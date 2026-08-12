use serde::{Deserialize, Serialize};

use crate::{Error, Paths, Result, write_atomic};

/// Bumped whenever `speakers.json`'s shape changes incompatibly.
///
/// Separate from every other schema version in this crate for the reason they are all
/// separate from each other: this file lives at the root rather than in a session, is
/// written by `enroll` and read by `transcribe`, and evolves on its own schedule.
///
/// # v1 -> v2, which is a change of *invariant* rather than of shape
///
/// The JSON is byte-for-byte the same shape in both versions. What moved is the rule about
/// it: v1 held **one row per name** and replaced that row when a person was named again; v2
/// holds **every confirmed recording of a person as its own row under their name** (see
/// [`EnrolledSpeakers`]). Every v1 file is therefore already a valid v2 file -- one row per
/// name is the degenerate reference set -- which is why v1 is migrated on read rather than
/// refused.
///
/// The version still has to move, because the hazard runs the other way: a v1-writing binary
/// reading a v2 file replaces only the *first* row for a name and silently keeps the rest as
/// stale claims about a voice. That downgrade is what [`crate::Error::UnsupportedSchema`]
/// catches from the other side.
pub const ENROLLED_SPEAKERS_SCHEMA_VERSION: u32 = 2;

/// The oldest `speakers.json` this build can read. Below this nothing exists to migrate from.
const OLDEST_SUPPORTED_SCHEMA_VERSION: u32 = 1;

/// How many recordings of one person are kept as references.
///
/// # Why there is a cap at all, and why it is five
///
/// The cost of the extra dot products is not the argument -- five 256-dim products per person
/// is nothing. The argument is **false accepts: every extra reference is another independent
/// draw at clearing the identification cut for a stranger.** TASK-027.01 measured the nearest
/// impostor pair over 1014 LibriSpeech pairs at 0.362 against a 0.350 cut, i.e. 0.012 of
/// headroom at one or two references, so multiplying the draws is not free. Against that, the
/// measured gain from the *second* reference was +2.1pp with overlapping Wilson intervals --
/// real but small -- and a fifth reference cannot plausibly be worth more than the second.
///
/// Five covers a person met in five different rooms while bounding the impostor exposure at
/// 5x. It is a judgement rather than a measurement, and the measurement that would revise it
/// is scoring `k = 1..3` references per person on the chapter cache.
///
/// # What happens at the cap: the new reference is refused, nothing stored is dropped
///
/// [`EnrolledSpeakers::store_reference`] returns [`Stored::AtCapacity`] and writes nothing.
/// The caller names the voice against that session instead -- the same `speaker_names.json`
/// path a below-floor voice takes -- so the transcript still reads the right person and the
/// recording simply does not contribute to recognising them.
///
/// The two alternatives are worse rather than merely different:
///
/// - **Drop the oldest.** The dropped reference may be the only thing naming a voice in some
///   past session, whose transcript then reads "Unknown N" on the next `enroll` run over it --
///   which is exactly the defect this reference set exists to end. Worse, nothing here records
///   provenance, so the tool could not even say which session it had broken.
/// - **Merge the nearest pair.** That is averaging, which TASK-027.01 measured halving the
///   impostor headroom (0.376 -> 0.362), and a blended vector equals no cluster on disk, so it
///   would silently stop [`EnrolledSpeakers::forget_reference`]'s exact-equality removal from
///   ever firing again.
pub const MAX_REFERENCES_PER_SPEAKER: usize = 5;

/// What [`EnrolledSpeakers::store_reference`] did, so a caller can say so in one line.
///
/// An enum rather than a bool-and-count, because the four cases want four different sentences
/// and the caller must not have to re-derive which it was by counting rows itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stored {
    /// The first reference for this name: a person who was not in the database is now.
    Enrolled,

    /// Another recording of somebody already here. `held` is their new total, `2..=MAX`.
    Added { held: usize },

    /// Bit-identical to a reference this name already has, so nothing was written.
    ///
    /// Re-answering the same voice with the same name is the common way to reach this -- a
    /// second `enroll` run over one session under `--correct` -- and duplicating the row would
    /// spend one of the five slots on no new information.
    AlreadyHeld,

    /// Nothing was stored: this name already holds [`MAX_REFERENCES_PER_SPEAKER`].
    AtCapacity { held: usize },
}

/// A name that lost a reference to somebody else's correction, and what it has left.
///
/// `remaining` exists because "Nate no longer has a reference" is a lie when Nate has three
/// and lost one -- and under a reference set that is the usual case rather than the rare one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Displaced {
    pub name: String,
    pub remaining: usize,
}

/// One confirmed recording of one person: the voice, and who it is of.
///
/// Deliberately just those two fields. Enrollment timestamps, source session ids and
/// re-enrollment history are all in service of *versioning* the database -- renaming,
/// removing, or re-enrolling someone whose embedding has drifted -- which this file does not
/// do. A field added later is cheap; a field written now and reinterpreted later is not.
///
/// Note what this is *not*: it is not "a person". Several rows can carry one name, and all of
/// them together are that person -- see [`EnrolledSpeakers`]. Code that iterates these rows
/// and means "people" is counting recordings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnrolledSpeaker {
    /// What this person is called in a transcript, exactly as the user typed it.
    pub name: String,

    /// The voice fingerprint: one enrolled cluster's centroid, L2-normalized, stored
    /// unmodified.
    ///
    /// Nothing is averaged and nothing is blended. That is load-bearing twice over: it keeps
    /// each reference an honest description of one recording rather than of a point between
    /// two, and it is what lets [`EnrolledSpeakers::forget_reference`] find the row built from
    /// a given cluster by exact equality.
    ///
    /// The order matters and is part of the contract, not an implementation detail, and it
    /// is the same order [`crate::SpeakerCluster::embedding`] is built in -- which is the
    /// whole point, because comparing the two is then a dot product. The mean of normalized
    /// vectors and the normalized mean are different vectors, so produce either side any
    /// other way and identification silently never fires.
    ///
    /// Length is whatever the embedding model emits (256 for the WeSpeaker checkpoint
    /// meethook ships against); this file does not pin it, so a future model change is a
    /// schema bump rather than a lie in a constant here.
    pub embedding: Vec<f32>,
}

/// `speakers.json`: everybody meethook can put a name to, across all sessions.
///
/// One file at the root of the meethook directory rather than one per session, because the
/// entire value of enrollment is that naming someone once names them in every meeting they
/// turn up in.
///
/// # A person is every row bearing their name
///
/// `speakers` is a flat list with no uniqueness constraint, and that is the design rather than
/// a gap in it. Rows under one name are independent recordings of one voice -- met in
/// different rooms, on different microphones -- in the order they were enrolled, and a person
/// is scored at the *nearest* of theirs. Identification gets that for free: its argmax already
/// runs over rows and returns the winning row's name, and it groups contenders by name before
/// the heard-at-once veto, so a person's several references collapse to one contender for one
/// name before any conflict is resolved.
///
/// TASK-027.01 measured this against replacing a person's reference each time they are named
/// (the v1 rule) and against averaging the two. Keeping both beat replacing on both corpora;
/// averaging matched keeping on accuracy but halved the impostor headroom, so nothing here is
/// blended.
///
/// The multiplicity is *not* something call sites should re-derive: [`Self::store_reference`],
/// [`Self::forget_reference`] and [`Self::references`] are the whole interface to it, and they
/// are what keep the cap, the duplicate rule and the correction rule in one place.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnrolledSpeakers {
    pub schema_version: u32,
    pub speakers: Vec<EnrolledSpeaker>,
}

impl EnrolledSpeakers {
    pub fn new(speakers: Vec<EnrolledSpeaker>) -> Self {
        EnrolledSpeakers {
            schema_version: ENROLLED_SPEAKERS_SCHEMA_VERSION,
            speakers,
        }
    }

    /// How many references this name holds. Zero for somebody who is not enrolled.
    pub fn references(&self, name: &str) -> usize {
        self.speakers.iter().filter(|s| s.name == name).count()
    }

    /// Records `embedding` as another recording of `name`, reporting what that did.
    ///
    /// Nothing existing is ever modified or removed: the only outcomes are one row appended or
    /// no change at all. That is what makes the caller's promise -- naming a voice never costs
    /// an earlier name -- true of the database layer rather than merely intended above it.
    ///
    /// Names match exactly, so "alice" and "Alice" are two people, which is the same rule the
    /// rest of the tool applies to a name the user typed.
    pub fn store_reference(&mut self, name: &str, embedding: Vec<f32>) -> Stored {
        let held = self.references(name);
        if self
            .speakers
            .iter()
            .any(|s| s.name == name && s.embedding == embedding)
        {
            return Stored::AlreadyHeld;
        }
        if held >= MAX_REFERENCES_PER_SPEAKER {
            return Stored::AtCapacity { held };
        }
        self.speakers.push(EnrolledSpeaker {
            name: name.to_string(),
            embedding,
        });
        if held == 0 {
            Stored::Enrolled
        } else {
            Stored::Added { held: held + 1 }
        }
    }

    /// Drops every reference *not* under `keeping` that was built from this exact `embedding`,
    /// reporting each name that lost one and how many it has left.
    ///
    /// This is the correction guarantee: a voice stored under the wrong name is a claim about a
    /// person it is not of, and it goes on competing in every future meeting -- winning
    /// whenever its name happens to sort first -- until it is removed. The user naming that
    /// voice somebody else is the only evidence that will ever arrive, so it is acted on here.
    ///
    /// **Exact equality is the whole condition**, and it generalises to a reference set without
    /// changing: a reference derived from another recording of that same wrong name is a
    /// different vector, a legitimate one, and is left alone. That only holds because nothing
    /// is averaged -- a blended vector equals no cluster on disk, and this removal would
    /// silently stop firing.
    pub fn forget_reference(&mut self, embedding: &[f32], keeping: &str) -> Vec<Displaced> {
        let losing: Vec<String> = self
            .speakers
            .iter()
            .filter(|s| s.name != keeping && s.embedding == embedding)
            .map(|s| s.name.clone())
            .collect();
        if losing.is_empty() {
            return Vec::new();
        }
        self.speakers
            .retain(|s| s.name == keeping || s.embedding != embedding);
        // Deduplicated in first-loss order rather than sorted: one name can lose two identical
        // rows, and it should be reported once, with the count it is actually left with.
        let mut displaced: Vec<Displaced> = Vec::new();
        for name in losing {
            if displaced.iter().any(|d| d.name == name) {
                continue;
            }
            let remaining = self.references(&name);
            displaced.push(Displaced { name, remaining });
        }
        displaced
    }

    pub fn write(&self, paths: &Paths) -> Result<()> {
        let path = paths.speakers_json();
        let mut json = serde_json::to_vec_pretty(self).map_err(|e| Error::json(&path, e))?;
        json.push(b'\n');
        write_atomic(&path, &json)
    }

    /// Reads the database, treating "there isn't one yet" as an empty one.
    ///
    /// Every session recorded before anybody was enrolled is a session with no enrolled
    /// speakers, and that is the *normal* first run rather than an error -- so the absent
    /// file is defined out of existence here instead of at every call site.
    ///
    /// A file that exists and does not parse is a different event entirely and stays an
    /// error: a user who enrolled ten people and then silently got ten Unknowns back has
    /// been failed quietly, which is the one outcome worth interrupting for.
    ///
    /// # The version gate, and why old is migrated where `speaker_clusters.json` refuses
    ///
    /// An older version is **accepted and normalized to the current one**; only a version this
    /// build does not recognise -- which can only come from a downgrade -- is refused, as
    /// [`Error::UnsupportedSchema`].
    ///
    /// That is a deliberate divergence from [`crate::SPEAKER_CLUSTERS_SCHEMA_VERSION`]'s
    /// precedent, which refuses an old file. It can: `transcribe --force` regenerates clusters
    /// from the audio. **References cannot be regenerated -- the audio they were built from may
    /// be long deleted -- so refusing an old database would destroy the names in it.**
    ///
    /// The v1 -> v2 migration is a normalization and nothing else, because a v1 file already
    /// satisfies v2's invariant: one row per name is the degenerate case of "a person is every
    /// row bearing their name". There is no data transformation to look for here.
    ///
    /// Normalizing **on read** rather than forcing the constant in [`Self::write`] is the point:
    /// a database this build accepted is one it can write, and carrying the old number in memory
    /// would write it straight back and leave the file claiming a version its contents no longer
    /// match.
    pub fn read_or_empty(paths: &Paths) -> Result<EnrolledSpeakers> {
        let path = paths.speakers_json();
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(EnrolledSpeakers::new(Vec::new()));
            }
            Err(e) => return Err(Error::io(&path, e)),
        };
        let mut speakers: EnrolledSpeakers =
            serde_json::from_slice(&bytes).map_err(|e| Error::json(&path, e))?;
        if !(OLDEST_SUPPORTED_SCHEMA_VERSION..=ENROLLED_SPEAKERS_SCHEMA_VERSION)
            .contains(&speakers.schema_version)
        {
            return Err(Error::UnsupportedSchema {
                path,
                found: speakers.schema_version,
                oldest: OLDEST_SUPPORTED_SCHEMA_VERSION,
                newest: ENROLLED_SPEAKERS_SCHEMA_VERSION,
            });
        }
        speakers.schema_version = ENROLLED_SPEAKERS_SCHEMA_VERSION;
        Ok(speakers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speakers() -> EnrolledSpeakers {
        EnrolledSpeakers::new(vec![
            EnrolledSpeaker {
                name: "Alice".to_string(),
                embedding: vec![0.6, 0.8],
            },
            EnrolledSpeaker {
                name: "Bob".to_string(),
                embedding: vec![0.8, -0.6],
            },
        ])
    }

    #[test]
    fn a_written_file_reads_back_identical() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let written = speakers();

        written.write(&paths).unwrap();

        let read = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(read, written);
        assert_eq!(read.schema_version, ENROLLED_SPEAKERS_SCHEMA_VERSION);
    }

    /// The representation is the contract: a reference is a unit vector, and the file has to
    /// hand it back unchanged. Matching is a bare dot product on the strength of it -- so a
    /// round trip that quietly rescaled a vector, by rounding it through a shorter float format
    /// or otherwise, would move every similarity and shift the threshold under everybody.
    #[test]
    fn a_normalized_reference_survives_a_round_trip_still_normalized() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());

        // A cluster centroid: mean-pooled over its segments, then normalized, exactly as
        // clustering produces it and exactly as it is stored here.
        let clips = [[0.31f32, 0.77, -0.55], [0.62, 0.19, 0.44]];
        let mut reference: Vec<f32> = (0..3).map(|i| (clips[0][i] + clips[1][i]) / 2.0).collect();
        let norm = reference.iter().map(|v| v * v).sum::<f32>().sqrt();
        reference.iter_mut().for_each(|v| *v /= norm);

        EnrolledSpeakers::new(vec![EnrolledSpeaker {
            name: "Alice".to_string(),
            embedding: reference.clone(),
        }])
        .write(&paths)
        .unwrap();

        let read = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(read.speakers[0].embedding, reference);
        let norm = read.speakers[0]
            .embedding
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "stored reference had norm {norm}"
        );
    }

    /// The first run of every install: nobody has enrolled anybody, so there is no file. That
    /// has to be an empty database rather than an error, or `transcribe` would refuse to work
    /// until `enroll` had been run at least once.
    #[test]
    fn an_absent_file_reads_as_an_empty_database() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());

        let read = EnrolledSpeakers::read_or_empty(&paths).unwrap();

        assert!(read.speakers.is_empty());
        assert_eq!(read.schema_version, ENROLLED_SPEAKERS_SCHEMA_VERSION);
    }

    /// The other half of that: a file that is *there* and unreadable is not the first-run
    /// case, and must not be silently downgraded into one.
    #[test]
    fn a_malformed_file_is_an_error_rather_than_an_empty_database() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        std::fs::write(paths.speakers_json(), b"{ this is not json").unwrap();

        let error = EnrolledSpeakers::read_or_empty(&paths).unwrap_err();

        assert!(
            matches!(error, Error::Json { .. }),
            "expected a JSON error, got {error:?}"
        );
        // The path is in the message, because "malformed JSON" with no file name is a
        // support question rather than an answer.
        assert!(error.to_string().contains("speakers.json"), "{error}");
    }

    /// Enrolling a second person rewrites the whole file, so the write must replace the
    /// previous contents rather than append to them -- the in-memory database is the whole
    /// truth, and a write that appended would double every row on the second one.
    #[test]
    fn rewriting_replaces_the_previous_contents() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());

        speakers().write(&paths).unwrap();
        EnrolledSpeakers::new(Vec::new()).write(&paths).unwrap();

        let read = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert!(read.speakers.is_empty());
    }

    /// The atomic write leaves no temp file behind, and lands on the one name the rest of the
    /// tool looks for.
    #[test]
    fn writing_leaves_exactly_one_file_in_the_root() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());

        speakers().write(&paths).unwrap();

        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(entries, [paths.speakers_json().file_name().unwrap()]);
    }

    /// Writes a `speakers.json` by hand at whatever version, since the point of these tests is
    /// files this build did not write.
    fn write_raw(paths: &Paths, version: u32, rows: &[(&str, &[f32])]) {
        let rows: Vec<String> = rows
            .iter()
            .map(|(name, embedding)| {
                format!(
                    "{{\"name\": \"{name}\", \"embedding\": {}}}",
                    serde_json::to_string(embedding).unwrap()
                )
            })
            .collect();
        let json = format!(
            "{{\n  \"schema_version\": {version},\n  \"speakers\": [{}]\n}}\n",
            rows.join(", ")
        );
        std::fs::write(paths.speakers_json(), json).unwrap();
    }

    /// A v1 database is the one file that must never be refused: the audio its references were
    /// built from may be long deleted, so refusing it would destroy names nothing can rebuild.
    /// Every v1 file already satisfies v2's invariant, so the rows come back untouched.
    #[test]
    fn a_v1_database_is_migrated_on_read_with_its_rows_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        write_raw(&paths, 1, &[("Alice", &[0.6, 0.8]), ("Bob", &[0.8, -0.6])]);

        let read = EnrolledSpeakers::read_or_empty(&paths).unwrap();

        assert_eq!(read.speakers, speakers().speakers);
        assert_eq!(read.schema_version, ENROLLED_SPEAKERS_SCHEMA_VERSION);
    }

    /// The other half of migrating on read: the version is normalized in memory, so the next
    /// write puts the new number on disk rather than writing the old one back beside contents
    /// it no longer describes.
    #[test]
    fn writing_back_a_migrated_v1_database_upgrades_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        write_raw(&paths, 1, &[("Alice", &[0.6, 0.8])]);

        EnrolledSpeakers::read_or_empty(&paths)
            .unwrap()
            .write(&paths)
            .unwrap();

        let json = std::fs::read_to_string(paths.speakers_json()).unwrap();
        assert!(
            json.contains(&format!(
                "\"schema_version\": {ENROLLED_SPEAKERS_SCHEMA_VERSION}"
            )),
            "{json}"
        );
    }

    /// A version from the future can only be a downgrade, and reading it as though it were this
    /// one would believe a shape nothing here has a rule for. Refused, with the path and a
    /// remedy, because "unsupported schema" with no file name is a support question.
    #[test]
    fn a_version_this_build_does_not_recognise_is_refused_with_a_remedy() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        write_raw(
            &paths,
            ENROLLED_SPEAKERS_SCHEMA_VERSION + 1,
            &[("Alice", &[0.6, 0.8])],
        );

        let error = EnrolledSpeakers::read_or_empty(&paths).unwrap_err();

        assert!(
            matches!(error, Error::UnsupportedSchema { .. }),
            "expected an unsupported-schema error, got {error:?}"
        );
        let message = error.to_string();
        assert!(message.contains("speakers.json"), "{message}");
        assert!(message.contains("upgrade meethook"), "{message}");
    }

    /// The gate is a range rather than a ceiling: version 0 was never written by anything, so
    /// there is no migration to run and believing it would be a guess.
    #[test]
    fn a_version_below_the_readable_range_is_refused_too() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        write_raw(&paths, 0, &[("Alice", &[0.6, 0.8])]);

        let error = EnrolledSpeakers::read_or_empty(&paths).unwrap_err();

        assert!(
            matches!(error, Error::UnsupportedSchema { found: 0, .. }),
            "{error:?}"
        );
    }

    #[test]
    fn the_first_reference_for_a_name_enrols_that_person() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());

        assert_eq!(
            speakers.store_reference("Alice", vec![1.0, 0.0]),
            Stored::Enrolled
        );
        assert_eq!(speakers.references("Alice"), 1);
        assert_eq!(speakers.references("Bob"), 0);
    }

    /// The whole point of v2: a second recording of somebody already enrolled is *added*, and
    /// the first one is still there afterwards. Under v1 this replaced it.
    #[test]
    fn a_second_recording_is_added_rather_than_replacing_the_first() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Alice", vec![1.0, 0.0]);

        assert_eq!(
            speakers.store_reference("Alice", vec![0.0, 1.0]),
            Stored::Added { held: 2 }
        );
        let stored: Vec<&[f32]> = speakers
            .speakers
            .iter()
            .map(|s| s.embedding.as_slice())
            .collect();
        assert_eq!(
            stored,
            [[1.0, 0.0].as_slice(), [0.0, 1.0].as_slice()],
            "enrollment order is part of the contract"
        );
    }

    /// Re-answering the same voice with the same name -- a second `enroll --correct` pass over
    /// one session -- must not spend a capped slot on information already held.
    #[test]
    fn a_byte_identical_reference_is_recognised_rather_than_duplicated() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Alice", vec![1.0, 0.0]);

        assert_eq!(
            speakers.store_reference("Alice", vec![1.0, 0.0]),
            Stored::AlreadyHeld
        );
        assert_eq!(speakers.references("Alice"), 1);
    }

    /// The same vector under two names is two different claims, and both are legitimate until
    /// a user says otherwise -- which is `forget_reference`'s job, not this one's.
    #[test]
    fn the_duplicate_rule_is_per_name_rather_than_across_the_database() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Alice", vec![1.0, 0.0]);

        assert_eq!(
            speakers.store_reference("Bob", vec![1.0, 0.0]),
            Stored::Enrolled
        );
        assert_eq!(speakers.references("Bob"), 1);
    }

    /// At the cap nothing stored is dropped: the new reference is refused and the caller is
    /// told what is held, so it can name the voice against the session instead. Dropping the
    /// oldest would un-name a voice in some past session, which is the defect v2 exists to end.
    #[test]
    fn at_the_cap_the_new_reference_is_refused_and_nothing_is_dropped() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        for i in 0..MAX_REFERENCES_PER_SPEAKER {
            let mut embedding = vec![0.0f32; MAX_REFERENCES_PER_SPEAKER + 1];
            embedding[i] = 1.0;
            assert!(matches!(
                speakers.store_reference("Alice", embedding),
                Stored::Enrolled | Stored::Added { .. }
            ));
        }
        let mut over = vec![0.0f32; MAX_REFERENCES_PER_SPEAKER + 1];
        over[MAX_REFERENCES_PER_SPEAKER] = 1.0;
        let held_before = speakers.speakers.clone();

        assert_eq!(
            speakers.store_reference("Alice", over),
            Stored::AtCapacity {
                held: MAX_REFERENCES_PER_SPEAKER
            }
        );
        assert_eq!(speakers.speakers, held_before);
    }

    /// The cap is per person, so somebody else being full says nothing about a new name.
    #[test]
    fn the_cap_is_per_person_rather_than_over_the_database() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        for i in 0..MAX_REFERENCES_PER_SPEAKER {
            let mut embedding = vec![0.0f32; MAX_REFERENCES_PER_SPEAKER];
            embedding[i] = 1.0;
            speakers.store_reference("Alice", embedding);
        }

        assert_eq!(
            speakers.store_reference("Bob", vec![1.0, 0.0]),
            Stored::Enrolled
        );
    }

    /// The correction guarantee, generalised: the wrong name loses the reference built from
    /// this voice and keeps the ones built from its own recordings, and the report says how
    /// many it has left -- "Nate no longer has a reference" being a lie when Nate has two.
    #[test]
    fn forgetting_a_reference_drops_only_that_row_and_reports_what_is_left() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Nate", vec![1.0, 0.0]);
        speakers.store_reference("Nate", vec![0.0, 1.0]);
        speakers.store_reference("Ryan", vec![0.0, 1.0]);

        // Ryan is being given the voice Nate's second reference was built from.
        let displaced = speakers.forget_reference(&[0.0, 1.0], "Ryan");

        assert_eq!(
            displaced,
            [Displaced {
                name: "Nate".to_string(),
                remaining: 1
            }]
        );
        assert_eq!(speakers.references("Nate"), 1);
        assert_eq!(speakers.speakers[0].embedding, vec![1.0, 0.0]);
        assert_eq!(speakers.references("Ryan"), 1);
    }

    /// The v1 case still reads the same way, which is what an existing enroll test asserts
    /// verbatim: a name whose only reference was of somebody else has none left.
    #[test]
    fn forgetting_a_names_only_reference_leaves_it_with_none() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Nate", vec![1.0, 0.0]);

        let displaced = speakers.forget_reference(&[1.0, 0.0], "Ryan");

        assert_eq!(displaced[0].remaining, 0);
        assert!(speakers.speakers.is_empty());
    }

    /// Exact equality is the condition, and nothing here is averaged, so a reference built from
    /// a *different* recording of that wrong name is a different vector and is left alone.
    #[test]
    fn forgetting_leaves_a_different_recording_of_the_wrong_name_alone() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Nate", vec![0.6, 0.8]);

        assert!(speakers.forget_reference(&[1.0, 0.0], "Ryan").is_empty());
        assert_eq!(speakers.references("Nate"), 1);
    }

    /// The name being stored keeps every one of its own references: somebody else's correction
    /// must not cost a person the recordings that are genuinely of them.
    #[test]
    fn forgetting_never_touches_the_name_being_kept() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Ryan", vec![1.0, 0.0]);
        speakers.store_reference("Ryan", vec![0.0, 1.0]);

        assert!(speakers.forget_reference(&[1.0, 0.0], "Ryan").is_empty());
        assert_eq!(speakers.references("Ryan"), 2);
    }
}
