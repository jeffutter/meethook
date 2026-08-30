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
///
/// # v2 -> v3, one optional field
///
/// v3 adds [`EnrolledSpeaker::clip_seconds`]: how much speech the row was built from, which is
/// what [`EnrolledSpeakers::store_reference`] now compares when the cap is full. The field is
/// optional and absent means *unknown*, so every v2 row is a valid v3 row and the migration is
/// again a normalization rather than a transformation.
///
/// The downgrade hazard is the same shape as before and is why the number moves: a v2-writing
/// binary round-tripping a v3 file drops the lengths, and every row silently becomes unmeasured
/// -- which does not lose a reference, but does lose the tool's ability to make room for a
/// better one.
pub const ENROLLED_SPEAKERS_SCHEMA_VERSION: u32 = 3;

/// The oldest `speakers.json` this build can read. Below this nothing exists to migrate from.
const OLDEST_SUPPORTED_SCHEMA_VERSION: u32 = 1;

/// How many recordings of one person are kept as references.
///
/// # Why there is a cap at all, and why it is ten
///
/// The cost of the extra dot products is not the argument -- ten 256-dim products per person is
/// nothing. The argument is **false accepts: every extra reference is another independent draw
/// at clearing the identification cut for a stranger.** TASK-027.01 measured the nearest
/// impostor pair over 1014 LibriSpeech pairs at 0.362, and `transcribe`'s `IDENTIFY_DISTANCE`
/// now sits at 0.400 -- so the draws are no longer free of each other the way they were when the
/// cut was 0.350 and the measured impostor population sat entirely above it.
///
/// Ten rather than five because the binding constraint in practice was not impostor exposure
/// but **running out of room for a better recording of somebody already enrolled**: a person met
/// in a sixth meeting had nothing to offer the database, however much cleaner the sixth
/// recording was than the five it could not displace. Five was already a judgement rather than
/// a measurement, and doubling it is the same judgement with the eviction rule below carrying
/// the risk that the extra draws add.
///
/// # What happens at the cap: the longest clips win, and only strictly better evidence displaces
///
/// [`EnrolledSpeakers::store_reference`] compares the new clip's length against the shortest
/// [`EnrolledSpeaker::clip_seconds`] this name holds. If the new one is longer it replaces that
/// shortest row ([`Stored::Replaced`]); otherwise nothing is written ([`Stored::AtCapacity`])
/// and the caller names the voice against that session instead -- the same `speaker_names.json`
/// path a below-floor voice takes -- so the transcript still reads the right person and the
/// recording simply does not contribute to recognising them.
///
/// A cap holding the ten *longest* recordings of a person is a better database than one holding
/// the first ten, and it is what pays for the doubled draw count: a longer clip yields a
/// tighter, less noisy centroid, so the references that survive are individually less likely to
/// clear the cut for a stranger than the ones they displaced.
///
/// **A row with no measured length is never the one evicted.** Absent lengths come from a
/// pre-v3 file, where the only thing knowable about the clip is that it cleared the reference
/// floor -- it could equally have been six seconds or six minutes. Treating unknown as short
/// would preferentially destroy exactly the long, good references written before the field
/// existed, so the rule declines to guess and those rows are removable only by hand.
///
/// # What this reverses, and why that is safe now
///
/// Before TASK-027.02's cap grew this rule, *nothing* stored was ever dropped, because "drop the
/// oldest" can take away the only reference naming a voice in some past session, whose
/// transcript then reads "Unknown N" on the next `enroll` run over it. That hazard is real and
/// has not gone away. What makes it acceptable here is that this is not "drop the oldest": a row
/// is displaced only by a **longer recording of the same person**, so the evidence that replaces
/// it is strictly stronger than the evidence removed, and a voice the short clip was naming is
/// more likely to be named by the long one than not. Callers report the eviction and its length
/// on the line that reports the enrollment, because an enrollment that vanishes without a line
/// about it is worse than the bug.
///
/// The alternative that stays rejected is **merging the nearest pair**. That is averaging, which
/// TASK-027.01 measured halving the impostor headroom (0.376 -> 0.362), and a blended vector
/// equals no cluster on disk, so it would silently stop
/// [`EnrolledSpeakers::forget_reference`]'s exact-equality removal from ever firing again.
///
/// Deliberate removal remains the escape hatch for everything this rule will not do: `meethook
/// speakers` says which voices each of a person's recordings is naming, and `meethook forget`
/// removes the one the user chooses after printing what that costs.
pub const MAX_REFERENCES_PER_SPEAKER: usize = 10;

/// What [`EnrolledSpeakers::store_reference`] did, so a caller can say so in one line.
///
/// An enum rather than a bool-and-count, because the five cases want five different sentences
/// and the caller must not have to re-derive which it was by counting rows itself.
///
/// Not `Copy`, unlike its first four variants alone: [`Stored::AtCapacity`] carries the
/// shortest length held so the caller can say *why* the recording was refused, and a future
/// field that is not a number should not be blocked by a derive nothing needs.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Stored {
    /// The first reference for this name: a person who was not in the database is now.
    Enrolled,

    /// Another recording of somebody already here. `held` is their new total, `2..=MAX`.
    Added { held: usize },

    /// Bit-identical to a reference this name already has, so nothing was written.
    ///
    /// Re-answering the same voice with the same name is the common way to reach this -- a
    /// second `enroll` run over one session under `--correct` -- and duplicating the row would
    /// spend one of the slots on no new information.
    AlreadyHeld,

    /// The name was full, and this recording was longer than the shortest it held, which was
    /// dropped to make room. `held` stays at [`MAX_REFERENCES_PER_SPEAKER`].
    ///
    /// `evicted_seconds` is what went, so the caller can print the trade it just made. The
    /// dropped row may have been the only thing naming a voice in some other session -- see
    /// [`MAX_REFERENCES_PER_SPEAKER`] for why that is accepted here and nowhere else -- which is
    /// why this is a distinct variant rather than an [`Stored::Added`] at the cap.
    Replaced { held: usize, evicted_seconds: f64 },

    /// Nothing was stored: this name already holds [`MAX_REFERENCES_PER_SPEAKER`], none of them
    /// shorter than what was offered.
    ///
    /// `shortest` is the smallest measured length held, or `None` when every row this name holds
    /// predates [`EnrolledSpeaker::clip_seconds`] and so cannot be compared against anything.
    /// The two cases want different sentences: one says the stored recordings are better, the
    /// other says their lengths are simply not known.
    AtCapacity { held: usize, shortest: Option<f64> },
}

/// A name that lost a reference to somebody else's correction, and what it has left.
///
/// `remaining` exists because "Nate no longer has a reference" is a lie when Nate has three
/// and lost one -- and under a reference set that is the usual case rather than the rare one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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

    /// How much speech the cluster this was built from held, in seconds. `None` for a row
    /// written before v3, where it is genuinely unknown rather than zero.
    ///
    /// The one thing this file records about a reference beyond the voice and the name, and it
    /// earns that place by being what [`EnrolledSpeakers::store_reference`] decides on at the
    /// cap: longer clip, tighter centroid, better reference, so the shortest is the one to
    /// displace. It is a *quality* measure, deliberately not provenance -- it does not say which
    /// session or cluster the row came from, which is still derived on demand by re-labelling.
    ///
    /// `None` is not `0.0` anywhere that matters: an unmeasured row never loses the comparison
    /// above, because "unknown" and "shortest" are different claims and only one of them is
    /// supported by the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_seconds: Option<f64>,
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

    /// Every enrolled name, deduplicated, in the order it first appears -- which is enrolment
    /// order, since nothing here ever reorders or replaces a row.
    ///
    /// The counterpart to [`Self::references`], which answers "how many" for a name somebody
    /// already has: this is what a caller with no name in hand asks first. Here rather than
    /// derived from `speakers` at the call site for the reason the type doc gives -- a caller
    /// iterating the rows and calling them people is counting recordings, and would list a
    /// person once per reference they hold.
    pub fn enrolled_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = Vec::new();
        for speaker in &self.speakers {
            if !names.contains(&speaker.name.as_str()) {
                names.push(&speaker.name);
            }
        }
        names
    }

    /// Records `embedding`, built from `clip_seconds` of speech, as another recording of `name`,
    /// reporting what that did.
    ///
    /// Below the cap the only outcomes are one row appended or no change at all. At the cap the
    /// one thing that can be removed is **another recording of this same name, shorter than this
    /// one** -- see [`MAX_REFERENCES_PER_SPEAKER`] for why that trade is made and why no other
    /// row is ever touched. So the caller's promise still holds where it was always the point:
    /// naming a voice never costs somebody *else* their name.
    ///
    /// Names match exactly, so "alice" and "Alice" are two people, which is the same rule the
    /// rest of the tool applies to a name the user typed.
    pub fn store_reference(
        &mut self,
        name: &str,
        embedding: Vec<f32>,
        clip_seconds: f64,
    ) -> Stored {
        let held = self.references(name);
        if self
            .speakers
            .iter()
            .any(|s| s.name == name && s.embedding == embedding)
        {
            return Stored::AlreadyHeld;
        }
        let row = EnrolledSpeaker {
            name: name.to_string(),
            embedding,
            clip_seconds: Some(clip_seconds),
        };
        if held < MAX_REFERENCES_PER_SPEAKER {
            self.speakers.push(row);
            return if held == 0 {
                Stored::Enrolled
            } else {
                Stored::Added { held: held + 1 }
            };
        }

        // Full. The shortest *measured* recording of this person is the only row that can go, and
        // only to something longer. Ties keep the incumbent -- an equal-length clip is not better
        // evidence, and churning the file on it would displace a reference for nothing.
        let shortest = self
            .speakers
            .iter()
            .enumerate()
            .filter(|(_, s)| s.name == name)
            .filter_map(|(index, s)| s.clip_seconds.map(|seconds| (index, seconds)))
            .min_by(|(_, a), (_, b)| a.total_cmp(b));
        match shortest {
            Some((index, seconds)) if clip_seconds > seconds => {
                self.speakers.remove(index);
                self.speakers.push(row);
                Stored::Replaced {
                    held,
                    evicted_seconds: seconds,
                }
            }
            _ => Stored::AtCapacity {
                held,
                shortest: shortest.map(|(_, seconds)| seconds),
            },
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

    /// This database without one of `name`'s references, addressed by its 1-based position
    /// among the rows bearing that name in file order -- the same position
    /// [`Self::enrolled_names`] and [`Self::references`] count in.
    ///
    /// `None` when that name does not hold that many, which covers a name that is not enrolled
    /// at all, a position past the end, and a zero. Those are one outcome rather than three
    /// because the caller does one thing with each: report that the reference the user named is
    /// not there.
    ///
    /// # Why this returns a database rather than removing a row
    ///
    /// Two callers need this and only one of them is a removal. A report on what a reference is
    /// currently naming needs the *counterfactual* -- label every session against the database
    /// as it stands, then again without one row, and the diff is what that row is doing -- and a
    /// mutating removal would make it clone-and-restore around every comparison. A removal is
    /// then `speakers.without(name, handle)?.write(paths)?`, so the mapping from a printed
    /// position back to a row exists in exactly one place: the contract between what a listing
    /// prints and what a removal acts on.
    ///
    /// Nothing here knows about renumbering. The positions are of the list as it was read, so a
    /// caller holding a stale one is removing a row the user did not choose -- which is why a
    /// removal should echo what the chosen reference names before it writes, rather than this
    /// trying to detect it.
    pub fn without(&self, name: &str, position: usize) -> Option<EnrolledSpeakers> {
        let index = self
            .speakers
            .iter()
            .enumerate()
            .filter(|(_, speaker)| speaker.name == name)
            .map(|(index, _)| index)
            .nth(position.checked_sub(1)?)?;
        let mut rest = self.clone();
        rest.speakers.remove(index);
        Some(rest)
    }

    /// This database without any of `name`'s references -- which is this person removed, since a
    /// person is every row bearing their name.
    ///
    /// `None` when nothing is stored under that name, the same miss [`Self::without`] reports for
    /// a position that name does not hold, and for the same reason: the caller does one thing with
    /// it, which is to say that what the user named is not there.
    ///
    /// # Why this is not [`Self::without`] with an `Option`
    ///
    /// Removing a person is a *different counterfactual* from removing each of their references in
    /// turn, and the difference is not cosmetic: for a name holding two rows that both match one
    /// voice, dropping either alone leaves the other naming it, so a per-reference diff reports no
    /// change from either while removing the person reverts that voice. A caller wanting to know
    /// what losing a person costs has to label against this, and cannot aggregate the other.
    ///
    /// Pure, like [`Self::without`], and for the same two callers: the preview that says what a
    /// removal would cost, and the removal itself, which is
    /// `speakers.without_person(name)?.write(paths)?`.
    pub fn without_person(&self, name: &str) -> Option<EnrolledSpeakers> {
        if self.references(name) == 0 {
            return None;
        }
        let mut rest = self.clone();
        rest.speakers.retain(|speaker| speaker.name != name);
        Some(rest)
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

    /// A clip length for the tests that are not about clip lengths. Well clear of any floor, and
    /// the same for every reference they store, so nothing in those tests turns on the ordering
    /// the eviction rule imposes -- the ones that do say their own numbers.
    const CLIP: f64 = 30.0;

    fn speakers() -> EnrolledSpeakers {
        EnrolledSpeakers::new(vec![
            EnrolledSpeaker {
                name: "Alice".to_string(),
                embedding: vec![0.6, 0.8],
                clip_seconds: None,
            },
            EnrolledSpeaker {
                name: "Bob".to_string(),
                embedding: vec![0.8, -0.6],
                clip_seconds: None,
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
            clip_seconds: None,
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
            speakers.store_reference("Alice", vec![1.0, 0.0], CLIP),
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
        speakers.store_reference("Alice", vec![1.0, 0.0], CLIP);

        assert_eq!(
            speakers.store_reference("Alice", vec![0.0, 1.0], CLIP),
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
        speakers.store_reference("Alice", vec![1.0, 0.0], CLIP);

        assert_eq!(
            speakers.store_reference("Alice", vec![1.0, 0.0], CLIP),
            Stored::AlreadyHeld
        );
        assert_eq!(speakers.references("Alice"), 1);
    }

    /// The same vector under two names is two different claims, and both are legitimate until
    /// a user says otherwise -- which is `forget_reference`'s job, not this one's.
    #[test]
    fn the_duplicate_rule_is_per_name_rather_than_across_the_database() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Alice", vec![1.0, 0.0], CLIP);

        assert_eq!(
            speakers.store_reference("Bob", vec![1.0, 0.0], CLIP),
            Stored::Enrolled
        );
        assert_eq!(speakers.references("Bob"), 1);
    }

    /// A full name, with `seconds[i]` of speech behind the reference on axis `i`.
    fn full(seconds: &[f64]) -> EnrolledSpeakers {
        assert_eq!(seconds.len(), MAX_REFERENCES_PER_SPEAKER);
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        for (i, &clip) in seconds.iter().enumerate() {
            let mut embedding = vec![0.0f32; MAX_REFERENCES_PER_SPEAKER + 1];
            embedding[i] = 1.0;
            assert!(matches!(
                speakers.store_reference("Alice", embedding, clip),
                Stored::Enrolled | Stored::Added { .. }
            ));
        }
        speakers
    }

    /// One past the cap on an axis nothing held occupies, so it is always a new reference rather
    /// than a duplicate.
    fn over_the_cap() -> Vec<f32> {
        let mut over = vec![0.0f32; MAX_REFERENCES_PER_SPEAKER + 1];
        over[MAX_REFERENCES_PER_SPEAKER] = 1.0;
        over
    }

    /// At the cap a recording no longer than the shortest held is refused and nothing is
    /// dropped, and the refusal carries that shortest length so the caller can say why.
    #[test]
    fn at_the_cap_a_recording_no_longer_than_the_shortest_held_is_refused() {
        let mut speakers = full(&[12.0, 40.0, 31.0, 55.0, 22.0, 90.0, 18.0, 33.0, 47.0, 61.0]);
        let held_before = speakers.speakers.clone();

        assert_eq!(
            speakers.store_reference("Alice", over_the_cap(), 12.0),
            Stored::AtCapacity {
                held: MAX_REFERENCES_PER_SPEAKER,
                shortest: Some(12.0),
            },
            "an equal-length clip is not better evidence, so the incumbent keeps the slot"
        );
        assert_eq!(speakers.speakers, held_before);
    }

    /// The other half: a longer recording of somebody full displaces the shortest one they hold,
    /// and the report names the length that went so the caller can print the trade.
    #[test]
    fn at_the_cap_a_longer_recording_displaces_the_shortest_held() {
        let mut speakers = full(&[12.0, 40.0, 31.0, 55.0, 22.0, 90.0, 18.0, 33.0, 47.0, 61.0]);
        let over = over_the_cap();

        assert_eq!(
            speakers.store_reference("Alice", over.clone(), 12.5),
            Stored::Replaced {
                held: MAX_REFERENCES_PER_SPEAKER,
                evicted_seconds: 12.0,
            }
        );
        assert_eq!(speakers.references("Alice"), MAX_REFERENCES_PER_SPEAKER);
        let lengths: Vec<Option<f64>> = speakers.speakers.iter().map(|s| s.clip_seconds).collect();
        assert!(
            !lengths.contains(&Some(12.0)),
            "the 12.0 s reference should be gone, held {lengths:?}"
        );
        assert_eq!(
            speakers.speakers.last().map(|s| s.embedding.as_slice()),
            Some(over.as_slice()),
            "the replacement goes on the end, like every other stored reference"
        );
    }

    /// Repeated displacement converges on the longest recordings rather than the latest ones:
    /// the point of the rule is that the database gets better as it is used, not merely newer.
    #[test]
    fn displacement_leaves_a_name_holding_its_longest_recordings() {
        let mut speakers = full(&[12.0, 40.0, 31.0, 55.0, 22.0, 90.0, 18.0, 33.0, 47.0, 61.0]);

        for (i, clip) in [70.0f64, 5.0, 25.0].into_iter().enumerate() {
            let mut embedding = vec![0.0f32; MAX_REFERENCES_PER_SPEAKER + 4];
            embedding[MAX_REFERENCES_PER_SPEAKER + i] = 1.0;
            speakers.store_reference("Alice", embedding, clip);
        }

        // 70.0 displaces the 12.0; 5.0 is refused, being shorter than the 18.0 that is now
        // shortest; 25.0 displaces that 18.0. What is left is the ten longest of the thirteen.
        let mut held: Vec<f64> = speakers
            .speakers
            .iter()
            .filter_map(|s| s.clip_seconds)
            .collect();
        held.sort_by(f64::total_cmp);
        assert_eq!(
            held,
            [22.0, 25.0, 31.0, 33.0, 40.0, 47.0, 55.0, 61.0, 70.0, 90.0]
        );
    }

    /// A reference with no measured length predates the field, and the only thing knowable about
    /// it is that it cleared the floor -- it could have been six seconds or six minutes. So it is
    /// never the row a longer clip displaces, and a name holding nothing else is simply full.
    #[test]
    fn a_reference_with_no_measured_length_is_never_the_one_displaced() {
        let mut speakers = EnrolledSpeakers::new(
            (0..MAX_REFERENCES_PER_SPEAKER)
                .map(|i| {
                    let mut embedding = vec![0.0f32; MAX_REFERENCES_PER_SPEAKER + 1];
                    embedding[i] = 1.0;
                    EnrolledSpeaker {
                        name: "Alice".to_string(),
                        embedding,
                        clip_seconds: None,
                    }
                })
                .collect(),
        );
        let held_before = speakers.speakers.clone();

        assert_eq!(
            speakers.store_reference("Alice", over_the_cap(), 600.0),
            Stored::AtCapacity {
                held: MAX_REFERENCES_PER_SPEAKER,
                shortest: None,
            },
            "nothing held is comparable, so there is no basis for calling one of them the worst"
        );
        assert_eq!(speakers.speakers, held_before);
    }

    /// The mixed case, which is what every existing database becomes on its first enrollment
    /// after v3: unmeasured rows sit the comparison out, and the shortest *measured* one is what
    /// a longer recording displaces. So a person carrying old references can still be improved,
    /// without those old references being guessed at.
    #[test]
    fn among_mixed_rows_only_the_shortest_measured_one_is_displaced() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        for i in 0..MAX_REFERENCES_PER_SPEAKER / 2 {
            let mut embedding = vec![0.0f32; MAX_REFERENCES_PER_SPEAKER + 1];
            embedding[i] = 1.0;
            speakers.speakers.push(EnrolledSpeaker {
                name: "Alice".to_string(),
                embedding,
                clip_seconds: None,
            });
        }
        for (offset, clip) in [44.0f64, 9.0, 61.0, 27.0, 38.0].into_iter().enumerate() {
            let mut embedding = vec![0.0f32; MAX_REFERENCES_PER_SPEAKER + 1];
            embedding[MAX_REFERENCES_PER_SPEAKER / 2 + offset] = 1.0;
            speakers.store_reference("Alice", embedding, clip);
        }

        assert_eq!(
            speakers.store_reference("Alice", over_the_cap(), 15.0),
            Stored::Replaced {
                held: MAX_REFERENCES_PER_SPEAKER,
                evicted_seconds: 9.0,
            }
        );
        assert_eq!(
            speakers
                .speakers
                .iter()
                .filter(|s| s.clip_seconds.is_none())
                .count(),
            MAX_REFERENCES_PER_SPEAKER / 2,
            "the unmeasured rows are untouched"
        );
    }

    /// The cap is per person, so somebody else being full says nothing about a new name.
    #[test]
    fn the_cap_is_per_person_rather_than_over_the_database() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        for i in 0..MAX_REFERENCES_PER_SPEAKER {
            let mut embedding = vec![0.0f32; MAX_REFERENCES_PER_SPEAKER];
            embedding[i] = 1.0;
            speakers.store_reference("Alice", embedding, CLIP);
        }

        assert_eq!(
            speakers.store_reference("Bob", vec![1.0, 0.0], CLIP),
            Stored::Enrolled
        );
    }

    /// The correction guarantee, generalised: the wrong name loses the reference built from
    /// this voice and keeps the ones built from its own recordings, and the report says how
    /// many it has left -- "Nate no longer has a reference" being a lie when Nate has two.
    #[test]
    fn forgetting_a_reference_drops_only_that_row_and_reports_what_is_left() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Nate", vec![1.0, 0.0], CLIP);
        speakers.store_reference("Nate", vec![0.0, 1.0], CLIP);
        speakers.store_reference("Ryan", vec![0.0, 1.0], CLIP);

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
        speakers.store_reference("Nate", vec![1.0, 0.0], CLIP);

        let displaced = speakers.forget_reference(&[1.0, 0.0], "Ryan");

        assert_eq!(displaced[0].remaining, 0);
        assert!(speakers.speakers.is_empty());
    }

    /// Exact equality is the condition, and nothing here is averaged, so a reference built from
    /// a *different* recording of that wrong name is a different vector and is left alone.
    #[test]
    fn forgetting_leaves_a_different_recording_of_the_wrong_name_alone() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Nate", vec![0.6, 0.8], CLIP);

        assert!(speakers.forget_reference(&[1.0, 0.0], "Ryan").is_empty());
        assert_eq!(speakers.references("Nate"), 1);
    }

    /// A person is every row bearing their name, so the list of people is the list of names --
    /// deduplicated, and in the order enrolment put them in rather than sorted.
    #[test]
    fn the_enrolled_names_are_deduplicated_in_first_appearance_order() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Silas", vec![1.0, 0.0], CLIP);
        speakers.store_reference("Alice", vec![0.0, 1.0], CLIP);
        speakers.store_reference("Silas", vec![0.6, 0.8], CLIP);

        assert_eq!(speakers.enrolled_names(), ["Silas", "Alice"]);
    }

    #[test]
    fn an_empty_database_has_nobody_enrolled() {
        assert!(
            EnrolledSpeakers::new(Vec::new())
                .enrolled_names()
                .is_empty()
        );
    }

    /// The handle-to-row mapping: position 2 of Alice's three is Alice's second row in file
    /// order, and nothing else in the database moves.
    #[test]
    fn without_drops_the_addressed_row_and_only_that_row() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Alice", vec![1.0, 0.0], CLIP);
        speakers.store_reference("Bob", vec![0.0, 1.0], CLIP);
        speakers.store_reference("Alice", vec![0.6, 0.8], CLIP);
        speakers.store_reference("Alice", vec![0.8, -0.6], CLIP);

        let rest = speakers.without("Alice", 2).unwrap();

        assert_eq!(rest.references("Alice"), 2);
        assert_eq!(rest.references("Bob"), 1);
        let held: Vec<(&str, &[f32])> = rest
            .speakers
            .iter()
            .map(|s| (s.name.as_str(), s.embedding.as_slice()))
            .collect();
        assert_eq!(
            held,
            [
                ("Alice", [1.0, 0.0].as_slice()),
                ("Bob", [0.0, 1.0].as_slice()),
                ("Alice", [0.8, -0.6].as_slice()),
            ],
            "file order is preserved either side of the removed row"
        );
        // Pure: the database it was asked of is untouched, which is what makes it usable as a
        // counterfactual rather than only as a write.
        assert_eq!(speakers.references("Alice"), 3);
    }

    /// Removing a name's last reference leaves that name with none, which is the case a caller
    /// has to be able to reach: the whole point of a per-reference handle is that it also
    /// addresses the degenerate one-row person.
    #[test]
    fn without_the_last_row_leaves_that_name_with_none() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Alice", vec![1.0, 0.0], CLIP);

        let rest = speakers.without("Alice", 1).unwrap();

        assert_eq!(rest.references("Alice"), 0);
        assert!(rest.enrolled_names().is_empty());
    }

    /// The three ways of naming a reference that is not there are one outcome, because the
    /// caller says the same thing about each: that is not a reference this name holds.
    #[test]
    fn without_a_reference_that_is_not_there_is_none() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Alice", vec![1.0, 0.0], CLIP);
        speakers.store_reference("Alice", vec![0.0, 1.0], CLIP);

        assert!(speakers.without("Alice", 3).is_none(), "past the end");
        assert!(
            speakers.without("Alice", 0).is_none(),
            "positions are 1-based"
        );
        assert!(speakers.without("Bob", 1).is_none(), "not enrolled at all");
    }

    /// Removing a person is removing every row bearing their name, and nobody else's: the other
    /// names keep their references, in the order they were in either side of the rows that went.
    #[test]
    fn without_person_drops_every_row_of_that_name_and_nothing_else() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Alice", vec![1.0, 0.0], CLIP);
        speakers.store_reference("Bob", vec![0.0, 1.0], CLIP);
        speakers.store_reference("Alice", vec![0.6, 0.8], CLIP);
        speakers.store_reference("Cara", vec![0.8, -0.6], CLIP);
        speakers.store_reference("Alice", vec![-0.6, 0.8], CLIP);

        let rest = speakers.without_person("Alice").unwrap();

        assert_eq!(rest.references("Alice"), 0);
        assert_eq!(rest.enrolled_names(), ["Bob", "Cara"]);
        let held: Vec<(&str, &[f32])> = rest
            .speakers
            .iter()
            .map(|s| (s.name.as_str(), s.embedding.as_slice()))
            .collect();
        assert_eq!(
            held,
            [
                ("Bob", [0.0, 1.0].as_slice()),
                ("Cara", [0.8, -0.6].as_slice()),
            ],
            "the survivors keep their file order"
        );
        // Pure, like `without`: the database it was asked of is what a preview then compares
        // against.
        assert_eq!(speakers.references("Alice"), 3);
    }

    /// The same miss `without` reports, for the same reason: the caller says "that is not stored"
    /// either way. Names match exactly, so a case slip is a different person and reads as absent.
    #[test]
    fn without_person_who_is_not_stored_is_none() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Alice", vec![1.0, 0.0], CLIP);

        assert!(speakers.without_person("Bob").is_none(), "not enrolled");
        assert!(
            speakers.without_person("alice").is_none(),
            "names match exactly, so alice is not Alice"
        );
        assert!(
            EnrolledSpeakers::new(Vec::new())
                .without_person("Alice")
                .is_none(),
            "nobody is enrolled at all"
        );
    }

    /// Removing the only person leaves an empty database rather than a special state: it is
    /// written as `"speakers": []` and read back as the empty one `read_or_empty` already
    /// collapses an absent file into, so a removal needs no second path to "nobody is enrolled".
    #[test]
    fn removing_the_only_person_round_trips_as_an_empty_database() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Alice", vec![1.0, 0.0], CLIP);
        speakers.store_reference("Alice", vec![0.0, 1.0], CLIP);

        speakers
            .without_person("Alice")
            .unwrap()
            .write(&paths)
            .unwrap();

        let read = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert!(read.speakers.is_empty());
        assert!(read.enrolled_names().is_empty());
        assert_eq!(read.schema_version, ENROLLED_SPEAKERS_SCHEMA_VERSION);
        assert!(
            paths.speakers_json().is_file(),
            "the file stays, holding an empty list, rather than being deleted"
        );
    }

    /// The name being stored keeps every one of its own references: somebody else's correction
    /// must not cost a person the recordings that are genuinely of them.
    #[test]
    fn forgetting_never_touches_the_name_being_kept() {
        let mut speakers = EnrolledSpeakers::new(Vec::new());
        speakers.store_reference("Ryan", vec![1.0, 0.0], CLIP);
        speakers.store_reference("Ryan", vec![0.0, 1.0], CLIP);

        assert!(speakers.forget_reference(&[1.0, 0.0], "Ryan").is_empty());
        assert_eq!(speakers.references("Ryan"), 2);
    }
}
