use serde::{Deserialize, Serialize};

use crate::{Error, Result, SessionId, SessionPaths, write_atomic};

/// Bumped whenever `speaker_names.json`'s shape changes incompatibly.
///
/// Separate from every other schema version in this crate for the reason they are all
/// separate from each other: this file is written by `enroll` and read by `transcribe`, lives
/// in one session, and evolves on its own schedule.
pub const SPEAKER_NAMES_SCHEMA_VERSION: u32 = 2;

/// One voice in one session that the user named by hand, without that name becoming a
/// reference anybody could be recognised by elsewhere.
///
/// This exists because a name and a reference are not the same act. Enrolment stores a voice
/// fingerprint that every future meeting is matched against; a fragment of speech too short to
/// be a trustworthy fingerprint can still be a person the user recognised and wants named in
/// *this* transcript. Without somewhere to record that, the only channel a name has is
/// `speakers.json`, and a name that cannot go there is a name `transcribe --force` silently
/// reverts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssignedName {
    /// The cluster this name was given to, as `speaker_clusters.json` numbered it at the time.
    ///
    /// Recorded so that the file can be read by a human, and so that `enroll` can replace one
    /// voice's row without searching. It is deliberately **not** what the name is resolved
    /// through: cluster ids are stable only within one clustering run, so a re-diarization
    /// that renumbered them would move this name onto a different person's words. See
    /// [`AssignedName::embedding`].
    pub cluster: u32,

    /// What the user called that voice, exactly as typed.
    pub name: String,

    /// That cluster's centroid as it stood when the name was given. This is the handle.
    ///
    /// A reader resolves a row by finding the cluster in the *current* clustering whose
    /// embedding is this vector, exactly. Matching none, or more than one, means the
    /// clustering this name was given against no longer exists, and the row is ignored rather
    /// than applied to whichever cluster now holds [`cluster`]. The same representation and
    /// ordering contract as [`crate::SpeakerCluster::embedding`], because it is a copy of one.
    ///
    /// [`cluster`]: AssignedName::cluster
    pub embedding: Vec<f32>,
}

/// One voice in one session that the user said was *not* somebody.
///
/// The second claim kind this file holds: where [`AssignedName`] records "this cluster, by
/// embedding, is this person", a denial records the opposite -- "this cluster, by embedding,
/// is **not** this person" -- so a rejected identification stops being re-suggested on every
/// later run. It is a separate top-level list rather than a flag on [`AssignedName`] because a
/// cluster legitimately needs both at once: one voice can be asserted to be Alex while also
/// having been told it is not Ivan, and the one-row-per-cluster invariant forbids mixing the
/// two kinds in one row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeniedName {
    /// The cluster this name was denied for, as `speaker_clusters.json` numbered it at the
    /// time.
    ///
    /// Recorded for exactly the same reasons [`AssignedName::cluster`] is: so the file reads
    /// by a human and lookups need no search. Deliberately **not** what the denial is resolved
    /// through either -- see [`DeniedName::embedding`].
    pub cluster: u32,

    /// Who that voice was said not to be, exactly as typed.
    pub name: String,

    /// That cluster's centroid as it stood when the denial was given. This is the handle.
    ///
    /// A reader resolves a row by finding the cluster in the *current* clustering whose
    /// embedding is this vector, exactly. Matching none, or more than one, means the
    /// clustering this denial was given against no longer exists, and the row is ignored
    /// rather than applied to whichever cluster now holds [`cluster`]. The same
    /// representation and ordering contract as [`crate::SpeakerCluster::embedding`], because
    /// it is a copy of one.
    ///
    /// [`cluster`]: DeniedName::cluster
    pub embedding: Vec<f32>,
}

/// `speaker_names.json`: the voices in one session the user named without enrolling.
///
/// One file per session, unlike `speakers.json`, and for the opposite reason. That file is at
/// the root because naming somebody once should name them in every meeting they turn up in;
/// this one records a claim scoped to a single recording, so deleting a session should take
/// its hand-given names with it.
///
/// Not a section of `speaker_clusters.json`, which is what diarization honestly knows about
/// the audio: `enroll` reads that file and never writes it, and this is the one thing in a
/// session that comes from a person rather than from the models.
///
/// Two claim kinds live here side by side: an affirmation ([`SpeakerNames::names`]) says a
/// cluster is somebody; a denial ([`SpeakerNames::denied`]) says a cluster is *not* somebody. Denials are keyed by
/// (cluster, name) rather than by cluster alone, because a voice can deny several names while
/// affirming one -- the one-row-per-cluster rule applies within each list, not across them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerNames {
    pub schema_version: u32,
    pub session_id: SessionId,

    /// One row per cluster, ordered by cluster id.
    ///
    /// Both halves are invariants [`SpeakerNames::assign`] maintains rather than things a
    /// reader has to tolerate: one voice cannot be asserted to be two people, and two runs
    /// that gave the same answers produce the same bytes.
    pub names: Vec<AssignedName>,

    /// The denials this session carries, ordered by cluster id then name.
    ///
    /// Written even when empty -- deliberately not `skip_serializing_if`, the settled
    /// convention in this crate (`Transcript.speaker_id_confidence`, `Meeting.fit`) -- so an
    /// *absent* key keeps meaning "written by an older tool" rather than blurring into
    /// "nothing denied". Readers made before denials existed ignore the key entirely (no
    /// `deny_unknown_fields` anywhere in this crate), and readers made after read its absence
    /// as an empty list via `#[serde(default)]`.
    #[serde(default)]
    pub denied: Vec<DeniedName>,
}

impl SpeakerNames {
    pub fn new(session_id: SessionId, names: Vec<AssignedName>) -> Self {
        SpeakerNames {
            schema_version: SPEAKER_NAMES_SCHEMA_VERSION,
            session_id,
            names,
            denied: Vec::new(),
        }
    }

    /// Records that one cluster is `name`, replacing whatever this session said about it.
    ///
    /// Replace rather than append, for the same reason `speakers.json` replaces: answering the
    /// same prompt twice is a correction, and two rows for one voice would make which of them
    /// wins a question about file order.
    ///
    /// The affirmation also purges the denial of the exact same pair: denying "Ivan?" and then
    /// affirming "Ivan" leaves the second answer standing, so a file asserting both "this
    /// cluster is Ivan" and "this cluster is not Ivan" would be a self-contradiction a human
    /// reading it could not trust. Only that pair goes -- denials of other names on the same
    /// cluster stand untouched, because a voice can be affirmed as Alex while still being
    /// denied as Ivan, exactly as [`SpeakerNames::deny`] records such coexistence.
    pub fn assign(&mut self, cluster: u32, name: &str, embedding: Vec<f32>) {
        let row = AssignedName {
            cluster,
            name: name.to_string(),
            embedding,
        };
        match self.names.binary_search_by_key(&cluster, |row| row.cluster) {
            Ok(at) => self.names[at] = row,
            Err(at) => self.names.insert(at, row),
        }
        self.denied
            .retain(|row| !(row.cluster == cluster && row.name == name));
    }

    /// Records that one cluster is *not* `name`, replacing any earlier denial of the same
    /// pair.
    ///
    /// Replace rather than append, for the same reason `assign` replaces: answering the same
    /// question twice is a correction, and two rows for one (cluster, name) would make which
    /// of them wins a question about file order. Rows stay ordered by cluster id then name
    /// however they arrived, so two runs that gave the same answers write the same bytes.
    pub fn deny(&mut self, cluster: u32, name: &str, embedding: &[f32]) {
        let row = DeniedName {
            cluster,
            name: name.to_string(),
            embedding: embedding.to_vec(),
        };
        // Compare field by field rather than as a (u32, &str) tuple: the tuple form hands
        // the compiler a reference whose lifetime it cannot relate to the search target.
        let before = |r: &DeniedName| {
            r.cluster < cluster || (r.cluster == cluster && r.name.as_str() < name)
        };
        let at = self.denied.partition_point(before);
        match self.denied.get(at) {
            Some(existing) if existing.cluster == cluster && existing.name == name => {
                self.denied[at] = row
            }
            _ => self.denied.insert(at, row),
        }
    }

    /// The denials recorded against one cluster, in the file's order.
    ///
    /// An iterator rather than a slice because a cluster's denials are not necessarily
    /// contiguous in the list once other clusters' rows are interleaved between them.
    pub fn denials_for(&self, cluster: u32) -> impl Iterator<Item = &DeniedName> {
        self.denied.iter().filter(move |row| row.cluster == cluster)
    }

    /// Drops everything this session recorded about one cluster -- its name and its denials
    /// alike -- reporting whether there was anything.
    ///
    /// What enrolment calls when a voice that was named for this session only is later named
    /// again above the reference floor: one voice gets one record, so a name that has become
    /// a reference must stop also being an assignment. A redrawn voice loses its denials too:
    /// both claim kinds were about the voice as it then stood, and neither survives the
    /// clustering they were given against existing.
    pub fn forget(&mut self, cluster: u32) -> bool {
        let removed_name = match self.names.binary_search_by_key(&cluster, |row| row.cluster) {
            Ok(at) => {
                self.names.remove(at);
                true
            }
            Err(_) => false,
        };
        let before = self.denied.len();
        self.denied.retain(|row| row.cluster != cluster);
        removed_name || self.denied.len() != before
    }

    pub fn write(&self, paths: &SessionPaths) -> Result<()> {
        let path = paths.speaker_names_json();
        let mut json = serde_json::to_vec_pretty(self).map_err(|e| Error::json(&path, e))?;
        json.push(b'\n');
        write_atomic(&path, &json)
    }

    /// Reads this session's hand-given names, treating "there aren't any" as none.
    ///
    /// Absent is the normal state of every session ever recorded -- the file appears only when
    /// somebody names a voice too quiet to enrol -- so it is defined out of existence here
    /// rather than at each call site, exactly as [`crate::EnrolledSpeakers::read_or_empty`]
    /// does for the database.
    ///
    /// A file that exists and does not parse stays an error, and for the same reason it does
    /// there: a user whose hand-given names silently stopped being applied has been failed
    /// quietly.
    ///
    /// `session_id` is what an empty one is stamped with, so that a caller which then assigns
    /// a name has a complete value to write. Asked for rather than recovered from the
    /// directory name because every caller is holding it already.
    pub fn read_or_empty(paths: &SessionPaths, session_id: &SessionId) -> Result<SpeakerNames> {
        let path = paths.speaker_names_json();
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(SpeakerNames::new(session_id.clone(), Vec::new()));
            }
            Err(e) => return Err(Error::io(&path, e)),
        };
        serde_json::from_slice(&bytes).map_err(|e| Error::json(&path, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session_id() -> SessionId {
        SessionId::parse("20260809-052600").unwrap()
    }

    fn names() -> SpeakerNames {
        SpeakerNames::new(
            session_id(),
            vec![
                AssignedName {
                    cluster: 1,
                    name: "Alex".to_string(),
                    embedding: vec![0.6, 0.8],
                },
                AssignedName {
                    cluster: 4,
                    name: "Ryan".to_string(),
                    embedding: vec![0.8, -0.6],
                },
            ],
        )
    }

    #[test]
    fn a_written_file_reads_back_identical() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path().join("20260809-052600"));
        std::fs::create_dir_all(paths.dir()).unwrap();
        let written = names();

        written.write(&paths).unwrap();

        let read = SpeakerNames::read_or_empty(&paths, &session_id()).unwrap();
        assert_eq!(read, written);
        assert_eq!(read.schema_version, SPEAKER_NAMES_SCHEMA_VERSION);
    }

    /// The whole resolution rule rests on the recorded embedding being the *same vector* the
    /// clustering holds, so a round trip that rounded it would make every assignment stale on
    /// the next run. Asserted bit for bit rather than approximately: this is a copy of an
    /// array through one file format, not a measurement.
    #[test]
    fn an_embedding_survives_a_round_trip_bit_for_bit() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path().join("20260809-052600"));
        std::fs::create_dir_all(paths.dir()).unwrap();

        // Values that need every bit of an f32 to state, unlike the 0.6/0.8 above.
        let embedding: Vec<f32> = (0..64)
            .map(|i| ((i as f32) * 0.437_591_3).sin() * 0.117_211_3)
            .collect();
        SpeakerNames::new(
            session_id(),
            vec![AssignedName {
                cluster: 0,
                name: "Alex".to_string(),
                embedding: embedding.clone(),
            }],
        )
        .write(&paths)
        .unwrap();

        let read = SpeakerNames::read_or_empty(&paths, &session_id()).unwrap();
        let round_tripped: Vec<u32> = read.names[0]
            .embedding
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let expected: Vec<u32> = embedding.iter().map(|v| v.to_bits()).collect();
        assert_eq!(round_tripped, expected);
    }

    /// The normal state of every session: nobody has named a quiet voice in it, so there is no
    /// file. That has to read as "no assignments" rather than as an error, or `transcribe`
    /// would refuse to work on every session recorded before this file existed.
    #[test]
    fn an_absent_file_reads_as_no_assignments() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path().join("20260809-052600"));
        std::fs::create_dir_all(paths.dir()).unwrap();

        let read = SpeakerNames::read_or_empty(&paths, &session_id()).unwrap();

        assert!(read.names.is_empty());
        assert_eq!(read.session_id, session_id());
        assert_eq!(read.schema_version, SPEAKER_NAMES_SCHEMA_VERSION);
    }

    /// The other half of that: a file that is *there* and unreadable is not the ordinary
    /// no-assignments case, and must not be silently downgraded into one -- a user whose
    /// hand-given names stopped being applied would have no way to tell.
    #[test]
    fn a_malformed_file_is_an_error_rather_than_no_assignments() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path().join("20260809-052600"));
        std::fs::create_dir_all(paths.dir()).unwrap();
        std::fs::write(paths.speaker_names_json(), b"{ this is not json").unwrap();

        let error = SpeakerNames::read_or_empty(&paths, &session_id()).unwrap_err();

        assert!(
            matches!(error, Error::Json { .. }),
            "expected a JSON error, got {error:?}"
        );
        assert!(error.to_string().contains("speaker_names.json"), "{error}");
    }

    /// One voice, one record: answering the same prompt again is a correction, not a second
    /// claim about that voice. And rows stay in cluster order however they arrived, so two
    /// runs that gave the same answers write the same bytes.
    #[test]
    fn assigning_a_cluster_twice_replaces_the_row_and_keeps_the_order() {
        let mut names = SpeakerNames::new(session_id(), Vec::new());

        names.assign(4, "Ryan", vec![0.8, -0.6]);
        names.assign(1, "Andrew", vec![0.6, 0.8]);
        names.assign(1, "Alex", vec![0.6, 0.8]);

        assert_eq!(
            names
                .names
                .iter()
                .map(|row| (row.cluster, row.name.as_str()))
                .collect::<Vec<_>>(),
            [(1, "Alex"), (4, "Ryan")]
        );
    }

    /// Denying a guess and then affirming the very same name is a retraction of the denial:
    /// the assignment stands alone, and every other denial on the cluster keeps standing
    /// beside it.
    #[test]
    fn assigning_a_denied_pair_purges_only_that_denial() {
        let mut names = SpeakerNames::new(session_id(), Vec::new());

        names.deny(1, "Ivan", &[0.6, 0.8]);
        names.deny(1, "Boris", &[0.6, 0.8]);
        names.assign(1, "Ivan", vec![0.6, 0.8]);

        assert_eq!(
            names
                .names
                .iter()
                .map(|row| (row.cluster, row.name.as_str()))
                .collect::<Vec<_>>(),
            [(1, "Ivan")]
        );
        assert_eq!(
            names
                .denied
                .iter()
                .map(|row| (row.cluster, row.name.as_str()))
                .collect::<Vec<_>>(),
            [(1, "Boris")],
            "the Boris denial stands; only the contradicted Ivan denial goes"
        );
    }

    /// A denial is the second claim kind this file holds: a cluster can be asserted to be
    /// Alex while also having been told it is not Ivan, so the round trip must carry both
    /// lists back intact, and stamp the new schema version on the way out.
    #[test]
    fn a_written_file_with_denials_reads_back_identical() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path().join("20260809-052600"));
        std::fs::create_dir_all(paths.dir()).unwrap();
        let mut written = names();
        written.deny(1, "Ivan", &[0.6, 0.8]);
        written.deny(4, "Alex", &[0.8, -0.6]);

        written.write(&paths).unwrap();

        assert_eq!(SPEAKER_NAMES_SCHEMA_VERSION, 2);
        let read = SpeakerNames::read_or_empty(&paths, &session_id()).unwrap();
        assert_eq!(read, written);
        assert_eq!(read.schema_version, SPEAKER_NAMES_SCHEMA_VERSION);
        assert_eq!(
            read.denied
                .iter()
                .map(|row| (row.cluster, row.name.as_str()))
                .collect::<Vec<_>>(),
            [(1, "Ivan"), (4, "Alex")]
        );
    }

    /// The forward half of the compatibility story: a file written by an older tool has no
    /// `denied` key at all, and it must read as "nothing denied" rather than fail -- every
    /// session recorded before denials existed would otherwise become unreadable.
    #[test]
    fn a_file_without_the_denied_key_reads_as_no_denials() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path().join("20260809-052600"));
        std::fs::create_dir_all(paths.dir()).unwrap();
        std::fs::write(
            paths.speaker_names_json(),
            r#"{
  "schema_version": 1,
  "session_id": "20260809-052600",
  "names": [
    {
      "cluster": 1,
      "name": "Alex",
      "embedding": [0.6, 0.8]
    }
  ]
}
"#, // An older tool's exact shape: three keys, no `denied`.
        )
        .unwrap();

        let read = SpeakerNames::read_or_empty(&paths, &session_id()).unwrap();

        assert_eq!(read.names.len(), 1);
        assert_eq!(read.names[0].name, "Alex");
        assert!(read.denied.is_empty());
    }

    /// The reverse half: a file that *does* carry denials must stay readable by a reader
    /// shaped like the old one. This crate has no `deny_unknown_fields`, but that guarantee
    /// is load-bearing -- the next release's readers will be exactly such structs -- so it is
    /// tested rather than assumed: the new file's bytes parse into a struct carrying only the
    /// old fields, with the names intact.
    #[test]
    fn a_file_with_denials_still_parses_into_the_old_shape() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path().join("20260809-052600"));
        std::fs::create_dir_all(paths.dir()).unwrap();
        let mut current = names();
        current.deny(1, "Ivan", &[0.6, 0.8]);
        current.write(&paths).unwrap();

        // Shaped like the struct a pre-denial build deserializes into.
        #[derive(Deserialize)]
        struct OldShape {
            schema_version: u32,
            session_id: String,
            names: Vec<AssignedName>,
        }
        let bytes = std::fs::read(paths.speaker_names_json()).unwrap();
        let old: OldShape = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(old.schema_version, 2);
        assert_eq!(old.session_id, "20260809-052600");
        assert_eq!(old.names.len(), 2);
        assert_eq!(old.names[0].name, "Alex");
        assert_eq!(old.names[1].name, "Ryan");
    }

    /// One row per (cluster, name): denying the same pair twice is a correction, and rows
    /// stay in (cluster, then name) order however they arrived, so two runs that gave the
    /// same answers write the same bytes.
    #[test]
    fn denying_a_pair_twice_replaces_the_row_and_keeps_the_order() {
        let mut names = SpeakerNames::new(session_id(), Vec::new());

        names.deny(4, "Alex", &[0.8, -0.6]);
        names.deny(1, "Ivan", &[0.6, 0.8]);
        names.deny(1, "Boris", &[0.6, 0.8]);
        names.deny(1, "Ivan", &[0.6, 0.8]); // Correction: replace, don't append.

        assert_eq!(
            names
                .denied
                .iter()
                .map(|row| (row.cluster, row.name.as_str()))
                .collect::<Vec<_>>(),
            [(1, "Boris"), (1, "Ivan"), (4, "Alex")]
        );
    }

    /// A redrawn voice loses both kinds of claims about it: the assignment and every denial
    /// recorded against the same cluster go together, and a cluster that carried only
    /// denials still reports that something was dropped.
    #[test]
    fn forgetting_a_cluster_purges_its_denials_too() {
        let mut names = names();
        names.deny(1, "Ivan", &[0.6, 0.8]);
        names.deny(7, "Alex", &[0.1, 0.2]);

        assert!(names.forget(1)); // Name and denial both gone.
        assert!(names.forget(7)); // Denial only.
        assert!(!names.forget(7)); // Nothing left.

        assert_eq!(
            names
                .names
                .iter()
                .map(|row| row.cluster)
                .collect::<Vec<_>>(),
            [4]
        );
        assert!(names.denied.is_empty());
    }

    /// Two runs that gave the same answers write the same bytes: the determinism invariant
    /// extends to the denial list, so byte-stability is asserted over a whole file, not just
    /// the names in it. Two separate directories hold the same session id, so nothing else in
    /// the file differs either.
    #[test]
    fn the_same_denies_produce_the_same_bytes() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();

        let bytes = |dir: &tempfile::TempDir| -> Vec<u8> {
            let paths = SessionPaths::new(dir.path().join("20260809-052600"));
            std::fs::create_dir_all(paths.dir()).unwrap();
            let mut names = names();
            names.deny(4, "Alex", &[0.8, -0.6]);
            names.deny(1, "Ivan", &[0.6, 0.8]);
            names.write(&paths).unwrap();
            std::fs::read(paths.speaker_names_json()).unwrap()
        };

        assert_eq!(bytes(&first), bytes(&second));
    }

    /// What enrolment calls when a voice named for one session is later enrolled properly:
    /// the two records are mutually exclusive, so one has to be able to go.
    #[test]
    fn forgetting_a_cluster_removes_only_that_row() {
        let mut names = names();

        assert!(names.forget(1));
        assert!(!names.forget(1));
        assert_eq!(
            names
                .names
                .iter()
                .map(|row| row.cluster)
                .collect::<Vec<_>>(),
            [4]
        );
    }

    /// The atomic write leaves no temp file behind, and lands on the one name the rest of the
    /// tool looks for.
    #[test]
    fn writing_leaves_exactly_one_file_in_the_session_directory() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path().join("20260809-052600"));
        std::fs::create_dir_all(paths.dir()).unwrap();

        names().write(&paths).unwrap();

        let entries: Vec<_> = std::fs::read_dir(paths.dir())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(entries, [paths.speaker_names_json().file_name().unwrap()]);
    }
}
