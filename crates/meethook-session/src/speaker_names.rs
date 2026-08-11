use serde::{Deserialize, Serialize};

use crate::{Error, Result, SessionId, SessionPaths, write_atomic};

/// Bumped whenever `speaker_names.json`'s shape changes incompatibly.
///
/// Separate from every other schema version in this crate for the reason they are all
/// separate from each other: this file is written by `enroll` and read by `transcribe`, lives
/// in one session, and evolves on its own schedule.
pub const SPEAKER_NAMES_SCHEMA_VERSION: u32 = 1;

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
}

impl SpeakerNames {
    pub fn new(session_id: SessionId, names: Vec<AssignedName>) -> Self {
        SpeakerNames {
            schema_version: SPEAKER_NAMES_SCHEMA_VERSION,
            session_id,
            names,
        }
    }

    /// Records that one cluster is `name`, replacing whatever this session said about it.
    ///
    /// Replace rather than append, for the same reason `speakers.json` replaces: answering the
    /// same prompt twice is a correction, and two rows for one voice would make which of them
    /// wins a question about file order.
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
    }

    /// Drops this session's name for one cluster, reporting whether there was one.
    ///
    /// What enrolment calls when a voice that was named for this session only is later named
    /// again above the reference floor: one voice gets one record, so a name that has become a
    /// reference must stop also being an assignment.
    pub fn forget(&mut self, cluster: u32) -> bool {
        match self.names.binary_search_by_key(&cluster, |row| row.cluster) {
            Ok(at) => {
                self.names.remove(at);
                true
            }
            Err(_) => false,
        }
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

    /// What enrolment calls when a voice named for one session is later enrolled properly: the
    /// two records are mutually exclusive, so one has to be able to go.
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
