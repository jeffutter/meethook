use serde::{Deserialize, Serialize};

use crate::{Error, Paths, Result, write_atomic};

/// Bumped whenever `speakers.json`'s shape changes incompatibly.
///
/// Separate from every other schema version in this crate for the reason they are all
/// separate from each other: this file lives at the root rather than in a session, is
/// written by `enroll` and read by `transcribe`, and evolves on its own schedule.
pub const ENROLLED_SPEAKERS_SCHEMA_VERSION: u32 = 1;

/// One person `enroll` has been told the name of, and the voice to recognise them by.
///
/// Deliberately just those two fields. Enrollment timestamps, source session ids and
/// re-enrollment history are all in service of *versioning* the database -- renaming,
/// removing, or re-enrolling someone whose embedding has drifted -- which v1 does not do.
/// A field added later is cheap; a field written now and reinterpreted later is not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnrolledSpeaker {
    /// What this person is called in a transcript, exactly as the user typed it.
    pub name: String,

    /// The voice fingerprint: the mean of the enrolled clusters' embeddings, L2-normalized
    /// *after* averaging.
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
    pub fn read_or_empty(paths: &Paths) -> Result<EnrolledSpeakers> {
        let path = paths.speakers_json();
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(EnrolledSpeakers::new(Vec::new()));
            }
            Err(e) => return Err(Error::io(&path, e)),
        };
        serde_json::from_slice(&bytes).map_err(|e| Error::json(&path, e))
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

    /// The representation is the contract: a reference is the *mean* of a speaker's clips,
    /// L2-normalized after averaging, and the file has to hand that back unchanged. Matching
    /// is a bare dot product on the strength of it -- so a round trip that quietly rescaled a
    /// vector, by rounding it through a shorter float format or otherwise, would move every
    /// similarity and shift the threshold under everybody.
    #[test]
    fn a_normalized_reference_survives_a_round_trip_still_normalized() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());

        // The mean of two clips, normalized afterwards, exactly as clustering produces it.
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

    /// Enrolling a second person rewrites the whole file, so the write must replace rather
    /// than append -- otherwise a re-enrollment would leave two entries with one name.
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
}
