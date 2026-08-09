use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{Error, Result, SessionId, SessionPaths, write_atomic};

/// Bumped whenever `speaker_clusters.json`'s shape changes incompatibly.
///
/// Separate from the transcript's and the session's versions for the same reason those two
/// are separate from each other: this file is written by diarization and read by `enroll`,
/// on its own schedule.
pub const SPEAKER_CLUSTERS_SCHEMA_VERSION: u32 = 1;

/// The shortest clip a representative segment is allowed to be.
///
/// `enroll` plays one of these and asks "who is this?". Under about a second and a half
/// there is not enough voice there for a person to answer -- a half-second fragment is a
/// syllable, and being asked to name a speaker from a syllable is a bad experience that
/// would be discovered long after this file was written.
///
/// Producers must satisfy this by widening a short segment into the surrounding track
/// rather than by dropping the cluster: a participant with no playable clip is a
/// participant who can never be named.
pub const MIN_REPRESENTATIVE_SECONDS: f64 = 1.5;

/// A stretch of `speaker.wav` to play back when asking who a cluster is.
///
/// `start` and `end` are seconds **into `speaker.wav`**, not seconds from session start.
/// This is the one place in the session contract that uses track time rather than the
/// shared timeline [`crate::Turn`] is on, and it is deliberate: the only thing anyone does
/// with these numbers is seek into that file and play, so making them offsets into it
/// removes a conversion -- and a chance to get it wrong -- from every consumer.
///
/// Guaranteed to span at least [`MIN_REPRESENTATIVE_SECONDS`], which may mean it includes
/// a little audio either side of the speech that earned it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RepresentativeSegment {
    pub start: f64,
    pub end: f64,
}

impl RepresentativeSegment {
    pub fn seconds(&self) -> f64 {
        self.end - self.start
    }
}

/// One voice on the speaker track, with no idea whose it is.
///
/// A cluster is what diarization can honestly produce on its own: everything here is
/// derived from the audio, and nothing from a person. Attaching a name is `enroll`'s job,
/// which is why there is no name field to leave empty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerCluster {
    /// Stable only within this session. Clusters are numbered by how much they spoke, most
    /// first, so `0` is the person who did most of the talking.
    pub id: u32,

    /// The voice fingerprint: the mean of this cluster's per-turn embeddings, L2-normalized
    /// *after* averaging.
    ///
    /// The order matters and is part of the contract, not an implementation detail. The
    /// mean of normalized vectors and the normalized mean are different vectors, and
    /// `speakers.json` stores enrolled speakers in this same representation so that
    /// comparing the two is a dot product. Produce it any other way and enrolment silently
    /// never matches.
    ///
    /// Length is whatever the embedding model emits (256 for the WeSpeaker checkpoint
    /// meethook ships against); this file does not pin it, so a future model change is a
    /// schema bump rather than a lie in a constant here.
    pub embedding: Vec<f32>,

    /// Total speech attributed to this cluster, in seconds. What the numbering is by, and
    /// how a consumer tells a participant from someone who coughed once.
    pub speech_seconds: f64,

    /// Clips to play when asking who this is, longest first.
    ///
    /// Each spans at least [`MIN_REPRESENTATIVE_SECONDS`]. Never empty for a cluster that
    /// made it into this file.
    pub representatives: Vec<RepresentativeSegment>,
}

/// `speaker_clusters.json`: the voices found on one session's speaker track.
///
/// Written once by diarization and read by `enroll`, which is the whole point of it being
/// on disk at all: naming speakers must not require re-running the models over the audio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerClusters {
    pub schema_version: u32,
    pub session_id: SessionId,
    pub clusters: Vec<SpeakerCluster>,
}

impl SpeakerClusters {
    pub fn new(session_id: SessionId, clusters: Vec<SpeakerCluster>) -> Self {
        SpeakerClusters {
            schema_version: SPEAKER_CLUSTERS_SCHEMA_VERSION,
            session_id,
            clusters,
        }
    }

    pub fn write(&self, paths: &SessionPaths) -> Result<()> {
        let path = paths.speaker_clusters_json();
        let mut json = serde_json::to_vec_pretty(self).map_err(|e| Error::json(&path, e))?;
        json.push(b'\n');
        write_atomic(&path, &json)
    }

    pub fn read(path: &Path) -> Result<SpeakerClusters> {
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::json(path, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clusters() -> SpeakerClusters {
        SpeakerClusters::new(
            SessionId::parse("20260809-052600").unwrap(),
            vec![SpeakerCluster {
                id: 0,
                embedding: vec![0.6, 0.8],
                speech_seconds: 42.5,
                representatives: vec![RepresentativeSegment {
                    start: 10.0,
                    end: 13.0,
                }],
            }],
        )
    }

    #[test]
    fn a_written_file_reads_back_identical() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path());
        let written = clusters();

        written.write(&paths).unwrap();

        let read = SpeakerClusters::read(&paths.speaker_clusters_json()).unwrap();
        assert_eq!(read, written);
        assert_eq!(read.schema_version, SPEAKER_CLUSTERS_SCHEMA_VERSION);
    }

    /// The atomic write leaves no temp file behind, and lands on the one name the rest of
    /// the tool looks for.
    #[test]
    fn writing_leaves_exactly_one_file_in_the_session_directory() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path());

        clusters().write(&paths).unwrap();

        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(
            entries,
            [paths.speaker_clusters_json().file_name().unwrap()]
        );
    }

    /// Rewriting is what a `--force` re-transcribe does, and it must not append or merge.
    #[test]
    fn rewriting_replaces_the_previous_contents() {
        let dir = tempfile::tempdir().unwrap();
        let paths = SessionPaths::new(dir.path());

        clusters().write(&paths).unwrap();
        let mut second = clusters();
        second.clusters.clear();
        second.write(&paths).unwrap();

        let read = SpeakerClusters::read(&paths.speaker_clusters_json()).unwrap();
        assert!(read.clusters.is_empty());
    }
}
