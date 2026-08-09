//! The meethook on-disk session contract.
//!
//! `record`, `transcribe`, and `enroll` share no process, no IPC, and no state other than
//! the directory layout described here. That makes this crate the single seam the whole
//! tool hangs off: every path, file name, id format, and metadata field lives here and
//! nowhere else, so a change to the layout is a change to one crate.
//!
//! Layout, rooted at `~/meethook` by default:
//!
//! ```text
//! <root>/
//!   speakers.json                 enrolled-speaker embedding DB
//!   models/                       lazily fetched model weights
//!   sessions/
//!     <YYYYMMDD-HHMMSS>[-N]/      local-time id, numeric suffix on same-second collision
//!       mic.wav                   raw microphone track (never modified after recording)
//!       mic.cleaned.wav           echo-cancelled derivative, written by transcribe
//!       speaker.wav               system/speaker track
//!       session.json              presence == "valid/complete session"
//!       speaker_clusters.json     diarization output reused by enroll
//!       transcript.json           presence == "already transcribed"
//!       transcript.md             human-readable rendering
//! ```

mod atomic;
mod discovery;
mod id;
mod metadata;
mod paths;
mod speaker_clusters;
mod speakers;
mod transcript;

pub use atomic::{write_atomic, write_atomic_with};
pub use discovery::{Classification, DiscoveredSession, discover_sessions};
pub use id::{SessionId, create_session_dir};
pub use metadata::{SESSION_SCHEMA_VERSION, SessionMetadata, TrackSync};
pub use paths::{Paths, SessionPaths};
pub use speaker_clusters::{
    MIN_REPRESENTATIVE_SECONDS, RepresentativeSegment, SPEAKER_CLUSTERS_SCHEMA_VERSION,
    SpeakerCluster, SpeakerClusters, unknown_labels,
};
pub use speakers::{ENROLLED_SPEAKERS_SCHEMA_VERSION, EnrolledSpeaker, EnrolledSpeakers};
pub use transcript::{
    SourceTrack, TRANSCRIPT_SCHEMA_VERSION, Transcript, Turn, YOU as SPEAKER_YOU, unknown_speaker,
};

use std::path::PathBuf;

/// Everything that can go wrong while working with the on-disk contract.
///
/// Note what is *not* in here: an orphaned session (WAVs but no `session.json`, i.e. a
/// crash mid-recording) is a normal, expected classification, never an error.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("malformed session id {0:?}: expected YYYYMMDD-HHMMSS with an optional -N suffix")]
    MalformedSessionId(String),

    #[error(
        "could not allocate a session directory for {base:?}: {tried} same-second ids already exist"
    )]
    SessionIdExhausted { base: String, tried: u32 },

    #[error("i/o error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("malformed JSON at {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn json(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        Error::Json {
            path: path.into(),
            source,
        }
    }
}
