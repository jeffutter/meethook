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
//!   transcript.md.jinja           optional template every transcript.md is rendered through
//!   models/                       lazily fetched model weights
//!   sessions/
//!     <YYYYMMDD-HHMMSS>[-N]/      local-time id, numeric suffix on same-second collision
//!       mic.wav                   raw microphone track (never modified after recording)
//!       mic.cleaned.wav           echo-cancelled derivative, written by transcribe
//!       speaker.wav               system/speaker track
//!       session.json              presence == "valid/complete session"
//!       speaker_clusters.json     diarization output reused by enroll
//!       speaker_names.json        voices named by hand in this session only
//!       transcript.json           presence == "already transcribed"
//!       transcript.md             human-readable rendering
//! ```

mod atomic;
mod discovery;
mod id;
mod metadata;
mod paths;
mod speaker_clusters;
mod speaker_names;
mod speakers;
mod transcript;

// The header spelling of the tracks named above. A module rather than a flat re-export
// because its surface is constructors: `wav::create` reads better than `create_wav`.
// (A plain comment, not a doc comment: outer docs on a `mod` item resolve their intra-doc
// links in *this* scope, which breaks the links in wav.rs's own module doc.)
pub mod wav;

pub use atomic::{write_atomic, write_atomic_with};
pub use discovery::{Classification, DiscoveredSession, discover_sessions};
pub use id::{SessionId, create_session_dir, discard_session_dir};
pub use metadata::{
    Attendee, AttendeeStatus, Meeting, SESSION_SCHEMA_VERSION, SessionMetadata, TrackSync,
};
pub use paths::{Paths, SessionPaths};
pub use speaker_clusters::{
    MIN_REPRESENTATIVE_SECONDS, RepresentativeSegment, SPEAKER_CLUSTERS_SCHEMA_VERSION,
    SpeakerCluster, SpeakerClusters, unknown_labels,
};
pub use speaker_names::{AssignedName, SPEAKER_NAMES_SCHEMA_VERSION, SpeakerNames};
pub use speakers::{
    Displaced, ENROLLED_SPEAKERS_SCHEMA_VERSION, EnrolledSpeaker, EnrolledSpeakers,
    MAX_REFERENCES_PER_SPEAKER, Stored,
};
pub use transcript::{
    SourceTrack, TRANSCRIPT_SCHEMA_VERSION, TimestampError, Transcript, TranscriptContext,
    TranscriptTemplate, TranscriptTime, Turn, VoiceAt, YOU as SPEAKER_YOU, unknown_speaker,
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

    /// A `transcript.md` template that would not compile, or that failed while rendering.
    ///
    /// A *missing* or unreadable template file is [`Error::Io`], which already names the path
    /// and the reason; this is the file that was read and then could not be used.
    ///
    /// The `{source}` here must stay a plain `{}` and must never become `{:#}`, and nothing may
    /// call `minijinja::Error::display_debug_info`. minijinja's alternate form dumps the
    /// template's variables, and those include `Meeting::notes` -- the verbatim invite body,
    /// which that field's own documentation commits to never leaving `session.json`. The plain
    /// form is a syntax or undefined-value diagnosis with a line number and nothing else, which
    /// is all a user needs to fix a template.
    #[error("could not use the transcript template at {path}: {source}")]
    Template {
        path: PathBuf,
        #[source]
        source: minijinja::Error,
    },

    /// A file that parsed fine but claims a schema version this build has never heard of. In
    /// practice that means a newer meethook wrote it and this one has been downgraded onto it.
    ///
    /// Distinct from [`Error::Json`] on purpose: that is "these bytes are not the shape I
    /// expected", and a user's remedy is to look at the file. This is "these bytes are a shape
    /// I have no rule for", and the remedy is to move the binary rather than the data -- so the
    /// remedy is in the `Display` here rather than appended by each caller, because both
    /// readers of `speakers.json` reach it outside any per-session loop and simply `?` it.
    ///
    /// The wording covers a version below the readable range as well as above it, because the
    /// gate is a range check and a message that assumed "newer" would be a guess. `oldest` and
    /// `newest` are the range this build reads, inclusive.
    #[error(
        "{path} claims schema_version {found}, which this build does not understand (it reads \
         {oldest} through {newest}) -- upgrade meethook if a newer one wrote that file, or move \
         it aside to start a new database"
    )]
    UnsupportedSchema {
        path: PathBuf,
        found: u32,
        oldest: u32,
        newest: u32,
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

    pub(crate) fn template(path: impl Into<PathBuf>, source: minijinja::Error) -> Self {
        Error::Template {
            path: path.into(),
            source,
        }
    }
}
