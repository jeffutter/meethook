use std::path::Path;

use jiff::Timestamp;
use serde::{Deserialize, Serialize};

use crate::{Error, Result, SessionId, write_atomic};

/// Bumped whenever `session.json`'s shape changes incompatibly.
///
/// `transcript.json` carries its own version; see [`crate::TRANSCRIPT_SCHEMA_VERSION`].
pub const SESSION_SCHEMA_VERSION: u32 = 1;

/// One track's first-sample host timestamp, in the form the hardware reported it.
///
/// Ticks are stored raw, together with the `mach_timebase_info` ratio needed to interpret
/// them, rather than pre-converted to nanoseconds. Converting at write time would round
/// once here and again when `transcribe` computes the mic/speaker offset; keeping the
/// native tick count lets `transcribe` do a single exact rational conversion.
///
/// Nanoseconds = `host_ticks * timebase_numer / timebase_denom`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrackSync {
    /// Raw `mach_absolute_time()` value at the track's first delivered sample.
    pub host_ticks: u64,
    /// `mach_timebase_info.numer`.
    pub timebase_numer: u32,
    /// `mach_timebase_info.denom`.
    pub timebase_denom: u32,
}

/// `session.json`: the marker that a session shut down cleanly, plus the sync data
/// `transcribe` needs to put the two tracks on one timeline.
///
/// Sample rate, channel count, and bit depth are deliberately absent. They live in the WAV
/// headers, which are the authority; duplicating them here would create two sources of
/// truth that can silently disagree after any format change in the recorder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub session_id: SessionId,
    pub schema_version: u32,
    /// Session start as an unambiguous instant, RFC 3339. The local wall-clock time is
    /// already encoded in the session id.
    pub start_time: Timestamp,
    pub mic: TrackSync,
    pub speaker: TrackSync,
}

impl SessionMetadata {
    pub fn new(
        session_id: SessionId,
        start_time: Timestamp,
        mic: TrackSync,
        speaker: TrackSync,
    ) -> Self {
        SessionMetadata {
            session_id,
            schema_version: SESSION_SCHEMA_VERSION,
            start_time,
            mic,
            speaker,
        }
    }

    /// Writes `session.json` atomically.
    ///
    /// Atomicity is what makes presence-of-file a trustworthy "session is complete" marker:
    /// a reader either sees no file or sees a whole one, never a truncated fragment that
    /// would classify a crashed session as valid.
    pub fn write(&self, path: &Path) -> Result<()> {
        let mut json = serde_json::to_vec_pretty(self).map_err(|e| Error::json(path, e))?;
        json.push(b'\n');
        write_atomic(path, &json)
    }

    pub fn read(path: &Path) -> Result<SessionMetadata> {
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        serde_json::from_slice(&bytes).map_err(|e| Error::json(path, e))
    }
}
