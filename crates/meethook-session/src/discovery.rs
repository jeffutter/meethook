use std::fmt;
use std::fs;
use std::io::ErrorKind;

use crate::{Error, Paths, Result, SessionId, SessionMetadata, SessionPaths};

/// What state a session directory is in.
///
/// The three outcomes are not mutually exclusive on disk -- a transcribed session still has
/// its `session.json` -- so classification applies a fixed precedence:
/// `Transcribed` > `Valid` > `Orphaned`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classification {
    /// `transcript.json` present: already transcribed, skipped on rerun unless `--force`.
    Transcribed,
    /// `session.json` present: recorded cleanly, ready to transcribe.
    Valid,
    /// Neither marker: the recorder died mid-session. A normal, expected state -- never an
    /// error. Callers skip these with a warning.
    Orphaned,
}

impl Classification {
    pub fn as_str(self) -> &'static str {
        match self {
            Classification::Transcribed => "transcribed",
            Classification::Valid => "valid",
            Classification::Orphaned => "orphaned",
        }
    }
}

impl fmt::Display for Classification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A session directory found under `sessions/`, with its state already determined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredSession {
    pub id: SessionId,
    pub paths: SessionPaths,
    pub classification: Classification,
}

impl DiscoveredSession {
    /// Reads `session.json` on demand.
    ///
    /// Metadata is not loaded during discovery because most callers only need the
    /// classification, and because a `Transcribed` session still needs its metadata
    /// readable when `--force` re-runs it.
    pub fn load_metadata(&self) -> Result<SessionMetadata> {
        SessionMetadata::read(&self.paths.session_json())
    }
}

/// Lists every session directory under `sessions/`, sorted by id.
///
/// A missing or empty sessions directory is the normal first-run case and yields an empty
/// list, not an error. Entries whose names are not session ids (`models/`, `.DS_Store`, a
/// stray note file) are ignored rather than reported.
pub fn discover_sessions(paths: &Paths) -> Result<Vec<DiscoveredSession>> {
    let sessions_dir = paths.sessions_dir();

    let entries = match fs::read_dir(&sessions_dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(Error::io(&sessions_dir, e)),
    };

    let mut sessions = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| Error::io(&sessions_dir, e))?;

        let file_type = entry.file_type().map_err(|e| Error::io(entry.path(), e))?;
        if !file_type.is_dir() {
            continue;
        }

        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        let Ok(id) = SessionId::parse(&name) else {
            continue;
        };

        let session_paths = SessionPaths::new(entry.path());
        sessions.push(DiscoveredSession {
            id,
            classification: classify(&session_paths),
            paths: session_paths,
        });
    }

    sessions.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(sessions)
}

fn classify(paths: &SessionPaths) -> Classification {
    if paths.transcript_json().is_file() {
        Classification::Transcribed
    } else if paths.session_json().is_file() {
        Classification::Valid
    } else {
        Classification::Orphaned
    }
}
