use std::path::{Path, PathBuf};

use crate::SessionId;

/// File names inside a session directory.
///
/// Several of these are written by slices that do not exist yet. They live here anyway:
/// the point of this module is that a name is spelled exactly once in the codebase, and a
/// name that is first spelled in the slice that needs it will be spelled twice by the time
/// two slices need it.
const MIC_WAV: &str = "mic.wav";
const MIC_CLEANED_WAV: &str = "mic.cleaned.wav";
const SPEAKER_WAV: &str = "speaker.wav";
const SESSION_JSON: &str = "session.json";
const SPEAKER_CLUSTERS_JSON: &str = "speaker_clusters.json";
const TRANSCRIPT_JSON: &str = "transcript.json";
const TRANSCRIPT_MD: &str = "transcript.md";

/// The meethook data directory and everything directly under it.
///
/// Construct one from the resolved root (`--root`, `$MEETHOOK_ROOT`, or `~/meethook`) and
/// never build a meethook path by string concatenation anywhere else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Paths {
    root: PathBuf,
}

impl Paths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Paths { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn sessions_dir(&self) -> PathBuf {
        self.root.join("sessions")
    }

    /// Lazily fetched model weights. Deliberately outside the Nix closure.
    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    /// The enrolled-speaker embedding database.
    pub fn speakers_json(&self) -> PathBuf {
        self.root.join("speakers.json")
    }

    pub fn session(&self, id: &SessionId) -> SessionPaths {
        SessionPaths::new(self.sessions_dir().join(id.as_str()))
    }
}

/// One session directory's contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionPaths {
    dir: PathBuf,
}

impl SessionPaths {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        SessionPaths { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The raw microphone track. Never rewritten after recording finishes.
    pub fn mic_wav(&self) -> PathBuf {
        self.dir.join(MIC_WAV)
    }

    /// The echo-cancelled derivative of [`Self::mic_wav`], written by `transcribe`.
    pub fn mic_cleaned_wav(&self) -> PathBuf {
        self.dir.join(MIC_CLEANED_WAV)
    }

    pub fn speaker_wav(&self) -> PathBuf {
        self.dir.join(SPEAKER_WAV)
    }

    /// Presence marks the session as valid/complete.
    pub fn session_json(&self) -> PathBuf {
        self.dir.join(SESSION_JSON)
    }

    pub fn speaker_clusters_json(&self) -> PathBuf {
        self.dir.join(SPEAKER_CLUSTERS_JSON)
    }

    /// Presence marks the session as already transcribed.
    pub fn transcript_json(&self) -> PathBuf {
        self.dir.join(TRANSCRIPT_JSON)
    }

    pub fn transcript_md(&self) -> PathBuf {
        self.dir.join(TRANSCRIPT_MD)
    }
}
