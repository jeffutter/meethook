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
const MEETING_OPUS: &str = "meeting.opus";
const SESSION_JSON: &str = "session.json";
const SPEAKER_CLUSTERS_JSON: &str = "speaker_clusters.json";
const SPEAKER_NAMES_JSON: &str = "speaker_names.json";
const CLEANING_JSON: &str = "cleaning.json";
const TRANSCRIPT_JSON: &str = "transcript.json";
const TRANSCRIPT_MD: &str = "transcript.md";
const TRANSCRIPT_VTT: &str = "transcript.vtt";

/// Apps excluded from the mic-activity trigger. Root-level, like the template: the list
/// applies to every session, and meethook never writes it.
const EXCLUSIONS_JSON: &str = "exclusions.json";

/// The optional user template `transcript.md` is rendered through. Root-level, not
/// per-session: see [`Paths::transcript_template`].
const TRANSCRIPT_TEMPLATE: &str = "transcript.md.jinja";

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

    /// The template every session's `transcript.md` is rendered through, when it exists.
    ///
    /// One file for the whole root rather than one per session, and that is the point rather
    /// than a convenience: `enroll` and `forget` re-render transcripts they did not write, and
    /// both reach a `Paths`. Resolving the template from the root is what makes them re-render
    /// through the same template `transcribe` used without anything having been recorded in the
    /// session directory -- and therefore without a rename ever being able to revert a
    /// transcript to the built-in default.
    ///
    /// Absent is the normal case: with no file here the built-in default is used.
    pub fn transcript_template(&self) -> PathBuf {
        self.root.join(TRANSCRIPT_TEMPLATE)
    }

    /// Apps the user has excluded from the mic-activity trigger, when the file exists.
    ///
    /// Root-level rather than per-session for the same reason as the template: the list
    /// applies to every session, and nothing in the pipeline writes it -- it is authored by
    /// the user and read once at `record` startup. Absent is the normal case and means no
    /// exclusions; see the exclusions module for what a corrupt one does instead.
    pub fn exclusions_json(&self) -> PathBuf {
        self.root.join(EXCLUSIONS_JSON)
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

    /// Both tracks mixed to one compressed stereo file, for listening back to the meeting.
    ///
    /// The `.opus` extension rather than `.ogg`: the contents are an Ogg stream either way,
    /// but every player that matters keys on `.opus` for an Opus-in-Ogg file, and it says
    /// which of the two Ogg payloads this is without opening it.
    ///
    /// Written by `transcribe`, and derived rather than authoritative -- deleting it costs a
    /// re-transcribe, not a recording. Nothing classifies a session by it.
    pub fn meeting_opus(&self) -> PathBuf {
        self.dir.join(MEETING_OPUS)
    }

    /// Presence marks the session as valid/complete.
    pub fn session_json(&self) -> PathBuf {
        self.dir.join(SESSION_JSON)
    }

    pub fn speaker_clusters_json(&self) -> PathBuf {
        self.dir.join(SPEAKER_CLUSTERS_JSON)
    }

    /// Voices in this session the user named without enrolling. Absent until one is.
    pub fn speaker_names_json(&self) -> PathBuf {
        self.dir.join(SPEAKER_NAMES_JSON)
    }

    /// Outcome of the AEC pre-pass over `mic.wav`. Absent for a session transcribed before this
    /// file existed.
    pub fn cleaning_json(&self) -> PathBuf {
        self.dir.join(CLEANING_JSON)
    }

    /// Presence marks the session as already transcribed.
    pub fn transcript_json(&self) -> PathBuf {
        self.dir.join(TRANSCRIPT_JSON)
    }

    pub fn transcript_md(&self) -> PathBuf {
        self.dir.join(TRANSCRIPT_MD)
    }

    /// The same turns as WebVTT captions, for players and for tools that read subtitles.
    ///
    /// Written beside `transcript.md` by everything that writes one, so a session never holds a
    /// caption file that disagrees with its transcript about who said what.
    pub fn transcript_vtt(&self) -> PathBuf {
        self.dir.join(TRANSCRIPT_VTT)
    }
}
