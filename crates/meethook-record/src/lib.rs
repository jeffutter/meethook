//! Dual-track macOS meeting capture.
//!
//! Two independent capture engines write two independent WAV tracks into one
//! [`meethook_session`] session directory:
//!
//! - **speaker** -- system-wide audio from ScreenCaptureKit (`SCStream`)
//! - **mic** -- the default input device, from a separate `AVAudioEngine` input tap
//!
//! The split is deliberate. macOS 15's unified `SCStream` microphone output is a known
//! source of corrupted recordings when combined with capture, and, more importantly, echo
//! cancellation later needs the speaker track to exist as an *independent* reference
//! signal. One stream carrying both would make that impossible.
//!
//! Neither track is resampled or format-converted here. Each is written as mono 32-bit
//! float at whatever rate its device reports, because 32-bit float is what both engines
//! deliver natively and any conversion in the recorder would be a lossy transform applied
//! to audio that can never be re-captured. Rate handling belongs to `transcribe`.
//!
//! Alignment data is *captured*, not computed: each track's first delivered buffer's host
//! timestamp is written to `session.json` as raw mach ticks plus the timebase ratio. The
//! offset arithmetic is `transcribe`'s job.

mod clock;
mod mic;
mod preflight;
mod speaker;
mod track;

pub use preflight::{Authorized, MissingPermissions, preflight};
pub use track::TrackSummary;

use std::path::PathBuf;
use std::time::Duration;

use jiff::{Timestamp, Zoned};
use meethook_session::{Paths, SessionId, SessionMetadata, SessionPaths};

/// Everything capture can fail with.
///
/// Every variant is phrased as something the user can act on, because the entire point of
/// this slice's error handling is that a recording problem is discovered *before* a meeting
/// rather than after it.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// `transparent` rather than `"{0}"`: the latter makes `anyhow` print this long,
    /// multi-line message twice -- once as the error and once as its own cause.
    #[error(transparent)]
    Permissions(#[from] MissingPermissions),

    #[error("ScreenCaptureKit reported no displays, so there is no system audio to capture")]
    NoDisplay,

    #[error("ScreenCaptureKit failed: {0}")]
    ScreenCaptureKit(String),

    #[error("ScreenCaptureKit did not respond within {0:?}")]
    ScreenCaptureKitTimeout(Duration),

    /// A zero-ish input format is what a missing input device or a revoked microphone grant
    /// looks like. Installing a tap on it yields a silent, empty file -- the silent failure
    /// this recorder exists to eliminate -- so it is an error at start, not a surprise later.
    #[error(
        "the default input device reported an unusable format ({sample_rate} Hz, {channels} channels); \
         check that an input device is connected and selected in System Settings > Sound > Input"
    )]
    UnusableInputFormat { sample_rate: f64, channels: u32 },

    #[error("AVAudioEngine failed to start: {0}")]
    AudioEngine(String),

    /// Writing `session.json` for a track that produced nothing would classify a broken
    /// recording as valid, which is the one thing the session contract must never do.
    #[error(
        "the {track} track received no audio during the whole session, so this recording is \
         not usable; leaving {dir} without session.json rather than marking it valid"
    )]
    SilentTrack { track: &'static str, dir: PathBuf },

    #[error("wav error at {path}: {source}")]
    Wav {
        path: PathBuf,
        #[source]
        source: hound::Error,
    },

    #[error("i/o error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("the wav writer thread for {path} panicked")]
    WriterPanic { path: PathBuf },

    #[error(transparent)]
    Session(#[from] meethook_session::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn wav(path: impl Into<PathBuf>, source: hound::Error) -> Self {
        Error::Wav {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}

/// A configured, not-yet-running recorder.
///
/// Construction demands an [`Authorized`] token, which only [`preflight`] can produce.
/// That is what keeps "permissions are checked before anything is written to disk" a
/// property of the type system rather than a convention a later edit can quietly break.
///
/// `start` -> [`RunningSession::finish`] may be called more than once on the same
/// `Recorder`. This slice calls it once; the auto start/stop slice calls it in a loop, and
/// designing for that now avoids redesigning the API then.
pub struct Recorder {
    _authorized: Authorized,
}

impl Recorder {
    pub fn new(authorized: Authorized) -> Result<Recorder> {
        Ok(Recorder {
            _authorized: authorized,
        })
    }

    /// Creates a session directory and starts both capture engines.
    ///
    /// Everything that can fail without touching the filesystem -- resolving the display to
    /// capture, reading the input device's format -- happens before the directory is
    /// created, so a failed start leaves nothing behind to clean up.
    pub fn start(&self, paths: &Paths, now: &Zoned) -> Result<RunningSession> {
        // Resolved per session rather than cached on the Recorder: displays and input
        // devices change between meetings, and a stale handle would fail confusingly.
        let display = speaker::default_display()?;
        let input = mic::InputDevice::open()?;

        let (id, session_paths) = meethook_session::create_session_dir(paths, now)?;

        let speaker = speaker::SpeakerCapture::start(&display, &session_paths.speaker_wav())?;
        let mic = match input.start(&session_paths.mic_wav()) {
            Ok(mic) => mic,
            Err(e) => {
                // The speaker stream is already live; stop it so a failed start does not
                // leave a capture running against a session nobody will finish.
                drop(speaker.stop());
                return Err(e);
            }
        };

        Ok(RunningSession {
            id,
            paths: session_paths,
            start_time: now.timestamp(),
            mic,
            speaker,
        })
    }
}

/// A session with both engines live.
pub struct RunningSession {
    id: SessionId,
    paths: SessionPaths,
    start_time: Timestamp,
    mic: mic::MicCapture,
    speaker: speaker::SpeakerCapture,
}

impl RunningSession {
    pub fn id(&self) -> &SessionId {
        &self.id
    }

    pub fn paths(&self) -> &SessionPaths {
        &self.paths
    }

    /// The input device's own rate, as reported. Printed at start so the user can see that
    /// the mic engine actually came up.
    pub fn mic_sample_rate(&self) -> u32 {
        self.mic.sample_rate()
    }

    pub fn mic_channels(&self) -> u32 {
        self.mic.channels()
    }

    pub fn speaker_sample_rate(&self) -> u32 {
        self.speaker.sample_rate()
    }

    /// Stops both engines, finalizes both WAV headers, and writes `session.json`.
    ///
    /// The metadata write is last and atomic, so the presence of `session.json` keeps
    /// meaning "this recording completed and both tracks are finalized".
    pub fn finish(self) -> Result<Recording> {
        let RunningSession {
            id,
            paths,
            start_time,
            mic,
            speaker,
        } = self;

        // Stop both before inspecting either result: returning early on a mic failure would
        // leave the speaker WAV unfinalized, which is a worse outcome than a late error.
        let mic_stop = mic.stop();
        let speaker_stop = speaker.stop();

        let mic_summary = mic_stop?;
        let speaker_summary = speaker_stop?;

        let mic_ticks = mic_summary.first_host_ticks.ok_or(Error::SilentTrack {
            track: "mic",
            dir: paths.dir().to_path_buf(),
        })?;
        let speaker_ticks = speaker_summary.first_host_ticks.ok_or(Error::SilentTrack {
            track: "speaker",
            dir: paths.dir().to_path_buf(),
        })?;

        let metadata = SessionMetadata::new(
            id.clone(),
            start_time,
            clock::track_sync(mic_ticks),
            clock::track_sync(speaker_ticks),
        );
        metadata.write(&paths.session_json())?;

        Ok(Recording {
            id,
            paths,
            metadata,
            mic: mic_summary,
            speaker: speaker_summary,
        })
    }
}

/// A completed, on-disk session.
pub struct Recording {
    pub id: SessionId,
    pub paths: SessionPaths,
    pub metadata: SessionMetadata,
    pub mic: TrackSummary,
    pub speaker: TrackSummary,
}
