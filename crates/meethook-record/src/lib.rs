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

mod activity;
mod clock;
mod exception;
mod mic;
mod preflight;
mod speaker;
mod track;

pub use activity::{Activity, MicActivityWatcher};
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

    /// An Apple framework raised an Objective-C exception instead of returning an error.
    ///
    /// Transient by nature: the calls that raise do so when the hardware moved out from under
    /// them mid-call, which is precisely what a dock being plugged in does. Reported as an
    /// ordinary error so the record loop's existing retry treats it as one -- see the
    /// `exception` module for why an uncaught one used to kill the process instead.
    #[error("{api} raised an Objective-C exception: {message}")]
    Framework { api: &'static str, message: String },

    /// Fatal at startup only: without listeners there is no trigger, so the recorder would
    /// sit there watching nothing while the user believes it is armed.
    #[error(
        "CoreAudio refused to watch {what} (status {status}), so meeting starts and ends \
         cannot be detected"
    )]
    CoreAudio { what: &'static str, status: i32 },

    #[error(
        "no default input device is selected, so there is no microphone to watch; \
         choose one in System Settings > Sound > Input"
    )]
    NoInputDevice,

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
/// Construction demands an [`Authorized`] token, which only [`preflight()`] can produce.
/// That is what keeps "permissions are checked before anything is written to disk" a
/// property of the type system rather than a convention a later edit can quietly break.
///
/// `start` -> [`RunningSession::finish`] may be called more than once on the same
/// `Recorder`, and is: the CLI drives it in a loop, one session per detected call, for as
/// long as the process runs.
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
    /// capture, finding a usable input device -- happens before the directory is created, so
    /// most failed starts never reach it. The two that can, because they are what bringing an
    /// engine up costs, discard the directory on their way out: a failed start leaves nothing
    /// behind either way.
    pub fn start(&self, paths: &Paths, now: &Zoned) -> Result<RunningSession> {
        // Resolved per session rather than cached on the Recorder: displays and input
        // devices change between meetings, and a stale handle would fail confusingly.
        let display = speaker::default_display()?;
        let input = mic::InputDevice::open()?;

        let (id, session_paths) = meethook_session::create_session_dir(paths, now)?;

        // Both engines are brought up inside a closure so that every failure past the
        // directory creation leaves by one path, and that path discards the directory. The
        // failures here are the transient ones a device swap produces, so the caller retries --
        // five times, in `begin` -- and without the discard each attempt would leave behind a
        // directory holding two empty WAV headers and no `session.json`, which at a glance
        // reads like a recording that went wrong rather than one that never started.
        let started = (|| {
            let speaker = speaker::SpeakerCapture::start(&display, &session_paths.speaker_wav())?;
            match input.start(&session_paths.mic_wav()) {
                Ok(mic) => Ok((speaker, mic)),
                Err(e) => {
                    // The speaker stream is already live; stop it so a failed start does not
                    // leave a capture running against a session nobody will finish.
                    drop(speaker.stop());
                    Err(e)
                }
            }
        })();

        let (speaker, mic) = match started {
            Ok(engines) => engines,
            Err(e) => {
                meethook_session::discard_session_dir(&session_paths);
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

    /// Whether the microphone track has stopped receiving audio.
    ///
    /// Cheap enough to ask on a poll the caller is already making: one relaxed atomic load
    /// and a comparison. Answers `true` for any reason the tap stopped delivering -- a
    /// sample-rate reconfiguration, an exclusive grab, a stream-format change, a sleep the
    /// engine did not return from -- and needs no notification to do it. A microphone that
    /// has not delivered its first buffer yet is never stalled; that case is a failed start,
    /// and [`RunningSession::finish`] already reports it as [`Error::SilentTrack`].
    pub fn mic_stalled(&mut self) -> bool {
        self.mic.stalled(std::time::Instant::now())
    }

    /// Frames the microphone tap has delivered so far. A diagnostic for the record loop's
    /// debug output, not the length of `mic.wav`.
    pub fn mic_frames_delivered(&self) -> u64 {
        self.mic.frames_delivered()
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

        let mic_ticks = mic_summary.first_host_ticks().ok_or(Error::SilentTrack {
            track: "mic",
            dir: paths.dir().to_path_buf(),
        })?;
        let speaker_ticks = speaker_summary
            .first_host_ticks()
            .ok_or(Error::SilentTrack {
                track: "speaker",
                dir: paths.dir().to_path_buf(),
            })?;

        report_first_buffer_timing(&mic_summary, &speaker_summary);

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

/// Prints how far each capture API's own timestamp sits from the moment its buffer was
/// actually delivered. Gated behind `MEETHOOK_TIMING_DEBUG`.
///
/// This exists because a stored timestamp is otherwise unfalsifiable: `session.json` says
/// the two tracks start N ms apart, and nothing in the recording can confirm or refute it.
/// A live click test showed the stored offset is wrong by at least 13 ms -- the mic appears
/// to hear a sound before the speaker emitted it -- but not *which* of the two timestamps
/// is at fault, since both APIs are asked the same question and only one can be lying.
///
/// The first buffer's gap alone cannot say this, because it cannot separate "the timestamp
/// is early" from "delivery was late" -- and the first buffer of a stream is exactly where
/// startup delay is worst. The **median** gap over the whole session is the reference that
/// makes it decidable: once a stream is running, delivery is paced by the buffer clock, so
/// the gap converges. Read `first - median`:
///
/// - `~= 0` means the stored timestamp describes its samples exactly the way every later
///   timestamp does; the recorder's arithmetic is sound and any residual sync error is
///   physical, not computed.
/// - Materially negative means the stored timestamp is *earlier* than that track's own
///   steady-state relationship to its samples, so `session.json` starts that track too
///   early.
///
/// `max drift` is a separate check: how far any buffer's timestamp strayed from the straight
/// line through the first timestamp at the track's nominal rate. A track whose timestamps
/// disagree with its own sample count has lost or gained audio, which corrupts alignment
/// however good the first timestamp was.
fn report_first_buffer_timing(mic: &track::TrackSummary, speaker: &track::TrackSummary) {
    if std::env::var_os("MEETHOOK_TIMING_DEBUG").is_none() {
        return;
    }

    eprintln!("\n[timing] diagnostics (MEETHOOK_TIMING_DEBUG)");
    for (label, summary) in [("mic", mic), ("speaker", speaker)] {
        let Some(b) = summary.first_buffer else {
            eprintln!("  {label:<8} no buffers received");
            continue;
        };
        // Signed: a timestamp *after* delivery would be a distinct and much stranger bug
        // than one before it, and must not silently wrap through u64.
        let first_gap = clock::ticks_to_millis(b.delivered_ticks as i64 - b.host_ticks as i64);
        let buffer_ms = f64::from(b.frames) / f64::from(summary.sample_rate) * 1000.0;
        eprintln!(
            "  {label:<8} buffer={} frames ({buffer_ms:.3} ms)   first gap={first_gap:+.3} ms \
             ({:.2} buffers)",
            b.frames,
            first_gap / buffer_ms,
        );

        let Some(t) = summary.timing.as_ref() else {
            continue;
        };
        let median = clock::ticks_to_millis(t.median_gap_ticks());
        let excess = first_gap - median;
        eprintln!(
            "  {:<8} {} buffers   gap median={median:+.3} min={:+.3} max={:+.3} ms   \
             max drift={:+.3} ms",
            "",
            t.buffers(),
            clock::ticks_to_millis(t.min_gap_ticks()),
            clock::ticks_to_millis(t.max_gap_ticks()),
            clock::ticks_to_millis(t.max_drift_ticks()),
        );
        eprintln!(
            "  {:<8} first - median = {excess:+.3} ms ({:.2} buffers){}",
            "",
            excess / buffer_ms,
            if excess.abs() > buffer_ms {
                "   <- OUTLIER: the stored timestamp is not typical of this track"
            } else {
                ""
            },
        );
    }

    if let (Some(m), Some(s)) = (mic.first_buffer, speaker.first_buffer) {
        let stored = clock::ticks_to_millis(m.host_ticks as i64 - s.host_ticks as i64);
        eprintln!(
            "  stored offset (mic - speaker)    {stored:+.3} ms   <- what session.json records"
        );
        // Deliberately no "corrected" offset is printed here. A first-vs-median excess is
        // tempting to subtract, but `max drift` above measures the first timestamp against
        // every later one in the same stream: when drift is ~0, the first timestamp is
        // already on the same line as the rest, and the excess is startup *delivery* -- the
        // pipeline is not yet full -- rather than a timestamp fault. Subtracting it would be
        // a calibration constant dressed up as a measurement.
    }

    // Nor is any hardware latency printed. That probe existed and was removed: CoreAudio's
    // figures proved unusable for this purpose (see `speaker.rs`), and a number on screen
    // that nobody should subtract is worse than no number at all.
    eprintln!();
}

/// A completed, on-disk session.
pub struct Recording {
    pub id: SessionId,
    pub paths: SessionPaths,
    pub metadata: SessionMetadata,
    pub mic: TrackSummary,
    pub speaker: TrackSummary,
}
