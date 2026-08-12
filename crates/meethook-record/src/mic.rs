//! Microphone capture via a dedicated `AVAudioEngine` input tap.
//!
//! This is a completely separate engine from the ScreenCaptureKit stream in `speaker.rs`,
//! and that separation is the design, not an accident. The reason is native-format capture:
//! `SCStreamConfiguration` fixes the audio format for the whole stream, so pulling the mic
//! through macOS 15's unified `SCStream` microphone output would deliver it resampled to the
//! stream's rate -- a 44.1 kHz USB mic silently converted to 48 kHz -- which is precisely the
//! lossy transform this recorder exists to avoid. Keeping our own engine also means an
//! unusable device format is visible here, before a session directory exists.
//!
//! Not the reason: the widely-cited report of corrupted output from `captureMicrophone`
//! describes an `AVAssetWriter` container error, and this recorder muxes nothing. Nor is it
//! echo cancellation -- SCK delivers microphone audio as its own output type rather than
//! mixed into system audio, so the unified API would leave the speaker track intact as an
//! independent reference signal.
//!
//! The tap writes whatever the device reports -- its own sample rate, its own float32
//! samples -- with no conversion. `transcribe` owns all rate handling.
//!
//! Whether the tap is still *alive* is judged from the track's own delivered frame count
//! rather than from any notification, because the frame count is the general signal. An
//! input tap delivers buffers continuously while its engine runs -- a silent room arrives as
//! zeros, not as an absence of buffers -- so a frame count standing still means the engine
//! has stopped, whatever stopped it: a sample-rate reconfiguration, an exclusive grab, a
//! stream-format change under the running engine, or a sleep the engine did not come back
//! from. None of those post a default-input-device change, and only some of them post
//! `AVAudioEngineConfigurationChangeNotification`. See [`MicLiveness`].

use std::path::Path;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

use block2::RcBlock;
use objc2::rc::Retained;
use objc2_avf_audio::{
    AVAudioEngine, AVAudioFormat, AVAudioInputNode, AVAudioNodeTapBlock, AVAudioPCMBuffer,
    AVAudioTime,
};

use crate::track::{TrackProgress, TrackSummary, TrackWriter};
use crate::{Error, Result};

/// Tap buffer size in frames. Large enough that the callback rate stays low (roughly 12 Hz
/// at 48 kHz), small enough that a stop never has to wait long for the last buffer.
const TAP_BUFFER_FRAMES: u32 = 4096;

/// How many buffer periods a tap may go without delivering before the only remaining
/// explanation is a dead engine.
///
/// A working tap is driven by the device clock and does not miss buffers, so this only has
/// to be past the noise floor rather than generous.
const STALL_BUFFER_PERIODS: u32 = 8;

/// The shortest stall limit the arithmetic is allowed to produce.
///
/// Exists so a fast device cannot yield a limit shorter than an ordinary scheduling hiccup
/// on the *reader* side. At 48 kHz eight 4096-frame periods is 683 ms, so in the common case
/// this floor is what governs.
const MIN_STALL_LIMIT: Duration = Duration::from_secs(1);

/// How long a tap delivering `buffer_frames`-frame buffers at `sample_rate` may go without
/// delivering one before the only remaining explanation is a dead engine.
///
/// Scaled to the buffer size *actually observed* rather than to [`TAP_BUFFER_FRAMES`],
/// because `installTapOnBus`'s buffer size is a request the framework need not honour, and a
/// limit computed from a size the device is not using would be a guess wearing a
/// computation's clothes.
///
/// Deliberately uncapped at the top: a Bluetooth device at 8 kHz delivers a 4096-frame
/// buffer only every 512 ms, and the same rule then allows it 4.1 s. The slower a device
/// delivers, the longer a gap has to be before it means anything.
fn stall_limit(buffer_frames: u32, sample_rate: u32) -> Duration {
    // `max(1)` guards the division rather than a reachable case: `InputDevice::open` refuses
    // a non-positive rate, so a zero here would be a bug elsewhere, and panicking in a
    // liveness check would be a worse way to report it.
    let period = Duration::from_secs_f64(f64::from(buffer_frames) / f64::from(sample_rate.max(1)));
    (period * STALL_BUFFER_PERIODS).max(MIN_STALL_LIMIT)
}

/// Whether the microphone tap is still delivering audio.
///
/// The signal is the track itself rather than any notification -- see the module docs for
/// why. Nothing is judged until the first buffer has arrived, and both halves of that rule
/// are load-bearing:
///
/// - A device still coming up has not stalled. Declaring one dead would finalize and restart
///   the session, whose new engine would also be slow to come up, and so on: a session
///   directory every few seconds for the length of the meeting, which is a worse failure than
///   the one being fixed.
/// - It defuses the only false positive the counter has. `MicCapture`'s tap drops buffers
///   with an invalid host time, so a device delivering only those never advances the count --
///   but it never arms either, so no stall is declared, the session runs to its natural end,
///   and `RunningSession::finish` reports `SilentTrack` exactly as it does today.
///
/// So a microphone that never delivers a single buffer is explicitly **not** what this
/// detects. That is a failed start, and it is already reported at finalize.
struct MicLiveness {
    sample_rate: u32,
    /// `None` until the first buffer arrives.
    armed: Option<Armed>,
}

/// The state of a track that has delivered at least one buffer.
struct Armed {
    /// How long a standstill has to last before it can only mean a dead engine.
    limit: Duration,
    /// The count at the last observation that advanced, and when that was.
    frames: u64,
    since: Instant,
}

impl MicLiveness {
    fn new(sample_rate: u32) -> MicLiveness {
        MicLiveness {
            sample_rate,
            armed: None,
        }
    }

    /// Records what the tap has delivered and answers whether it has stopped.
    ///
    /// `now` is a parameter rather than read inside, so the rule is decidable in a test
    /// without sleeping -- the same reason the record loop takes its `Timing` as a value.
    ///
    /// The standstill is measured from the last observation that *advanced*, not from the
    /// last call, so asking more often can neither delay nor provoke a stall.
    fn observe(&mut self, progress: &TrackProgress, now: Instant) -> bool {
        let frames = progress.frames();
        let Some(armed) = self.armed.as_mut() else {
            // Not armed yet. `first_buffer` is `Some` by the time `frames` is non-zero, since
            // the callback publishes both before it returns, so this is also where the
            // observed buffer size for the limit comes from.
            if let Some(first) = progress.first_buffer().filter(|_| frames > 0) {
                self.armed = Some(Armed {
                    limit: stall_limit(first.frames, self.sample_rate),
                    frames,
                    since: now,
                });
            }
            return false;
        };

        if frames > armed.frames {
            armed.frames = frames;
            armed.since = now;
            return false;
        }
        now.saturating_duration_since(armed.since) > armed.limit
    }
}

/// The default input device, opened and validated but not yet tapped.
///
/// Splitting this from [`MicCapture`] is what lets [`crate::Recorder::start`] discover an
/// unusable input *before* it creates a session directory.
pub struct InputDevice {
    engine: Retained<AVAudioEngine>,
    input: Retained<AVAudioInputNode>,
    format: Retained<AVAudioFormat>,
    sample_rate: u32,
    channels: u32,
}

impl InputDevice {
    pub fn open() -> Result<InputDevice> {
        let engine = unsafe { AVAudioEngine::new() };
        // SAFETY: `inputNode` is a plain property read; on macOS it is always present, and
        // its *format* is what reveals whether a usable device is actually behind it.
        let input = unsafe { engine.inputNode() };
        let format = unsafe { input.outputFormatForBus(0) };

        // SAFETY: plain property reads on a live format object.
        let sample_rate = unsafe { format.sampleRate() };
        let channels = unsafe { format.channelCount() };

        // A zero format is exactly what a missing input device or a revoked microphone grant
        // looks like. Installing a tap on it yields a silent, empty file, which is the silent
        // failure this recorder exists to eliminate -- so refuse here instead.
        if sample_rate.is_nan() || sample_rate <= 0.0 || channels == 0 {
            return Err(Error::UnusableInputFormat {
                sample_rate,
                channels,
            });
        }

        Ok(InputDevice {
            engine,
            input,
            format,
            sample_rate: sample_rate as u32,
            channels,
        })
    }

    /// Installs the tap and starts the engine, writing to `path`.
    pub fn start(self, path: &Path) -> Result<MicCapture> {
        let InputDevice {
            engine,
            input,
            format,
            sample_rate,
            channels,
        } = self;

        let writer = TrackWriter::create(path, sample_rate)?;
        let sink = writer.sink();

        // Non-interleaved float32 is what AVAudioEngine delivers for an input tap, so stride
        // is 1; reading it from the format anyway means an interleaved device degrades to
        // correct-but-slower rather than to garbage.
        let stride = if unsafe { format.isInterleaved() } {
            channels as usize
        } else {
            1
        };

        let tap = RcBlock::new(
            move |buffer: NonNull<AVAudioPCMBuffer>, when: NonNull<AVAudioTime>| {
                // Read first, before any work, so it measures when this callback arrived
                // rather than how long it took.
                let delivered_ticks = crate::clock::now_ticks();

                // SAFETY: AVAudioEngine passes live objects that outlive the callback.
                let buffer = unsafe { buffer.as_ref() };
                let when = unsafe { when.as_ref() };

                // An invalid host time cannot be aligned against the speaker track, and a
                // fabricated one would be worse than a slightly later first timestamp.
                if !unsafe { when.isHostTimeValid() } {
                    return;
                }
                let host_ticks = unsafe { when.hostTime() };

                let frames = unsafe { buffer.frameLength() } as usize;
                if frames == 0 {
                    return;
                }

                let channel_data = unsafe { buffer.floatChannelData() };
                if channel_data.is_null() {
                    return;
                }

                // SAFETY: `floatChannelData` returns an array of at least `channelCount`
                // pointers, each valid for `frameLength * stride` samples. `NonNull<f32>` and
                // `*const f32` have identical layout.
                let mono = unsafe {
                    crate::track::channel_zero(
                        channel_data.cast::<*const f32>().cast_const(),
                        frames,
                        stride,
                    )
                };
                sink.push(host_ticks, delivered_ticks, mono);
            },
        );

        // SAFETY: the tap block is retained by the framework and kept alive by the `RcBlock`
        // stored in the returned struct, which lives until `removeTapOnBus`.
        unsafe {
            let block: AVAudioNodeTapBlock = (&*tap as *const block2::DynBlock<_>).cast_mut();
            input.installTapOnBus_bufferSize_format_block(
                0,
                TAP_BUFFER_FRAMES,
                Some(&format),
                block,
            );
        }

        // SAFETY: the graph is a bare input tap, which needs no output connection on macOS.
        unsafe { engine.prepare() };
        if let Err(e) = unsafe { engine.startAndReturnError() } {
            // SAFETY: `input` is the node the tap was just installed on.
            unsafe { input.removeTapOnBus(0) };
            let _ = writer.finish();
            return Err(Error::AudioEngine(e.localizedDescription().to_string()));
        }

        Ok(MicCapture {
            engine,
            input,
            _tap: tap,
            progress: writer.progress(),
            liveness: MicLiveness::new(sample_rate),
            writer,
            sample_rate,
            channels,
        })
    }
}

/// A live microphone capture.
pub struct MicCapture {
    engine: Retained<AVAudioEngine>,
    input: Retained<AVAudioInputNode>,
    /// Held so the block -- and the [`crate::track::TrackSink`] it captured -- outlives the
    /// tap. The framework retains it too, but letting a local drop it would be a use-after
    /// free waiting for the next refactor.
    _tap: RcBlock<dyn Fn(NonNull<AVAudioPCMBuffer>, NonNull<AVAudioTime>)>,
    progress: TrackProgress,
    liveness: MicLiveness,
    writer: TrackWriter,
    sample_rate: u32,
    channels: u32,
}

impl MicCapture {
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Frames the tap has delivered into `mic.wav` so far. A diagnostic, not the file's
    /// length: see [`TrackProgress::frames`].
    pub fn frames_delivered(&self) -> u64 {
        self.progress.frames()
    }

    /// Whether the tap has stopped delivering audio into `mic.wav`.
    ///
    /// Costs one relaxed atomic load, so it is cheap enough to ask on a poll the caller is
    /// already making. `&mut self` because the standstill is measured across calls.
    pub fn stalled(&mut self, now: Instant) -> bool {
        self.liveness.observe(&self.progress, now)
    }

    /// Stops the engine and finalizes the WAV.
    pub fn stop(self) -> Result<TrackSummary> {
        let MicCapture {
            engine,
            input,
            _tap,
            writer,
            ..
        } = self;

        // Stop first, then remove the tap: the reverse order leaves a window in which a
        // callback can fire into a torn-down writer.
        // SAFETY: both are live objects owned by this struct.
        unsafe {
            engine.stop();
            input.removeTapOnBus(0);
        }

        writer.finish()
    }
}

/// The stall rule, decided with no microphone anywhere near it.
///
/// Every test here drives a real [`TrackWriter`] and its real [`crate::track::TrackSink`] --
/// no FFI, no device -- because the rule's input is a delivered frame count and a buffer
/// size, and those are exactly what a sink publishes. What these cannot decide is whether a
/// real `AVAudioEngine` whose device reconfigures actually stops calling its tap block; that
/// needs hardware.
#[cfg(test)]
mod tests {
    use super::*;

    /// A live track, plus a sink to push buffers through it.
    fn track() -> (tempfile::TempDir, TrackWriter) {
        let dir = tempfile::tempdir().unwrap();
        let writer = TrackWriter::create(&dir.path().join("mic.wav"), 48_000).unwrap();
        (dir, writer)
    }

    #[test]
    fn the_limit_is_eight_buffer_periods_with_a_floor() {
        // 4096 frames at 48 kHz is 85.3 ms, so eight periods is 683 ms -- under the floor.
        assert_eq!(stall_limit(4096, 48_000), MIN_STALL_LIMIT);
        // The same buffer at 8 kHz takes 512 ms to fill, so the same rule allows 4.096 s.
        assert_eq!(stall_limit(4096, 8_000), Duration::from_millis(4096));
        // A big buffer is given proportionally longer rather than being capped: 16384 frames
        // at 48 kHz is 341.3 ms, so eight periods is 2.731 s. Compared at millisecond
        // precision because the exact value is a repeating fraction.
        assert_eq!(stall_limit(16_384, 48_000).as_millis(), 2730);
        // Not reachable -- `InputDevice::open` refuses it -- but a liveness check must not
        // panic on arithmetic. A rate of 1 Hz is the guarded reading, and its limit is
        // absurdly long rather than absurdly short, which is the safe direction.
        assert_eq!(stall_limit(4096, 0), Duration::from_secs(4096 * 8));
    }

    /// A device that is still coming up has not stalled. Getting this wrong would finalize and
    /// restart the session every few seconds for the length of the meeting.
    #[test]
    fn a_tap_that_has_not_delivered_a_buffer_yet_has_not_stalled() {
        let (_dir, writer) = track();
        let progress = writer.progress();
        let mut liveness = MicLiveness::new(48_000);

        let t0 = Instant::now();
        assert!(!liveness.observe(&progress, t0));
        assert!(!liveness.observe(&progress, t0 + Duration::from_secs(3600)));
    }

    /// The false-positive guard, at the level the rule is decided: the count advances the same
    /// way for a silent room as for a loud one, because nothing here looks at a sample value.
    #[test]
    fn a_frame_count_that_keeps_advancing_is_never_a_stall() {
        let (_dir, writer) = track();
        let progress = writer.progress();
        let sink = writer.sink();
        let mut liveness = MicLiveness::new(48_000);

        let t0 = Instant::now();
        for buffer in 0..20u32 {
            // Silence, deliberately: a live tap delivers a quiet room as zeros.
            sink.push(u64::from(buffer), u64::from(buffer), vec![0.0; 4096]);
            // Well past the limit each time, so only the advance can be what keeps it quiet.
            let now = t0 + Duration::from_millis(1500) * (buffer + 1);
            assert!(
                !liveness.observe(&progress, now),
                "buffer {buffer} was read as a stall"
            );
        }
    }

    #[test]
    fn a_frame_count_that_stands_still_becomes_a_stall_once_the_limit_passes() {
        let (_dir, writer) = track();
        let progress = writer.progress();
        let mut liveness = MicLiveness::new(48_000);

        let t0 = Instant::now();
        writer.sink().push(1, 2, vec![0.5; 4096]);
        assert!(!liveness.observe(&progress, t0), "arming is not a stall");

        assert!(!liveness.observe(&progress, t0 + Duration::from_millis(999)));
        assert!(liveness.observe(&progress, t0 + Duration::from_millis(1001)));
    }

    /// A rule that reset its clock on every question would never fire, because the record loop
    /// asks far more often than the limit.
    #[test]
    fn a_standstill_is_measured_from_the_last_advance_and_not_from_the_last_question() {
        let (_dir, writer) = track();
        let progress = writer.progress();
        let mut liveness = MicLiveness::new(48_000);

        let t0 = Instant::now();
        writer.sink().push(1, 2, vec![0.5; 4096]);
        assert!(!liveness.observe(&progress, t0));

        assert!(!liveness.observe(&progress, t0 + Duration::from_millis(600)));
        assert!(!liveness.observe(&progress, t0 + Duration::from_millis(900)));
        // 200 ms since the previous question, 1100 ms since the last advance.
        assert!(liveness.observe(&progress, t0 + Duration::from_millis(1100)));
    }
}
