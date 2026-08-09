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

use std::path::Path;
use std::ptr::NonNull;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2_avf_audio::{
    AVAudioEngine, AVAudioFormat, AVAudioInputNode, AVAudioNodeTapBlock, AVAudioPCMBuffer,
    AVAudioTime,
};

use crate::track::{TrackSummary, TrackWriter};
use crate::{Error, Result};

/// Tap buffer size in frames. Large enough that the callback rate stays low (roughly 12 Hz
/// at 48 kHz), small enough that a stop never has to wait long for the last buffer.
const TAP_BUFFER_FRAMES: u32 = 4096;

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

        // SAFETY: a plain property read on the live input node. Read after `start`, because
        // before the engine runs the node reports 0 rather than the hardware's figure.
        let presentation_latency = unsafe { input.presentationLatency() };

        Ok(MicCapture {
            engine,
            input,
            _tap: tap,
            writer,
            sample_rate,
            channels,
            presentation_latency,
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
    writer: TrackWriter,
    sample_rate: u32,
    channels: u32,
    /// Seconds of hardware latency AVAudioEngine reports for this input, captured at start.
    /// Diagnostic only; see [`crate::latency`].
    presentation_latency: f64,
}

impl MicCapture {
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// What AVAudioEngine says this input's hardware latency is, in seconds.
    pub fn presentation_latency(&self) -> f64 {
        self.presentation_latency
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
