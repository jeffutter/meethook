//! One capture track: a lock-free hand-off from a real-time callback to a FLAC writer.
//!
//! The invariant this module exists to hold is that **no disk I/O ever happens on a capture
//! callback**. A callback copies its samples into a `Vec<f32>`, hands it to a channel, and
//! returns; a dedicated thread owns the FLAC writer. Overrunning an audio callback drops audio
//! at the driver, which is unrecoverable, so the copy-and-send is the only work the callback is
//! allowed to do.
//!
//! What lands on disk is **not** what the callback delivered. The writer thread resamples each
//! track from its native capture rate to 16 kHz, quantises to 16-bit integers and streams the
//! result into a FLAC file as it arrives (`mic.flac`, `speaker.flac`), holding about one
//! block of audio per track. The original 48 kHz float stream is therefore not kept: the
//! recorder used to promise a lossless native-rate WAV, and that promise is dropped on purpose,
//! because everything downstream (the echo canceller, the diarizer, Whisper) resamples to
//! 16 kHz and 16 bits first, and a recording hour was ~690 MB per track for audio nothing
//! listened to at that quality. Timing is the exception: [`TimingAccumulator`] and
//! [`TrackSummary`] keep counting *native* frames, so `session.json`'s `mic_rate` and speaker
//! rate still describe the capture, and the host-tick alignment is unchanged -- the resampler
//! is delay-compensated, so the first stored sample is the first captured one.
//!
//! The channel is unbounded on purpose. Mono 32-bit float at 48 kHz is 192 KB/s, so a stall
//! would have to last minutes before memory mattered, whereas a bounded channel would drop
//! audio the moment the disk hiccupped. Losing meeting audio to save a megabyte is the
//! wrong trade.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};

use meethook_session::{FlacTrackWriter, Resampler, TRACK_RATE};

use crate::{Error, Result};

enum Message {
    Samples {
        host_ticks: u64,
        delivered_ticks: u64,
        samples: Vec<f32>,
    },
    Stop,
}

/// Per-buffer timing for a whole track, accumulated on the writer thread.
///
/// Exists to answer one question the stored first-buffer timestamp cannot answer about
/// itself: *is it typical?* Delivery settles into a steady rhythm once a stream is running,
/// so the gap between an API's timestamp and the buffer's arrival converges to a constant.
/// A first-buffer gap that sits on that constant means the stored timestamp describes its
/// samples the same way every later timestamp does. A first-buffer gap that is an outlier
/// means the one value written to `session.json` is the one value the API got wrong.
///
/// Gated behind `MEETHOOK_TIMING_DEBUG`, checked once when the track is created rather than
/// per buffer, so an unset variable costs a single `bool` test on the writer thread.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TimingProfile {
    first_gap_ticks: i64,
    /// Ascending, so percentiles are a plain index.
    sorted_gaps: Vec<i64>,
    /// The furthest any buffer's timestamp strayed from the straight line drawn through the
    /// first timestamp at the track's nominal sample rate. Large values mean the timestamps
    /// and the sample stream disagree about how much time has passed -- which would corrupt
    /// alignment no matter how accurate the first timestamp was.
    max_drift_ticks: i64,
}

impl TimingProfile {
    pub fn buffers(&self) -> usize {
        self.sorted_gaps.len()
    }

    /// The gap belonging to the buffer whose timestamp is stored in `session.json`.
    pub fn first_gap_ticks(&self) -> i64 {
        self.first_gap_ticks
    }

    /// The gap a settled stream converges to. Median rather than mean because a single
    /// scheduling stall would drag a mean far more than it distorts the typical case.
    pub fn median_gap_ticks(&self) -> i64 {
        self.sorted_gaps
            .get(self.sorted_gaps.len() / 2)
            .copied()
            .unwrap_or(0)
    }

    pub fn min_gap_ticks(&self) -> i64 {
        self.sorted_gaps.first().copied().unwrap_or(0)
    }

    pub fn max_gap_ticks(&self) -> i64 {
        self.sorted_gaps.last().copied().unwrap_or(0)
    }

    pub fn max_drift_ticks(&self) -> i64 {
        self.max_drift_ticks
    }
}

/// Accumulates [`TimingProfile`] as buffers arrive.
struct TimingAccumulator {
    ticks_per_frame: f64,
    first_host_ticks: Option<u64>,
    frames_before: u64,
    gaps: Vec<i64>,
    first_gap_ticks: i64,
    max_drift_ticks: i64,
}

impl TimingAccumulator {
    fn new(sample_rate: u32) -> TimingAccumulator {
        let (numer, denom) = crate::clock::timebase();
        // ticks/second = 1e9 * denom / numer, so ticks/frame divides that by the rate.
        let ticks_per_second = 1e9 * f64::from(denom) / f64::from(numer);
        TimingAccumulator {
            ticks_per_frame: ticks_per_second / f64::from(sample_rate.max(1)),
            first_host_ticks: None,
            frames_before: 0,
            gaps: Vec::new(),
            first_gap_ticks: 0,
            max_drift_ticks: 0,
        }
    }

    fn observe(&mut self, host_ticks: u64, delivered_ticks: u64, frames: u64) {
        // Signed throughout: a timestamp later than its own delivery is a distinct and much
        // stranger fault than one earlier, and must not wrap silently through `u64`.
        let gap = delivered_ticks as i64 - host_ticks as i64;
        let first = *self.first_host_ticks.get_or_insert_with(|| {
            self.first_gap_ticks = gap;
            host_ticks
        });
        self.gaps.push(gap);

        let expected = first as f64 + self.frames_before as f64 * self.ticks_per_frame;
        let drift = (host_ticks as f64 - expected) as i64;
        if drift.abs() > self.max_drift_ticks.abs() {
            self.max_drift_ticks = drift;
        }
        self.frames_before += frames;
    }

    fn finish(mut self) -> TimingProfile {
        self.gaps.sort_unstable();
        TimingProfile {
            first_gap_ticks: self.first_gap_ticks,
            sorted_gaps: self.gaps,
            max_drift_ticks: self.max_drift_ticks,
        }
    }
}

/// Everything known about the buffer whose timestamp becomes the track's stored
/// `host_ticks`.
///
/// `delivered_ticks` exists to make the capture API's timestamp *falsifiable*. A stored
/// timestamp on its own cannot be checked against anything; paired with the moment the
/// buffer actually reached us, the gap between them reveals what the API means by "when":
/// a gap of roughly one buffer duration means the timestamp refers to the buffer's first
/// sample, a gap near zero means it refers to delivery, and a gap materially larger than a
/// buffer means there is latency the timestamp is not accounting for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FirstBuffer {
    pub host_ticks: u64,
    pub delivered_ticks: u64,
    pub frames: u32,
}

/// What the callback side of a track has delivered, readable while the track is live.
///
/// Both fields are written from a capture callback and read from another thread, which is
/// the whole reason this is one shared allocation rather than two: they are published
/// together, on every buffer, from the one place every buffer passes through.
struct Delivery {
    first: OnceLock<FirstBuffer>,
    /// Frames handed to the writer, not frames on disk. See [`TrackProgress::frames`].
    frames: AtomicU64,
}

/// A reader's view of what a live track's callback side has delivered so far.
///
/// Exists so that "is this track still receiving audio?" can be answered *while* the track
/// runs, rather than only from [`TrackSummary`] once it has been finished. A handle rather
/// than accessors on [`TrackWriter`] because the type is what carries the documentation of
/// what the number means, and because the reader is usually a different object from the one
/// that owns the writer.
#[derive(Clone)]
pub struct TrackProgress(Arc<Delivery>);

impl TrackProgress {
    /// Frames **delivered to the writer**, not frames written to the file.
    ///
    /// The two differ transiently by whatever is queued, and permanently if the writer
    /// thread has failed. This is therefore a liveness signal for the capture callback and
    /// nothing else: [`TrackSummary::frames`] remains the only number that describes the
    /// file on disk.
    ///
    /// Monotonic, and never resets. Zero means no buffer has been delivered yet.
    pub fn frames(&self) -> u64 {
        self.0.frames.load(Ordering::Relaxed)
    }

    /// The first delivered buffer, once one has arrived.
    ///
    /// `Some` implies [`TrackProgress::frames`] is non-zero, since both are published from
    /// the same callback before it returns.
    pub fn first_buffer(&self) -> Option<FirstBuffer> {
        self.0.first.get().copied()
    }
}

/// A cheap, cloneable handle for a capture callback to push samples through.
#[derive(Clone)]
pub struct TrackSink {
    tx: Sender<Message>,
    delivery: Arc<Delivery>,
}

impl TrackSink {
    /// Records the first buffer's timing if this is the track's first buffer, counts the
    /// frames delivered, then queues `mono`.
    ///
    /// `delivered_ticks` should be read at the very top of the capture callback, before any
    /// sample copying, so it measures the callback's arrival rather than its duration.
    ///
    /// Returns immediately and performs no I/O or allocation beyond taking ownership of a
    /// buffer the caller already allocated.
    ///
    /// A closed channel is ignored: the writer thread has already stopped or failed, and
    /// [`TrackWriter::finish`] is where that gets reported. There is nothing useful a
    /// real-time callback could do about it.
    pub fn push(&self, host_ticks: u64, delivered_ticks: u64, mono: Vec<f32>) {
        // Mono, so one sample is one frame.
        let _ = self.delivery.first.set(FirstBuffer {
            host_ticks,
            delivered_ticks,
            frames: mono.len() as u32,
        });
        // `Relaxed` on purpose: nothing else is published through this counter, and its
        // reader wants only a monotonic number it can compare against the last one it saw.
        // An acquire/release pair on an audio callback would be cost for an ordering
        // guarantee no reader uses.
        self.delivery
            .frames
            .fetch_add(mono.len() as u64, Ordering::Relaxed);
        let _ = self.tx.send(Message::Samples {
            host_ticks,
            delivered_ticks,
            samples: mono,
        });
    }
}

/// What a finished track turned out to contain.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrackSummary {
    pub sample_rate: u32,
    pub frames: u64,
    /// `None` means the track never received a single buffer.
    pub first_buffer: Option<FirstBuffer>,
    /// `None` unless `MEETHOOK_TIMING_DEBUG` was set for this run.
    pub timing: Option<TimingProfile>,
}

impl TrackSummary {
    /// The timestamp that becomes this track's `host_ticks` in `session.json`.
    pub fn first_host_ticks(&self) -> Option<u64> {
        self.first_buffer.map(|b| b.host_ticks)
    }

    pub fn seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.frames as f64 / f64::from(self.sample_rate)
    }
}

/// Owns a track's writer thread and the FLAC file behind it.
pub struct TrackWriter {
    path: PathBuf,
    sample_rate: u32,
    sink: TrackSink,
    handle: JoinHandle<Result<(u64, Option<TimingProfile>)>>,
}

impl TrackWriter {
    /// Opens `path` as a 16 kHz mono 16-bit FLAC and starts its writer thread.
    ///
    /// `sample_rate` is the rate the capture callback delivers at, not the rate stored: the
    /// writer thread resamples to 16 kHz as audio arrives (nothing is done when the capture
    /// is already 16 kHz). It stays the rate [`TrackSummary`] and the timing accumulator
    /// count in, so a track's `frames` and `seconds()` describe the capture.
    ///
    /// This is a lossy transform and deliberately so; see the module note. Crash recovery is
    /// the FLAC's own: every frame is self-contained and STREAMINFO is a placeholder until
    /// [`TrackWriter::finish`] patches it, so a recorder killed mid-session leaves a file that
    /// decodes up to its last whole frame -- about a second short of what was captured.
    pub fn create(path: &Path, sample_rate: u32) -> Result<TrackWriter> {
        let file = std::fs::File::create(path).map_err(|e| Error::io(path, e))?;
        let mut flac = FlacTrackWriter::new(file).map_err(|e| Error::io(path, e))?;
        let mut resampler = if sample_rate == TRACK_RATE {
            None
        } else {
            Some(Resampler::new(sample_rate, TRACK_RATE).map_err(|e| Error::resample(path, e))?)
        };

        let (tx, rx) = mpsc::channel::<Message>();
        let thread_path = path.to_path_buf();
        // Read once here rather than per buffer: the diagnostic must not put an environment
        // lookup anywhere near the write loop.
        let timing = std::env::var_os("MEETHOOK_TIMING_DEBUG").map(|_| ());

        let handle = thread::Builder::new()
            .name("meethook-track".to_owned())
            .spawn(move || {
                let mut frames: u64 = 0;
                let mut timing = timing.map(|()| TimingAccumulator::new(sample_rate));
                // Reused across buffers, so the resampled copy of a buffer is the only
                // allocation-sized thing this loop holds besides the FLAC's one block.
                let mut resampled: Vec<f32> = Vec::new();

                // `Stop` rather than sender-disconnect, because a capture callback may still
                // be holding a `TrackSink` clone when `finish` runs; waiting for every clone
                // to drop would hang.
                while let Ok(Message::Samples {
                    host_ticks,
                    delivered_ticks,
                    samples,
                }) = rx.recv()
                {
                    match resampler.as_mut() {
                        Some(resampler) => {
                            resampled.clear();
                            resampler
                                .push(&samples, &mut resampled)
                                .map_err(|e| Error::resample(&thread_path, e))?;
                            flac.push(&resampled)
                        }
                        None => flac.push(&samples),
                    }
                    .map_err(|e| Error::io(&thread_path, e))?;

                    // Native frames, not stored ones: mono means every sample is a frame, and
                    // the timing arithmetic is in capture-rate ticks.
                    frames += samples.len() as u64;
                    if let Some(timing) = timing.as_mut() {
                        timing.observe(host_ticks, delivered_ticks, samples.len() as u64);
                    }
                }

                // The resampler's tail is still inside its filter.
                if let Some(resampler) = resampler {
                    resampled.clear();
                    resampler
                        .finish(&mut resampled)
                        .map_err(|e| Error::resample(&thread_path, e))?;
                    flac.push(&resampled)
                        .map_err(|e| Error::io(&thread_path, e))?;
                }
                flac.finish()
                    .map_err(|e| Error::io(&thread_path, e))
                    .map(|_| (frames, timing.map(TimingAccumulator::finish)))
            })
            .map_err(|e| Error::io(path, e))?;

        Ok(TrackWriter {
            path: path.to_path_buf(),
            sample_rate,
            sink: TrackSink {
                tx,
                delivery: Arc::new(Delivery {
                    first: OnceLock::new(),
                    frames: AtomicU64::new(0),
                }),
            },
            handle,
        })
    }

    /// A handle to hand to a capture callback. Clone freely; they all feed one file.
    pub fn sink(&self) -> TrackSink {
        self.sink.clone()
    }

    /// A handle for reading what the callback side has delivered while the track is live.
    pub fn progress(&self) -> TrackProgress {
        TrackProgress(Arc::clone(&self.sink.delivery))
    }

    /// Drains the queue, flushes the resampler, finalizes the FLAC header, and reports what was
    /// written.
    ///
    /// Joining the writer thread is what makes a write error surface to the caller instead
    /// of vanishing into a detached thread.
    pub fn finish(self) -> Result<TrackSummary> {
        let TrackWriter {
            path,
            sample_rate,
            sink,
            handle,
        } = self;

        let delivery = Arc::clone(&sink.delivery);
        let _ = sink.tx.send(Message::Stop);
        drop(sink);

        let (frames, timing) = match handle.join() {
            Ok(result) => result?,
            Err(_) => return Err(Error::WriterPanic { path }),
        };

        Ok(TrackSummary {
            sample_rate,
            frames,
            // Read after the join, so any callback still in flight when `stop` was called
            // has already had its chance to set it.
            first_buffer: delivery.first.get().copied(),
            timing,
        })
    }
}

/// Copies channel 0 out of a float32 buffer.
///
/// `stride` is the distance in samples between consecutive frames of one channel: 1 for the
/// non-interleaved layouts both capture engines use, and the channel count for an
/// interleaved one.
///
/// Channel 0 is taken rather than a downmix of all channels. On macOS a multi-channel
/// default input is far more often a multi-input audio interface (where channel 1 is an
/// unplugged jack emitting noise) or a duplicated mono signal than a true stereo capture of
/// one voice. Averaging corrupts the first case and gains nothing in the second; channel 0
/// is right in both, and in the rare true-stereo case it merely narrows the field, which
/// does not affect intelligibility.
///
/// # Safety
///
/// `channels` must point to at least one valid channel pointer, and that channel pointer
/// must be valid for `frames * stride` readable `f32`s.
pub unsafe fn channel_zero(channels: *const *const f32, frames: usize, stride: usize) -> Vec<f32> {
    debug_assert!(!channels.is_null());
    debug_assert!(stride >= 1);

    // SAFETY: the caller guarantees `channels` points to at least one channel pointer.
    let channel = unsafe { *channels };
    if channel.is_null() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(frames);
    if stride == 1 {
        // SAFETY: the caller guarantees `frames` readable samples at `channel`.
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(channel, frames) });
    } else {
        for frame in 0..frames {
            // SAFETY: `frame * stride < frames * stride`, which the caller guarantees is
            // readable.
            out.push(unsafe { *channel.add(frame * stride) });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_track(name: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        (dir, path)
    }

    /// The i16 samples a FLAC holds, stopping quietly at a cut frame the way every reader of a
    /// killed recorder's file has to.
    fn decode(path: &Path) -> Vec<i32> {
        let bytes = std::fs::read(path).unwrap();
        let mut decoder = rusty_flac::Decoder::new(&bytes).unwrap();
        let mut samples = Vec::new();
        let mut block = Vec::new();
        loop {
            block.clear();
            match decoder.next_frame_interleaved::<i32>(&mut block) {
                Ok(Some(_)) => samples.extend_from_slice(&block),
                Ok(None) | Err(rusty_flac::DecodeError::Truncated) => break,
                Err(e) => panic!("decode failed: {e}"),
            }
        }
        samples
    }

    /// Samples already on the 16-bit grid, so a 16 kHz track must come back exactly.
    fn on_grid(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 31) % 4001) as f32 - 2000.0) / 32_768.0)
            .collect()
    }

    #[test]
    fn samples_round_trip_through_the_flac_file() {
        let (_dir, path) = temp_track("round-trip.flac");
        let writer = TrackWriter::create(&path, 16_000).unwrap();
        let sink = writer.sink();

        let first = on_grid(5_000);
        let second = on_grid(3_000);
        sink.push(1_000, 1_500, first.clone());
        sink.push(2_000, 2_500, second.clone());

        let summary = writer.finish().unwrap();
        assert_eq!(summary.frames, 8_000);
        assert_eq!(summary.sample_rate, 16_000);
        // The *first* buffer's ticks, not the last.
        assert_eq!(summary.first_host_ticks(), Some(1_000));

        let bytes = std::fs::read(&path).unwrap();
        let decoder = rusty_flac::Decoder::new(&bytes).unwrap();
        let info = decoder.streaminfo();
        assert_eq!(
            (info.sample_rate, info.channels, info.bits_per_sample),
            (16_000, 1, 16)
        );
        assert_eq!(info.total_samples, 8_000, "a clean stop patches the total");

        let expected: Vec<i32> = first
            .into_iter()
            .chain(second)
            .map(meethook_session::quantize_sample)
            .collect();
        assert_eq!(decode(&path), expected);
    }

    /// A track that never received audio must still leave a valid (empty) file -- but it must
    /// report `None`, so the caller can refuse to write `session.json` rather than fabricating
    /// a timestamp for a broken recording.
    #[test]
    fn a_silent_track_finalizes_to_a_valid_empty_flac() {
        let (_dir, path) = temp_track("silent.flac");
        let writer = TrackWriter::create(&path, 48_000).unwrap();

        let summary = writer.finish().unwrap();
        assert_eq!(summary.frames, 0);
        assert_eq!(summary.first_host_ticks(), None);

        assert!(decode(&path).is_empty());
        let bytes = std::fs::read(&path).unwrap();
        let decoder = rusty_flac::Decoder::new(&bytes).unwrap();
        assert_eq!(decoder.streaminfo().sample_rate, 16_000);
    }

    /// The crash-safety property: a recorder killed mid-session leaves a file that decodes up
    /// to roughly the last second, with no finish ever having run.
    #[test]
    fn a_file_that_was_never_finished_decodes_up_to_the_last_flush() {
        let (_dir, path) = temp_track("killed.flac");
        let writer = TrackWriter::create(&path, 16_000).unwrap();
        writer.sink().push(7, 9, on_grid(40_000));

        // Poll rather than sleep-and-hope: the writer thread is asynchronous by design.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let decoded = loop {
            let bytes = std::fs::read(&path).unwrap();
            if rusty_flac::Decoder::new(&bytes).is_ok() {
                let decoded = decode(&path);
                // One second of audio is 16 000 samples: at least that much was pushed to
                // the OS by the writer's own flush.
                if decoded.len() >= 16_000 {
                    break decoded;
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "nothing was flushed to disk"
            );
            thread::sleep(std::time::Duration::from_millis(20));
        };
        let expected: Vec<i32> = on_grid(decoded.len())
            .into_iter()
            .map(meethook_session::quantize_sample)
            .collect();
        assert_eq!(decoded, expected);

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(
            meethook_session::declared_samples(&bytes).map(|(_, total)| total),
            Some(0),
            "an unfinished file declares no length"
        );

        // Leak deliberately: dropping would send nothing, but joining is exactly what a
        // killed process does not get to do, and that is the case under test.
        std::mem::forget(writer);
    }

    /// Whatever the capture rate, the stored track is the same duration, and the summary still
    /// counts native frames -- the figure `session.json`'s rates and the timing arithmetic use.
    #[test]
    fn a_native_rate_track_is_stored_at_16_khz_and_counted_natively() {
        for rate in [48_000u32, 44_100] {
            let (_dir, path) = temp_track("native.flac");
            let writer = TrackWriter::create(&path, rate).unwrap();
            let sink = writer.sink();
            let native_frames = rate as usize * 3 + 123;
            for (i, piece) in vec![0.25f32; native_frames].chunks(480).enumerate() {
                sink.push(i as u64, i as u64, piece.to_vec());
            }
            let summary = writer.finish().unwrap();
            assert_eq!(summary.frames, native_frames as u64);
            assert_eq!(summary.sample_rate, rate);

            let expected = native_frames * 16_000 / rate as usize;
            let stored = decode(&path).len();
            // The only slack is the sub-32-sample tail flacenc cannot hold, padded with silence.
            assert!(
                (expected..expected + 32).contains(&stored),
                "{rate} Hz: stored {stored}, expected {expected}"
            );
        }
    }

    /// The resampler is delay-compensated, so a step at a known capture frame is at the same
    /// instant in the stored track: the host-tick of the first buffer still names sample zero.
    #[test]
    fn the_stored_track_stays_aligned_with_the_capture_timeline() {
        let (_dir, path) = temp_track("aligned.flac");
        let writer = TrackWriter::create(&path, 48_000).unwrap();
        let sink = writer.sink();
        let mut capture = vec![0.0f32; 48_000];
        capture[24_000..].fill(0.5);
        for (i, piece) in capture.chunks(512).enumerate() {
            sink.push(i as u64, i as u64, piece.to_vec());
        }
        writer.finish().unwrap();

        let stored = decode(&path);
        let onset = stored.iter().position(|s| *s > 8_192).unwrap();
        assert!(
            (onset as i64 - 8_000).abs() <= 8,
            "step at capture frame 24000 (0.5 s) landed at stored sample {onset}, expected 8000"
        );
    }

    /// The counter a live liveness check reads has to move as the callback pushes, and has to
    /// agree with the file once the writer has drained.
    #[test]
    fn frames_delivered_advance_as_the_callback_pushes() {
        let (_dir, path) = temp_track("delivered.flac");
        let writer = TrackWriter::create(&path, 48_000).unwrap();
        let progress = writer.progress();
        let sink = writer.sink();

        assert_eq!(progress.frames(), 0);
        assert_eq!(progress.first_buffer(), None);

        sink.push(1_000, 1_500, vec![0.25; 4096]);
        assert_eq!(progress.frames(), 4096);
        assert_eq!(progress.first_buffer().map(|b| b.frames), Some(4096));

        sink.push(2_000, 2_500, vec![0.5; 512]);
        assert_eq!(progress.frames(), 4608);
        // Still the *first* buffer, not the latest.
        assert_eq!(progress.first_buffer().map(|b| b.host_ticks), Some(1_000));

        let summary = writer.finish().unwrap();
        assert_eq!(
            summary.frames, 4608,
            "delivered frames disagree with the file"
        );
    }

    /// A silent room is delivered as zeros, not as an absence of buffers. This is the property
    /// the stall rule rests on: counting frames cannot mistake quiet for a dead engine,
    /// because nothing in the count looks at a sample value.
    #[test]
    fn a_track_delivering_silence_advances_the_delivered_count() {
        let (_dir, path) = temp_track("silence-counts.flac");
        let writer = TrackWriter::create(&path, 48_000).unwrap();
        let progress = writer.progress();

        writer.sink().push(1, 2, vec![0.0; 4096]);
        assert_eq!(progress.frames(), 4096);

        let summary = writer.finish().unwrap();
        assert_eq!(summary.frames, 4096);
    }

    #[test]
    fn channel_zero_takes_the_first_channel_of_a_non_interleaved_pair() {
        let left = [1.0f32, 2.0, 3.0];
        let right = [-1.0f32, -2.0, -3.0];
        let channels = [left.as_ptr(), right.as_ptr()];

        // SAFETY: both channel pointers are valid for 3 samples each.
        let mono = unsafe { channel_zero(channels.as_ptr(), 3, 1) };
        assert_eq!(mono, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn channel_zero_strides_over_an_interleaved_pair() {
        let interleaved = [1.0f32, -1.0, 2.0, -2.0, 3.0, -3.0];
        let channels = [interleaved.as_ptr()];

        // SAFETY: the single channel pointer is valid for 3 frames of stride 2.
        let mono = unsafe { channel_zero(channels.as_ptr(), 3, 2) };
        assert_eq!(mono, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn seconds_are_derived_from_frames_and_rate() {
        let summary = TrackSummary {
            sample_rate: 48_000,
            frames: 24_000,
            timing: None,
            first_buffer: Some(FirstBuffer {
                host_ticks: 1,
                delivered_ticks: 2,
                frames: 24_000,
            }),
        };
        assert!((summary.seconds() - 0.5).abs() < f64::EPSILON);
    }
}
