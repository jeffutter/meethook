//! One capture track: a lock-free hand-off from a real-time callback to a WAV writer.
//!
//! The invariant this module exists to hold is that **no disk I/O ever happens on a capture
//! callback**. A callback copies its samples into a `Vec<f32>`, hands it to a channel, and
//! returns; a dedicated thread owns the `hound::WavWriter`. Overrunning an audio callback
//! drops audio at the driver, which is unrecoverable, so the copy-and-send is the only work
//! the callback is allowed to do.
//!
//! The channel is unbounded on purpose. Mono 32-bit float at 48 kHz is 192 KB/s, so a stall
//! would have to last minutes before memory mattered, whereas a bounded channel would drop
//! audio the moment the disk hiccupped. Losing meeting audio to save a megabyte is the
//! wrong trade.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use hound::{SampleFormat, WavSpec, WavWriter};

use crate::{Error, Result};

/// How often the writer thread rewrites the RIFF/data sizes in place.
///
/// `hound`'s `flush` is a checkpoint, not merely a buffer flush: it writes correct sizes
/// and seeks back, leaving the file valid on disk. A hard-killed recorder therefore loses
/// at most this much audio instead of leaving an unplayable file.
const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(5);

enum Message {
    Samples(Vec<f32>),
    Stop,
}

/// A cheap, cloneable handle for a capture callback to push samples through.
#[derive(Clone)]
pub struct TrackSink {
    tx: Sender<Message>,
    first: Arc<OnceLock<u64>>,
}

impl TrackSink {
    /// Records `host_ticks` if this is the track's first buffer, then queues `mono`.
    ///
    /// Returns immediately and performs no I/O or allocation beyond taking ownership of a
    /// buffer the caller already allocated.
    ///
    /// A closed channel is ignored: the writer thread has already stopped or failed, and
    /// [`TrackWriter::finish`] is where that gets reported. There is nothing useful a
    /// real-time callback could do about it.
    pub fn push(&self, host_ticks: u64, mono: Vec<f32>) {
        let _ = self.first.set(host_ticks);
        let _ = self.tx.send(Message::Samples(mono));
    }
}

/// What a finished track turned out to contain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackSummary {
    pub sample_rate: u32,
    pub frames: u64,
    /// `None` means the track never received a single buffer.
    pub first_host_ticks: Option<u64>,
}

impl TrackSummary {
    pub fn seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.frames as f64 / f64::from(self.sample_rate)
    }
}

/// Owns a track's writer thread and the WAV file behind it.
pub struct TrackWriter {
    path: PathBuf,
    sample_rate: u32,
    sink: TrackSink,
    handle: JoinHandle<Result<u64>>,
}

impl TrackWriter {
    /// Opens `path` as a mono 32-bit-float WAV at `sample_rate` and starts its writer
    /// thread.
    ///
    /// Float32 rather than a smaller integer format because it is what both capture engines
    /// deliver natively; narrowing to i16 here would be exactly the lossy transform this
    /// recorder promises never to apply.
    pub fn create(path: &Path, sample_rate: u32) -> Result<TrackWriter> {
        TrackWriter::create_with_checkpoint(path, sample_rate, CHECKPOINT_INTERVAL)
    }

    fn create_with_checkpoint(
        path: &Path,
        sample_rate: u32,
        checkpoint: Duration,
    ) -> Result<TrackWriter> {
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(path, spec).map_err(|e| Error::wav(path, e))?;

        let (tx, rx) = mpsc::channel::<Message>();
        let thread_path = path.to_path_buf();

        let handle = thread::Builder::new()
            .name("meethook-wav".to_owned())
            .spawn(move || {
                let mut frames: u64 = 0;
                let mut last_checkpoint = Instant::now();

                // `Stop` rather than sender-disconnect, because a capture callback may still
                // be holding a `TrackSink` clone when `finish` runs; waiting for every clone
                // to drop would hang.
                while let Ok(Message::Samples(samples)) = rx.recv() {
                    for sample in &samples {
                        writer
                            .write_sample(*sample)
                            .map_err(|e| Error::wav(&thread_path, e))?;
                    }
                    // Mono means every sample is a complete frame, so a checkpoint can never
                    // land mid-frame.
                    frames += samples.len() as u64;

                    if last_checkpoint.elapsed() >= checkpoint {
                        writer.flush().map_err(|e| Error::wav(&thread_path, e))?;
                        last_checkpoint = Instant::now();
                    }
                }

                writer
                    .finalize()
                    .map_err(|e| Error::wav(&thread_path, e))
                    .map(|()| frames)
            })
            .map_err(|e| Error::io(path, e))?;

        Ok(TrackWriter {
            path: path.to_path_buf(),
            sample_rate,
            sink: TrackSink {
                tx,
                first: Arc::new(OnceLock::new()),
            },
            handle,
        })
    }

    /// A handle to hand to a capture callback. Clone freely; they all feed one file.
    pub fn sink(&self) -> TrackSink {
        self.sink.clone()
    }

    /// Drains the queue, finalizes the WAV header, and reports what was written.
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

        let first = Arc::clone(&sink.first);
        let _ = sink.tx.send(Message::Stop);
        drop(sink);

        let frames = match handle.join() {
            Ok(result) => result?,
            Err(_) => return Err(Error::WriterPanic { path }),
        };

        Ok(TrackSummary {
            sample_rate,
            frames,
            // Read after the join, so any callback still in flight when `stop` was called
            // has already had its chance to set it.
            first_host_ticks: first.get().copied(),
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

    fn temp_wav(name: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        (dir, path)
    }

    #[test]
    fn samples_round_trip_through_the_wav_file() {
        let (_dir, path) = temp_wav("round-trip.wav");
        let writer = TrackWriter::create(&path, 44_100).unwrap();
        let sink = writer.sink();

        let first: Vec<f32> = vec![0.0, 0.25, -0.5, 1.0];
        let second: Vec<f32> = vec![-1.0, 0.125];
        sink.push(1_000, first.clone());
        sink.push(2_000, second.clone());

        let summary = writer.finish().unwrap();
        assert_eq!(summary.frames, 6);
        assert_eq!(summary.sample_rate, 44_100);
        // The *first* buffer's ticks, not the last.
        assert_eq!(summary.first_host_ticks, Some(1_000));

        let mut reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 44_100);
        assert_eq!(spec.bits_per_sample, 32);
        assert_eq!(spec.sample_format, SampleFormat::Float);
        assert_eq!(reader.duration(), 6);

        let read: Vec<f32> = reader.samples::<f32>().map(|s| s.unwrap()).collect();
        let expected: Vec<f32> = first.into_iter().chain(second).collect();
        assert_eq!(read, expected);
    }

    /// A track that never received audio must still leave a valid, playable (empty) file --
    /// but it must report `None`, so the caller can refuse to write `session.json` rather
    /// than fabricating a timestamp for a broken recording.
    #[test]
    fn a_silent_track_finalizes_to_a_valid_empty_wav() {
        let (_dir, path) = temp_wav("silent.wav");
        let writer = TrackWriter::create(&path, 48_000).unwrap();

        let summary = writer.finish().unwrap();
        assert_eq!(summary.frames, 0);
        assert_eq!(summary.first_host_ticks, None);

        let reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.duration(), 0);
        assert_eq!(reader.spec().sample_rate, 48_000);
    }

    /// The crash-safety property: a recorder killed mid-session leaves a file that plays up
    /// to the last checkpoint, rather than one with a placeholder header.
    #[test]
    fn a_checkpointed_file_is_playable_without_finish() {
        let (_dir, path) = temp_wav("checkpointed.wav");
        // Zero interval: every message checkpoints, which is the same code path a 5-second
        // interval takes, just without a 5-second test.
        let writer =
            TrackWriter::create_with_checkpoint(&path, 16_000, Duration::from_secs(0)).unwrap();
        writer.sink().push(7, vec![0.5; 128]);

        // Poll rather than sleep-and-hope: the writer thread is asynchronous by design.
        let deadline = Instant::now() + Duration::from_secs(5);
        let frames = loop {
            if let Ok(reader) = hound::WavReader::open(&path)
                && reader.duration() == 128
            {
                break reader.duration();
            }
            assert!(Instant::now() < deadline, "checkpoint never landed on disk");
            thread::sleep(Duration::from_millis(20));
        };
        assert_eq!(frames, 128);

        // Leak deliberately: dropping would send nothing, but joining is exactly what a
        // killed process does not get to do, and that is the case under test.
        std::mem::forget(writer);
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
            first_host_ticks: Some(1),
        };
        assert!((summary.seconds() - 0.5).abs() < f64::EPSILON);
    }
}
