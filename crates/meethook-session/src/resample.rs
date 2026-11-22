//! Fixed-ratio sample-rate conversion for the tracks meethook writes and reads.
//!
//! One streaming implementation shared by the recorder (native capture rate down to the 16 kHz
//! a raw track is stored at) and by `transcribe` (any rate down to what Whisper accepts), so the
//! anti-aliasing filter, its delay compensation and the exact output length are decided once.
//! Push-shaped because both callers have audio arriving in pieces and neither can afford to
//! hold an hour of 48 kHz audio *and* its 16 kHz result at once (~900 MB).
//!
//! Rubato 1.0 and later route every `process` call through the `audioadapter` buffer traits
//! instead of taking raw slices. `SequentialSliceOfVecs` is their name for the old
//! `&[&[f32]]` shape -- one vector per channel -- so the chunk loop below keeps its old form
//! and only pays for the wrapper struct, not for copying samples into an intermediate buffer.

use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Fft, FixedSync, Resampler as _, WindowFunction};

/// Input frames handed to the resampler at a time.
///
/// Together with `SUB_CHUNKS` this sets the internal FFT size, and therefore both the
/// anti-aliasing filter's length and its group delay. These values give a filter of a few
/// hundred taps and a delay of a few milliseconds -- ample rejection for a 3:1 decimation,
/// and a delay far below the 10 ms granularity Whisper reports timestamps at.
const CHUNK_FRAMES: usize = 4096;
const SUB_CHUNKS: usize = 16;

/// Why a conversion could not be set up or run. Carries rubato's own wording, or a diagnosis
/// of the rates, as text: nothing branches on which it was.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ResampleError(pub String);

/// A mono sample-rate conversion, fed in pieces and drained into a caller's buffer.
///
/// Delay-compensated: the filter is linear-phase, so its output lags its input by a known,
/// constant number of frames, and exactly that many leading output frames are dropped. The
/// first output sample therefore lines up with the first input sample, and after
/// [`Resampler::finish`] the total output is exactly `pushed * target / source` samples (rounded
/// down), so a 16 kHz -> 48 kHz -> 16 kHz round trip returns the input's length and timeline.
pub struct Resampler {
    resampler: Fft<f32>,
    /// Input frames still waiting for a full [`CHUNK_FRAMES`] to hand the resampler.
    chunk: Vec<f32>,
    /// Leading output frames belonging to the filter's own group delay, not yet dropped.
    delay_left: usize,
    source_rate: u32,
    target_rate: u32,
    pushed: u64,
    /// Output frames handed to the caller so far, delay already removed.
    emitted: usize,
}

impl Resampler {
    pub fn new(source_rate: u32, target_rate: u32) -> Result<Self, ResampleError> {
        // Checked here rather than left to rubato, both because a rate of zero would divide
        // by zero below and because "a header claiming 0 Hz" is a diagnosis, where a
        // resampler-internal message would not be.
        if source_rate == 0 || target_rate == 0 {
            return Err(ResampleError(
                "the file's header reports a sample rate of 0".to_string(),
            ));
        }

        // BlackmanHarris2 is what the pre-1.0 `FftFixedIn` used internally, so the
        // anti-aliasing filter is the same one the timestamps were measured against.
        let resampler = Fft::<f32>::new_custom(
            source_rate as usize,
            target_rate as usize,
            CHUNK_FRAMES,
            SUB_CHUNKS,
            1,
            WindowFunction::BlackmanHarris2,
            FixedSync::Input,
        )
        .map_err(|e| ResampleError(e.to_string()))?;

        let delay_left = resampler.output_delay();
        Ok(Resampler {
            resampler,
            chunk: Vec::with_capacity(CHUNK_FRAMES),
            delay_left,
            source_rate,
            target_rate,
            pushed: 0,
            emitted: 0,
        })
    }

    /// How many output frames the whole conversion yields for `frames` input frames.
    fn expected_for(&self, frames: u64) -> usize {
        (u128::from(frames) * u128::from(self.target_rate) / u128::from(self.source_rate)) as usize
    }

    /// Appends the output for `samples` to `out`. Output arrives in whole filter chunks, so
    /// a call may append nothing, and the tail waits for [`Resampler::finish`].
    pub fn push(&mut self, samples: &[f32], out: &mut Vec<f32>) -> Result<(), ResampleError> {
        self.pushed += samples.len() as u64;
        let mut rest = samples;
        while !rest.is_empty() {
            let take = (CHUNK_FRAMES - self.chunk.len()).min(rest.len());
            self.chunk.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.chunk.len() == CHUNK_FRAMES {
                // Swap out the full chunk for a fresh pre-sized buffer: `process_chunk` needs
                // `&mut self`, so the chunk cannot be borrowed across the call.
                let full = std::mem::replace(&mut self.chunk, Vec::with_capacity(CHUNK_FRAMES));
                self.process_chunk(&full, None, usize::MAX, out)?;
            }
        }
        Ok(())
    }

    /// Flushes the filter and appends the remaining output to `out`.
    pub fn finish(mut self, out: &mut Vec<f32>) -> Result<(), ResampleError> {
        let expected = self.expected_for(self.pushed);
        if !self.chunk.is_empty() {
            let partial = std::mem::take(&mut self.chunk);
            self.process_chunk(&partial, Some(partial.len()), expected, out)?;
        }

        // Silence pumped through the filter pushes the samples still sitting in it out, so
        // the last fraction of a second of speech is not lost to the resampler's own
        // latency. Pump until the expected length is reached rather than flushing a fixed
        // number of times: the clamp in `process_chunk` discards the excess either way.
        let zeros = vec![0.0f32; self.resampler.input_frames_next()];
        while self.emitted < expected {
            self.process_chunk(&zeros, None, expected, out)?;
        }
        Ok(())
    }

    /// Hands `chunk` to the resampler. `partial_len` says how many of its frames are real
    /// when the last chunk is shorter than [`CHUNK_FRAMES`]; the resampler pads the missing
    /// tail with silence itself, which is what `process_partial` did in the pre-1.0 API.
    ///
    /// Output past `limit` total frames is dropped, which is how the filter's trailing ring
    /// is trimmed to the exact expected length.
    fn process_chunk(
        &mut self,
        chunk: &Vec<f32>,
        partial_len: Option<usize>,
        limit: usize,
        out: &mut Vec<f32>,
    ) -> Result<(), ResampleError> {
        let frames = partial_len.unwrap_or(chunk.len());
        // One vector per channel; mono means the single vector is the whole track.
        let buffer = SequentialSliceOfVecs::new(std::slice::from_ref(chunk), 1, frames)
            .map_err(|e| ResampleError(format!("{e:?}")))?;
        let indexing = partial_len.map(|len| rubato::Indexing {
            input_offset: 0,
            output_offset: 0,
            partial_len: Some(len),
            active_channels_mask: None,
        });
        let produced = self
            .resampler
            .process(&buffer, indexing.as_ref())
            .map_err(|e| ResampleError(e.to_string()))?;
        // Mono: interleaved order is the sample order, so the buffer is the output itself.
        let data = produced.take_data();
        let skip = self.delay_left.min(data.len());
        self.delay_left -= skip;
        let keep = (data.len() - skip).min(limit.saturating_sub(self.emitted));
        out.extend_from_slice(&data[skip..skip + keep]);
        self.emitted += keep;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, frames: usize, hz: f32) -> Vec<f32> {
        (0..frames)
            .map(|i| (i as f32 / rate as f32 * hz * std::f32::consts::TAU).sin() * 0.5)
            .collect()
    }

    fn run(source: u32, target: u32, input: &[f32], piece: usize) -> Vec<f32> {
        let mut resampler = Resampler::new(source, target).unwrap();
        let mut out = Vec::new();
        for part in input.chunks(piece.max(1)) {
            resampler.push(part, &mut out).unwrap();
        }
        resampler.finish(&mut out).unwrap();
        out
    }

    #[test]
    fn output_length_is_the_exact_ratio_whatever_the_piece_size() {
        for (source, target) in [(48_000, 16_000), (44_100, 16_000), (16_000, 48_000)] {
            for frames in [0usize, 1, 4095, 4096, 4097, 100_000] {
                let input = tone(source, frames, 440.0);
                let expected = frames * target as usize / source as usize;
                for piece in [1, 480, 4096, 1_000_000] {
                    assert_eq!(
                        run(source, target, &input, piece).len(),
                        expected,
                        "{source}->{target}, {frames} frames, pieces of {piece}"
                    );
                }
            }
        }
    }

    #[test]
    fn piece_size_does_not_change_the_samples() {
        let input = tone(48_000, 50_000, 440.0);
        assert_eq!(
            run(48_000, 16_000, &input, 480),
            run(48_000, 16_000, &input, 50_000)
        );
    }

    #[test]
    fn the_first_output_sample_lines_up_with_the_first_input_sample() {
        // A step at input frame 24_000 (0.5 s) must appear at output frame 8_000.
        let mut input = vec![0.0f32; 48_000];
        input[24_000..].fill(0.5);
        let out = run(48_000, 16_000, &input, 480);
        let onset = out.iter().position(|s| *s > 0.25).unwrap();
        assert!(
            (onset as i64 - 8_000).abs() <= 8,
            "step landed at {onset}, expected about 8000"
        );
    }

    #[test]
    fn a_rate_of_zero_is_refused() {
        assert!(Resampler::new(0, 16_000).is_err());
        assert!(Resampler::new(16_000, 0).is_err());
    }
}
