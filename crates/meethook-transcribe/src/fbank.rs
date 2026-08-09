//! Log-mel filterbank features, in the exact shape WeSpeaker expects to be fed.
//!
//! The embedding graph does not take audio. Its input is 80 kaldi-compatible log-mel
//! energies per 10 ms frame, and every parameter below -- window length, window *function*,
//! pre-emphasis, the mel scale, the flooring -- has to match what the checkpoint was
//! trained on. Get one wrong and nothing fails: embeddings still come out, still look like
//! embeddings, and quietly stop putting the same voice near itself.
//!
//! Two of those parameters are WeSpeaker-specific rather than general kaldi practice, and
//! both are called out where they happen: the int16 scaling in [`Fbank::compute`] and the
//! per-utterance mean normalization at the end of it.
//!
//! This is a from-scratch implementation over the `realfft` dependency the workspace
//! already has, rather than a binding to kaldi's C++. That is only defensible because it
//! is checked against the real thing: `testdata/kaldi-fbank-hamming.bin` holds a fixed
//! input and the features kaldi-native-fbank produces from it, and the test at the bottom
//! asserts this module reproduces them. Changing anything here without that test still
//! passing means the features no longer match the ones the network learned on.

use std::sync::Arc;

use realfft::RealToComplex;
use realfft::num_complex::Complex;

use crate::audio::TARGET_RATE;

/// Mel bins per frame -- the 80 of the graph's `[B, T, 80]` input.
pub const MEL_BINS: usize = 80;

/// 25 ms of analysis every 10 ms, kaldi's defaults and WeSpeaker's.
const FRAME_LENGTH: usize = (TARGET_RATE as usize * 25) / 1000;
const FRAME_SHIFT: usize = (TARGET_RATE as usize * 10) / 1000;

/// The FFT length: the frame rounded up to a power of two, zero-padded.
const PADDED_LENGTH: usize = 512;

/// Mel bank edges. 20 Hz cuts rumble below speech; the top is the Nyquist frequency.
const LOW_FREQ: f32 = 20.0;
const HIGH_FREQ: f32 = TARGET_RATE as f32 / 2.0;

/// First-difference filter that flattens the ~-6 dB/octave tilt of voiced speech.
const PREEMPHASIS: f32 = 0.97;

/// One triangular mel filter, stored as the run of FFT bins it actually touches.
struct MelFilter {
    first_bin: usize,
    weights: Vec<f32>,
}

/// A reusable feature extractor: FFT plan, window, and mel bank, built once.
///
/// Construction is the expensive part (a 512-point real FFT plan and 80 triangular
/// filters), so a caller embedding hundreds of turns should build one of these and keep it.
pub struct Fbank {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    filters: Vec<MelFilter>,
    /// Scratch, reused per frame so a long meeting does not allocate per 10 ms of it.
    frame: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    power: Vec<f32>,
}

impl Fbank {
    pub fn new() -> Self {
        Fbank {
            fft: realfft::RealFftPlanner::<f32>::new().plan_fft_forward(PADDED_LENGTH),
            window: hamming_window(),
            filters: mel_filters(),
            frame: vec![0.0; PADDED_LENGTH],
            spectrum: vec![Complex::default(); PADDED_LENGTH / 2 + 1],
            power: vec![0.0; PADDED_LENGTH / 2 + 1],
        }
    }

    /// Features for one utterance: `[T, 80]` row-major, ready to be fed as `[1, T, 80]`.
    ///
    /// `samples` is 16 kHz mono in `[-1, 1]`. Anything shorter than one 25 ms frame yields
    /// an empty vector rather than an error -- there is no such thing as a partial frame
    /// here, and a caller that has to distinguish "too short" from "failed" has been handed
    /// a decision it does not want.
    ///
    /// Frames are snipped at the edges, kaldi's default: only whole frames that fit inside
    /// the audio are emitted, so `T == 1 + (samples - 400) / 160`.
    pub fn compute(&mut self, samples: &[f32]) -> Vec<f32> {
        if samples.len() < FRAME_LENGTH {
            return Vec::new();
        }
        let frames = 1 + (samples.len() - FRAME_LENGTH) / FRAME_SHIFT;
        let mut out = vec![0.0f32; frames * MEL_BINS];

        for f in 0..frames {
            let start = f * FRAME_SHIFT;
            // WeSpeaker convention, not general kaldi practice: the network was trained on
            // fbank computed over a waveform in int16 range, and everything in this
            // workspace carries audio as floats in [-1, 1]. Skip this and every energy is
            // 90 dB low, which the log turns into a large constant offset -- survivable in
            // principle, since the mean normalization below removes constants, but not
            // where the log floor bites into quiet bins.
            for (dst, src) in self.frame[..FRAME_LENGTH]
                .iter_mut()
                .zip(&samples[start..start + FRAME_LENGTH])
            {
                *dst = src * 32768.0;
            }
            self.frame[FRAME_LENGTH..].fill(0.0);

            self.frame_energies(&mut out[f * MEL_BINS..(f + 1) * MEL_BINS]);
        }

        // Per-utterance cepstral mean normalization, again a WeSpeaker convention rather
        // than part of fbank: subtract each bin's mean over time. This is what makes an
        // embedding describe the voice instead of the microphone and room it was recorded
        // through -- and what makes two clips of one person comparable at all.
        subtract_mean_over_time(&mut out, frames);
        out
    }

    /// Turns `self.frame` -- one padded frame, already scaled -- into its 80 log-mel
    /// energies.
    fn frame_energies(&mut self, into: &mut [f32]) {
        let frame = &mut self.frame[..FRAME_LENGTH];

        // DC offset, then pre-emphasis, then the window, in that order. Kaldi's order, and
        // not interchangeable: pre-emphasising a frame that still has its DC offset leaves
        // 3% of that offset behind in every sample.
        let mean = frame.iter().sum::<f32>() / FRAME_LENGTH as f32;
        frame.iter_mut().for_each(|s| *s -= mean);
        for i in (1..FRAME_LENGTH).rev() {
            frame[i] -= PREEMPHASIS * frame[i - 1];
        }
        frame[0] -= PREEMPHASIS * frame[0];
        for (s, w) in frame.iter_mut().zip(&self.window) {
            *s *= w;
        }

        self.fft
            .process(&mut self.frame, &mut self.spectrum)
            .expect("the FFT plan and its buffers are built to the same fixed length");
        for (p, c) in self.power.iter_mut().zip(&self.spectrum) {
            *p = c.re * c.re + c.im * c.im;
        }

        for (energy, filter) in into.iter_mut().zip(&self.filters) {
            let bins = &self.power[filter.first_bin..filter.first_bin + filter.weights.len()];
            let sum: f32 = bins
                .iter()
                .zip(&filter.weights)
                .map(|(p, w)| p * w)
                .sum::<f32>();
            // Floor before the log so silence is a large negative number rather than
            // negative infinity. A single -inf poisons the mean below and, through it,
            // every frame of the utterance.
            *energy = sum.max(f32::EPSILON).ln();
        }
    }
}

impl Default for Fbank {
    fn default() -> Self {
        Fbank::new()
    }
}

/// Subtracts each of the 80 dimensions' mean over the `frames` rows of `features`.
fn subtract_mean_over_time(features: &mut [f32], frames: usize) {
    let mut means = [0.0f32; MEL_BINS];
    for row in features.chunks_exact(MEL_BINS) {
        for (m, v) in means.iter_mut().zip(row) {
            *m += v;
        }
    }
    for m in &mut means {
        *m /= frames as f32;
    }
    for row in features.chunks_exact_mut(MEL_BINS) {
        for (v, m) in row.iter_mut().zip(&means) {
            *v -= m;
        }
    }
}

/// The window WeSpeaker trains with.
///
/// Hamming, explicitly -- and this is the one parameter here that a reasonable person would
/// get wrong. Kaldi's default is Povey, and so is the default of every off-the-shelf
/// extractor that gets pointed at this checkpoint, but `wespeaker/dataset/processor.py`
/// passes `window_type='hamming'` when it builds the features the network is trained on.
/// The two windows differ by a few percent in the taper, which is enough to move
/// embeddings without breaking anything visibly.
///
/// The `frame_length - 1` denominator (a symmetric window, not a periodic one) is kaldi's,
/// and is what the reference fixture pins.
fn hamming_window() -> Vec<f32> {
    let a = std::f64::consts::TAU / (FRAME_LENGTH - 1) as f64;
    (0..FRAME_LENGTH)
        .map(|i| (0.54 - 0.46 * (a * i as f64).cos()) as f32)
        .collect()
}

/// The 80 triangular filters, evenly spaced on the HTK mel scale between 20 Hz and Nyquist.
///
/// Adjacent triangles overlap by half: filter `b` rises from the centre of `b - 1` and
/// falls to the centre of `b + 1`, which is why the spacing divides by `bins + 1` rather
/// than by `bins`.
fn mel_filters() -> Vec<MelFilter> {
    // Only the first half of the spectrum carries distinct frequencies, and the Nyquist bin
    // itself is excluded by the strict comparison below, exactly as in kaldi.
    let fft_bins = PADDED_LENGTH / 2;
    let bin_width = TARGET_RATE as f32 / PADDED_LENGTH as f32;
    let (low_mel, high_mel) = (mel(LOW_FREQ), mel(HIGH_FREQ));
    let delta = (high_mel - low_mel) / (MEL_BINS + 1) as f32;

    (0..MEL_BINS)
        .map(|b| {
            let left = low_mel + b as f32 * delta;
            let centre = low_mel + (b + 1) as f32 * delta;
            let right = low_mel + (b + 2) as f32 * delta;

            let mut first_bin = 0;
            let mut weights = Vec::new();
            for i in 0..fft_bins {
                let m = mel(bin_width * i as f32);
                if m <= left || m >= right {
                    continue;
                }
                if weights.is_empty() {
                    first_bin = i;
                }
                weights.push(if m <= centre {
                    (m - left) / (centre - left)
                } else {
                    (right - m) / (right - centre)
                });
            }
            MelFilter { first_bin, weights }
        })
        .collect()
}

/// The HTK mel scale. Single precision on purpose: kaldi computes the filter weights in
/// `float`, and the fixture this module is checked against inherits that.
fn mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fixed input and the features kaldi-native-fbank computes from it.
    ///
    /// Generated once, outside this workspace, by the real C++ implementation (the `knf-rs`
    /// crate, patched to WeSpeaker's Hamming window) -- see this ticket's notes for the
    /// harness. Layout: `MHFBANK1`, then `samples`, `frames` and `bins` as little-endian
    /// `u32`, then that many `f32` samples in `[-1, 1]`, then `frames * bins` `f32`
    /// features, row-major, after mean normalization.
    ///
    /// Committed rather than regenerated because the point is to be able to check this
    /// implementation forever without a C++ toolchain, a cmake build, or a network.
    const REFERENCE: &[u8] = include_bytes!("../testdata/kaldi-fbank-hamming.bin");

    fn reference() -> (Vec<f32>, usize, Vec<f32>) {
        assert_eq!(&REFERENCE[..8], b"MHFBANK1");
        let word = |i: usize| {
            u32::from_le_bytes(REFERENCE[8 + i * 4..12 + i * 4].try_into().unwrap()) as usize
        };
        let (samples, frames, bins) = (word(0), word(1), word(2));
        assert_eq!(bins, MEL_BINS);

        let floats: Vec<f32> = REFERENCE[20..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(floats.len(), samples + frames * bins);
        let (audio, features) = floats.split_at(samples);
        (audio.to_vec(), frames, features.to_vec())
    }

    /// The test the rest of this module exists to pass.
    ///
    /// Every parameter -- Hamming window, 0.97 pre-emphasis, DC removal, the 512-point
    /// zero-padded FFT, the HTK mel bank from 20 Hz to Nyquist, the epsilon floor, the
    /// int16 scaling, the mean normalization -- is pinned by this comparison at once. A
    /// tolerance of 1e-3 on a log-energy is roughly a tenth of a percent in the linear
    /// domain, and covers kaldi doing its FFT in double precision where this does it in
    /// single.
    #[test]
    fn features_match_kaldi_native_fbank_on_a_fixed_input() {
        let (audio, frames, expected) = reference();

        let got = Fbank::new().compute(&audio);

        assert_eq!(got.len(), frames * MEL_BINS, "frame count");
        let worst = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "worst deviation from kaldi was {worst}");
    }

    /// The trailing tenth of a second of the fixture is exact silence, where every mel
    /// energy is zero and the log has to be floored rather than allowed to reach -inf.
    ///
    /// Asserted separately from the comparison above because a NaN or an infinity compares
    /// unequal to everything, including itself, and would show up there as an unhelpful
    /// "worst deviation was NaN".
    #[test]
    fn silence_produces_finite_features() {
        let (audio, _, _) = reference();
        let got = Fbank::new().compute(&audio);
        assert!(got.iter().all(|v| v.is_finite()), "non-finite feature");
    }

    #[test]
    fn audio_shorter_than_one_frame_produces_no_features() {
        let mut fbank = Fbank::new();
        assert!(fbank.compute(&[]).is_empty());
        assert!(fbank.compute(&vec![0.1; FRAME_LENGTH - 1]).is_empty());
        assert_eq!(fbank.compute(&vec![0.1; FRAME_LENGTH]).len(), MEL_BINS);
    }

    /// One frame every 10 ms, and only whole frames: a second of audio is 98 frames, not
    /// 100. Cheap to state and the first thing to check when timings look stretched.
    #[test]
    fn frame_count_follows_the_snip_edges_rule() {
        let mut fbank = Fbank::new();
        let one_second = vec![0.01; TARGET_RATE as usize];
        assert_eq!(fbank.compute(&one_second).len() / MEL_BINS, 98);
    }

    /// Mean normalization is over time, per mel bin -- not over the 80 bins of a frame.
    /// Both are one-line loops and only one is right; this is the difference.
    #[test]
    fn every_mel_bin_averages_to_zero_over_the_utterance() {
        let (audio, frames, _) = reference();
        let got = Fbank::new().compute(&audio);

        for bin in 0..MEL_BINS {
            let mean: f32 = got.iter().skip(bin).step_by(MEL_BINS).sum::<f32>() / frames as f32;
            assert!(mean.abs() < 1e-3, "bin {bin} has mean {mean}");
        }
    }
}
