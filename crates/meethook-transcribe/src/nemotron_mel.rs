//! The 128-bin log-mel features Nemotron-3-Diarization reads.
//!
//! This is a different recipe from [`crate::fbank`] (80 Kaldi bins for WeSpeaker), so it is a
//! separate extractor. It reuses the `realfft` plan pattern from there; no mel filterbank
//! crate is in the dependency tree, and `fbank`'s bank is HTK-scale, so the Slaney bank below
//! is built by hand (the formula of `transformers.audio_utils.mel_filter_bank` and
//! `librosa.filters.mel`).
//!
//! The recipe follows the Hugging Face feature extractor the model was trained against, as
//! recorded in `testdata/nemotron/NEMOTRON-FIXTURES.md` (not the doc-015 sketch, which differs
//! in two places):
//!
//! - pre-emphasis `y[t] = x[t] - 0.97 x[t-1]`, `y[0] = x[0]`, applied before padding;
//! - 256 zeros of centre padding each side, `n_fft` 512, hop 160, so `1 + n / 160` frames;
//! - a *symmetric* Hann window of 400 samples, centred inside the 512-point FFT;
//! - power spectrum, 128 Slaney-scale Slaney-normalised mel filters over 0..8000 Hz,
//!   `ln(mel + 2^-24)`, no normalisation;
//! - frames at and after `n / 160` are exactly `0.0` (the processor's attention mask), not the
//!   log of silence.
//!
//! The graph stacks [`STACK`] mel frames into one encoder frame, so a caller feeding it chunks
//! must start and size them on multiples of [`STACK`].

use std::sync::Arc;

use realfft::RealToComplex;
use realfft::num_complex::Complex;

/// Mel bins per frame.
pub const MEL_BINS: usize = 128;

/// Mel frames the graph stacks into one encoder frame; chunk boundaries fall on multiples.
pub const STACK: usize = 8;

const SAMPLE_RATE: f32 = 16_000.0;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP: usize = 160;
const PAD: usize = N_FFT / 2;
/// Where the window starts inside the FFT frame (torch centre-pads a short window).
const WINDOW_OFFSET: usize = (N_FFT - WIN_LENGTH) / 2;
const PREEMPHASIS: f32 = 0.97;
const LOG_FLOOR: f32 = 1.0 / 16_777_216.0; // 2^-24
const FFT_BINS: usize = N_FFT / 2 + 1;

/// Mel frames produced for `samples` samples of audio, including the trailing partial one.
pub fn frame_count(samples: usize) -> usize {
    1 + samples / HOP
}

/// Frames carrying real signal; later ones are zero.
pub fn valid_frames(samples: usize) -> usize {
    samples / HOP
}

struct MelFilter {
    first_bin: usize,
    weights: Vec<f32>,
}

/// A reusable extractor: FFT plan, window and mel bank, built once. Pure: no model, no weights.
pub struct NemotronMel {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    filters: Vec<MelFilter>,
    frame: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    power: Vec<f32>,
}

impl NemotronMel {
    pub fn new() -> Self {
        let window = (0..WIN_LENGTH)
            .map(|i| {
                0.5 - 0.5
                    * (2.0 * std::f64::consts::PI * i as f64 / (WIN_LENGTH - 1) as f64).cos() as f32
            })
            .collect();
        NemotronMel {
            fft: realfft::RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT),
            window,
            filters: slaney_filters(),
            frame: vec![0.0; N_FFT],
            spectrum: vec![Complex::default(); FFT_BINS],
            power: vec![0.0; FFT_BINS],
        }
    }

    /// Features for every frame of the track, `[frame_count, 128]` row-major.
    pub fn compute_all(&mut self, samples: &[f32]) -> Vec<f32> {
        self.compute_range(samples, 0, frame_count(samples.len()))
    }

    /// Mel frames `first..first + count` of the whole track, `[count, 128]` row-major, without
    /// computing any other frame. `samples` is the whole track; only the samples under the
    /// requested frames are read (plus the one before each, for pre-emphasis), so memory is
    /// proportional to `count`. A range running past the last frame is truncated.
    pub fn compute_range(&mut self, samples: &[f32], first: usize, count: usize) -> Vec<f32> {
        let total = frame_count(samples.len());
        let end = total.min(first.saturating_add(count));
        if first >= end {
            return Vec::new();
        }
        let valid = valid_frames(samples.len());
        let mut out = vec![0.0f32; (end - first) * MEL_BINS];

        for f in first..end.min(valid) {
            // The window starts at padded position f*HOP + WINDOW_OFFSET, which is signal
            // index f*HOP + WINDOW_OFFSET - PAD.
            let origin = (f * HOP + WINDOW_OFFSET) as isize - PAD as isize;
            self.frame.fill(0.0);
            for (i, w) in self.window.iter().enumerate() {
                let t = origin + i as isize;
                if t < 0 || t as usize >= samples.len() {
                    continue;
                }
                let t = t as usize;
                let y = if t == 0 {
                    samples[0]
                } else {
                    samples[t] - PREEMPHASIS * samples[t - 1]
                };
                self.frame[WINDOW_OFFSET + i] = y * w;
            }
            self.fft
                .process(&mut self.frame, &mut self.spectrum)
                .expect("the FFT plan and its buffers are built to the same fixed length");
            for (p, c) in self.power.iter_mut().zip(&self.spectrum) {
                *p = c.re * c.re + c.im * c.im;
            }
            let row = &mut out[(f - first) * MEL_BINS..(f - first + 1) * MEL_BINS];
            for (energy, filter) in row.iter_mut().zip(&self.filters) {
                let bins = &self.power[filter.first_bin..filter.first_bin + filter.weights.len()];
                let sum: f32 = bins.iter().zip(&filter.weights).map(|(p, w)| p * w).sum();
                *energy = (sum + LOG_FLOOR).ln();
            }
        }
        out
    }
}

impl Default for NemotronMel {
    fn default() -> Self {
        NemotronMel::new()
    }
}

fn hz_to_mel(hz: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        min_log_mel + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        hz / F_SP
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = 6.4f64.ln() / 27.0;
    if mel >= min_log_mel {
        MIN_LOG_HZ * (logstep * (mel - min_log_mel)).exp()
    } else {
        mel * F_SP
    }
}

/// 128 triangular filters on the Slaney mel scale with Slaney (area) normalisation, 0..8000 Hz.
fn slaney_filters() -> Vec<MelFilter> {
    let mel_max = hz_to_mel(SAMPLE_RATE as f64 / 2.0);
    let edges: Vec<f64> = (0..MEL_BINS + 2)
        .map(|i| mel_to_hz(mel_max * i as f64 / (MEL_BINS + 1) as f64))
        .collect();
    let fft_freq = |k: usize| k as f64 * (SAMPLE_RATE as f64 / 2.0) / (FFT_BINS - 1) as f64;

    (0..MEL_BINS)
        .map(|m| {
            let (lo, mid, hi) = (edges[m], edges[m + 1], edges[m + 2]);
            let norm = 2.0 / (hi - lo);
            let weights: Vec<(usize, f64)> = (0..FFT_BINS)
                .filter_map(|k| {
                    let f = fft_freq(k);
                    let w = ((f - lo) / (mid - lo)).min((hi - f) / (hi - mid));
                    (w > 0.0).then_some((k, w * norm))
                })
                .collect();
            // A triangle narrower than one FFT bin can touch none; keep an empty filter.
            match (weights.first(), weights.last()) {
                (Some(&(first, _)), Some(&(last, _))) => MelFilter {
                    first_bin: first,
                    weights: (first..=last)
                        .map(|k| {
                            weights
                                .iter()
                                .find(|(b, _)| *b == k)
                                .map_or(0.0, |(_, w)| *w as f32)
                        })
                        .collect(),
                },
                _ => MelFilter {
                    first_bin: 0,
                    weights: Vec::new(),
                },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CLIP: &[u8] = include_bytes!("../testdata/nemotron/nemotron-clip.mulaw");
    const FEATURES: &[u8] = include_bytes!("../testdata/nemotron/nemotron-features.bin");

    fn decode_mulaw(code: u8) -> f32 {
        let c = !code;
        let sign = c & 0x80;
        let exp = (c >> 4) & 7;
        let mant = (c & 0x0F) as i32;
        let x = (((mant << 3) + 0x84) << exp) - 0x84;
        (if sign != 0 { -x } else { x }) as f32 / 32768.0
    }

    fn clip() -> Vec<f32> {
        CLIP.iter().map(|&b| decode_mulaw(b)).collect()
    }

    fn u32_at(i: usize) -> u32 {
        u32::from_le_bytes(FEATURES[i * 4..i * 4 + 4].try_into().unwrap())
    }

    fn floats(start: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let o = start + i * 4;
                f32::from_le_bytes(FEATURES[o..o + 4].try_into().unwrap())
            })
            .collect()
    }

    #[test]
    fn matches_the_huggingface_fixtures_at_both_edges() {
        let (frames, bins, head, tail) = (
            u32_at(0) as usize,
            u32_at(1) as usize,
            u32_at(2) as usize,
            u32_at(3) as usize,
        );
        assert_eq!((frames, bins), (10001, MEL_BINS));
        let head_ref = floats(16, head * bins);
        let tail_ref = floats(16 + head * bins * 4, tail * bins);

        let samples = clip();
        assert_eq!(frame_count(samples.len()), frames);
        let mut mel = NemotronMel::new();
        let all = mel.compute_all(&samples);
        assert_eq!(all.len(), frames * bins);

        let max_diff = |got: &[f32], want: &[f32]| {
            got.iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };
        let head_diff = max_diff(&all[..head * bins], &head_ref);
        let tail_diff = max_diff(&all[(frames - tail) * bins..], &tail_ref);
        assert!(head_diff <= 1e-3, "head max abs diff {head_diff}");
        assert!(tail_diff <= 1e-3, "tail max abs diff {tail_diff}");
        // The trailing partial frame is masked to exactly zero, not ln(floor).
        assert!(all[(frames - 1) * bins..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn any_range_equals_the_same_slice_of_the_whole_track() {
        let samples = clip();
        let mut mel = NemotronMel::new();
        let all = mel.compute_all(&samples);
        let total = frame_count(samples.len());
        for (first, count) in [
            (0, 8),
            (8, 64),
            (4000, 340),
            (9937, 64),
            (9900, 500),
            (0, total),
        ] {
            let got = mel.compute_range(&samples, first, count);
            let end = total.min(first + count);
            assert_eq!(
                got,
                all[first * MEL_BINS..end * MEL_BINS],
                "range {first}+{count}"
            );
        }
        assert!(mel.compute_range(&samples, total, 10).is_empty());
    }

    #[test]
    fn first_frame_sees_zero_padding_and_unchanged_first_sample() {
        // A lone impulse at x[0]: pre-emphasis leaves it at 1.0 (not 1.0 - 0.97 * pad).
        let mut samples = vec![0.0f32; 800];
        samples[0] = 1.0;
        let mut mel = NemotronMel::new();
        let first = mel.compute_range(&samples, 0, 1);
        // Same impulse shifted by one sample would see y[1] = -0.97 also; with the impulse at 0
        // the frame must be finite and above the floor in at least one bin.
        assert!(first.iter().all(|v| v.is_finite()));
        assert!(first.iter().any(|&v| v > LOG_FLOOR.ln() + 1.0));
    }

    #[test]
    fn tiny_inputs_do_not_panic_and_follow_the_frame_formula() {
        let mut mel = NemotronMel::new();
        for n in [0usize, 159, 160, 400] {
            let samples = vec![0.1f32; n];
            let out = mel.compute_all(&samples);
            assert_eq!(out.len(), (1 + n / 160) * MEL_BINS, "n = {n}");
        }
    }

    #[test]
    fn filterbank_has_128_area_normalised_nonempty_filters() {
        let filters = slaney_filters();
        assert_eq!(filters.len(), MEL_BINS);
        for (i, f) in filters.iter().enumerate() {
            assert!(!f.weights.is_empty(), "filter {i} is empty");
            assert!(f.weights.iter().all(|&w| w >= 0.0));
        }
        // Slaney normalisation: each triangle's area in Hz is about 1, so the sum of weights
        // times the 31.25 Hz bin width is about 1 for the filters wide enough to sample well.
        for f in &filters[80..] {
            let area: f32 = f.weights.iter().sum::<f32>() * 31.25;
            assert!((area - 1.0).abs() < 0.1, "area {area}");
        }
    }
}
