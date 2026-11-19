//! Voice isolation: strips room noise from a track with Inworld's voice isolator.
//!
//! [`VoiceIsolator::isolate`] takes 48 kHz mono `f32` samples and returns the isolated voice at
//! the same length and rate. The STFT, the 8 s windows with 2 s crossfades and the inverse STFT
//! are all inside, so a caller passes audio and nothing else.
//!
//! The graph is the community ONNX conversion of the Apache-2.0 64M-parameter denoiser, pinned
//! as [`crate::VOICE_ISOLATOR_MODEL`]. It reads the real/imaginary STFT of the audio (n_fft
//! 2048, hop 480, periodic Hann, `center=True` with reflect padding) as input `spec`
//! `[1, frames, 1025, 2]` and returns a complex `correction` of the same shape; the voice is
//! `audio + istft(correction, length = len(audio))`. It was trained on 8 s windows (801
//! frames), so longer audio runs as 8 s windows overlapped by 2 s and blended with a linear
//! crossfade.
//!
//! No resampling is done here: the caller owns the rate. (`rubato` is already a dependency for
//! that, in [`crate::audio`].)
//!
//! # CoreML, CPU and speed
//!
//! Measured with `examples/isolate-voice` on `sample_14.wav` (10.77 s, which is two windows),
//! release build, on a MacBook Pro 18,1 (Apple M1 Pro, 32 GB), macOS, ONNX Runtime from the
//! flake:
//!
//! | path | result |
//! |---|---|
//! | CoreML ([`VoiceIsolator::open_coreml`], via [`crate::open_session`]) | **Accepted at load** (28.6 s; whether every node or only a partition went to CoreML was not inspected), **then unusable**: the first inference ran past 2 minutes without finishing one window and the process was killed for memory (peak footprint about 48 GB on a 32 GB machine; a second run was terminated at 150 s). |
//! | CPU ([`VoiceIsolator::open`]) | Loads in 23 s. 10.77 s of audio in 40.3 s wall = **0.27x realtime** (audio seconds per wall second). Peak resident memory about 12.9 GB. |
//! | Python package on PyTorch / Apple Metal (the earlier trial) | about 0.41x realtime |
//!
//! Consequences for the caller (TASK-080): [`VoiceIsolator::open`] is CPU on purpose. At 0.27x
//! realtime one hour of microphone track is about 3.7 hours of work, so isolating a whole track
//! is not free: run it over speech regions only, and expect the 13 GB working set (the graph
//! attends over the whole 801-frame window). Whether a different CoreML configuration (for
//! example static shapes or `MLProgram` format) helps was not tried.
//!
//! # Parity
//!
//! `matches_the_official_python_package` (ignored: it needs the 260 MB download) compares the
//! output on `testdata/voice-isolator/sample_14.wav` with the official `inworld-voice-isolation`
//! 1.0.0 package's, produced by `scripts/voice_isolator_reference.py`. Measured SNR: 46.7 dB
//! (required: 35 dB). The sample is Apache-2.0 from `inworld-ai/voice-isolator`; no meeting
//! audio is used anywhere.

use std::path::Path;
use std::sync::Arc;

use meethook_models::ModelSpec;
use ort::session::Session;
use ort::value::TensorRef;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::onnx::{open_cpu_session, open_session};
use crate::{Error, Result, VOICE_ISOLATOR_MODEL};

/// The only rate the graph works at.
pub const SAMPLE_RATE: u32 = 48_000;

const N_FFT: usize = 2048;
const HOP: usize = 480;
const BINS: usize = N_FFT / 2 + 1;
const PAD: usize = N_FFT / 2;
/// 801 frames: the length the model was trained on.
const WINDOW: usize = 800 * HOP;
/// 2 s of overlap between neighbouring windows.
const OVERLAP: usize = 2 * SAMPLE_RATE as usize;
const STRIDE: usize = WINDOW - OVERLAP;

/// A loaded voice-isolator graph and the FFT plans around it.
pub struct VoiceIsolator {
    session: Session,
    /// True only for [`VoiceIsolator::open_coreml`] when CoreML took the graph.
    pub accelerated: bool,
    stft: Stft,
}

impl VoiceIsolator {
    /// Loads the graph from `models_dir`, the directory the registry installs into, on the CPU.
    ///
    /// CPU rather than the CoreML-then-CPU policy of [`crate::open_session`], because on the
    /// measured machine CoreML accepts this graph at load and then exhausts memory on the first
    /// inference (see the module docs). A default that can take a machine down is not a default.
    pub fn open(models_dir: &Path) -> Result<Self> {
        let path = installed(models_dir)?;
        Ok(Self {
            session: open_cpu_session(&path)?,
            accelerated: false,
            stft: Stft::new(),
        })
    }

    /// Loads the graph through [`crate::open_session`] (CoreML, then CPU). Kept for measuring
    /// the accelerator; see [`VoiceIsolator::open`] for why it is not the default. Running it
    /// can exhaust memory.
    pub fn open_coreml(models_dir: &Path) -> Result<Self> {
        let path = installed(models_dir)?;
        let loaded = open_session(&path)?;
        Ok(Self {
            session: loaded.session,
            accelerated: loaded.accelerated,
            stft: Stft::new(),
        })
    }

    /// The isolated voice: same length and rate as `audio`, which must be 48 kHz mono.
    ///
    /// Empty input returns empty without touching the graph. Input shorter than one window is
    /// zero-padded to one, run, and cropped back.
    pub fn isolate(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        let windows = plan_windows(audio.len());
        let mut out = vec![0.0f32; audio.len()];
        let mut chunk = vec![0.0f32; WINDOW];
        for (index, window) in windows.iter().enumerate() {
            let take = (audio.len() - window.start).min(WINDOW);
            chunk[..take].copy_from_slice(&audio[window.start..window.start + take]);
            chunk[take..].fill(0.0);

            let voice = self.run_window(&chunk)?;

            let first = index == 0;
            let last = index + 1 == windows.len();
            for j in 0..take {
                let weight = crossfade_weight(j, first, last);
                out[window.start + j] += weight * voice[j];
            }
        }
        Ok(out)
    }

    /// One full window through the graph: `audio + istft(correction)`.
    fn run_window(&mut self, window: &[f32]) -> Result<Vec<f32>> {
        debug_assert_eq!(window.len(), WINDOW);
        let spec = self.stft.forward(window);
        let frames = spec.len() / (BINS * 2);
        let input = TensorRef::from_array_view(([1usize, frames, BINS, 2], &spec[..]))
            .map_err(inference_failed)?;
        let outputs = self
            .session
            .run(ort::inputs!["spec" => input])
            .map_err(inference_failed)?;
        let (shape, correction) = outputs["correction"]
            .try_extract_tensor::<f32>()
            .map_err(inference_failed)?;
        if correction.len() != spec.len() {
            return Err(Error::VoiceIsolator(format!(
                "output `correction` has shape {shape:?}, expected [1, {frames}, {BINS}, 2]"
            )));
        }
        let mut voice = self.stft.inverse(correction, window.len());
        for (v, a) in voice.iter_mut().zip(window) {
            *v += a;
        }
        Ok(voice)
    }
}

fn installed(models_dir: &Path) -> Result<std::path::PathBuf> {
    let spec: &ModelSpec = &VOICE_ISOLATOR_MODEL;
    let path = models_dir.join(spec.file_name);
    if !path.is_file() {
        return Err(Error::VoiceIsolator(format!(
            "{} is not installed; run `cargo run --example fetch-onnx-models`",
            path.display()
        )));
    }
    Ok(path)
}

fn inference_failed(source: ort::Error) -> Error {
    Error::VoiceIsolator(source.to_string())
}

/// One graph window: where it starts in the input. Every window is [`WINDOW`] samples long;
/// the last is zero-padded.
#[derive(Debug, PartialEq, Eq)]
struct Window {
    start: usize,
}

/// The windows covering `len` samples: none for empty input, one up to a full window, then a
/// window every [`STRIDE`] samples until one reaches the end.
fn plan_windows(len: usize) -> Vec<Window> {
    if len == 0 {
        return Vec::new();
    }
    let mut windows = vec![Window { start: 0 }];
    while windows.last().is_some_and(|w| w.start + WINDOW < len) {
        let start = windows.last().map_or(0, |w| w.start) + STRIDE;
        windows.push(Window { start });
    }
    windows
}

/// The blend weight of sample `j` of a window. The first window has no left neighbour and the
/// last no right one, so those edges stay at 1; elsewhere a 2 s linear ramp that sums to 1
/// with the neighbour's opposite ramp.
fn crossfade_weight(j: usize, first: bool, last: bool) -> f32 {
    let mut weight = 1.0;
    if !first && j < OVERLAP {
        weight *= (j as f32 + 0.5) / OVERLAP as f32;
    }
    if !last && j >= WINDOW - OVERLAP {
        weight *= 1.0 - ((j - (WINDOW - OVERLAP)) as f32 + 0.5) / OVERLAP as f32;
    }
    weight
}

/// `torch.stft` / `torch.istft` with `n_fft` 2048, hop 480, a periodic Hann window and
/// `center=True` with reflect padding, which is what the graph's `spec` input is.
struct Stft {
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
}

impl Stft {
    fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        // Periodic (denominator N), not the symmetric window `scipy.signal.hann` defaults to.
        let window = (0..N_FFT)
            .map(|i| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32
            })
            .collect();
        Self {
            forward: planner.plan_fft_forward(N_FFT),
            inverse: planner.plan_fft_inverse(N_FFT),
            window,
        }
    }

    /// `1 + len / HOP` frames of [`BINS`] bins, as an interleaved `(re, im)` flat buffer in
    /// `[frame][bin][2]` order. `x` must be longer than [`PAD`] samples (reflect padding).
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        assert!(n > PAD, "reflect padding needs more than {PAD} samples");
        let reflect = |i: isize| -> f32 {
            // i indexes the unpadded signal; mirror about the first and last samples.
            let last = n as isize - 1;
            let mut i = i;
            if i < 0 {
                i = -i;
            }
            if i > last {
                i = 2 * last - i;
            }
            x[i as usize]
        };
        let frames = 1 + n / HOP;
        let mut out = vec![0.0f32; frames * BINS * 2];
        let mut frame = vec![0.0f32; N_FFT];
        let mut spectrum = vec![Complex::new(0.0f32, 0.0); BINS];
        for f in 0..frames {
            let origin = (f * HOP) as isize - PAD as isize;
            for (k, slot) in frame.iter_mut().enumerate() {
                *slot = reflect(origin + k as isize) * self.window[k];
            }
            self.forward
                .process(&mut frame, &mut spectrum)
                .expect("buffer lengths match the plan");
            let row = &mut out[f * BINS * 2..(f + 1) * BINS * 2];
            for (b, c) in spectrum.iter().enumerate() {
                row[2 * b] = c.re;
                row[2 * b + 1] = c.im;
            }
        }
        out
    }

    /// The inverse of [`Stft::forward`] for `spec` in the same layout: windowed overlap-add
    /// normalised by the summed squared window, centre padding trimmed, `length` samples.
    fn inverse(&self, spec: &[f32], length: usize) -> Vec<f32> {
        let frames = spec.len() / (BINS * 2);
        let total = (frames - 1) * HOP + N_FFT;
        let mut signal = vec![0.0f32; total];
        let mut envelope = vec![0.0f32; total];
        let mut spectrum = vec![Complex::new(0.0f32, 0.0); BINS];
        let mut frame = vec![0.0f32; N_FFT];
        let scale = 1.0 / N_FFT as f32;
        for f in 0..frames {
            let row = &spec[f * BINS * 2..(f + 1) * BINS * 2];
            for (b, c) in spectrum.iter_mut().enumerate() {
                *c = Complex::new(row[2 * b], row[2 * b + 1]);
            }
            // A real signal has no imaginary DC or Nyquist component; the inverse requires it.
            spectrum[0].im = 0.0;
            spectrum[BINS - 1].im = 0.0;
            self.inverse
                .process(&mut spectrum, &mut frame)
                .expect("buffer lengths match the plan");
            let at = f * HOP;
            for k in 0..N_FFT {
                signal[at + k] += frame[k] * scale * self.window[k];
                envelope[at + k] += self.window[k] * self.window[k];
            }
        }
        (0..length)
            .map(|i| {
                let e = envelope[PAD + i];
                if e > 1e-11 { signal[PAD + i] / e } else { 0.0 }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic noise plus two tones, so the round trip is not flattered by a single bin.
    fn signal(n: usize) -> Vec<f32> {
        let mut state = 0x2545_f491u32;
        (0..n)
            .map(|i| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let noise = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
                let t = i as f32 / SAMPLE_RATE as f32;
                0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 3150.0 * t).sin()
                    + 0.1 * noise
            })
            .collect()
    }

    #[test]
    fn the_hann_window_is_periodic() {
        let w = Stft::new().window;
        assert_eq!(w.len(), N_FFT);
        assert_eq!(w[0], 0.0);
        assert!((w[N_FFT / 2] - 1.0).abs() < 1e-6);
        // Periodic: w[i] == w[N - i], and the last sample is not zero (the symmetric one is).
        for i in 1..N_FFT {
            assert!((w[i] - w[N_FFT - i]).abs() < 1e-6, "{i}");
        }
        assert!(w[N_FFT - 1] > 0.0);
    }

    #[test]
    fn stft_then_istft_recovers_the_signal() {
        let stft = Stft::new();
        for n in [PAD + 1, 1500, 4800, 48_000, WINDOW] {
            let x = signal(n);
            let spec = stft.forward(&x);
            assert_eq!(spec.len(), (1 + n / HOP) * BINS * 2, "frames for {n}");
            let y = stft.inverse(&spec, n);
            assert_eq!(y.len(), n);
            let worst = x
                .iter()
                .zip(&y)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 1e-4, "n = {n}: worst error {worst}");
        }
    }

    #[test]
    fn a_full_window_is_801_frames() {
        let spec = Stft::new().forward(&vec![0.0; WINDOW]);
        assert_eq!(spec.len() / (BINS * 2), 801);
    }

    #[test]
    fn a_pure_tone_lands_in_its_bin() {
        // 48000 / 2048 Hz per bin: bin 64 is 1500 Hz exactly.
        let n = 8192;
        let x: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 1500.0 * i as f32 / 48_000.0).sin())
            .collect();
        let spec = Stft::new().forward(&x);
        let frame = 8;
        let mag = |b: usize| {
            let o = (frame * BINS + b) * 2;
            spec[o].hypot(spec[o + 1])
        };
        assert!(mag(64) > 100.0 * mag(200), "{} vs {}", mag(64), mag(200));
    }

    #[test]
    fn window_planning_covers_the_input() {
        assert!(plan_windows(0).is_empty());
        for len in [1, 1000, WINDOW - 1, WINDOW] {
            assert_eq!(plan_windows(len), vec![Window { start: 0 }], "{len}");
        }
        assert_eq!(
            plan_windows(WINDOW + 1),
            vec![Window { start: 0 }, Window { start: STRIDE }]
        );
        let len = 5 * SAMPLE_RATE as usize * 10;
        let windows = plan_windows(len);
        assert!(
            windows
                .windows(2)
                .all(|p| p[1].start - p[0].start == STRIDE)
        );
        let end = windows.last().unwrap().start + WINDOW;
        assert!(end >= len);
        // The last window is needed: the one before it stops short of the end.
        assert!(windows[windows.len() - 2].start + WINDOW < len);
    }

    #[test]
    fn crossfades_sum_to_one_across_every_overlap() {
        for j in (0..OVERLAP).step_by(997) {
            let outgoing = crossfade_weight(WINDOW - OVERLAP + j, true, false);
            let incoming = crossfade_weight(j, false, true);
            assert!((outgoing + incoming - 1.0).abs() < 1e-6, "{j}");
        }
        assert_eq!(crossfade_weight(0, true, true), 1.0);
        assert_eq!(crossfade_weight(WINDOW - 1, true, true), 1.0);
        // The middle of a middle window is untouched.
        assert_eq!(crossfade_weight(WINDOW / 2, false, false), 1.0);
    }

    // ---- need the 260 MB weights; run with `-- --ignored` ----

    fn models_dir() -> std::path::PathBuf {
        let root = match std::env::var_os("MEETHOOK_ROOT") {
            Some(root) => std::path::PathBuf::from(root),
            None => std::env::home_dir()
                .expect("no home directory; set MEETHOOK_ROOT")
                .join("meethook"),
        };
        root.join("models")
    }

    fn open_installed() -> VoiceIsolator {
        let dir = models_dir();
        // Same call the real command will make, so this downloads and verifies lazily.
        meethook_models::ensure_model(&dir, &VOICE_ISOLATOR_MODEL, &mut |_, _| {})
            .expect("the voice isolator must download");
        VoiceIsolator::open(&dir).expect("the voice isolator must load")
    }

    fn read_wav(path: &Path) -> Vec<f32> {
        let mut reader = hound::WavReader::open(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let spec = reader.spec();
        assert_eq!(spec.channels, 1, "{}", path.display());
        assert_eq!(spec.sample_rate, SAMPLE_RATE, "{}", path.display());
        match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
            hound::SampleFormat::Int => {
                let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i32>()
                    .map(|s| s.unwrap() as f32 / scale)
                    .collect()
            }
        }
    }

    fn testdata(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/voice-isolator")
            .join(name)
    }

    #[test]
    #[ignore = "downloads the 260 MB voice isolator"]
    fn the_graph_has_the_documented_contract() {
        let isolator = open_installed();
        let inputs = isolator.session.inputs();
        let outputs = isolator.session.outputs();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].name(), "spec");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name(), "correction");
        eprintln!("accelerated: {}", isolator.accelerated);
    }

    #[test]
    #[ignore = "downloads the 260 MB voice isolator"]
    fn short_exact_and_empty_input_keep_their_length() {
        let mut isolator = open_installed();
        for n in [0, 1, 4800, WINDOW - 1, WINDOW] {
            let out = isolator.isolate(&signal(n)).unwrap();
            assert_eq!(out.len(), n);
            assert!(out.iter().all(|s| s.is_finite()), "{n}");
        }
    }

    /// Needs `testdata/voice-isolator/{sample_14.wav, sample_14.reference.wav}`; regenerate
    /// with `scripts/voice_isolator_reference.py`.
    #[test]
    #[ignore = "downloads the 260 MB voice isolator"]
    fn matches_the_official_python_package() {
        let (input, reference) = (
            testdata("sample_14.wav"),
            testdata("sample_14.reference.wav"),
        );
        if !input.is_file() || !reference.is_file() {
            eprintln!("skipping: run scripts/voice_isolator_reference.py to create the sample");
            return;
        }
        let audio = read_wav(&input);
        let want = read_wav(&reference);
        let got = open_installed().isolate(&audio).unwrap();
        assert_eq!(got.len(), want.len());
        let signal: f64 = want.iter().map(|&s| f64::from(s).powi(2)).sum();
        let noise: f64 = want
            .iter()
            .zip(&got)
            .map(|(&a, &b)| f64::from(a - b).powi(2))
            .sum();
        let snr = 10.0 * (signal / noise.max(1e-30)).log10();
        eprintln!("SNR against the Python reference: {snr:.1} dB");
        assert!(snr >= 35.0, "SNR {snr:.1} dB is below 35 dB");
    }
}
