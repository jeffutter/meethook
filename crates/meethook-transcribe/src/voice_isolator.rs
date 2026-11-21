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
//! crossfade, the last window ending at the end of the audio. Audio up to one window runs
//! unpadded: the graph's frame axis is dynamic.
//!
//! No resampling is done here: the caller owns the rate. (`rubato` is already a dependency for
//! that, in [`crate::audio`].)
//!
//! # Engines and speed
//!
//! [`VoiceIsolator::open`] picks the engine this build has: the network on the GPU through MLX
//! (the `mlx` cargo feature, on by default on macOS, `crate::mlx_isolator`) or the ONNX graph on the CPU.
//! Both run the same STFT, windows and crossfades from this module and give the same output (the
//! MLX engine is fp16, see Parity). Measured with `examples/isolate-voice` on a MacBook Pro 18,1
//! (Apple M1 Pro, 32 GB), release build:
//!
//! | engine | `sample_14.wav` (10.77 s, two windows) |
//! |---|---|
//! | MLX (`VoiceIsolator::open_mlx`) | **about 1.3 s per 8 s window warm; 129 s of audio in 31 s (4.2x realtime)**; loads in 0.04 s (lazily mapped), the first call takes about 3 s more while MLX compiles its Metal kernels; peak footprint 2 GB |
//! | ONNX on the CPU ([`VoiceIsolator::open_onnx`]) | 31-33 s (0.33x realtime) on the 8 performance cores; 40 s on all 10 cores, where ONNX Runtime's default thread count includes the 2 efficiency cores the others then wait on; loads in 20 s; about 6 GB per window |
//! | ONNX through CoreML ([`VoiceIsolator::open_coreml`]) | **unusable in every configuration tried**, below |
//! | Python package on PyTorch / Apple Metal (an earlier trial) | about 0.41x realtime |
//!
//! Through the whole mic-isolation stage on the first 30 minutes of a real recording (117
//! speech regions, 433 s of audio sent through the isolator in 41 batches): **106 s on MLX (17x
//! realtime, 2.3 GB peak) against 1,191 s on the ONNX CPU engine (1.5x realtime, 12.3 GB peak)**,
//! about 11x.
//!
//! **CoreML via ONNX Runtime.** The graph has 15,568 nodes, almost all of them the Mamba scans
//! exported as a blocked scan for WebGPU. With ONNX Runtime's defaults CoreML takes 6,598 of
//! them in 189 partitions, accepts the load and dies in the first inference. `MLProgram` with
//! static input shapes fails to compile (`model.mil`, error -14). Pinning the `frames` axis to
//! 801 (a free dimension override) and using `MLProgram` lets CoreML take 14,360 of 14,396 nodes
//! in 13 partitions, but with every compute-unit setting (CPU only, CPU and GPU, CPU and Neural
//! Engine) the footprint passed 16 GB within seconds of the first inference and the run was
//! killed (the load alone took 154 s on the Neural Engine setting). So the partitioning is not
//! the problem; CoreML's execution of this graph is. The MLX engine runs the same network with
//! hand-written kernels for the scan and the attention head layout, which is why it fits in
//! 2.3 GB.
//!
//! **Caching.** MLX keeps freed buffers for reuse, keyed by size, and every span of the mic
//! stage is a different length, so `crate::mlx_isolator` clears that cache after each window.
//! Without it a 30 minute track grew past 19 GB by 40%.
//!
//! # Parity
//!
//! `matches_the_official_python_package` (ignored: it needs the 260 MB download) compares the
//! output on `testdata/voice-isolator/sample_14.wav` with the official `inworld-voice-isolation`
//! 1.0.0 package's, produced by `scripts/voice_isolator_reference.py`. Measured SNR: 125.6 dB
//! (required: 90 dB; before the last window was aligned to the end of the audio instead of
//! zero-padded it was 46.7 dB). The MLX engine (fp16) measures 75.3 dB, required 70 dB, the same
//! as the Swift package it is ported from (75.4 dB). The sample is Apache-2.0 from
//! `inworld-ai/voice-isolator`; no meeting audio is used anywhere.

use std::path::Path;
use std::sync::Arc;

use meethook_models::ModelSpec;
use ort::session::Session;
use ort::value::TensorRef;
use realfft::num_complex::Complex;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

#[cfg(all(feature = "mlx", target_os = "macos"))]
use crate::VOICE_ISOLATOR_MLX_MODEL;
#[cfg(all(feature = "mlx", target_os = "macos"))]
use crate::mlx_isolator::MlxNetwork;
use crate::onnx::{open_cpu_session_with_threads, open_session, performance_cores};
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

/// What turns a window's spectrum into the complex correction: the ONNX graph on the CPU or,
/// with the `mlx` feature on macOS, the same network on the GPU.
enum Engine {
    Onnx(Session),
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    Mlx(Box<MlxNetwork>),
}

/// A loaded voice-isolator network and the FFT plans around it.
pub struct VoiceIsolator {
    engine: Engine,
    /// True when the network runs on an accelerator: the MLX engine, or CoreML through
    /// [`VoiceIsolator::open_coreml`] when it took the graph.
    pub accelerated: bool,
    stft: Stft,
}

impl VoiceIsolator {
    /// The model files [`VoiceIsolator::open`] needs in the models directory, for the caller to
    /// install first: the MLX weights when this build has the MLX engine, else the ONNX graph.
    pub fn required_model() -> &'static ModelSpec {
        #[cfg(all(feature = "mlx", target_os = "macos"))]
        {
            &VOICE_ISOLATOR_MLX_MODEL
        }
        #[cfg(not(all(feature = "mlx", target_os = "macos")))]
        {
            &VOICE_ISOLATOR_MODEL
        }
    }

    /// Loads the best engine this build has from `models_dir`, the directory the registry
    /// installs into: MLX on the GPU with the `mlx` feature on macOS, otherwise the ONNX graph
    /// on the CPU. [`VoiceIsolator::required_model`] says which file must be installed.
    ///
    /// The ONNX path is CPU rather than the CoreML-then-CPU policy of [`crate::open_session`],
    /// because on the measured machine CoreML accepts this graph at load and then exhausts
    /// memory on the first inference (see the module docs). A default that can take a machine
    /// down is not a default.
    pub fn open(models_dir: &Path) -> Result<Self> {
        #[cfg(all(feature = "mlx", target_os = "macos"))]
        {
            Self::open_mlx(models_dir)
        }
        #[cfg(not(all(feature = "mlx", target_os = "macos")))]
        {
            Self::open_onnx(models_dir)
        }
    }

    /// The ONNX graph on the CPU, whatever else this build has. Uses the performance cores only
    /// where the machine has two kinds.
    pub fn open_onnx(models_dir: &Path) -> Result<Self> {
        let path = installed(models_dir, &VOICE_ISOLATOR_MODEL)?;
        Ok(Self {
            engine: Engine::Onnx(open_cpu_session_with_threads(&path, performance_cores())?),
            accelerated: false,
            stft: Stft::new(),
        })
    }

    /// The MLX engine on the GPU: about 11x the ONNX CPU path through the whole mic stage on the
    /// measured machine (see the module docs).
    #[cfg(all(feature = "mlx", target_os = "macos"))]
    pub fn open_mlx(models_dir: &Path) -> Result<Self> {
        let path = installed(models_dir, &VOICE_ISOLATOR_MLX_MODEL)?;
        Ok(Self {
            engine: Engine::Mlx(Box::new(MlxNetwork::load(&path)?)),
            accelerated: true,
            stft: Stft::new(),
        })
    }

    /// Loads the ONNX graph through [`crate::open_session`] (CoreML, then CPU). Kept for
    /// measuring the accelerator; see [`VoiceIsolator::open`] for why it is not the default.
    /// Running it can exhaust memory.
    pub fn open_coreml(models_dir: &Path) -> Result<Self> {
        let path = installed(models_dir, &VOICE_ISOLATOR_MODEL)?;
        let loaded = open_session(&path)?;
        Ok(Self {
            engine: Engine::Onnx(loaded.session),
            accelerated: loaded.accelerated,
            stft: Stft::new(),
        })
    }

    /// The isolated voice: same length and rate as `audio`, which must be 48 kHz mono.
    ///
    /// Empty input returns empty without touching the graph. Audio up to one window runs as it
    /// is (the graph's frame axis is dynamic). Longer audio runs as `WINDOW`-sample windows
    /// every `STRIDE` samples with the last one ending at the end of the audio, blended by
    /// their weights; this is the official package's plan, so the tail is never zero-padded.
    pub fn isolate(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        if audio.len() <= WINDOW {
            return self.run_window(audio);
        }
        let mut total = vec![0.0f32; audio.len()];
        let mut weight = vec![0.0f32; audio.len()];
        for (start, w) in plan_windows(audio.len()) {
            let voice = self.run_window(&audio[start..start + WINDOW])?;
            for j in 0..WINDOW {
                total[start + j] += voice[j] * w[j];
                weight[start + j] += w[j];
            }
        }
        for (t, w) in total.iter_mut().zip(&weight) {
            *t /= w.max(1e-6);
        }
        Ok(total)
    }

    /// One window of any length through the graph: `audio + istft(correction)`. Audio too short
    /// for the STFT's reflect padding is zero-padded to the shortest length it accepts and
    /// cropped back.
    fn run_window(&mut self, window: &[f32]) -> Result<Vec<f32>> {
        if window.is_empty() {
            return Ok(Vec::new());
        }
        if window.len() <= PAD {
            let mut padded = window.to_vec();
            padded.resize(PAD + 1, 0.0);
            let mut voice = self.run_window(&padded)?;
            voice.truncate(window.len());
            return Ok(voice);
        }
        let spec = self.stft.forward(window);
        let correction = match &mut self.engine {
            Engine::Onnx(session) => onnx_correction(session, &spec)?,
            #[cfg(all(feature = "mlx", target_os = "macos"))]
            Engine::Mlx(network) => network.correction(&spec)?,
        };
        let mut voice = self.stft.inverse(&correction, window.len());
        for (v, a) in voice.iter_mut().zip(window) {
            *v += a;
        }
        Ok(voice)
    }
}

fn installed(models_dir: &Path, spec: &ModelSpec) -> Result<std::path::PathBuf> {
    let path = models_dir.join(spec.file_name);
    if !path.is_file() {
        return Err(Error::VoiceIsolator(format!(
            "{} is not installed; `meethook transcribe` downloads it, or run \
             `cargo run --example isolate-voice` on a wav",
            path.display()
        )));
    }
    Ok(path)
}

/// The ONNX graph's correction of `spec`, flat `[frames][bin][re, im]`.
fn onnx_correction(session: &mut Session, spec: &[f32]) -> Result<Vec<f32>> {
    let frames = spec.len() / (BINS * 2);
    let input =
        TensorRef::from_array_view(([1usize, frames, BINS, 2], spec)).map_err(inference_failed)?;
    let outputs = session
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
    Ok(correction.to_vec())
}

fn inference_failed(source: ort::Error) -> Error {
    Error::VoiceIsolator(source.to_string())
}

/// `torch.linspace(start, end, steps)` in float32 as PyTorch's CPU kernel evaluates it: the first
/// half counts up from `start`, the second half counts back from `end`, so both ends are exact.
fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![start];
    }
    let step = (end - start) / (steps - 1) as f32;
    (0..steps)
        .map(|i| {
            if i < steps / 2 {
                start + step * i as f32
            } else {
                end - step * (steps - i - 1) as f32
            }
        })
        .collect()
}

/// The windows covering `len > WINDOW` samples, each as its start and a [`WINDOW`]-long weight.
/// Starts are every [`STRIDE`] samples while that leaves more than a window, then one final
/// window ending at `len`. A window's weight is 1 except at the edges it shares: the first
/// `fade_in` samples (where the previous window is still at full weight) are 0, then a
/// [`OVERLAP`] ramp up, and a ramp down over the last [`OVERLAP`] unless it is the last window.
fn plan_windows(len: usize) -> Vec<(usize, Vec<f32>)> {
    let mut starts: Vec<usize> = (0..len - WINDOW).step_by(STRIDE).collect();
    starts.push(len - WINDOW);
    let up = linspace(0.0, 1.0, OVERLAP);
    let down = linspace(1.0, 0.0, OVERLAP);
    starts
        .iter()
        .enumerate()
        .map(|(i, &start)| {
            let mut w = vec![1.0f32; WINDOW];
            if i > 0 {
                let fade_in = starts[i - 1] + STRIDE - start;
                w[..fade_in.min(WINDOW)].fill(0.0);
                for (k, r) in up.iter().enumerate() {
                    if fade_in + k < WINDOW {
                        w[fade_in + k] = *r;
                    }
                }
            }
            if start + WINDOW < len {
                w[WINDOW - OVERLAP..].copy_from_slice(&down);
            }
            (start, w)
        })
        .collect()
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
    fn windows_cover_the_input_and_the_last_ends_at_the_end() {
        for len in [
            WINDOW + 1,
            WINDOW + STRIDE,
            WINDOW + STRIDE + 1,
            5 * SAMPLE_RATE as usize * 10,
        ] {
            let plan = plan_windows(len);
            let (last, _) = plan.last().unwrap();
            assert_eq!(last + WINDOW, len, "{len}: the last window ends at the end");
            assert_eq!(plan[0].0, 0);
            // Blending a constant must give the constant back: every sample is covered with
            // positive total weight.
            let mut weight = vec![0.0f32; len];
            for (start, w) in &plan {
                for (j, x) in w.iter().enumerate() {
                    weight[start + j] += x;
                }
            }
            assert!(
                weight.iter().all(|&w| w > 0.5),
                "{len}: a sample is under-covered"
            );
        }
    }

    #[test]
    fn a_window_that_overlaps_more_than_the_stride_fades_in_late() {
        // len = WINDOW + 1: the second window starts at 1, so the first is at full weight until
        // its own ramp, and the second must be silent where the first still is.
        let plan = plan_windows(WINDOW + 1);
        assert_eq!(plan.len(), 2);
        let fade_in = plan[0].0 + STRIDE - plan[1].0;
        assert!(plan[1].1[..fade_in].iter().all(|&w| w == 0.0));
        assert_eq!(plan[1].1[fade_in], 0.0);
        assert_eq!(plan[1].1[fade_in + OVERLAP - 1], 1.0);
        assert!(
            plan[1].1[WINDOW - 1] == 1.0,
            "the last window never fades out"
        );
    }

    #[test]
    fn linspace_hits_both_ends_exactly() {
        let up = linspace(0.0, 1.0, OVERLAP);
        assert_eq!((up[0], up[OVERLAP - 1]), (0.0, 1.0));
        assert!(up.windows(2).all(|p| p[1] >= p[0]));
        assert_eq!(linspace(1.0, 0.0, 1), vec![1.0]);
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

    /// The ONNX engine, downloading its graph first the way the real command does.
    fn open_installed() -> VoiceIsolator {
        let dir = models_dir();
        meethook_models::ensure_model(&dir, &VOICE_ISOLATOR_MODEL, &mut |_, _| {})
            .expect("the voice isolator must download");
        VoiceIsolator::open_onnx(&dir).expect("the voice isolator must load")
    }

    /// Whatever [`VoiceIsolator::open`] picks in this build, downloading its model first.
    fn open_default() -> VoiceIsolator {
        let dir = models_dir();
        meethook_models::ensure_model(&dir, VoiceIsolator::required_model(), &mut |_, _| {})
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

    /// The MLX engine runs fp16: 75.3 dB measured against the reference, the same as the Swift
    /// port it comes from (75.4 dB).
    const MLX_FLOOR_DB: f64 = 70.0;

    fn testdata(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/voice-isolator")
            .join(name)
    }

    #[test]
    #[ignore = "downloads the 260 MB voice isolator"]
    #[allow(clippy::infallible_destructuring_match)] // two engines with the `mlx` feature
    fn the_graph_has_the_documented_contract() {
        let isolator = open_installed();
        let session = match &isolator.engine {
            Engine::Onnx(session) => session,
            #[cfg(all(feature = "mlx", target_os = "macos"))]
            Engine::Mlx(_) => panic!("open_installed opens the ONNX engine"),
        };
        let inputs = session.inputs();
        let outputs = session.outputs();
        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].name(), "spec");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name(), "correction");
        eprintln!("accelerated: {}", isolator.accelerated);
    }

    #[test]
    #[ignore = "downloads the 260 MB voice isolator"]
    fn short_exact_and_empty_input_keep_their_length() {
        let mut isolator = open_default();
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
        let mut isolator = open_default();
        let floor = if isolator.accelerated && cfg!(feature = "mlx") {
            MLX_FLOOR_DB
        } else {
            90.0
        };
        let got = isolator.isolate(&audio).unwrap();
        assert_eq!(got.len(), want.len());
        let signal: f64 = want.iter().map(|&s| f64::from(s).powi(2)).sum();
        let noise: f64 = want
            .iter()
            .zip(&got)
            .map(|(&a, &b)| f64::from(a - b).powi(2))
            .sum();
        let snr = 10.0 * (signal / noise.max(1e-30)).log10();
        eprintln!("SNR against the Python reference: {snr:.1} dB");
        assert!(snr >= floor, "SNR {snr:.1} dB is below {floor} dB");
    }
}
