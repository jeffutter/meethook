//! Voice activity detection: which stretches of a track hold speech at all.
//!
//! This exists so nothing has to ask the decoder to decode silence. Whisper is a sequence
//! model with a language prior, and a 30 s window holding nothing but room noise is the case
//! where that prior fills the gap with whatever the training data made likely -- "Thank you.",
//! "Thanks for watching." A detector that says "there is no speech here" is what lets a caller
//! decline to decode it.
//!
//! # Why FireRedVAD, streaming
//!
//! The detector is FireRedVAD's streaming model (a DFSMN, 2 MB, ONNX), run with the
//! probabilities smoothed causally. It replaced Silero v5.1.2, which this module used to run
//! through whisper.cpp. The comparison that decided it was made on one real session (a
//! 49-minute call, both tracks) against Silero at the same region walk, and the facts that
//! matter are:
//!
//! - It finds nearly a superset of what Silero finds. Silero missed short backchannels ("yeah",
//!   a laugh) that FireRed keeps, and FireRed lacked under a minute of Silero's speech on the
//!   speaker track.
//! - A finite receptive field means its output depends on a bounded window of audio and
//!   nothing before it (its cache is 19 frames in each of 8 layers; the test at the bottom of
//!   this file pins that the reach is under two seconds). SAVA, the other candidate, did not:
//!   after a few seconds of stream its
//!   output for quiet speech dropped out, and dropped out again on every run of the same
//!   audio. A gate that can silently miss minutes of speech is worse than no gate, so that
//!   property is the one worth keeping and is tested for below.
//! - Joining. With the same region walk, FireRed's probabilities cut a continuous speaker's
//!   turn into fewer pieces than Silero's did.
//!
//! The 0.35 threshold in [`VadTuning::default`] was chosen by listening to this corpus, not by
//! a metric: lower admits more of the quiet words at the edges of a turn, and below about 0.3
//! the speaker track starts joining regions across speakers. It is a setting for this tool's
//! recordings (a mic and a call), not FireRed's published operating point.
//!
//! The pyannote segmentation graph was rejected as a detector for the reason it was rejected
//! before: the invariant at the top of [`crate`] -- the microphone track is never handed to the
//! diarization models -- is checked against the samples themselves, and gating the mic with
//! that graph would make it false.
//!
//! # The frontend
//!
//! The network reads 80 log-mel energies per 10 ms frame, computed over a waveform in int16
//! range with Kaldi's defaults (Povey window, 25 ms frames), then normalised with a mean and
//! inverse standard deviation **fixed at training time**. [`crate::fbank::Fbank`] is the
//! extractor ([`Window::Povey`], [`Fbank::compute_unnormalized`]); the fixed statistics are
//! `assets/firered-vad-cmvn.ark`, the file FireRedVAD ships beside the weights, embedded in the
//! binary. They are embedded rather than downloaded because they are 1.3 KB, they are only
//! meaningful for the exact weights pinned in [`crate::FIRERED_VAD_MODEL`], and a model path
//! that cannot find its own normalisation is a failure mode that should not exist. Both
//! files must move together.
//!
//! # The boundary
//!
//! Everything crossing this module's edge is seconds into the audio that was handed in, the
//! same contract [`crate::AsrSegment`] and [`crate::LocalTurn`] use. No `ort` type is visible
//! outside this file.
//!
//! The detector runs on the CPU ([`crate::onnx::open_cpu_session`]), for the reason documented
//! there.

use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;

use crate::audio::TARGET_RATE;
use crate::fbank::{Fbank, MEL_BINS, Window};
use crate::onnx::open_cpu_session;
use crate::progress::Phase;
use crate::{Error, Result};

/// One stretch of audio the detector believes holds speech.
///
/// Both fields are seconds from the start of the audio handed to
/// [`FireRedVad::speech_regions`] -- not session time, which a detector knows nothing about.
/// Regions come back in ascending order, do not overlap, and lie inside `0.0..=duration` of
/// that audio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeechRegion {
    pub start_s: f64,
    pub end_s: f64,
}

impl SpeechRegion {
    /// How long this region lasts, in seconds. Always positive: a region that clamping left
    /// with no duration is dropped rather than returned.
    pub fn duration_s(&self) -> f64 {
        self.end_s - self.start_s
    }
}

/// The knobs that decide where speech starts and stops.
///
/// Seconds throughout, matching [`crate::RUN_BRIDGE_S`], [`crate::SPLICE_GAP_S`] and
/// [`crate::MIC_SILENCE_S`]. The region walk that applies them is whisper.cpp's
/// (`whisper_vad_segments_from_probs`, minus its maximum-region-length handling, which no
/// caller here uses), ported so that moving off Silero changed the detector and nothing
/// downstream of it: the same hysteresis, the same 200 ms merge, the same padding rule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadTuning {
    /// Probability at or above which a frame counts as speech.
    ///
    /// This moves *both* edges of a region, not one. The closing threshold is derived as
    /// `threshold - 0.15` (floored at 0.01): speech starts when the probability reaches
    /// `threshold` and ends when it falls below that lower value. One knob, two edges,
    /// deliberate hysteresis -- so lowering it both admits quieter speech and holds regions
    /// open longer.
    pub threshold: f32,

    /// Regions shorter than this are discarded as noise rather than reported.
    pub min_speech_s: f64,

    /// How much silence has to pass before a region is considered ended.
    ///
    /// **There is an effective floor of 0.2 s that this field cannot go under.** After the
    /// threshold walk, any two regions separated by less than 200 ms are merged, whatever this
    /// says. Values below that produce nearly identical output; a sweep that appears not to
    /// respond is this, not a bug.
    pub min_silence_s: f64,

    /// Added to both ends of every region, so a region does not clip the edges of speech.
    /// Where two regions are closer than twice this, the gap between them is split evenly
    /// instead, so padding never makes regions overlap.
    pub speech_pad_s: f64,
}

impl Default for VadTuning {
    fn default() -> Self {
        VadTuning {
            threshold: 0.35,
            min_speech_s: 0.25,
            min_silence_s: 0.1,
            speech_pad_s: 0.03,
        }
    }
}

/// FireRedVAD's streaming model, reusable across tracks.
///
/// Each call to [`Self::speech_regions`] starts from a zeroed cache, so tracks do not leak into
/// each other and a track's result does not depend on what ran before it.
pub struct FireRedVad {
    session: Session,
    fbank: Fbank,
    /// Per-bin mean and inverse standard deviation, from the embedded CMVN statistics.
    mean: [f32; MEL_BINS],
    inverse_std: [f32; MEL_BINS],
}

/// Frames per graph call. Not a property of the model -- it accepts any length -- but of
/// the cost: one call per 0.5 s of audio keeps the call overhead small without holding more
/// than a few thousand floats of features at a time.
const BLOCK_FRAMES: usize = 50;

/// 25 ms analysis windows every 10 ms at 16 kHz, in samples.
const FRAME_LENGTH: usize = TARGET_RATE as usize * 25 / 1000;
const FRAME_SHIFT: usize = TARGET_RATE as usize / 100;

/// The graph's cache is `[LAYERS, 1, PROJECTION, CACHE_FRAMES]`: the last `CACHE_FRAMES` of
/// each DFSMN layer's memory input. The graph returns more than it takes in, so the carried
/// state is its last `CACHE_FRAMES` columns.
const CACHE_LAYERS: usize = 8;
const CACHE_PROJECTION: usize = 128;
const CACHE_FRAMES: usize = 19;

/// Width of the causal moving average over the speech probability, in frames (50 ms).
///
/// FireRedVAD's own default smoothing, applied causally here because the streaming model is
/// evaluated with no lookahead; a centred window would make a frame's value depend on audio
/// that comes after it.
const SMOOTHING_FRAMES: usize = 5;

/// FireRedVAD's own training statistics, as shipped with the weights: a Kaldi binary double
/// matrix of two rows (sum, sum of squares) by 81 columns (80 bins and the frame count).
const CMVN: &[u8] = include_bytes!("../assets/firered-vad-cmvn.ark");

impl FireRedVad {
    /// Loads the streaming graph ([`crate::FIRERED_VAD_MODEL`]).
    pub fn load(model_path: &Path) -> Result<FireRedVad> {
        let session = open_cpu_session(model_path)?;
        let (mean, inverse_std) = parse_cmvn(CMVN)?;
        Ok(FireRedVad {
            session,
            fbank: Fbank::with_window(Window::Povey),
            mean,
            inverse_std,
        })
    }

    /// The frame-by-frame probability that each 10 ms of `audio_16k_mono` is speech, smoothed.
    ///
    /// Frame `i` is reported at `i * 10 ms`, the convention FireRedVAD's own tooling uses: the
    /// frame's analysis window covers 25 ms starting there, and the network's receptive field
    /// reaches back from it. Only whole frames are produced (`1 + (n - 400) / 160`), so a track
    /// shorter than 25 ms has none.
    pub fn probabilities(&mut self, audio_16k_mono: &[f32]) -> Result<Vec<f32>> {
        let frames = if audio_16k_mono.len() < FRAME_LENGTH {
            0
        } else {
            1 + (audio_16k_mono.len() - FRAME_LENGTH) / FRAME_SHIFT
        };
        let mut probabilities = Vec::with_capacity(frames);
        let mut cache = vec![0.0f32; CACHE_LAYERS * CACHE_PROJECTION * CACHE_FRAMES];
        let mut next_cache = vec![0.0f32; cache.len()];
        let mut features = Vec::new();
        let mut phase = Phase::start("speech gate: detecting");

        for first in (0..frames).step_by(BLOCK_FRAMES) {
            phase.at(first, frames);
            let count = BLOCK_FRAMES.min(frames - first);
            let start = first * FRAME_SHIFT;
            let end = start + FRAME_LENGTH + (count - 1) * FRAME_SHIFT;

            features.clear();
            features.extend(self.fbank.compute_unnormalized(&audio_16k_mono[start..end]));
            debug_assert_eq!(features.len(), count * MEL_BINS);
            for row in features.as_chunks_mut::<MEL_BINS>().0 {
                for ((v, m), s) in row.iter_mut().zip(&self.mean).zip(&self.inverse_std) {
                    *v = (*v - m) * s;
                }
            }

            let feat = TensorRef::from_array_view(([1usize, count, MEL_BINS], &features[..]))
                .map_err(inference_failed)?;
            let caches = TensorRef::from_array_view((
                [CACHE_LAYERS, 1, CACHE_PROJECTION, CACHE_FRAMES],
                &cache[..],
            ))
            .map_err(inference_failed)?;
            let outputs = self
                .session
                .run(ort::inputs!["feat" => feat, "caches_in" => caches])
                .map_err(inference_failed)?;

            let (shape, probs) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(inference_failed)?;
            if probs.len() != count {
                return Err(Error::Vad(format!(
                    "the model returned {} probabilities (shape {shape:?}) for {count} frames",
                    probs.len()
                )));
            }
            probabilities.extend_from_slice(probs);

            let (shape, returned) = outputs[1]
                .try_extract_tensor::<f32>()
                .map_err(inference_failed)?;
            let &[layers, 1, projection, width] = &shape[..] else {
                return Err(Error::Vad(format!(
                    "the model's cache has shape {shape:?}, not [layers, 1, projection, frames]"
                )));
            };
            let (layers, projection, width) =
                (layers as usize, projection as usize, width as usize);
            if layers != CACHE_LAYERS || projection != CACHE_PROJECTION || width < CACHE_FRAMES {
                return Err(Error::Vad(format!(
                    "the model's cache has shape {shape:?}, expected \
                     [{CACHE_LAYERS}, 1, {CACHE_PROJECTION}, >= {CACHE_FRAMES}]"
                )));
            }
            for (row, out) in returned
                .chunks_exact(width)
                .zip(next_cache.as_chunks_mut::<CACHE_FRAMES>().0)
            {
                out.copy_from_slice(&row[width - CACHE_FRAMES..]);
            }
            std::mem::swap(&mut cache, &mut next_cache);
        }
        phase.done();

        Ok(smooth_causal(&probabilities, SMOOTHING_FRAMES))
    }

    /// Reports the stretches of `audio_16k_mono` that hold speech.
    ///
    /// The rate is in the argument name rather than in a parameter because it is not a choice:
    /// [`TARGET_RATE`] is what the frontend is written for. Same contract as
    /// [`crate::SpeechToText::transcribe`].
    ///
    /// An empty track yields no regions rather than an error -- a session where nobody spoke is
    /// a real session -- and so does a track shorter than `min_speech_s`.
    ///
    /// `tuning` is per call so a caller can sweep several settings over one loaded model. The
    /// cost is a full pass per setting; use [`Self::probabilities`] and [`regions_from`] to
    /// sweep without it.
    pub fn speech_regions(
        &mut self,
        audio_16k_mono: &[f32],
        tuning: VadTuning,
    ) -> Result<Vec<SpeechRegion>> {
        let probabilities = self.probabilities(audio_16k_mono)?;
        Ok(regions_from(&probabilities, audio_16k_mono.len(), tuning))
    }
}

fn inference_failed(source: ort::Error) -> Error {
    Error::Vad(format!("detection failed ({source})"))
}

/// Parses a Kaldi binary double matrix of CMVN statistics into the mean and inverse standard
/// deviation the network was trained against.
///
/// Row 0 holds per-bin sums with the frame count in the last column; row 1 holds sums of
/// squares. The variance floor is FireRedVAD's own (`1e-20`).
fn parse_cmvn(bytes: &[u8]) -> Result<([f32; MEL_BINS], [f32; MEL_BINS])> {
    let bad = |why: &str| Error::Vad(format!("the embedded CMVN statistics are unusable: {why}"));

    // `\0BDM ` then, for each dimension, a size byte (4) and a little-endian i32.
    let header = bytes.get(..15).ok_or_else(|| bad("truncated"))?;
    if &header[..5] != b"\0BDM " || header[5] != 4 || header[10] != 4 {
        return Err(bad("not a Kaldi binary double matrix"));
    }
    let rows = i32::from_le_bytes(header[6..10].try_into().unwrap());
    let cols = i32::from_le_bytes(header[11..15].try_into().unwrap());
    if rows != 2 || cols as usize != MEL_BINS + 1 {
        return Err(bad(&format!("shape {rows} x {cols}, expected 2 x 81")));
    }
    let body = &bytes[15..];
    if body.len() != 2 * (MEL_BINS + 1) * 8 {
        return Err(bad("wrong length"));
    }
    let value = |row: usize, col: usize| {
        let at = (row * (MEL_BINS + 1) + col) * 8;
        f64::from_le_bytes(body[at..at + 8].try_into().unwrap())
    };

    let count = value(0, MEL_BINS);
    if count <= 0.0 {
        return Err(bad("frame count is not positive"));
    }
    let mut mean = [0.0f32; MEL_BINS];
    let mut inverse_std = [0.0f32; MEL_BINS];
    for bin in 0..MEL_BINS {
        let m = value(0, bin) / count;
        let variance = (value(1, bin) / count - m * m).max(1e-20);
        mean[bin] = m as f32;
        inverse_std[bin] = (1.0 / variance.sqrt()) as f32;
    }
    Ok((mean, inverse_std))
}

/// A trailing moving average: each value is the mean of itself and up to `window - 1` values
/// before it. Fewer than `window` values exist at the start and the mean is over those.
fn smooth_causal(values: &[f32], window: usize) -> Vec<f32> {
    let mut sum = 0.0f64;
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            sum += f64::from(v);
            if i >= window {
                sum -= f64::from(values[i - window]);
            }
            (sum / (i + 1).min(window) as f64) as f32
        })
        .collect()
}

/// Turns a probability per 10 ms frame into speech regions.
///
/// whisper.cpp's `whisper_vad_segments_from_probs` (1.8.3), without its maximum-region-length
/// branch, over 160-sample frames and in integer samples so the arithmetic is the original's.
/// `audio_samples` is the length of the audio the probabilities came from.
///
/// The walk, in the original's order of events: a probability at or above `threshold` opens a
/// region and cancels any pending end; a probability below `threshold - 0.15` starts a pending
/// end at that frame; the region closes once the pending end is `min_silence_s` old. Regions
/// shorter than `min_speech_s` go, regions closer than 200 ms merge, and padding is applied
/// last -- split down the middle where two regions are closer than twice the pad.
pub fn regions_from(
    probabilities: &[f32],
    audio_samples: usize,
    tuning: VadTuning,
) -> Vec<SpeechRegion> {
    const HOP: i64 = FRAME_SHIFT as i64;
    let rate = i64::from(TARGET_RATE);
    let samples = |seconds: f64| (seconds * rate as f64).round() as i64;

    let min_silence = samples(tuning.min_silence_s);
    let min_speech = samples(tuning.min_speech_s);
    let pad = samples(tuning.speech_pad_s);
    let merge_gap = rate * 200 / 1000;
    let on = tuning.threshold;
    let off = (tuning.threshold - 0.15).max(0.01);
    // Like whisper.cpp, the track is a whole number of frames long as far as the walk is
    // concerned; the final clamp to the real length is at the bottom.
    let length = probabilities.len() as i64 * HOP;

    let mut speeches: Vec<(i64, i64)> = Vec::new();
    let mut in_speech = false;
    let mut start = 0i64;
    // Zero means "no pending end", exactly as in the original: a pending end at sample 0 is
    // indistinguishable from none, and cannot matter, because a region cannot be open yet.
    let mut pending_end = 0i64;

    for (i, &p) in probabilities.iter().enumerate() {
        let at = i as i64 * HOP;
        if p >= on {
            pending_end = 0;
            if !in_speech {
                in_speech = true;
                start = at;
            }
            continue;
        }
        if in_speech && p < off {
            if pending_end == 0 {
                pending_end = at;
            }
            if at - pending_end >= min_silence {
                if pending_end - start > min_speech {
                    speeches.push((start, pending_end));
                }
                pending_end = 0;
                in_speech = false;
            }
        }
    }
    if in_speech && length - start > min_speech {
        speeches.push((start, length));
    }

    let mut merged: Vec<(i64, i64)> = Vec::with_capacity(speeches.len());
    for (s, e) in speeches {
        match merged.last_mut() {
            Some(last) if s - last.1 < merge_gap => last.1 = e,
            _ => merged.push((s, e)),
        }
    }
    merged.retain(|&(s, e)| e - s >= min_speech);

    // Padding. The first region's start and the last region's end have nothing to share a gap
    // with; every other boundary is a gap between two neighbours.
    for i in 0..merged.len() {
        if i == 0 {
            merged[i].0 = (merged[i].0 - pad).max(0);
        }
        if i + 1 < merged.len() {
            let gap = merged[i + 1].0 - merged[i].1;
            if gap < 2 * pad {
                merged[i].1 += gap / 2;
                merged[i + 1].0 = (merged[i + 1].0 - gap / 2).max(0);
            } else {
                merged[i].1 = (merged[i].1 + pad).min(length);
                merged[i + 1].0 = (merged[i + 1].0 - pad).max(0);
            }
        } else {
            merged[i].1 = (merged[i].1 + pad).min(length);
        }
    }

    let duration_s = audio_samples as f64 / f64::from(TARGET_RATE);
    merged
        .into_iter()
        .filter_map(|(s, e)| {
            let start_s = (s as f64 / f64::from(TARGET_RATE)).clamp(0.0, duration_s);
            let end_s = (e as f64 / f64::from(TARGET_RATE)).clamp(0.0, duration_s);
            (end_s > start_s).then_some(SpeechRegion { start_s, end_s })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::FIRERED_VAD_MODEL;

    /// Where `meethook transcribe` would have put the weights, resolved the way the CLI
    /// resolves it so a developer who has run the tool once already has the file.
    fn models_dir() -> Option<PathBuf> {
        let root = match std::env::var_os("MEETHOOK_ROOT") {
            Some(root) => PathBuf::from(root),
            None => std::env::home_dir()?.join("meethook"),
        };
        Some(root.join("models"))
    }

    /// Loads the detector, or `None` if the weights are not installed.
    ///
    /// Skipping rather than downloading, matching the `onnx` module's graph-contract tests: a
    /// `cargo test` that silently reaches for a model fails on a plane, in CI, and on a
    /// machine that has never run the tool, none of which is the failure these tests exist to
    /// catch.
    fn load_if_installed() -> Option<FireRedVad> {
        let path = models_dir()?.join(FIRERED_VAD_MODEL.file_name);
        if !path.is_file() {
            eprintln!(
                "skipping: {} is not installed; \
                 run `cargo run --release --example vad-regions -- <session-dir>` to fetch it",
                path.display()
            );
            return None;
        }
        Some(FireRedVad::load(&path).expect("installed weights must load"))
    }

    /// Frames of probability `p` for `seconds`.
    fn frames(p: f32, seconds: f64) -> Vec<f32> {
        vec![p; (seconds * 100.0).round() as usize]
    }

    fn walk(probabilities: &[f32], tuning: VadTuning) -> Vec<(f64, f64)> {
        let samples = probabilities.len() * FRAME_SHIFT + (FRAME_LENGTH - FRAME_SHIFT);
        regions_from(probabilities, samples, tuning)
            .iter()
            .map(|r| (r.start_s, r.end_s))
            .collect()
    }

    fn near(a: f64, b: f64) -> bool {
        (a - b).abs() < 0.011
    }

    #[test]
    fn the_embedded_statistics_parse_to_sane_values() {
        let (mean, inverse_std) = parse_cmvn(CMVN).unwrap();
        // Log-mel of int16-range audio: means are large and positive, deviations a few units.
        assert!(
            mean.iter().all(|m| m.is_finite() && *m > 5.0 && *m < 25.0),
            "{mean:?}"
        );
        assert!(
            inverse_std
                .iter()
                .all(|s| s.is_finite() && *s > 0.05 && *s < 5.0),
            "{inverse_std:?}"
        );
    }

    #[test]
    fn a_corrupted_statistics_file_is_refused() {
        assert!(parse_cmvn(&CMVN[..100]).is_err());
        let mut wrong_magic = CMVN.to_vec();
        wrong_magic[1] = b'X';
        assert!(parse_cmvn(&wrong_magic).is_err());
    }

    #[test]
    fn smoothing_is_a_trailing_average_over_what_exists() {
        let got = smooth_causal(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5);
        let want = [1.0, 0.5, 1.0 / 3.0, 0.25, 0.2, 0.0, 0.0];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?}");
        }
        assert!(smooth_causal(&[], 5).is_empty());
    }

    #[test]
    fn no_speech_is_no_regions() {
        assert_eq!(walk(&frames(0.0, 5.0), VadTuning::default()), []);
        assert_eq!(regions_from(&[], 0, VadTuning::default()), []);
    }

    #[test]
    fn a_stretch_of_speech_becomes_one_padded_region() {
        let mut p = frames(0.0, 1.0);
        p.extend(frames(0.9, 2.0));
        p.extend(frames(0.0, 1.0));
        let got = walk(&p, VadTuning::default());
        assert_eq!(got.len(), 1, "{got:?}");
        // Opened at 1.0 s and closed at the first sub-threshold frame (3.0 s), padded 30 ms.
        assert!(near(got[0].0, 0.97) && near(got[0].1, 3.03), "{got:?}");
    }

    /// The hysteresis, the one thing the threshold means: a probability between the closing and
    /// opening values keeps a region open but cannot start one.
    #[test]
    fn the_middle_band_holds_a_region_open_but_cannot_open_one() {
        let tuning = VadTuning::default(); // opens at 0.35, closes below 0.20
        let mut held = frames(0.9, 1.0);
        held.extend(frames(0.25, 1.0));
        held.extend(frames(0.9, 1.0));
        held.extend(frames(0.0, 1.0));
        assert_eq!(walk(&held, tuning).len(), 1, "0.25 holds the region");

        let mut never = frames(0.25, 3.0);
        never.extend(frames(0.0, 1.0));
        assert_eq!(walk(&never, tuning), [], "0.25 never opens one");
    }

    /// Two stretches of speech with a 0.4 s pause are two regions at 0.1 s min silence and one
    /// at 0.5 s: the knob the gate turns to leave a speaker's turn in one piece.
    #[test]
    fn a_longer_minimum_silence_joins_a_pause() {
        let mut p = frames(0.9, 1.0);
        p.extend(frames(0.0, 0.4));
        p.extend(frames(0.9, 1.0));
        p.extend(frames(0.0, 1.0));
        let short = VadTuning::default();
        let long = VadTuning {
            min_silence_s: 0.5,
            ..short
        };
        assert_eq!(walk(&p, short).len(), 2);
        assert_eq!(walk(&p, long).len(), 1);
    }

    /// whisper.cpp's merge floor, documented on [`VadTuning::min_silence_s`]: a pause under
    /// 200 ms joins whatever the minimum silence says.
    #[test]
    fn a_pause_under_200_ms_always_merges() {
        let mut p = frames(0.9, 1.0);
        p.extend(frames(0.0, 0.15));
        p.extend(frames(0.9, 1.0));
        p.extend(frames(0.0, 1.0));
        assert_eq!(walk(&p, VadTuning::default()).len(), 1);
    }

    #[test]
    fn a_blip_shorter_than_the_minimum_speech_is_dropped() {
        let mut p = frames(0.0, 1.0);
        p.extend(frames(0.9, 0.1));
        p.extend(frames(0.0, 1.0));
        assert_eq!(walk(&p, VadTuning::default()), []);
    }

    /// Padding never makes two regions overlap: a gap narrower than twice the pad is split down
    /// the middle instead.
    #[test]
    fn padding_splits_a_narrow_gap_rather_than_overlapping() {
        let mut p = frames(0.9, 1.0);
        p.extend(frames(0.0, 0.3));
        p.extend(frames(0.9, 1.0));
        p.extend(frames(0.0, 1.0));
        let tuning = VadTuning {
            speech_pad_s: 0.2,
            ..VadTuning::default()
        };
        let got = walk(&p, tuning);
        assert_eq!(got.len(), 2, "{got:?}");
        assert!(got[0].1 <= got[1].0, "{got:?}");
        // The 0.3 s gap (1.0 to 1.3) is split down the middle.
        assert!(near(got[0].1, 1.15) && near(got[1].0, 1.15), "{got:?}");
    }

    #[test]
    fn the_last_region_is_clamped_to_the_audio() {
        let p = frames(0.9, 2.0);
        let samples = p.len() * FRAME_SHIFT; // a little shorter than the walk's own length
        let regions = regions_from(&p, samples - 100, VadTuning::default());
        let duration_s = (samples - 100) as f64 / f64::from(TARGET_RATE);
        assert_eq!(regions.len(), 1);
        assert!(regions[0].end_s <= duration_s, "{regions:?}");
    }

    /// Not "the detector answers nothing" but "the detector is never reached": an empty track
    /// has no frames, so no graph call happens. Runs with weights installed only because
    /// loading needs them.
    #[test]
    fn an_empty_track_yields_no_regions_rather_than_an_error() {
        let Some(mut vad) = load_if_installed() else {
            return;
        };
        assert_eq!(vad.speech_regions(&[], VadTuning::default()).unwrap(), []);
        // And a track under one analysis frame, which has no features.
        assert_eq!(
            vad.speech_regions(&[0.1; 100], VadTuning::default())
                .unwrap(),
            []
        );
    }

    #[test]
    fn digital_silence_holds_no_speech() {
        let Some(mut vad) = load_if_installed() else {
            return;
        };
        let silence = vec![0.0f32; TARGET_RATE as usize * 5];
        assert_eq!(
            vad.speech_regions(&silence, VadTuning::default()).unwrap(),
            []
        );
    }

    /// The property this detector was chosen for, and the one SAVA failed: what it says about a
    /// stretch of audio does not depend on what came before it. Run the same synthetic burst
    /// alone and after a long lead-in of silence, and the probabilities must agree once the
    /// receptive field has passed the seam; the 2 s used below is what was measured to be enough.
    ///
    /// Deliberately not "this burst is detected as speech": a tone is not speech and there is no
    /// speech fixture in this repository. Whether the detector finds real talk is checked by a
    /// person against a real session, with `examples/vad-regions.rs`.
    #[test]
    fn what_it_says_about_audio_does_not_depend_on_what_preceded_it() {
        let Some(mut vad) = load_if_installed() else {
            return;
        };
        let rate = TARGET_RATE as usize;
        let burst: Vec<f32> = (0..rate * 6)
            .map(|i| {
                let t = i as f32 / TARGET_RATE as f32;
                (t * 240.0 * std::f32::consts::TAU).sin() * 0.3
                    + (t * 1_700.0 * std::f32::consts::TAU).sin() * 0.1
            })
            .collect();
        let alone = vad.probabilities(&burst).unwrap();

        let lead = 12 * rate;
        let mut with_lead_in = vec![0.0f32; lead];
        with_lead_in.extend_from_slice(&burst);
        let after = vad.probabilities(&with_lead_in).unwrap();

        // The lead-in is a whole number of frames, so frame `i` of the burst is frame
        // `lead / 160 + i` of the longer track. Skip the first two seconds of each: the
        // receptive field and the smoothing window both reach back across the seam there.
        let offset = lead / FRAME_SHIFT;
        let worst = alone
            .iter()
            .enumerate()
            .skip(200)
            .map(|(i, p)| (p - after[offset + i]).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.002, "lead-in changed a probability by {worst}");
    }

    /// Every region the detector reports is ordered, disjoint and inside the track, for a
    /// length that is not a whole number of frames.
    #[test]
    fn whatever_is_found_is_ordered_non_overlapping_and_inside_the_track() {
        let Some(mut vad) = load_if_installed() else {
            return;
        };
        let rate = TARGET_RATE as usize;
        let samples = rate * 4 + 137;
        let audio: Vec<f32> = (0..samples)
            .map(|i| {
                let t = i as f32 / TARGET_RATE as f32;
                let voiced = (t * 240.0 * std::f32::consts::TAU).sin() * 0.4
                    + (t * 1_700.0 * std::f32::consts::TAU).sin() * 0.2;
                if ((t * 2.0) as u32).is_multiple_of(2) {
                    voiced
                } else {
                    0.0
                }
            })
            .collect();
        let duration_s = samples as f64 / f64::from(TARGET_RATE);

        let regions = vad.speech_regions(&audio, VadTuning::default()).unwrap();

        let mut previous_end = 0.0;
        for region in &regions {
            assert!(region.duration_s() > 0.0, "{region:?}");
            assert!(region.start_s >= previous_end, "{regions:?}");
            assert!(region.end_s <= duration_s, "{region:?} past {duration_s} s");
            previous_end = region.end_s;
        }
    }
}
