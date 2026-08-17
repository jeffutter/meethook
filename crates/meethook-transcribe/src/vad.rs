//! Voice activity detection: which stretches of a track hold speech at all.
//!
//! This exists so nothing has to ask the decoder to decode silence. Whisper is a sequence
//! model with a language prior, and a 30 s window holding nothing but room noise is the case
//! where that prior fills the gap with whatever the training data made likely -- "Thank you.",
//! "Thanks for watching." A detector that says "there is no speech here" is what lets a caller
//! decline to decode it.
//!
//! # Why Silero standalone
//!
//! Two other detectors were available and both were rejected.
//!
//! The pyannote segmentation graph is already downloaded, and the union of its non-silence
//! powerset classes would answer the same question. But the invariant stated at the top of
//! [`crate`] -- the microphone track is never handed to the diarization models -- is a claim
//! this crate's tests check against the samples themselves, and routing the mic track through
//! that graph to gate it would make the claim false. Silero keeps it literally true, and keeps
//! gating inside the recogniser's own neighbourhood, which is where "do not decode silence"
//! belongs.
//!
//! whisper.cpp's *internal* VAD (`whisper_full_params.vad`) is not reachable from here at all.
//! `whisper_vad` is called only from `whisper_full` and `whisper_full_parallel`, never from
//! `whisper_full_with_state` -- and `whisper_full_with_state` is what `WhisperState::full`,
//! and therefore [`crate::WhisperEngine`], calls. Setting that flag would compile and do
//! nothing.
//!
//! # The boundary
//!
//! Everything crossing this module's edge is seconds into the audio that was handed in, the
//! same contract [`crate::AsrSegment`] and [`crate::LocalTurn`] use. whisper.cpp reports VAD
//! timestamps in centiseconds and exposes a per-frame probability array; neither of those, and
//! no `whisper_rs` type, is visible outside this file.
//!
//! The detector runs on the CPU. That is whisper.cpp's own default for this graph, and it is
//! also what makes a diagnostic built on this module work in an environment with no reachable
//! Metal device: [`SileroVad::load`] does not call [`crate::gpu::use_gpu`], so unlike
//! [`crate::WhisperEngine::load`] it can never fail with [`crate::NoMetalDevice`] and needs no
//! `MEETHOOK_CPU`.

use std::path::Path;

use whisper_rs::{WhisperVadContext, WhisperVadContextParams, WhisperVadParams};

use crate::asr;
use crate::audio::TARGET_RATE;
use crate::{Error, Result};

/// One stretch of audio the detector believes holds speech.
///
/// Both fields are seconds from the start of the audio handed to
/// [`SileroVad::speech_regions`] -- not session time, which a detector knows nothing about.
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
/// [`crate::MIC_SILENCE_S`]; the conversion to whisper.cpp's milliseconds is internal. The
/// defaults are whisper.cpp's own (`whisper_vad_default_params`), so a caller that sets
/// nothing gets the behaviour upstream considers correct.
///
/// `samples_overlap` is deliberately absent even though whisper.cpp's parameter struct has
/// one: on this path it is never read. It is used only inside `whisper_vad`, the internal
/// filter that splices its own buffer -- which this module does not go through. How much
/// overlap to keep when splicing regions is the splicer's decision, taken at splice time; a
/// knob here that silently did nothing would be worse than no knob.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadTuning {
    /// Probability above which a frame counts as speech.
    ///
    /// This moves *both* edges of a region, not one. whisper.cpp derives the closing
    /// threshold as `threshold - 0.15` (floored at 0.01): speech starts when the probability
    /// rises above `threshold` and ends when it falls below that lower value. One knob, two
    /// edges, deliberate hysteresis -- so lowering it both admits quieter speech and holds
    /// regions open longer.
    pub threshold: f32,

    /// Regions shorter than this are discarded as noise rather than reported.
    pub min_speech_s: f64,

    /// How much silence has to pass before a region is considered ended.
    ///
    /// **There is an effective floor of 0.2 s that this field cannot go under.** After the
    /// threshold walk, whisper.cpp unconditionally merges any two regions separated by less
    /// than 200 ms, whatever this says. Values below that produce nearly identical output; a
    /// sweep that appears not to respond is this, not a bug.
    pub min_silence_s: f64,

    /// Added to both ends of every region, so a region does not clip the edges of speech.
    pub speech_pad_s: f64,

    /// Force a region to end after this long, or `None` for unbounded.
    ///
    /// `Option` rather than whisper.cpp's `FLT_MAX` sentinel, so "unbounded" cannot be
    /// written wrong.
    pub max_speech_s: Option<f64>,
}

impl Default for VadTuning {
    fn default() -> Self {
        VadTuning {
            threshold: 0.5,
            min_speech_s: 0.25,
            min_silence_s: 0.1,
            speech_pad_s: 0.03,
            max_speech_s: None,
        }
    }
}

/// A loaded Silero VAD, reusable across tracks.
///
/// A standalone detector rather than the pyannote segmentation graph already installed, and
/// rather than whisper.cpp's own `whisper_full_params.vad`. The first would hand the microphone
/// track to a diarization model, which the invariant at the top of [`crate`] says never
/// happens; the second is unreachable from the entry point [`crate::WhisperEngine`] uses and
/// would compile while doing nothing. The `vad` module's own documentation carries the long
/// version.
///
/// One context serves every track a run measures: whisper.cpp clears the recurrent state
/// buffer at the start of each detection pass, so a second load would buy nothing.
pub struct SileroVad {
    context: WhisperVadContext,
}

impl SileroVad {
    /// Loads the Silero VAD weights ([`crate::SILERO_VAD_MODEL`]).
    ///
    /// Deliberately CPU-only, stated rather than inherited from
    /// [`WhisperVadContextParams::default`]: this is an 885 KB LSTM evaluated once per 512
    /// samples, which does not earn a second Metal allocation next to the Whisper checkpoint
    /// already on the device. `n_threads` is left at whisper.cpp's own 4 for the same reason
    /// -- the cost of a long track is the number of sequential graph evaluations, not the
    /// width of any one of them.
    pub fn load(model_path: &Path) -> Result<SileroVad> {
        // Before the load, not after: loading and detection both log at INFO, and without the
        // hooks installed first that goes straight to stderr interleaved with the caller's
        // own output.
        asr::install_logging_hooks();

        let mut params = WhisperVadContextParams::default();
        params.set_use_gpu(false);

        // `WhisperVadContext::new` takes `&str`, so a path that is not UTF-8 cannot be
        // handed to it at all. That lands here rather than as a panic further in.
        let path = model_path.to_str().ok_or_else(|| {
            Error::Vad(format!(
                "the model path {} is not valid UTF-8",
                model_path.display()
            ))
        })?;

        let context = WhisperVadContext::new(path, params).map_err(|e| {
            // whisper-rs returns a bare `NullPointer` here and sends the real diagnosis to
            // the logging hooks, so this message has to carry what it can: which file, and
            // the failure that is overwhelmingly the reason.
            Error::Vad(format!(
                "could not load the voice activity detection model at {} ({e}); \
                 the weights may be missing or truncated",
                model_path.display()
            ))
        })?;

        Ok(SileroVad { context })
    }

    /// Reports the stretches of `audio_16k_mono` that hold speech.
    ///
    /// The rate is in the argument name rather than in a parameter because it is not a
    /// choice: `WHISPER_SAMPLE_RATE` is compiled into the detector, and [`TARGET_RATE`] is
    /// already that rate. Same contract as [`crate::SpeechToText::transcribe`].
    ///
    /// An empty track yields no regions rather than an error -- a session where nobody spoke
    /// is a real session -- and so does a track shorter than `min_speech_s`.
    ///
    /// `tuning` is per call rather than stored at load, mirroring whisper.cpp's own split
    /// (context parameters at load, detection parameters per pass). It keeps [`Self::load`]
    /// one argument wide, and it is what lets a caller sweep several thresholds over one
    /// loaded model. The cost is a full detection pass per setting, which is the honest price
    /// of a sweep either way: the probabilities are recomputed for each.
    pub fn speech_regions(
        &mut self,
        audio_16k_mono: &[f32],
        tuning: VadTuning,
    ) -> Result<Vec<SpeechRegion>> {
        // Two lines to make the guarantee above a property of this code rather than of a C
        // function nobody re-reads, and it skips a pointless graph evaluation besides.
        if audio_16k_mono.is_empty() {
            return Ok(Vec::new());
        }

        let mut params = WhisperVadParams::default();
        params.set_threshold(tuning.threshold);
        params.set_min_speech_duration(milliseconds(tuning.min_speech_s));
        params.set_min_silence_duration(milliseconds(tuning.min_silence_s));
        params.set_speech_pad(milliseconds(tuning.speech_pad_s));
        params.set_max_speech_duration(match tuning.max_speech_s {
            Some(seconds) => seconds as f32,
            None => f32::MAX,
        });

        let segments = self
            .context
            .segments_from_samples(params, audio_16k_mono)
            .map_err(|e| Error::Vad(format!("detection failed ({e})")))?;

        // whisper.cpp measures the track as a whole number of 512-sample windows, zero-padding
        // the last one, and rounds timestamps to the nearest centisecond -- so the final
        // region's end can sit up to one window plus 5 ms past the samples handed in. Clamping
        // is not cosmetic: a caller that slices its audio with these numbers would panic on an
        // end past the buffer.
        let duration_s = audio_16k_mono.len() as f64 / f64::from(TARGET_RATE);
        let mut regions = Vec::new();
        for segment in segments {
            let start_s = (f64::from(segment.start) / 100.0).clamp(0.0, duration_s);
            let end_s = (f64::from(segment.end) / 100.0).clamp(0.0, duration_s);
            if end_s > start_s {
                regions.push(SpeechRegion { start_s, end_s });
            }
        }
        Ok(regions)
    }
}

/// Seconds to whisper.cpp's milliseconds, rounded rather than truncated so 0.25 s does not
/// become 249 ms on a float that landed just short.
fn milliseconds(seconds: f64) -> i32 {
    (seconds * 1000.0).round() as i32
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::SILERO_VAD_MODEL;

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
    fn load_if_installed() -> Option<SileroVad> {
        let path = models_dir()?.join(SILERO_VAD_MODEL.file_name);
        if !path.is_file() {
            eprintln!(
                "skipping: {} is not installed; \
                 run `cargo run --release --example vad-regions -- <session-dir>` to fetch it",
                path.display()
            );
            return None;
        }
        Some(SileroVad::load(&path).expect("installed weights must load"))
    }

    /// Not "the detector answers nothing" but "the detector is never reached": the early
    /// return is what makes the empty case a property of this module. Runs with no weights
    /// installed, which is the other half of why it is written this way.
    #[test]
    fn an_empty_track_yields_no_regions_rather_than_an_error() {
        let Some(mut vad) = load_if_installed() else {
            return;
        };
        assert_eq!(vad.speech_regions(&[], VadTuning::default()).unwrap(), []);
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

    /// The assertion that catches an unclamped region end, and the only positive claim these
    /// tests can honestly make.
    ///
    /// Deliberately *not* "this synthetic track is detected as speech". Silero is trained on
    /// speech; a tone burst is not speech, and there is no speech fixture in this repository.
    /// Whether the detector finds real talk is checked by a person against a real session --
    /// which is what `examples/vad-regions.rs` exists for.
    #[test]
    fn whatever_is_found_is_ordered_non_overlapping_and_inside_the_track() {
        let Some(mut vad) = load_if_installed() else {
            return;
        };
        let rate = TARGET_RATE as usize;
        // A deliberately awkward length: not a whole number of 512-sample windows and not a
        // whole number of centiseconds, so the last region's end has to be clamped back.
        let samples = rate * 4 + 137;
        let audio: Vec<f32> = (0..samples)
            .map(|i| {
                let t = i as f32 / TARGET_RATE as f32;
                // Bursts of a formant-ish pair, gated on and off twice a second.
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

    /// Seconds are this module's unit; whisper.cpp's are milliseconds. A truncating conversion
    /// would turn every default into one millisecond less than it claims.
    #[test]
    fn tuning_seconds_convert_to_whole_milliseconds() {
        assert_eq!(milliseconds(0.25), 250);
        assert_eq!(milliseconds(0.1), 100);
        assert_eq!(milliseconds(0.03), 30);
        assert_eq!(milliseconds(0.0), 0);
    }
}
