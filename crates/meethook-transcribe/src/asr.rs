//! Speech recognition: the [`SpeechToText`] seam, and the whisper.cpp engine behind it.

use std::path::Path;
use std::sync::Once;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::TARGET_RATE;
use crate::gate;
use crate::gpu;
use crate::vad::{SileroVad, VadTuning};
use crate::{Error, Result};

/// One stretch of recognized speech, timed from the start of the audio that was handed to
/// the engine.
///
/// Deliberately not a `Turn`: turns carry a speaker and a session timeline, neither of
/// which an ASR engine knows anything about.
#[derive(Debug, Clone, PartialEq)]
pub struct AsrSegment {
    pub start_s: f64,
    pub end_s: f64,
    pub text: String,
}

/// The one thing transcription needs from a recognizer.
///
/// One method wide on purpose. It exists so batch behaviour -- skipping, `--force`, orphan
/// handling -- can be tested without a 1.6 GB model download, and so a future engine can be
/// swapped in without touching the orchestration; it is not a plugin framework.
pub trait SpeechToText {
    /// Recognizes `audio_16k_mono`, which must be 16 kHz mono `f32`.
    fn transcribe(&mut self, audio_16k_mono: &[f32]) -> Result<Vec<AsrSegment>>;
}

/// Fixed to English in v1.
///
/// Whisper's auto-detection decides from the opening seconds, which in a meeting recording
/// are often silence or keyboard noise; a whole meeting transcribed as the wrong language is
/// a much worse failure than not supporting a language this user does not speak. A flag can
/// be added the day someone needs one.
const LANGUAGE: &str = "en";

/// whisper.cpp refuses input shorter than one second, so a very short (or silent) track is
/// padded rather than rejected.
const MIN_SAMPLES: usize = TARGET_RATE as usize;

/// Beyond this, decoding starts landing on efficiency cores, where it costs more than it
/// buys.
const MAX_THREADS: usize = 8;

/// whisper.cpp emits this instead of text when a window contains no speech.
///
/// The only string this module compares, and it is whisper.cpp's own marker rather than a
/// hallucination. Nothing here filters by text: a blocklist would delete "Thank you." when a
/// user really says it and would still leave every other silence hallucination in place, which
/// is why the gate below removes the silence instead of the phrase.
const BLANK_AUDIO: &str = "[BLANK_AUDIO]";

/// The detector settings the speech gate runs at.
///
/// `min_silence_s` is raised from whisper.cpp's 0.1 s default. Measured with
/// `examples/vad-regions` on `20260810-093047`'s `mic.cleaned.wav`: 0.1 s gives 463 regions
/// holding 713.4 s of speech, 0.5 s gives 315 regions holding 791.6 s. That is 148 fewer seams
/// for 78 s more audio -- roughly three more 30 s decoder windows -- and a seam is the only
/// place rule 4 of [`gate::Splice::to_original`] can fire, so fewer of them is the axis worth
/// buying on.
///
/// `threshold` stays at whisper.cpp's 0.5. A sweep over 0.30..0.95 moved the detected speech on
/// that track from 792 s to 454 s, and the criterion that must not break is the user's quietest
/// real speech surviving, so there is nothing in that range worth taking.
fn gate_tuning() -> VadTuning {
    VadTuning {
        min_silence_s: 0.5,
        ..VadTuning::default()
    }
}

/// Routes ggml's and whisper.cpp's own log output through Rust, once per process.
///
/// Without this, both write their progress and load messages straight to stderr, interleaved
/// with whatever the CLI or a diagnostic is printing. Every whisper.cpp entry point in this
/// crate has to call it before loading anything -- [`crate::SileroVad`] as well as
/// [`WhisperEngine`] -- which is why the `Once` lives here rather than inside `load`. One
/// global gets one piece of state; a second `Once` elsewhere would be two states for one
/// global, and that is how a global ends up half-initialised.
pub(crate) fn install_logging_hooks() {
    static HOOKS: Once = Once::new();
    HOOKS.call_once(whisper_rs::install_logging_hooks);
}

/// A loaded Whisper model, reusable across sessions.
///
/// Loading is the expensive part -- gigabytes read and uploaded to the GPU -- so a batch
/// builds one of these and runs every session through it.
///
/// The voice-activity detector is part of the engine rather than something a caller may attach.
/// An engine that had quietly stopped gating would produce exactly the transcript this gate
/// exists to prevent, and there is no caller that wants that state to be reachable -- so
/// [`Self::load`] requires the VAD weights and there is no builder to forget.
pub struct WhisperEngine {
    context: WhisperContext,
    vad: SileroVad,
    tuning: VadTuning,
    threads: i32,
    accelerated: bool,
}

impl WhisperEngine {
    /// Loads a ggml Whisper checkpoint.
    ///
    /// Metal is used via the `metal` cargo feature, which is also what makes
    /// `WhisperContextParameters::use_gpu` default to true. That default is asserted rather
    /// than set: if the feature is ever dropped, this fails loudly at startup instead of
    /// quietly transcribing on the CPU at a fraction of the speed.
    ///
    /// The GPU decision is taken here rather than at CLI entry because this is the exact call
    /// that crashes without a device, and because the CLI opens engines lazily -- a check at
    /// entry would fail a run that had nothing to transcribe, on a machine it was never going
    /// to touch. See the `gpu` module for why a missing device is an error rather than a silent
    /// CPU fallback.
    ///
    /// `vad_model_path` is [`crate::SILERO_VAD_MODEL`], and it is loaded **first**: 885 KB on
    /// the CPU, so a missing or truncated file fails in under a second rather than after 1.6 GB
    /// has been read and uploaded to the GPU.
    pub fn load(model_path: &Path, vad_model_path: &Path) -> Result<WhisperEngine> {
        install_logging_hooks();

        let vad = SileroVad::load(vad_model_path)?;

        let mut params = WhisperContextParameters::default();
        assert!(
            params.use_gpu,
            "whisper-rs was built without a GPU backend; the `metal` feature is required"
        );
        // Must happen before `new_with_params`. Loading with `use_gpu` still set on a process
        // that cannot reach a device gets as far as ggml's Metal allocator, which does not
        // null-check a failed allocation and dies with SIGSEGV -- not something a `?` further
        // down could have caught.
        let accelerated = gpu::use_gpu()?;
        params.use_gpu(accelerated);

        let context =
            WhisperContext::new_with_params(model_path, params).map_err(|e| Error::Asr {
                source: Box::new(e),
            })?;

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(MAX_THREADS) as i32;

        Ok(WhisperEngine {
            context,
            vad,
            tuning: gate_tuning(),
            threads,
            accelerated,
        })
    }

    /// False when recognition is running on the CPU because `MEETHOOK_CPU` asked it to.
    ///
    /// Mirrors [`crate::OnnxDiarizer::accelerated`], for the same reason and with the same
    /// rule: a CPU run is correct and many times slower, and reporting is the caller's job,
    /// so loading prints on no path at all.
    pub fn accelerated(&self) -> bool {
        self.accelerated
    }
}

impl SpeechToText for WhisperEngine {
    /// Recognizes only the stretches of `audio_16k_mono` that hold speech.
    ///
    /// The gate is here, below the [`SpeechToText`] seam, rather than in the orchestration
    /// above it. That keeps the trait one method wide, keeps [`crate::merge()`] and the batch
    /// tests running against a fake recogniser, and means both of a session's tracks are gated
    /// by one code path with no branch to get wrong.
    fn transcribe(&mut self, audio_16k_mono: &[f32]) -> Result<Vec<AsrSegment>> {
        let regions = self.vad.speech_regions(audio_16k_mono, self.tuning)?;
        let plan = gate::Splice::plan(&regions, audio_16k_mono.len());
        report(audio_16k_mono.len(), plan.as_ref());

        // Before `create_state`, deliberately: "a track with no speech is not handed to the
        // recogniser at all" is then a property of the order of these lines rather than of a
        // comment claiming it.
        let Some(splice) = plan else {
            return Ok(Vec::new());
        };

        let spliced = splice.build(audio_16k_mono);

        let mut state = self.context.create_state().map_err(|e| Error::Asr {
            source: Box::new(e),
        })?;

        // Padding the *spliced* buffer, not the track. The no-speech case returned above, so
        // this is only ever reached with real speech in the buffer -- a silent track is no
        // longer padded up to a decodable second and then decoded. A timestamp that lands in
        // the pad maps to the last piece's end by rule 3 of `Splice::to_original`.
        let padded;
        let audio = if spliced.len() < MIN_SAMPLES {
            padded = {
                let mut buf = spliced;
                buf.resize(MIN_SAMPLES, 0.0);
                buf
            };
            &padded[..]
        } else {
            &spliced[..]
        };

        // Greedy rather than beam search: a beam is several times the compute for an
        // accuracy difference that does not show up in meeting speech, and this runs over
        // whole meetings.
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_language(Some(LANGUAGE));
        params.set_translate(false);
        params.set_n_threads(self.threads);
        // whisper.cpp will otherwise print its own transcript to stdout, corrupting the
        // CLI's output.
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // States the intent that each call starts from no transcript history rather than
        // inheriting one. Three things about it are worth writing down, because all three are
        // easy to assume wrongly:
        //
        // 1. whisper.cpp 1.8.3 already defaults it to true (`whisper.cpp:5916`, and
        //    `FullParams::new` is a thin wrapper over `whisper_full_default_params`), so this
        //    line changes no behaviour. whisper-rs's own doc comment saying "Defaults to false"
        //    is stale, in exactly the way its `no_speech_thold` comment is.
        // 2. Its scope is the *call*, not the window: it clears the prompt history once on
        //    entry to `whisper_full_with_state` (`:6900`), and a fresh state is created above
        //    on every call, so the history it clears was already empty.
        // 3. What bounds priming *between* the 30 s windows of one call is `n_max_text_ctx`
        //    (`:7090`), not this -- `prompt_past` is rebuilt after each window (`:7591`) and
        //    fed to the next (`:7107`) regardless of `no_context`. It is left at its default
        //    because the gate removes the windows that were repeating, which is measured rather
        //    than assumed: re-transcribing `20260810-093047` with the gate took the longest run
        //    of identical consecutive turns from 63 to 7 on the mic track (the 63 were
        //    "Thank you." over 723-2612 s) and from 38 to 7 on the speaker track, with the 76
        //    silence hallucinations reduced to one 0.15 s turn inside real detected speech.
        //    So the repetition went with the silent windows, and no decoder flag was needed to
        //    take it. Setting `n_max_text_ctx` to 0 costs the rolling prompt outright, and the
        //    runs that survive are seconds long rather than the 25 minutes that motivated this,
        //    so it is not worth spending yet. The residue is a real ceiling, not a clean zero:
        //    the surviving speaker-track run of 7 (308-311 s) repeats one phrase over audio
        //    where the pre-gate transcript had different words, so it is a within-window
        //    repetition artefact and this is the knob to reach for if that becomes the
        //    complaint.
        params.set_no_context(true);

        // Every other decoder parameter is left at whisper.cpp's default, checked rather than
        // assumed against the vendored 1.8.3: `suppress_blank` is already true (`:5946`),
        // `no_speech_thold` is 0.6 and *is* live (`:7555`, `:7585`) despite whisper-rs's stale
        // comment, and `temperature` 0.0 / `temperature_inc` 0.2 / `entropy_thold` 2.4 /
        // `logprob_thold` -1.0 are the temperature-fallback loop that already rejects
        // degenerate output. None of them is what let the silence hallucinations through, so
        // none of them is changed here.

        state.full(params, audio).map_err(|e| Error::Asr {
            source: Box::new(e),
        })?;

        let mut segments = Vec::new();
        for segment in state.as_iter() {
            let text = segment
                .to_str_lossy()
                .map_err(|e| Error::Asr {
                    source: Box::new(e),
                })?
                .trim()
                .to_string();
            if text.is_empty() || text == BLANK_AUDIO {
                continue;
            }
            // whisper.cpp reports centiseconds, and reports them against the spliced buffer it
            // was handed. Everything above this seam expects seconds on the original track.
            let (start_s, end_s) = splice.to_original(
                segment.start_timestamp() as f64 / 100.0,
                segment.end_timestamp() as f64 / 100.0,
            );
            segments.push(AsrSegment {
                start_s,
                end_s,
                text,
            });
        }
        Ok(segments)
    }
}

/// Says how much of the track the gate is about to decode, one line per call on stderr.
///
/// A track with nothing to decode gets a line too, rather than silence: "no speech detected" is
/// the answer a user most needs to see, since it is the one case where an empty transcript is
/// the gate's doing rather than the recogniser's.
///
/// A gate that quietly stopped gating is the same class of failure as a mic track that quietly
/// stopped being cleaned, which [`crate::transcribe_session`] already reports a line about --
/// and the number that makes it visible is how much of each track was actually decoded.
///
/// stderr rather than the `progress` writer the cleaning note uses, and that is a considered
/// second best. The gate sits below the [`SpeechToText`] seam so that the trait stays one method
/// wide; getting these numbers to the orchestration would mean either widening the trait or
/// giving the engine a sink that every construction site has to remember to set, and a report
/// that can be forgotten is the failure this one exists to prevent. stderr is where this tool
/// already puts operational notes that are not the batch's per-session record -- download
/// progress and both accelerator notes -- and it cannot corrupt the greppable stdout log.
///
/// The line cannot name its track, because the seam never tells the engine which one it holds.
/// They arrive in a fixed order instead: cleaned mic track, then speaker track, because that is
/// the order [`crate::transcribe_session`] calls in.
fn report(total_samples: usize, plan: Option<&gate::Splice>) {
    match plan {
        Some(splice) => {
            let total = splice.total_duration_s();
            let speech = splice.speech_duration_s();
            eprintln!(
                "speech gate: {total:.1} s in, {speech:.1} s of speech in {} region(s) ({:.1}%), \
                 {} window(s) to decode",
                splice.piece_count(),
                speech / total.max(f64::MIN_POSITIVE) * 100.0,
                splice.decode_windows()
            );
        }
        // `total_samples` is the argument rather than the plan's, because there is no plan on
        // this path: a track with no speech is exactly the case that produced no `Splice`.
        None => {
            let total = total_samples as f64 / f64::from(TARGET_RATE);
            eprintln!("speech gate: {total:.1} s in, no speech detected; nothing decoded");
        }
    }
}
