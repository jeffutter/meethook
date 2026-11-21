//! Speech recognition: the [`SpeechToText`] seam, and the whisper.cpp engine behind it.

use std::path::Path;
use std::sync::Once;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::TARGET_RATE;
use crate::gate;
use crate::gpu;
use crate::vad::{FireRedVad, VadTuning};
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

    /// Biases recognition toward `terms`, replacing any earlier vocabulary. Returns how many
    /// terms were dropped to fit the prompt window. Recognizers with no prompt ignore it.
    fn set_vocabulary(&mut self, _terms: &[String]) -> usize {
        0
    }

    /// The prompt [`set_vocabulary`](Self::set_vocabulary) built, if any: the one place the
    /// text the recognizer got can be read back, so what is recorded is what was used.
    fn vocabulary_prompt(&self) -> Option<&str> {
        None
    }
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
/// `min_silence_s` is raised from the detector's 0.1 s default. Measured with
/// `examples/vad-regions` on `20260810-093047`'s `mic.cleaned.wav` under the previous detector
/// (Silero): 0.1 s gave 463 regions holding 713.4 s of speech, 0.5 s gave 315 regions holding
/// 791.6 s. That is 148 fewer seams for 78 s more audio -- roughly three more 30 s decoder
/// windows -- and a seam is the only place rule 4 of [`gate::Splice::to_original`] can fire, so
/// fewer of them is the axis worth buying on. It also keeps one speaker's turn in one piece
/// across a short pause. Re-measure before treating those two numbers as FireRed's.
///
/// `threshold` is [`VadTuning::default`]'s 0.35, chosen by listening to this corpus; see
/// [`crate::FireRedVad`].
pub(crate) fn gate_tuning() -> VadTuning {
    VadTuning {
        min_silence_s: 0.5,
        ..VadTuning::default()
    }
}

/// Routes ggml's and whisper.cpp's own log output through Rust, once per process.
///
/// Without this, both write their progress and load messages straight to stderr, interleaved
/// with whatever the CLI or a diagnostic is printing. Every whisper.cpp entry point in this
/// crate has to call it before loading anything -- which is why the `Once` lives here rather
/// than inside `load`. One global gets one piece of state; a second `Once` elsewhere would be
/// two states for one global, and that is how a global ends up half-initialised.
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
    vad: FireRedVad,
    tuning: VadTuning,
    threads: i32,
    accelerated: bool,
    /// Initial prompt built from vocabulary terms; `None` leaves the recognition call untouched.
    vocabulary_prompt: Option<String>,
}

impl WhisperEngine {
    /// Loads a ggml Whisper checkpoint.
    ///
    /// On macOS, Metal is used via the `metal` cargo feature, which is also what makes
    /// `WhisperContextParameters::use_gpu` default to true. That default is asserted rather
    /// than set: if the feature is ever dropped, this fails loudly at startup instead of
    /// quietly transcribing on the CPU at a fraction of the speed. Off macOS the build is
    /// CPU-only by design, `use_gpu` defaults to false, and there is nothing to assert -- the
    /// platform *is* the opt-in to slowness, chosen at build time rather than runtime.
    ///
    /// The GPU decision is taken here rather than at CLI entry because this is the exact call
    /// that crashes without a device, and because the CLI opens engines lazily -- a check at
    /// entry would fail a run that had nothing to transcribe, on a machine it was never going
    /// to touch. See the `gpu` module for why a missing device is an error rather than a silent
    /// CPU fallback.
    ///
    /// `vad_model_path` is [`crate::FIRERED_VAD_MODEL`], and it is loaded **first**: 2.3 MB on
    /// the CPU, so a missing or truncated file fails in under a second rather than after 1.6 GB
    /// has been read and uploaded to the GPU.
    pub fn load(model_path: &Path, vad_model_path: &Path) -> Result<WhisperEngine> {
        install_logging_hooks();

        let vad = FireRedVad::load(vad_model_path)?;

        let mut params = WhisperContextParameters::default();
        // Only meaningful where a GPU backend was compiled in. A macOS build without the
        // `metal` feature would silently run every batch on the CPU at a fraction of the
        // speed, so the pairing is pinned here; off macOS `false` is the intended default
        // and asserting it would just restate the build configuration.
        #[cfg(target_os = "macos")]
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
            vocabulary_prompt: None,
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

/// Lead-in of the vocabulary prompt. A bare list of terms did not fix a misheard name in
/// experiment; a short sentence naming the terms did.
const PROMPT_LEAD_IN: &str = "Meeting notes: ";

/// Byte budget for the whole prompt sentence. Whisper's prompt window is 224 tokens and no
/// tokenizer is available here, so this is a conservative approximation: 400 bytes is at
/// most about 2 bytes per token for unusual names, comfortably under the window.
const PROMPT_MAX_BYTES: usize = 400;

/// The result of [`build_vocabulary_prompt`].
#[derive(Debug, PartialEq, Eq)]
pub struct VocabularyPrompt {
    /// The prompt sentence, or `None` when there are no usable terms.
    pub text: Option<String>,
    /// Terms left out because they would not fit [`PROMPT_MAX_BYTES`]. Duplicates and blank
    /// terms are not counted.
    pub dropped: usize,
}

/// Builds the initial-prompt sentence `Meeting notes: A, B, C.` from `terms`.
///
/// Terms are trimmed, blanks skipped, and duplicates (case-insensitive) removed keeping the
/// first spelling: a term repeated in the prompt makes the model echo it into the transcript.
/// Order is preserved; terms that do not fit the byte budget are dropped from the end, never
/// truncated.
pub(crate) fn build_vocabulary_prompt(terms: &[String]) -> VocabularyPrompt {
    let mut seen = std::collections::HashSet::new();
    let mut text = String::from(PROMPT_LEAD_IN);
    let mut kept = 0usize;
    let mut dropped = 0usize;
    for term in terms {
        // A NUL would panic inside whisper-rs's CString conversion.
        let term = term.replace('\0', "");
        let term = term.trim();
        if term.is_empty() || !seen.insert(term.to_lowercase()) {
            continue;
        }
        // Once one term does not fit, later ones are dropped too, so the cut is from the end.
        let extra = if kept == 0 {
            term.len()
        } else {
            2 + term.len()
        };
        if dropped > 0 || text.len() + extra + 1 > PROMPT_MAX_BYTES {
            dropped += 1;
            continue;
        }
        if kept > 0 {
            text.push_str(", ");
        }
        text.push_str(term);
        kept += 1;
    }
    if kept == 0 {
        return VocabularyPrompt {
            text: None,
            dropped,
        };
    }
    text.push('.');
    VocabularyPrompt {
        text: Some(text),
        dropped,
    }
}

impl SpeechToText for WhisperEngine {
    /// Biases decoding toward the spelling of `terms` (names, jargon) with an initial prompt.
    /// Replaces any earlier vocabulary; an empty list clears it. The prompt only ever reaches
    /// the gated path in [`transcribe`](Self::transcribe).
    fn set_vocabulary(&mut self, terms: &[String]) -> usize {
        let built = build_vocabulary_prompt(terms);
        self.vocabulary_prompt = built.text;
        built.dropped
    }

    fn vocabulary_prompt(&self) -> Option<&str> {
        self.vocabulary_prompt.as_deref()
    }

    /// Recognizes only the stretches of `audio_16k_mono` that hold speech.
    ///
    /// The gate is here, below the [`SpeechToText`] seam, rather than in the orchestration
    /// above it. That keeps the trait one method wide, keeps [`crate::merge()`] and the batch
    /// tests running against a fake recogniser, and means both of a session's tracks are gated
    /// by one code path with no branch to get wrong.
    fn transcribe(&mut self, audio_16k_mono: &[f32]) -> Result<Vec<AsrSegment>> {
        let segments = self.decode(audio_16k_mono)?;
        self.repair_loops(audio_16k_mono, segments)
    }
}

impl WhisperEngine {
    /// Re-decodes every stretch where the decoder got stuck repeating itself.
    ///
    /// The visible effect is one sentence printed dozens of times across 30 s of a turn, in place
    /// of what was said. whisper.cpp's greedy decoder falls into a loop it cannot leave, and its
    /// own guards (`entropy_thold`, `logprob_thold`) pick the loop anyway because a loop scores as
    /// highly confident. Measured on session `20261120-125850`'s speaker track: 17 runs of 3 or
    /// more identical consecutive segments, the longest 31, and which windows loop differs from
    /// run to run.
    ///
    /// The loop comes from the text carried between the 30 s windows of one call, so a span
    /// decoded as its own call, with no preceding text, does not loop. `n_max_text_ctx = 0`
    /// would remove the carry-over everywhere, but measured over the same track it left 159 long
    /// segments with no punctuation and about 8% fewer words, so only the broken spans pay.
    ///
    /// A span that still loops on the second attempt is collapsed to one segment: the repeats
    /// carry no information, and a single sentence is a smaller lie than thirty.
    fn repair_loops(
        &mut self,
        audio_16k_mono: &[f32],
        segments: Vec<AsrSegment>,
    ) -> Result<Vec<AsrSegment>> {
        /// Context on each side of a loop, so the fresh decode starts in silence or the
        /// previous sentence rather than mid-word.
        const MARGIN_S: f64 = 2.0;

        let mut out = Vec::with_capacity(segments.len());
        let mut i = 0;
        while i < segments.len() {
            let run = loop_run_len(&segments[i..]);
            if run == 0 {
                out.push(segments[i].clone());
                i += 1;
                continue;
            }
            let looped = &segments[i..i + run];
            let (first, last) = (&looped[0], &looped[run - 1]);
            // Stay inside the neighbours so the redo cannot duplicate a sentence that was
            // already decoded correctly on either side.
            let lo =
                (first.start_s - MARGIN_S).max(out.last().map_or(0.0, |s: &AsrSegment| s.end_s));
            let hi =
                (last.end_s + MARGIN_S).min(segments.get(i + run).map_or(f64::MAX, |s| s.start_s));
            let rate = f64::from(TARGET_RATE);
            let from = ((lo * rate) as usize).min(audio_16k_mono.len());
            let to = ((hi * rate) as usize).clamp(from, audio_16k_mono.len());
            let mut redone = self.decode(&audio_16k_mono[from..to])?;
            for s in &mut redone {
                s.start_s += from as f64 / rate;
                s.end_s += from as f64 / rate;
            }
            if (0..redone.len()).any(|k| loop_run_len(&redone[k..]) > 0) {
                eprintln!(
                    "repeat guard: {:.1}-{:.1} s still repeats after a fresh decode; keeping one",
                    first.start_s, last.end_s
                );
                out.push(AsrSegment {
                    start_s: first.start_s,
                    end_s: last.end_s,
                    text: first.text.clone(),
                });
            } else {
                eprintln!(
                    "repeat guard: {:.1}-{:.1} s repeated one sentence {run} times; re-decoded",
                    first.start_s, last.end_s
                );
                out.extend(redone);
            }
            i += run;
        }
        Ok(out)
    }

    /// The gated decode of one buffer; see [`SpeechToText::transcribe`].
    fn decode(&mut self, audio_16k_mono: &[f32]) -> Result<Vec<AsrSegment>> {
        // Opens the bracket the detector's own heartbeat ticks inside (a `Phase`, silent unless
        // the scan is slow) and the `report` line below closes with the answer.
        eprintln!(
            "speech gate: scanning {:.0} s for speech...",
            audio_16k_mono.len() as f64 / f64::from(TARGET_RATE)
        );
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

        // Below the gate on purpose: ungated, a prompt on a mostly-silent window produced runs
        // of "I don't know." hallucinations. With no vocabulary no call is made, so the
        // parameters are exactly what they were before.
        if let Some(prompt) = &self.vocabulary_prompt {
            params.set_initial_prompt(prompt);
        }

        // Decoding a meeting is minutes of work behind one call, so whisper.cpp's own progress
        // hook is attached: it fires from the head of the main loop, once per 30 s decoder
        // window, with a percentage of the spliced buffer -- ~85 calls on a 49-minute speaker
        // track, which the phase's own throttle turns into a handful of lines.
        //
        // Two properties of this API are load-bearing and neither is obvious:
        //
        // 1. whisper-rs requires `FnMut(i32) + 'static`, so the closure cannot borrow the
        //    engine or a caller-supplied writer. It owns its phase and prints to stderr, which
        //    is where `report` below already puts the gate's numbers for the same reason.
        // 2. `set_progress_callback_safe` hands the boxed closure to `Box::into_raw` and
        //    nothing ever reclaims it -- `FullParams` has no `Drop` and clears its own safe
        //    field. So this leaks one small box per call, two per session. That is acceptable
        //    for a per-meeting cost, but it is deliberate rather than missed.
        let mut phase = crate::progress::Phase::start("transcribing");
        params.set_progress_callback_safe(move |percent: i32| {
            phase.at(percent.max(0) as usize, 100);
        });

        // Two settings follow screenpipe's whisper_rs use (`screenpipe-audio`,
        // `transcription/whisper/batch.rs`), adopted on trust rather than measured here:
        //
        // - `suppress_nst` bans the non-speech tokens (music notes, bracketed sound tags) so
        //   they cannot be emitted. Off by default in whisper.cpp (`:5947`).
        // - `logprob_thold` -2.0 instead of the default -1.0. It gates both the temperature
        //   fallback (`:7555`) and the drop of a window judged silent (`:7585`), so a lower
        //   bar means fewer retries and a window needs a worse score to be dropped.
        //
        // Neither is aimed at decoder loops, which score as confident; see `repair_loops`.
        params.set_suppress_nst(true);
        params.set_logprob_thold(-2.0);

        // Every other decoder parameter is left at whisper.cpp's default, checked rather than
        // assumed against the vendored 1.8.3: `suppress_blank` is already true (`:5946`),
        // `no_speech_thold` is 0.6 and *is* live (`:7555`, `:7585`) despite whisper-rs's stale
        // comment, and `temperature` 0.0 / `temperature_inc` 0.2 / `entropy_thold` 2.4 are the
        // temperature-fallback loop. None of them is what let the silence hallucinations
        // through, so none of them is changed here.

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

/// Length of the run of identical consecutive segments at the start of `segments`, or 0 when it
/// is too short to be a decoder loop.
///
/// Three or more of a sentence of three or more words, or five or more of anything: a real
/// speaker says "Yeah." twice, and the loops seen in practice are twelve to thirty-one long.
/// Text is compared lowercased with punctuation dropped, because the repeats are not always
/// byte-identical ("Yeah" / "Yeah.").
fn loop_run_len(segments: &[AsrSegment]) -> usize {
    fn key(text: &str) -> String {
        text.chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .flat_map(char::to_lowercase)
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    let Some(first) = segments.first() else {
        return 0;
    };
    let k = key(&first.text);
    let run = segments.iter().take_while(|s| key(&s.text) == k).count();
    let words = k.split(' ').count();
    if (run >= 3 && words >= 3) || run >= 5 {
        run
    } else {
        0
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

#[cfg(test)]
mod vocabulary_tests {
    use super::*;

    fn v(terms: &[&str]) -> Vec<String> {
        terms.iter().map(|t| t.to_string()).collect()
    }

    #[test]
    fn sentence_form_in_order() {
        let p = build_vocabulary_prompt(&v(&["Brog", "Roll20", "D&D"]));
        assert_eq!(p.text.as_deref(), Some("Meeting notes: Brog, Roll20, D&D."));
        assert_eq!(p.dropped, 0);
    }

    fn seg(text: &str) -> AsrSegment {
        AsrSegment {
            start_s: 0.0,
            end_s: 1.0,
            text: text.to_string(),
        }
    }

    #[test]
    fn a_decoder_loop_is_a_run_of_the_same_sentence() {
        let run: Vec<_> = (0..4)
            .map(|_| seg("So, we're going to be at least."))
            .collect();
        assert_eq!(loop_run_len(&run), 4);
    }

    #[test]
    fn punctuation_and_case_do_not_hide_a_loop() {
        let run: Vec<_> = ["Yeah", "yeah.", "Yeah.", "YEAH", "yeah!"].map(seg).into();
        assert_eq!(loop_run_len(&run), 5);
    }

    #[test]
    fn ordinary_backchannel_is_not_a_loop() {
        let run: Vec<_> = (0..3).map(|_| seg("Yeah.")).collect();
        assert_eq!(loop_run_len(&run), 0);
        let two: Vec<_> = (0..2).map(|_| seg("So you can see it.")).collect();
        assert_eq!(loop_run_len(&two), 0);
    }

    #[test]
    fn a_run_is_counted_from_the_start_only() {
        let mut v = vec![seg("Hello there friend.")];
        v.extend((0..4).map(|_| seg("Same words here.")));
        assert_eq!(loop_run_len(&v), 0);
        assert_eq!(loop_run_len(&v[1..]), 4);
    }

    #[test]
    fn dedup_is_case_insensitive_and_keeps_first() {
        let p = build_vocabulary_prompt(&v(&["Brog", "brog", " BROG ", "Roll20"]));
        assert_eq!(p.text.as_deref(), Some("Meeting notes: Brog, Roll20."));
        assert_eq!(p.dropped, 0);
    }

    #[test]
    fn empty_and_blank_give_none() {
        assert_eq!(
            build_vocabulary_prompt(&[]),
            VocabularyPrompt {
                text: None,
                dropped: 0
            }
        );
        assert_eq!(build_vocabulary_prompt(&v(&["", "  "])).text, None);
    }

    #[test]
    fn bound_drops_from_the_end_and_counts() {
        let terms: Vec<String> = (0..100).map(|i| format!("Term{i:03}xxxxxxxx")).collect();
        let p = build_vocabulary_prompt(&terms);
        let text = p.text.unwrap();
        assert!(text.len() <= PROMPT_MAX_BYTES);
        assert!(p.dropped > 0);
        let kept = text.matches("Term").count();
        assert_eq!(kept + p.dropped, 100);
        assert!(text.contains("Term000"));
        assert!(!text.contains("Term099"));
    }

    #[test]
    fn oversized_term_is_dropped_not_truncated() {
        let p = build_vocabulary_prompt(&["x".repeat(500), "Brog".to_string()]);
        assert_eq!(p.dropped, 2);
        assert_eq!(p.text, None);
    }
}
