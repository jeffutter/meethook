//! Speech recognition: the [`SpeechToText`] seam, and the whisper.cpp engine behind it.

use std::path::Path;
use std::sync::Once;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::TARGET_RATE;
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
const BLANK_AUDIO: &str = "[BLANK_AUDIO]";

/// A loaded Whisper model, reusable across sessions.
///
/// Loading is the expensive part -- gigabytes read and uploaded to the GPU -- so a batch
/// builds one of these and runs every session through it.
pub struct WhisperEngine {
    context: WhisperContext,
    threads: i32,
}

impl WhisperEngine {
    /// Loads a ggml Whisper checkpoint.
    ///
    /// Metal is used via the `metal` cargo feature, which is also what makes
    /// `WhisperContextParameters::use_gpu` default to true. That default is asserted rather
    /// than set: if the feature is ever dropped, this fails loudly at startup instead of
    /// quietly transcribing on the CPU at a fraction of the speed.
    pub fn load(model_path: &Path) -> Result<WhisperEngine> {
        static HOOKS: Once = Once::new();
        // Without this, ggml and whisper.cpp write their own progress and load messages
        // straight to stderr, interleaved with the CLI's output.
        HOOKS.call_once(whisper_rs::install_logging_hooks);

        let params = WhisperContextParameters::default();
        assert!(
            params.use_gpu,
            "whisper-rs was built without a GPU backend; the `metal` feature is required"
        );

        let context =
            WhisperContext::new_with_params(model_path, params).map_err(|e| Error::Asr {
                source: Box::new(e),
            })?;

        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(MAX_THREADS) as i32;

        Ok(WhisperEngine { context, threads })
    }
}

impl SpeechToText for WhisperEngine {
    fn transcribe(&mut self, audio_16k_mono: &[f32]) -> Result<Vec<AsrSegment>> {
        let mut state = self.context.create_state().map_err(|e| Error::Asr {
            source: Box::new(e),
        })?;

        let padded;
        let audio = if audio_16k_mono.len() < MIN_SAMPLES {
            padded = {
                let mut buf = audio_16k_mono.to_vec();
                buf.resize(MIN_SAMPLES, 0.0);
                buf
            };
            &padded[..]
        } else {
            audio_16k_mono
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
            segments.push(AsrSegment {
                // whisper.cpp reports centiseconds.
                start_s: segment.start_timestamp() as f64 / 100.0,
                end_s: segment.end_timestamp() as f64 / 100.0,
                text,
            });
        }
        Ok(segments)
    }
}
