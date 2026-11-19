//! Running a whole speaker track through Nemotron-3-Diarization.
//!
//! Joins the three parts that each know one thing: [`NemotronMel`] makes features for a range
//! of frames, [`NemotronGraph`] turns one chunk of features plus the cached embeddings into
//! logits, and [`SpeakerCache`] decides what to remember between chunks. This module is only
//! the loop that feeds one into the next (doc-015's `stream.py` is the reference).
//!
//! Memory stays flat: only the mel frames of the chunk under way (plus its right context,
//! about 1.5 MB) are ever materialised, never the roughly 400 MB matrix of a two hour track.

use crate::Result;
use crate::nemotron_cache::SpeakerCache;
use crate::nemotron_cache::config::{CHUNK, HIDDEN, RIGHT, SPEAKERS};
use crate::nemotron_graph::{NemotronGraph, StepOutput, encoder_frames, encoder_valid};
use crate::nemotron_mel::{MEL_BINS, NemotronMel, STACK, frame_count, valid_frames};
use crate::progress::Phase;

/// One call of the graph, so the chunk and cache behaviour can be tested without weights.
pub trait SpeakerStep {
    /// As [`NemotronGraph::step`].
    fn step(&mut self, features: &[f32], cached: &[f32], valid: &[bool]) -> Result<StepOutput>;
}

impl SpeakerStep for NemotronGraph {
    fn step(&mut self, features: &[f32], cached: &[f32], valid: &[bool]) -> Result<StepOutput> {
        NemotronGraph::step(self, features, cached, valid)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Per-slot speech probabilities for a 16 kHz mono track: one row of `SPEAKERS` values per
/// 10 ms mel frame, `frame_count(samples.len()) * SPEAKERS` values in all, slot-minor. An
/// empty track returns no rows and never touches the graph.
pub fn speaker_probabilities(
    graph: &mut impl SpeakerStep,
    mel: &mut NemotronMel,
    samples: &[f32],
) -> Result<Vec<f32>> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }
    let frames = frame_count(samples.len());
    let valid_total = valid_frames(samples.len());
    let n_enc = encoder_frames(frames);
    let steps = n_enc.div_ceil(CHUNK);

    let mut probs = Vec::with_capacity(frames.div_ceil(STACK) * STACK * SPEAKERS);
    let mut cache = SpeakerCache::new();
    let mut phase = Phase::start("diarize: nemotron");

    for (step, start) in (0..n_enc).step_by(CHUNK).enumerate() {
        phase.at(step, steps);
        let n_chunk = CHUNK.min(n_enc - start);
        let first_mel = start * STACK;
        let features = mel.compute_range(samples, first_mel, (n_chunk + RIGHT) * STACK);
        let range_len = features.len() / MEL_BINS;
        let valid = encoder_valid(
            valid_total.saturating_sub(first_mel).min(range_len),
            range_len,
        );

        let cached = cache.cached_embeds();
        let cached_frames = cached.len() / HIDDEN;
        let out = graph.step(&features, &cached, &valid)?;

        let rows = cached_frames * STACK * SPEAKERS..(cached_frames + n_chunk) * STACK * SPEAKERS;
        probs.extend(out.logits[rows].iter().map(|&x| sigmoid(x)));
        // The cache wants the attention mask over the cached rows too: ones, then the chunk's.
        let mut mask = vec![true; cached_frames];
        mask.extend_from_slice(&valid);
        cache.update(
            &out.chunk_embeds[..n_chunk * HIDDEN],
            &out.logits,
            &mask,
            &out.silence,
        );
    }
    phase.done();

    // The last encoder frame stacks eight mel frames, some past the end of the track.
    probs.truncate(frames * SPEAKERS);
    Ok(probs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nemotron_cache::config::{CACHE_LEN, FIFO};

    /// What the fake saw in one call.
    #[derive(Debug)]
    struct Call {
        mel_frames: usize,
        cached_frames: usize,
        valid_len: usize,
    }

    /// A graph with no weights that records what it was given and returns zero logits.
    struct Fake {
        calls: Vec<Call>,
        fail_at: Option<usize>,
    }

    impl SpeakerStep for Fake {
        fn step(&mut self, features: &[f32], cached: &[f32], valid: &[bool]) -> Result<StepOutput> {
            let mel_frames = features.len() / MEL_BINS;
            let cached_frames = cached.len() / HIDDEN;
            self.calls.push(Call {
                mel_frames,
                cached_frames,
                valid_len: valid.len(),
            });
            if self.fail_at == Some(self.calls.len() - 1) {
                return Err(crate::Error::Nemotron("fake failure".into()));
            }
            let enc = encoder_frames(mel_frames);
            Ok(StepOutput {
                logits: vec![0.0; (cached_frames + enc) * STACK * SPEAKERS],
                chunk_embeds: (0..enc * HIDDEN).map(|i| (i % 13) as f32).collect(),
                silence: vec![0.5; HIDDEN],
            })
        }
    }

    fn fake() -> Fake {
        Fake {
            calls: Vec::new(),
            fail_at: None,
        }
    }

    fn track(frames: usize) -> Vec<f32> {
        // Deterministic non-silent audio of `frames` 10 ms frames.
        (0..frames * 160)
            .map(|i| ((i * 31 % 2003) as f32 / 2003.0 - 0.5) * 0.2)
            .collect()
    }

    #[test]
    fn empty_track_returns_nothing_and_never_calls_the_graph() {
        let mut g = fake();
        let out = speaker_probabilities(&mut g, &mut NemotronMel::new(), &[]).unwrap();
        assert!(out.is_empty());
        assert!(g.calls.is_empty());
    }

    #[test]
    fn output_covers_every_mel_frame_for_awkward_lengths() {
        let one_chunk = CHUNK * STACK * 160;
        for len in [
            1,
            159,
            160,
            161,
            5_000,
            one_chunk - 1,
            one_chunk,
            one_chunk + 1,
            700_000,
        ] {
            let samples = &track(len / 160 + 1)[..len];
            let mut g = fake();
            let out = speaker_probabilities(&mut g, &mut NemotronMel::new(), samples).unwrap();
            assert_eq!(out.len(), frame_count(len) * SPEAKERS, "{len} samples");
        }
    }

    #[test]
    fn chunks_start_every_340_frames_with_right_context_except_the_last() {
        // 100 s is the fixture shape: 10001 mel frames, 1251 encoder frames, four chunks.
        let samples = track(10_000);
        let mut g = fake();
        speaker_probabilities(&mut g, &mut NemotronMel::new(), &samples).unwrap();
        let frames = frame_count(samples.len());
        assert_eq!(frames, 10_001);
        assert_eq!(g.calls.len(), 4);
        for call in &g.calls[..3] {
            assert_eq!(call.mel_frames, (CHUNK + RIGHT) * STACK);
            assert_eq!(call.valid_len, CHUNK + RIGHT);
        }
        let last = &g.calls[3];
        // Chunk starts at 3 * 340 * 8 = 8160; the remainder to 10001 has no right context.
        assert_eq!(last.mel_frames, frames - 3 * CHUNK * STACK);
        assert_eq!(last.valid_len, encoder_frames(last.mel_frames));
    }

    #[test]
    fn cache_and_fifo_sizes_follow_the_cache_rules() {
        let samples = track(10_000);
        let mut g = fake();
        speaker_probabilities(&mut g, &mut NemotronMel::new(), &samples).unwrap();
        let cached: Vec<usize> = g.calls.iter().map(|c| c.cached_frames).collect();
        // Nothing cached at the start; after the first chunk the cache holds its 264 rows plus
        // 40 FIFO rows, which is what the Hugging Face fixtures show too.
        assert_eq!(cached[0], 0);
        assert_eq!(cached[1], CACHE_LEN + FIFO);
        assert!(cached.iter().all(|&c| c <= CACHE_LEN + FIFO), "{cached:?}");
    }

    #[test]
    fn a_failing_step_propagates() {
        let mut g = fake();
        g.fail_at = Some(1);
        let err = speaker_probabilities(&mut g, &mut NemotronMel::new(), &track(10_000));
        assert!(err.is_err());
        assert_eq!(g.calls.len(), 2);
    }

    #[test]
    fn each_row_comes_from_its_own_mel_frame() {
        // Rather than decode the real mel, make the fake's logit a function of the position
        // within its call and check the loop places each call's rows at the right offset.
        struct ByPosition;
        impl SpeakerStep for ByPosition {
            fn step(&mut self, f: &[f32], c: &[f32], v: &[bool]) -> Result<StepOutput> {
                let mel = f.len() / MEL_BINS;
                let cached = c.len() / HIDDEN;
                let enc = encoder_frames(mel);
                assert_eq!(v.len(), enc);
                let mut logits = vec![0.0; (cached + enc) * STACK * SPEAKERS];
                // Cached rows are poison: if the loop reads them, probabilities hit 1.
                logits[..cached * STACK * SPEAKERS].fill(100.0);
                // Chunk rows: a value that encodes the position within this call.
                for r in 0..enc * STACK {
                    logits[(cached * STACK + r) * SPEAKERS] = (r % 5) as f32 - 2.0;
                }
                Ok(StepOutput {
                    logits,
                    chunk_embeds: vec![0.0; enc * HIDDEN],
                    silence: vec![0.0; HIDDEN],
                })
            }
        }
        let samples = track(7_000);
        let out =
            speaker_probabilities(&mut ByPosition, &mut NemotronMel::new(), &samples).unwrap();
        let chunk_rows = CHUNK * STACK;
        for (i, row) in out.chunks(SPEAKERS).enumerate() {
            // Calls start at multiples of the chunk, so position-in-call is i % chunk_rows.
            let want = sigmoid(((i % chunk_rows) % 5) as f32 - 2.0);
            assert!(
                (row[0] - want).abs() < 1e-6,
                "row {i}: {} vs {want}",
                row[0]
            );
            assert!(row[1..].iter().all(|&p| (p - 0.5).abs() < 1e-6), "row {i}");
        }
    }

    // ---- weights needed ----

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    const PROBS: &[u8] = include_bytes!("../testdata/nemotron/nemotron-probs.bin");

    /// The whole fixture clip through the real graph against the Hugging Face probabilities.
    /// Skips when the weights are not installed.
    #[test]
    fn the_fixture_clip_matches_the_reference() {
        let root = match std::env::var_os("MEETHOOK_ROOT") {
            Some(r) => std::path::PathBuf::from(r),
            None => match std::env::home_dir() {
                Some(h) => h.join("meethook"),
                None => return,
            },
        };
        let dir = root.join("models");
        if !crate::NEMOTRON_DIARIZATION_MODELS
            .iter()
            .all(|s| dir.join(s.file_name).is_file())
        {
            eprintln!("skipping: Nemotron pair not installed; run fetch-onnx-models");
            return;
        }
        let mut graph = NemotronGraph::open(&dir).expect("the installed pair must load");
        let samples = crate::nemotron_mel::tests::clip();
        let got = speaker_probabilities(&mut graph, &mut NemotronMel::new(), &samples).unwrap();
        let want = f32s(&PROBS[8..]);
        assert_eq!(got.len(), want.len());
        let max = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let flips = got
            .iter()
            .zip(&want)
            .filter(|(a, b)| (**a > 0.5) != (**b > 0.5))
            .count();
        eprintln!("whole clip: max abs probability diff {max}, 0.5 flips {flips}");
        assert!(max <= PROB_TOL, "probability diff {max}");
    }

    const PROB_TOL: f32 = 5e-3;
}
