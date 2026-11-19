//! The streaming speaker cache Nemotron-3-Diarization needs between chunks.
//!
//! The graph is stateless: each call is given the encoder frames cached so far and returns
//! logits for the chunk. Which earlier frames are worth keeping, once the cache is full, is
//! decided here, and a mistake in it only shows up as slowly worse speaker tracking over a long
//! meeting. So this is pure functions and plain `Vec<f32>` state, with no ONNX in it, in the
//! way `decode_powerset` and `merge` are: it is tested in microseconds against the Hugging Face
//! reference states in `testdata/nemotron/`, with no model weights present.
//!
//! The algorithm is `Nemotron3DiarizationSpeakerCache` of transformers (the doc-015 numpy port
//! with the two corrections recorded in `testdata/nemotron/NEMOTRON-FIXTURES.md`: the attention
//! mask is honoured, so padding frames pool to probability 0). Per step:
//!
//! 1. the logits of the whole step are squashed with a sigmoid and averaged over each group of
//!    [`STACK`] mel frames into one probability row per encoder frame, then multiplied by the
//!    validity mask ([`pool_probs`]);
//! 2. the step's chunk frames are appended to the FIFO of the most recent frames;
//! 3. when the FIFO overflows, its oldest frames move to the cache, and if that makes the
//!    cache longer than [`CACHE_LEN`] it is compressed to exactly [`CACHE_LEN`] rows: every
//!    frame is scored per speaker ([`frame_scores`]), the best ones are boosted ([`boost`]),
//!    and the top [`CACHE_LEN`] speaker-major (speaker, frame) entries are kept, grouped by
//!    speaker and in frame order within a speaker ([`compress`]).
//!
//! # Tie-breaking
//!
//! Frames with equal scores are common (a run of identical probabilities, or all `-inf`).
//! Hugging Face breaks them with `torch.topk(sorted=False)`, which is unspecified; the doc-015
//! port uses a stable descending argsort. This module uses the stable rule, written as a total
//! order: higher score first, and among equal scores the lower index first. "Index" is the
//! frame index for [`boost`] (so the earlier frame wins) and the speaker-major flat index
//! (`speaker * (frames + 1) + frame`) for the final selection (so the lower speaker, then the
//! earlier frame, wins). Two runs on one input therefore always give one cache. Scores are
//! compared with [`f32::total_cmp`], so a stray NaN still has a defined place.
//!
//! The graph that produces the logits and embeddings is TASK-088.05; the chunk loop that drives
//! this is TASK-088.06.

/// Where each tunable comes from: `Nemotron3DiarizationStreamingConfig` in transformers, whose
/// values for this checkpoint are in its `config.json` (offline mode, revision
/// `52a432672732425e4c19765b9fec9b5810e6f4ba`), cross-checked with doc-015.
pub mod config {
    pub use crate::nemotron_mel::STACK;

    /// Encoder frames per chunk (`chunk_len`, offline mode). Used by the chunk loop.
    pub const CHUNK: usize = 340;
    /// Encoder frames of right context fed with a chunk but not stored (`chunk_right_context`).
    pub const RIGHT: usize = 40;
    /// Left context frames (`chunk_left_context`); zero in offline mode, so nothing is fed twice.
    pub const LEFT: usize = 0;
    /// Capacity of the FIFO of the most recent frames (`fifo_length`).
    pub const FIFO: usize = 40;
    /// Frames moved from the FIFO to the cache when it overflows (`speaker_cache_update_period`).
    pub const UPDATE_PERIOD: usize = 300;
    /// Rows in a compressed cache (`speaker_cache_length`).
    pub const CACHE_LEN: usize = 264;
    /// Reserved silence rows per speaker in a compressed cache
    /// (`speaker_cache_silence_frames_per_speaker`).
    pub const SILENCE_PER_SPEAKER: usize = 1;
    /// Speaker slots the model predicts (`num_speakers`).
    pub const SPEAKERS: usize = 8;
    /// Encoder hidden size; the width of a cached embedding row.
    pub const HIDDEN: usize = 512;
    /// Probabilities are clamped to at least this inside the log-odds score
    /// (`prediction_score_threshold`).
    pub const SCORE_FLOOR: f32 = 0.25;
    /// Added to the score of frames newly popped from the FIFO (`latest_frames_score_boost`).
    pub const LATEST_BOOST: f32 = 0.05;
    /// Share of a speaker's budget that must score positive before weak frames are dropped
    /// (`min_positive_scores_rate`).
    pub const MIN_POS_RATE: f32 = 0.5;
    /// Share of a speaker's budget that gets the strong boost (`strong_boost_rate`).
    pub const STRONG_RATE: f32 = 0.75;
    /// Share of a speaker's budget that gets the weak boost (`weak_boost_rate`).
    pub const WEAK_RATE: f32 = 1.5;

    /// Rows each speaker may hold besides its silence rows.
    pub const BUDGET: usize = CACHE_LEN / SPEAKERS - SILENCE_PER_SPEAKER;
    /// Positive-scoring frames a speaker needs for weak frames to be dropped.
    pub const MIN_POS: usize = (BUDGET as f32 * MIN_POS_RATE) as usize;
    /// Frames per speaker given the strong boost, `2 * ln 2` (the doc-015 `-2 ln 0.5`).
    pub const N_STRONG: usize = (BUDGET as f32 * STRONG_RATE) as usize;
    /// Frames per speaker given the weak boost, `ln 2`.
    pub const N_WEAK: usize = (BUDGET as f32 * WEAK_RATE) as usize;
}

use config::{
    CACHE_LEN, FIFO, HIDDEN, LATEST_BOOST, MIN_POS, N_STRONG, N_WEAK, SCORE_FLOOR,
    SILENCE_PER_SPEAKER, SPEAKERS, STACK, UPDATE_PERIOD,
};

/// Probability rows (one per encoder frame, [`SPEAKERS`] wide) from the step's raw logits.
///
/// `logits` is `mel_frames * SPEAKERS` values; a trailing group shorter than [`STACK`] mel
/// frames is dropped. `valid` has one entry per encoder frame; an invalid (padding) frame pools
/// to probability 0.
pub fn pool_probs(logits: &[f32], valid: &[bool]) -> Vec<f32> {
    let frames = logits.len() / SPEAKERS / STACK;
    assert!(
        valid.len() >= frames,
        "mask has {} entries for {frames} encoder frames",
        valid.len()
    );
    let mut out = vec![0.0f32; frames * SPEAKERS];
    for (f, row) in out.as_chunks_mut::<SPEAKERS>().0.iter_mut().enumerate() {
        if !valid[f] {
            continue;
        }
        for (s, p) in row.iter_mut().enumerate() {
            let sum: f32 = (0..STACK)
                .map(|k| {
                    let x = logits[(f * STACK + k) * SPEAKERS + s];
                    1.0 / (1.0 + (-x).exp())
                })
                .sum();
            *p = sum / STACK as f32;
        }
    }
    out
}

/// Importance of each (frame, speaker): log-odds that the speaker is the one talking, `-inf`
/// for frames that are not speech or are too weak once the speaker has enough strong ones.
///
/// `probs` is frame-major, [`SPEAKERS`] wide; the result has the same shape.
pub fn frame_scores(probs: &[f32]) -> Vec<f32> {
    let frames = probs.len() / SPEAKERS;
    let mut scores = vec![f32::NEG_INFINITY; probs.len()];
    let mut is_pos = vec![false; probs.len()];
    let mut pos_count = [0usize; SPEAKERS];
    for f in 0..frames {
        let row = &probs[f * SPEAKERS..(f + 1) * SPEAKERS];
        let log_comp: [f32; SPEAKERS] =
            std::array::from_fn(|s| (1.0 - row[s]).max(SCORE_FLOOR).ln());
        let comp_sum: f32 = log_comp.iter().sum();
        for s in 0..SPEAKERS {
            if row[s] > 0.5 {
                let v = row[s].max(SCORE_FLOOR).ln() - log_comp[s] + comp_sum - 0.5f32.ln();
                scores[f * SPEAKERS + s] = v;
                if v > 0.0 {
                    is_pos[f * SPEAKERS + s] = true;
                    pos_count[s] += 1;
                }
            }
        }
    }
    for (i, score) in scores.iter_mut().enumerate() {
        // speech that did not score positive is dropped once the speaker has enough that did
        if probs[i] > 0.5 && !is_pos[i] && pos_count[i % SPEAKERS] >= MIN_POS {
            *score = f32::NEG_INFINITY;
        }
    }
    scores
}

/// Adds `amount` to the `n` best-scoring frames of every speaker (ties: the earlier frame).
/// `scores` is frame-major, [`SPEAKERS`] wide. `-inf` stays `-inf`.
pub fn boost(scores: &mut [f32], n: usize, amount: f32) {
    let frames = scores.len() / SPEAKERS;
    if n == 0 {
        return;
    }
    for s in 0..SPEAKERS {
        let mut order: Vec<usize> = (0..frames).collect();
        order.sort_by(|&a, &b| {
            scores[b * SPEAKERS + s]
                .total_cmp(&scores[a * SPEAKERS + s])
                .then(a.cmp(&b))
        });
        for &f in order.iter().take(n) {
            scores[f * SPEAKERS + s] += amount;
        }
    }
}

/// Which `CACHE_LEN` rows survive compression of `frames` rows, as frame indices into the
/// input; the value `frames` stands for the silence row. Ascending speaker-major order, the
/// order the rows are stored in.
pub fn select_frames(probs: &[f32]) -> Vec<usize> {
    let frames = probs.len() / SPEAKERS;
    let mut scores = frame_scores(probs);
    // rows beyond the cache capacity are the ones just popped from the FIFO
    for s in scores.iter_mut().skip(CACHE_LEN * SPEAKERS) {
        *s += LATEST_BOOST;
    }
    boost(&mut scores, N_STRONG, -2.0 * 0.5f32.ln());
    boost(&mut scores, N_WEAK, -0.5f32.ln());

    // One extra frame per speaker holds the silence rows, scored +inf so they always survive.
    let scored = frames + SILENCE_PER_SPEAKER;
    let sentinel = scored * SPEAKERS;
    let mut flat: Vec<(f32, usize)> = Vec::with_capacity(scored * SPEAKERS);
    for s in 0..SPEAKERS {
        for f in 0..scored {
            let score = if f < frames {
                scores[f * SPEAKERS + s]
            } else {
                f32::INFINITY
            };
            flat.push((score, s * scored + f));
        }
    }
    flat.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut top: Vec<usize> = flat
        .iter()
        .take(CACHE_LEN)
        .map(|&(score, idx)| {
            if score == f32::NEG_INFINITY {
                sentinel
            } else {
                idx
            }
        })
        .collect();
    top.sort_unstable();
    top.into_iter()
        .map(|i| {
            if i == sentinel {
                frames
            } else {
                (i % scored).min(frames)
            }
        })
        .collect()
}

/// Compresses `embeds` (`frames * HIDDEN`) and `probs` (`frames * SPEAKERS`) to `CACHE_LEN`
/// rows. Returns the rows and the frame indices they came from (`frames` is the silence row).
pub fn compress(
    embeds: &[f32],
    probs: &[f32],
    silence: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    assert_eq!(silence.len(), HIDDEN);
    let frames = probs.len() / SPEAKERS;
    assert_eq!(embeds.len(), frames * HIDDEN);
    let picked = select_frames(probs);
    let mut e = Vec::with_capacity(picked.len() * HIDDEN);
    let mut p = Vec::with_capacity(picked.len() * SPEAKERS);
    for &f in &picked {
        if f == frames {
            e.extend_from_slice(silence);
            p.extend(std::iter::repeat_n(0.0, SPEAKERS));
        } else {
            e.extend_from_slice(&embeds[f * HIDDEN..(f + 1) * HIDDEN]);
            p.extend_from_slice(&probs[f * SPEAKERS..(f + 1) * SPEAKERS]);
        }
    }
    (e, p, picked)
}

/// The caller-owned state between chunks: the speaker cache and the FIFO of recent frames.
#[derive(Debug, Clone, Default)]
pub struct SpeakerCache {
    embeds: Vec<f32>,
    probs: Vec<f32>,
    fifo: Vec<f32>,
    compressed: bool,
}

impl SpeakerCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Rows in the cache.
    pub fn cache_len(&self) -> usize {
        self.probs.len() / SPEAKERS
    }

    /// Rows in the FIFO.
    pub fn fifo_len(&self) -> usize {
        self.fifo.len() / HIDDEN
    }

    /// Cache embeddings, `cache_len() * HIDDEN`.
    pub fn embeds(&self) -> &[f32] {
        &self.embeds
    }

    /// Probabilities stored beside the cache rows, `cache_len() * SPEAKERS`.
    pub fn probs(&self) -> &[f32] {
        &self.probs
    }

    /// FIFO embeddings, `fifo_len() * HIDDEN`.
    pub fn fifo(&self) -> &[f32] {
        &self.fifo
    }

    /// The `cached_embeds` input of the next graph call: cache rows, then FIFO rows.
    pub fn cached_embeds(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.embeds.len() + self.fifo.len());
        v.extend_from_slice(&self.embeds);
        v.extend_from_slice(&self.fifo);
        v
    }

    /// Takes in one processed step.
    ///
    /// - `chunk_embeds`: the `n_chunk` embedding rows of the chunk itself (`n_chunk * HIDDEN`),
    ///   i.e. the rows of the graph's `chunk_embeds` after the cached ones, without the right
    ///   context that follows them (those are fed again next step).
    /// - `logits`: the step's raw logits, `mel_frames * SPEAKERS`, covering the cached rows,
    ///   the chunk and its right context.
    /// - `valid`: one entry per encoder frame of the step, `false` for padding frames
    ///   (the attention mask: ones over the cached frames).
    /// - `silence`: the graph's silence embedding.
    pub fn update(
        &mut self,
        chunk_embeds: &[f32],
        logits: &[f32],
        valid: &[bool],
        silence: &[f32],
    ) {
        assert_eq!(chunk_embeds.len() % HIDDEN, 0);
        let n_chunk = chunk_embeds.len() / HIDDEN;
        let (nc, nf) = (self.cache_len(), self.fifo_len());
        let probs = pool_probs(logits, valid);
        let start = nc + nf;
        assert!(
            probs.len() >= (start + n_chunk) * SPEAKERS,
            "logits do not cover the cached rows and the chunk"
        );

        let mut fifo = std::mem::take(&mut self.fifo);
        fifo.extend_from_slice(chunk_embeds);
        let fifo_rows = fifo.len() / HIDDEN;
        if fifo_rows > FIFO {
            let popped = UPDATE_PERIOD.max(fifo_rows - FIFO).min(fifo_rows);
            let mut fifo_probs = probs[nc * SPEAKERS..start * SPEAKERS].to_vec();
            fifo_probs.extend_from_slice(&probs[start * SPEAKERS..(start + n_chunk) * SPEAKERS]);
            // An uncompressed cache still holds plain chunk frames, whose probabilities this step
            // re-estimates; a compressed one is out of order, so its stored probabilities are the
            // only ones that line up with its rows.
            let mut cache_probs = if self.compressed {
                std::mem::take(&mut self.probs)
            } else {
                probs[..nc * SPEAKERS].to_vec()
            };
            cache_probs.extend_from_slice(&fifo_probs[..popped * SPEAKERS]);
            let mut cache_embeds = std::mem::take(&mut self.embeds);
            cache_embeds.extend_from_slice(&fifo[..popped * HIDDEN]);
            fifo.drain(..popped * HIDDEN);

            if cache_embeds.len() / HIDDEN > CACHE_LEN {
                let (e, p, _) = compress(&cache_embeds, &cache_probs, silence);
                cache_embeds = e;
                cache_probs = p;
                self.compressed = true;
            }
            self.embeds = cache_embeds;
            self.probs = cache_probs;
        }
        self.fifo = fifo;
    }

    #[cfg(test)]
    fn from_parts(embeds: Vec<f32>, probs: Vec<f32>, fifo: Vec<f32>, compressed: bool) -> Self {
        Self {
            embeds,
            probs,
            fifo,
            compressed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP: [&[u8]; 4] = [
        include_bytes!("../testdata/nemotron/nemotron-cache-step0.bin"),
        include_bytes!("../testdata/nemotron/nemotron-cache-step1.bin"),
        include_bytes!("../testdata/nemotron/nemotron-cache-step2.bin"),
        include_bytes!("../testdata/nemotron/nemotron-cache-step3.bin"),
    ];
    const UPDATE: [&[u8]; 4] = [
        include_bytes!("../testdata/nemotron/nemotron-update-step0.bin"),
        include_bytes!("../testdata/nemotron/nemotron-update-step1.bin"),
        include_bytes!("../testdata/nemotron/nemotron-update-step2.bin"),
        include_bytes!("../testdata/nemotron/nemotron-update-step3.bin"),
    ];

    /// Embedding and FIFO rows are compared at 1e-3 and probabilities at 1e-4; the fixture notes
    /// record the actual differences as at most 3.7e-4 and 9.5e-7.
    const EMBED_TOL: f32 = 1e-3;
    const PROB_TOL: f32 = 1e-4;

    fn u32s(b: &[u8], n: usize) -> Vec<usize> {
        (0..n)
            .map(|i| u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()) as usize)
            .collect()
    }

    fn f32s(b: &[u8], count: usize) -> (Vec<f32>, &[u8]) {
        let (head, rest) = b.split_at(count * 4);
        (
            head.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            rest,
        )
    }

    struct Fixture {
        embeds: Vec<f32>,
        probs: Vec<f32>,
        fifo: Vec<f32>,
    }

    fn state(step: usize) -> Fixture {
        let h = u32s(STEP[step], 4);
        let (nc, nf) = (h[0], h[1]);
        assert_eq!((h[2], h[3]), (HIDDEN, SPEAKERS));
        let (embeds, rest) = f32s(&STEP[step][16..], nc * HIDDEN);
        let (probs, rest) = f32s(rest, nc * SPEAKERS);
        let (fifo, _) = f32s(rest, nf * HIDDEN);
        Fixture {
            embeds,
            probs,
            fifo,
        }
    }

    struct Input {
        chunk_embeds: Vec<f32>,
        logits: Vec<f32>,
        valid: Vec<bool>,
        silence: Vec<f32>,
    }

    fn input(step: usize) -> Input {
        let h = u32s(UPDATE[step], 6);
        let (n_chunk, n_in) = (h[2], h[3]);
        assert_eq!((h[4], h[5]), (HIDDEN, SPEAKERS));
        let (logits, rest) = f32s(&UPDATE[step][24..], n_in * STACK * SPEAKERS);
        let (mask, rest) = f32s(rest, n_in);
        let (chunk_embeds, rest) = f32s(rest, n_chunk * HIDDEN);
        let (silence, rest) = f32s(rest, HIDDEN);
        assert!(rest.is_empty());
        Input {
            chunk_embeds,
            logits,
            valid: mask.iter().map(|&m| m > 0.5).collect(),
            silence,
        }
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn check(cache: &SpeakerCache, want: &Fixture, step: usize) {
        assert_eq!(
            cache.cache_len(),
            want.probs.len() / SPEAKERS,
            "step {step} cache rows"
        );
        assert_eq!(
            cache.fifo_len(),
            want.fifo.len() / HIDDEN,
            "step {step} fifo rows"
        );
        // Equal embedding rows mean the same source frames were kept, in the same order.
        assert!(
            max_diff(cache.embeds(), &want.embeds) <= EMBED_TOL,
            "step {step} embeds"
        );
        assert!(
            max_diff(cache.probs(), &want.probs) <= PROB_TOL,
            "step {step} probs"
        );
        assert!(
            max_diff(cache.fifo(), &want.fifo) <= EMBED_TOL,
            "step {step} fifo"
        );
    }

    #[test]
    fn reproduces_the_reference_cache_after_every_step() {
        let mut cache = SpeakerCache::new();
        for step in 0..4 {
            let i = input(step);
            cache.update(&i.chunk_embeds, &i.logits, &i.valid, &i.silence);
            check(&cache, &state(step), step);
        }
    }

    #[test]
    fn each_step_matches_when_started_from_the_reference_state() {
        // isolates a step from drift accumulated in earlier ones
        for step in 1..4 {
            let prev = state(step - 1);
            let mut cache = SpeakerCache::from_parts(prev.embeds, prev.probs, prev.fifo, true);
            let i = input(step);
            cache.update(&i.chunk_embeds, &i.logits, &i.valid, &i.silence);
            check(&cache, &state(step), step);
        }
    }

    #[test]
    fn the_final_step_masks_padding_frames() {
        let i = input(3);
        assert!(
            i.valid.iter().any(|&v| !v),
            "fixture must exercise the mask"
        );
        let probs = pool_probs(&i.logits, &i.valid);
        let last_invalid = i.valid.iter().rposition(|&v| !v).unwrap();
        assert!(
            probs[last_invalid * SPEAKERS..(last_invalid + 1) * SPEAKERS]
                .iter()
                .all(|&p| p == 0.0)
        );
    }

    #[test]
    fn the_same_input_gives_the_same_cache() {
        let run = || {
            let mut cache = SpeakerCache::new();
            for step in 0..4 {
                let i = input(step);
                cache.update(&i.chunk_embeds, &i.logits, &i.valid, &i.silence);
            }
            cache
        };
        let (a, b) = (run(), run());
        assert_eq!(a.embeds(), b.embeds());
        assert_eq!(a.probs(), b.probs());
    }

    #[test]
    fn all_equal_scores_keep_the_lowest_speaker_major_index() {
        // Every frame is a weak but positive-scoring entry for every speaker: identical scores
        // everywhere, so the stable rule alone decides. 270 frames > CACHE_LEN.
        let frames = CACHE_LEN + 6;
        let probs = vec![0.9f32; frames * SPEAKERS];
        let picked = select_frames(&probs);
        assert_eq!(picked.len(), CACHE_LEN);
        let again = select_frames(&probs);
        assert_eq!(picked, again);
        let silence_rows = picked.iter().filter(|&&f| f == frames).count();
        assert_eq!(silence_rows, SPEAKERS, "one silence row per speaker");
    }

    #[test]
    fn boost_ties_go_to_the_earlier_frame() {
        let mut scores = vec![1.0f32; 5 * SPEAKERS];
        boost(&mut scores, 2, 10.0);
        for f in 0..5 {
            for s in 0..SPEAKERS {
                let want = if f < 2 { 11.0 } else { 1.0 };
                assert_eq!(scores[f * SPEAKERS + s], want, "frame {f} speaker {s}");
            }
        }
    }

    #[test]
    fn frame_scores_drop_silence_and_extra_weak_frames() {
        // speaker 0 only: frame 0 silent, frame 1 strongly speech, frame 2 weakly speech
        let mut probs = vec![0.0f32; 3 * SPEAKERS];
        probs[0] = 0.2;
        probs[SPEAKERS] = 0.95;
        probs[2 * SPEAKERS] = 0.51;
        probs[2 * SPEAKERS + 1] = 0.9; // overlap makes speaker 0's score negative
        let s = frame_scores(&probs);
        assert_eq!(s[0], f32::NEG_INFINITY, "below 0.5 is never kept");
        assert!(s[SPEAKERS] > 0.0);
        // too few positives to reach MIN_POS, so the weak frame keeps its (negative) score
        assert!(s[2 * SPEAKERS].is_finite() && s[2 * SPEAKERS] < 0.0);
        // once speaker 0 has MIN_POS positive frames the weak one goes to -inf
        let mut many = vec![0.0f32; (MIN_POS + 1) * SPEAKERS];
        for f in 0..MIN_POS {
            many[f * SPEAKERS] = 0.95;
        }
        many[MIN_POS * SPEAKERS] = 0.51;
        many[MIN_POS * SPEAKERS + 1] = 0.9;
        let s = frame_scores(&many);
        assert_eq!(s[MIN_POS * SPEAKERS], f32::NEG_INFINITY);
        assert!(s[0] > 0.0);
    }

    /// Small deterministic generator so the synthetic test needs no dependency and never flakes.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / (1u64 << 24) as f32
        }
    }

    #[test]
    fn synthetic_steps_keep_the_cache_and_fifo_bounded_and_silence_slots_filled() {
        let mut rng = Lcg(7);
        let silence = vec![123.0f32; HIDDEN];
        let mut cache = SpeakerCache::new();
        for step in 0..14 {
            let n_chunk = 100 + (step * 37) % 250;
            let n_in = cache.cache_len() + cache.fifo_len() + n_chunk;
            let logits: Vec<f32> = (0..n_in * STACK * SPEAKERS)
                .map(|_| (rng.next() - 0.5) * 8.0)
                .collect();
            let chunk: Vec<f32> = (0..n_chunk * HIDDEN).map(|_| rng.next()).collect();
            cache.update(&chunk, &logits, &vec![true; n_in], &silence);
            assert!(
                cache.cache_len() <= CACHE_LEN,
                "step {step} cache {}",
                cache.cache_len()
            );
            assert!(
                cache.fifo_len() <= FIFO.max(n_chunk),
                "step {step} fifo {}",
                cache.fifo_len()
            );
            assert!(
                cache.fifo_len() <= FIFO || cache.cache_len() == 0,
                "step {step} fifo over limit after a pop"
            );
            if cache.cache_len() == CACHE_LEN {
                let silent = cache
                    .embeds()
                    .as_chunks::<HIDDEN>()
                    .0
                    .iter()
                    .filter(|r| r.as_slice() == silence.as_slice())
                    .count();
                assert!(
                    silent >= SPEAKERS * SILENCE_PER_SPEAKER,
                    "step {step} silence rows {silent}"
                );
                for (row, p) in cache
                    .embeds()
                    .as_chunks::<HIDDEN>()
                    .0
                    .iter()
                    .zip(cache.probs().as_chunks::<SPEAKERS>().0.iter())
                {
                    if row.as_slice() == silence.as_slice() {
                        assert!(
                            p.iter().all(|&x| x == 0.0),
                            "silence rows carry zero probabilities"
                        );
                    }
                }
            }
        }
        assert_eq!(cache.cache_len(), CACHE_LEN);
    }

    #[test]
    fn module_has_no_onnx_dependency() {
        let src = include_str!("nemotron_cache.rs");
        let needle = ["use ", "ort"].concat();
        assert!(!src.contains(&needle));
    }
}
