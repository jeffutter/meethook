//! One call through the Nemotron-3-Diarization ONNX graph.
//!
//! The graph is stateless: it is given the mel frames of a chunk (plus its right context) and
//! the encoder frames cached so far, and returns logits for cached and chunk frames together,
//! the chunk's own embeddings, and the silence embedding. What to cache afterwards is
//! [`crate::nemotron_cache`]'s business and looping over a track is TASK-088.06's; this
//! module is only the contract with the graph, which doc-015 read from the graph and observed
//! running rather than from documentation, so the weights-gated tests here pin it.

use std::path::Path;

use ort::session::Session;
use ort::value::TensorRef;

use crate::nemotron_cache::config::{CACHE_LEN, CHUNK, FIFO, HIDDEN, RIGHT, SPEAKERS};
use crate::nemotron_mel::{MEL_BINS, STACK};
use crate::onnx::open_session;
use crate::{Error, NEMOTRON_DIARIZATION_DATA, NEMOTRON_DIARIZATION_GRAPH, Result};

/// `max_position_embeddings` of the graph: cache + FIFO + chunk encoder frames per call.
/// One more and ONNX Runtime fails with an obscure RotaryEmbedding cache message.
pub const MAX_POSITIONS: usize = 5000;

/// A loaded Nemotron graph.
pub struct NemotronGraph {
    session: Session,
    /// False when the whole graph is running on CPU because CoreML would not take it.
    pub accelerated: bool,
}

/// What one step returns.
pub struct StepOutput {
    /// Raw logits, `(cached + encoder frames) * STACK` rows of `SPEAKERS`.
    pub logits: Vec<f32>,
    /// Embeddings of the step's own encoder frames, `encoder frames * HIDDEN`; the cached rows
    /// are not echoed back.
    pub chunk_embeds: Vec<f32>,
    /// The graph's silence embedding, `HIDDEN` values.
    pub silence: Vec<f32>,
}

/// Encoder frames for `mel_frames` mel frames (each encoder frame stacks [`STACK`]).
pub fn encoder_frames(mel_frames: usize) -> usize {
    mel_frames.div_ceil(STACK)
}

/// The attention mask of a step's own encoder frames, as Hugging Face builds it: encoder frame
/// `f` is valid when its first mel frame, `f * STACK`, is one of the `valid_mel_frames` frames
/// that carry signal. Without it the all-zero trailing mel frame of a track attends like
/// speech and changes the last chunk's probabilities.
pub fn encoder_valid(valid_mel_frames: usize, mel_frames: usize) -> Vec<bool> {
    (0..encoder_frames(mel_frames))
        .map(|f| f * STACK < valid_mel_frames)
        .collect()
}

/// Checks the sizes of a step's inputs and the position limit, returning the encoder frame
/// count. Separate from [`NemotronGraph::step`] so it needs no session.
fn check_step(features: &[f32], cached: &[f32], valid: &[bool]) -> Result<usize> {
    if features.is_empty() || !features.len().is_multiple_of(MEL_BINS) {
        return Err(Error::Nemotron(format!(
            "features hold {} values, not a non-zero multiple of {MEL_BINS} mel bins",
            features.len()
        )));
    }
    if !cached.len().is_multiple_of(HIDDEN) {
        return Err(Error::Nemotron(format!(
            "cached embeddings hold {} values, not a multiple of {HIDDEN}",
            cached.len()
        )));
    }
    let enc = encoder_frames(features.len() / MEL_BINS);
    if valid.len() != enc {
        return Err(Error::Nemotron(format!(
            "mask has {} entries for {enc} encoder frames",
            valid.len()
        )));
    }
    let cached_frames = cached.len() / HIDDEN;
    if cached_frames + enc > MAX_POSITIONS {
        return Err(Error::NemotronPositionLimit {
            cached: cached_frames,
            chunk: enc,
            limit: MAX_POSITIONS,
            cache_len: CACHE_LEN,
            fifo: FIFO,
            chunk_len: CHUNK,
            right: RIGHT,
        });
    }
    Ok(enc)
}

/// The `attention_mask` input: ones over the cached frames, then the chunk's own mask.
fn attention_mask(cached_frames: usize, valid: &[bool]) -> Vec<i64> {
    let mut mask = vec![1i64; cached_frames];
    mask.extend(valid.iter().map(|&v| i64::from(v)));
    mask
}

fn inference_failed(source: ort::Error) -> Error {
    Error::Nemotron(source.to_string())
}

impl NemotronGraph {
    /// Loads the graph from `models_dir`, the directory the registry installs into, trying
    /// CoreML first.
    pub fn open(models_dir: &Path) -> Result<Self> {
        let graph = models_dir.join(NEMOTRON_DIARIZATION_GRAPH.file_name);
        let data = models_dir.join(NEMOTRON_DIARIZATION_DATA.file_name);
        // ONNX Runtime resolves the external data beside the graph by name and reports a
        // missing file as a bare parse failure, so say which file and what to run.
        for path in [&graph, &data] {
            if !path.is_file() {
                return Err(Error::Nemotron(format!(
                    "{} is not installed; run `cargo run --example fetch-onnx-models`",
                    path.display()
                )));
            }
        }
        let loaded = open_session(&graph)?;
        Ok(Self {
            session: loaded.session,
            accelerated: loaded.accelerated,
        })
    }

    /// As [`NemotronGraph::open`] but CPU only, for comparing against the accelerated path.
    pub fn open_cpu(models_dir: &Path) -> Result<Self> {
        let graph = models_dir.join(NEMOTRON_DIARIZATION_GRAPH.file_name);
        if !graph.is_file() {
            return Err(Error::Nemotron(format!(
                "{} is not installed; run `cargo run --example fetch-onnx-models`",
                graph.display()
            )));
        }
        Ok(Self {
            session: crate::onnx::open_cpu_session(&graph)?,
            accelerated: false,
        })
    }

    /// Runs one step.
    ///
    /// - `features`: `mel_frames * MEL_BINS` log-mel values of the chunk and its right context.
    /// - `cached`: `cached_frames * HIDDEN`, the cache rows then the FIFO rows
    ///   (`crate::nemotron_cache::SpeakerCache::cached_embeds`).
    /// - `valid`: one entry per encoder frame of `features` ([`encoder_valid`]).
    pub fn step(&mut self, features: &[f32], cached: &[f32], valid: &[bool]) -> Result<StepOutput> {
        let enc = check_step(features, cached, valid)?;
        let mel_frames = features.len() / MEL_BINS;
        let cached_frames = cached.len() / HIDDEN;
        let mask = attention_mask(cached_frames, valid);

        let input_features = TensorRef::from_array_view(([1usize, mel_frames, MEL_BINS], features))
            .map_err(inference_failed)?;
        let cached_embeds = TensorRef::from_array_view(([1usize, cached_frames, HIDDEN], cached))
            .map_err(inference_failed)?;
        let attention = TensorRef::from_array_view(([1usize, cached_frames + enc], &mask[..]))
            .map_err(inference_failed)?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input_features" => input_features,
                "cached_embeds" => cached_embeds,
                "attention_mask" => attention,
            ])
            .map_err(inference_failed)?;

        let take = |name: &str, want: usize| -> Result<Vec<f32>> {
            let (_, data) = outputs[name]
                .try_extract_tensor::<f32>()
                .map_err(inference_failed)?;
            if data.len() != want {
                return Err(Error::Nemotron(format!(
                    "output `{name}` holds {} values, expected {want}",
                    data.len()
                )));
            }
            Ok(data.to_vec())
        };
        Ok(StepOutput {
            logits: take("logits", (cached_frames + enc) * STACK * SPEAKERS)?,
            chunk_embeds: take("chunk_embeds", enc * HIDDEN)?,
            silence: take("silence_embeds", HIDDEN)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ort::value::{TensorElementType, ValueType};

    use super::*;
    use crate::nemotron_mel::NemotronMel;
    use crate::nemotron_mel::tests::clip;

    fn models_dir() -> Option<PathBuf> {
        let root = match std::env::var_os("MEETHOOK_ROOT") {
            Some(root) => PathBuf::from(root),
            None => std::env::home_dir()?.join("meethook"),
        };
        Some(root.join("models"))
    }

    /// The installed graph, or `None` (with a note) when the weights are absent.
    fn open_if_installed() -> Option<NemotronGraph> {
        let dir = models_dir()?;
        let all_there = crate::NEMOTRON_DIARIZATION_MODELS
            .iter()
            .all(|s| dir.join(s.file_name).is_file());
        if !all_there {
            eprintln!("skipping: Nemotron pair not installed; run fetch-onnx-models");
            return None;
        }
        let graph = NemotronGraph::open(&dir).expect("the installed Nemotron pair must load");
        if !graph.accelerated {
            eprintln!("Nemotron: CoreML declined this graph; running on CPU");
        }
        Some(graph)
    }

    // ---- no weights needed ----

    #[test]
    fn position_limit_names_the_limit_and_not_ort() {
        let features = vec![0.0; 8 * MEL_BINS];
        let cached = vec![0.0; MAX_POSITIONS * HIDDEN];
        let err = check_step(&features, &cached, &[true]).unwrap_err();
        let msg = err.to_string();
        assert!(matches!(err, Error::NemotronPositionLimit { .. }));
        assert!(msg.contains("5000 encoder frames (400 s)"), "{msg}");
        assert!(!msg.contains("RotaryEmbedding"), "{msg}");

        // Exactly at the limit is allowed.
        let cached = vec![0.0; (MAX_POSITIONS - 1) * HIDDEN];
        assert_eq!(check_step(&features, &cached, &[true]).unwrap(), 1);
    }

    #[test]
    fn mismatched_inputs_are_errors_not_panics() {
        let ok = vec![0.0; 16 * MEL_BINS];
        assert!(check_step(&[], &[], &[]).is_err());
        assert!(check_step(&ok[..MEL_BINS + 1], &[], &[true]).is_err());
        assert!(check_step(&ok, &[0.0; HIDDEN + 1], &[true, true]).is_err());
        assert!(check_step(&ok, &[], &[true]).is_err());
        assert_eq!(check_step(&ok, &[], &[true, true]).unwrap(), 2);
    }

    #[test]
    fn the_mask_is_ones_for_cache_then_the_chunk_validity() {
        assert_eq!(attention_mask(2, &[true, false]), vec![1, 1, 1, 0]);
        // 10001 mel frames of which 10000 carry signal: the last encoder frame is masked.
        let v = encoder_valid(10_000, 10_001);
        assert_eq!(v.len(), 1251);
        assert!(v[..1250].iter().all(|&b| b));
        assert!(!v[1250]);
    }

    // ---- weights-gated ----

    #[test]
    fn the_graph_contract_matches_doc_015() {
        let Some(graph) = open_if_installed() else {
            return;
        };
        let check =
            |outlet: &ort::value::Outlet, name: &str, ty: TensorElementType, rank: usize| {
                assert_eq!(outlet.name(), name);
                let ValueType::Tensor { ty: got, shape, .. } = outlet.dtype() else {
                    panic!("{name} is not a tensor: {:?}", outlet.dtype());
                };
                assert_eq!(*got, ty, "{name} element type");
                assert_eq!(shape.len(), rank, "{name} rank");
                shape.to_vec()
            };
        let inputs = graph.session.inputs();
        let outputs = graph.session.outputs();
        assert_eq!(inputs.len(), 3);
        assert_eq!(outputs.len(), 3);
        fn by_name<'a>(list: &'a [ort::value::Outlet], name: &str) -> &'a ort::value::Outlet {
            list.iter()
                .find(|o| o.name() == name)
                .unwrap_or_else(|| panic!("no `{name}`"))
        }
        let f = TensorElementType::Float32;
        let i = check(by_name(inputs, "input_features"), "input_features", f, 3);
        assert_eq!(i[2], MEL_BINS as i64);
        let c = check(by_name(inputs, "cached_embeds"), "cached_embeds", f, 3);
        assert_eq!(c[2], HIDDEN as i64);
        check(
            by_name(inputs, "attention_mask"),
            "attention_mask",
            TensorElementType::Int64,
            2,
        );
        let l = check(by_name(outputs, "logits"), "logits", f, 3);
        assert_eq!(l[2], SPEAKERS as i64);
        let e = check(by_name(outputs, "chunk_embeds"), "chunk_embeds", f, 3);
        assert_eq!(e[2], HIDDEN as i64);
        let s = check(by_name(outputs, "silence_embeds"), "silence_embeds", f, 1);
        assert_eq!(s[0], HIDDEN as i64);
    }

    const UPDATE0: &[u8] = include_bytes!("../testdata/nemotron/nemotron-update-step0.bin");
    const UPDATE1: &[u8] = include_bytes!("../testdata/nemotron/nemotron-update-step1.bin");
    const CACHE0: &[u8] = include_bytes!("../testdata/nemotron/nemotron-cache-step0.bin");
    const PROBS: &[u8] = include_bytes!("../testdata/nemotron/nemotron-probs.bin");

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    fn u32_at(b: &[u8], i: usize) -> usize {
        u32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()) as usize
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "length");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    /// Speaker-activity decisions (probability above 0.5) that differ from the reference, and
    /// how many of those the reference itself was confident about (more than PROB_TOL from 0.5).
    fn flips(got: &[f32], want: &[f32]) -> (usize, usize) {
        let flipped: Vec<f32> = got
            .iter()
            .zip(want)
            .filter(|(a, b)| (**a > 0.5) != (**b > 0.5))
            .map(|(_, b)| *b)
            .collect();
        let confident = flipped
            .iter()
            .filter(|b| (**b - 0.5).abs() > PROB_TOL)
            .count();
        (flipped.len(), confident)
    }

    /// Step 0 (empty cache) and step 1 (cache + FIFO from the step-0 fixture) of the fixture
    /// clip, with the reference logits and chunk embeddings of each.
    #[test]
    fn steps_return_logits_for_cached_plus_encoder_frames_and_match_the_reference() {
        let Some(mut graph) = open_if_installed() else {
            return;
        };
        let samples = clip();
        let mel = NemotronMel::new().compute_all(&samples);

        // Step 0: chunk 340 + right 40 encoder frames from the start, no cache.
        let mel0 = (CHUNK + RIGHT) * STACK;
        let valid0 = vec![true; CHUNK + RIGHT];
        let out0 = graph.step(&mel[..mel0 * MEL_BINS], &[], &valid0).unwrap();
        assert_eq!(out0.logits.len(), (CHUNK + RIGHT) * STACK * SPEAKERS);

        // Reference logits and chunk embeddings from the HF update fixture.
        let n_in = u32_at(UPDATE0, 3);
        assert_eq!(n_in, CHUNK + RIGHT);
        let want_logits = f32s(&UPDATE0[24..24 + n_in * STACK * SPEAKERS * 4]);
        let embeds_at = 24 + n_in * STACK * SPEAKERS * 4 + n_in * 4;
        let want_embeds = f32s(&UPDATE0[embeds_at..embeds_at + CHUNK * HIDDEN * 4]);
        let logit_diff = max_diff(&out0.logits, &want_logits);
        let embed_diff = max_diff(&out0.chunk_embeds[..CHUNK * HIDDEN], &want_embeds);
        eprintln!("step 0: max abs logit diff {logit_diff}, chunk embed diff {embed_diff}");

        // Probabilities for the chunk's own mel frames against nemotron-probs.bin.
        let probs = f32s(&PROBS[8..]);
        let rows = CHUNK * STACK;
        let got: Vec<f32> = out0.logits[..rows * SPEAKERS]
            .iter()
            .map(|&x| sigmoid(x))
            .collect();
        let prob_diff = max_diff(&got, &probs[..rows * SPEAKERS]);
        let (flips0, confident0) = flips(&got, &probs[..rows * SPEAKERS]);
        eprintln!(
            "step 0: max abs probability diff {prob_diff}, 0.5 flips {flips0} of {} \
             ({confident0} where the reference is confident)",
            got.len()
        );

        // Step 1: the step-0 cache rows then FIFO rows as cached_embeds, next chunk's mel.
        let (n_cache, n_fifo) = (u32_at(CACHE0, 0), u32_at(CACHE0, 1));
        let cached = f32s(&CACHE0[16..16 + n_cache * HIDDEN * 4]);
        let fifo_at = 16 + n_cache * HIDDEN * 4 + n_cache * SPEAKERS * 4;
        let fifo = f32s(&CACHE0[fifo_at..fifo_at + n_fifo * HIDDEN * 4]);
        let cached: Vec<f32> = cached.into_iter().chain(fifo).collect();
        let cached_frames = n_cache + n_fifo;
        assert_eq!(cached_frames, 304);
        let first = CHUNK * STACK;
        let mel1 = &mel[first * MEL_BINS..(first + (CHUNK + RIGHT) * STACK) * MEL_BINS];
        let out1 = graph.step(mel1, &cached, &valid0).unwrap();
        assert_eq!(
            out1.logits.len(),
            (cached_frames + CHUNK + RIGHT) * STACK * SPEAKERS
        );
        let n_in1 = u32_at(UPDATE1, 3);
        assert_eq!(n_in1, cached_frames + CHUNK + RIGHT);
        let want1 = f32s(&UPDATE1[24..24 + n_in1 * STACK * SPEAKERS * 4]);
        let logit_diff1 = max_diff(&out1.logits, &want1);
        let got1: Vec<f32> = out1.logits.iter().map(|&x| sigmoid(x)).collect();
        let want1_probs: Vec<f32> = want1.iter().map(|&x| sigmoid(x)).collect();
        let prob_diff1 = max_diff(&got1, &want1_probs);
        let (flips1, confident1) = flips(&got1, &want1_probs);
        eprintln!(
            "step 1: max abs logit diff {logit_diff1}, probability diff {prob_diff1}, \
             0.5 flips {flips1} of {} ({confident1} where the reference is confident)",
            got1.len()
        );

        // Tolerances set from the measured differences of the fp16 graph against the fp32
        // Hugging Face reference (see the TASK-088.05 notes for the numbers).
        assert!(prob_diff <= PROB_TOL, "probability diff {prob_diff}");
        assert_eq!(
            confident0, 0,
            "0.5 threshold flips the reference is confident about"
        );
        assert!(embed_diff <= EMBED_TOL, "chunk embed diff {embed_diff}");
        assert!(logit_diff1 <= LOGIT_TOL, "step 1 logit diff {logit_diff1}");
        assert!(
            prob_diff1 <= PROB_TOL,
            "step 1 probability diff {prob_diff1}"
        );
        assert_eq!(
            confident1, 0,
            "step 1 flips the reference is confident about"
        );
    }

    // fp16 graph against the fp32 reference. ONNX Runtime 1.27: logits 0.020 (both steps),
    // chunk embeddings 0.031, probabilities 1.6e-3. 1.31 (the static macOS build): logits
    // 0.033 and 0.068, embeddings 0.065, probabilities 2.4e-3, one flip of a frame within
    // 2.3e-3 of 0.5. The fp32 fixtures' own 1e-4 does not apply to fp16 weights.
    const PROB_TOL: f32 = 5e-3;
    const EMBED_TOL: f32 = 0.1;
    const LOGIT_TOL: f32 = 0.1;

    #[test]
    fn a_step_over_5000_frames_returns_our_error() {
        let Some(mut graph) = open_if_installed() else {
            return;
        };
        let cached = vec![0.0; MAX_POSITIONS * HIDDEN];
        let err = graph
            .step(&vec![0.0; 8 * MEL_BINS], &cached, &[true])
            .err()
            .expect("5001 frames must fail");
        assert!(matches!(err, Error::NemotronPositionLimit { .. }), "{err}");
    }
}
