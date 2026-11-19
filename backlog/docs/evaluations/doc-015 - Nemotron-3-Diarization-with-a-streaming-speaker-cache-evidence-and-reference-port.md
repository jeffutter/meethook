---
id: doc-015
title: >-
  Nemotron-3-Diarization with a streaming speaker cache: evidence and reference
  port
type: other
created_date: '2026-11-19T17:23:00Z'
updated_date: '2026-11-19T17:23:00Z'
---
This is the evidence and reference material behind the work to replace pyannote diarization with Nemotron-3-Diarization and its streaming speaker cache. It was gathered in an exploratory session, with throwaway scripts that no longer exist, so everything a worker needs is here. Read the caveats at the end before trusting any number.

## What was compared

Three diarizers over one real session: a 129 minute `speaker.wav` of a tabletop game with three named people on the speaker track (Discord audio, 48 kHz mono float).

1. **Pyannote (current).** `OnnxDiarizer`: segmentation-3.0 plus WeSpeaker plus constrained agglomerative clustering. 364 clusters on that track (reproduces the session's saved `speaker_clusters.json`).
2. **Nemotron in independent 30 s chunks, linked by WeSpeaker.** Each chunk's slots handed to `cluster_speaker_turns` as `(window = chunk, local_speaker = slot)`. 146 clusters; the top 3 hold 82% of speech (pyannote's top 3: 73%).
3. **Nemotron with the streaming speaker cache, one pass over the track.** 3 slots used in total, one per person. 91-99% of each person's speech sits in a single slot.

Agreement with the pipeline's own transcript names (on frames where both name one of the three people): route 2 97.2% over 3396 s, route 3 97.0% over 2616 s. **This is circular**: the names come from pyannote plus enrollment, so agreement shows the approaches agree, not that either is right. No ground truth exists yet; the evaluation ticket exists to get one.

Pyannote's small unnamed clusters (about 315 s) were mostly assigned by Nemotron to the three main people. The user judged route 3 clearly best from the comparison page (three lanes against the same audio) and asked for it to be implemented.

## The model and its ONNX export

- Base model: `nvidia/Nemotron-3-Diarization` (about 100M parameters, 31-layer RoPE transformer, hidden size 512, 8 speaker slots, 80 ms encoder frames). Weights license: OpenMDW-1.1 (the card says commercial and non-commercial use allowed; check before shipping).
- Export used: https://huggingface.co/onnx-community/Nemotron-3-Diarization-ONNX, folder `onnx/`. One graph per precision, each with an external data file that must sit beside it: `model_fp16.onnx` (352 kB) + `model_fp16.onnx_data` (199 MB) was used. Also fp32 (398 MB), q4 (83 MB), q4f16 (73 MB), `model_quantized` (120 MB). Quantized variants were not evaluated for accuracy. The repo's current diarization weights total about 32 MB.
- Graph contract (opset 21, uses `com.microsoft` contrib ops `RotaryEmbedding` and `SkipLayerNormalization`, which ONNX Runtime's CPU provider runs; whether the CoreML provider takes them was not established, it showed no speedup and no error):

  | | name | type | shape |
  |---|---|---|---|
  | in | `input_features` | f32 | `[batch, mel_frames, 128]` |
  | in | `cached_embeds` | f32 | `[batch, cached_frames, 512]` |
  | in | `attention_mask` | i64 | `[batch, cached_frames + encoder_frames]` |
  | out | `logits` | f32 | `[batch, (cached_frames + encoder_frames) * 8, 8]` |
  | out | `chunk_embeds` | f32 | `[batch, encoder_frames, 512]` |
  | out | `silence_embeds` | f32 | `[512]` |

  `encoder_frames = ceil(mel_frames / 8)`. The mask is all ones in offline use. `logits` are raw logits, not probabilities (observed range -16.8 to +4.9): apply a sigmoid. Output frames are 10 ms (8 per encoder frame); a speaker is active where the sigmoid exceeds the threshold. The graph is stateless: the caller owns the streaming cache.
- Position limit: `max_position_embeddings` is 5000 encoder frames (400 s) for cache + FIFO + chunk together. A single call over about 340 s ran, one over 400 s failed with "Updating cos_cache and sin_cache in RotaryEmbedding is not currently supported". Running a whole 300 s clip in one call with an empty cache was clearly worse than 30 s chunks on one clip (agreement with pyannote 0.80 vs 0.91, recall of pyannote's under-1 s turns 0.45 vs 0.86); the model card's DER figures are for 30.4 s and 0.32 s latency settings, not long single calls.
- Features (preprocessor_config.json, extractor type `NemotronAsrStreamingFeatureExtractor`): 16 kHz mono, 128 mel bins, n_fft 512, window 400 (Hann), hop 160, pre-emphasis 0.97, right padding. The reference port used Slaney-scale, Slaney-normalized mel (librosa `htk=False, norm="slaney"`), `log(power_mel + 2^-24)`, no normalization, zero (constant) centre padding. **None of that was verified numerically against the model's own extractor.** It produced sensible output, and a wrong recipe would degrade accuracy silently.
- The first encoder step is `Nemotron3DiarizationFeatureStacking`: it stacks 8 consecutive mel frames (zero-padding an incomplete last group) and projects them, so embeddings are local to groups of 8 mel frames and chunks aligned to 8 mel frames embed identically to a whole-track pass.

## The streaming algorithm (offline mode of the Hugging Face implementation)

Source of truth: Hugging Face transformers (Apache-2.0), `src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py`, class `Nemotron3DiarizationSpeakerCache` (methods `update`, `_num_popped_frames`, `_get_frame_scores`, `_boost_scores`, `_compress`, `_pool_probs`, `get_embeds`) and the chunk loop in `Nemotron3DiarizationForAudioFrameClassification.forward`. Fetch the current file, since only excerpts were read. Another implementation, https://github.com/warith-harchaoui/vocal-helper-rs (BSD-3-Clause), ports NVIDIA's NeMo code but targets a **different export** (inputs `chunk`/`spkcache`/`fifo`) and differs in its pop rule and cache-probability refresh; use it only as a cross-check.

Constants (the export's `config.json`; offline mode uses the top-level chunk values and `streaming_config` for the cache):

| constant | value |
|---|---|
| chunk length / right context / left context | 340 / 40 / 0 encoder frames (27.2 s + 3.2 s) |
| FIFO length | 40 |
| speaker cache update period | 300 |
| speaker cache length | 264 |
| silence frames per speaker | 1 |
| prediction score threshold / latest-frames boost | 0.25 / 0.05 |
| strong boost rate / weak boost rate / min positive rate | 0.75 / 1.5 / 0.5 |
| speakers / subsampling | 8 / 8 |

Per chunk: run the graph on the mel frames of `[chunk + right context]` with `cached_embeds = concat(cache, FIFO)`; slice the chunk's logits at `[(cached + 0) * 8, (cached + chunk_frames) * 8)`; then update the cache from the step's probabilities (sigmoid of the logits average-pooled by 8 per encoder frame). The FIFO accepts the chunk's embeddings; when it overflows, `max(update_period, overflow)` oldest frames move into the cache; when the cache exceeds its length it is compressed by per-speaker scores. `chunk_embeds` (the graph's output) are the embeddings that go into the FIFO.

The reference port below reproduces this and ran the whole 129 minute track on CPU in 149 s (Python, onnxruntime), with a steady 264-frame cache and 40-frame FIFO. It was checked for shapes and sensible output only.

## Post-processing found to work

Decode each slot's sigmoid probabilities: enter above 0.5, stay active until below 0.25 (hysteresis), merge gaps up to 1.0 s, drop turns under 0.5 s. On two 5-minute windows this cut raw turn counts from 83 to 46 (and 24 to 18) with no loss of agreement with pyannote's speech map (F1 about 0.84-0.93), while the threshold itself barely mattered between 0.25 and 0.5 (0.7 was worse). These came from one session and a metric against pyannote, so treat them as a starting point to re-measure. Overlapping speech comes out as simultaneous active slots.

## Where the transcript differences come from

Nemotron-only speech (speech with no transcript turn) is not a diarization finding: pyannote and the VAD cover it too, and `merge` gives unmatched text to the nearest turn. Pyannote's own raw turns cover about the same speech as Nemotron's (window 40-45 min: 198 s vs 169 s at a stricter threshold). Where text was lost is after the VAD and unverified (Whisper output on those regions is the suspect). Do not treat extra Nemotron speech as a reason to switch.

## Reference port (Python, numpy plus onnxruntime)

A numpy port of the Hugging Face offline mode that drives the onnx-community export. It is a reference for parity fixtures, not production code. It needs `numpy soundfile librosa onnxruntime scipy`.

```python
"""Nemotron-3-Diarization with the real streaming speaker cache, over the whole speaker track.

A numpy port of the Hugging Face transformers implementation's offline mode
(models/nemotron3_diarization/modeling_nemotron3_diarization.py:
Nemotron3DiarizationSpeakerCache + the chunk loop in ...ForAudioFrameClassification.forward),
driving the onnx-community ONNX export. Constants are the model's config.json.

usage: stream.py [max_chunks]   -> writes stream_probs.npy (float16, [frames at 10 ms, 8])
"""
import math, os, sys, time
import numpy as np, soundfile as sf, librosa, onnxruntime as ort
from scipy.signal import resample_poly

D = os.environ["D"]
WAV = os.path.expanduser("~/meethook/sessions/20261118-195036/speaker.wav")

# config.json (offline mode: top-level chunk/fifo/update-period, streaming_config for the cache).
CHUNK, RIGHT, LEFT = 340, 40, 0
FIFO, PERIOD = 40, 300
CACHE_LEN, SIL_PER_SPK, NSPK, SUB = 264, 1, 8, 8
SCORE_FLOOR, LATEST_BOOST = 0.25, 0.05
MIN_POS_RATE, STRONG_RATE, WEAK_RATE = 0.5, 0.75, 1.5
BUDGET = CACHE_LEN // NSPK - SIL_PER_SPK
MIN_POS, N_STRONG, N_WEAK = math.floor(BUDGET * MIN_POS_RATE), math.floor(BUDGET * STRONG_RATE), math.floor(BUDGET * WEAK_RATE)

sess = ort.InferenceSession(D + "/nemo/model_fp16.onnx", providers=["CPUExecutionProvider"])
fb = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm="slaney", htk=False)


def load_16k():
    """Whole track at 16 kHz mono, resampled in 5 min blocks."""
    out = []
    with sf.SoundFile(WAV) as f:
        while True:
            x = f.read(300 * f.samplerate, dtype="float32")
            if len(x) == 0:
                break
            out.append(resample_poly(x if x.ndim == 1 else x.mean(1), 16000, f.samplerate).astype(np.float32))
    return np.concatenate(out)


def log_mel(x, block=6000):
    """[frames, 128] log-mel, same framing as librosa.stft(center=True), computed in blocks."""
    x = np.concatenate([x[:1], x[1:] - 0.97 * x[:-1]])  # preemphasis
    n_frames = 1 + len(x) // 160
    xp = np.pad(x, (256, 256 + 512))
    out = np.empty((n_frames, 128), np.float32)
    for f0 in range(0, n_frames, block):
        nf = min(block, n_frames - f0)
        seg = xp[f0 * 160 : f0 * 160 + (nf - 1) * 160 + 512]
        S = np.abs(librosa.stft(seg, n_fft=512, win_length=400, hop_length=160, window="hann", center=False)) ** 2
        out[f0 : f0 + nf] = np.log(fb @ S + 2.0**-24).T[:nf]
    return out


def frame_scores(probs):
    """Port of _get_frame_scores. probs [frames, 8] -> scores [frames, 8]."""
    lp = np.log(np.clip(probs, SCORE_FLOOR, None))
    lc = np.log(np.clip(1.0 - probs, SCORE_FLOOR, None))
    scores = lp - lc + lc.sum(-1, keepdims=True) - math.log(0.5)
    is_speech = probs > 0.5
    scores = np.where(is_speech, scores, -np.inf)
    is_pos = scores > 0
    enough = is_pos.sum(0, keepdims=True) >= MIN_POS
    return np.where(~is_pos & is_speech & enough, -np.inf, scores)


def boost(scores, n, amount):
    """Port of _boost_scores: add `amount` to each speaker's top-n frames."""
    if n <= 0:
        return scores
    idx = np.argsort(-scores, axis=0, kind="stable")[:n]  # [n, 8]
    for s in range(scores.shape[1]):
        scores[idx[:, s], s] += amount
    return scores


def compress(embeds, probs, silence):
    """Port of _compress. embeds [F,512], probs [F,8] -> (embeds, probs) of CACHE_LEN rows."""
    F = len(embeds)
    scores = frame_scores(probs)
    scores[CACHE_LEN:] += LATEST_BOOST
    scores = boost(scores, N_STRONG, -2.0 * math.log(0.5))
    scores = boost(scores, N_WEAK, -math.log(0.5))
    scores = np.concatenate([scores, np.full((SIL_PER_SPK, NSPK), np.inf)], 0)
    embeds = np.concatenate([embeds, silence[None]], 0)
    probs = np.concatenate([probs, np.zeros((1, NSPK), probs.dtype)], 0)
    scored = F + SIL_PER_SPK
    sentinel = scored * NSPK
    flat = scores.T.reshape(-1)  # speaker-major, like transpose(1,2).reshape
    top = np.argsort(-flat, kind="stable")[:CACHE_LEN]
    top = np.where(np.isneginf(flat[top]), sentinel, top)
    top = np.sort(top)
    frame = np.where(top == sentinel, F, np.minimum(top % scored, F))
    return embeds[frame], probs[frame]


class Cache:
    def __init__(self):
        self.embeds = np.zeros((0, 512), np.float32)
        self.probs = np.zeros((0, NSPK), np.float32)
        self.fifo = np.zeros((0, 512), np.float32)
        self.compressed = False

    def cached(self):
        return np.concatenate([self.embeds, self.fifo], 0)

    def update(self, input_embeds, logits, silence, n_chunk):
        """Port of Nemotron3DiarizationSpeakerCache.update (no look-back frames)."""
        probs = 1 / (1 + np.exp(-logits))
        probs = probs[: len(probs) // SUB * SUB].reshape(-1, SUB, NSPK).mean(1)  # avg_pool1d over 8
        nc, nf = len(self.embeds), len(self.fifo)
        start = nc + nf
        fifo_embeds = np.concatenate([self.fifo, input_embeds[start : start + n_chunk]], 0)
        if len(fifo_embeds) > FIFO:
            popped = min(max(PERIOD, len(fifo_embeds) - FIFO), len(fifo_embeds))
            fifo_probs = np.concatenate([probs[nc : nc + nf], probs[start : start + n_chunk]], 0)
            stored = self.probs[:nc] if self.compressed else probs[:nc]
            cache_embeds = np.concatenate([self.embeds, fifo_embeds[:popped]], 0)
            cache_probs = np.concatenate([stored, fifo_probs[:popped]], 0)
            fifo_embeds = fifo_embeds[popped:]
            if len(cache_embeds) > CACHE_LEN:
                cache_embeds, cache_probs = compress(cache_embeds, cache_probs, silence)
                self.compressed = True
            self.embeds, self.probs = cache_embeds.astype(np.float32), cache_probs.astype(np.float32)
        self.fifo = fifo_embeds.astype(np.float32)


def main(max_chunks=None):
    t0 = time.time()
    x = load_16k()
    mel = log_mel(x)
    n_mel = len(mel)
    n_enc = -(-n_mel // SUB)
    print(f"audio {len(x)/16000/60:.1f} min, mel frames {n_mel}, encoder frames {n_enc}  ({time.time()-t0:.0f}s)", flush=True)

    cache = Cache()
    out = []
    for step, start in enumerate(range(0, n_enc, CHUNK)):
        if max_chunks and step >= max_chunks:
            break
        end = min(start + CHUNK, n_enc)
        n_chunk = end - start
        ctx_end = min(end + RIGHT, n_enc)
        feats = mel[start * SUB : ctx_end * SUB]
        cached = cache.cached()
        enc_in = -(-len(feats) // SUB)
        logits, chunk_embeds, silence = sess.run(None, {
            "input_features": feats[None],
            "cached_embeds": cached[None].astype(np.float32),
            "attention_mask": np.ones((1, len(cached) + enc_in), np.int64),
        })
        logits, chunk_embeds = logits[0], chunk_embeds[0]
        input_embeds = np.concatenate([cached, chunk_embeds], 0)
        out.append(logits[len(cached) * SUB : (len(cached) + n_chunk) * SUB])
        cache.update(input_embeds, logits, silence, n_chunk)
        if step % 5 == 0:
            print(f"step {step}: cache {len(cache.embeds)} fifo {len(cache.fifo)} chunk {n_chunk} "
                  f"logits {logits.shape} ({time.time()-t0:.0f}s)", flush=True)

    L = np.concatenate(out, 0)[:n_mel]
    P = (1 / (1 + np.exp(-L))).astype(np.float16)
    np.save(D + "/stream_probs.npy", P)
    print("wrote", P.shape, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
```

## Caveats (read before trusting the numbers)

- **Not independently verified.** All agreement figures use names from the pipeline's own transcript. Ground truth has to come from a person listening, in the evaluation ticket.
- **One easy recording.** Three speakers, one session. The cache's compression only matters with many speakers, people joining late, or more than 8 speakers (the model has 8 slots; what happens beyond that is untested). An earlier decision record mentions a 43 minute, 7-person session, and `asr.rs` comments mention session `20260810-093047`; neither was on the disk this was done on, so a many-speaker recording must be found or recorded.
- **The port was not compared numerically** with the Hugging Face reference; only its shapes and its behaviour were checked. The feature recipe was not compared with the model's own extractor either. Parity fixtures from the reference are the first ticket for that reason.
- **Speed and size are unmeasured in Rust.** 149 s for 129 minutes was Python on CPU. The fp16 weights are about six times today's diarization weights. CoreML behaviour is unknown.
- **Speaker identity across sessions still needs voice embeddings.** Nemotron says who spoke when within a recording; it produces no fingerprint. `speaker_clusters.json`, `enroll` and identification all run on a per-cluster WeSpeaker embedding, so each slot's turns must still be embedded and averaged, exactly as `SpeakerCluster` documents.
- **Window comparisons so far** covered only 2 five-minute clips plus whole-track aggregates; the throwaway comparison page and scripts are gone.
