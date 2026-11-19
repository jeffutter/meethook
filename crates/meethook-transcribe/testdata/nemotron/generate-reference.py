"""Stage 2: Hugging Face reference fixtures + parity check of the doc-015 numpy port.

usage: generate-reference.py OUT_DIR
  reads  OUT_DIR/nemotron-clip.mulaw   (written by gen_clip.py)
  writes OUT_DIR/nemotron-features.bin, nemotron-cache-step{0..3}.bin, nemotron-probs.bin
  prints the parity report (HF torch vs ONNX graph vs the doc-015 numpy port).
Nothing here runs from cargo; it only documents how the fixtures were made.
"""
import math, sys
import numpy as np, torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
from transformers.models.nemotron3_diarization import modeling_nemotron3_diarization as M

OUT = sys.argv[1]
torch.manual_seed(0)
torch.set_num_threads(1)  # single thread keeps the float32 sums in a fixed order

# --- clip: decode mu-law to int16, then float32 in [-1, 1) exactly as soundfile would give for a 16-bit WAV
code = np.fromfile(f"{OUT}/nemotron-clip.mulaw", np.uint8)
c = (~code.astype(np.int32)) & 0xFF
sign, exp, mant = c & 0x80, (c >> 4) & 7, c & 0x0F
x16 = np.where(sign != 0, -((((mant << 3) + 0x84) << exp) - 0x84), (((mant << 3) + 0x84) << exp) - 0x84).astype(np.int16)
audio = (x16.astype(np.float32) / 32768.0)

NVIDIA = "nvidia/Nemotron-3-Diarization"
ONNX_REPO = "onnx-community/Nemotron-3-Diarization-ONNX"
proc = AutoProcessor.from_pretrained(NVIDIA)
model = AutoModelForAudioFrameClassification.from_pretrained(NVIDIA, dtype=torch.float32).eval()

# --- features (offline, center=True, the model's own extractor)
inputs = proc(audio, sampling_rate=16000)
feats = inputs["input_features"][0].numpy().astype(np.float32)  # [frames, 128]
n_mel = feats.shape[0]
print("mel frames", n_mel, "mask sum", int(inputs["attention_mask"].sum()))
HEAD, TAIL = 64, 64
with open(f"{OUT}/nemotron-features.bin", "wb") as f:
    np.array([n_mel, 128, HEAD, TAIL], "<u4").tofile(f)
    feats[:HEAD].astype("<f4").tofile(f)
    feats[n_mel - TAIL:].astype("<f4").tofile(f)

# --- HF offline forward, recording the cache after each update
states = []
orig_update = M.Nemotron3DiarizationSpeakerCache.update
def recording_update(self, *a, **k):
    orig_update(self, *a, **k)
    nc, nf = self.num_cache_frames, self.num_fifo_frames
    states.append(dict(
        embeds=self.embeds[0, :nc].clone().numpy(), probs=self.probs[0, :nc].clone().numpy(),
        fifo=self.fifo[0, :nf].clone().numpy(), compressed=self.is_compressed))
M.Nemotron3DiarizationSpeakerCache.update = recording_update
with torch.no_grad():
    logits_hf = model(**inputs).logits[0].numpy()  # [mel frames, 8]
M.Nemotron3DiarizationSpeakerCache.update = orig_update
probs_hf = (1 / (1 + np.exp(-logits_hf.astype(np.float64)))).astype(np.float32)
print("HF steps", len(states), "logits", logits_hf.shape)
for i, s in enumerate(states):
    print(f"  step {i}: cache {s['embeds'].shape} probs {s['probs'].shape} fifo {s['fifo'].shape} compressed {s['compressed']}")
for i in range(4):
    s = states[i]
    with open(f"{OUT}/nemotron-cache-step{i}.bin", "wb") as f:
        np.array([len(s["embeds"]), len(s["fifo"]), 512, 8], "<u4").tofile(f)
        s["embeds"].astype("<f4").tofile(f)
        s["probs"].astype("<f4").tofile(f)
        s["fifo"].astype("<f4").tofile(f)
with open(f"{OUT}/nemotron-probs.bin", "wb") as f:
    np.array([n_mel, 8], "<u4").tofile(f)
    probs_hf.astype("<f4").tofile(f)

# --- cross-check 1: the ONNX graph (fp32) driven by the doc-015 numpy port, on the same clip
import onnxruntime as ort
import librosa

onnx_dir = snapshot_download(ONNX_REPO, allow_patterns=["onnx/model.onnx*"])
sess = ort.InferenceSession(onnx_dir + "/onnx/model.onnx", providers=["CPUExecutionProvider"])

CHUNK, RIGHT = 340, 40
FIFO, PERIOD = 40, 300
CACHE_LEN, SIL_PER_SPK, NSPK, SUB = 264, 1, 8, 8
SCORE_FLOOR, LATEST_BOOST = 0.25, 0.05
MIN_POS_RATE, STRONG_RATE, WEAK_RATE = 0.5, 0.75, 1.5
BUDGET = CACHE_LEN // NSPK - SIL_PER_SPK
MIN_POS, N_STRONG, N_WEAK = math.floor(BUDGET * MIN_POS_RATE), math.floor(BUDGET * STRONG_RATE), math.floor(BUDGET * WEAK_RATE)
fb = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm="slaney", htk=False)


def log_mel_port(x, block=6000, window="hann", mask_last=False):
    """doc-015 port; `window` and `mask_last` select the HF-faithful variants (default = verbatim doc-015)."""
    x = np.concatenate([x[:1], x[1:] - 0.97 * x[:-1]])
    n_frames = 1 + len(x) // 160
    xp = np.pad(x, (256, 256 + 512))
    out = np.empty((n_frames, 128), np.float32)
    for f0 in range(0, n_frames, block):
        nf = min(block, n_frames - f0)
        seg = xp[f0 * 160 : f0 * 160 + (nf - 1) * 160 + 512]
        S = np.abs(librosa.stft(seg, n_fft=512, win_length=400, hop_length=160, window=window, center=False)) ** 2
        out[f0 : f0 + nf] = np.log(fb @ S + 2.0**-24).T[:nf]
    if mask_last:
        out[len(x) // 160 :] = 0.0  # HF zeroes frames past floor(L / hop)
    return out


def frame_scores(probs):
    lp = np.log(np.clip(probs, SCORE_FLOOR, None))
    lc = np.log(np.clip(1.0 - probs, SCORE_FLOOR, None))
    scores = lp - lc + lc.sum(-1, keepdims=True) - math.log(0.5)
    is_speech = probs > 0.5
    scores = np.where(is_speech, scores, -np.inf)
    is_pos = scores > 0
    enough = is_pos.sum(0, keepdims=True) >= MIN_POS
    return np.where(~is_pos & is_speech & enough, -np.inf, scores)


def boost(scores, n, amount):
    if n <= 0:
        return scores
    idx = np.argsort(-scores, axis=0, kind="stable")[:n]
    for s in range(scores.shape[1]):
        scores[idx[:, s], s] += amount
    return scores


def compress(embeds, probs, silence):
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
    flat = scores.T.reshape(-1)
    top = np.argsort(-flat, kind="stable")[:CACHE_LEN]
    top = np.where(np.isneginf(flat[top]), sentinel, top)
    top = np.sort(top)
    frame = np.where(top == sentinel, F, np.minimum(top % scored, F))
    return embeds[frame], probs[frame], frame


class Cache:
    def __init__(self):
        self.embeds = np.zeros((0, 512), np.float32)
        self.probs = np.zeros((0, NSPK), np.float32)
        self.fifo = np.zeros((0, 512), np.float32)
        self.compressed = False
        self.last_frames = None

    def cached(self):
        return np.concatenate([self.embeds, self.fifo], 0)

    def update(self, input_embeds, logits, silence, n_chunk, step_mask=None):
        probs = 1 / (1 + np.exp(-logits))
        probs = probs[: len(probs) // SUB * SUB].reshape(-1, SUB, NSPK).mean(1)
        if step_mask is not None:
            probs = probs * step_mask[:, None]
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
            self.last_frames = None
            if len(cache_embeds) > CACHE_LEN:
                cache_embeds, cache_probs, self.last_frames = compress(cache_embeds, cache_probs, silence)
                self.compressed = True
            self.embeds, self.probs = cache_embeds.astype(np.float32), cache_probs.astype(np.float32)
        self.fifo = fifo_embeds.astype(np.float32)


def run_port(mel, record, n_valid=None):
    """doc-015 chunk loop on the ONNX graph; `record` collects (cache, fifo, probs, frames) after each step."""
    n_enc = -(-len(mel) // SUB)
    cache, out = Cache(), []
    for start in range(0, n_enc, CHUNK):
        end = min(start + CHUNK, n_enc)
        n_chunk = end - start
        ctx_end = min(end + RIGHT, n_enc)
        f = mel[start * SUB : ctx_end * SUB]
        cached = cache.cached()
        enc_in = -(-len(f) // SUB)
        am = np.ones((1, len(cached) + enc_in), np.int64)
        step_mask = None
        if n_valid is not None:  # HF: attention_mask[:, ::8] over the chunk, ones over the cached frames
            valid = (np.arange(start * SUB, ctx_end * SUB, SUB) < n_valid)[:enc_in]
            am[0, len(cached):] = valid
            step_mask = am[0].astype(np.float32)
        logits, chunk_embeds, silence = sess.run(None, {
            "input_features": f[None], "cached_embeds": cached[None].astype(np.float32),
            "attention_mask": am})
        logits, chunk_embeds = logits[0], chunk_embeds[0]
        input_embeds = np.concatenate([cached, chunk_embeds], 0)
        out.append(logits[len(cached) * SUB : (len(cached) + n_chunk) * SUB])
        cache.update(input_embeds, logits, silence, n_chunk, step_mask)
        record.append(dict(embeds=cache.embeds.copy(), probs=cache.probs.copy(), fifo=cache.fifo.copy(), frames=cache.last_frames))
    return np.concatenate(out, 0)[:len(mel)]


def mad(a, b):
    if a.shape != b.shape:
        return f"SHAPE {a.shape} vs {b.shape}"
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0


report = {}
# features: HF extractor vs port recipe
from scipy.signal import get_window
sym_window = get_window("hann", 400, fftbins=False)
feats_port = log_mel_port(audio)
report["features HF vs port recipe: shape"] = (feats.shape, feats_port.shape)
if feats.shape == feats_port.shape:
    d = np.abs(feats.astype(np.float64) - feats_port)
    report["features max abs diff (all)"] = float(d.max())
    report["features mean abs diff"] = float(d.mean())
    report["features max abs diff first 64"] = float(d[:HEAD].max())
    report["features max abs diff last 64"] = float(d[-TAIL:].max())
    report["features argmax (frame, bin)"] = tuple(int(i) for i in np.unravel_index(d.argmax(), d.shape))

# Port A: ONNX graph fed HF features (isolates the graph + cache logic from the mel recipe)
recA = []
logitsA = run_port(feats, recA)
# Port B: ONNX graph fed the port's own features (the whole doc-015 pipeline as it ran)
recB = []
logitsB = run_port(feats_port, recB)

# Port C: everything HF does that the port skipped: symmetric Hann, last frame zeroed, attention mask honoured
recC = []
feats_c = log_mel_port(audio, window=sym_window, mask_last=True)
d = np.abs(feats.astype(np.float64) - feats_c)
report["features (symmetric Hann + last frame zeroed) max abs diff all"] = float(d.max())
report["features (symmetric Hann + last frame zeroed) mean abs diff"] = float(d.mean())
d1 = np.abs(feats.astype(np.float64) - log_mel_port(audio, window=sym_window))[:-1]
report["features symmetric Hann only, frames 0..N-2, max abs diff"] = float(d1.max())
d2 = np.abs(feats[:-1].astype(np.float64) - feats_port[:-1])
report["features verbatim port, frames 0..N-2, max abs diff"] = float(d2.max())
report["features verbatim port, frames 0..N-2, mean abs diff"] = float(d2.mean())
logitsC = run_port(feats_c, recC, n_valid=10000)
probsC = 1 / (1 + np.exp(-logitsC.astype(np.float64)))
report["probs: port with HF-faithful features+mask vs HF torch, max abs"] = mad(probsC, probs_hf)
report["probs: port C, mean abs"] = float(np.mean(np.abs(probsC - probs_hf)))
report["probs: port C, flips of >0.5"] = int(((probsC > 0.5) != (probs_hf > 0.5)).sum())
for i in range(4):
    r, h = recC[i], states[i]
    report[f"cache step {i} C: embeds max abs"] = mad(r["embeds"], h["embeds"])
    report[f"cache step {i} C: stored probs max abs"] = mad(r["probs"], h["probs"])
    report[f"cache step {i} C: fifo max abs"] = mad(r["fifo"], h["fifo"])
probsA = 1 / (1 + np.exp(-logitsA.astype(np.float64)))
probsB = 1 / (1 + np.exp(-logitsB.astype(np.float64)))
report["probs: ONNX-graph+port-cache on HF feats vs HF torch, max abs"] = mad(probsA, probs_hf)
report["probs: port A, mean abs"] = float(np.mean(np.abs(probsA - probs_hf)))
report["probs: port A, flips of >0.5"] = int(((probsA > 0.5) != (probs_hf > 0.5)).sum())
report["probs: whole doc-015 pipeline (port feats) vs HF torch, max abs"] = mad(probsB, probs_hf)
report["probs: whole pipeline, mean abs"] = float(np.mean(np.abs(probsB - probs_hf)))
report["probs: whole pipeline, flips of >0.5"] = int(((probsB > 0.5) != (probs_hf > 0.5)).sum())
bad = np.nonzero(np.abs(probsA - probs_hf).max(1) > 1e-3)[0]
report["probs: port A, frames differing > 1e-3 (first, last, count)"] = (int(bad.min()), int(bad.max()), len(bad)) if len(bad) else None
report["logits: graph vs HF torch max abs"] = mad(logitsA, logits_hf)
for i in range(4):
    for tag, rec in (("A (HF feats)", recA), ("B (port feats)", recB)):
        r, h = rec[i], states[i]
        report[f"cache step {i} {tag}: shapes"] = (r["embeds"].shape, h["embeds"].shape, r["fifo"].shape, h["fifo"].shape)
        report[f"cache step {i} {tag}: embeds max abs"] = mad(r["embeds"], h["embeds"])
        report[f"cache step {i} {tag}: stored probs max abs"] = mad(r["probs"], h["probs"])
        report[f"cache step {i} {tag}: fifo max abs"] = mad(r["fifo"], h["fifo"])
        if r["embeds"].shape == h["embeds"].shape:
            # an embedding row is a frame of the input; rows equal within 1e-3 mean the same frame was kept
            report[f"cache step {i} {tag}: rows differing > 1e-3"] = int((np.abs(r["embeds"] - h["embeds"]).max(1) > 1e-3).sum())
for k, v in report.items():
    print(k, "=", v)
