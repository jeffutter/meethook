"""Stage 1: build the redistributable test clip (LibriSpeech test-clean, CC BY 4.0).

Three speakers take turns, one utterance each, 0.4 s of silence between turns, cut to exactly 100 s.
The clip is stored mu-law encoded (8 bit, ITU-T G.711, 1 byte per 16 kHz sample) to keep the fixtures small;
every reference number is computed from the DECODED int16 samples, so the encoding loses nothing.

usage: generate-clip.py OUT_DIR   (needs network: lists LibriSpeech test-clean rows through the HF datasets-server)
"""
import io, json, sys, urllib.request
import numpy as np, soundfile as sf

OUT = sys.argv[1]
SPEAKERS = [6930, 1320, 260]
SR, TOTAL = 16000, 100 * 16000

def list_rows():
    """speaker_id -> [(utterance id, wav url)] for the first 500 rows of librispeech test-clean, in dataset order."""
    sp = {}
    for off in range(0, 500, 100):
        url = ("https://datasets-server.huggingface.co/rows?dataset=openslr/librispeech_asr"
               f"&config=clean&split=test&offset={off}&length=100")
        with urllib.request.urlopen(url, timeout=60) as r:
            for x in json.load(r)["rows"]:
                row = x["row"]
                sp.setdefault(str(row["speaker_id"]), []).append((row["id"], row["audio"][0]["src"]))
    return sp

sp = list_rows()

def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        x, sr = sf.read(io.BytesIO(r.read()), dtype="float32")
    assert sr == SR
    return x

def mulaw_encode(x16):
    """G.711 mu-law, 16-bit linear in -> 8 bit code out (bias 0x84, clip 32635)."""
    x = x16.astype(np.int32)
    sign = (x < 0)
    x = np.minimum(np.abs(x), 32635) + 0x84
    exp = np.floor(np.log2(x)).astype(np.int32) - 7
    mant = (x >> (exp + 3)) & 0x0F
    return (~((sign.astype(np.int32) << 7) | (exp << 4) | mant) & 0xFF).astype(np.uint8)

def mulaw_decode(c):
    c = (~c.astype(np.int32)) & 0xFF
    sign, exp, mant = c & 0x80, (c >> 4) & 7, c & 0x0F
    x = (((mant << 3) + 0x84) << exp) - 0x84
    return np.where(sign != 0, -x, x).astype(np.int16)

parts, next_utt, turn = [], {s: 0 for s in SPEAKERS}, 0
gap = np.zeros(int(0.4 * SR), np.float32)
while sum(len(p) for p in parts) < TOTAL:
    s = SPEAKERS[turn % 3]
    uid, url = sp[str(s)][next_utt[s]]
    next_utt[s] += 1
    parts += [fetch(url), gap]
    print("turn", turn, "speaker", s, uid, len(parts[-2]) / SR, "s", flush=True)
    turn += 1
x = np.concatenate(parts)[:TOTAL]
x16 = np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16)
code = mulaw_encode(x16)
dec = mulaw_decode(code)
code.tofile(f"{OUT}/nemotron-clip.mulaw")
print("samples", len(dec), "snr dB", 10 * np.log10((x16.astype(float) ** 2).sum() / ((x16.astype(float) - dec) ** 2).sum()))
