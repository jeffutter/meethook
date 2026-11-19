# Nemotron-3-Diarization parity fixtures

Reference numbers for porting `nvidia/Nemotron-3-Diarization` and its streaming speaker cache to Rust
(TASK-088.03 features, .04 cache, .05/.06 graph and end-to-end probabilities). They come from the Hugging Face
transformers implementation, run offline in Python. **Nothing in `cargo build` or `cargo test` runs Python.** The two
`generate-*.py` files beside this note only document how the fixtures were made; they need torch and the weights
and are not required by any teammate.

Background and the algorithm: `backlog/docs/evaluations/doc-015`.

## Files

All integers and floats are little-endian. Total size is about 7.6 MB.

| file | content |
|---|---|
| `nemotron-clip.mulaw` | 1,600,000 bytes, the clip: 100.0 s, 16 kHz mono, 8-bit mu-law (G.711), one byte per sample |
| `nemotron-features.bin` | header 4 x u32 `[mel_frames=10001, bins=128, head=64, tail=64]`, then `f32[64][128]` (frames 0..63), then `f32[64][128]` (frames 9937..10000) |
| `nemotron-cache-step{0,1,2,3}.bin` | header 4 x u32 `[n_cache, n_fifo, hidden=512, speakers=8]`, then `f32[n_cache][512]` cache embeddings, `f32[n_cache][8]` stored cache probabilities, `f32[n_fifo][512]` FIFO embeddings |
| `nemotron-update-step{0,1,2,3}.bin` | what `Nemotron3DiarizationSpeakerCache.update` consumed at that step (TASK-088.04): header 6 x u32 `[n_cache, n_fifo, n_chunk, n_in, hidden=512, speakers=8]` (cache and FIFO rows held *before* the step, so the state to start from is `nemotron-cache-step{N-1}.bin`; none at step 0), then `f32[n_in*8][8]` raw logits of the step, `f32[n_in]` attention mask (1.0 valid, 0.0 padding), `f32[n_chunk][512]` chunk embeddings that join the FIFO, `f32[512]` silence embedding. Sizes 797 KB, 876 KB, 876 KB, 614 KB; `n_in` is 380, 684, 684, 535 |
| `nemotron-probs.bin` | header 2 x u32 `[frames=10001, speakers=8]`, then `f32[10001][8]` sigmoid probabilities, one row per 10 ms mel frame |
| `generate-clip.py`, `generate-reference.py` | the generators (documentation only) |

Cache sizes: steps 0, 1, 2 are `n_cache=264, n_fifo=40`; step 3 (the last chunk, no right context) is
`n_cache=264, n_fifo=0`. Every step is compressed (see below), so the cache holds 264 rows, in speaker-major order
with the reserved silence rows, as `_compress` returns them. State is the one after `update` of that step, i.e. what
the next step receives as `cached_embeds`.

### The clip

Not a meeting recording. LibriSpeech `test-clean` (OpenSLR 12, CC BY 4.0, redistributable with attribution), speakers
6930, 1320 and 260 take turns, one utterance each, 0.4 s of silence between turns, cut at 100.0 s:
`6930-75918-0000..0003`, `1320-122617-0000..0002`, `260-123440-0000..0002` (and the start of the next 1320
utterance). Utterances are read through the Hugging Face datasets-server. Audio was converted to 16-bit and
stored mu-law encoded to keep the fixtures under 5 MB; **every reference number is computed from the decoded
samples, never from the original**. Decode (the G.711 formula):

```
c = !code (as 8 bit); sign = c & 0x80; exp = (c >> 4) & 7; mant = c & 0x0F
x = (((mant << 3) + 0x84) << exp) - 0x84;  sample = sign ? -x : x      // i16
float = sample / 32768.0
```

Decoded-vs-original SNR is 37.2 dB. 100 s gives 10001 mel frames, 1251 encoder frames, and with chunk 340 four chunks
of 340, 340, 340 and 231 encoder frames, so the first chunk already pops 300 frames into an empty cache (over the
264 limit) and compression runs at every step.

## Versions and how it was made

- Python 3.14.10, macOS arm64, CPU, `torch.set_num_threads(1)`.
- torch 2.15.0, **transformers 5.23.0 (released, PyPI; the `nemotron3_diarization` model is in the release, so no
  source revision is needed)**, numpy 2.5.5, onnxruntime 1.32.0, librosa 1.0.1, scipy 1.18.1, soundfile 0.14.0,
  huggingface_hub 2.4.0, safetensors 0.8.0.
- Weights `nvidia/Nemotron-3-Diarization` at revision `52a432672732425e4c19765b9fec9b5810e6f4ba`
  (`model.safetensors` sha256 `c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6`), loaded as float32.
- ONNX cross-check `onnx-community/Nemotron-3-Diarization-ONNX` at `353b6f8ad2cac3580e982d7fbdf0a010786b0406`:
  `onnx/model.onnx` (fp32, sha256 `12a7c98f...3e9d`) with `model.onnx_data` (sha256 `c293d9b5...b1dbd6`). The fp32 graph was
  used, not fp16, for the tightest parity; fp16 was not run.

```sh
python3 -m venv v
v/bin/pip install torch==2.15.0 transformers==5.23.0 numpy==2.5.5 onnxruntime==1.32.0 librosa==1.0.1 \
    scipy==1.18.1 soundfile==0.14.0 huggingface_hub==2.4.0
mkdir out
v/bin/python generate-clip.py out        # writes out/nemotron-clip.mulaw
v/bin/python generate-reference.py out   # writes the other fixture kinds, prints the parity report
```

Both scripts were run twice from scratch (the clip was rebuilt from the network; the reference was regenerated)
and every output file compared byte-identical with `cmp`.

## What the reference computes

- **Features**: `NemotronAsrStreamingFeatureExtractor` through `AutoProcessor`, offline (`center=True`), unchanged.
  Recipe, read from its source: pre-emphasis 0.97 (`x[t] - 0.97 x[t-1]`, `x[0]` unchanged); `torch.stft` with
  `n_fft=512`, `win_length=400`, `hop=160`, constant (zero) centre padding of 256 each side; **symmetric** Hann window
  (`torch.hann_window(400, periodic=False)`), placed in the 512 FFT by torch; power spectrum; 128 mel filters from
  `transformers.audio_utils.mel_filter_bank` (Slaney scale, Slaney norm, 0 to 8000 Hz); `ln(mel + 2^-24)`; no
  normalisation; then **frames at and after index `floor(samples / 160)` are multiplied by the attention mask, i.e.
  set to 0.0** (not the log of silence). For this clip the processor returns 10001 frames with 10000 valid: frame 10000
  is all zeros.
- **Cache and probabilities**: `AutoModelForAudioFrameClassification` forward in offline mode on those features and the
  attention mask (chunk 340, right context 40, left 0, FIFO 40, update period 300, cache 264). `Nemotron3DiarizationSpeakerCache.update`
  was wrapped to copy the state after each call. `nemotron-probs.bin` is `sigmoid(logits)` (computed in float64, stored f32) of the
  end-to-end forward. The cache holds the output of the feature-stacking embedder (`audio_tower.embedder`), not of the
  transformer; this is what the ONNX graph returns as `chunk_embeds`.
- The attention mask matters: HF passes `mask[:, ::8]` for the chunk (ones for cached frames), so the last encoder
  frame (index 1250, built from the zeroed mel frame) is masked out of attention and gets probability 0 in the pooled
  cache probabilities.
- The selected frame indices of `_compress` are not observable from the HF class. They are covered instead by the
  stored embeddings and probabilities, which identify the kept rows.

## Does the doc-015 numpy port match?

**Not as written. With two small fixes it matches.** `generate-reference.py` runs three variants against the HF fixtures
(ONNX fp32 graph driven by the port's chunk loop and cache, which are verbatim doc-015):

| variant | features | probabilities vs HF | cache after steps 0-3 vs HF |
|---|---|---|---|
| doc-015 verbatim (port features, no mask) | max abs 11.02 (frame 10000), 1.46 on frames 0..9999 | max abs 0.185; no flip of the 0.5 threshold | 256 of 264 rows differ by > 1e-3 at every step, max abs up to 134 at step 3 |
| port cache on HF features, no mask | n/a | max abs 0.078, all in frames 8160..10000 (the last chunk, 1681 frames); no flip of 0.5 | steps 0-2 match (embeds <= 2.5e-4, probs <= 1.1e-6, FIFO <= 1.7e-4); step 3 diverges, 154 of 264 rows differ, max abs 98 |
| port with HF window and mask | max abs 3.2e-4 over all frames, mean 9.6e-7 | max abs 2.4e-6, mean 1.2e-8; no flip | embeds <= 3.7e-4, stored probs <= 9.5e-7, FIFO <= 1.7e-4, all four steps |

Where the verbatim port differs:

1. **Window.** doc-015 uses `librosa.stft(window="hann")`, a periodic Hann. The model uses the symmetric Hann
   (`periodic=False`). This alone changes features by up to 1.46 (mean 0.0059) and shifts the cache embeddings by up
   to about 0.4, which makes 256 of 264 cache rows differ by more than 1e-3 at every step. No probability crossed the 0.5 threshold on this clip.
2. **Final partial frame and attention mask.** HF zeroes the mel frames at and after `floor(samples / hop)` (not
   `ln(2^-24)`), and masks the matching encoder frame out of attention and out of the cache probabilities. The port's
   all-ones mask lets the last frame in. Effect: the last chunk's logits change (max prob difference 0.078 on HF
   features) and so does the final compression (step 3). Steps 0-2 are unaffected.
3. The remaining 3e-4 on features is float32 rounding plus the mel filter bank construction
   (`mel_filter_bank` versus `librosa.filters.mel`); it does not change any decision on this clip.

A Rust port should follow the HF recipe above (symmetric Hann, zeroed trailing frames, attention mask honoured at
chunk ends) rather than the doc-015 recipe. Suggested tolerances: features 1e-3, probabilities 1e-4 (the fixture
differences above are well inside), embeddings 1e-3.

## Limits

- One clip, three speakers, no overlap, no more than 3 of 8 slots used; many-speaker input is TASK-088.11.
- Only offline mode (chunk 340 / right context 40) was captured; the model's streaming latency modes were not.
- The probabilities fixture is the HF torch result. The ONNX fp32 graph agrees with it (max abs 2.4e-6 once the mask is
  passed); the fp16 export and CoreML were not checked.

## Expected turns (TASK-088.07)

`expected-turns.py` decodes `nemotron-probs.bin` into speaker turns (enter above 0.5, stay while
at least 0.25, merge gaps up to 1.0 s, drop turns under 0.5 s) with a loop written separately from
`nemotron_turns::slot_turns`, and prints `(start_frame, end_frame, slot)` tuples. The test
`the_fixture_probabilities_decode_to_the_independent_expectation` holds that output as a literal.
Rerun the script and update the literal if the constants change.
