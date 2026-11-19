---
updated_date: '2026-11-19T14:38:00Z'
id: decision-002
title: ASR pipeline (Whisper)
date: '2026-08-27 04:38'
status: accepted
---
Transcription runs on `whisper-rs` (whisper.cpp's Rust bindings) with Metal acceleration and no Core ML, chosen over Parakeet-TDT (better published accuracy, but every Rust-embeddable runtime path on Apple Silicon — ONNX/ort, Candle — was still immature at evaluation time) and FluidAudio (best raw performance, but Swift-only with no Rust FFI). Metal alone is fast enough for this tool's batch, non-real-time use, and Core ML would require a one-time Python/coremltools conversion step the project exists to avoid.

Whisper hallucinated the literal string "Thank you." across long silent stretches of the mic track — 76 of 115 turns in one real 43-minute session, almost all exactly one 30-second Whisper window long — because handing an entire mostly-silent track to Whisper undivided lets its language-model prior fill silence with plausible-sounding text. The fix runs whisper.cpp's own Silero VAD as a standalone detector, splices only the detected speech regions into one buffer for a single decode pass, and maps decoder timestamps back to the original timeline by exact sample-integer translation rather than the interpolation whisper.cpp uses internally (which carries a measurable scale error). Reusing the pyannote diarization model as a VAD was ruled out because it would route mic audio through the diarization seam, violating the standing rule that mic audio never touches diarization models; whisper.cpp's own built-in VAD flag was found to be dead code on the only call path `whisper-rs` exposes safely. A text blocklist for the literal phrase was rejected outright — it would delete real speech containing those words and miss every other hallucination.

The investigation initially suspected Whisper's decoder was carrying context across windows and "priming" the repetition; reading the vendored C++ confirmed that mechanism was already disabled by default and the real fix was the VAD gate alone (measured collapse from 63 to 7 mic turns). That correction is recorded so a future engineer doesn't reopen a dead lead. Separately, Core ML was evaluated for the diarization ONNX graphs (not Whisper) and rejected: Apple's newer `MLProgram` format can't even load the segmentation model, and a compiled-model cache saves roughly 0.3 seconds of cold start against a run dominated by minutes of Whisper decoding — not worth the disk cost and cache-invalidation complexity it would add.

## Considered options

- Parakeet-TDT via ONNX/ort or Candle — better numbers on paper, immature Rust runtime support.
- FluidAudio — best accuracy and speed, but no Rust FFI.
- Reusing pyannote segmentation as VAD — would cross a deliberately maintained architectural boundary.
- A text blocklist for hallucinated phrases — destroys real speech, doesn't generalize.
- Core ML `MLProgram` for diarization, plus a compiled-model cache — doesn't load the model at all; the cache's benefit is smaller than measurement noise.

## Consequences

<!-- SECTION:CONSEQUENCES:BEGIN -->
The speech gate's detector is FireRedVAD's streaming model, not Silero (amended 2026-11-19; the Silero text above is the original reasoning, and the gate's shape - splice, one decode pass, exact timestamp map-back - is unchanged). It runs through `ort` on the CPU with a Kaldi-style fbank frontend in `fbank.rs`, the training-time CMVN statistics embedded in the crate, and whisper.cpp's region walk ported into `vad.rs`, so the dependency on whisper.cpp's VAD is gone entirely. Chosen on one real session (129 minutes, both tracks) against Silero, and against SAVA, which was rejected because its output depended on stream history and dropped minutes of quiet speech. FireRed keeps short backchannels Silero cut, joins a speaker's turn into fewer pieces, and its output depends only on a bounded window of audio. Threshold 0.35 was picked by listening, not by a metric.

Costs: about 20% more audio goes to Whisper on the speaker track than under Silero (61.9 min against 50.6 min at the same 0.1 s minimum silence; the Silero figure comes from a Python approximation of the walk and the FireRed figure from this crate, so the two are not exactly like for like). The weights come from the FireRedVAD GitHub repo at a pinned commit rather than Hugging Face, and `assets/firered-vad-cmvn.ark` must move in step with them. The 0.5 s `min_silence_s` in `gate_tuning` was measured under Silero and has not been re-measured for FireRed.
Vocabulary initial prompt (amended 2026-11-19). `WhisperEngine` can carry an optional initial prompt built from a list of vocabulary terms (`set_vocabulary`, `build_vocabulary_prompt` in `asr.rs`); with no terms the recognition call is unchanged. Whisper spells names and jargon it has not seen by sound (the barbarian "Brog" came out as "Rog" or "Roger"). On a 5-minute real clip with large-v3-turbo, a bare list did not fix it, while a short sentence did ("Meeting notes: Brog, Brog's rage, Roll20, D&D, D20."), so the prompt is always one sentence, `Meeting notes: A, B, C.`. Risks: repeating a term makes the model echo it into the transcript ("Brog, Brog, Brog. We are to you"), so terms are de-duplicated case-insensitively; and without the speech gate a prompt on a mostly-silent window produced runs of "I don't know.", so the prompt is set only after the gate, on the gated path. Whisper's prompt window is 224 tokens and no tokenizer is available, so the prompt is bounded by a 400-byte budget and terms beyond it are dropped from the end and counted for the caller. Where the terms come from is a separate decision (the vocabulary file).
<!-- SECTION:CONSEQUENCES:END -->

