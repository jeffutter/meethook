---
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

