---
id: doc-003
title: ASR model and runtime evaluation for Apple Silicon
type: other
created_date: '2026-08-09 02:07'
updated_date: '2026-08-09 02:07'
---
# ASR Model and Runtime Evaluation for Apple Silicon (meethook)

Scope: batch (non-realtime), local-first, macOS-only, Apple Silicon (M-series) speech-to-text for meethook. Must be embeddable directly inside a Rust binary — no external Python process. vLLM/llama.cpp-hosted LLMs are out of scope (reserved for a future summarization/search feature).

## 1. Parakeet-family models (NVIDIA NeMo)

**Accuracy/speed reputation vs Whisper:** On the Open ASR Leaderboard's aggregate benchmark (AMI, Earnings22, Gigaspeech, LibriSpeech, SPGI, TED-LIUM, VoxPopuli), Parakeet-TDT-0.6B-v3 scores ~6.32% average WER vs ~7.44% for Whisper-large-v3, while running at dramatically higher throughput (RTFx ~3332 vs ~145) ([Open ASR Leaderboard paper](https://arxiv.org/pdf/2510.06961)). Whisper still wins on multilingual coverage (99+ languages vs Parakeet v3's 25 European languages, no CJK) and showed better robustness on some noisy/accented benchmarks (e.g. AfriSpeech-MultiBench: Whisper-large-v3 33.79 vs Parakeet-TDT-v2 40.89 error rate) — see [BibiGPT comparison](https://bibigpt.co/en/blog/posts/parakeet-vs-whisper-transcription) and [AfriSpeech-MultiBench](https://arxiv.org/pdf/2511.14255). Parakeet is English/European-focused; Whisper is the safer choice for broad language coverage.

**Rust-embeddability on Apple Silicon — several real paths, all with caveats:**

- **ONNX export + `ort` (Rust ONNX Runtime bindings):** NeMo natively exports Parakeet to ONNX (`model.export("model.onnx")`); pre-converted models are published at [istupakov/parakeet-tdt-0.6b-v2-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx) and [v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx). The Rust crate **[parakeet-rs](https://github.com/altunenes/parakeet-rs)** (also on [crates.io](https://crates.io/crates/parakeet-rs)) wraps `ort` for this. **Important caveat for Apple Silicon:** the crate's own source comments state CoreML "currently fails with this model due to unsupported operations," and WebGPU (Metal-backed) is "experimental and may produce incorrect results" — despite the README recommending WebGPU/CPU. In practice this currently means **CPU-only is the reliable path** on Mac (reported as still faster than Whisper-Metal for this model, but the ecosystem is young and NeMo's own ONNX export has open regressions, e.g. [NeMo issue #15040](https://github.com/NVIDIA-NeMo/NeMo/issues/15040)).
- **Pure Rust via `candle` + Metal:** **[gpu-cli/parakeet-rs](https://github.com/gpu-cli/parakeet-rs)** is a from-scratch Candle implementation of Parakeet-TDT-0.6B-v3 with native Metal GPU acceleration, avoiding ONNX/CoreML entirely. It reports 3.83% WER on LibriSpeech test-clean vs NVIDIA's published 1.93% — a real accuracy regression indicating an immature/incomplete port.
- **`ort`'s general macOS story:** `ort` itself supports static linking (build ONNX Runtime from source without `--build_shared_lib`, point `ORT_LIB_PATH` at the static libs) which avoids Python and dylib/rpath fragility — see [ort linking docs](https://ort.pyke.io/setup/linking). This is a solid embedding path in principle; the fragility is specific to Parakeet's CoreML/WebGPU execution providers, not to `ort`/ONNX Runtime generally.
- **Alternative acceleration path (not Rust):** Swift/CoreML via **[FluidAudio](https://github.com/FluidInference/FluidAudio)** running Parakeet TDT on the Apple Neural Engine reports 2.1% average WER on LibriSpeech test-clean at ~128x RTFx ([Benchmarks.md](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md)) — notably better than either ONNX or Candle Parakeet ports above. This isn't directly embeddable in a Rust binary without an Objective-C/Swift FFI bridge, but it's evidence that Parakeet-on-ANE has real headroom once the Rust tooling catches up.
- A unified crate, **[cjpais/transcribe-rs](https://github.com/cjpais/transcribe-rs)** (extracted from the Handy macOS app), wraps both Parakeet (ONNX) and Whisper (whisper.cpp) behind one `SpeechModel` trait with Metal acceleration on macOS — useful if we want to keep engine choice swappable rather than committing early.

**Conclusion for point 1:** Parakeet is genuinely attractive on accuracy/throughput paper stats, but every current Rust-embeddable path on Apple Silicon has a real maturity gap (CoreML broken, WebGPU unverified, Candle port under-accurate vs NVIDIA's numbers). Not yet a safe default for v1.

## 2. Whisper-family models

**Rust bindings:** **[whisper-rs](https://github.com/tazz4843/whisper-rs)** (mirrored on GitHub, canonical repo on Codeberg) provides mature, long-maintained Rust bindings to whisper.cpp, with Cargo feature flags for `coreml`, `metal`, `cuda`, `vulkan`, etc. ([Cargo.toml](https://github.com/tazz4843/whisper-rs/blob/master/Cargo.toml)). A convenience wrapper, **[mutter](https://github.com/sigaloid/mutter)**, adds audio decoding/resampling around it.

**Apple Silicon acceleration:** whisper.cpp has first-class Metal support (ggml compiles Metal Shading Language kernels for encoder/decoder matmuls) giving roughly 2-4x speedup over CPU-only, and optional Core ML encoder acceleration on top (offloads the encoder to the Apple Neural Engine, reported as >3x over CPU-only, or a further ~15-20% over Metal alone on some workloads) — see [whisper.cpp Metal blog](https://fazm.ai/blog/whisper-cpp-metal-apple-silicon) and [Apple Silicon Whisper benchmark](https://justvoice.ai/blog/whisper-benchmark-apple-silicon-m3-m4). Reported real-time factors with Metal: tiny 48-60x realtime, small 22-34x, large-v3 5-14x depending on chip generation (M1→M5 Pro), large-v3-turbo 14-18x on M5. A 60-minute recording on an M2 Pro with large-v3 (Q5_1 quantized) transcribes in roughly 8-12 minutes.

**Accuracy:** large-v3 average WER ~7.4% on the Open ASR Leaderboard aggregate (worse than Parakeet's ~6.3%), but Whisper covers 99+ languages and has shown better robustness on some noisy/accented audio in independent tests. distil-whisper and whisper large-v3-turbo trade a little accuracy for significant speed and are also usable via whisper-rs/whisper.cpp (same ggml runtime, no separate Rust binding needed since they ship as ggml-compatible checkpoints).

**faster-whisper (CTranslate2) note:** Explicitly weaker fit here — it has no Metal backend and falls back to CPU-only on macOS, and it isn't natively Rust (its core is C++/CTranslate2 typically driven via Python); whisper.cpp with Metal already outperforms faster-whisper on Mac, so it's not worth pursuing for this ticket.

**Conclusion for point 2:** Whisper via whisper-rs is the most mature, battle-tested, genuinely Rust-native, Apple-Silicon-accelerated (Metal + optional CoreML-encoder) option available today, at the cost of somewhat worse WER and speed than Parakeet's best-case numbers.

## 3. Other candidates (candle / ONNX Runtime / MLX-adjacent)

- **`candle` (Hugging Face's Rust ML framework):** Whisper is a first-class, actively maintained example in candle-transformers with CUDA/Metal support and even WASM/browser demos ([candle whisper example](https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper)). This gives a second, fully-pure-Rust path to Whisper (no C++ FFI at all) if avoiding whisper.cpp's C++ dependency is a priority — though whisper-rs/whisper.cpp is more battle-tested for production accuracy/perf tuning (quantization, Core ML encoder, years of community bug fixes).
- **`candle` + Parakeet:** only via the third-party, unofficial [gpu-cli/parakeet-rs](https://github.com/gpu-cli/parakeet-rs) port discussed above (accuracy gap vs NVIDIA's numbers).
- **MLX-adjacent (Python/Swift, not Rust):** [parakeet-mlx](https://github.com/senstella/parakeet-mlx) (Python, Apple's MLX) and a stalled [swift-parakeet-mlx](https://github.com/FluidInference/swift-parakeet-mlx) port exist but neither is Rust-embeddable; MLX itself doesn't have official Rust bindings. Relevant mainly as a datapoint that Parakeet-on-Apple-Silicon-without-ONNX is an active but unsettled area (the Swift team abandoned MLX in favor of CoreML/ANE for better hardware utilization).
- **ONNX Runtime (`ort`) generally:** Viable and Python-free at runtime (static linking avoids even a bundled dylib), and is the substrate several of the above crates (parakeet-rs, transcribe-rs) build on. The friction is model-specific (Parakeet's CoreML/WebGPU EPs), not with `ort`/ONNX Runtime as infrastructure.
- **[transcribe-rs](https://github.com/cjpais/transcribe-rs):** worth calling out again here as an architecture pattern — a single Rust crate/trait abstracting over whisper.cpp and multiple ONNX ASR models (Parakeet, Moonshine, SenseVoice, GigaAM), extracted from a real shipping macOS app (Handy). Adopting this trait/abstraction (or its design) would let meethook start on Whisper and swap to Parakeet later without a rewrite.

## 4. Recommendation

**Use whisper-rs (whisper.cpp bindings) with Metal + optional Core ML encoder acceleration, running a ggml Whisper model (large-v3-turbo or distil-whisper for speed, or large-v3 if maximizing accuracy per-recording matters more than turnaround time), as the v1 ASR engine for meethook.**

Why:
- **Maturity/risk:** whisper-rs and whisper.cpp are years-old, widely deployed, and have dedicated, tested Apple Silicon acceleration (Metal always, Core ML encoder optionally). Every current Rust-embeddable Parakeet path (ONNX/ort, or the from-scratch Candle port) has a documented, current-state gap on Apple Silicon: CoreML EP broken for Parakeet, WebGPU EP unverified/"may produce incorrect results" per the maintainer's own code comments, and the pure-Rust Candle port undershoots NVIDIA's published WER by roughly 2x. For a personal tool where correctness of transcripts matters, that's a meaningful reliability risk to take on as the default.
- **Good enough performance for batch use:** Since meethook doesn't need real-time transcription, whisper.cpp's Metal-accelerated 5-18x-realtime range for large/turbo models on Apple Silicon is comfortably fast for personal meeting-length recordings, even without Parakeet's much larger RTFx headroom.
- **Acceptable accuracy trade-off:** Whisper large-v3's ~7.4% WER vs Parakeet's ~6.3% (both on the standard leaderboard aggregate) is a modest gap, and Whisper's broader language coverage and stronger noisy/accented robustness in some benchmarks are useful properties for real meeting audio (cross-talk, accents, background noise) that a personal meeting recorder will actually encounter.
- **No Python, no fragile export pipeline:** whisper.cpp ships ggml model files directly (no ONNX export step, no NeMo toolchain, no dynamic_axes/dynamo regressions to track); whisper-rs links against it directly. This is the simplest, most self-contained path to "model + runtime embedded directly inside a Rust binary."

**Path to revisit Parakeet later:** Track NeMo's ONNX export stability, the CoreML EP fix for parakeet-rs, and gpu-cli/parakeet-rs's WER convergence toward NVIDIA's published numbers. If/when one of those paths matures, adopting the `transcribe-rs`-style abstraction (a small internal trait wrapping the ASR engine) would let meethook add Parakeet as a second backend, or switch to it, without redesigning the surrounding pipeline. Given Parakeet's throughput and accuracy ceiling, it's worth a fast-follow evaluation once the Apple Silicon tooling gap closes — but it is not a safe bet for v1.

## Sources

- [Open ASR Leaderboard paper (arXiv 2510.06961)](https://arxiv.org/pdf/2510.06961)
- [Parakeet vs Whisper 2026 comparison (BibiGPT)](https://bibigpt.co/en/blog/posts/parakeet-vs-whisper-transcription)
- [AfriSpeech-MultiBench (arXiv 2511.14255)](https://arxiv.org/pdf/2511.14255)
- [Canary-1B-v2 & Parakeet-TDT-0.6B-v3 paper (arXiv 2509.14128)](https://arxiv.org/pdf/2509.14128)
- [parakeet-rs (altunenes) GitHub](https://github.com/altunenes/parakeet-rs)
- [parakeet-rs crates.io](https://crates.io/crates/parakeet-rs)
- [gpu-cli/parakeet-rs (Candle, pure Rust) GitHub](https://github.com/gpu-cli/parakeet-rs)
- [istupakov/parakeet-tdt-0.6b-v2-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx)
- [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx)
- [NeMo ONNX export dynamic_shapes issue #15040](https://github.com/NVIDIA-NeMo/NeMo/issues/15040)
- [ort (Rust ONNX Runtime bindings) linking docs](https://ort.pyke.io/setup/linking)
- [ort crates.io](https://crates.io/crates/ort)
- [whisper-rs GitHub (tazz4843)](https://github.com/tazz4843/whisper-rs)
- [whisper-rs Cargo.toml feature flags](https://github.com/tazz4843/whisper-rs/blob/master/Cargo.toml)
- [mutter (whisper-rs wrapper) GitHub](https://github.com/sigaloid/mutter)
- [whisper.cpp Metal on Apple Silicon (Fazm blog)](https://fazm.ai/blog/whisper-cpp-metal-apple-silicon)
- [Whisper benchmark on Apple Silicon M1-M5 (JustVoice)](https://justvoice.ai/blog/whisper-benchmark-apple-silicon-m3-m4)
- [candle (Hugging Face Rust ML framework) GitHub](https://github.com/huggingface/candle)
- [candle Whisper example](https://github.com/huggingface/candle/tree/main/candle-examples/examples/whisper)
- [cjpais/transcribe-rs GitHub](https://github.com/cjpais/transcribe-rs)
- [transcribe-rs crates.io](https://crates.io/crates/transcribe-rs)
- [parakeet-mlx (senstella) GitHub](https://github.com/senstella/parakeet-mlx)
- [swift-parakeet-mlx GitHub](https://github.com/FluidInference/swift-parakeet-mlx)
- [FluidAudio GitHub](https://github.com/FluidInference/FluidAudio)
- [FluidAudio Benchmarks.md](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/Benchmarks.md)
