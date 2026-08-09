---
id: doc-002
title: Diarization and speaker-ID model/runtime evaluation
type: other
created_date: '2026-08-09 02:07'
updated_date: '2026-08-09 02:08'
---
## Scope

TASK-001.03: which diarization approach + speaker-embedding model should meethook use for named speaker-ID against a small fixed set of recurring speakers, with no external Python dependency, embeddable directly in the Rust binary (candle / ONNX runtime bindings / similar) on Apple Silicon, batch (non-real-time) processing.

## 1. Diarization: is there a Rust/ONNX path to pyannote's models without a Python runtime?

Yes — this is a well-trodden path as of 2026, not a research gap.

- **pyannote/segmentation-3.0** (speaker-turn / overlap-aware segmentation) exports cleanly to ONNX via standard `torch.onnx.export`. The official checkpoint is MIT-licensed but gated behind an HF click-through agreement; the **`onnx-community/pyannote-segmentation-3.0`** mirror republishes the same MIT weights, already in ONNX form, ungated. There's also a dedicated conversion project, **pengzhendong/pyannote-onnx**.
- **pyannote's own embedding model** (the newer WeSpeaker-based checkpoint) currently *fails* ONNX export: a Feb 2026 pyannote-audio discussion (#1929) shows `torch.onnx.export` erroring on the internal torchaudio/kaldi `fbank` extraction (`Unsupported value kind: Tensor`). This is a real limitation of the *newest* pyannote embedding checkpoint specifically, not of ONNX/pyannote generally.
- The practical workaround already used by every Rust implementation below: skip the pyannote embedding model and pair segmentation-3.0 with a **WeSpeaker ONNX embedding model** instead (WeSpeaker exports fine — see §2). This is exactly what upstream pyannote.audio itself does internally now (their "new wespeaker" embedding).
- Existing Rust crates already wire this whole pipeline end-to-end, proving it out:
  - **`pyannote-rs`** (thewh1teagle/pyannote-rs, MIT, ~125 stars, actively maintained): segmentation-3.0 (10s sliding window) + `wespeaker-voxceleb-resnet34-LM` embedding, fbank features via companion crate `knf-rs`, cosine-similarity speaker matching, runs on `onnxruntime` via the `ort` crate. Explicitly notes CoreML acceleration on macOS and processes ~1hr audio in <1 min on CPU. Zero Python at runtime.
  - **RustedBytes/pyannote-rs** and **native-pyannote-rs**: same model pair, but pure-Rust inference via **Burn** (ndarray backend) with bundled weights — no `onnxruntime` C++ dependency at all, still zero Python.
  - **OpenASR's pyannote-segmentation-3.0 pack**: repackages the ungated ONNX weights as a pure-Rust `.oasr` raw-f32 format, claiming bit-exact match (~7e-5 max abs error) vs. upstream ONNX logits.
  - **sherpa-onnx** (k2-fsa, official cross-platform C++/ONNX toolkit with an official Rust binding crate): ships a full speaker-diarization pipeline (segmentation + embedding, pairs with 3D-Speaker or NeMo embedding models, plus pyannote per its changelog) and the Rust crate exposes an **in-memory named-speaker-embedding index** — i.e., enrollment/matching against a fixed known-speaker set is a first-class primitive, not something to hand-roll. This is the most mature/officially-maintained option of the group.

Conclusion for §1: no Python needed anywhere at runtime for diarization. Multiple independent Rust implementations already exist; the only real friction is that the *newest* pyannote embedding checkpoint doesn't export, so pair segmentation-3.0 with a WeSpeaker (or 3D-Speaker/NeMo) embedding model instead.

## 2. Speaker-embedding models for identification (WeSpeaker, ECAPA-TDNN, TitaNet)

- **WeSpeaker** (wenet-e2e/wespeaker): purpose-built "research and production" toolkit; **first-class ONNX export is a documented, supported feature** (torch JIT or ONNX, straight from the training checkpoint). Pretrained ONNX exports (e.g. `Wespeaker/wespeaker-ecapa-tdnn512-LM`, `wespeaker-voxceleb-resnet34-LM`) are published on Hugging Face and are exactly what `pyannote-rs` and sherpa-onnx already consume. No official Rust runtime package, but that doesn't matter — the model is a plain ONNX graph, loadable via any Rust ONNX binding (`ort`). This is the most turnkey option.
- **ECAPA-TDNN**: the underlying architecture WeSpeaker's best checkpoints use. It's explicitly designed to optimize cosine-distance separability between embeddings, which maps directly onto the enrollment workflow needed here: collect a handful of clean utterances per known speaker, extract embeddings, mean + L2-normalize into one reference vector per person, then score each diarized segment's embedding against all references via cosine similarity with a decision threshold (below threshold → "unknown speaker"). This is standard practice, not a novel design.
- **NVIDIA NeMo TitaNet**: strong benchmarks, used inside NeMo's own cascaded diarization pipeline (MarbleNet VAD → TitaNet embedding → spectral clustering), and sherpa-onnx does support NeMo-flavoured embedding models as one of its pairings. However, NeMo's own tooling is Python/PyTorch-centric (`.nemo` checkpoints, export via NeMo's Python export scripts) and TitaNet-specific ONNX conversion is far less documented/turnkey than WeSpeaker's built-in exporter. Usable (via sherpa-onnx's pre-converted ONNX releases) but more friction if meethook ever needed to convert or fine-tune a checkpoint itself.

Conclusion for §2: WeSpeaker (ECAPA-TDNN or ResNet34-LM variant) ONNX models are the most practical embedding choice — first-party ONNX export support, already proven inside two independent Rust pipelines, and cosine-similarity matching against a small enrolled set is exactly the workflow the architecture was designed around.

## 3. Newer/all-in-one alternatives considered

- **NeMo Sortformer** (end-to-end transformer diarizer, NVIDIA's 2025/2026 successor to the cascaded VAD+embedding+clustering pipeline): architecturally attractive (single model, no separate clustering step), but **ONNX export is currently broken** for the streaming variants that would be relevant here. Multiple open NVIDIA-NeMo GitHub issues from Sept 2025 through March 2026 (#14733, #15077, #15536) show export failures from dynamic tensor slicing in `sortformer_modules.py` that isn't traceable to a static ONNX graph, plus feature/chunk dimension mismatches. Not viable to embed today; worth revisiting later if NVIDIA lands a fix.
- **FluidAudio** (FluidInference/FluidAudio, Swift, MIT/Apache-2.0): a fully native Apple-platform (macOS 14+/iOS 17+, Apple Silicon only) speaker diarization + embedding SDK that runs pyannote-derived models through CoreML on the **Apple Neural Engine** rather than CPU/GPU — explicitly built on top of sherpa-onnx's diarization approach, then re-converted to CoreML for ANE execution. It's already used in shipping meeting-transcription apps. This is likely the *best raw performance/power option on Apple Silicon* (ANE beats CPU ONNX Runtime for sustained batch workloads), and it has zero Python dependency. The catch: it's a **Swift** library, not Rust — embedding it inside meethook's Rust binary means bridging across a Swift/Rust FFI boundary (a small C-ABI shim or an XPC helper), not a native Rust crate. That's a real integration cost (second toolchain, Swift Package Manager in the build), but it is fundamentally different from a Python subprocess dependency: no separate runtime/interpreter to launch or manage, no IPC protocol to design — it's a compiled native library call, closer in spirit to linking libonnxruntime than to shelling out to a Python process. Worth flagging as a strong follow-up path if ANE-level performance/power becomes a priority later, but out of scope for a first pass given the added toolchain complexity.

## 4. Recommendation

**Diarization:** `pyannote/segmentation-3.0`, via the ungated MIT `onnx-community` ONNX mirror, run through ONNX Runtime.

**Speaker identification:** a WeSpeaker ONNX embedding model (ECAPA-TDNN or ResNet34-LM variant), with enrollment done as mean+L2-normalized reference embeddings per known speaker and cosine-similarity matching (with an "unknown" threshold) at inference time.

**Runtime:** the `ort` crate (Rust bindings for ONNX Runtime) with the `coreml` execution-provider feature enabled, so both models get Apple Neural Engine/GPU acceleration on Apple Silicon through one runtime stack. `pyannote-rs` (thewh1teagle) already demonstrates this exact pipeline end-to-end (segmentation-3.0 + wespeaker-voxceleb-resnet34-LM + `ort` + CoreML) and can be used directly or as a reference implementation to vendor/fork given its modest (125-star) maintenance footprint. sherpa-onnx is the fallback if a more heavily-maintained, officially-supported toolkit is preferred — its Rust bindings additionally ship a ready-made named-speaker-embedding index that maps directly onto the "fixed set of known speakers" requirement.

**No Python dependency anywhere in this recommended path**, at build time or runtime — all weights are pre-converted ONNX artifacts consumed directly by Rust/ONNX Runtime. The one caveat: the *newest* pyannote embedding checkpoint itself cannot currently be ONNX-exported (torchaudio fbank tracing failure); the recommendation sidesteps this by using WeSpeaker's embedding model instead, which is what the existing Rust implementations already do.

**Flagged but not recommended for v1:** NeMo Sortformer (ONNX export currently broken upstream) and FluidAudio (best likely Apple Silicon performance via ANE, zero Python, but Swift-native — would require a Swift/Rust FFI bridge rather than fitting the "single Rust binary, ONNX/candle" constraint as directly; worth a follow-up evaluation if CPU-based ONNX Runtime performance/power turns out insufficient in practice).

## Sources

- https://github.com/pyannote/pyannote-audio/discussions/1929
- https://huggingface.co/onnx-community/pyannote-segmentation-3.0
- https://huggingface.co/pyannote/segmentation-3.0
- https://github.com/pengzhendong/pyannote-onnx
- https://huggingface.co/deepghs/pyannote-embedding-onnx
- https://github.com/thewh1teagle/pyannote-rs
- https://github.com/RustedBytes/pyannote-rs
- https://crates.io/crates/native-pyannote-rs
- https://huggingface.co/OpenASR/pyannote-segmentation-3.0
- https://github.com/k2-fsa/sherpa-onnx
- https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html
- https://docs.rs/sherpa-onnx/latest/sherpa_onnx/
- https://github.com/wenet-e2e/wespeaker
- https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md
- https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM
- https://arxiv.org/abs/2210.17016 (WeSpeaker paper)
- https://arxiv.org/abs/2005.07143 (ECAPA-TDNN paper)
- https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html
- https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_recognition/models.html
- https://github.com/NVIDIA-NeMo/NeMo/issues/15536
- https://github.com/NVIDIA-NeMo/NeMo/issues/15077
- https://github.com/NVIDIA-NeMo/NeMo/issues/14733
- https://github.com/FluidInference/FluidAudio
- https://github.com/FluidInference/FluidAudio/blob/main/README.md
- https://crates.io/crates/ort
- https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
