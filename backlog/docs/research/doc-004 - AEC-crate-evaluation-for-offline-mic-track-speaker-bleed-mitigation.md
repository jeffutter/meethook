---
id: doc-004
title: AEC crate evaluation for offline mic-track speaker-bleed mitigation
type: other
created_date: '2026-08-09 04:53'
updated_date: '2026-08-09 04:54'
---
# AEC Crate Evaluation for TASK-001.13 — Offline Batch Mic-Track Echo Cancellation

Research conducted against primary sources: crates.io API, GitHub API (commits/issues/releases), raw source files, docs.rs, and the nixpkgs source tree. All three named candidates were confirmed to exist, though one's identity differs from the guessed name.

---

## Candidate 1: `aec3` (repo: `RubyBit/aec3-rs`)

**Identity:** Crate name is `aec3` (not `aec3-rs` — that's the repo name). Repo: [github.com/RubyBit/aec3-rs](https://github.com/RubyBit/aec3-rs). Crate: [crates.io/crates/aec3](https://crates.io/crates/aec3). Author: Angelos Mangos (GitHub `RubyBit`). Confirmed as "an acoustic echo canceller written in rust based on the WebRTC aec3 project."

**Maturity:** Very young. Repo created 2025-11-18, crate first published same date. Latest version 0.3.1 published 2026-06-24; most recent GitHub commit also 2026-06-24. 44 stars, 4 forks, 1 open issue, 12 published versions, ~23.5K total downloads (~17K "recent"), so adoption is growing but the project is under 9 months old. The README **explicitly self-describes as work-in-progress**: "NOTE: This is a work in progress and the API is expected to evolve," and the maintainer says they're "still validating internally if this design is useful." As of v0.2 it underwent a full internal architecture rewrite (fixed pipeline → generic graph execution model), which is a signal of churn, not stability.

**Offline-batch fitness:** Supported via `examples/file_to_file.rs`, but there is no single whole-buffer call — like every AEC implementation checked in this research, processing is strictly frame-based (10ms frames via `AudioFormat::ten_ms`). The pattern is: build a `linear::builder(...).build()` pipeline, then loop calling `pipeline.handle_render_frame(...)` then `pipeline.process_capture_frame(...)` per 10ms chunk, zero-padding the tail. This is trivial to adapt for a batch tool but requires writing that loop yourself — no "process(full_mic_buf, full_speaker_buf) -> Vec<f32>" convenience API exists.

**Output-quality evidence:** Thin. There is a `cargo test` suite but no benchmark suite and no documented comparison against reference WebRTC AEC3 test vectors or output. The README calls it "aligned with WebRTC reference algorithms" but does not cite validation against WebRTC's own C++ conformance/unit tests. This is meaningfully weaker evidence than exists for the C++-backed candidate (see below) or for third-party pure-Rust alternatives like `dignifiedquire/sonora`, which explicitly validates against "the C++ reference test suite (WebRTC M145, 2400+ tests)" — [github.com/dignifiedquire/sonora](https://github.com/dignifiedquire/sonora) (surfaced during research; not one of the three named candidates, flagged here only as an aside worth knowing about, not evaluated in depth per scope).

**Nix-packageability:** Excellent on paper. Confirmed via `crates.io/api/v1/crates/aec3/0.3.1/dependencies` that the crate's only non-dev dependencies are `crossbeam-channel`, `log`, `num-complex`, `rustfft` — all pure-Rust, no `-sys` crate, no `build.rs`/`links` key. This means `cargo build` works with zero C/C++ toolchain, trivially buildable in a Nix flake via naersk/crane with no OS-specific glue.

**Minor caveat:** GitHub's repo metadata reports license `NOASSERTION` despite crates.io showing a proper SPDX license (`MIT OR BSD-3-Clause`) — likely just GitHub's license-detector choking on a non-standard LICENSE file layout (the repo also ships a separate `PATENT` file, mirroring WebRTC's own BSD-3-Clause + PATENTS structure — a sign of licensing diligence, not a red flag, but worth a manual look before depending on it).

---

## Candidate 2: `webrtc-audio-processing` (repo: `tonarino/webrtc-audio-processing`)

**Identity:** Confirmed. Crate: [crates.io/crates/webrtc-audio-processing](https://crates.io/crates/webrtc-audio-processing), repo [github.com/tonarino/webrtc-audio-processing](https://github.com/tonarino/webrtc-audio-processing). This wraps the **PulseAudio project's repackaging of WebRTC's real production C++ Audio Processing Module (APM)** — not a reimplementation. A lower-level `webrtc-audio-processing-sys` crate provides the raw FFI. Maintained by Tonari, Inc.

**Maturity:** By far the most established of the three. GitHub repo created 2019-12-05 (~6.5 years old), 323 stars, 48 forks, 189+ commits, actively pushed as recently as 2026-07-16 (`Fix clippy 1.97 lints (#101)`). Crate has 16 published versions on crates.io going back to 2019-12-09, latest `2.1.0` released 2026-05-13, ~90.7K total downloads. License: BSD-3-Clause. This is a mature, continuously-maintained project by an organization actually using it in production (Tonari builds telepresence hardware).

**C++ dependency shape:** The WebRTC APM C++ source is vendored as a **git submodule** in the repo (PulseAudio's `webrtc-audio-processing` fork, tracked at GitLab freedesktop.org). Two build modes:
- **Default (dynamic link):** links against a system-installed `webrtc-audio-processing` library (e.g. via `apt`/`pacman`/nixpkgs), found via pkg-config.
- **`bundled` feature:** compiles the vendored C++ source itself (needs clang/gcc, pkg-config, meson, ninja), with symbol-mangling so multiple major versions can coexist.

The crate's major version deliberately tracks the upstream PulseAudio APM major version (currently 2.x ↔ v2.1), so this isn't a random re-vendoring — it's a thin, versioned wrapper over the actual upstream library.

**Offline-batch fitness:** Confirmed via `src/lib.rs` — `Processor::new(sample_rate_hz)`, then `process_render_frame(&self, frame: F) where F: IntoIterator<Item: AsMut<[f32]>>` for the far-end/speaker reference and `process_capture_frame(...)` for the near-end/mic signal (mutated in place to become the cleaned output). Also `analyze_render_frame` variant. Like all AEC APIs surveyed, it is **frame-based at a fixed size**: exactly `num_samples_per_frame()` (`sample_rate_hz / 100`, i.e. 10ms, hardcoded in WebRTC) — the docs state it explicitly **panics** if you pass the wrong length. Again, no single whole-buffer call exists; you write a simple loop feeding 10ms frames from your two WAV buffers. This is the cleanest, best-documented API of the three (thread-safe, `Send + Sync`, explicit panic contracts, `set_config`/`get_stats`/`reinitialize` for runtime control).

**Output-quality evidence:** Strongest of the three by a wide margin, precisely because this *is* WebRTC's real AEC3 C++ implementation — the same code Chrome/production softphones ship — not a reimplementation that must be independently validated. Quality claims rest on WebRTC's own multi-year production track record rather than a young port's self-testing.

**Nix-packageability:** Confirmed nixpkgs has this exact library, matching version, with Apple Silicon support. Found at `pkgs/by-name/we/webrtc-audio-processing/package.nix` (plus `_1` and `_0_3` variants for other major versions consumers pin to): version **2.1**, sourced from GitLab (`pulseaudio/webrtc-audio-processing` tag `v2.1`), build via `meson`/`ninja`/`pkg-config`, `buildInputs = [ abseil-cpp ]`. `meta.platforms` intersects upstream arch support (incl. `aarch64`) with upstream OS support (incl. `darwin`) → **aarch64-darwin is supported**. This means the flake devShell can simply add `pkgs.webrtc-audio-processing` (v2.1, matching the crate's default dynamic-link expectation) + `pkg-config` to build inputs and skip the crate's `bundled` C++ compile path entirely.

**Caveat found:** Two relevant open issues — [#90 "Documentation build on docs.rs fails"](https://github.com/tonarino/webrtc-audio-processing/issues/90) (Feb 2026, docs.rs sandbox lacks build deps — not a real usage blocker) and, more importantly, [**#102 "Support MSVC targets, and fix Apple cross-compilation, in bundled builds"**](https://github.com/tonarino/webrtc-audio-processing/issues/102) filed 2026-08-08, which explicitly flags Apple cross-compilation problems in the **`bundled`** build path specifically. This reinforces that the right integration path for this project is dynamic-linking against the nixpkgs-provided `webrtc-audio-processing` v2.1 package (which nixpkgs already builds correctly for aarch64-darwin via meson), **not** enabling the crate's `bundled` feature — that sidesteps the reported Apple build issue entirely.

---

## Candidate 3: `aec-rs` (repo: `thewh1teagle/aec`)

**Identity:** Confirmed, published as `aec-rs` on crates.io ([crates.io/crates/aec-rs](https://crates.io/crates/aec-rs)) and mirrored as `pyaec` on PyPI. Repo: [github.com/thewh1teagle/aec](https://github.com/thewh1teagle/aec). MIT licensed. This is FFI bindings to speexdsp's echo-canceller (the MDF algorithm), confirmed directly from source.

**Maturity — the weakest of the three.** Repo created 2024-12-06. Crate has only **2 published versions ever** (0.1.0 and 1.0.0, both in Dec 2024). Last GitHub push was 2025-04-25 (a README update + adding a LICENSE file) — over 15 months stale as of the research date. 95 stars, 18 forks, 7 open issues, 0 open PRs. The published 1.0.0 crate contains only **72 lines of Rust code** (per crates.io's own linecount metadata) — it is a minimal, thin FFI shim, not a substantial implementation of its own.

**Source confirmed via `src/lib.rs`:** the entire public API is:
```rust
pub struct AecConfig { pub frame_size: usize, pub filter_length: i32, pub sample_rate: u32, pub enable_preprocess: bool }
pub struct Aec { /* raw speex_echo_state / speex_preprocess_state pointers */ }
impl Aec {
    pub fn new(config: &AecConfig) -> Self
    pub fn cancel_echo(&self, rec_buffer: &[i16], echo_buffer: &[i16], out_buffer: &mut [i16])
}
```
It's a direct `unsafe` pass-through to `speex_echo_cancellation`/`speex_preprocess_run` via a `-sys` crate, with **no length validation or chunking logic** — the caller must supply exactly `frame_size`-length `i16` slices (default 160 samples @16kHz, `filter_length: 1600`). Uses `i16` PCM, not `f32` — an extra conversion step versus the other two candidates if the pipeline is float-based.

**Offline-batch fitness:** Workable but the thinnest wrapper of the three — same frame-loop requirement as the others, but with the least safety/ergonomics (raw pointers, manual `Drop`, no bounds checking) and an `i16` rather than `f32` interface.

**Output-quality evidence:** None found beyond the author's own description. No tests, benchmarks, or third-party validation surfaced in the repo or README.

**Algorithmic quality vs. AEC3 — is "generally lower quality" substantiated?** Partially, and with an important nuance. Speex's echo canceller is a classical NLMS-style Multi-Delay Filter (MDF) — a single adaptive filter with no secondary nonlinear residual-echo suppression stage. WebRTC's own stated design goal for AEC3 (replacing AEC2, WebRTC's own MDF-family predecessor) was explicitly to leak much less echo than the old NLMS-style approach while preserving transparency, via a multi-stage pipeline (delay estimation robust to clock drift/buffering variance, adaptive linear filter, plus a nonlinear residual suppressor for what the linear filter can't model, e.g. speaker distortion on cheap hardware). This is a legitimate, sourceable engineering rationale, not just a vague reputation — see the WebRTC team's own description of AEC3's goals ([switchboard.audio/hub/how-webrtc-aec3-works](https://switchboard.audio/hub/how-webrtc-aec3-works/), [webrtc.github.io blog on AEC evolution](https://webrtc.github.io/webrtc-org/blog/2011/07/11/webrtc-improvement-optimized-aec-acoustic-echo-cancellation.html)). Countervailing evidence: at least one practitioner report (Google's own `discuss-webrtc` group) found Speex correctly cancelling echo in a scenario where AEC3 flattened/muted the mic output — suggesting AEC3 is more sensitive to correct configuration (sample rate, frame alignment) and can fail ungracefully if misconfigured, whereas MDF degrades more predictably. **Net assessment: "lower quality than AEC3" is a reasonably well-founded characterization for well-configured use, but not an absolute law — AEC3 buys more echo suppression at the cost of being pickier about correct setup**, which matters for an offline batch tool where you fully control sample rate/frame alignment and can validate output before committing to it.

**Nix-packageability:** Straightforward. speexdsp confirmed in nixpkgs at `pkgs/development/libraries/speexdsp/default.nix`, version 1.2.1, BSD-3-Clause, `meta.platforms = lib.platforms.unix ++ lib.platforms.windows` (covers aarch64-darwin), built via `autoreconfHook` + `pkg-config`, only extra dep is optional `fftw`. This is the simplest C dependency of the three to add to a flake — but the crate wrapping it is the least maintained.

---

## Recommendation

**Primary choice: `webrtc-audio-processing` (tonarino), linked against nixpkgs' `webrtc-audio-processing` v2.1 package via the default (non-`bundled`) dynamic-link path.**

Rationale, weighing correctness (the dominant factor, since this feeds ASR accuracy and CPU cost is a non-issue in offline batch):

- It is the **only candidate running the actual, production-proven WebRTC AEC3 C++ implementation**, not a young/independent reimplementation whose correctness is unverified against reference behavior. For a step that materially affects downstream ASR accuracy, this outweighs the marginal Rust-build-purity advantage of `aec3`.
- nixpkgs already builds this exact library (v2.1, matching the crate's expected major version) for aarch64-darwin, via a standard meson/ninja/pkg-config recipe — no bespoke C++ build needs to be authored in the flake, and the specific Apple-cross-compilation bug reported in tonarino/webrtc-audio-processing#102 only affects the crate's own `bundled`-C++-build feature, which this integration path avoids entirely.
- The API is clean, documented, `Send + Sync`, and its 10ms-frame requirement is a universal constraint of every AEC library evaluated here (including the other two candidates) — not a differentiator.
- Downside: it's an FFI crate (unlike `aec3`'s pure-Rust build), so the flake needs `pkg-config` + the nixpkgs library as a build input rather than a bare `cargo build`. This is a small, well-trodden Nix pattern (same shape already accepted for `ort`/onnxruntime in TASK-001.03), not a real obstacle.

**Fallback: `aec3` (RubyBit/aec3-rs)** if the tonarino crate proves unworkable during implementation (e.g. FFI/linking friction against the nixpkgs package turns out worse than expected, or the panic-on-frame-mismatch contract collides badly with resampling/edge-case WAV files). It's pure Rust with zero C dependencies, so build friction risk is near zero — but budget time to validate its AEC3 port's correctness (e.g. spot-check output against a known-echo test recording) since no third-party validation evidence exists yet for this young (9-month-old), self-described work-in-progress project.

**Do not use `aec-rs` (thewh1teagle/aec)** as primary: it's a stale (15+ months untouched), extremely thin (72 LOC) unsafe FFI shim around speex's older MDF algorithm, with no tests/benchmarks of its own. If both other options fail outright, speexdsp itself is easy to package in Nix, but at that point it would be worth writing a slightly more careful safe wrapper directly against `speexdsp` (or `rust-av/speexdsp-rs`, which surfaced during research as a more actively maintained alternative offering both bindings and a pure-Rust implementation) rather than depending on this specific abandoned crate.
