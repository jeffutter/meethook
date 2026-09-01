# Building and running meethook on Linux

The workspace builds and tests on Linux (x86_64), and `meethook transcribe` runs
end-to-end there on CPU Whisper (whisper.cpp) and CPU ONNX Runtime. What does not
come along:

- **`record`** exists only on macOS. The `meethook-record` crate is pure Apple
  frameworks (ScreenCaptureKit, EventKit, ...) and lives in its own standalone
  workspace; the binary pulls it in through a target-gated dependency, so on Linux
  the subcommand simply does not exist.
- **Calendar-backed halves of `meeting`** are macOS-only too. On Linux
  `meeting <id>` lists no candidates and points at `--clear`, which never consults
  the calendar and works everywhere.
- **Accelerators** (Metal/CoreML) are macOS-only by construction; off macOS the
  pipeline reports `accelerated = false` and runs on CPU. That is expected, not a
  failure.
- **Enroll clip playback** falls back from `afplay` to the first of
  `paplay`, `aplay`, `ffplay`, `mpv` found on `PATH`; with none of them present,
  enrollment degrades to text snippets instead of failing.

## The easy way: `nix develop`

The flake defines an `x86_64-linux` devShell alongside the macOS one. It carries
the whole toolchain -- Rust (with clippy/rustfmt/rust-analyzer), cmake, pkg-config,
onnxruntime, meson, ninja, clang, plus lefthook, cargo-nextest, cargo-audit and
cargo-outdated --
and sets the two environment variables the build needs (`LIBCLANG_PATH` for
bindgen, `LD_LIBRARY_PATH` for the onnxruntime and libstdc++ shared libraries,
which `nix develop` would otherwise leave unset). Entering the shell also runs
`lefthook install`, so the pre-commit/pre-push gates activate automatically.

```sh
nix develop
cargo nextest run --all-features --workspace
cargo run -p meethook -- transcribe --help
```

Model weights are not part of the Nix closure; `transcribe` downloads them on
first use into `<root>/models/` (default root `~/meethook`).

## Without Nix

The same requirements, by hand:

| Need | Why |
| --- | --- |
| A recent Rust toolchain + `clippy`, `rustfmt` | the repo gates are `-D warnings` |
| `cargo-nextest` (e.g. `cargo binstall cargo-nextest`) | the pre-push test gate runs the suite through it |
| `cmake` and a C/C++ toolchain | whisper.cpp compiles from source |
| `meson`, `ninja`, `clang` | off macOS the `webrtc-audio-processing` crate builds AEC3 from source, and its `meson.build` hard-codes `clang` as the compiler |
| `libclang` on `LIBCLANG_PATH` | bindgen invokes libclang directly, outside any cc wrapper |
| `libonnxruntime` findable via `pkg-config` | `ort-sys` probes for it at build time and links it dynamically |

At *run* time the ELF binaries additionally need `libonnxruntime.so.1` and
`libstdc++.so.6` resolvable; if your package manager does not put both on the
standard loader path, add their directories to `LD_LIBRARY_PATH`.

Then the usual:

```sh
cargo build --workspace
cargo nextest run --all-features --workspace
cargo run -p meethook -- transcribe [SESSION_ID]...
```

## Gates

`lefthook.yml` runs the record-crate steps (fmt/clippy/test on
`crates/meethook-record`) only on Darwin; on Linux those steps print a skip notice
and pass, and the rest of the chain (fmt, clippy, test, doc, audit) runs as
usual.

## Downloaded release binaries

The `.tar.gz` binaries the CD workflow (`.github/workflows/cd.yml`) attaches to
each GitHub release are plain `cargo build --release` output -- not the Nix
package below -- so they carry the same dynamic-linking requirements as
"Without Nix" above: `libonnxruntime` findable via the loader (on both
platforms), plus `libwebrtc_audio_processing` on macOS specifically (Linux
compiles AEC3 from source into the binary instead, so it has no such runtime
dependency there). Neither is bundled in the tarball. A machine without those
libraries already installed -- via Nix, Homebrew, or a distro package -- will
fail to start the binary with a dynamic-linker error, not a clean one.

`nix build .#meethook` (or `nix profile install`/`nix run` against this flake)
is the reliable install path: it resolves those libraries from the same Nix
store the build already depends on, rather than assuming the target machine
happens to have them. Prefer it over the tarball unless you already know the
target machine has onnxruntime (and, on macOS, webrtc-audio-processing)
available outside Nix.
