# Third-party license notices

The licenses of what meethook includes (MIT, BSD, Apache-2.0) require their notices to accompany
the binary. The macOS release is a single file, so the notices are built into it: `meethook
licenses` prints them. The same text is `THIRD_PARTY_LICENSES.md` at the repository root, and the
release workflow attaches it to each GitHub release.

`THIRD_PARTY_LICENSES.md` is **generated**; do not edit it. CI (`licenses` job) fails when it is
out of date.

```
python3 scripts/gen_licenses.py            # regenerate (needs cargo-about; `nix develop` has it)
python3 scripts/gen_licenses.py --check    # what CI runs
```

It has two parts.

## Rust crates: `cargo about`

Configured by `about.toml`, rendered by `about.hbs`. It reads cargo metadata for both release
targets (`aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`, so the output does not depend on the
machine that runs it) with all features, ignoring build and dev dependencies, which are not linked
into the binary, and our own unpublished crates.

- **A new dependency** changes the output: regenerate and commit. If `cargo about` stops with an
  unaccepted license, read that license before adding it to `accepted` in `about.toml`; the list
  is deliberately short and has nothing copyleft in it.
- **A crate that bundles C or C++ source** (a `-sys` crate) is covered by `cargo about` only for
  the crate's own license, not for the bundled code. Those go in the native part below. Today:
  `whisper-rs-sys` (whisper.cpp), `mlx-sys` (MLX and what MLX fetches), `zstd-sys` (zstd).

## Native code: `licenses/native/`

Verbatim upstream texts, listed with their version and how each reaches the binary in `NATIVE` in
`scripts/gen_licenses.py`.

| File(s) | Source |
|---|---|
| `onnxruntime.*` | the official `onnxruntime` 1.31.0 macOS wheel on PyPI (`LICENSE`, `ThirdPartyNotices.txt`), matching the version of the static build `ort-sys` downloads |
| `mlx.LICENSE`, `fmt.LICENSE`, `nlohmann-json.LICENSE`, `gguflib.LICENSE`, `metal-cpp.LICENSE` | the MLX build tree: `target/.../build/mlx-sys-*/out/build/_deps/*-src/` |
| `mlx-c.LICENSE` | `mlx-sys-0.7.0/src/mlx-c/LICENSE` in the cargo registry |
| `whisper.cpp.LICENSE` | `whisper-rs-sys-*/whisper.cpp/LICENSE` in the cargo registry |
| `zstd.LICENSE` | `zstd-sys-*/zstd/LICENSE` in the cargo registry (zstd's BSD license) |
| `VoiceIsolatorMLX.*` | <https://github.com/starkdmi/VoiceIsolatorMLX> at the commit named in `scripts/gen_licenses.py` |

**When one of those is upgraded** (a new ONNX Runtime from `ort`, a new MLX from `mlx-sys`, a
new `whisper-rs-sys`): replace the text from the new source, update the version in `NATIVE`, and
regenerate. Nothing checks these versions automatically, so a bump of `ort`, `mlx-sys`,
`whisper-rs-sys` or `zstd-sys` in `Cargo.toml` is the cue. `MLX.md` lists what else changes with
`mlx-sys`.

Not covered, on purpose: the model weights meethook downloads at run time (they are separate works
and not part of the binary), and system libraries and Apple frameworks it links on macOS.
