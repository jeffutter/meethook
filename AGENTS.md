# AGENTS.md

This file provides guidance to AI coding agents (including Claude Code, via `CLAUDE.md`)
when working with code in this repository.

## What this is

`meethook` is a local-first macOS/Linux meeting recorder and transcriber: it records both
sides of a call to independent WAV tracks, diarizes and transcribes them, matches speakers
against an enrolled-voice database, and renders a `transcript.md`. The design record for
*why* lives in `backlog/decisions/decision-001` through `decision-012` — read the relevant
one before making an architectural change in that area (ASR, diarization, speaker matching,
reference lifecycle, echo cancellation, calendar fit, transcript rendering, mixdown,
interactive enroll UI).

## Commands

Enter the dev environment first — it provides the pinned Rust toolchain, `cargo-nextest`,
`lefthook`, and (off macOS) the C/C++ toolchain the native deps need:

```sh
nix develop        # `.envrc` also does this automatically via direnv
```

```sh
cargo build --workspace
cargo nextest run --all-features --workspace     # full suite (matches the pre-push gate)
cargo nextest run -p meethook-transcribe          # one crate
cargo nextest run -p meethook-session -- speakers # one test by substring
cargo clippy --all-targets --all-features --workspace -- -D warnings
rustfmt --edition 2024 <files>                    # pre-commit uses bare rustfmt, not `cargo fmt`
cargo fmt --all --check                           # what pre-push actually runs
cargo doc --no-deps --document-private-items --all-features --workspace   # RUSTDOCFLAGS=-D warnings at push time
cargo run -p meethook -- transcribe --help
```

`crates/meethook-record` is **excluded** from the root workspace (see the comment in
`Cargo.toml`) because it's pure Apple-framework bindings that don't compile off macOS. On
Darwin it roots its own workspace and its own gates run from inside its directory:

```sh
cd crates/meethook-record && cargo fmt --all --check && cargo clippy --all-targets --workspace -- -D warnings && cargo nextest run --workspace
```

`lefthook.yml` encodes the exact gates and their order (fmt → clippy → test → doc → audit,
plus the record-crate variants, each skipped with a printed notice off Darwin). Read it
before assuming what "the gates" check. `lefthook install` runs automatically on `nix
develop` entry. `cargo audit` needs real network/fs access and is expected to fail in a
sandboxed agent run — use `LEFTHOOK_EXCLUDE=audit lefthook run pre-push` there.

Useful runtime env vars: `MEETHOOK_ROOT` (data dir, default `~/meethook`; `--root` overrides
it), `MEETHOOK_TEMPLATE` (transcript template override), `MEETHOOK_CPU=1` (opt out of
GPU/CoreML acceleration — see `meethook-transcribe::gpu`, which otherwise hard-fails rather
than silently falling back).

See `LINUX.md` for what does and doesn't work off macOS (no `record`, no calendar
correction, no accelerators, clip playback falls back through `paplay`/`aplay`/`ffplay`/`mpv`).

## Architecture

Five crates, one binary. **`record`, `transcribe`, and `enroll` run as entirely separate
processes with no IPC and no shared state — they only ever communicate through the on-disk
session directory.** That contract is the thing to understand before touching any of them.

- **`meethook-session`** — the on-disk contract itself: session directory layout, ids,
  atomic writes, and every JSON schema (`session.json`, `speaker_clusters.json`,
  `speaker_names.json`, `speakers.json`, `transcript.json`/`.md`, `cleaning.json`). Its
  module doc lays out the full directory tree. Every other crate depends on this one and on
  nothing else in the workspace for shared state — a layout change is a change to this one
  crate.
- **`meethook-record`** (macOS only, own workspace) — dual-track capture via ScreenCaptureKit
  (speaker/system audio) and a separate `AVAudioEngine` input tap (mic), plus calendar
  lookup via EventKit for meeting-fit guessing. Deliberately two independent streams rather
  than one unified capture, both to dodge a known macOS 15 corruption bug and because echo
  cancellation later needs the speaker track as an independent reference signal.
- **`meethook-transcribe`** — the batch pipeline: AEC pre-pass, VAD (Silero), diarization
  (ONNX), speaker embedding/matching, ASR (whisper.cpp via `whisper-rs`), turn merging, and
  the `mixdown` module (public, unlike its siblings) that produces `meeting.opus`. The mic
  track is never diarized — there's exactly one local speaker, always labelled `You`; the
  speaker track is diarized and each turn attributed to a distinct voice.
- **`meethook-enroll`** — the one interactive path in the tool, but built so almost none of
  it actually is: which sessions/voices to ask about, what an answer writes, and what a
  correction costs are all decided here against two one-method seams (`Interviewer` for
  naming, `MeetingSource` for calendar correction) with no terminal or audio device on this
  side. The live terminal implementation lives in the CLI crate; enroll's own tests answer
  from a script. Also owns the `speakers`/`forget` read/report and removal logic, and
  `resolve()`, which decides what a typed name means (exact/case-sensitive match, shortlist,
  or new person — never an automatic pick between two candidates).
- **`meethook-models`** — lazy, hash-verified download of model weights into
  `<root>/models/`. Knows nothing about whisper, ONNX, sessions, or transcripts; just turns a
  `ModelSpec` into a verified local path, with progress reported through a callback rather
  than printed directly (batch commands must never prompt).
- **`meethook`** (bin crate) — the CLI (`clap`) and everything that can't be tested without a
  terminal: printing, prompting, playing audio clips, and the full-screen `enroll` TUI
  (`ratatui`, under `src/screen/`). Subcommand bodies in `commands.rs` are thin dispatch onto
  the library crates above.

### Invariants worth knowing before changing code

- A rewritten transcript is always exactly what `transcribe --force` would now produce —
  `enroll` and `transcribe` must never become two sources of truth about a transcript.
  `enroll` brings any stale transcript up to date on the way past, even for sessions where
  nothing was asked.
- "Unresolved" is decided against `speakers.json` as it stands *right now*, never against
  transcript text — deduplication of who's who across sessions is enrollment itself.
- Every enroll answer/removal is preceded by a preview path (`Voice::preview`,
  `Consequence`, `Forget`'s cost report) that runs the same logic as the real write over a
  copy, so a preview and a write can't drift apart. Prefer extending that shared path over
  adding a second computation for `--dry-run`/`--list`/report output.
- An orphaned session (WAVs with no `session.json` — a crash mid-recording) is a normal,
  expected classification throughout the codebase, never an error.
- Numeric CLI options that come from measured/settled constants (mixdown pan, LUFS target,
  boost cap, bitrate) are range-refused at the parser edge rather than clamped — clamping a
  user-typed value silently changes what they asked for.

### Code style already in force here

This codebase writes unusually long, load-bearing comments — module docs and inline comments
routinely explain *why* a decision was made (a measurement, a rejected alternative, a bug
worked around), not just what the code does. Match that when the reasoning is genuinely
non-obvious; don't add narration for things the code already states plainly. Read a module's
top-of-file doc comment before editing it — the constraints it exists under are usually
recorded there rather than rediscoverable from the code alone.
