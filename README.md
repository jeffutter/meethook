# meethook

A local-first meeting recorder and transcriber. `meethook` records both sides of a call to
independent audio tracks, transcribes and diarizes them, matches speakers against voices you've
enrolled, and renders a `transcript.md` — all on your own machine, with no audio or transcript
ever leaving it.

- **Record** (macOS only) — watches the default microphone and records each call as a session,
  capturing your mic and the system/speaker audio as two independent tracks.
- **Transcribe** (macOS and Linux) — runs voice-activity detection, diarization, speaker
  matching, and Whisper ASR over a recorded session, then renders `transcript.md` and a
  compressed `meeting.opus` mixdown.
- **Enroll** — names the voices transcription couldn't identify, either interactively (a
  full-screen terminal UI) or by scripted answer, and keeps every existing transcript in sync
  as you do.

Recording depends on Apple's ScreenCaptureKit and EventKit frameworks and is macOS-only.
Transcription runs on both macOS and Linux — see [LINUX.md](./LINUX.md) for exactly what does
and doesn't come along off macOS.

## Installing

### With Nix (recommended)

The flake resolves `meethook`'s runtime dependencies (onnxruntime, and on macOS
webrtc-audio-processing) from the Nix store, so the binary it produces doesn't depend on
anything already being installed on the target machine:

```sh
nix profile install github:jeffutter/meethook
# or, to try it without installing:
nix run github:jeffutter/meethook -- --help
```

### Release binaries

Prebuilt binaries for macOS (arm64) and Linux (x86_64) are attached to each
[GitHub release](https://github.com/jeffutter/meethook/releases). They're built with a plain
`cargo build --release`, so unlike the Nix package they are **not** self-contained: the machine
running them needs `libonnxruntime` on the loader path already (and, on macOS,
`libwebrtc-audio-processing`), or the binary will fail to start with a dynamic-linker error.
Prefer the Nix install above unless you already have those libraries from somewhere else (Nix,
Homebrew, or your distro's package manager).

```sh
tar xzf meethook-vX.Y.Z-<os>-<arch>.tar.gz
./meethook --help
```

### From source

```sh
git clone https://github.com/jeffutter/meethook.git
cd meethook
nix develop            # pulls in the pinned Rust toolchain and every native dependency
cargo build --release --workspace
./target/release/meethook --help
```

Building without Nix is possible but means installing the native toolchain (cmake, a C/C++
compiler, meson/ninja/clang, libclang, libonnxruntime) yourself — see
[LINUX.md](./LINUX.md#without-nix) for the full list and why each one is needed.

Model weights (Whisper, diarization, speaker embedding) are downloaded on first use into
`<data dir>/models/` rather than bundled — see [Data directory](#data-directory) below.

## Usage

A typical session looks like:

```sh
meethook record                    # macOS: leave running, join your calls as usual
# ... later, once you've recorded some meetings ...
meethook transcribe                # transcribe every session that doesn't have one yet
meethook enroll                    # name any voices it couldn't identify
```

`transcribe` and `enroll` both work over every discovered session by default, or over specific
sessions if you pass session ids (the directory name each one is recorded under, e.g.
`20260809-052600`).

### `meethook record`

*macOS only.* Watches the default microphone and records each call as a session — your mic and
the system/speaker audio as two independent tracks — until interrupted (Ctrl-C). Takes no
options; there's nothing to configure that the tool can't detect itself.

### `meethook transcribe [SESSION_ID...]`

Transcribes recorded sessions: an AEC pre-pass, voice-activity detection, diarization, speaker
matching against your enrolled voices, Whisper ASR, and turn merging, then writes
`transcript.md`/`transcript.json` and a `meeting.opus` mixdown. With no session ids, every
discovered session that doesn't already have a transcript is considered.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--force` | off | Re-transcribe sessions that already have a transcript |
| `--bitrate <BPS>` | 32000 | Bitrate of the `meeting.opus` mixdown, in bits per second |
| `--pan <WIDTH>` | 0.3 | How far each track is panned from centre: `0.0` is mono, `1.0` is hard left/right |
| `--target-lufs <LUFS>` | -16 | Loudness each track is normalized to before mixing, in LUFS |
| `--max-boost-db <DB>` | 18 | Most a quiet track may be turned up on its way to the target, in dB |

`--pan`, `--target-lufs`, and `--max-boost-db` are refused outside the range the mixdown
arithmetic admits, rather than silently clamped, so a mistyped value is reported instead of
quietly reinterpreted.

### `meethook enroll [SESSION_ID...]`

Names speakers that transcription couldn't identify. With no session ids, every session with
unresolved speakers is considered. Opens a full-screen terminal UI by default, or a plain
line-by-line prompt when run from a script or without `--plain` on a real terminal (a
non-terminal stdin/stdout falls back to plain automatically).

| Flag | Meaning |
| --- | --- |
| `--voice <VOICE>` | Ask about one voice only — either its number (`"Unknown 3"` → `3`) or its current name. Requires exactly one session id. |
| `--at <MM:SS>` | Ask about whoever was speaking at this timestamp, exactly as `transcript.md` prints it. Requires exactly one session id; conflicts with `--voice`. |
| `--name <NAME>` | Answer with this name instead of prompting. Requires `--voice` or `--at`. |
| `--all` | Ask about every unresolved voice, including ones normally too quiet to be offered. |
| `--correct` | Also ask about voices already named, to fix a wrong identification. |
| `--force-reference` | Store a reference even from a voice too short to make a reliable one (otherwise such a voice is named for that session only). |
| `--plain` | Ask line by line instead of opening the full-screen UI. |
| `--one-speaker <NAME>` | Assert the whole session's speaker track is one person and name every voice on it, without asking. Requires exactly one session id; conflicts with `--voice`/`--at`/`--name`. |
| `--list` | Print every voice this run would offer, ranked by resemblance to enrolled speakers, and write nothing. |
| `--dry-run` | Show what answering with `--name` would do, without writing it. Requires `--name`. |
| `--json` | Print `--list`/`--dry-run` output as versioned JSON instead of text. |

### `meethook speakers`

Reports who is enrolled and which stored voice recording is naming them — useful before
`forget`, since a person can hold several recordings that are otherwise indistinguishable.
Takes no options; reads every transcribed session and writes nothing.

### `meethook forget <NAME> [--reference N] [--yes]`

Removes a stored recording of somebody, or removes them entirely. `<NAME>` is the name exactly
as `meethook speakers` prints it.

| Flag | Meaning |
| --- | --- |
| `--reference <N>` | Remove only this one recording (the number `meethook speakers` gives it), instead of every recording of that person. |
| `--yes` | Perform the removal. Without it, the consequences are printed — voices that stop being named, ones that start reading somebody else, ones that gain a name — and nothing is written. |

A removed reference can't be rebuilt: the audio it was built from isn't consulted and may be
long gone. Every affected session's transcript is brought in line in the same run.

### `meethook meeting <SESSION_ID> [--event N | --clear]`

Corrects, or clears, the calendar meeting a session was labelled with — the automatic match is
a guess over start/end time, and sometimes it's wrong. With neither flag, prints the label the
session currently carries and the candidate meetings around it, numbered, and writes nothing.

| Flag | Meaning |
| --- | --- |
| `--event <N>` | Attach the Nth meeting from that numbered list. |
| `--clear` | Record that the session wasn't recorded during any meeting. Works without calendar access, unlike `--event`. |

### Global options

| Flag / env var | Default | Meaning |
| --- | --- | --- |
| `--root <PATH>` / `MEETHOOK_ROOT` | `~/meethook` | The meethook data directory (`sessions/`, `models/`, `speakers.json`) |
| `--template <PATH>` / `MEETHOOK_TEMPLATE` | built-in | Jinja template every `transcript.md` is rendered through |

### Data directory

Everything meethook writes lives under one root (`~/meethook` by default, override with
`--root` or `MEETHOOK_ROOT`):

- `sessions/<id>/` — one directory per recorded session: raw tracks, `session.json`,
  `transcript.md`/`.json`, `meeting.opus`
- `models/` — downloaded model weights (Whisper, diarization, speaker embedding), fetched and
  hash-verified on first use
- `speakers.json` — enrolled voice references, shared across every session

Nothing here is ever uploaded anywhere; recording, transcription, and enrollment all run
entirely on-device.

## Developing

```sh
nix develop                                                  # pinned toolchain + native deps
cargo build --workspace
cargo nextest run --all-features --workspace                 # full test suite
cargo clippy --all-targets --all-features --workspace -- -D warnings
rustfmt --edition 2024 <files>                                # what pre-commit runs
cargo fmt --all --check                                       # what pre-push runs
```

`nix develop` (or `direnv`, via `.envrc`) installs [lefthook](https://github.com/evilmartians/lefthook)
hooks automatically, so `fmt`/`clippy`/`test` run on commit/push the same way CI does —
`lefthook.yml` is the source of truth for the exact gates.

`crates/meethook-record` (macOS capture) roots its own Cargo workspace and is excluded from the
root's, because it can't compile off macOS — see the comment at the top of the root `Cargo.toml`.
On Darwin, gate it separately:

```sh
cd crates/meethook-record && cargo fmt --all --check && cargo clippy --all-targets --workspace -- -D warnings && cargo nextest run --workspace
```

See [AGENTS.md](./AGENTS.md) for the full architecture (how the five crates divide
responsibility, and the invariants worth knowing before changing any of them), and
[`backlog/decisions/`](./backlog/decisions/) for the design record behind specific choices (ASR,
diarization, speaker matching, echo cancellation, calendar fit, transcript rendering, and more).

## License

[MIT](./LICENSE)
