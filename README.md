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
the system/speaker audio as two independent tracks. Two ways out, because "end this call" and
"stop the recorder" are different questions: in the full-screen UI `s` ends the session that is
recording, finalizes it, and goes back to watching for the next call, while Ctrl-C or Ctrl-D
stops and exits. Run with `--plain`, which prints one line per event and reads no keys, Ctrl-C is
the only stop.

It starts a session whenever *another* app opens the microphone, so an app that does that
without it being a meeting — a dictation tool, say — can be named as one that never counts,
in `<data dir>/exclusions.json`:

```json
{
  "schema_version": 1,
  "bundle_ids": ["com.example.voiceink"],
  "executables": ["/Applications/VoiceInk.app/Contents/MacOS/VoiceInk"]
}
```

Entries match exactly — no wildcards or prefixes, so a near-miss spelling excludes nothing.
A bundle id names a bundled app; an executable entry is the real binary inside
`.app/Contents/MacOS/` (not the bundle directory), which is what plain binaries that report
no bundle id are matched by. Find an app's bundle id with `mdls -name kMDItemCFBundleIdentifier
/path/to/App.app`. The file is read once when `meethook record` starts, so restart `record`
after editing it; with no file, or empty lists, nothing is excluded.

#### Troubleshooting: the recording will not stop

The orange microphone dot in the menu bar is a weak instrument. It lights when any input device
is initialized — a virtual device included, audible sound not required — and it has been seen to
stay lit some twenty seconds after the app that held the mic let go, Apple's own Voice Memos
among them. Clicking Control Center names the app macOS blames, which makes the dot a good first
look and a poor measurement. To find out whether meethook is recording, ask the lock:

```sh
ROOT="${MEETHOOK_ROOT:-$HOME/meethook}"
lsof "$ROOT/record.lock"
```

That prints COMMAND and PID for whoever has the file open, and `record` keeps it open for exactly
as long as it holds the lock; with nobody recording it prints nothing at all. The `ROOT=` line is
not decoration: without it an unset `MEETHOOK_ROOT` leaves the path as a literal `/sessions/*`,
and an empty data directory then looks like a healthy one. On a machine where `meethook record`
has never run there is no `record.lock` yet, and `lsof` reports that rather than staying quiet.
This beats `ps | grep meethook`, which answers "has meethook ever run here?" instead — tried
here, it matched the shell doing the matching and a ten-day-old leftover `enroll`. With a pid in
hand, say what it is:

```sh
ps -p <pid> -o pid,ppid,lstart,etime,args
```

A PPID of 1 means no terminal owns that process any more. Then check that it is still capturing
rather than merely alive:

```sh
ls -l "$ROOT"/sessions/<id>/mic.wav
sleep 5
ls -l "$ROOT"/sessions/<id>/mic.wav
```

A size that grew is the proof. `du -sh "$ROOT"/sessions/*` sizes the damage instead — each track
runs close to 690 MB an hour, so both together are close to 1.4 GB — and says nothing about
liveness. `lsof "$ROOT"/sessions/<id>/mic.wav` asks the same question in one shot: it names the
writer, and prints nothing once no process holds the file. Read sizes with `ls -l` or `wc -c <`;
inside the dev shell GNU coreutils `stat` shadows BSD `stat`, where `-f` means "file system".

With the full-screen UI on that terminal, `s` ends just that session: the audio is finalized and
the recorder stays up, watching for the next call. To stop the whole run instead, use Ctrl-C in its
terminal, or `kill -INT <pid>` from another shell. A plain `kill` (SIGTERM) and a closed terminal
(SIGHUP) reach the same quit path and finalize the session too. What does not finalize is
`kill -9`, an abort, or a crash. Deleting `record.lock` accomplishes nothing: the lock is held by
the operating system, not by the file.

What you get back depends on how it ended. An interrupted run writes `session.json` and
transcribes normally. A panicked or unwound run leaves an orphan whose WAVs were finalized on the
way out anyway, complete to the last sample. A `kill -9` leaves an orphan that is valid up to the
last five-second checkpoint — the header stops there, so `afinfo` reports a duration up to five
seconds shorter than what was recorded, and those last seconds sit on disk unread. Either orphan
is skipped, never repaired:

```text
20260818-143027  skipped: no session.json: no transcript is possible; the speaker track holds 4.3 s past the end its header declares
```

`transcribe` prints that and exits 0; `enroll` passes the session over in the same words; `meethook
sessions` lists every such directory at once, without running either. The
audio still plays on the Mac that recorded it, though a strict parser elsewhere may refuse the
file until told to ignore the header length.

To keep it from happening again, name whatever keeps the trigger true:

```sh
MEETHOOK_ACTIVITY_DEBUG=1 meethook record --plain 2>activity.log
```

Run this after stopping the runaway, not instead of it: `record` takes the lock before it checks
permissions or prints anything, so while a runaway holds it this command refuses and not one
`[activity]` line appears. It is `record`, so it needs the same permissions and starts capturing
the moment something opens the mic. Its `[activity]` lines go to stderr — one summary per
recomputation, then one per holder with its pid, bundle id, `exe=`, `devices=[...]` and
`on-default=`. Browsers capture through helpers (`com.google.Chrome.helper`), any WebKit
embedder reports `com.apple.WebKit.GPU`, and a plain binary reports `(no bundle id)` and is
matched by its `exe=` path. `com.apple.CoreSpeech` beside `devices=[]` holds no device at all and
is a bystander; `on-default=no` is not an acquittal, because aggregate and virtual devices are
objects of their own that can contain the built-in microphone. Names worth excluding go in
`exclusions.json` above, followed by a `record` restart. Two ways to see the same list without
capturing anything: the probe in the source tree, which opens no device and needs no permission
but does need the checkout,

```sh
cd crates/meethook-record && MEETHOOK_ACTIVITY_DEBUG=1 cargo run --example mic-activity -- 60
```

and `lsaudio`, a third-party tool that lists the processes holding audio devices and can kill
them by pid.

Why it happens comes down to what the trigger counts: a session stays open while *another*
process has registered input IO, not while anyone is audible. An always-on dictation tool, an
assistant daemon, or a virtual-device companion app can therefore hold a session open
indefinitely, and nothing in the recording itself will end it. Ending one is something you do:
`s` in the full-screen UI closes that session and leaves the watcher watching, and a signal to the
run finalizes and quits.

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

A process that cannot reach the GPU stops rather than quietly degrading: `transcribe` reports
`no usable Metal device` and exits nonzero. Set `MEETHOOK_CPU` to any non-empty value (`0`
counts, an empty value does not) to choose the CPU deliberately, which `transcribe` confirms
once at startup with a `Note:` line on stderr. The switch covers speech recognition only;
diarization keeps whatever CoreML decided, and off macOS there is nothing to opt out of.

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

### `meethook sessions`

Reports what every session directory under the data directory became — the standing answer to
"which of my recordings survived?", which you'd otherwise have to ask by running `transcribe` or
`enroll` over everything. Each session is listed with the state it's actually in: `transcribed`,
`valid` (recorded cleanly, not yet transcribed), or `orphaned`. An orphan says which of its two
tracks reached disk, how far short of its own declaration either one stops when the header shows
it — those seconds are on disk but no player will find them — and why no transcript can be built
from it: `session.json`, the single clock both tracks share, was never written. An orphan is an
expected shape, not a failure, and nothing here offers to repair one.

```text
4 session(s) in /Users/you/meethook/sessions: 1 transcribed, 1 valid, 2 orphaned

20260809-052500  orphaned
    no session.json: no transcript is possible.
    That file held the single clock both tracks share, so neither can be placed on a common timeline however much of either one plays.
    The mic track declares 0.2 s more audio than the file holds, and that part is not on disk.
    The speaker track holds 0.1 s past the end its header declares; players stop at the declaration, so that part does not play.
    Nothing about this needs fixing: the audio that reached disk is kept as recorded.
20260809-052600  orphaned
    …
20260809-052700  transcribed
20260809-052800  valid
```

While a `record` holds the root, the report says so once at the top and declines to say that any
unfinished directory was left behind — one in that shape may be the call happening now. Takes
no options, like `speakers`; reads only, writes nothing, and exits 0 whatever it finds, including
an empty or missing `sessions/`. It works on Linux too, where `record` does not exist: that
asymmetry is exactly why this is a command rather than something printed when recording starts.

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
| `MEETHOOK_ACTIVITY_DEBUG` | unset | Print which processes hold the microphone, and why `record` started or stayed running, to stderr. Setting it at all counts, an empty value included |
| `MEETHOOK_CPU` | unset | Run speech recognition on the CPU rather than Metal, many times slower for the same transcript. Any non-empty value counts, `0` included |
| `MEETHOOK_CALENDAR_DEBUG` | unset | Print each calendar lookup to stderr: the access status, the candidates found, and which one matched, counting attendees without naming them. Honoured by `record` and `meeting`, on macOS only. Setting it at all counts, an empty value included |

### Data directory

Everything meethook writes lives under one root (`~/meethook` by default, override with
`--root` or `MEETHOOK_ROOT`):

- `sessions/<id>/` — one directory per recorded session: raw tracks, `session.json`,
  `transcript.md`/`.json`, `meeting.opus`
- `models/` — downloaded model weights (Whisper, diarization, speaker embedding), fetched and
  hash-verified on first use
- `speakers.json` — enrolled voice references, shared across every session
- `exclusions.json` — apps excluded from the mic-activity trigger (`record` only); user-
  authored, absent by default
- `record.lock` — held by a live `meethook record` so a second one refuses to start; written by
  `record`, never deleted, and its presence alone does not mean something is recording

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
