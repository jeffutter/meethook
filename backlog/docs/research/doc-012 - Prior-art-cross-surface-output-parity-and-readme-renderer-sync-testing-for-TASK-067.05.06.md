---
id: doc-012
title: 'Prior art: cross-surface output parity through a built binary, and pinning README samples to the renderer, for TASK-067.05.06'
type: research
created_date: '2026-09-22 02:40'
updated_date: '2026-09-22 02:40'
---

## Purpose

Gathered for planning `task-067.05.06 - Automate the two close-out checks that are still hand-run:
cross-surface parity through the built binary, and README's samples against the renderers`. Covers
(a) what other projects do about the same two problems, (b) facts measured on this repo's own built
binary that decide how the tests have to be shaped, and (c) which existing seams the plan can lean on.
Not a plan.

Web-search provider availability on this run was partial (only one provider answered), so the prior
art below is thin on blogs but complete on the crate documentation that matters.

## Prior art, outside this repo

### Testing a CLI through its built binary

- Cargo's own convention is the one already used three times here: `env!("CARGO_BIN_EXE_<name>")`
  gives an integration test the path to the compiled binary, documented in *Command Line Applications
  in Rust*, [Testing](https://rust-cli.github.io/book/tutorial/testing.html). Nothing new needed.
- [`assert_cmd`](https://docs.rs/assert_cmd) (`Command::cargo_bin`, `.args()`,
  `output.get_output().status.code()`) and `predicates` are the usual conveniences for the same thing;
  [`trycmd`](https://docs.rs/trycmd) goes further and runs whole stdout/stderr/exit comparisons from
  case files or `README.md` blocks. Both are **conveniences, not capabilities** - what they buy is
  less boilerplate. This repo already spawns `std::process::Command` directly in
  `crates/meethook/tests/{sessions_report,record_single_instance,headless_enroll}.rs` with no such
  dependency, and both AC#1 ("assert the ... sentences agree across all three surfaces") and AC#3
  ("if it starts fetching a model, that is a regression ... not a workaround") need *computed*
  assertions over several runs rather than literal expected-output files. Adding a dev-dependency to
  gain nothing would fragment the style of the existing bin-driven tests.
- The differential pattern the ticket asks for in AC#4 is exactly what
  `record_single_instance.rs::a_live_recording_does_not_disturb_any_other_command` already does: run
  each command twice against otherwise-identical roots, scrub the root path out, and compare
  `(code, stdout, stderr)` as one tuple. That test is the precedent to extend, not to reinvent.

### Keeping documentation byte-pinned to code

Three established shapes, in increasing order of machinery:

1. **A test that reconstructs the sample from the code and diffs it** (no markdown parsing):
   build the same fixture the report renders from, call the real renderer or the real binary, and
   compare against the README lines obtained by `include_str!` + line slicing. Cheapest, zero deps,
   and the anchor problem (which README lines?) is solved by asserting a stable header line lands
   where the slice says it should. Same idea as golden-master / approval testing
   ([characterization testing overview](https://understandlegacycode.com/blog/characterization-tests-or-approval-tests/)):
   the oracle is the produced artifact, compared whole, never a hand-typed transcription.
2. **Markdown block extraction** - mdBook's [`test`](https://rust-lang.github.io/mdBook/cli/test.html)
   flow, [`doubter`](https://users.rust-lang.org/t/announcement-doubter-a-crate-for-testing-code-blocks-in-markdown/21185),
   `pytest-codeblocks` for Python, `markdown-code-runner-rs` (rewrites outputs in place). All of these
   exist to *execute* code blocks; the README's two blocks here are `text` samples plus `sh` usage
   lines, so an executor is the wrong tool and a parser would be a new dependency for ~2 blocks.
3. **Generated READMEs** - `cargo readme` / [`cargo rdme --check`](https://github.com/orium/cargo-rdme)
   make the README a build artifact of rustdoc. Rejected by this repo's shape: README here is
   hand-written prose with two rendered samples, not rustdoc-derived. Note also that `cargo rdme
   --check`'s `--check` mode (verify drift, exit nonzero, don't rewrite) is the right *posture* for
   AC#5: the test must fail when the README has drifted, not silently repair the README (which would
   let a wording change pass by copying itself).

### Deterministic path scrubbing in snapshots

`insta` documents [redactions](https://insta.rs/docs/snapshot-types/) for replacing volatile parts
(a tempdir path) with a placeholder inside an inline snapshot. This repo has no `insta`; its existing
equivalent is hand-rolled and already correct - `record_single_instance.rs::outcome()` replaces
`root.display()` with `<root>` before comparing, and `sessions_report.rs` interpolates
`paths.sessions_dir().display()` into the expected string. AC#5's "modulo the path placeholder"
should reuse that exact substitution rather than invent a second one.

## Measured on this build (not documentation reading)

Built with `cargo build -p meethook` (dev shell), then drove `target/debug/meethook` over synthetic
roots. Everything below is observed output.

### 1. Which stream each surface prints to (this decides how the parity assertion reads)

| Surface | unfinished-directory sentence | other output | exit |
| --- | --- | --- | --- |
| `sessions` | stdout | census header + per-directory detail block | 0 |
| `transcribe <orphan-id>` | stdout (`<id>  skipped: <brief>`) | also prints `Nothing to transcribe.` on stdout | 0 |
| `transcribe` (whole root) | stdout | reaches models - see #4 | 1 once a valid session fails |
| `enroll --list` | **stderr** | stdout carries only the `--list` document | 0 if nothing failed |

That asymmetry is deliberate and already documented (`headless.rs`: "Run commentary goes to stderr
(the narrator is a `Lines` pointed at it); stdout carries only the document"), and it even contradicts
the adjacent comment in `commands.rs:645-652` claiming narration goes to stdout for every face. So a
parity test must collect `(stdout, stderr)` per command and search both, or it will pass by reading
nothing. Do not "fix" the stream - AC#6 forbids production changes, and `enroll --list --json | jq`
purity is the reason for it.

### 2. Exit 0 everywhere is reachable, but only with named ids

With the four-shape root (one transcribed, one valid, two orphans):

- `transcribe <orphan-id>` → exit 0. Partitioning happens before any model is touched, so the lazy
  factory is never called.
- `transcribe` with no ids → tries the `valid` directory too → opened all four models (measured:
  fetched silero 1 MB, segmentation 6 MB, embedding 27 MB, whisper 1.6 GB) → then failed on the
  fixture's placeholder `session.json` → exit 1. **This is the trap AC#3 names.** Batch form is not
  usable over a root containing a `valid` directory.
- `enroll --list` → exit 0 when every named directory is passed over; exit 1 if a named `transcribed`
  directory lacks a readable `speaker_clusters.json` (`failed: ... -- re-transcribe this session with
  --force`).

So: `transcribe` names orphan ids only; `enroll --list` names the same orphan ids (plus optionally a
properly-built transcribed directory - `headless_enroll.rs::fixture` shows how to write clusters,
names and a transcript programmatically if you want that row exercised). `sessions` takes the whole
root, since its entire job is the census.

### 3. The README brief line reproduces byte-for-byte from the renderer

Fixture: an orphan whose mic track agrees with its own header (so it contributes no clause) and whose
speaker track holds 4.3 s past what its header declares. Output:

```text
20260818-143027  skipped: no session.json: no transcript is possible; the speaker track holds 4.3 s past the end its header declares
```

`diff` against `README.md:168`: identical. Two consequences:

- The brief skip line IS reproducible at 4.3 s with the same truncation/append arithmetic the
  existing fixtures already use (64 000 bytes of `data` per second → 4.3 s = 275 200 appended bytes).
  No need to weaken AC#5 to substring comparison.
- A naive reuse of the `mixed_root` orphan does **not** reproduce README's line: there the mic is
  short of its declaration, so the real line reads
  `...; the mic track declares 0.2 s more than it holds` (measured), and appending mic data afterwards
  yields yet another sentence (`both tracks are on disk whole as recorded`, or a joined
  `<mic>, and <speaker>` pair). Whichever fixture is reused, the line the test compares must be the
  line that run actually produces.

### 4. Brief-vs-detail agreement is currently provable only by eye

`interrupted.rs` renders the two lengths from one `match` (`track_says` returns both halves - the
comment calls that "the point"), but no test asserts they describe the same evidence for the same
directory. On the mixed fixture the same directory printed:

- brief (transcribe/enroll): `the mic track declares 0.2 s more than it holds`
- detail (sessions): `The mic track declares 0.2 s more audio than the file holds, and that part is not on disk.`
- detail, speaker row: `The speaker track is on disk whole as its own header records, and plays through.`
  while the brief said **nothing at all** about the speaker track.

The last pair is consistent with the module's documented rule ("A track that agrees with its own
header contributes no clause of its own" in `interrupted_brief`) - `interrupted_detail` pushes a line
per track unconditionally. A parity test therefore cannot assert "same sentence on all three
surfaces"; it must assert what is genuinely shared: the `no session.json: no transcript is possible`
stem (brief adds `; <body>`, detail adds a `.` and follows with more lines) plus, per track, whichever
clause that surface's form defines. If the plan wants a stronger claim, state it as a derived subset
of clauses from one `Unfinished`, not as string equality across forms.

### 5. AC#4 needs a parent-process holder, and probe cost says why

`RecordLock` uses **OFD** locks (`F_OFD_SETLK`, `record_lock.rs:23`), which are owned by the open file
description, so a parent that acquires the lock in-process *cannot* hand it to a child through
inheritance - and even POSIX `flock(2)` locks drop on `execve`. Meanwhile the read-only probe
(`F_OFD_GETLK` over an `O_RDONLY` open) costs up to its timeout **per unfinished directory** while a
recorder is live. Consequences for AC#4:

- Add `&["sessions"]` to `record_single_instance.rs`'s `COMMANDS` and hold the lock in the test
  process, as that test already does for its five commands. There is no way to give a spawned child
  the lock, and no need for one.
- With the four forensic shapes under a held lock, `sessions` makes up to four timed probes, and
  `transcribe <orphan-id>` / `enroll --list` each probe again per named orphan via `unfinished_now`.
  Worth measuring rather than assuming: that test's own guard panics at `REFUSAL_TIMEOUT = 90 s` per
  command, and a slow filesystem could turn a green suite into a timeout flake. Alternative that keeps
  both claims in one place: spawn a tiny helper process that holds the lock (a long-sleeping child
  holding it, or reuse `cargo run -p meethook-record --example mic-hold`-style patterns) - strictly
  more moving parts, so prefer the in-process parent unless measurement says otherwise.
- With a recorder live, `sessions` prints the hedge and suppresses every unfinished block
  (`RootNow::detail_for` returns empty), so AC#1's parity run must hold **no** lock; AC#4's run is the
  opposite. Keep them separate - `sessions_report.rs::a_recorder_holding_the_root_elsewhere_...`
  already pins the hedged half.

## Existing seams the plan should build on

- One fixture, three surfaces: `crates/meethook/tests/sessions_report.rs::mixed_root` already builds
  the AC#2 shapes (an orphan with mic-short + speaker-past-declaration, an orphan with nothing, one
  transcribed, one valid) with honest WAV forensics, and `commands.rs:863-891` holds the near-identical
  unit-side copy. Extracting one shared builder (probably into `tests/common/mod.rs`, which already
  exists for cross-target helpers and is `#![allow(dead_code)]` by design) is the natural third home;
  a third hand-copied fixture is the failure mode to avoid.
- Model-fetch guard for AC#3 without network assumptions: `Paths::new(root)` puts weights at
  `<root>/models`, and `<root>` is always a `tempfile::tempdir()`, so asserting "no model was fetched"
  is just `assert!(files_under(root)["models"].is_empty())` (plus scanning stderr for `Fetching `,
  which is the literal `fetch`/`DownloadProgress` emits). Prefer that over an env-var kill switch: AC#3
  wants a regression in lazy fetching to *fail loudly*, and a pre-seeded fake weight file in
  `<root>/models` would instead mask it - hash verification would reject a stub anyway. (`HF_TOKEN` is
  the only env knob in `meethook-models`; there is no offline switch.)
- Where the parity test belongs: `crates/meethook/tests/` beside `sessions_report.rs`, using the piped
  stdio spawn idiom (`stdin(Stdio::null())`) rather than `common::Driver`, which exists only for pty
  frames. CI matrix is `macos-26` + `ubuntu-latest` (`.github/workflows/ci.yml:16`), and all four
  commands are platform-independent, so no `#[cfg]` gates and no `#[ignore]` (AC#6).
- Markdown-free README comparison: `include_str!("../../../README.md")` is shortest (path-sensitivity
  is fine because `cargo nextest run -p meethook` resolves the manifest dir), and locating the block by
  searching for the census-header line *inside the include* (then taking N following lines) turns
  "README moved" into a clear failure rather than an off-by-N diff. Only 2 of README's fenced blocks
  are `text`, so a full parser buys nothing.

## Risks / decisions this leaves open for the plan

1. Whether to assert strict equality of the shared stem only, or to model the brief/detail
   relationship explicitly (#4 above). The latter proves more and costs a small amount of derivation
   logic in the test; the former stays honest about what `interrupted.rs` actually promises.
2. Whether the AC#1 root's `transcribed` row is a placeholder (fine for `sessions`, but then
   `enroll --list` on that id exits 1) or a real cluster/transcript fixture (copy from
   `headless_enroll.rs`, ~100 lines). Naming only orphan ids for the two batch commands avoids the
   question entirely; leaving it open lets the plan choose per-surface ids.
3. Probe-cost measurement under a held lock (#5) is cheap to run before deciding where AC#4 lives.
4. If the README check parses blocks for the skip line too, decide whether to anchor by line number
   today (fails loudly on move) or by content search (robust, slightly vaguer failure). Existing
   ticket text ("compared against what the code renders, not against a transcription of it") is
   satisfied either way.
