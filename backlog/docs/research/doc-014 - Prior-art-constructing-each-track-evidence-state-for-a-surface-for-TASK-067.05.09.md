# doc-014 - Prior art: constructing every `TrackEvidence` state for a surface (TASK-067.05.09)

Research only, no plan. Everything under "Measured here" was run on this machine
(macOS, uid 501) against the current tree at `4db7a03`, using throwaway test targets
under `crates/meethook-session/tests/` that were deleted afterwards.

## Measured here: each remaining state and the cheapest filesystem means that produces it

| State | Construction that works | Observed |
| --- | --- | --- |
| `HeaderOnly` | `common::clip(&path, 0.0)` (already exported from the shared fixture) - hound writes its header declaring zero `data` bytes and nothing else | existing `crates/meethook/tests/zz_probe.rs` asserts it, green |
| `NotAWav` | `std::fs::write(&mic_wav, [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a])` - an honest PNG renamed `.wav`, i.e. exactly the support-request shape the ticket names | `NotAWav` |
| `Unreadable` | `set_permissions(mode 0o000)` on a real WAV | `Unreadable` |
| `Unreadable` (guard-free alternative) | `std::fs::create_dir(session.mic_wav())` - a *directory* named `mic.wav`; `File::open` succeeds on macOS and the subsequent `read` fails `EISDIR`, which `track` maps through its `read_to_end` arm | `Unreadable`, and it does **not** depend on permission bits, so no root guard is needed for it (`wav.rs:414-437`) |
| `Unknown` | Four variants all work; the cheapest is a valid 16-byte `fmt ` chunk followed by nothing (no `data` at all) | `Unknown` |
| `Unknown` ("walk ran off the read window", the wording in the variant's own doc) | Put a prelude chunk ahead of `fmt ` whose `ckSize` exceeds `HEADER_WINDOW` (64 KiB): `chunk(b"JUNK", 70_000, &[0u8; 1_000])` then `fmt ` then `data `, with the file ~100 KiB. The walk stops inside the prelude, so neither `fmt ` nor `data` is ever seen | `Unknown` |
| `Unknown` | A 14-byte file (`RIFF....WAVEfm`) - window ends between chunk headers | `Unknown` |
| `Unknown` | A parsed-but-refused `fmt ` (zero sample rate, or zero block align) over 1 600 real audio bytes | `Unknown` |
| `NoDeclaredLength` (TASK-067.05.08's ninth state, for reference) | hand-assembled `RIFF\xFF\xFF\xFF\xFF WAVE` + good `fmt ` + `data` declaring `u32::MAX` over 64 000 bytes | `NoDeclaredLength { bytes: 64000, millis: 1000 }` |

Consequence for AC#5: **no arm needs deleting.** All four target states plus `Unknown`
are producible for a session directory with plain writes plus one `set_permissions`, and
`Unreadable` has a permissions-free route too, so the suite keeps real coverage even where
the guard fires.

Two construction traps worth recording, both hit while probing:

- Forgetting a `WavWriter` via `mem::forget` leaves a **zero-byte** file, not a
  header-only one (`NotAWav`, not `HeaderOnly`). `wav.rs`'s own doc for `HeaderOnly`
  describes hound writing its header "the instant the writer is created"; through the
  public create path the observable result at `meethook_session::wav::track` is
  `NotAWav`. `clip(&p, 0.0)` is the working route (and `common::clip` is already public
  to test targets), and there is already a committed proof of that in `zz_probe.rs`.
- Making `Unknown` by pushing `data` past the read window does **not** work by declaring a
  giant `data` chunk: `Chunks::next` returns a chunk whose body runs past the window with
  the bytes in hand (`wav.rs:269-277`), so the measurement still happens and the answer is
  `ShortBy(TrackSpan { bytes: 3_999_899_900, millis: 62_498_435 })`. Only moving the
  dispatch point into a pre-*`fmt`* prelude chunk yields `Unknown`.
- `PermissionExt` (`std::os::unix::fs::PermissionsExt`) must be imported for
  `Permissions::from_mode`; `wav.rs`'s existing unit test has it in scope, a new test
  target will not.

## Prior art and practice found

- **Exhaustive-per-variant tests.** The recurring Rust answer to "every enum variant must
  reach an assertion" is to drive the cases from data rather than restate prose: `rstest`'s
  `#[case]` / `#[values]` table tests (docs.rs/rstest), or plain exhaustive `match` in a
  test so the compiler fails when a variant is added (users.rust-lang.org "How do you write
  exhaustive tests for enums"). `rstest` has no way to enumerate variants automatically
  (rstest#264), so a hand-listed case table over the nine `TrackEvidence` variants is the
  idiom; the repo already gets the compile-time half of that from the existing exhaustive
  `match` at `three_surfaces_one_directory.rs:258-283`, and adding fixture rows is what
  makes those arms run. Note the ticket explicitly wants the *existing* structural relation
  reused rather than new assertions, so a case table would be a second mechanism.
  https://docs.rs/rstest/latest/rstest/attr.rstest.html ·
  https://github.com/la10736/rstest/issues/264 ·
  https://users.rust-lang.org/t/how-do-you-write-exhaustive-tests-for-enums/94541
- **The `chmod 000` + root skip is standard practice, and its known weakness is that the
  guard silently drops coverage.** Real repos first add a `geteuid() == 0` early-return,
  then get complaints that the assertion "looks green because it never runs" in the
  container CI it was written for, and finally replace the second copy with a mocked
  failure instead of duplicating guards (observed as a commit series: skip-under-root →
  extend-the-existing-guard → prove-the-path-without-chmod). That sequence is external
  support for the ticket's instruction to reuse `wav.rs:1098-1103`'s guard rather than
  invent a second one, and for preferring a guard-free producer (the directory-named
  `mic.wav`) where one exists. `tempfile::TempDir::drop` also cannot remove a mode-0000
  file cleanly in some setups, so restoring the mode before returning is cheap insurance.
  https://github.com/dshakes/distil/commit/cca07759d905feb721e9d7c7d673552b23227314 ·
  https://github.com/dshakes/distil/commit/71f80bca8bffb70d988ba9f4a7aafc9edbf61091 ·
  https://github.com/dshakes/distil/pull/147
- **CLI parity testing.** Where teams snapshot whole CLI output (`insta-cmd`'s
  `assert_cmd_snapshot!` + `get_cargo_bin`, insta.rs/docs/cmd) the maintenance cost is
  that every word change churns a golden. This repo has deliberately gone the other way -
  structural relation asserted against live renderer output (`interrupted_brief` /
  `interrupted_detail` recomputed from the same files), plus byte-pinned README blocks kept
  in sync by `readme_quotes_the_renderers.rs`. Staying with the structural relation for the
  new shapes is consistent with that stance; nothing found argues for introducing snapshots
  here. Relatedly, insta's own guidance is to use inline snapshots where a human reviews
  the words - which is what README already is for these sentences.
  https://insta.rs/docs/cmd/
- **Test-fixture shape.** The long-standing recommendation for multi-shape fixtures is
  named factory methods per meaning rather than one builder with flags, precisely so a
  caller cannot compose a nonsense state (Naty Pryce "Test Data Builders"; reflectoring.io
  Object Mother + Builder). `common::mixed_root` is already that kind of named mother, and
  `readme_quotes_the_renderers.rs:155-183` documents the same reasoning for keeping its
  fifth shape local ("only this test wants a fifth shape, and folding it into `mixed_root`
  would churn the golden listing every other target pins") - direct in-repo precedent for
  the ticket's step-2 choice of a *second* root helper.
  http://www.natpryce.com/articles/000714.html
- **WAV forensics grounding for the exotic shapes.** The 1991 RIFF spec makes `ckSize` a
  u32 that excludes the header and the odd-size pad byte, and explicitly permits a reader
  to skip unknown chunks by `ckSize` + pad - which is why an arbitrary `LIST`/`JUNK`
  prelude (the `Unknown` route above) is a legitimate foreign artifact rather than a
  synthetic one, matching `HEADER_WINDOW`'s own doc comment. Modern parsers documented in
  this space validate child chunks against the parent's declared budget and surface "clean
  EOF before the parent budget is satisfied" as a distinct truncated-parent error rather
  than a guess (oxideav-riff's chunk reader) - the same posture `wav.rs` takes when it
  returns `Unknown` instead of a number.
  https://modland.com/pub/documents/format_documentation/RIFF%20WAVE%20%28.wav%29.txt ·
  https://docs.rs/crate/oxideav-riff/latest/source/src/chunk.rs
- **Root-user access on this platform:** none applicable - `id -u` is 501 here, so the
  `geteuid()==0` guard does not fire locally; it matters only for a root gate run. Worth
  stating in the notes rather than discovering later, since a guarded state that never runs
  is precisely the class of unfalsifiable branch AC#5 is about - the directory-named
  `mic.wav` route is what keeps `Unreadable` genuinely executed everywhere.

## Local facts a planner needs

- `crates/meethook/Cargo.toml` dev-deps currently hold only `libc` (plus workspace `tempfile`
  as a normal dep), and `hound` is *not* available to the CLI crate's test targets - so any
  hand-assembled header bytes in the new fixture helper must come from raw `Vec<u8>` writes
  (or from `common::clip` + truncation/append), not from `hound`. Adding `hound` as a
  dev-dependency is the alternative.
- Test targets that would see the new rows: `three_surfaces_one_directory.rs`,
  `sessions_report.rs` (greps the census line `4 session(s) in ...: 1 transcribed, 1 valid,
  2 orphaned` and pins id classifications), `record_single_instance.rs`,
  `headless_enroll.rs`, `interrupt_group_commit.rs`, `readme_quotes_the_renderers.rs`.
  Adding directories to `mixed_root` therefore has blast radius beyond the README parity
  test - three_surfaces itself hardcodes the census text.
- Orphans in a new root cost nothing at the model layer only if they stay orphans: the
  existing test passes ids explicitly to `transcribe`/`enroll --list` because a no-id batch
  run opens the 1.6 GB weights, and `run()` carries the no-download tripwire. Any extra
  orphan id must be named, and the census-length assertions updated in whatever target uses
  the enlarged root.
- `interrupted.rs` binds `TrackSpan` at line 56 but does not name it in `track_says`; the
  unused-import warning surfaced only once the planned refactor touched that function (seen
  while running a scratch build). Trivial, but a planner should expect it if code moves.
- Coverage tooling exists in the ecosystem for detecting never-executed arms
  (`cargo-tarpaulin`, and `lmmx/isotarp` for "which test uniquely covers which lines"), but
  neither is in this repo's gates, and tarpaulin's `--ignore-test` / dead-code-linking
  behaviour makes it a poor proxy for "did a user-facing sentence print". The
  arms-execute-because-a-fixture-produces-the-state framing in this ticket is stronger than
  a coverage number.
