---
id: doc-007
title: >-
  Prior art: write-error-swallowing writers and panic containment for TASK-067.08
type: other
created_date: '2026-09-10 12:52'
updated_date: '2026-09-10 13:03'
---

Research for TASK-067.08 (debug-gated stderr prints inside `capture.finish` still panic on a
hung-up terminal). Sources: Rust std/reference/nomicon/Unstable Book, Rust RFC 2945, Unix tty
man pages and kernel commits, cargo/ripgrep issue trackers, published writer adapters, the
CoreAudio listener contract in Apple docs, and the vendored `block2`/`dispatch2` sources this
workspace already builds against. Items marked *(verify)* are leads, not facts.

## 1. Why these prints are fatal at all, and why SIGPIPE does not help

- Rust's runtime sets `SIGPIPE` to `SIG_IGN` before `fn main()` (and restores `SIG_DFL` before a
  child `exec`), so a write to a vanished reader comes back as an error rather than killing the
  process. Documented in the Unstable Book's `-Zon-broken-pipe` table
  (<https://doc.rust-lang.org/unstable-book/compiler-flags/on-broken-pipe.html>); history and
  the "is this still right?" debate in rust-lang/rust#62569
  (<https://github.com/rust-lang/rust/issues/62569>). The stable knob for opting out is the
  unstable `#[unix_sigpipe = "sig_dfl"]` attribute
  (<https://dev-doc.rust-lang.org/beta/unstable-book/language-features/unix-sigpipe.html>).
- `print!`/`eprintln!` panic when the write fails — stated outright in the std docs
  ("Panics if writing to `io::stdout()` fails", <https://doc.rust-lang.org/std/macro.print.html>).
  That is the whole bug: SIGPIPE being ignored converts a lethal signal into a panic instead.
- A hung-up pty really is sticky: with the master gone, the slave side answers I/O with `EIO`
  repeatedly (Linux/OpenBSD discussion of the closed-slave case
  <https://unix.stackexchange.com/questions/478815/read2-blocking-behaviour-changes-when-pts-is-closed-resulting-in-read-return>;
  FreeBSD changed exactly this special case twice — 2008 "Fix pts(4) error codes when slave
  device is closed" <https://lists.freebsd.org/pipermail/cvs-all/2008-August/268458.html> and
  2024 "kern: pts: do not special case closed slave side"
  <https://lists.freebsd.org/archives/dev-commits-src-branches/2024-January/014906.html>;
  OpenBSD `pty(4)` <https://man.openbsd.org/OpenBSD-7.9/pty.4>). Which errno a macOS Terminal.app
  window close yields (`EIO` vs `ENXIO`, and whether it flips mid-teardown) is *(verify)* — it is
  TASK-067.07's hardware proof, and argues for swallowing every kind rather than branching.
- Branching on the kind is not available anyway: std's Unix errno→`ErrorKind` table has no arm
  for `EIO`, so it lands in `Uncategorized`
  (<https://doc.rust-lang.org/nightly/src/std/sys/io/error/unix.rs.html>; the variant itself is
  unstable and documented as "not recommended to match against"
  <https://doc.rust-lang.org/stable/std/io/enum.ErrorKind.html>). This corroborates the existing
  `Narration` comment in `crates/meethook/src/record.rs` ("retrying is pointless and the error
  kind is not something this type can usefully branch on"). Practical consequence for tests:
  `io::Error::from_raw_os_error(5)` gives an EIO-shaped failure with no `libc` dev-dependency,
  since `EIO == 5` on both macOS and Linux.
- A run launched by `launchd` (no controlling terminal, `StandardErrorPath` unset or pointing at a
  file) never hits this — the failure mode belongs to tty-launched runs
  (`launchd.plist(5)` <https://keith.github.io/xcode-man-pages/launchd.plist.5.html>). Apple's
  sanctioned channel for diagnostics from a non-tty process is the unified logging system
  (<https://developer.apple.com/documentation/os/logging>, TN2083 on daemons
  <https://developer.apple.com/library/archive/technotes/tn2083/_index.html>) — a possible later
  home for these debug blocks, not needed here.

## 2. Prior art for "keep printing, swallow the refusal"

- **cargo did the same fix, same reasoning.** PR #8236 "Ignore broken console output in some
  situations" (merged for 1.45) routes *all* output, stdout included, through one `Shell` seam so
  no call site decides policy, and keeps genuine status-message failures fatal to match `make`
  (<https://github.com/rust-lang/cargo/pull/8236>). The motivating report is cargo#5234 and the
  general std complaint is rust-lang/rust#46016
  (<https://github.com/rust-lang/rust/issues/46016>). Reviewer concern worth copying: *"it seems
  like it's going to be very difficult to remember to always use not-eprintln"* — resolved by
  landing a CI test that scans the codebase for `println!`/`eprintln!` macros. That is the
  anti-rot companion to "one seam, don't sprinkle `let _ = writeln!`".
- **ripgrep is the counter-case.** It swallowed print errors everywhere and then kept searching
  full trees with nobody reading (issue #200
  <https://github.com/BurntSushi/ripgrep/issues/200>, originally #22
  <https://github.com/BurntSushi/ripgrep/issues/22>): where early exit has value, the error must
  survive. Here it does not — `finish`'s job (finalize WAVs, write `session.json`) strictly
  outranks the diagnostic, and the loop already has its own quit path.
- **Published adapters, and why none fits:** `pipecheck::Writer` kills the process on `BrokenPipe`
  (<https://docs.rs/pipecheck/latest/pipecheck/>) — the opposite of what a recorder wants;
  `yaks`' `broken_pipe_guard` absorbs `BrokenPipe` only (<https://github.com/mattwynne/yaks/blob/main/src/adapters/broken_pipe_guard.rs>)
  — too narrow given `EIO`; `ufmt_utils::Ignore<W>` is the right *shape* (a `Write` adapter whose
  `Error = Infallible`, <https://docs.rs/ufmt-utils/latest/ufmt_utils/struct.Ignore.html>) but is a
  embedded `no_std` stack; `sashiko::logging::IgnoreBrokenPipe` wraps a `log` `MakeWriter`
  (<https://docs.rs/sashiko/latest/sashiko/logging/struct.IgnoreBrokenPipe.html>). Cargo's own
  merged PR is the strongest argument that a ~30-line local latch beats a dependency, and this
  repo already owns a better in-house model: `record::Narration` (forward until first refusal,
  then discard, latch on failed flush too).
- **There is an unsolved-libstd gap worth naming in the plan:** the internals thread
  "Defusing `println!` when stdout is closed"
  (<https://internals.rust-lang.org/t/defusing-println-when-stdout-is-closed/20325>) records that
  no std-level option means "ignore the signal *and* the write error", so callers must wrap — i.e.
  the local writer is the accepted workaround, not a smell.
- **Library crates shouldn't own the printing policy at all.** The `log` facade's stated rule is
  that libraries emit through the facade and the *consumer* picks the implementation
  (<https://docs.rs/log/latest/log/>). Read as a design pointer, the cleaner sibling of "a second
  per-crate writer latch" is for `finish` to hand back the timing report (or formatted lines) and
  let the bin crate print it through the one `Narration` that already guards the quit path —
  consistent with the repo invariant that the sink is `record`'s single printing point. Trade-off
  to weigh explicitly in the plan: that changes a cross-crate signature; the latch writer does not.

## 3. How others test an unwritable stderr without a pty

- Found precedent: integration tests that swap the process's real fd 2 for the write end of a pipe
  whose read end is already closed, via raw `pipe()`/`dup2` — "Logging to a stderr that cannot be
  written must not panic"
  (<https://docs.rs/crate/logwise_runtime/latest/source/tests/stderr_unwritable.rs>; same trick in
  <https://docs.rs/crate/louke/latest/source/tests/default_sink_dropped_events.rs>). Both put it in
  a **separate integration-test binary** precisely because replacing fd 2 is process-global.
- The hazard with doing that in unit tests is real and observed: process-global `dup2` redirection
  races concurrent writers and produces flaky suites
  (<https://github.com/randomparity/bzr/issues/192>; forum thread on `dup2`+tempfile behaving
  differently inside `#[test]` <https://users.rust-lang.org/t/stdout-redirection-with-dup2-tempfile-works-in-main-rs-but-fails-in-test-why/130096>).
  Under `nextest` (every test its own process) the fd-2 swap is safer than under plain
  `cargo test`, but writer injection remains the deterministic choice — matching the ticket's
  constraint and the existing `record::tests::ScriptedStream` pattern.

## 4. AC #2's premise ("the panic only takes the watcher down") is not supported

- Where the code actually runs: listener work is dispatched onto a private serial GCD queue
  (`crates/meethook-record/src/activity.rs`, queue created at `:286`, `Trigger::Install` log on the
  caller's thread at `:470`). So `MEETHOOK_ACTIVITY_DEBUG` logs land either on a libdispatch worker
  or on the main thread at install time — not on a Rust-owned thread anywhere.
- Those bindings do not stop panics: `block2` declares block trampolines `extern "C-unwind"`
  (vendored `block2-0.6.2`: `src/abi.rs:239`, `src/global.rs:193-203`, `src/stack.rs:90`) and
  `dispatch2-0.3.1` contains no `catch_unwind` on the async path — it *aborts* deliberately in its
  `once` because it will not unwind either (`dispatch2-0.3.1/src/once.rs:15`, `:43`). A panic
  therefore continues from the Rust closure into libdispatch frames.
- That is outside what Rust promises. RFC 2945 (`-unwind` ABIs exist precisely because unwinding
  across C frames is only "relies on compatibility" between mechanisms, and non-`-unwind` boundaries
  are where the compiler stops the panic)
  (<https://rust-lang.github.io/rfcs/2945-c-unwind-abi.html>); the Reference lists "unwinding past a
  stack frame that does not allow unwinding" under behavior considered undefined
  (<https://dev-doc.rust-lang.org/stable/reference/behavior-considered-undefined.html>); the
  Rustonomicon's FFI page gives the same instruction — never let a panic cross into foreign code,
  use `catch_unwind` in the trampoline
  (<https://doc.rust-lang.org/nomicon/ffi.html>). Worst realistic outcomes here are process abort or
  a wedged serial queue (no further notifications, watchers silently dead), not a clean thread exit;
  `*(verify)*` only by experiment, and the ticket's own no-device/no-pty constraints make that
  experiment hard to gate. Recommendation for AC #2: make the logging unable to fail (shared latch
  writer) and record that containment could not be relied on, rather than declaring the panic cost
  acceptable.
- Two supporting details: the watcher already treats mutex poisoning as non-fatal
  (`activity.rs:349` `unwrap_or_else(|e| e.into_inner())`, with the comment that poisoning must not
  permanently disable triggering — nominal behaviour in std docs
  <https://doc.rust-lang.org/stable/std/sync/struct.Mutex.html>, critique in the Rustonomicon
  <https://doc.rust-lang.org/nomicon/poisoning.html>), so a panic there leaves the machinery running
  and *silent*, which is exactly the "nothing says why" outcome the ticket complains about. And the
  CoreAudio listener contract asks listeners to be cheap — Apple's AudioUnit-events note warns
  notifications may come from real-time priority threads
  (<https://leopard-adc.pepas.com/technotes/tn2002/tn2104.html>; the `AudioObjectPropertyListenerProc`
  contract at <https://developer.apple.com/documentation/coreaudio/audioobjectpropertylistenerproc>)
  — while `log()` does a sweep of cross-process property reads plus formatting allocations per line.

## 5. Census gaps to fold into the plan

- `RunningSession::finish` also reaches a *second* gated print path the ticket does not list:
  `calendar::meeting_at(start_time)` prints through `calendar/mod.rs:570` when
  `MEETHOOK_CALENDAR_DEBUG` is set, and it runs inside `finish` before `session.json` is written —
  same lost-meeting class as `report_first_buffer_timing`, so AC #1's "nothing printed by finish can
  abort it" is not satisfied by fixing the timing block alone.
- Total ungated-by-the-bin-crate print surface in `crates/meethook-record` is 14 `eprintln!` sites
  (7 in `lib.rs`, 6 in `activity.rs`, 1 in `calendar/mod.rs`), across three env
  vars (`MEETHOOK_TIMING_DEBUG`, `MEETHOOK_ACTIVITY_DEBUG`, `MEETHOOK_CALENDAR_DEBUG`). One writer
  per crate covering all three is a smaller surface than per-site repair, and matches the ticket's
  "same decision, held once per crate" direction.
- Anti-rot option with provenance: cargo pairs its seam with a source-scan test banning `println!`
  /`eprintln!`. In this repo such a test belongs in `crates/meethook-record`'s own suite (its gate
  set is separate, and AC #3 requires those gates pass unchanged), and can be a plain
  `include_str!`/`WalkDir`-free token scan rather than cargo's `syn` dependency.
