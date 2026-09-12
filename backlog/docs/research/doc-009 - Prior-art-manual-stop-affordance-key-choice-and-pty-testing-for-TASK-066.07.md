# TASK-066.07 research — manual stop that keeps the watcher running

Gathered for planning `task-066.07 - Add-a-record-frame-key-that-ends-the-current-session-and-keeps-watching`.
Prior art plus the two facts that constrain the design (raw-mode key delivery, and what a pty
test of `record` can actually reach). Not a plan.

## Peers that separate "end this recording" from "quit the app"

- **Ariso oats** ([PR #157/#170](https://github.com/ariso-ai/oats/pull/170)) — the origin of this
  ticket's framing. Its mic monitor *only* fires when the triggering PIDs release the mic for 8s,
  so in a back-to-back transition where one app holds the mic continuously, call A's recording
  bleeds into call B. Its answer is a calendar-driven meeting-end **stop prompt** mirroring an
  existing silence-stop prompt. Related [issue #174](https://github.com/ariso-ai/oats/issues/174)
  asks for the *start*-side twin: continue the existing thread or start a new one. Both are
  stop-with-the-app-still-alive; quit is never the affordance offered.
- **edwinv/minutes**
  ([8ab68d2 `stop_when_call_ends`](https://github.com/edwinv/minutes/commit/8ab68d22856f8c14bba4aa15f514a88c66fc3973),
  [#132 settings UI](https://github.com/edwinv/minutes/commit/e6ce063f6f48f945e5a26ecbf51375e2a7ee6410)) —
  a countdown card with explicit **Stop now / Keep recording**, opt-in, scoped to detector-started
  sessions only ("a manual `minutes record` is never auto-stopped"). Then
  [v0.13.3 "Auto-stop actually auto-stops now"](https://github.com/silverstein/minutes/releases/tag/v0.13.3):
  the countdown logged `call_ended` and a state race killed it mid-tick — the fix was that *once
  armed, the countdown thread alone owns the lifecycle*, no atomics in the detector poll loop.
  That is the same ownership question as meethook's `await_end` grace (record.rs:1239): a manual
  stop must not share lifecycle ownership with a device-edge veto. Supports the ticket's
  own-`Event`-variant / own-`Recording`-variant decision rather than a reuse of `Stopped`.
- **Wispr Flow Notetaker**
  ([recording](https://docs.wisprflow.ai/articles/9238501024-recording-a-meeting-with-notetaker-beta),
  [settings](https://docs.wisprflow.ai/articles/9319084321-notetaker-settings-explained-beta),
  [troubleshooting](https://docs.wisprflow.ai/articles/3089221553-troubleshooting-notetaker-recording-and-audio-beta)) —
  pill's stop button "only stops"; "**Stop always works**: from the pill or the meeting window,
  even if system audio dropped and even during a connection attempt. Flow always shows a toast
  first — recordings are never stopped silently." Auto-stop is suppressed while *other*
  participants are audible, but your own continuing voice does not hold it. Note the asymmetry:
  the manual stop is unconditional, the automatic one is the vetoable one.
- Nobody in this space requires a confirmation **step** to stop; all three make stop immediate and
  show the consequence afterwards.

## Confirmation vs. reflex-safety (the ticket's open question)

- NN/g [confirmation dialogs](https://www.nngroup.com/articles/confirmation-dialog/), A List Apart
  [Never Use a Warning When You Mean Undo](https://alistapart.com/article/neveruseawarning/),
  [undo is better than confirm](https://thunderguy.com/semicolon/2009-11-11/undo-is-better-than-confirm/)
  agree: a confirm slows the users who meant it and does not protect the ones who didn't — the
  warning becomes the thing you click to make it go away. Undo/delay beats confirm when undo exists.
- The meethook analogue of "undo" is already true structurally: finalize-then-watch leaves the WAVs
  finalized on disk and the very next mic edge opens a fresh session, so an accidental press costs
  the tail of one call and nothing else. That weakens decision-012's preview-before-you-can-press
  analogy — enroll's preview exists because a *write to the speaker database* is not undoable; a
  session finalize is largely reversible from the user's seat.
- Where a cheap reflex guard *is* used in Rust TUIs it is a two-stage gesture, not a modal: Codex
  [issue #8936 double-press Ctrl+C/Ctrl+D to quit](https://github.com/openai/codex/issues/8936),
  entanglement [ADR-0087 two-stage
  Ctrl+C](https://github.com/xmiksay/entanglement/blob/master/docs/adr/0087-two-stage-ctrl-c.md)
  (first press clears input, second quits; intercepted once at the top of `handle_event`),
  recursive-agent's double-press-within-a-window Ctrl+C arm. A second-press-within-window would be
  the pattern to copy if any guard is wanted, since there is no modal chrome in either frame today.

## Key-choice precedent

- `q` = quit vs. `s`/`start`/`stop` = transport is the established split: DriveWorks
  [recorder-tui](https://developer.nvidia.com/docs/drive/driveworks-nvgssi/6.0.5/public/driveworks-nvsdk/dwx_recorder_textui_tool.html)
  (`s` toggle recording, `start`, `stop`, `q` quit),
  [cbuffer](https://github.com/kuremu/cbuffer) (`q` or ctrl-c stop, any other key toggles
  record/buffer), [asciinema](https://docs.asciinema.org/manual/cli/quick-start/) (ctrl-d/exit ends
  the recording because recording *is* the session there),
  [`arecord --interactive`](https://commandmasters.com/commands/arecord-linux/) (space/enter drive
  transport). Record-specific tools put transport on a printable letter and keep the control byte
  for the exit; meethook's record frame has printables free (unlike enroll's, where they go to the
  filter), and `s` is in the ticket's free set.

## Raw-mode / signal mechanics worth citing

- crossterm [`terminal` module docs](https://docs.rs/crossterm/latest/crossterm/terminal/index.html):
  in raw mode input is not forwarded to the screen and line/signals handling is off — which is why
  the frame's comment at record_screen.rs:483-491 says "raw mode means no SIGINT arrives, so the
  keys are the interrupt".
- crossterm [issue #554](https://github.com/crossterm-rs/crossterm/issues/554): no built-in signal
  events, hence the separate `ctrlc` handler for plain mode — i.e. the frame-only key leaves the
  plain presenter with signals only, exactly as the ticket expects.
- kitty keyboard-protocol
  [discussion #7179](https://github.com/kovidgoyal/kitty/discussions/7179): under Kitty's protocol
  Ctrl-C is delivered as CSI-u instead of SIGINT, and
  crossterm [issue #931](https://github.com/crossterm-rs/crossterm/issues/931) covers remapped
  `stty intr`. Any assumption that a specific control *byte* means quit is environment-dependent;
  the frame already reads both cases (`'c' | 'C' | 'd' | 'D'`) for that reason.

## Testing prior art and the AC #4 feasibility flag

- Harnesses exist (`microsoft/tui-test`, `raibid-labs/ratatui-testlib`) but the repo already has a
  hand-rolled `openpty` Driver in `crates/meethook/tests/interrupt_group_commit.rs`
  (`open_pty`, `feed`, `wait_for_frame_or_exit`, `wait_exit`, `tail`) — reuse it rather than add a dep.
- **Flag:** a pty test that observes a real finalize needs a real mic edge. `Capture` is
  `pub(crate)` (record.rs:184) and `SessionCapture` wraps ScreenCaptureKit; the only env hooks in
  the whole tree are `MEETHOOK_ROOT`, `MEETHOOK_ACTIVITY_DEBUG`, `MEETHOOK_TIMING_DEBUG`,
  `MEETHOOK_CALENDAR_DEBUG` — there is no fake-capture seam at binary level. So AC #4 needs macOS +
  a granted TCC mic grant + something that opens the input device briefly:
  `crates/meethook-record/examples/mic-hold.rs` is exactly such a holder. Without the grant the
  child walks into a permission prompt and hangs, which is why `record_single_instance.rs` uses
  `spawn` + bounded `try_wait` instead of `output()` and gates that test `#[cfg(target_os =
  "macos")]`. Expect the pty test to need the same gate and possibly a real-Mac confirmation
  sub-ticket, as TASK-066.04.02 and .05.01 were.
- Good news on cost: `finish` writes headers + `session.json` only (meethook-record lib.rs:354 →
  `Note::Recorded`); transcribing is a separate command, so the pty assertion does not need the
  1.6 GB model weights that kept `record_single_instance.rs` differential.

## Gate note

Nothing here points at `crates/meethook-record` (key map, actions, state, hint rendering and the
loop all live in the `meethook` bin crate), so AC #6's record-crate gate set should stay a printed
skip — unless the mic-holder helper gets promoted into that crate for the test.
