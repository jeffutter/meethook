---
id: TASK-066.07.03
title: >-
  Confirm on a real Mac that s ends the live call and the recorder keeps
  watching
status: Done
assignee:
  - '@human'
created_date: '2026-09-12 01:55'
updated_date: '2026-09-18 17:37'
labels:
  - spike
  - record
  - planned
dependencies:
  - TASK-066.07.01
  - TASK-066.07.02
parent_task_id: TASK-066.07
priority: medium
type: spike
ordinal: 54100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-066.07.01 puts a hand-stop route into the record loop and proves the sequencing with a fake capture; TASK-066.07.02 drives the real frame over a pty. Neither can answer the two questions that matter to the person who presses the key: does the audio that was already captured survive complete, and does the recorder really keep working afterwards? Both need a machine whose terminal holds the Microphone and Screen & System Audio Recording grants, which CI has never had and an agent cannot grant - so this is hardware-gated for the same reason TASK-066.04.02 and TASK-066.05.01 are.

What is being confirmed, in one sentence: `s` ends *this* call and leaves the watcher running, while Ctrl-C keeps meaning "leave the run".

Recipe (repo root, inside `nix develop`, in a terminal that holds both TCC grants):
1. `cargo build -p meethook --bin meethook`, and build the stand-in that plays the other party: `cd crates/meethook-record && cargo build --example mic-hold`.
2. Shell A - pin the trigger true so the session will not end on its own, which is the whole situation the key exists for: `./crates/meethook-record/target/debug/examples/mic-hold 240`.
3. Shell B - start the recorder against a scratch root with the frame up: `MEETHOOK_ROOT=/tmp/t06607 ./target/debug/meethook record`. Wait for the header to read `recording`; note the session id and directory shown.
4. Press `s`. Write down what the frame says at each step - the hint line before the press, the header word, the notice, and what it settles on.
5. Ask the disk: `ls -l /tmp/t06607/sessions/*/*/`, `afinfo` on both WAVs, and the first lines of `session.json`. The durations should match roughly how long the session ran, not zero and not a truncated tail.
6. Ask whether it is still recording-as-a-run rather than exited: `lsof /tmp/t06607/record.lock` should still name the same pid, and `ps -p <pid>` should still list it.
7. With the holder *still* holding the mic, wait ~30 seconds and confirm no second session directory appeared (the key stops the call; it is not supposed to reopen one under a call that is still up). Then let `mic-hold` exit and start it again - a fresh session directory should appear without restarting `record`.
8. Mispress behaviour, three ways: press `s` while the frame reads `watching` with nothing recording; press `s` while the meeting selector or the roster pane is open; and press `r` then `n` and type a name that contains an `s` - the letter must land in the field, never stop the recording.
9. Narrow the terminal to about 40 columns and repeat 4 once: say whether the hint still communicates what `s` does when it wraps.

Paste raw output and copied cell text into this ticket. Anything that disagreed with the wording goes back to TASK-066.07 by its AC #5, and anything that disagreed with the sequencing goes back as a bug.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 HUMAN: in the same granted terminal, also run the automated live proof from TASK-066.07.02 and paste its full output: `cargo nextest run -p meethook --run-ignored only -E 'binary(live_record_hand_stop)' --nocapture` (the stand-in from step 1 must be built first). MEETHOOK_ACTIVITY_DEBUG changes nothing there - the pty driver scrubs it, because the proof reads the screen out of the same stream the diagnostics would print into - so debug a failing proof by hand with steps 1-7 above. If the proof cannot run at all, paste the refusal verbatim and name which machine condition failed.
<!-- AC:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-09-12 06:21
---
TASK-066.07.02 is now planned and this ticket depends on it, so the physical recipe and the automated pty proof get run in one sitting rather than two. One correction worth carrying into the recipe: the trigger counts *which processes are capturing* (`IsRunningInput` per process object, crates/meethook-record/src/activity.rs:1-100), not how loud anything is - there is no level threshold anywhere in the workspace. Steps 2 and 7 are therefore exactly right as written, and no amount of playing sound would substitute for the holder.
---

created: 2026-09-16 17:23
---
The automated proof this recipe now runs beside is green on hardware: TASK-066.07.02 passed in 27.14 s in a granted terminal, verbatim output on that ticket. Its AC #1 is therefore a ~27-second command rather than an open question — build the holder, then `cargo nextest run -p meethook --run-ignored only -E 'binary(live_record_hand_stop)' --nocapture`, and paste both that and your own keystrokes.

What remains yours alone are the two things no harness can answer: whether `s` is discoverable sitting next to Ctrl-C when you're mid-call, and whether pressing it actually rescues the runaway case — a call that ended while something kept the trigger pinned true. That second one is the reason the key exists.
---

created: 2026-09-16 17:41
---
Human run, partial (steps 1–3), verbatim report:

- hint row before the press: **did mention `s`** — the affordance is discoverable where it works.
- header immediately after: "not sure if it said finalizing, if it did it was very fast".
- settled on: `Watching for the next call`.
- overall read: "It seems like it's working correctly from my POV."

The middle line is a find, filed as TASK-066.07.04 with source evidence: `Note::Stopping` is painted when sent and replaced when `capture.finish` returns (`record.rs:1092-1112`, notes channel `record_screen.rs:132-137`, `got_note` draw gate `:399`), so its visible lifetime *is* the finalize duration and there is no display floor. The notice that exists to explain why recording stopped explains nothing, and the state left behind reads as though nothing happened. Steps 4–8 still owed; automated proof already green on .02.
---

created: 2026-09-17 01:11
---
Status after the human run, and an explicit waiver agreed with @human rather than an implied pass.

**Verified on hardware.** Steps 1–5: `s` advertised in the hint row before the press; frame settled on `Watching for the next call`; audio complete on both tracks for every finalized session — `123454` ≈15 s, `123534` ≈10 s (the hand stop), `124152` 838.6 s mic against 838.8 s speaker, i.e. 13m58s agreeing within 0.2 s. Nothing truncated. Findings filed rather than absorbed: the unreadable end-of-session notice went to TASK-066.07.04 (widened there to cover `Note::FinishFailed` as well), and `123326` — `speaker.wav` at 68 bytes, zero samples, no `session.json` — is the silent-track gate refusing before metadata exists (`meethook-record/src/lib.rs:417-427`); its cause (nothing playing vs. grant landing mid-session vs. SCK emitting nothing) is unresolved and nobody has claimed it.

**Not checked, and why it is acceptable.** `lsof`/`ps` after the press — the process had already been ended; the same fact is asserted four ways inside TASK-066.07.02's green run. `s` with the *meeting selector* open, and the ~40-column hint wrap — no calendar event existed on the day of the run. The selector case is partly pinned off-hardware: `record_screen.rs:659` asserts `s` in `KeyContext::Selector` resolves to `Action::StopSession`, and TASK-066.07.01's scripted tests cover finalizing while a pane is open. Unproven residue is therefore presentational only — what the screen looks like afterwards — and @human will file bugs if he hits it.

**Still owed, needs no calendar event, ~3 minutes:** step 7 (holder holds → no second directory for ~30 s; release and re-hold → new session with no restart of `record`), step 8's roster half (`r`, `n`, type `Tess` — the letter must land in the field, never stop the recording) and its watching-state no-op, and AC #1's 27-second nextest paste from this same granted terminal.
---

created: 2026-09-17 01:11
---
#3/#4 above stand: this ticket's single AC is the nextest paste, so the manual findings live here as evidence and route onward — wording to TASK-066.07, sequencing and notices to their own tickets.
---

created: 2026-09-18 17:37
---
Run complete on hardware 2026-09-16 (evening session, scratch root `/tmp/t06607c`, `mic-hold` as the foreign holder). All three manual tests and the automated proof reported **pass** by @human:

- **Test 1** (step 7, first half): pressed `s` mid-session; frame settled on `Watching for the next call`; one session directory; after `sleep 30` with `mic-hold` *still holding*, still exactly one. So the hand stop ended the session without reopening capture under a call that was still up.
- **Test 2** (step 7, second half): released the holder, waited ~10 s, started it again — a second session directory appeared on its own and the frame read `recording`. The `record` process was never restarted.
- **Test 3** (step 8, idle case): with the frame reading `watching` and nothing recording, `s` did nothing — no directory, no exit, no error.
- **AC #1**: `cargo nextest run -p meethook --run-ignored only -E 'binary(live_record_hand_stop)' --nocapture` passed from the same granted terminal with the microphone free (~27 s).

Stated precisely so this isn't an implied green: the outputs were reported verbally rather than pasted, so the verbatim nextest block is not attached here (scrollback can be appended later if it turns up). Step 8's roster half (`r`, `n`, type a name containing `s`) could not be attempted — see the next comment for why that is a real gap rather than a skip.
---

created: 2026-09-18 17:37
---
Two further findings from the same sitting, both closing old questions:

1. **The hint row does not advertise `r` when no meeting is attached**, which is the frame honouring its own rule (`render.rs:200-204`) that a key which cannot work in the current context is not offered. The silent `r` observed earlier was therefore correct behaviour, not a dead key — `open_roster` is gated on `roster.is_some()` (`state.rs:429-434`), and there is no roster without a meeting. Consequence for this ticket: the mispress battery's name-field case is genuinely untestable without a calendar invite, and is parked rather than waived — printable-letter collisions inside `KeyContext::RosterEditing` are pinned only by the unit assertion at `record_screen.rs:772`.
2. **A real call recorded the two tracks at different sample rates** — `mic.wav` at 24 kHz (the device follows the call) against `speaker.wav` at 48 kHz — and `transcribe` over that session produced a transcript @human reports as not garbled. So the per-track resample path at `meethook-transcribe/src/audio.rs:71-81` works on live hardware. Worth noting that no fixture in the workspace has ever written mismatched track rates, so this is currently proven by one afternoon on one Mac and pinned by nothing.
---
<!-- COMMENTS:END -->
