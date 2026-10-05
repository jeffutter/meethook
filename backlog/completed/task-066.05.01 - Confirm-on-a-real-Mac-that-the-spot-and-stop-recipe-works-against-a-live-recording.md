---
id: TASK-066.05.01
title: >-
  Confirm on a real Mac that the spot-and-stop recipe works against a live
  recording
status: Done
assignee:
  - '@human'
created_date: '2026-09-10 23:58'
updated_date: '2026-09-18 17:54'
labels:
  - record
  - docs
  - planned
dependencies: []
parent_task_id: TASK-066.05
priority: high
type: spike
ordinal: 44100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The hardware half of TASK-066.05's AC#1 ("every command in it was run verbatim on macOS, with the real output pasted into this ticket"). Most of that recipe is static — `lsof`, `ls`, `du`, the unset-`MEETHOOK_ROOT` glob trap — and an agent verifies those in 066.05 itself. What cannot be settled by reading code or watching a build is the half that needs a recording actually running: naming a live recorder through `record.lock`, stopping it by signal from a second shell, and hearing what a `kill -9` left behind.

Recipe under test, as 066.05 will write it:

```sh
ROOT="${MEETHOOK_ROOT:-$HOME/meethook}"
lsof "$ROOT/record.lock"                                  # names COMMAND + PID of the live recorder
ps -p <pid> -o pid,ppid,lstart,etime,args                 # what that pid is; PPID 1 => nobody owns it
ls -l "$ROOT"/sessions/<id>/mic.wav; sleep 5; ls -l "$ROOT"/sessions/<id>/mic.wav   # bigger => still capturing
kill -INT <pid>                                           # finalizes; kill -9 does not
```

Setup that gets `record` into a session without anyone speaking: from a TCC-granted terminal (Ghostty/iTerm/WezTerm/VS Code — plain Terminal.app has no Microphone grant here), start `crates/meethook-record`'s `mic-hold` example (`cargo run --example mic-hold -- 120`) so `record` takes its already-active path immediately. Same trick TASK-067.01 uses. Use the current build, not the `v0.3.0` baseline 067.01 measures against.

Scope discipline: do NOT re-run the whole death matrix. TASK-067.01 and TASK-067.07 own the per-signal table — cite them for SIGTERM/SIGHUP rather than repeating rows. Only Ctrl-C, `kill -INT` and `kill -9` matter here, because those are the three things the README tells a user to do.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 HUMAN: With a live meethook record on the current build from a TCC-granted terminal, lsof "$ROOT/record.lock" named it (COMMAND and PID), and the same command printed nothing (or "No such file") once it had stopped.
- [x] #2 HUMAN: kill -INT <pid> from a second shell finalized the run: it printed the plain-mode stopping lines, the session directory gained session.json, and meethook transcribe accepted that session rather than skipping it.
- [x] #3 HUMAN: A separate run killed with kill -9 left both WAVs and no session.json; transcribe printed "skipped: no session.json (the recorder crashed mid-session)" at exit 0; the two-sample ls -l showed the track growing before the kill and static after; afinfo reported a duration within 5 seconds of the recorded wall-clock length and the file played on that Mac.
- [x] #4 HUMAN: Anywhere the recipe did not behave as README claims — empty output where output is promised, a wrong column, a playability claim that did not hold — got fixed in README.md or reported back on TASK-066.05 before this ticket closes.
- [x] #5 HUMAN: Raw command outputs appended to this ticket (notes, or a file under backlog/docs/reports/) so TASK-066.05 cites them rather than re-measuring.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
No code changes. Run, verbatim and on a real Mac with the current build, each command in README.md's new troubleshooting subsection, having started a live recording first via `mic-hold` as the description says.

1. Start `meethook record --plain` (plain output keeps the status lines visible in the log) in one window, with `mic-hold` holding the mic so a session actually begins; note the wall-clock start time so the afinfo duration check has something to be compared against.
2. In a second window run `lsof "$ROOT/record.lock"`, then `ps -p <pid> -o pid,ppid,lstart,etime,args` on the pid it named, then the two-sample `ls -l` on the newest `mic.wav`. Paste all three outputs.
3. `kill -INT <pid>`. Confirm the stopping lines appeared, that `session.json` exists in the new session directory, and that `meethook transcribe <that id>` produced a transcript rather than the skip line.
4. Repeat with `kill -9 <pid>` on a fresh session and collect the same evidence plus `afinfo` on both tracks; play one back to confirm audio is present up to within about five seconds of the kill.
5. Append the outputs as notes here, one block per step, and state per step whether README's wording matched what happened. Anything that did not match goes back on TASK-066.05 as a comment before closing.

Do not open a follow-up ticket for a signal-behaviour surprise — that belongs on TASK-067 (its death matrix owns those rows); this ticket only reports whether the documented recipe is true.
<!-- SECTION:PLAN:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-09-18 17:54
---
Run complete on hardware 2026-09-18, current build, scratch root `/tmp/t06605`, `mic-hold` as the holder so a session started without anyone speaking. Verbatim outputs below.

**AC #1 - naming a live recorder.**

```text
❯ lsof /tmp/t06605/record.lock
COMMAND    PID      USER   FD   TYPE DEVICE SIZE/OFF     NODE NAME
meethook 41117 jeffutter    3u   REG   1,13      129 14081689 /private/tmp/t06605/record.lock

❯ ps -p 41117 -o pid,ppid,lstart,etime,args
  PID  PPID STARTED                      ELAPSED ARGS
41117 48861 Fri Sep 18 12:41:02 2026       00:31 ./target/debug/meethook record --plain
```

`PPID 48861` is the launching shell, i.e. somebody owns the process - the healthy case the recipe's `PPID 1 => nobody owns it` note exists to catch. After the stop, the same `lsof` printed nothing, twice. Liveness by two samples: 14M then 15M five seconds later.

**AC #2 - `kill -INT` finalizes.** Plain-mode scrollback, kept whole:

```text
❯ MEETHOOK_ROOT=/tmp/t06605 ./target/debug/meethook record --plain
Watching the default microphone. Press Ctrl-C to stop.
A microphone is already in use; recording immediately.
Session 20260918-124102
  /tmp/t06605/sessions/20260918-124102
  mic       48000 Hz, 1 channel(s) reported by the input device
  speaker   48000 Hz
Recording... press Ctrl-C to stop.
Stopping...
Recorded 20260918-124102 (120.1s mic, 120.2s speaker) to /tmp/t06605/sessions/20260918-124102
```

`session.json` appeared, and `transcribe` **accepted** the session rather than skipping it - echo cancellation reasoned about it, the speech gate scanned both tracks, found none (nobody spoke), and wrote a transcript with zero turns. Three readers agree to a tenth of a second: the printed `120.1s / 120.2s`, the speech gate's `120.1 s / 120.2 s`, and `afinfo`.

**AC #3 - `kill -9` leaves what the recipe says.** Growth 2.0M then 3.0M; killed; both WAVs static at 8.5M with **no** `session.json`; `afinfo` reporting 40.30 s mic and 39.92 s speaker against a start of 12:45:21 and a kill around 12:46. Then:

```text
❯ ./target/debug/meethook --root /tmp/t06605 transcribe 20260918-124521; echo "exit $status"
20260918-124521  skipped: no session.json: no transcript is possible; the mic track holds 3.8 s past the end its header declares, and the speaker track holds 4.3 s past the end its header declares
Nothing to transcribe.
exit 0
```

Stated precisely rather than overclaiming: `afplay` was run and returned without complaint, but nobody asserted audible content, and the recording contains no speech by construction. So "the file played" rests on it decoding and exiting cleanly, not on anybody hearing it. A run with real audio would make that half solid.
---

created: 2026-09-18 17:54
---
**AC #4 - does the recipe match what ships?** Yes, and the mismatch was in this ticket, not in README. AC #3 quotes `skipped: no session.json (the recorder crashed mid-session)`, which is how the plan phrased it; the shipped string comes from `meethook-session/src/interrupted.rs:185-217` and reads `no session.json: no transcript is possible` followed by one clause per track. README.md:163-167 states exactly that, including the five-second checkpoint explanation and an example line carrying the speaker clause alone. Reality carried **both** clauses (mic 3.8 s, speaker 4.3 s), which is the documented per-track form rather than a deviation. Nothing to fix in README; the stale wording lived here.

Two corroborations worth keeping upward:

1. The measured overruns - 3.8 s and 4.3 s - land inside README's promise that `afinfo` reports "a duration up to five seconds shorter than what was recorded" (`README.md:160-162`). That sentence is now measured on hardware rather than asserted.
2. `mic 48000 Hz ... reported by the input device` here, where the holder is a passive tap, against 24 kHz during a real call the same evening. Same binary, same device. TASK-078's premise confirmed from the other side.

Demonstrated incidentally: after `kill -9` the `record.lock` file remained and the next `record` took it without complaint, which is README.md:155 saying the lock is held by the operating system rather than by the file.

Shell note for whoever runs this again: the recipe's paths are `"$ROOT"/sessions/<id>/mic.wav`, one level down. A plan-side glob written `sessions/*/*/mic.wav` matches nothing, and under fish an unmatched wildcard is an error rather than a literal path handed to `ls`. Both hiccups in this run were the harness's, not the product's.
---
<!-- COMMENTS:END -->
