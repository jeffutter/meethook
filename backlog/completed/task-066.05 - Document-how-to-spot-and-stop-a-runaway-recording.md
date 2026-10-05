---
id: TASK-066.05
title: Document how to spot and stop a runaway recording
status: Done
assignee: []
created_date: '2026-09-09 23:04'
updated_date: '2026-09-18 17:56'
labels:
  - planned
dependencies:
  - TASK-066.05.01
documentation:
  - README.md
  - crates/meethook-record/examples/mic-activity.rs
parent_task_id: TASK-066
priority: medium
type: docs
ordinal: 25100
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Docs only, README.md. AC#5 of TASK-066 asks for "a stated way to spot and stop" a runaway recording. Today README documents how to stop a recording in six words ("until interrupted (Ctrl-C)", README.md:84-86), has no troubleshooting section at all, never mentions orphans or crashed sessions, and does not document any diagnostic env var — `MEETHOOK_ACTIVITY_DEBUG` appears nowhere in README.md or LINUX.md, only in code (activity.rs:262).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 #1 README gained the troubleshooting recipe and every command in it was run verbatim on macOS, with the real output pasted into this ticket.
- [x] #2 #2 MEETHOOK_ACTIVITY_DEBUG is documented in the Global options table beside --root and --template.
- [x] #3 #3 The two claims that could mislead are accurate as written: the orange mic dot is a lagging coarse signal, and a killed session's WAVs stay playable to within the 5-second checkpoint (track.rs:27-30) while still being skipped as orphans by transcribe and enroll.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs only, one file: README.md. Adds a troubleshooting subsection to the `record` section plus one row to the Global options table. TASK-066 AC#5 asks for "a stated way to spot and stop" a runaway recording; this is that statement.

**Why the earlier draft of this plan is rewritten rather than executed.** Two of its claims went stale underneath it. (a) It offered "two meethooks holding each other recording" as a live cause; TASK-066.03 closed that — `record` takes an OFD lock on `<root>/record.lock` and refuses a second instance by pid, start time and argv (`crates/meethook/src/record.rs:808`, refusal built at `:1305`). (b) It promised "Ctrl-C finalizes; `kill` or a closed terminal does not"; TASK-067.02/.03/.04 closed that — the root workspace builds ctrlc with `features = ["termination"]` (`Cargo.toml:74-80`), so SIGINT, SIGTERM and SIGHUP all reach the quit path, pinned by `sigterm_and_sighup_are_handled_so_a_kill_reaches_the_quit_path` (`record.rs:3081`). Only SIGKILL, SIGABRT and an Objective-C abort still die without finalizing. Neither stale claim may reappear; TASK-067.06 was written expecting to correct (b) inside our text, and now has a planner note saying not to.

**1. Heading and placement.** Insert after README.md:104 (the exclusions prose), before the blank :105 and ``### `meethook transcribe [SESSION_ID...]` `` at :106, as a fourth-level heading:

    #### Troubleshooting: the recording will not stop

Fourth level, not third, because the material is macOS-only and specific to `record`: a `###` would drop a non-command section between two command headings under `## Usage`. Nothing in README uses `####` yet, so the level is free. Keep the heading text exactly as above — TASK-067.06 adds its sibling `####` paragraph immediately after ours and cross-references it, so do not number the bullets or fold its salvage half into this one. Budget roughly 55 lines including fences. House style, evidenced: `sh`-tagged fences and no `$` prompts anywhere in README, imperative voice, heavy em-dashes, no emoji or exclamation marks, caveats phrased as flat declaratives (compare :196's "its presence alone does not mean something is recording").

**2. Body, in this order.** Every factual claim below is already cited from code in the tree or from `backlog/docs/research/doc-006`; none of it needs new research, only writing.

*a. The menu-bar dot proves very little (two or three lines).* Apple's own wording is that the indicator appears when the mic "is in use or has been used recently", and clicking Control Center names the app — a good first look, a poor measurement. The dot lights for any initialized HAL input unit, virtual devices included, regardless of audible sound, and has been observed lagging the real release by some twenty seconds even in Apple's own Voice Memos. Cite doc-006 §5 rather than re-deriving it.

*b. Is a meethook recording right now?* Every recipe begins `ROOT="${MEETHOOK_ROOT:-$HOME/meethook}"`, because with the variable unset `"$MEETHOOK_ROOT"/sessions/*` expands to a literal `/sessions/*` and an empty data directory looks like a healthy one. Then:

```sh
lsof "$ROOT/record.lock"
```

which prints COMMAND and PID for whoever holds the lock, and prints nothing when nobody does — the live counterpart to :195-196's promise that the file's presence alone proves nothing. Say why this beats `ps | grep meethook`: run here, that grep matched the shell performing the grep and an unrelated leftover `enroll`, so it answers "has meethook ever run on this machine?", not "is it recording?". With a pid in hand, `ps -p <pid> -o pid,ppid,lstart,etime,args` describes it; PPID 1 means no terminal owns it any more.

*c. Is it still writing?* Two samples of the newest track five seconds apart — `ls -l "$ROOT"/sessions/<id>/mic.wav`, `sleep 5`, again — growth being the actual proof. `du -sh "$ROOT"/sessions/*` sizes the damage (roughly 700 MB/hour, per the parent ticket) but says nothing about liveness. `lsof "$ROOT"/sessions/<id>/mic.wav` is the one-shot form: it names the writer, and prints nothing when no process holds the file. Warn off `stat -f %z`: in the dev shell GNU coreutils `stat` shadows BSD `stat` and `-f` there means "file system", so sizes come from `ls -l` or `wc -c <`.

*d. Stop it.* Ctrl-C in its terminal, or `kill -INT <pid>` from another shell. Plain `kill <pid>` (SIGTERM) and a hangup (SIGHUP) now take the same finalize path — state it, because users reach for those two constantly and the honest answer changed with TASK-067. What does not finalize: `kill -9`, an abort, a crash. Deleting `record.lock` accomplishes nothing; reuse the refusal's own words, "the lock is held by the operating system, not by the file."

*e. What you get back.* Three outcomes, and AC#3 is only accurate once they are split, because the draft's single sentence describes just the last: an interrupted run writes `session.json` and transcribes normally; a panicked or unwound run leaves an orphan whose WAVs the drop path still finalized, complete to the last sample (`crates/meethook-record/src/teardown.rs:29-36`); a `kill -9` leaves an orphan valid up to the last 5-second RIFF checkpoint (`crates/meethook-record/src/track.rs:25-30`). Either orphan is skipped, never repaired: transcribe prints `<id>  skipped: no session.json (the recorder crashed mid-session)` (`crates/meethook-transcribe/src/lib.rs:403`) and enroll `<id>  passed over: no session.json (...)` (`crates/meethook-enroll/src/narration.rs:515`), both at exit 0 (`crates/meethook/src/commands.rs:95-96`). Keep the playability claim scoped to what was measured: on the Mac that recorded it, `afinfo` reports an "estimated duration" clamped to the bytes actually present and CoreAudio readers play the file; strict parsers elsewhere are the ones that refuse (sox wants `--ignore-length`). Write "plays on the Mac that recorded it", not "every tool agrees". Turning orphaned audio into a transcript is TASK-067.06's paragraph — cross-reference it, do not write it.

*f. Make it stop happening.* Name what the trigger counts, using the installed binary:

```sh
MEETHOOK_ACTIVITY_DEBUG=1 meethook record --plain 2>activity.log
```

`[activity]` lines go to stderr: one summary per recomputation plus one per holder, with pid, bundle id, `exe=`, `devices=[...]` and `on-default=` (`crates/meethook-record/src/activity.rs:163-190`, formatting at `:911-925`). Two caveats that are easy to get wrong and must appear: the command refuses to start while a runaway still holds `record.lock` — the lock is taken at `record.rs:808`, before preflight and before any debug line prints — so it belongs *after* step d, as a way to decide what to exclude; and it is `record`, needing both TCC grants and starting to capture the moment something opens the mic. Reading the names it prints: browsers capture through helpers (`com.google.Chrome.helper`), any WebKit embedder reports `com.apple.WebKit.GPU`, plain binaries report `(no bundle id)` and are matched by their `exe=` path, `com.apple.CoreSpeech` beside `devices=[]` is a bystander, and `on-default=no` is not an acquittal because virtual devices aggregate. Exclusion goes in `exclusions.json` (:87-104) followed by a `record` restart. Then one sentence each for two alternatives: the source-tree probe `cd crates/meethook-record && MEETHOOK_ACTIVITY_DEBUG=1 cargo run --example mic-activity -- 60`, which opens no device and needs no permission but does need the tree and `nix develop`; and `lsaudio` (Homebrew, third-party, macOS 14+), the outsider that answers the same question — doc-006 §6.

*g. Why it happens stays two sentences.* A session runs while *another* process has registered input IO — not while anyone is audible (README:87 already carries the first half; doc-006 §1 and decision-006 hold the rest), so an always-on dictation tool, an assistant daemon or a virtual-device companion app can hold a session open indefinitely. Hand the overrun question to TASK-066.04; do not duplicate it and do not relitigate the predicate here.

**3. Global options table (AC#2).** One row appended after :181:

| `MEETHOOK_ACTIVITY_DEBUG` | unset | Print which processes hold the microphone, and why `record` started or stayed running, to stderr |

The header already reads "Flag / env var", so a flag-less first cell fits the existing shape; no trailing period, matching :180-181. Leave `MEETHOOK_CPU`, `MEETHOOK_TIMING_DEBUG` and `MEETHOOK_CALENDAR_DEBUG` out — filed as their own ticket so this one documents exactly one variable.

**4. Static verification an agent does here.** Run each added shell line verbatim and record outputs in Implementation Notes (convention: summarize, then quote the load-bearing lines — see TASK-065's notes). Planning already established the following, so re-run to confirm rather than discover: `lsof` on a path held by an OFD-locking stand-in printed `hold 15869 jeffutter 3u REG ... /private/tmp/lockdemo/record.lock`, and `lsof -t` printed the bare pid; `lsof` on a closed `mic.wav` printed nothing and exited 1; unset `MEETHOOK_ROOT` left the glob literal; `ps -eo pid,ppid,lstart,etime,args | grep -i '[m]eethook'` and `pgrep -fl meethook` both returned false positives on this machine; a 10-second header over 6 seconds of samples made `afinfo` report "estimated duration: 6.000000 sec" and made Python's `wave` return 6 seconds of frames without erroring; `stat -f %z` failed as described. If quoting the `mic-activity` example, rebuild it first — the binary cached under `crates/meethook-record/target` predates TASK-066.01 and prints holder lines without `exe=` or `devices=`.

**5. Verification that needs hands lives in TASK-066.05.01 (@human).** Starting a real recording, stopping it by signal from a second shell, and listening to what a kill left behind. Write the text so it stands on code and tests, cite that ticket's findings for the empirical half, and let this ticket close after it: the SIGINT/SIGTERM/SIGHUP wiring is pinned by test today, while the measured per-signal table belongs to TASK-067.01 and .07.

**6. Final checks.** `git diff --stat` shows README.md alone. Confirm the new heading nests (`###` then `####`, with ``### `meethook transcribe` `` intact below it), the options table still has three columns, and nothing contradicted :85's "until interrupted (Ctrl-C)" or :195-196's lock caveat. No Rust changed, so the gates reduce to the commit hook.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
README.md gained `#### Troubleshooting: the recording will not stop` under `### meethook record` (+94 lines, README.md only) and one `MEETHOOK_ACTIVITY_DEBUG` row in Global options. Every claim verified on this Mac (macOS 26.6.2) with the current build; the hardware half is still owed by TASK-066.05.01.

Static verification — commands run verbatim in the dev shell (GNU coreutils on PATH):

1. Lock holder naming, via a stand-in OFD holder (python taking F_OFD_SETLK on /tmp/t066-05/root/record.lock, pid 35216): `lsof "$ROOT/record.lock"` printed `python3.1 35216 jeffutter 3u REG 1,13 96 ... /private/tmp/t066-05/root/record.lock`; `lsof -t` printed `35216`. `ps -p 35216 -o pid,ppid,lstart,etime,args` printed `35216 35213 Thu Sep 10 19:21:05 2026 00:20 .../python3 /tmp/t066-05/holder.py ...`. After the holder died (file present, nobody holding): lsof printed nothing, exit 1.
2. One plan claim found WRONG: with no `record.lock` at all (a machine that never recorded), lsof does NOT print nothing — it prints `lsof: status error on /Users/jeffutter/meethook/record.lock: No such file or directory` plus its usage banner, exit 1. README now states both cases separately.
3. Second-instance refusal while held (real target/debug/meethook): exit 1, printing `/tmp/t066-05/root/record.lock is held by pid 35216`, `started 2026-09-10 19:21:05 -05:00`, `started as: meethook record --plain`, and `Deleting record.lock does not: the lock is held by the operating system, not by the file.` The same command with `MEETHOOK_ACTIVITY_DEBUG=1` printed zero `[activity]` lines (grep -c = 0), so the lock really is taken before preflight and before any debug output — which is why the debug recipe belongs after stopping the runaway.
4. Orphan handling (synthetic root: WAVs, no session.json): `transcribe` printed `20260910-010000  skipped: no session.json (the recorder crashed mid-session)` then `Nothing to transcribe.`, exit 0. `enroll --plain` printed `20260910-010000  passed over: no session.json (the recorder crashed mid-session)` then `0 named, 0 skipped, 1 session(s) passed over`, exit 0.
5. Kill -9 shape (header/checkpoint declares 5 s, file holds 6 s of samples): afinfo reported `estimated duration: 5.000000 sec`, audio bytes 480000, while 576000 data bytes are present; python's wave opened it at 240000 frames = 5.0 s without erroring; afplay streamed it (a 1-second-declared twin exited 0 in 1.4 s; the 5-second one was still playing when timeout killed it). So the playability claim holds as scoped: plays on the Mac that recorded it, last <=5 s on disk but unread.
6. Liveness and size recipes: two `ls -l` samples 5 s apart (identical size == dead writer); `du -sh "$ROOT"/sessions/*` gave 1.2M for the synthetic root and 1.7G / 1.1G for the two real sessions; `lsof .../mic.wav` with no writer printed nothing, exit 1; `wc -c < mic.wav` gave 576044; `stat -f %z mic.wav` failed as promised with `stat: cannot read file system information for '%z'` plus a filesystem dump, exit 1 (/nix coreutils stat shadows /usr/bin/stat; the BSD form gives 576044).
7. Glob trap: `echo "$MEETHOOK_ROOT"/sessions/*` with the var unset printed the literal `/sessions/*`; with `${MEETHOOK_ROOT:-$HOME/meethook}` it printed the two real session dirs.
8. ps/pgrep false positives: `ps -eo pid,ppid,lstart,etime,args | grep -i '[m]eethook'` returned three lines, none a recorder — the bash -c wrapper doing the grepping, and a ten-day-old `tmux new-session ... ./target/debug/meethook --root /tmp/probe3 enroll 20260809-052600` sitting at PPID 1. `pgrep -fl meethook` returned the same pair.
9. The mic-activity example is current (the cached-binary warning in the plan no longer applies): `cd crates/meethook-record && MEETHOOK_ACTIVITY_DEBUG=1 cargo run --example mic-activity -- 6` ran with no TCC grant, opened no device, and printed `pid=36679 (no bundle id) IsRunningInput=false exe=/Users/.../examples/mic-activity devices=[] on-default=unknown   <- meethook`.

Two numbers corrected against the plan and the parent ticket:

- The growth rate is ~690 MB/hour PER TRACK, not per session: afinfo reports the real tracks as 48 kHz Float32 mono (1536000 bit/s = 192000 B/s), and session 20260818-132033 holds a 736,339,268-byte mic.wav and a 736,366,148-byte speaker.wav across 3835 s (~64 min). README says ~690 MB an hour per track and close to 1.4 GB an hour for the pair. TASK-066's "~700 MB/hour observed locally" reads as a session figure and understates a pair by half.
- `lsof` names whoever has record.lock OPEN rather than who holds the OFD lock. That is safe because RecordLock keeps the descriptor for the whole life of the process (meethook-session/src/record_lock.rs), and README says it in exactly those terms rather than implying lsof reads the lock.

Not run here, stays with TASK-066.05.01: `kill -INT` and `kill -9` against a live `meethook record`. This session cannot start one — preflight refuses with `missing macOS permissions` (Screen & System Audio Recording and Microphone are attributed to the launching app, which here holds neither grant), so the live half genuinely needs a TCC-granted terminal.

Parked, not Done, and the writing is finished and committed (2958b80, README.md only). What remains is AC#1's last clause, and it is genuinely hardware-gated rather than unfinished effort: the recipe tells a user to `kill -INT <pid>` a live recorder and promises that run finalizes, and `kill -9` and promises it does not. No agent can produce that evidence from this machine state — starting a recording is refused by preflight with `missing macOS permissions`, because TCC attributes Screen & System Audio Recording and Microphone to the launching app (crates/meethook/src/record.rs preflight, reached only after RecordLock::acquire at record.rs:808), and no app in this session's chain holds either grant. TASK-066.05.01 is @human for exactly this reason, and its AC#1-#3 are those three observations. Its plan says the same thing this park does: write the text so it stands on code and tests, cite that ticket's findings for the empirical half, close after it.

Next actionable step, in order: (1) the human runs TASK-066.05.01's five steps and pastes the raw outputs there — including the two checks added to it today, lsof on a machine that HAS recorded before, and what du shows for a runaway of known length; (2) anything that disagreed with README gets fixed here (its AC#4 routes findings back); (3) re-check AC#1 here against those outputs and mark Done. Until then `backlog task list -s Blocked --ready` will release this ticket the moment 066.05.01 closes.

AC#2 and AC#3 are checked: the env var row is in the Global options table beside --root/--template, and both misleading claims were rewritten against measurement rather than the earlier draft — the dot paragraph cites the lag and the virtual-device false positive, and the outcome split now distinguishes interrupted (session.json, transcribes) from panicked (orphan, headers finalized complete to the last sample) from killed (orphan valid to the last 5 s RIFF checkpoint, measured: afinfo reported 5.0 s over 6 s of samples), with playability scoped to "plays on the Mac that recorded it".

**AC #1 closed against TASK-066.05.01, now Done.** Every command in the shipped recipe has been run verbatim on macOS with the current build from a TCC-granted terminal, raw outputs pasted there: `lsof "$ROOT/record.lock"` named `meethook 41117` while live and printed nothing twice over after the stop; `ps -p` described it (`PPID 48861`, i.e. terminal-owned rather than orphaned); two `ls -l` samples five seconds apart showed 14M then 15M; `kill -INT` produced `Stopping...` and `Recorded 20260918-124102 (120.1s mic, 120.2s speaker)` with a `session.json` that `transcribe` accepted; and a separate `kill -9` left both WAVs static with no `session.json`, `afinfo` at 40.30 s / 39.92 s, and exit 0.

One wording question this run settled, because two observations of the same skip looked like a contradiction. This ticket's own notes record a synthetic orphan printing `skipped: no session.json (the recorder crashed mid-session)` on 2026-09-10, while tonight's real kill -9 printed `skipped: no session.json: no transcript is possible; the mic track holds 3.8 s past the end its header declares, and the speaker track holds 4.3 s past the end its header declares`. Both are true of the tree each was measured against: `ed34f54` (2026-09-11, *say what an unfinished directory proves right now, not what happened to it*) retired the parenthetical form in favour of `interrupted.rs:185-217`'s per-track diagnosis, and **updated README.md:166 in the same commit** — `git blame` confirms the fence is `ed34f541` inside a block otherwise written by `2958b80`. So the documentation tracked the code, and the only place the dead string still lives is ticket prose: this ticket's plan step *e* and .05.01's AC #3 quote it. Nothing to fix in README.

The real run carried two clauses where README's example carries one, which is the documented per-track shape rather than drift. It also put the measured overruns — 3.8 s and 4.3 s — inside the promise at README.md:160-162 that `afinfo` reports a duration up to five seconds short, so that sentence is now hardware-measured rather than asserted.
<!-- SECTION:NOTES:END -->

## Comments

<!-- COMMENTS:BEGIN -->
created: 2026-09-12 00:44
---
Heads-up while you finish AC#1: TASK-066.04 (planned) will make `record` print, once a session passes its attached meeting's end plus two minutes, a block like 'Recording has run 12m past the end of "Standup" and will not stop on its own. Press Ctrl-C to stop.' followed by '  holding   pid=… exe=… devices=[…] on-default=…' lines (or '  holding   none reported by the trigger'). Your section g already hands the overrun question to it; when that lands, one sentence in your troubleshooting text can point at the line instead of describing the invisible state as invisible. Do not block on it — your ACs hold without it, and the wording may still move if TASK-066.04.02's live run finds it misleading.
---
<!-- COMMENTS:END -->
