---
id: doc-006
title: >-
  Prior art: mic-holder attribution, runaway stop, and single-instance locking
  for TASK-066
type: other
created_date: '2026-09-09 21:48'
updated_date: '2026-09-09 21:55'
---

Research for TASK-066 (recording runs on after the meeting ends, and after quitting).
Sources: Apple headers/docs and developer forums, CoreAudio bindings in active use
(`sbooth/CAAudioHardware`, `corti-coreaudio`, `mlaify/whositwhatsit`, `mickeyl/lsaudio`,
`Coalesce-Software-Inc/sountop`), a production write-up of the same detection problem
(Mac Note Taker), peer meeting-recorder implementations (Granola docs, OpenOats, Ariso oats,
clovy), Unix locking references (XNU `fcntl.h`, PostgreSQL, `fd-lock`, pidfile critiques),
and crash-recovery patterns for WAV writers. Nothing here was measured on the reporting
machine; items marked *(verify)* are leads, not facts.

## 1. What `kAudioProcessPropertyIsRunningInput` really says

The predicate's meaning is the crux of candidate (a) — "something reports true just because
the device is streaming".

- It is **1 when the process has at least one active registered input stream**, not when it
  is capturing audible sound. `corti-coreaudio` states this flat out in its module docs:
  "`IsRunningInput` reflects IO registration, not non-silent audio — so a muted Zoom still
  reads as the owner" ([docs.rs source](https://docs.rs/crate/corti-coreaudio/latest/source/src/process.rs));
  Mac Note Taker independently describes it as "1 when the process has at least one ACTIVE
  INPUT stream" ([field notes](https://macnotetaker.com/blog/which-app-is-using-mic-coreaudio-process-objects)).
  So the predicate answers "who asked the HAL for an input stream", and any piece of software
  that keeps an input unit initialized forever — a virtual-device companion app, an assistant
  daemon, a DAW's always-on monitor — pins it. That is consistent with the ticket's revised
  ranking, and it is why "which driver is installed there" decides reproducibility.
- The related property nobody in this codebase uses yet: **`kAudioProcessPropertyDevices`**,
  readable per scope (`kAudioObjectPropertyScopeInput`) on each process object, returns the
  devices that process is using
  ([CAAudioHardware `AudioProcess.swift`](https://github.com/sbooth/CAAudioHardware/blob/main/Sources/CAAudioHardware/AudioProcess.swift)).
  This is the sharpest available lever against candidate (a): require the holder to be running
  input *on the input device we care about* rather than on any device, so a host parked on
  BlackHole/Loopback/Wave-Link stops counting as somebody's meeting — and, separately, name
  the device in the debug output so the holder is identifiable without guessing from a pid.
  Also available: `kAudioHardwarePropertyTranslatePIDToProcessObject` (pid → object), which is
  the reverse lookup a `ps`-style diagnostic needs.
- `kAudioProcessPropertyIsMuted` is declared in `AudioHardware.h` but has no public definition
  (open TODO in the same binding), so muting cannot be used to refine the predicate.
- Availability conflicts across sources: the bindings gate the whole class on **macOS 14.2**,
  Mac Note Taker says 14.2 (same release as Core Audio taps), while `corti-coreaudio` and the
  comment at `activity.rs:16` say 14.4. Worth pinning against the SDK header rather than
  trusting either write-up *(verify)*.
- Enumerating process objects needs **no TCC grant** — it is coreaudiod state, not audio
  content ([Mac Note Taker FAQ](https://macnotetaker.com/blog/which-app-is-using-mic-coreaudio-process-objects)).
  Confirmed by `mic-activity`'s own docs; it means the diagnostic is runnable anywhere, which
  is what AC#1 depends on.
- Names are not meeting-shaped. Browsers capture through helpers: a Meet call reports
  `com.google.Chrome.helper`, and generic WebKit capture surfaces as `com.apple.WebKit.GPU`
  (prefix mapping needed, and Safari/WebKit is genuinely ambiguous). Any feature that displays
  "who started this session" needs a mapping table, and the `exclusions.json` exact-match rule
  will frequently need the *helper* id, not the app the user sees in the Dock — a usability
  wrinkle worth calling out for anyone told to add an entry (AC#3).

## 2. Listener behaviour: the codebase's existing choices are the ones the field converged on

- Real deployments get **notification storms, not edges**: Mac Note Taker measured 9+ callbacks
  per second during a Meet join with AirPods connected, most with no actual state change, and
  their postmortem rule is "dedupe against last-known state, and rate-limit expensive work
  triggered by system listeners" — after a livelock that pinned a main thread at 100% CPU.
  That is exactly `edge()` plus the wake-a-recomputation design of `State::notified`
  (`activity.rs:437`), and it corroborates (from outside this repo) dropping the per-process
  `IsRunningInput` listener: the recommendation is not to trust those triggers to carry state.
- Independent polling as a *safety net* is standard practice, not a crutch: the closed-source
  and open peers below poll ownership on a ~1 s cadence with multi-sample confirmation
  (clovy requires 15 consecutive negative probes before finishing). The 2-second live-session
  `recheck` is in line with that; making the interval configurable is probably not worth it.

## 3. Peer products do not have one stop trigger

Every comparable recorder ends a session on several independent signals, several of which are
deliberately *not* derived from the mic predicate — precisely because the mic predicate can be
wrong, which is this ticket's bug.

| Product | Stop triggers |
|---|---|
| Granola | Detected call end (transcript length, whether you're still using the call app, **calendar scheduled end**) **and a hard "15 minutes with no new audio"** ([docs](https://docs.granola.ai/help-center/taking-notes/transcription)) |
| OpenOats (issue [#186](https://github.com/yazinsai/OpenOats/issues/186)) | Meeting-app process exit; configurable silence timeout (default 15 min); system-level trigger |
| Ariso oats (PR [#170](https://github.com/ariso-ai/oats/pull/170)) | Calendar `end_at` + 2 min grace → **user prompt** Stop / Keep recording; ignoring it keeps recording, explicitly to survive back-to-back calls |
| clovy (PR [#955](https://github.com/open-software-network/os-clovy/pull/955)) | Keeps monitoring CoreAudio ownership after start, records the *originating app families* at start, and requires every one of them absent for 15 consecutive 1 s probes (≥15 s elapsed) before auto-finishing |

Two transferable shapes here, both independent of identifying the offending holder:

- **A ceiling that does not depend on the predicate**: an elapsed-time / no-new-audio cap
  (peers use 15 min) that finalizes or at minimum prompts. meethook already measures frames
  continuously (`TrackWriter::progress`), so a no-audio-progress cap needs no new plumbing.
- **A calendar backstop that asks rather than acts.** meethook already knows the matched event
  (`decision-009`), and every peer that uses the calendar prompts instead of auto-stopping —
  auto-stop on schedule would cut back-to-back calls, which is why they don't.
- Related: a session whose duration has blown past the matched event should show *why* it won't
  stop (holder identity), which is the ticket's "make a runaway visible" candidate and matches
  the prompt-not-silent-stop convention above.

## 4. Single-instance locking, if the guard finally gets built

`decision-006` line 17 records that a pidfile/advisory-lock guard was considered and deferred;
building it now is reversing that recorded decision, so the plan should say so explicitly.
What the field says about how:

- **Advisory kernel locks beat pid files**, decisively: the kernel drops the lock when the
  process dies, whereas a pidfile needs a liveness check that is inherently racy (pid reuse,
  PID namespaces, reparenting) — see ["Nobody does pidfiles right"](https://yakking.branchable.com/posts/procrun-2-pidfiles/)
  and the [pidfile anatomy](https://digitalgarden.bhekani.com/pid-files/) write-up. Postgres's
  namespace-confusion bug (its lockfile said "stale" for a live instance in another PID
  namespace) is the cautionary tale for reading identity out of file *contents*
  ([patch discussion](https://www.postgresql.org/message-id/attachment/187046/v1-0001-Use-open-file-description-locks-for-data-director.patch)).
  Contents (pid, argv, boot time) belong in the file **for messaging only** — "another record is
  running as pid N, started …" — never for deciding whether the lock is held.
- **Prefer OFD locks (`F_OFD_SETLK`) over `fcntl` POSIX locks.** POSIX record locks are keyed
  per-process-per-inode, so closing *any* descriptor for that file anywhere in the process drops
  the lock — a genuine hazard in a program that touches its own root directory for sessions,
  exclusions, and transcripts. Open-file-description locks are per-descriptor like `flock` and
  are what Postgres migrated to. Both XNU and Linux expose them
  (`F_OFD_SETLK`/`F_OFD_SETLKW`/`F_OFD_GETLK` in
  [`darwin-xnu/bsd/sys/fcntl.h`](https://github.com/apple/darwin-xnu/blob/main/bsd/sys/fcntl.h)),
  so they're usable on both supported platforms. `flock(2)` itself is available on Darwin too
  ([man page](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/flock.2.html));
  `fs2`'s `try_lock_*` wraps it. Confirm the Rust wrapper chosen doesn't silently fall back to
  POSIX locks *(verify)*.
- **`FD_CLOEXEC` is load-bearing here, not boilerplate.** An inherited lock descriptor turns
  any long-lived spawned child into a lock holder with no owner — and this ticket already
  documents exactly such a child (the reparented `enroll`, PPID 1, alive 4 days 16 hours). Every
  spawn path (`paplay`/`aplay`/`ffplay`/`mpv` clip playback, spawned test binaries) must not
  inherit it.
- **Acquisition should be non-blocking with a clear refusal**, matching the range-refusal
  convention already in the CLI (refuse rather than clamp). A wait-or-take-over option invites
  the two-instances-holding-each-other-recording failure the guard is meant to prevent.
- Advisory locks are opt-in and freely ignorable: coordination, not enforcement
  ([`fd-lock`](https://github.com/yoshuawuyts/fd-lock) README is explicit it "should never be
  used for security purposes"). Fine for this purpose, but it means the guard does not protect
  against a *different* tool recording — which the predicate still has to handle.
- One asymmetry to design deliberately: the guard belongs to `record` alone, not to `enroll` /
  `transcribe` / `speakers`, which are safe concurrently and — per the invariant that `enroll`
  must be able to bring any stale transcript up to date — must not be blocked by a live
  recording.

## 5. Teardown, signals, and what the mic light actually proves

TASK-067 owns this, but two findings bear directly on TASK-066's AC#5 severity framing.

- **The orange indicator is a lagging, coarse signal.** It lights when *any* HAL Audio Unit
  input is initialized — including a virtual device, and regardless of audible sound
  ([Happy Mac Admin](https://happymacadmin.wordpress.com/2022-02-22/orange-is-the-new-mac/);
  [lurar's process-tap switch](https://github.com/lsjoberg/lurar/commit/5840d2105cb31fe5c1fadbac2e8d475d0b9383cc)
  exists purely to stop lighting it). The menu-bar dot has been observed to lag the real
  release by ~20 s, including in Apple's own Voice Memos
  ([Apple dev forum](https://developer.apple.com/forums/thread/742272)), and stuck-dot reports
  persist with no microphone-capable app running at all
  ([Apple discussions](https://discussions.apple.com/thread/253840109)).
  Consequence for the repro: "the mic light stayed on after I quit" is weak evidence that
  meethook kept capturing — it must be corroborated by growing WAVs or a `IsRunningInput`
  reading, not treated as the measurement.
- **Signal coverage.** `ctrlc` covers the terminal interrupt; SIGHUP/SIGTERM need
  `signal-hook` / `tokio::signal::unix`, handlers must stay async-signal-safe (flag + channel —
  the shape `record.rs:704-709` already uses is the documented-correct one), signals coalesce,
  and a second Ctrl-C should force-exit a stuck finalize
  ([rust-cli book](https://rust-cli.github.io/book/in-depth/signals.html),
  [`signal_hook`](https://docs.rs/signal-hook/latest/signal_hook/)).
- **Half-written WAVs are much less catastrophic than the ticket assumes.**
  `track.rs:24-29` checkpoints the RIFF/data sizes every 5 s via `hound`'s flush-as-checkpoint,
  so a killed recorder leaves a *playable* file missing at most the last 5 s — not an unreadable
  placeholder-header file. Peer projects converge on the same conclusion and add the missing
  half: a **startup repair pass** that rewrites unfinalized headers before orphan recovery
  (`pasrom/meeting-transcriber` commit `b1e0d98`), a reader that clamps a truncated final
  `data` chunk to the bytes that exist
  ([WAV-on-disk deep dive](https://gophertrunk.org/blog/deep-dives/recording-streaming-05-wav-on-disk/)),
  and a user-facing repair tool for pathological cases ([`wavfix`](https://github.com/agfline/wavfix)).
  For AC#5 that reframes the cleanup: runaway sessions on disk are mostly *salvageable audio*,
  and `transcribe` over them (or a repair step) is likely the answer rather than deletion.
- **Capture can outlive the client — and leaks accumulate.** Projects hitting this describe
  ScreenCaptureKit sessions living outside the client process boundary and accumulating across
  restarts when `.stop()` is never called ([loop #4](https://github.com/tadaspetra/loop/issues/4)),
  plus `replayd`/ControlCenter churn attributed to persistent streams
  ([screenpipe #2676](https://github.com/screenpipe/screenpipe/issues/2676),
  [OBS #10636](https://github.com/obsproject/obs-studio/issues/10636)). So "several stacked
  runaway sessions with no second meethook" is a coherent story, not only a second instance.
  If the fix ever becomes "supervise the recorder so it dies with the UI": macOS has no
  `PR_SET_PDEATHSIG`; the non-polling answers are kqueue `EVFILT_PROC`/`NOTE_EXIT`
  (Apple [TN2050](https://developer.apple.com/library/archive/technotes/tn2050/_index.html);
  the `mothership` crate implements exactly this) or the parent-death-is-a-closed-pipe trick.

## 6. Cross-checks available on the reporting machine without writing code

Independent of `MEETHOOK_ACTIVITY_DEBUG`, these read the same underlying state, so a
disagreement between them and meethook is itself a result:

- `sountop` — TUI/log-mode monitor over the same per-process CoreAudio APIs, showing input and
  output activity per process ([repo](https://github.com/Coalesce-Software-Inc/sountop)).
- `lsaudio` — lists processes playing/recording audio and can kill them by pid, which is close
  to the "spot and stop one" recipe AC#5 asks for ([repo](https://github.com/mickeyl/lsaudio)).
- `OverSight` — identifies the process behind each mic activation, and works when the culprit
  never appears in any app list ([Objective-See](https://objective-see.org/products/oversight.html),
  described in [The Art of Mac Malware, ch. 12](https://taomm.org/vol2/pdfs/CH%2012%20Mic%20and%20Webcam%20Monitor.pdf)).
- Removing and re-adding an app under System Settings → Privacy → Screen & System Audio
  Recording is the reset path users reach for when capture state goes stale
  ([omi #6640](https://github.com/BasedHardware/omi/issues/6640)) — useful as a dichotomy test:
  if a stuck `IsRunningInput` clears only after that, the state is held framework-side.
- In-repo, `mic-hold` plus `mic-activity` already form the automated matrix: hold the device
  from a second process and watch for a *third* pid at `IsRunningInput=true` while meethook
  records. Per its own docs, that third pid appearing is precisely candidate (a) reproducing
  with no meeting involved.

## 7. Diagnostic gaps found locally while reading

Two things the current debug output cannot answer, which AC#1/#3 need:

1. `State::log` prints **bundle id and pid only** (`activity.rs:556-586`). The classifier
   already resolves the canonicalized executable via `proc_pidpath` for every candidate, but the
   log never prints it — so the most likely culprit shape in this ticket, a bare binary or a
   helper with no bundle id, renders as `pid=NNNN (no bundle id) IsRunningInput=true`, which
   names nothing. Printing the resolved path (and, ideally, the device(s) from
   `kAudioProcessPropertyDevices`) is a small change that turns the reporter's run into an
   answer instead of a puzzle.
2. Nothing in the debug output distinguishes "this holder is on my default input device" from
   "this holder is parked on some virtual device", which is the distinction candidate (a) hinges
   on. Same property read closes it.

Neither is a fix to the trigger; both are instrumentation the acceptance criteria implicitly
require.
