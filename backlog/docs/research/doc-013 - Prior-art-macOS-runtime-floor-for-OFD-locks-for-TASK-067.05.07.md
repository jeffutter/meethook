---
id: doc-013
title: 'Prior art: how far back macOS runtime support for F_OFD_* goes, for TASK-067.05.07'
type: research
created_date: '2026-09-22 03:40'
updated_date: '2026-09-22 06:05'
---

## Purpose

Gathered for planning `task-067.05.07 - Establish the macOS runtime floor for OFD locks, and tell
users what meethook does below it`. It supplies the evidence that closes the `(verify)` doc-008 §4
left in its "Availability." bullet, and the matching open item 4 of doc-008 §6. Those two live in
doc-008 itself, not here: this file holds the citations, doc-008 holds the conclusions, so line
numbers into doc-008 are deliberately not quoted (they shifted when the answer landed). Not a plan;
no production-code conclusions are drawn here beyond quoting what the code already does.

Method note up front: the question is answerable from first-party sources, but only by reading
*source*, not docs. Apple's `fcntl(2)` carries no per-command availability information, and its
HISTORY section did not begin to mention OFD locks until macOS 14 - where what it says is "Open file
description locks first appeared in Linux 3.15" (§3), pointing away from Apple rather than at a
release. The SDK headers carry **no** `API_AVAILABLE` / `__AVAILABILITY_INTERNAL_*` annotation on the
`F_OFD_*` defines in any SDK on this machine (15, 15.4, 26, 26.5 - grepped for
`API_AVAILABLE|__AVAILABILITY` in `…/MacOSX26.5.sdk/usr/include/sys/fcntl.h`: zero hits). A grep of
Apple's docs for a macOS version would have returned nothing forever. The two artifacts that actually record the answer are Apple's own OSS xnu releases
and the release tags that name them.

Items marked *(verify)* are leads, not facts.

## 1. The answer, in one line each

Two different floors, and conflating them is what kept the number unfound:

- **Kernel (runtime) floor: OS X 10.11 El Capitan, Darwin 15, xnu-3247/xnu-3248.** The `fcntl`
  switch itself and the lockf layer implement cmds 90/91/92 in the first El Capitan source drop and
  are absent from its predecessor. Below that, cmd 90 falls through to `EINVAL`.
- **Header (compile-time/public API) floor: macOS 14 Sonoma, Darwin 23, xnu-10002.** That is the
  first release where 90/91/92/93 moved out of Apple's internal `#ifdef PRIVATE` into
  `#if __DARWIN_C_LEVEL >= __DARWIN_C_FULL`, and simultaneously the first release whose `fcntl(2)`
  man page documents OFD locks at all. Every user-visible, citable statement Apple makes about this
  feature therefore begins at macOS 14, even though the syscall answered it eight years earlier.

Practical consequence for this ticket: **there is no version of macOS that meethook can plausibly
run `record` on where the kernel refuses `F_OFD_SETLK`.** `record` needs ScreenCaptureKit (macOS
12.3+) and the trigger's CoreAudio process-object list (macOS 14.4+, per doc-005), so the shipped
feature floor is years above both numbers above. AC #4 ("if the floor turns out to be newer than
meethook's own stated support floor, spin a ticket") does not fire.

## 2. Evidence for the kernel floor

Apple's OSS xnu distributions repo publishes one tag per released build, so the introduction is a
bisect rather than folklore. Tag list used:
<https://github.com/apple-oss-distributions/xnu/tags> (all 30 major families, 123 → 12377).

| tag | release | `F_OFD_*` in `bsd/sys/fcntl.h` | cmds handled in `bsd/kern/kern_descrip.c` | `F_OFD_LOCK` in `bsd/kern/kern_lockf.c` |
| --- | --- | --- | --- | --- |
| `xnu-2422.115.4` | 10.9.5 / Darwin 13 | absent | absent | n/a |
| `xnu-2782.40.9` | 10.10.5 / Darwin 14 | absent | absent | n/a |
| `xnu-3247.10.11` | 10.11 pre-release seed / Darwin 15 | present, under `#ifdef PRIVATE` | 7 hits (`F_OFD_SETLK`/`SETLKW`/`SETLKWTIMEOUT`/`GETLK`) | 6 hits |
| `xnu-3248.60.10` | 10.11 GM / Darwin 15 | present, `PRIVATE` | yes | yes |
| `xnu-4903.270.47` | 10.14.x | present, `PRIVATE` | yes | yes |
| `xnu-6153.141.1` | 10.15.x | present, `PRIVATE` | yes | yes |
| `xnu-8792.81.2` | 13.x | present, `PRIVATE` | yes | yes |
| `xnu-10002.81.5` | 14.x | **present, public** (`__DARWIN_C_FULL`) | yes | yes |
| `xnu-12377.121.6` | 25.x/26 | present, public | yes | yes |

Fetch these directly (raw URLs, all HTTP 200):
- `https://raw.githubusercontent.com/apple-oss-distributions/xnu/xnu-2782.40.9/bsd/sys/fcntl.h` -
  zero `F_OFD` hits.
- `https://raw.githubusercontent.com/apple-oss-distributions/xnu/xnu-3247.10.11/bsd/sys/fcntl.h` -
  the defines land: `#ifdef PRIVATE` / `F_OFD_SETLK 90 … F_OFD_GETLKPID 94 … F_SETCONFINED 95`, plus
  the internal flag `#define F_OFD_LOCK 0x400 /* Use "OFD" semantics for lock */`.
- `…/xnu-3247.10.11/bsd/kern/kern_descrip.c` and `…/kern_lockf.c` - the implementation lands in the
  same drop. Note the dispatch lives in **`kern_descrip.c`, not `kern_fcntl.c`** (that path does not
  exist in these trees); a plan that re-verifies should not go looking there.

What the fall-through looks like on a pre-El-Capitan kernel (`xnu-2782.40.9`, `kern_descrip.c`,
`kern_fcntl` default arm): unrecognized cmds are treated as ioctl-ish selectors -
`if ((uap->cmd & IOC_VOID) && (uap->cmd & IOC_INOUT)) { error = EINVAL; }` then non-vnodes get
`EBADF`, otherwise the cmd is handed to `VNOP_IOCTL`. For `fcntl(90)` with a `struct flock *` the
result is an error (`EINVAL`, occasionally `ENOTTY` from a filesystem's ioctl fallback), never a
successful lock. So the pre-10.11 answer really is "refused", which is exactly the case
`record_lock.rs:252-304` maps to `LockState::Unknown` on the read side and to a hard error on the
write side.

Kernel caveat worth a comment if the plan quotes it: from the very first drop,
`kern_lockf.c` says *"OFD byte-range locks currently do NOT support deadlock detection"*
(`xnu-3248.60.10/bsd/kern/kern_lockf.c:526`). Same sentence survives to today's `main` and into the
released man page ("No deadlock detection is performed for OFD file locks"). Irrelevant to
single-instance locking of one file, but it is the one substantive difference the docs never lead with.

Version↔name mapping cross-checks (second-party, needed because Apple does not publish a
Darwin↔product table with xnu branch strings):
- macOS 15.0 = Darwin 24.0.0 = `xnu-11215.1.10~2` — <https://en.wikipedia.org/wiki/MacOS_Sequoia>
- macOS 14.0 = Darwin 23.0.0 = `root:xnu-10002.1.13~1` (from a real `uname -a`) —
  <http://www.3rz.de/howto/apple-silicon-macos-unix-kernel-system-versions.txt>, corroborated by
  <https://github.com/calmsacibis995/xnu-history>

## 3. Evidence for the public-header / documentation floor

- `bsd/man/man2/fcntl.2` count of `F_OFD`: `xnu-3248` → **0**, `xnu-4903` → 0, `xnu-6153` → 0,
  `xnu-8792` → 0, **`xnu-10002.81.5` → 16**. First appearance is a full section ("Open file
  description (OFD) locks are locks on the file associated with the open file description used to
  acquire them, and not with the process that created them… Only the last close of the last file
  descriptor in any process still…") plus the `l_pid = -1` rule doc-008 §4 already cites. So the
  man text doc-008 relies on is a macOS 14 artifact.
- `xnu-10002.81.5/bsd/sys/fcntl.h` shows the guard flip precisely: 90/91/92/93 under
  `#if __DARWIN_C_LEVEL >= __DARWIN_C_FULL`, while 94/95/96 (`F_OFD_GETLKPID`, `F_SETCONFINED`,
  `F_GETCONFINED`) *stay* `PRIVATE`. That confirms and sharpens doc-008 §4's "do not reach for
  `F_OFD_GETLKPID`": those were never public, in any release.
- Downstream projects independently date the constants from the same era and were confused by the
  same split:
  - rust-lang/libc [PR #3563](https://github.com/rust-lang/libc/pull/3563) (merged 2024-08-13) adds
    the three consts for apple targets; the author notes *"the constants being 4 years old according
    to the kernel changelog"* and wonders whether CI's macos-13 runners can even exercise them
    (they could - the kernel had answered since 10.11; what was missing was only the header).
    libc 0.2.189 exposes them at `src/unix/bsd/apple/mod.rs:2442-2444`.
  - golang/go [issue #73351](https://github.com/golang/go/issues/73351): same complaint - the
    commands work on macOS but the generated `zerrors_darwin` consts were missing.
  - CPython never gained them at all: they appear in the macOS-side gap list in
    [python/cpython#113092](https://github.com/python/cpython/issues/113092), and
    `fcntl.flock`/`lockf` still has no OFD path
    ([bpo-22367](https://bugs.python.org/issue22367), still open as an enhancement). Python's docs
    mention `F_OFD_*` **only** as "On Linux(>=3.15)"
    (<https://docs.python.org/3/library/fcntl.html>) - a good reminder that the *documentation*
    record, not just ours, reads Linux-only.

Nothing in any of these threads asserts a minimum macOS release; every claim about a version in the
wild is either about Linux/glibc or about the SDK header. That is why the `(verify)` survived.

## 4. How Apple words the refusal - and why it is not a version signal

`xnu-10002.81.5/bsd/man/man2/fcntl.2`, `EINVAL` entry for `F_OFD_GETLK`/`F_OFD_SETLK`/`F_OFD_SETLKW`:
*"the data to which `arg` points is not valid, or `fildes` refers to a file that does not support
locking."* Compare glibc
(<https://www.sourceware.org/glibc/manual/latest/html_node/Open-File-Description-Locks.html>):
*"…doesn't specify valid lock information, **the operating system kernel doesn't support open file
description locks**, or the file associated with filedes doesn't support locks."*

Apple documented the errno without ever acknowledging the "kernel doesn't support it" third case
glibc names. So the man page cannot be used to establish a floor (this is presumably why doc-008's
author read it as load-bearing rather than as an answer). Two consequences the plan can use:

- Mapping `EINVAL` → *unknown* stays correct whatever the floor: on Apple's spelling `EINVAL` means a
  bad `struct flock` or a non-lockable file (NFS/SMB shares, some special files), i.e. genuinely
  "cannot tell", never "definitely free". No change to the errno map (AC #5).
- The hedge users need is about the **filesystem**, not the OS version. Network mounts and
  non-vnode paths are reachable on every supported macOS; a specific macOS release is not.

## 5. Where meethook's real floor sits (for AC #4 and the wording)

Nothing in the repo states a minimum macOS product version, and neither CI nor CD sets one
(`.github/workflows/ci.yml`, `cd.yml` run `macos-26`, target `aarch64-apple-darwin`; no
`MACOSX_VERSION_MIN`/`-mmacosx-version-min` anywhere). The effective floors are de facto, from the
frameworks the code calls:

- ScreenCaptureKit (`SCStream`, dual-track capture) - macOS 12.3+.
- The mic-activity trigger's `kAudioHardwarePropertyProcessObjectList` - **macOS 14.4+**, stated in
  doc-005 as "the only OS version floor this spec states"; `crates/meethook-record/src/activity.rs:17`
  repeats "macOS 14.4+".
- Calendar full-access request path and TCC downgrade behavior - macOS 14+
  (`crates/meethook-record/src/calendar/mod.rs:39`).
- The live-proof test ignores itself below macOS 14.2 for the same kind of reason
  (`crates/meethook/tests/live_record_hand_stop.rs:1077`, task-066.07.02).

So the honest sentence available to README/LINUX.md is not "requires macOS X or the lock silently
lies" but: **on anything old enough to lack OFD locks, `record` cannot start at all, and it is
already refusing for another reason.** The read path (`meethook sessions`) is the one surface where
an unsupporting root is a live possibility, and there the failure mode is silence + once-per-run
hedge, which is a *filesystem* story (network mount) more than an OS story.

## 6. Wording material: precedent for stating a floor you cannot pin to a number

- Linux's man pages are the gold standard for what Apple never wrote and we cannot copy:
  "This lock type is Linux-specific, and available since Linux 3.15"
  (<https://man7.org/linux/man-pages/man2/fcntl_locking.2.html>).
- apenwarr's *Everything you never wanted to know about file locking*
  (<https://apenwarr.ca/log/20101213-everything-you-never-wanted-to-know-about-file-locking>)
  remains the canonical explanation of *why* OFD locks exist (fork/close semantics) and notes, as of
  2015, that other OSes might copy it - useful only as background for a one-clause explanation, not
  a citation.
- FreeRADIUS hit the practical portability question head-on
  ([freeradius-server#5489](https://github.com/FreeRADIUS/freeradius-server/issues/5489)): "Linux
  and apparently OSX support `F_OFD_SETLK`… It's not clear if FreeBSD has them." The "apparently"
  from a locking-experienced maintainer is the best available evidence for *why* the macOS floor was
  unfindable, and a caution against promising any reader can look it up.
- Precedent inside this repo for the shape of the sentence: task-066.07.02's live-proof gate states
  its floor with the consequence attached ("macOS 14.2+ - below that floor the CoreAudio process
  objects the…"), rather than asserting a version alone. Same voice as doc-008 §5's "report the
  measurement".

## 7. Open threads the plan must settle (not answered here)

Settled during TASK-067.05.07 rather than left open: items 1, 4 and 5 were decided by
TASK-067.05.07.01, which amended doc-008 §4 in place so that section is now the text of record; items
2 and 3 belong to TASK-067.05.07.02, which splits the user-facing facts one per surface across
README's data-directory entry, README's `sessions` hedge paragraph and LINUX.md.

1. Does the doc-008 rewrite state two floors (runtime 10.11, documented/public 14) or collapse to
   one? Both are true; collapsing loses the reason the `(verify)` sat open for 8 years.
2. README's `record.lock` recipe (`README.md:116-127`) is `lsof`-based and works wherever `lsof`
   does, so a version sentence belongs at the *data-directory* entry (`README.md:344-346`) and/or
   the `sessions` prose, not inline in the `lsof` block. Which surface owns "what `record` does when
   the lock cannot be taken"?
3. Is a `LINUX.md` line warranted at all? The network-mount/NFS/SMB caveat is the part users can hit
   today; the OS-version part is unreachable given the framework floors. Deciding this is the ticket's
   real judgment call, and AC #3 asks for it beside paragraphs users already read.
4. Whether to record the pre-10.11 fall-through detail (`ioctl` path ⇒ `EINVAL`/`ENOTTY`) as fact
   from source reading, or to mark it *(verify)* because nobody ran a 10.10 box. Source-derived, not
   measured - doc-008's rule is that unverified leads get marked, so state it as "read from
   `xnu-2782.40.9`" rather than "measured".
5. Nobody checked Apple's release *notes* or Security Update PDFs for OFD mentions. Given §1 (public
   man page begins at macOS 14) and that xnu release engineering hides these deltas, a docs search is
   unlikely to beat 10.11 - recorded here so a successor does not spend a round rediscovering it.
   *(verify)* as unattempted.
