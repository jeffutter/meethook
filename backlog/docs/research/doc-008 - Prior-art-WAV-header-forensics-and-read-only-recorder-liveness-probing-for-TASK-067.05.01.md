---
id: doc-008
title: >-
  Prior art: WAV header forensics and read-only recorder-liveness probing for
  TASK-067.05.01
type: other
created_date: '2026-09-11 14:21'
updated_date: '2026-09-22 06:05'
---

Research for TASK-067.05.01 (per-track forensics over an unfinished session directory, one home
for the user-facing phrases, and a read-only "is a recorder live right now" probe). Sources: the
IBM/Microsoft RIFF spec and Microsoft's current RIFF overview, the vendored `hound` 3.5.1 source
(`Cargo.lock` pins 3.5.1; `Cargo.toml` asks for `hound = "3.5"`), Rust std docs, Linux man-pages
and Apple's own `fcntl(2)` plus xnu source, libsndfile and FFmpeg (the two reference WAV readers),
and measurements taken on this machine (macOS 26.6.2, arm64) with Apple's `afinfo`/`afconvert`.
Items marked *(verify)* are leads, not facts.

## 1. What a dead writer leaves, per hound's actual mechanics

All line numbers below are the vendored crate (`~/.cargo/registry/src/*/hound-3.5.1`).

- **The header is born lying.** `WavWriter::new_with_spec_ex` writes `"RIFF"`, then `0` for the
  file size, and a `data` chunk header whose size is likewise a placeholder filled in later
  (`write.rs:283-299`); the sizes are only ever filled by `update_header`
  (`write.rs:488-508`), called from `flush` (`:527-535`), `finalize` (`:540-548`) and `Drop`
  (`:579-590`). So *before the first checkpoint* a track declares **zero** data bytes.
- **`flush` really is a checkpoint**, as `meethook-record/src/track.rs:24-28` claims: it seeks to 4
  and to `data_len_offset`, writes both sizes, flushes, and seeks back. Combined with std's
  documented rule that "seeking always writes out the internal buffer before seeking"
  (<https://doc.rust-lang.org/std/io/struct.BufWriter.html#impl-Seek-for-BufWriter<W>>), the
  ordering is *audio first, then the sizes that describe it*. Consequence worth stating as an
  invariant in the new code: **a killed recorder never declares more than the file contains** —
  "declared longer than real" means truncation-by-copy or a different writer, never our own death.
- **Nothing else buffers.** hound says so in its own doc comment ("`WavWriter` employs *no*
  buffering internally", `write.rs:222-224`); the only userspace buffer above hound is
  `wav::create`'s `BufWriter::new` (`crates/meethook-session/src/wav.rs:154-160`), default
  capacity 8 KiB. Two caveats: std documents that default as *"currently 8 KiB, but may change in
  the future"*, so **never derive a user-visible number from the buffer size** — derive it from
  the file, which is what AC #1 says anyway. And a kill loses only whatever sat in that buffer
  (≈43 ms of 48 kHz mono float32), which is a rounding error next to the ≤5 s checkpoint gap;
  the samples past the last checkpoint are *on disk*, just undeclared.
- **Correct the ticket's table.** Both tracks are recorded mono 32-bit float at the device rate
  (`crates/meethook-record/src/track.rs:299-303`, `channels: 1`), not "48 kHz stereo". At 8 KiB the
  unflushed tail is ~43 ms at 48 kHz and ~128 ms at 16 kHz — not "~250 ms". The bigger correction
  is conceptual: the bytes past the declaration are **not additional to** the ≤5 s since the last
  checkpoint, they are *that interval minus* the <8 KiB still buffered. So "≤5 s hidden, <8 KiB
  gone" is the honest pair of statements, and only the first is knowable from the file.
- **u32 ceiling.** `data_bytes_written` is a `u32` ("because WAVE cannot accommodate more data",
  `write.rs:173-175`) and is incremented with plain `+=` (`:435`). At 192 KB/s (48 kHz mono f32)
  4 GiB arrives in ≈6 h 15 min: overflow-checks off (release) wraps and the declarations become
  small garbage, a debug build panics the writer thread. Precedents that this is a real-world
  shape, not a curiosity: libsndfile "Clamp wav header filelength and datalength"
  (<https://github.com/libsndfile/libsndfile/pull/930>) and FFmpeg trac #11696 "WAV ≥ 4 GiB
  abnormal decoding due to length field overflow"
  (<https://www.mail-archive.com/ffmpeg-trac@avcodec.org/msg71917.html>). Given TASK-066's whole
  subject is recordings that outrun their meeting, the forensic API needs a fourth answer besides
  complete/short: **file much larger than declared**, rather than reading it as "complete".

## 2. Reading the chunks: spec rules and defensive habits

- Word alignment is normative, and the size excludes the pad byte: "The data is always padded to
  the nearest WORD boundary. `chunkSize` gives the size of the valid data in the chunk. It does not
  include the padding, the size of `chunkID`, or the size of `chunkSize`" — Microsoft's current
  RIFF overview (<https://learn.microsoft.com/en-us/windows/win32/xaudio2/resource-interchange-file-format--riff-}).
  This is what `wav::channel_mask_of` already implements (`wav.rs:173-193`); reuse its walk.
- Unknown chunks must be skipped, not fatal: "Programs must expect (and ignore) any unknown chunks
  encountered, as with all RIFF forms" (IBM/Microsoft, *Multimedia Programming Interface and Data
  Specifications 1.0*, Aug 1991 — scanned PDFs at <https://mu.krj.st/wave/riff_1.pdf> and
  <https://www.robotplanet.dk/audio/wav_meta_data/riff_mci.pdf>; the 1991 WAVE form laid out
  chunk-by-chunk, with `LIST`/`INFO` metadata legitimately preceding `fmt `, is reproduced at
  <https://wavref.til.cafe/spec/riff1991/>). A foreign file can therefore put arbitrary megabytes
  ahead of `fmt `/`data`: **bound the window you read** and report "unknown" when the walk runs off
  it, rather than reading the whole track.
- Convert bytes→ms from the file's own `fmt ` fields, as planned: the spec defines
  `wBlockAlign` = `channels × bits_per_sample / 8` and `dwAvgBytesPerSec` as the transfer rate
  (both cited at the wavref page above), so `ms = bytes × 1000 / (dwSamplesPerSec × wBlockAlign)`.
  Prefer block-align × sample rate over `dwAvgBytesPerSec`, which the spec describes as an estimate
  for buffer sizing, and treat a nonsense `fmt ` (zero block align, zero rate) as "unknown", not as
  a divide-by-zero or a huge millisecond count.
- Guard the arithmetic the way the mature readers do. libsndfile bails out of the chunk parser on
  implausible sizes (`if (chunk_size >= 0xffff0000) … Exiting parser`, `if (chunk_size >=
  psf->filelength) … Chunk size > file length. Exiting parser`) and warns rather than errors on an
  odd `data` length (`src/wav.c`, <https://github.com/libsndfile/libsndfile/blob/master/src/wav.c>).
- There *can* be more than one `data` chunk in the wild (the 1991 form allows one inside a `'wavl'`
  LIST too), and a `fact` chunk carries sample counts for compressed payloads where the `data` size
  is not authoritative. Irrelevant to what we write, relevant to what a stranger's file might say:
  take the **first** top-level `data` chunk after `fmt `, and don't try to interpret anything else.

## 3. What readers actually do with a lying header (measured here, macOS 26.6.2)

Synthetic files with hound's exact header shape (48 kHz mono float32, `WAVE_FORMAT_EXTENSIBLE`,
`cbSize` 22, mask `0x4`, 68-byte header), written by python as
`b"RIFF" + u32(len(body)+payload-8) + b"WAVE" + b"fmt " + u32(40) + <extensible body> + b"data" +
u32(declared)` followed by `960 000` zero bytes (5 s of payload on disk in every row but the last).

Provenance for the whole table: measured 2026-09-21 on macOS 26.6.2 arm64. Apple tools are
`/usr/bin/afinfo` and `/usr/bin/afconvert`; ffmpeg is `ffprobe version 7.1.5` / `ffmpeg version
7.1.5` from `nix build --no-link --print-out-paths 'nixpkgs#ffmpeg_7'`; hound is the vendored 3.5.1
this repo pins, driven by a throwaway reader that counts the samples it actually returns. Per file:

```sh
afinfo F.wav
afconvert -f WAVE -d LEF32@48000 F.wav out.wav && afinfo out.wav   # and `wc -c out.wav`
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 F.wav
ffmpeg -hide_banner -i F.wav -y copy.wav                            # read the stderr and the output length
```

| declared `data` | bytes present | `afinfo` | `afconvert` -> LEF32 | `ffprobe` duration | `ffmpeg -i F.wav out.wav` |
|---|---|---|---|---|---|
| 960 000 B (5 s) | 5 s | 4.999958 s (959 992 B, 239 998 pkts) | 964 088 B, 4.999958 s | 5.000000 | 5.000000 s, no warning |
| 768 000 B (4 s) | 5 s | 4.000000 s | 772 096 B, 4.000000 s | 4.000000 | 4.000000 s, no warning |
| **0** | 5 s | **0.000000 s** | **4 096 B, nothing** | **5.000000** | **5.000000 s** |
| **0xFFFFFFFF** | 5 s | **4.999958 s** | 964 088 B, 4.999958 s | **5.000000** | **5.000000 s** |
| 1 920 000 B (10 s) | 5 s | 4.999958 s | 964 088 B, 4.999958 s | 5.000000 | 5.000000 s (clamped) |
| 0 | 0 | 0.000000 s | 4 096 B, nothing | `N/A` | no frames decoded |

The rows where the declaration is 0 or `0xFFFFFFFF` print `[wav @ 0x…] Ignoring maximum wav data
size, file may be invalid` at ffmpeg's default verbosity; the 0, `0xFFFFFFFF` and over-long rows
also print `Estimating duration from bitrate, this may be inaccurate`. `-ignore_length 1` does not
change any of those four, and changes only the short-nonzero row: `ffmpeg -ignore_length 1 -i` on
the 4 s declaration writes a 5.000000 s output where the same command without the option writes
4.000000 s (`ffprobe -ignore_length 1 -count_packets` corroborates: 59 packets versus 47). Note
that `ffprobe -show_entries format=duration` still reports 4.000000 *with* the option set - that
field comes from the header, so verify an `ignore_length` effect on packet count or output length,
not on it.

Two things earlier drafts of this section got wrong, both corrected by the table above.

- **CoreAudio honours the `0xFFFFFFFF` sentinel.** It is not a corruption shape but a deliberate
  third thing an author writes to mean "unknown length", and CoreAudio and libavformat both honour
  it by reading to EOF. Treat it as a value a writer may legitimately emit, not one to avoid.
- **The RIFF size is not ignored by everyone.** Varying the size at offset 4 changed nothing for
  libavformat (every short-RIFF variant still reported 5 s, or 4 s for the 4 s declaration), but it
  caps CoreAudio exactly once the `data` declaration is nonzero: with the RIFF size declaring 500
  000 bytes, `afinfo` reported 2.603875 s, and declaring 100 bytes it reported 0.000229 s - for the
  fully-declared, the 4 s-declared and the sentinel-shaped file alike. A declared-zero file reads as
  zero seconds whatever the RIFF size says. So `wav.rs`'s refusal to consult the RIFF size is a
  choice about which quantity to report (bytes the recorder captured), not a claim that readers
  ignore that field.

Read the declared-zero row against §1: because hound's initial declarations are zero and the first
checkpoint is at 5 s, **a recorder killed inside its first five seconds leaves seconds of real
audio that CoreAudio reports as a zero-second file.** "Your audio survived to within about five
seconds" would then be doing nothing for the user. What that state does *not* do is sit unhearable:
libavformat reads those bytes to EOF unasked and prints the discrepancy while doing it. So the
argument for keeping the states apart is not "nothing plays until somebody repairs it" - it is that
what the user hears depends on which reader they happen to use, and a report that folds
"declared zero" into "empty" loses seconds of real audio for every reader that would have found
them. Keep "declared shorter than the file" distinct from "declared zero", and neither folded into
"empty".

Cross-reader prior art for the same shapes:

- **libsndfile** clamps a too-long declaration to the file and *prints the measurement*: `data : %D
  (should be %D)`; and for the never-finalized shape it says outright `*** Looks like a WAV file
  which wasn't closed properly. Fixing it.` and takes the real byte count (`src/wav.c`, condition
  `chunk_size == 0 && RIFFsize == 8 && psf->filelength > 44`). Both precedents report an observed
  discrepancy instead of inventing a cause — exactly the tone this ticket family wants.
- **libavformat (`ffmpeg`/`ffprobe`) treats 0 and `0xFFFFFFFF` as unknown and reads to EOF with no
  option set**, printing `Ignoring maximum wav data size, file may be invalid` while doing so; it
  honours a short *nonzero* declaration unless told otherwise. `ignore_length` is therefore the
  opt-in for the third case only — a nonzero-but-short declaration — not the general key to the
  hidden tail: "Ignore the size of the `data` chunk and keep reading until the end of the file if
  set. May be useful to read broken or partial files where the header was not properly updated, but
  will misinterpret files with non-audio chunks after the `data` chunk. Default is disabled"
  (`doc/demuxers.texi`, <https://github.com/FFmpeg/FFmpeg/blob/d998016f/doc/demuxers.texi>). An
  earlier draft of this bullet asserted the opposite — that FFmpeg will not read the hidden tail by
  default — and that claim is what licensed the shipped report sentence "players stop at the
  declaration", which is false for a killed recorder's most common early death. See
  <https://github.com/FFmpeg/FFmpeg/blob/master/libavformat/wavdec.c> for the code path.
- **hound's own reader trusts the declaration** (`num_samples = data_len / bytes_per_sample`,
  `read.rs:598-616`) and refuses a file whose declared length isn't a whole multiple of sample size
  or channel count (`:607-617`). Measured against the six shapes above: full → 240 000 samples, 4 s
  declaration → 192 000, declared zero → 0, over-long declaration → 240 000 then `Failed to read
  enough bytes.` at EOF, and the `0xFFFFFFFF` declaration → the file is refused outright (`Ill-formed
  WAVE file: data chunk length is not a multiple of sample size`). Since checkpoints and finalization
  only ever write whole-sample sizes, the refusal mostly matters if a `write()` landed partially at a
  block boundary — a case where the strict reader rejects the *whole* file while lenient ones play
  part of it. Which is the point worth carrying into the phrasing: how much of the kept audio a
  listener reaches depends on the reader, so report the measurement and condition any talk of
  playback on it rather than promising either way.

## 4. The liveness probe: what the two kernels actually promise

Both platforms support the query, and both refuse to name the holder — which confirms
`record_lock.rs`'s existing choice that identity comes from the file's contents and liveness comes
from the kernel.

- **Apple documents it.** `fcntl(2)` lists `F_OFD_GETLK`/`F_OFD_SETLK`/`F_OFD_SETLKW`, describes OFD
  locks as owned by the open file description with unlocking only on the last close, and states for
  `F_OFD_GETLK`: "If a lock that does not support the discovery of lock ownership by process (such
  as an OFD lock …, one created by `flock(2)` … or `open(2)` with `O_SHLOCK`/`O_EXLOCK`) is found,
  `l_pid` is set to -1" (<https://keith.github.io/xcode-man-pages/fcntl.2.html>, mirror of the Xcode
  man page; same text in the xnu man source at
  <https://newosxbook.com/src.php?file=%2Fbsd%2Fman%2Fman2%2Ffcntl.2&tree=xnu>). That is a
  first-party citation for the measurement recorded in `record_lock.rs:36-45` — the module doc can
  cite the man page rather than only a local run.
  Why: in xnu's `kern_lockf.c`, "For `F_OFD_*` locks, `lf_id` is the fileglob. Record an `lf_owner`
  iff this is a confined fd" (<https://github.com/apple/darwin-xnu/blob/master/bsd/kern/kern_lockf.c>)
  — the owner is a file description, not a process. Do **not** reach for `F_OFD_GETLKPID` (94) /
  `F_SETCONFINED` (95): those exist only under `PRIVATE` in xnu's own header and are not public API.
- **Availability.** Closed by TASK-067.05.07 (full evidence in
  `doc-013 - Prior-art-macOS-runtime-floor-for-OFD-locks-for-TASK-067.05.07.md`). There are two
  floors, eight years apart, and conflating them is why the question looked unanswerable.
  - **Apple's own manual dates our dependency to Linux.** `fcntl(2)` gained a HISTORY paragraph at
    macOS 14 reading "Open file description locks first appeared in Linux 3.15"
    (`xnu-10002.81.5/bsd/man/man2/fcntl.2`, `.Sh HISTORY` at :1037, the sentence at :1043; still
    verbatim at `xnu-12377.121.6`, and live on the keith mirror cited above). At `xnu-8792.81.2`
    (macOS 13) that same section is the 4.2BSD line alone. First-party text pointing at Linux for a
    command Apple's kernel had answered since 2015 is the reason nobody found a macOS number: the
    documentation said Linux, so nobody looked for an Apple release.
  - **Kernel/runtime floor: OS X 10.11 El Capitan (Darwin 15).** Bisected on Apple's own OSS tags at
    <https://github.com/apple-oss-distributions/xnu/tags>: `xnu-2782.40.9` (10.10.5) has zero `F_OFD`
    hits in `bsd/sys/fcntl.h` *and* zero in `bsd/kern/kern_descrip.c`; `xnu-3247.10.11` (the 10.11
    seed) defines 90/91/92/93 under `#ifdef PRIVATE` (`bsd/sys/fcntl.h:353-358`) with 14 `F_OFD` hits
    in `bsd/kern/kern_descrip.c` and 7 in `bsd/kern/kern_lockf.c`; `xnu-3248.60.10` (10.11 GM) is the
    same. The dispatch lives in `kern_descrip.c` - there is no `bsd/kern/kern_fcntl.c` in these
    trees, so anyone re-running this bisect should not go looking there.
  - **Public-header and documentation floor: macOS 14 Sonoma (Darwin 23, `xnu-10002`).** There
    `bsd/sys/fcntl.h:386-391` moves 90/91/92/93 out of `PRIVATE` into
    `#if __DARWIN_C_LEVEL >= __DARWIN_C_FULL`, while 94/95/96 (`F_OFD_GETLKPID`, `F_SETCONFINED`,
    `F_GETCONFINED`) stay `PRIVATE` at :394 - never public in any release, which is the firmer basis
    for the "do not reach for `F_OFD_GETLKPID`" above. The count of `F_OFD` in `bsd/man/man2/fcntl.2`
    is 0 / 0 / 0 / 0 / 16 across `xnu-3248` / `4903` / `6153` / `8792` / `10002`, so the man-page text
    this section leans on, `l_pid = -1` included, is itself a macOS 14 artifact. Values 90/91/92 are
    exposed in today's SDK under `__DARWIN_C_FULL` (`…/MacOSX.sdk/usr/include/sys/fcntl.h:309-313`)
    and `libc` exports them for apple targets
    (`libc-0.2.189/src/unix/bsd/apple/mod.rs:2442-2444`); Linux uses 36/37/38, gated since Linux 3.15
    and glibc-exposed.
  - **There is no SDK route to the runtime floor.** `API_AVAILABLE` / `__AVAILABILITY` occurs 0 times
    in `usr/include/sys/fcntl.h` of every SDK on this machine (15, 15.4, 26, 26.5), and Apple's
    `fcntl(2)` carries no per-command availability. This was only ever answerable by reading xnu
    source, which is what made it cost eight years instead of one grep.
  - **What an errno can and cannot mean.** An earlier draft treated Apple's `EINVAL` wording as
    load-bearing for the *version* question, because the runtime floor was then unknown; with the
    floor established, that reading is superseded - `EINVAL` is a *filesystem* signal. Apple documents
    it for `F_OFD_GETLK` when "the data to which arg points is not valid, or `fildes` refers to a file
    that does not support locking" (`xnu-10002.81.5/bsd/man/man2/fcntl.2:892`), and never
    acknowledges the third case glibc names ("the operating system kernel doesn't support open file
    description locks"), so `EINVAL` can never be read as "this macOS is too old". Map `EINVAL` (and
    `ENOLCK`, which Linux documents as including "a remote locking protocol failed", i.e. the NFS/SMB
    caveat already in `record_lock.rs:63-70`) to *unknown*, never to *free* - which is right for the
    reason Apple does state.
  - Read from kernel source rather than run on hardware, so marked per the rule at the top of this
    document: pre-10.11, cmd 90 falls through to the ioctl-ish default arm of `kern_fcntl` and yields
    an error (`EINVAL`, occasionally `ENOTTY` from a filesystem's ioctl fallback), never a lock
    (`xnu-2782.40.9/bsd/kern/kern_descrip.c`); and from its first drop `kern_lockf.c` says "OFD
    byte-range locks currently do NOT support deadlock detection"
    (`xnu-3248.60.10/bsd/kern/kern_lockf.c:525-526`), mirrored in the man page from macOS 14 on
    (`xnu-10002.81.5/bsd/man/man2/fcntl.2:617`). Neither matters to locking one file whole, but the
    second is the substantive difference the documentation never leads with. Nobody ran a 10.10 box,
    and nobody searched Apple's release notes or Security Update PDFs for OFD mentions - that search
    stays unattempted *(verify)*, though given that the public man page begins at macOS 14 it is
    unlikely to beat 10.11.
- **Linux details that shape the call.** From the split man pages
  (<https://man7.org/linux/man-pages/man2/F_OFD_GETLK.2const.html>, same content as
  `fcntl_locking(2)`):
  - The answer to a query is in the struct, not the errno: on success, no conflicting lock leaves
    `l_type` = `F_UNLCK`; a conflicting one fills in `l_type`/`l_whence`/`l_start`/`l_len`, with
    "**If the conflicting lock is an open file description lock, then `l_pid` is set to -1**".
  - "**Note that the returned information may already be out of date by the time the caller
    inspects it**" — the TOCTOU hedge in the wording is documented behaviour, not paranoia.
  - `EAGAIN`/`EACCES` belong to `SETLK` only; a probe must not go looking for them. Real probe
    errors are `EBADF`, `EFAULT`, `EINTR` (Linux lists it even for `*_GETLK`), `EINVAL`, `ENOLCK`.
    Retry once on `EINTR`; everything else is "unknown".
  - Gotcha worth encoding in a comment: `EINVAL` when "`l_pid` was not specified as zero" in the
    request. The existing `try_ofd_write_lock` already zeroes it; a copy-pasted probe must too.
  - OFD locks conflict across *different* open file descriptions **of the same process** ("Conflicting
    lock combinations … where one lock is an open file description lock and the other is a
    traditional record lock conflict even when they are acquired by the same process"; and OFD locks
    "may conflict with each other when they are acquired via different open file descriptions").
    That is precisely what makes AC #3 testable in-process with two `open()`s — and it is why a
    POSIX `F_GETLK` probe would report "free" against our own lock. Keep the probe OFD on both
    platforms for that reason, not only for symmetry with `acquire`.
- **Query needs less privilege than take.** Both manuals phrase the open-mode requirement as being
  about *placing* a lock ("In order to place a write lock, `fd` must be open for writing"; Apple's
  `EBADF` entry names only the `SETLK` variants). So the probe can open `O_RDONLY | O_CLOEXEC`,
  without `O_CREAT`, and never trips the "must not disturb the holder" rule at
  `paths.rs:93-100` / `speakers.rs:617-630`. Absent path ⇒ free, straight from `ENOENT`; do not
  create, do not unlink, do not `acquire()`.
- **Three answers, not two.** Held / Free / Unknown, with *Unknown behaving like Held* at the call
  sites: an unreadable answer must not let a consumer assert "this session was interrupted" about a
  call that is happening now, since asserting exactly that is the defect being fixed. (Precedent for
  the asymmetry inside this repo: `record_lock.rs:130-135` treats any unexpected errno as a loud
  error rather than a silent pass.) The existing root-snapshot tests (`speakers.rs:615-630`,
  `record_lock.rs`'s "only acquiring touches the lock file") are the convention for AC #3's
  "entry set unchanged" assertion.

## 5. Phrasing: the precedent is to report the measurement

Across every tool surveyed, nobody asserts a cause; they report the observed disagreement and what
follows from it — libsndfile's `data : X (should be Y)` and "wasn't closed properly. Fixing it",
FFmpeg's opt-in `ignore_length` documented as being for files "where the header was not properly
updated", and OBS's answer being a container that survives abortion plus a *manual* remux (already
covered in the parent plan's corrections). Three consequences for the phrase library:

- Distinguish three magnitudes that today's single sentence conflates: audio **kept and playable**,
  audio **kept but undeclared** (present, undeclared — quantifiable in bytes and ms, and reachable or
  not depending on the reader, §3), and
  audio **never written** (unknowable, because the clock died with the recorder — say nothing about
  it beyond that).
- Say what a transcript lacks concretely enough to be believed: `session.json` held the one clock
  shared by the two tracks, so *neither track can be placed on a common timeline*, regardless of how
  much of either plays. Each track checkpoints independently (`track.rs` runs one writer thread per
  track), so the two declarations can disagree with each other — reporting both is information, not
  noise.
- While a recorder holds the lock, none of these sentences apply to the newest directory: a
  WAV-with-no-`session.json` is also what a call in progress looks like (`record_lock.rs:51-56`,
  decision-006). Withhold rather than hedge in-place, since the kernel answer is instantaneous but
  stale by inspection time (§4).

## 6. Things to settle in the plan (not answered here)

1. Name the fifth and sixth shapes explicitly: *not a WAV at all* (AC #1's own example), *zero-byte
   file*, *walk ran off the read window* ⇒ unknown, and *file far larger than declared* (u32 wrap,
   §1). A state named `complete_as_declared` should not be reachable when the excess exceeds the
   checkpoint window.
2. Pick the read-window bound and say why in a comment (libsndfile's behaviour and the 1991
   "ignore unknown chunks" rule both argue for a bounded scan with an explicit unknown outcome).
3. Decide whether the probe returns the holder's self-description (reusing `read_holder`) or only a
   boolean; returning it costs nothing and saves `.02`/`.03` each reaching for the path themselves —
   but it risks becoming a second home for holder formatting.
4. Settled by TASK-067.05.07: the minimum macOS runtime with working `F_OFD_*` is OS X 10.11
   (Darwin 15), bisected on Apple's OSS xnu tags, with the public-header and man-page floor at
   macOS 14 - both written up in the "Availability." bullet in §4 and evidenced in doc-013. It is
   *older* than meethook's own de facto floor (ScreenCaptureKit needs macOS 12.3+, the mic-activity
   trigger needs 14.4+, and the release binary's load command declares `minos 14.0`), so nothing that
   can run `record` lacks the lock, and `EINVAL → unknown` is not the story of old systems. It is the
   story of the filesystem: the network-mount case, whose user-facing sentences belong to README's
   `record.lock` entry and to `LINUX.md` (TASK-067.05.07.02).
5. Measure, for the record (TASK-067.07 territory): kill the real `record` at t = 0.5 s / 4 s / 6 s
   / 20 s and log the four (declared, actual) pairs, to turn "≈5 s" from arithmetic into a measured
   distribution — including the declared-zero case from §3, which the ticket's table does not have.
