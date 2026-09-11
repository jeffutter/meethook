---
id: doc-008
title: >-
  Prior art: WAV header forensics and read-only recorder-liveness probing for
  TASK-067.05.01
type: other
created_date: '2026-09-11 14:21'
updated_date: '2026-09-11 14:45'
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
`cbSize` 22, mask `0x4`), rebuilt with python; re-run with `python3` writing
`b"RIFF" + u32(len(body)+declared-8) + b"WAVE" + b"fmt "…` plus N zero bytes, then `afinfo F.wav`
and `afconvert -f WAVE -d LEF32 F.wav out.wav`:

| file | declared `data` | bytes present | `afinfo` duration | `afconvert` output |
|---|---|---|---|---|
| declared short | 4 s (768 000 B) | 5 s (960 000 B) | **3.999979 s** | 4 s — the extra second is dropped |
| declared zero | 0 | 5 s (960 000 B) | **0.000000 s** | 4 KB file, i.e. nothing |
| declared long | 10 s | 5 s | **5.000000 s** | 5 s (clamped to file size) |
| header only | 0 | 0 | 0 s | — |

Read that middle row against §1: because hound's initial declarations are zero and the first
checkpoint is at 5 s, **a recorder killed inside its first five seconds leaves seconds of real
audio that CoreAudio reports as a zero-second file.** "Your audio survived to within about five
seconds" would then be doing nothing for the user, and "may not play" would be too soft — for that
state the truth is *nothing plays until someone repairs it*, while the bytes are still there. The
forensic states should therefore keep "declared shorter than the file" distinct from "declared
zero", and neither should be folded into "empty".

Cross-reader prior art for the same shapes:

- **libsndfile** clamps a too-long declaration to the file and *prints the measurement*: `data : %D
  (should be %D)`; and for the never-finalized shape it says outright `*** Looks like a WAV file
  which wasn't closed properly. Fixing it.` and takes the real byte count (`src/wav.c`, condition
  `chunk_size == 0 && RIFFsize == 8 && psf->filelength > 44`). Both precedents report an observed
  discrepancy instead of inventing a cause — exactly the tone this ticket family wants.
- **FFmpeg** will *not* read the hidden tail by default; it ships an explicit opt-in instead:
  `ignore_length` — "Ignore the size of the `data` chunk and keep reading until the end of the file
  if set. May be useful to read broken or partial files where the header was not properly updated,
  but will misinterpret files with non-audio chunks after the `data` chunk. Default is disabled"
  (`doc/demuxers.texi`, <https://github.com/FFmpeg/FFmpeg/blob/d998016f/doc/demuxers.texi>). So
  "it will not play" is the default behaviour of the two biggest readers, and recovering it is a
  deliberate act — consistent with decision-001's refusal to reopen finalized files and with the
  parent plan's "name the gap, don't silently close it".
- **hound's own reader trusts the declaration** (`num_samples = data_len / bytes_per_sample`,
  `read.rs:598-616`) and refuses a file whose declared length isn't a whole multiple of sample size
  or channel count (`:607-617`). Since checkpoints and finalization only ever write whole-sample
  sizes, that mostly matters if a `write()` landed partially at a block boundary — a case where the
  strict reader rejects the *whole* file while lenient ones play part of it. Worth one sentence in
  the long-form message: how much of the kept audio plays depends on the player, which is another
  reason not to promise playback.

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
- **Availability.** Values 90/91/92 are exposed in the SDK under `__DARWIN_C_LEVEL >=
  __DARWIN_C_FULL` (`…/MacOSX.sdk/usr/include/sys/fcntl.h:308-314`) and `libc` exports them for
  apple targets (`libc-0.2.189/src/unix/bsd/apple/mod.rs:2442-2444`); Linux uses 36/37/38, gated
  since Linux 3.15 and glibc-exposed. How far back the *runtime* support goes on macOS is
  unresolved *(verify)* — which makes the `EINVAL` path load-bearing: Apple lists `EINVAL` for
  `F_OFD_GETLK` when "the data to which arg points is not valid, or `fildes` refers to a file that
  does not support locking". Map `EINVAL` (and `ENOLCK`, which Linux documents as including "a
  remote locking protocol failed", i.e. the NFS/SMB caveat already in `record_lock.rs:63-70`) to
  *unknown*, never to *free*.
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
  audio **kept but undeclared** (present, invisible to players — quantifiable in bytes and ms), and
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
4. Confirm *(verify)* minimum macOS runtime with working `F_OFD_*`; if it is newer than meethook's
   floor, the `EINVAL → unknown` mapping becomes the whole story on old systems and deserves a line
   in `LINUX.md`-adjacent docs.
5. Measure, for the record (TASK-067.07 territory): kill the real `record` at t = 0.5 s / 4 s / 6 s
   / 20 s and log the four (declared, actual) pairs, to turn "≈5 s" from arithmetic into a measured
   distribution — including the declared-zero case from §3, which the ticket's table does not have.
