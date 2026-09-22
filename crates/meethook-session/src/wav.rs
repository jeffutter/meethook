//! The WAV header every meethook track is written with.
//!
//! This module exists for four bytes. hound picks `WAVE_FORMAT_EXTENSIBLE` for any spec with
//! more than 16 bits per sample -- which is every track here, all 32-bit float -- and then
//! fills that format's `dwChannelMask` field with `(1 << channels) - 1`. For one channel that
//! is `0x1`, `SPEAKER_FRONT_LEFT`: an instruction to route the only channel to the left
//! speaker and nothing to the right. Players honour it. It is why a mono recording arrives in
//! one ear.
//!
//! hound has no API to override the field (`write.rs` carries the author's own `TODO: add the
//! option to specify the channel mask`), so it is corrected on the way past: [`ChannelMask`]
//! wraps the writer hound writes into and rewrites those four bytes as the header streams
//! through. Nothing else in the file is touched.
//!
//! It lives in this crate for the same reason the file names do. The session contract is not
//! only *where* the tracks are, it is *what* they are; a header is as much a part of
//! `mic.wav`'s meaning as its path, and one spelling means one place to fix.
//!
//! Nothing inside meethook reads this field -- hound's own reader parses `dwChannelMask` and
//! discards it -- so the AEC, resampler, diarizer, and ASR cannot observe the change. The
//! blast radius is a human with headphones on, which is the one moment they meet these files.
//!
//! The second half of this module is the mirror image of the first: not writing a header,
//! reading one that somebody else -- or a killed process -- left behind. [`track`] answers what
//! a single track file proves about itself, which is how an unfinished session directory gets
//! described in measured terms rather than in guesses about what happened to the recorder. It
//! lives here because this crate owns the byte layout of these files, and nowhere else.

use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use hound::{WavSpec, WavWriter};

use crate::SessionPaths;

/// `SPEAKER_FRONT_CENTER`: the mask a single-channel stream should carry.
///
/// Empirically the right value, not merely the spec-legal one. Patching the field of a
/// hound-written mono float WAV and asking CoreAudio what it sees:
///
/// | `dwChannelMask` | `afinfo` reports        |
/// |-----------------|-------------------------|
/// | `0x1` (hound)   | `Channel layout: Left`  |
/// | `0x3`           | `Channel layout: Left`  |
/// | `0x0`           | no channel layout at all|
/// | `0x4`           | `Channel layout: Mono`  |
///
/// `0x0` ("unspecified, player decides") also stops the mis-routing, but it says nothing where
/// `0x4` says the true thing. An output with no centre speaker downmixes front-centre to both,
/// which is the desired result; if some player is ever found to do worse, `0x0` is the fallback
/// and it is this one constant.
pub const MONO_CHANNEL_MASK: u32 = 0x4;

/// `wFormatTag` for `WAVE_FORMAT_EXTENSIBLE`, the only fmt kind that has a channel mask.
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// Byte offset of `wFormatTag` in hound's header: `RIFF`+size+`WAVE`+`fmt `+cksize.
const FORMAT_TAG_AT: usize = 20;

/// Byte offset of `dwChannelMask` in hound's `WAVE_FORMAT_EXTENSIBLE` header.
///
/// Fixed, because hound emits the whole 68-byte header in one write at position 0 and its
/// later header updates (`flush`, `finalize`) rewrite only the two size fields at `4` and
/// `64`. The fmt chunk is written once and never revisited, so a mask corrected here stays
/// corrected through every checkpoint -- including in a file a killed process never finalized.
const CHANNEL_MASK_AT: usize = 40;

/// The mask to write for `channels`.
///
/// Agree with hound except where it is wrong. Its `(1 << channels) - 1` is genuinely correct
/// for two channels (`0x3` really is front-left plus front-right) and mono is the only count it
/// gets backwards, so overriding just that leaves a future stereo track needing no second
/// decision.
fn channel_mask(channels: u16) -> u32 {
    if channels == 1 {
        MONO_CHANNEL_MASK
    } else {
        // hound's default, including its clamp to the 18 non-reserved bits.
        (1u32 << channels.min(18)) - 1
    }
}

/// A `Write + Seek` shim that corrects `dwChannelMask` in the header passing through it.
///
/// Transparent in every other respect: it counts bytes so it can recognise the header, and
/// forwards everything else verbatim.
pub struct ChannelMask<W> {
    inner: W,
    /// Absolute position in `inner`, tracked so the patch can be confined to offset 0.
    pos: u64,
    mask: u32,
}

impl<W> ChannelMask<W> {
    /// Wraps `inner`, correcting the channel mask of a `WAVE_FORMAT_EXTENSIBLE` header written
    /// at offset 0 to `mask`.
    pub fn new(inner: W, mask: u32) -> ChannelMask<W> {
        ChannelMask {
            inner,
            pos: 0,
            mask,
        }
    }

    /// Whether `buf` is unambiguously hound's initial header.
    ///
    /// Three conditions, and all three matter. At any position but 0 the bytes are audio or a
    /// size-field update. Shorter than 44 bytes there is no mask field to correct. And without
    /// `WAVE_FORMAT_EXTENSIBLE` at offset 20 the fmt chunk is a 16-byte `PCMWAVEFORMAT`, where
    /// offset 40 is *sample data*. The tag check is what makes this safe rather than merely
    /// correct: a hound that changed its mind about which format to emit would make this shim
    /// stop patching, not start corrupting.
    fn is_header(&self, buf: &[u8]) -> bool {
        self.pos == 0
            && buf.len() >= CHANNEL_MASK_AT + 4
            && buf[FORMAT_TAG_AT..FORMAT_TAG_AT + 2] == WAVE_FORMAT_EXTENSIBLE.to_le_bytes()
    }
}

impl<W: Write> Write for ChannelMask<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.is_header(buf) {
            let mut header = buf.to_vec();
            header[CHANNEL_MASK_AT..CHANNEL_MASK_AT + 4].copy_from_slice(&self.mask.to_le_bytes());
            // `write_all`, not `write`: a short write that stopped before offset 44 would send
            // the mask bytes back through as a second, unpatched write at a non-zero position.
            // Failing partway here fails `WavWriter::new` outright, so there is no half-written
            // header for a caller to keep using.
            self.inner.write_all(&header)?;
            self.pos += header.len() as u64;
            return Ok(header.len());
        }

        let written = self.inner.write(buf)?;
        self.pos += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

impl<W: Seek> Seek for ChannelMask<W> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        // The inner writer's answer is the truth, not our arithmetic on it.
        self.pos = self.inner.seek(pos)?;
        Ok(self.pos)
    }
}

/// [`hound::WavWriter::create`], with a channel mask a player will not read as front-left.
///
/// Buffered like hound's own `create`, and for the same reason: hound writes each sample
/// straight through, and an hour of audio is tens of millions of them. The shim sits *outside*
/// the [`BufWriter`] so it sees hound's writes at hound's own offsets rather than whatever
/// chunking a buffer chooses to produce.
///
/// The error type is hound's because the caller's failures are its own -- every call site
/// already maps [`hound::Error`] into its own error, so this is a drop-in replacement.
pub fn create(
    path: &Path,
    spec: WavSpec,
) -> hound::Result<WavWriter<ChannelMask<BufWriter<File>>>> {
    let file = File::create(path)?;
    new(BufWriter::new(file), spec)
}

/// [`hound::WavWriter::new`], with a channel mask a player will not read as front-left.
///
/// For the caller that already owns its sink -- a track written into a temp file by
/// [`crate::write_atomic_with`], say. `writer` must be at offset 0, exactly as hound requires,
/// and should be buffered by the caller.
pub fn new<W: Write + Seek>(writer: W, spec: WavSpec) -> hound::Result<WavWriter<ChannelMask<W>>> {
    WavWriter::new(ChannelMask::new(writer, channel_mask(spec.channels)), spec)
}

/// The `dwChannelMask` of a WAV already in memory, or `None` if it has no such field.
///
/// `None` covers "not a RIFF/WAVE file", "no `fmt ` chunk", and -- the common case -- a fmt
/// chunk that is a plain `PCMWAVEFORMAT`, which stops before the mask. So `None` means "this
/// file says nothing about speaker placement", never "look at offset 40 anyway".
///
/// Bytes rather than a path so it needs no error type and no I/O, and it *walks* the chunk list
/// rather than assuming `fmt ` comes first: a `LIST` chunk legitimately precedes it in files
/// other tools write.
pub fn channel_mask_of(wav: &[u8]) -> Option<u32> {
    if wav.len() < 12 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return None;
    }

    for chunk in Chunks::new(&wav[12..]) {
        if chunk.id != *b"fmt " {
            continue;
        }
        // wFormatTag(2) + WAVEFORMAT(14) + wBitsPerSample(2) + cbSize(2)
        // + wValidBitsPerSample(2) puts dwChannelMask 20 bytes into the chunk.
        // A `fmt ` whose body is clipped in hand answers "nothing", like any other fmt that is
        // too short to carry the field; it does not send the walk looking for a second one.
        let fmt = chunk.body;
        if fmt.len() < 24
            || u16::from_le_bytes(fmt[0..2].try_into().ok()?) != WAVE_FORMAT_EXTENSIBLE
        {
            return None;
        }
        return Some(u32::from_le_bytes(fmt[20..24].try_into().ok()?));
    }

    None
}

/// One top-level chunk: what it is, what it says it holds, and what is actually in hand.
struct Chunk<'a> {
    id: [u8; 4],
    /// The `ckSize` field, which a stranger's file is free to get wrong. `body` is the truth
    /// about what this reader was given; `declared` is the truth about what the file claims.
    /// Keeping both is the whole point of the forensic walk below.
    declared: u32,
    body: &'a [u8],
}

/// The top-level chunks of a RIFF list, walked rather than indexed.
///
/// Two rules from the 1991 RIFF form, both already honoured by the version of
/// [`channel_mask_of`] this iterator replaces:
///
/// - Unknown chunks are skipped, never fatal. `LIST`/`INFO` metadata legitimately precedes
///   `fmt `, so assuming fixed offsets works on our own files and fails on other people's.
/// - Chunk bodies are word-aligned: an odd `ckSize` carries one pad byte that the size itself
///   excludes, so advancing by `ckSize` alone desynchronises the walk.
///
/// Iteration ends -- rather than erroring -- when there is no room for another chunk header. A
/// chunk whose body is clipped by the buffer is yielded once, holding whatever bytes are in
/// hand alongside what it *claims*: a file that stops halfway through a chunk still said what it
/// meant, and the caller decides what that proves. Nothing after such a chunk is ever walked,
/// because its position is exactly what the unread bytes would have had to locate.
struct Chunks<'a> {
    rest: &'a [u8],
}

impl<'a> Chunks<'a> {
    /// The chunks listed in `payload`: everything after the 12-byte `RIFF` + size + form-type
    /// prologue, which is the caller's job to have checked.
    fn new(payload: &'a [u8]) -> Chunks<'a> {
        Chunks { rest: payload }
    }
}

impl<'a> Iterator for Chunks<'a> {
    type Item = Chunk<'a>;

    fn next(&mut self) -> Option<Chunk<'a>> {
        if self.rest.len() < 8 {
            return None;
        }
        let id = self.rest[0..4].try_into().ok()?;
        let declared = u32::from_le_bytes(self.rest[4..8].try_into().ok()?);
        let body = &self.rest[8..];

        // Saturating because `declared` is a u32 read out of a file: on the 64-bit targets this
        // crate builds for the sum cannot actually overflow, but arithmetic fed by somebody
        // else's header should not be able to try.
        let step = 8u64
            .saturating_add(u64::from(declared))
            .saturating_add(u64::from(declared & 1));
        let index = usize::try_from(step).unwrap_or(usize::MAX);

        // A chunk whose body runs past what this reader was handed is still *returned*, with the
        // bytes that are in hand as its body, because the thing it declares is itself the
        // evidence: a track truncated by a copy still carries the `data` length recorded before
        // the copy, and dropping the chunk would turn "short by 1.6 kB" into "I have no idea".
        // Advancing stops the walk there -- nothing behind an unread chunk can be located.
        let Some(after) = self.rest.get(index..) else {
            self.rest = &self.rest[self.rest.len()..];
            return Some(Chunk { id, declared, body });
        };
        self.rest = after;

        Some(Chunk { id, declared, body })
    }
}

/// A quantity of audio in one track, counted in bytes and in the milliseconds those bytes are
/// worth at the file's own rate.
///
/// Deliberately not named after disagreement: which audio a span points at -- the part that is
/// missing, the part no header declared, or the part the file simply holds -- is the business of
/// the [`TrackEvidence`] variant that carries it, not of the number itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackSpan {
    pub bytes: u64,
    pub millis: u64,
}

/// What one track file proves about itself.
///
/// Every answer is a state, never an error: the question is asked of a file that may be
/// anything -- absent, zero bytes, a JPEG, a header whose chunks claim gigabytes -- and a
/// report that prints nothing because one file was unreadable tells the user less than one
/// odd row does. An orphaned directory is a normal classification
/// ([`crate::Classification`]), so the thing that reads it has to be able to say so calmly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackEvidence {
    /// No file at all. A track that never started is a different fact from a track that
    /// stopped, and saying "both tracks" when only one was ever created is the same class of
    /// untruth as saying "crashed": a dead input device leaves exactly one track on disk.
    Absent,
    /// Too short to hold the `RIFF`/`WAVE` prologue, or carrying some other magic. A zero-byte
    /// file lands here rather than in a variant of its own: hound writes its header the instant
    /// the writer is created, so "created and then nothing" describes no event anyone will see.
    NotAWav,
    /// The file is there and could not be opened or read (permissions, EIO, a directory).
    Unreadable,
    /// A WAV-shaped beginning, but the walk ran off the read window, or `fmt ` is unusable or
    /// missing, or there is no `data ` chunk to weigh. Deliberately not folded into any of the
    /// measurements: guessing a number is worse than reporting none.
    Unknown,
    /// Valid header declaring zero bytes and holding zero bytes: a header that declares nothing,
    /// with nothing written onto it. Hound's initial header declares zero, and finalizing a writer
    /// that never received a sample produces the identical bytes, so no event is named here.
    HeaderOnly,
    /// Real bytes and declared bytes agree exactly, and more than zero. Note this says the file
    /// agrees with *itself*, which is all a header can be asked; whether it agrees with the
    /// meeting is a different question, and the one `session.json` used to answer.
    CompleteAsDeclared,
    /// The header counts more than the file holds: audio that was declared and is now gone.
    /// Never our own death -- see [`track`]'s note on hound's write order -- so this is
    /// truncation by copy, by a truncate, or by a writer that is not us.
    ShortBy(TrackSpan),
    /// The file holds more than the header counts: audio on disk past what the header declares.
    /// What the bytes prove is about the writer, not about anybody's player -- the header stopped
    /// being updated while bytes kept landing, and neither another `flush` after the last one nor
    /// `finalize`/`Drop` ever ran.
    ///
    /// How much of it a listener reaches depends on which reader they use, and the readers were
    /// measured rather than assumed (doc-008 §3): CoreAudio stops at a short *nonzero*
    /// declaration, calls a declared-zero file zero seconds long, and honours the `0xFFFFFFFF`
    /// "unknown length" sentinel by reading to EOF; libavformat treats 0 and `0xFFFFFFFF` as
    /// unknown and reads to EOF unaided, honouring a short nonzero declaration unless told
    /// `-ignore_length`; hound's reader divides the declared length by the sample size and trusts
    /// whatever number it finds. So this hands over bytes and milliseconds instead of a promise
    /// about playback, which is also what the report prints.
    BeyondDeclaration(TrackSpan),
}

/// What an unfinished session directory proves, one entry per track.
///
/// Public and comparable so a report can assert an exact expectation instead of fabricating a
/// file to produce it, and per-track rather than blended because the two tracks are written by
/// independent threads that checkpoint independently: their declarations are allowed to
/// disagree with each other, and reporting both is information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unfinished {
    pub mic: TrackEvidence,
    pub speaker: TrackEvidence,
}

/// How many leading bytes [`track`] reads, and the only bound on how deep it looks.
///
/// Our own header is 68 bytes -- `RIFF` + size + `WAVE`, a 40-byte extensible `fmt `, then
/// `data` -- so anything measured comes from well inside the first kilobyte. The window is wide
/// because of the 1991 form's permission for arbitrary `LIST`/`JUNK` metadata to precede
/// `fmt `: a foreign prologue is real, but it is not 64 KiB deep. Reading the whole track to
/// find out what its header says is the thing this function exists to avoid -- a track is
/// hundreds of megabytes, and the answer is always in the first 68 bytes of ours.
const HEADER_WINDOW: u64 = 64 * 1024;

/// What one track file proves about itself.
///
/// The verdict is the `data` chunk's declared length against the bytes that follow it, and
/// nothing else. The `RIFF` size is deliberately not consulted: it lies in the same breath as
/// `data` does, and the audio is what `data` counts, so weighing both would be averaging two
/// claims rather than checking one against reality. (No reader is obliged to agree with that
/// choice: a short `RIFF` size caps what CoreAudio reads once `data` declares something, though
/// libavformat ignores it -- doc-008 §3.)
///
/// The direction of disagreement means something, because hound's mechanics make one direction
/// impossible for us: its header is *born* declaring zero `data` bytes, the two size fields are
/// filled in only by `flush`, `finalize` and `Drop`, and `flush` seeks -- which std documents as
/// draining the buffer first. Audio therefore always reaches the disk before the sizes that
/// describe it. **A killed recorder never declares more than the file contains**, so
/// [`TrackEvidence::ShortBy`] is somebody else's doing, and [`TrackEvidence::BeyondDeclaration`]
/// is ours.
///
/// Milliseconds come from this file's own `fmt ` chunk -- sample rate times block align --
/// because those numbers exist in the header and nowhere else. Nothing here consults
/// `session.json`, another track, or a constant from the recorder: a fact about a file is read
/// out of the file.
pub fn track(path: &Path) -> TrackEvidence {
    let mut file = match File::open(path) {
        Ok(file) => file,
        // ENOENT is the one open failure that is a fact about the session rather than about the
        // filesystem, so it gets its own answer.
        Err(e) if e.kind() == io::ErrorKind::NotFound => return TrackEvidence::Absent,
        Err(_) => return TrackEvidence::Unreadable,
    };

    let file_len = match file.metadata() {
        Ok(metadata) => metadata.len(),
        Err(_) => return TrackEvidence::Unreadable,
    };

    // Sparse files make this cheaper than it looks: `take` bounds the read, not the file, and a
    // 4 GiB track whose audio was never written costs one window to classify.
    let mut window = Vec::new();
    // Qualified because `File` is both `Read` and `Write` and `by_ref` would be ambiguous.
    if std::io::Read::by_ref(&mut file)
        .take(file_len.min(HEADER_WINDOW))
        .read_to_end(&mut window)
        .is_err()
    {
        return TrackEvidence::Unreadable;
    }

    evidence(&window, file_len)
}

/// What the two tracks of an unfinished session prove.
///
/// Read-only, infallible, and safe against a directory holding one track, no tracks, or files
/// that are not WAVs at all: each side answers for itself.
pub fn unfinished(session: &SessionPaths) -> Unfinished {
    Unfinished {
        mic: track(&session.mic_wav()),
        speaker: track(&session.speaker_wav()),
    }
}

/// The `fmt ` fields needed to turn bytes back into time.
#[derive(Debug, Clone, Copy)]
struct Fmt {
    sample_rate: u32,
    block_align: u16,
}

impl Fmt {
    /// Parses the fields out of a `fmt ` chunk body, refusing a chunk that cannot support a
    /// conversion.
    ///
    /// A zero rate or zero block align -- or a body too short to carry either -- is refused
    /// rather than trusted, because the alternative is a divide-by-zero or a millisecond count
    /// that reads like a measurement. `None` becomes [`TrackEvidence::Unknown`], which is what
    /// an unusable header actually says.
    fn parse(body: &[u8]) -> Option<Fmt> {
        if body.len() < 16 {
            return None;
        }
        let fmt = Fmt {
            sample_rate: u32::from_le_bytes(body[4..8].try_into().ok()?),
            block_align: u16::from_le_bytes(body[12..14].try_into().ok()?),
        };
        (fmt.sample_rate != 0 && fmt.block_align != 0).then_some(fmt)
    }

    /// Duration of `bytes` of audio at this file's own rate, saturating rather than wrapping.
    ///
    /// Block align times sample rate, not `dwAvgBytesPerSec`: the spec calls the latter an
    /// estimate for buffer sizing, and this number goes in front of a user.
    fn millis(&self, bytes: u64) -> u64 {
        let bytes_per_second = u128::from(self.sample_rate) * u128::from(self.block_align);
        let millis = u128::from(bytes)
            .saturating_mul(1_000)
            .checked_div(bytes_per_second)
            .unwrap_or(u128::MAX);
        u64::try_from(millis).unwrap_or(u64::MAX)
    }
}

/// Weighs a track's declaration against the bytes, given only the header window and the length.
///
/// Split out from [`track`] so the states are testable from hand-assembled bytes without a
/// filesystem, and so the sparse-file case can be produced by `set_len` rather than by writing
/// four gigabytes.
fn evidence(window: &[u8], file_len: u64) -> TrackEvidence {
    if window.len() < 12 || &window[0..4] != b"RIFF" || &window[8..12] != b"WAVE" {
        return TrackEvidence::NotAWav;
    }

    let mut fmt: Option<Fmt> = None;
    // Whether a `fmt ` has been consumed, separately from whether it parsed: the first one wins
    // even when it is unusable, so a second `fmt ` behind it cannot quietly supply better numbers
    // for the same audio.
    let mut saw_fmt = false;
    // The first top-level `data` chunk: its declaration, and the absolute offset of its first
    // audio byte. Further `data` chunks, `fact` chunks and `'wavl'` lists are left alone -- the
    // 1991 form allows several, and interpreting more than one audio chunk per file would be
    // inventing a format nobody recorded.
    let mut data: Option<(u64, u64)> = None;
    let mut offset = 12u64;

    for chunk in Chunks::new(&window[12..]) {
        let declared = u64::from(chunk.declared);
        match &chunk.id {
            b"fmt " if !saw_fmt => {
                saw_fmt = true;
                fmt = Fmt::parse(chunk.body);
            }
            // The first one only; further `data` chunks fall through to the skip arm below.
            b"data" if data.is_none() => {
                data = Some((declared, offset + 8));
            }
            _ => {}
        }
        offset = offset
            .saturating_add(8)
            .saturating_add(declared)
            .saturating_add(declared & 1);
    }

    // No `data` means the walk never found audio to weigh. A walk that met no `fmt ` at all may
    // simply have stopped before reaching it. Both are `Unknown`. A `fmt ` that was seen but
    // refused is likewise `Unknown`: reporting a byte count with no way to say what those bytes
    // are worth is a number no user should be shown.
    let (Some(fmt), Some((declared, audio_at))) = (fmt, data) else {
        return TrackEvidence::Unknown;
    };

    // Everything from the chunk header to EOF is audio the recorder captured, which is the
    // quantity a user cares about even when a trailer chunk sits after it. Whether a given reader
    // reaches all of it is a question about that reader, not about these bytes (doc-008 §3).
    let held = file_len.saturating_sub(audio_at);
    let gap = |bytes: u64| TrackSpan {
        bytes,
        millis: fmt.millis(bytes),
    };

    if declared == 0 && held == 0 {
        TrackEvidence::HeaderOnly
    } else if declared == held {
        TrackEvidence::CompleteAsDeclared
    } else if declared > held {
        TrackEvidence::ShortBy(gap(declared - held))
    } else {
        TrackEvidence::BeyondDeclaration(gap(held - declared))
    }
}

/// The header, asserted by layout rather than only by outcome.
///
/// A hound upgrade that moved `dwChannelMask` would make the shim quietly stop patching, so
/// these tests pin the offsets around it -- the format tag, `cbSize`, the SubFormat GUID, and
/// where `data` starts -- and fail loudly instead.
#[cfg(test)]
mod tests {
    use std::os::unix::fs::PermissionsExt;

    use hound::SampleFormat;

    use super::*;

    /// hound's `KSDATAFORMAT_SUBTYPE_IEEE_FLOAT`, which lives immediately after the mask and
    /// is therefore what a mis-aimed patch would destroy.
    const SUBTYPE_IEEE_FLOAT: [u8; 16] = [
        0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b,
        0x71,
    ];

    fn mono_float(sample_rate: u32) -> WavSpec {
        WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        }
    }

    fn u16_at(wav: &[u8], at: usize) -> u16 {
        u16::from_le_bytes(wav[at..at + 2].try_into().unwrap())
    }

    fn u32_at(wav: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(wav[at..at + 4].try_into().unwrap())
    }

    #[test]
    fn a_mono_float_track_is_tagged_front_center_and_nothing_else_moved() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");

        let mut writer = create(&path, mono_float(48_000)).unwrap();
        for sample in [0.0f32, 0.25, -0.5, 1.0] {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();

        let wav = std::fs::read(&path).unwrap();
        assert_eq!(u16_at(&wav, 20), WAVE_FORMAT_EXTENSIBLE, "wFormatTag");
        assert_eq!(u16_at(&wav, 36), 22, "cbSize");
        assert_eq!(u32_at(&wav, 40), MONO_CHANNEL_MASK, "dwChannelMask");
        assert_eq!(&wav[44..60], &SUBTYPE_IEEE_FLOAT, "SubFormat GUID");
        assert_eq!(&wav[60..64], b"data");
        assert_eq!(u32_at(&wav, 4) as usize, wav.len() - 8, "RIFF size");
        assert_eq!(u32_at(&wav, 64) as usize, 4 * 4, "data size");

        assert_eq!(channel_mask_of(&wav), Some(MONO_CHANNEL_MASK));
    }

    #[test]
    fn samples_read_back_through_hound_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");
        let samples: Vec<f32> = (0..512).map(|i| (i as f32 / 512.0) - 0.5).collect();

        let mut writer = create(&path, mono_float(16_000)).unwrap();
        for sample in &samples {
            writer.write_sample(*sample).unwrap();
        }
        writer.finalize().unwrap();

        let mut reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.spec(), mono_float(16_000));
        let read: Vec<f32> = reader.samples::<f32>().map(|s| s.unwrap()).collect();
        assert_eq!(read, samples);
    }

    /// The crash path. hound's `flush` rewrites the two size fields and seeks back, so this is
    /// what proves the correction survives a checkpoint -- i.e. that a recording killed
    /// mid-session is centred too, not just a cleanly finalized one.
    #[test]
    fn a_checkpointed_header_keeps_the_corrected_mask() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpointed.wav");

        let mut writer = create(&path, mono_float(16_000)).unwrap();
        for _ in 0..64 {
            writer.write_sample(0.5f32).unwrap();
        }
        writer.flush().unwrap();

        let checkpointed = std::fs::read(&path).unwrap();
        assert_eq!(channel_mask_of(&checkpointed), Some(MONO_CHANNEL_MASK));
        assert_eq!(u32_at(&checkpointed, 64) as usize, 64 * 4, "data size");

        for _ in 0..64 {
            writer.write_sample(-0.5f32).unwrap();
        }
        writer.finalize().unwrap();

        let wav = std::fs::read(&path).unwrap();
        assert_eq!(u32_at(&wav, 40), MONO_CHANNEL_MASK, "dwChannelMask");
        assert_eq!(&wav[44..60], &SUBTYPE_IEEE_FLOAT, "SubFormat GUID");
        assert_eq!(u32_at(&wav, 4) as usize, wav.len() - 8, "RIFF size");
        assert_eq!(u32_at(&wav, 64) as usize, 128 * 4, "data size");
    }

    /// A 16-bit mono spec makes hound write `PCMWAVEFORMAT`, which has no mask at all. The
    /// shim must leave it alone -- offset 40 there is audio -- and the reader must say `None`
    /// rather than reporting whatever sample happens to sit at that offset.
    #[test]
    fn a_16_bit_file_has_no_mask_to_report_and_none_is_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pcm.wav");
        let spec = WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let mut writer = create(&path, spec).unwrap();
        for i in 0..64i16 {
            writer.write_sample(i * 300).unwrap();
        }
        writer.finalize().unwrap();

        let wav = std::fs::read(&path).unwrap();
        assert_eq!(u16_at(&wav, 20), 1, "wFormatTag should be WAVE_FORMAT_PCM");
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(channel_mask_of(&wav), None);

        let mut reader = hound::WavReader::open(&path).unwrap();
        let read: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(read, (0..64i16).map(|i| i * 300).collect::<Vec<_>>());
    }

    /// Stereo keeps hound's `0x3`: front-left plus front-right is what two channels are.
    #[test]
    fn a_stereo_track_keeps_hounds_own_mask() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stereo.wav");
        let spec = WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };

        let mut writer = create(&path, spec).unwrap();
        for sample in [0.1f32, -0.1, 0.2, -0.2] {
            writer.write_sample(sample).unwrap();
        }
        writer.finalize().unwrap();

        assert_eq!(channel_mask_of(&std::fs::read(&path).unwrap()), Some(0x3));
    }

    #[test]
    fn hounds_default_mask_is_reproduced_for_every_count_it_gets_right() {
        assert_eq!(channel_mask(0), 0);
        assert_eq!(channel_mask(2), 0x3);
        assert_eq!(channel_mask(3), 0x7);
        assert_eq!(channel_mask(8), 0xFF);
        assert_eq!(channel_mask(18), 0x3_FFFF);
        assert_eq!(channel_mask(64), 0x3_FFFF, "clamped to non-reserved bits");
        // The one it gets wrong.
        assert_eq!(channel_mask(1), MONO_CHANNEL_MASK);
    }

    /// `fmt ` is not always the first chunk. Foreign files put `LIST` metadata ahead of it, so
    /// the reader has to walk rather than index -- and it has to respect the pad byte an
    /// odd-sized chunk carries.
    #[test]
    fn the_mask_is_found_behind_a_preceding_odd_sized_chunk() {
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&0u32.to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        // A 5-byte LIST chunk: odd, so it is followed by one pad byte.
        wav.extend_from_slice(b"LIST");
        wav.extend_from_slice(&5u32.to_le_bytes());
        wav.extend_from_slice(b"INFO\0");
        wav.push(0);

        let mut fmt = Vec::new();
        fmt.extend_from_slice(&WAVE_FORMAT_EXTENSIBLE.to_le_bytes());
        fmt.extend_from_slice(&1u16.to_le_bytes()); // nChannels
        fmt.extend_from_slice(&16_000u32.to_le_bytes()); // nSamplesPerSec
        fmt.extend_from_slice(&64_000u32.to_le_bytes()); // nAvgBytesPerSec
        fmt.extend_from_slice(&4u16.to_le_bytes()); // nBlockAlign
        fmt.extend_from_slice(&32u16.to_le_bytes()); // wBitsPerSample
        fmt.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        fmt.extend_from_slice(&32u16.to_le_bytes()); // wValidBitsPerSample
        fmt.extend_from_slice(&MONO_CHANNEL_MASK.to_le_bytes());
        fmt.extend_from_slice(&SUBTYPE_IEEE_FLOAT);

        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        wav.extend_from_slice(&fmt);

        assert_eq!(channel_mask_of(&wav), Some(MONO_CHANNEL_MASK));
    }

    #[test]
    fn a_file_that_is_not_a_wav_reports_no_mask_rather_than_a_guess() {
        assert_eq!(channel_mask_of(b""), None);
        assert_eq!(channel_mask_of(b"not a riff file at all"), None);
        // RIFF, but no fmt chunk.
        let mut headerless = Vec::from(*b"RIFF\0\0\0\0WAVE");
        headerless.extend_from_slice(b"data");
        headerless.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(channel_mask_of(&headerless), None);
    }

    // ---------------------------------------------------------------------------
    // Forensics: what a track file proves about itself.
    // ---------------------------------------------------------------------------

    /// Assembles one chunk, including the pad byte an odd-sized body carries.
    fn chunk_bytes(id: &[u8; 4], declared: u32, body: &[u8]) -> Vec<u8> {
        let mut chunk = Vec::new();
        chunk.extend_from_slice(id);
        chunk.extend_from_slice(&declared.to_le_bytes());
        chunk.extend_from_slice(body);
        if body.len() % 2 == 1 {
            chunk.push(0);
        }
        chunk
    }

    /// A `RIFF`/`WAVE` buffer holding `chunks` in order.
    ///
    /// The `RIFF` size is written as zero and left wrong: the forensic verdict deliberately
    /// ignores that field, and every test here is stronger for staying right even when it is a
    /// lie.
    fn riff_with(chunks: &[Vec<u8>]) -> Vec<u8> {
        let mut wav = Vec::from(*b"RIFF\0\0\0\0WAVE");
        for chunk in chunks {
            wav.extend_from_slice(chunk);
        }
        wav
    }

    /// A 16-byte `PCMWAVEFORMAT` body -- enough for the two fields the conversion needs.
    fn fmt_body(sample_rate: u32, block_align: u16) -> Vec<u8> {
        let mut body = Vec::new();
        body.extend_from_slice(&1u16.to_le_bytes()); // wFormatTag: PCM
        body.extend_from_slice(&1u16.to_le_bytes()); // nChannels
        body.extend_from_slice(&sample_rate.to_le_bytes());
        body.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
        body.extend_from_slice(&block_align.to_le_bytes());
        body.extend_from_slice(&(block_align * 8).to_le_bytes()); // wBitsPerSample
        body
    }

    /// The length a buffer's audio occupies, given a `data` chunk that starts at the end of it.
    fn declared_at(wav: &[u8], at: usize) -> u64 {
        u64::from(u32_at(wav, at))
    }

    /// Writes `before` samples, optionally checkpoints them, writes `after` more, and then
    /// abandons the writer.
    ///
    /// `mem::forget` is the whole trick, and the reason this is a helper rather than a line in
    /// each test: dropping a `WavWriter` finalizes it, which rewrites the two size fields and
    /// erases the exact state under test. The leak is deliberate and confined to this one place.
    ///
    /// hound's only post-creation writes are the two size fields at offsets 4 and 64, so
    /// *patching those two is what a kill looks like* -- which is the recipe a consumer needs
    /// too when it wants an unfinished session of its own. Note that `after` has to exceed the
    /// 8 KiB [`BufWriter`] before any of it reaches the disk: unflushed bytes still sitting in
    /// that buffer die with the process, which is the truest available model of a kill.
    fn abandoned_track(path: &Path, spec: WavSpec, before: u32, checkpoint: bool, after: u32) {
        let mut writer = create(path, spec).unwrap();
        for i in 0..before {
            writer.write_sample((i % 7) as f32 * 0.1).unwrap();
        }
        if checkpoint {
            writer.flush().unwrap();
        }
        for i in 0..after {
            writer.write_sample((i % 5) as f32 * -0.1).unwrap();
        }
        std::mem::forget(writer);
    }

    /// What a cleanly finished track proves: it agrees with itself.
    #[test]
    fn a_finalized_track_is_complete_as_declared() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");

        let mut writer = create(&path, mono_float(48_000)).unwrap();
        for _ in 0..1_000 {
            writer.write_sample(0.25f32).unwrap();
        }
        writer.finalize().unwrap();

        assert_eq!(track(&path), TrackEvidence::CompleteAsDeclared);
    }

    /// The shape a run that never got a second checkpoint leaves, and the reason the two
    /// directions of disagreement are two states: the checkpoint froze at `before`, the audio kept
    /// going, and everything past the declaration is bytes a reader that trusts the declaration
    /// never reaches.
    #[test]
    fn a_checkpointed_track_killed_midway_holds_more_than_it_declares() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("speaker.wav");

        // 48 kHz mono float32: 192 000 bytes a second. 5 000 samples checkpointed (26 ms), then
        // far more than the 8 KiB buffer's worth unwritten-but-unflushed.
        abandoned_track(&path, mono_float(48_000), 5_000, true, 60_000);

        let wav = std::fs::read(&path).unwrap();
        let declared = declared_at(&wav, 64);
        let held = u64::try_from(wav.len()).unwrap() - 68;
        assert!(
            declared > 0,
            "the checkpoint should have declared some audio"
        );
        assert!(
            held > declared,
            "audio written after the checkpoint has to have reached the disk: {held} vs {declared}"
        );

        let expected = TrackSpan {
            bytes: held - declared,
            millis: (held - declared) * 1_000 / (48_000 * 4),
        };
        assert!(expected.bytes > 0, "computed from the file itself");
        assert_eq!(track(&path), TrackEvidence::BeyondDeclaration(expected));
    }

    /// A stop inside the first checkpoint interval: seconds of real audio, and a header that
    /// declares none of it. `afinfo` calls this file zero seconds long while `ffprobe` reports the
    /// full five seconds for the same bytes, which is why this state may not be folded into either
    /// "complete" or "empty": merged away, the audio is lost for every reader that would have
    /// found it.
    #[test]
    fn a_track_killed_before_its_first_checkpoint_declares_nothing_and_holds_audio() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");

        abandoned_track(&path, mono_float(48_000), 0, false, 60_000);

        let wav = std::fs::read(&path).unwrap();
        assert_eq!(
            declared_at(&wav, 64),
            0,
            "hound's first header declares zero"
        );
        let held = u64::try_from(wav.len()).unwrap() - 68;
        assert!(held > 0, "and yet the audio is there: {held} bytes");

        let evidence = track(&path);
        match evidence {
            TrackEvidence::BeyondDeclaration(gap) => {
                assert_eq!(gap.bytes, held);
                assert_eq!(gap.millis, held * 1_000 / (48_000 * 4));
            }
            other => panic!("expected the undeclared-audio state, got {other:?}"),
        }
    }

    /// Bytes that were declared and are now gone. Our own death cannot produce this -- see
    /// `track`'s note on hound's write order -- so the fixture makes it the way real life does:
    /// a finished file that gets shortened afterwards.
    #[test]
    fn a_truncated_track_is_short_by_the_missing_amount() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");

        let mut writer = create(&path, mono_float(16_000)).unwrap();
        for _ in 0..4_000 {
            writer.write_sample(0.1f32).unwrap();
        }
        writer.finalize().unwrap();

        let len = std::fs::metadata(&path).unwrap().len();
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(len - 1_600)
            .unwrap();

        assert_eq!(
            track(&path),
            TrackEvidence::ShortBy(TrackSpan {
                bytes: 1_600,
                // 16 kHz mono float32: 64 000 bytes a second, so 1 600 bytes is a quarter-second.
                millis: 25,
            })
        );
    }

    #[test]
    fn a_header_written_and_never_filled_in_holds_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");

        // Dropped unwritten: hound finalizes, and finalizing zero samples declares zero.
        let writer = create(&path, mono_float(48_000)).unwrap();
        drop(writer);

        assert_eq!(std::fs::metadata(&path).unwrap().len(), 68);
        assert_eq!(track(&path), TrackEvidence::HeaderOnly);
    }

    #[test]
    fn a_track_that_was_never_created_is_absent_rather_than_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(track(&dir.path().join("mic.wav")), TrackEvidence::Absent);
    }

    /// Zero bytes folds into `NotAWav` rather than getting a state of its own: hound writes the
    /// header as it creates the file, so nothing ever leaves this behind.
    #[test]
    fn a_file_that_is_not_a_wav_at_all_says_so() {
        let dir = tempfile::tempdir().unwrap();

        let empty = dir.path().join("empty.wav");
        std::fs::File::create(&empty).unwrap();
        assert_eq!(track(&empty), TrackEvidence::NotAWav);

        let text = dir.path().join("notes.wav");
        std::fs::write(&text, b"not a riff file at all").unwrap();
        assert_eq!(track(&text), TrackEvidence::NotAWav);
    }

    /// Foreign files put `LIST` metadata ahead of `fmt `, and an odd-sized chunk carries a pad
    /// byte its own size excludes. Same walk `channel_mask_of` uses, same fixture grown up: a
    /// complete file behind that prologue still measures complete, and the mask is still found.
    #[test]
    fn a_preceding_odd_sized_list_chunk_does_not_move_the_verdict() {
        let audio = vec![0u8; 4_000];
        let wav = riff_with(&[
            // A 5-byte LIST: odd, so a pad byte follows it.
            chunk_bytes(b"LIST", 5, b"INFO\0"),
            chunk_bytes(b"fmt ", 16, &fmt_body(48_000, 4)),
            chunk_bytes(b"data", 4_000, &audio),
        ]);

        assert_eq!(
            evidence(&wav, u64::try_from(wav.len()).unwrap()),
            TrackEvidence::CompleteAsDeclared
        );
    }

    /// A chunk that claims more bytes than the read window holds stops the walk instead of
    /// sending it looking: the answer is `Unknown`, arrived at without reading the track.
    #[test]
    fn a_chunk_claiming_more_than_the_window_stops_the_walk_without_reading_the_track() {
        let wav = riff_with(&[
            // Declares nearly 4 GiB; the file is a few dozen bytes, so a verdict here could only
            // have come from stopping at the window rather than from chasing the claim.
            chunk_bytes(b"JUNK", u32::MAX - 8, b"junk"),
            chunk_bytes(b"fmt ", 16, &fmt_body(48_000, 4)),
            chunk_bytes(b"data", 4_000, &[0u8; 4_000]),
        ]);
        assert!(u64::try_from(wav.len()).unwrap() < HEADER_WINDOW);

        assert_eq!(
            evidence(&wav, u64::try_from(wav.len()).unwrap()),
            TrackEvidence::Unknown
        );
    }

    /// An unusable `fmt ` is `Unknown`, never a divide-by-zero and never a plausible-sounding
    /// number of milliseconds.
    #[test]
    fn an_unusable_fmt_chunk_is_answered_with_unknown_rather_than_a_number() {
        let audio = vec![0u8; 4_000];
        let len = u64::try_from(audio.len()).unwrap() + 12 + 24 + 8;

        let zero_rate = riff_with(&[
            chunk_bytes(b"fmt ", 16, &fmt_body(0, 4)),
            chunk_bytes(b"data", 4_000, &audio),
        ]);
        assert_eq!(evidence(&zero_rate, len), TrackEvidence::Unknown);

        let zero_align = riff_with(&[
            chunk_bytes(b"fmt ", 16, &fmt_body(48_000, 0)),
            chunk_bytes(b"data", 4_000, &audio),
        ]);
        assert_eq!(evidence(&zero_align, len), TrackEvidence::Unknown);

        // A `fmt ` body that stops before the fields live in.
        let stub = riff_with(&[
            chunk_bytes(b"fmt ", 4, b"\x01\x00\x01\x00"),
            chunk_bytes(b"data", 4_000, &audio),
        ]);
        assert_eq!(evidence(&stub, len), TrackEvidence::Unknown);

        // RIFF/WAVE with no `data` at all: nothing to weigh.
        let no_data = riff_with(&[chunk_bytes(b"fmt ", 16, &fmt_body(48_000, 4))]);
        assert_eq!(
            evidence(&no_data, u64::try_from(no_data.len()).unwrap()),
            TrackEvidence::Unknown
        );
    }

    /// hound counts the `data` length in a `u32`, so a recording long enough to wrap it leaves a
    /// file much bigger than its own declaration. That must not be read as "complete", and the
    /// arithmetic describing it must not overflow on the release build a user actually runs.
    ///
    /// The file is sparse: `set_len` reserves the length without writing it, so this costs
    /// nothing on APFS and exercises the same arithmetic a real six-hour recording would.
    #[test]
    fn a_wrapped_declaration_on_an_enormous_file_is_measured_not_panicked() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("marathon.wav");

        let mut writer = create(&path, mono_float(48_000)).unwrap();
        writer.write_sample(0.5f32).unwrap();
        writer.finalize().unwrap();

        // Wrap the declaration down to something tiny and reserve four gigabytes behind it.
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        use std::os::unix::fs::FileExt;
        file.write_all_at(&16u32.to_le_bytes(), 64).unwrap();
        file.set_len(4 * 1_024 * 1_024 * 1_024 + 68).unwrap();
        drop(file);

        let held = 4 * 1_024 * 1_024 * 1_024;
        let expected = TrackSpan {
            bytes: held - 16,
            millis: (held - 16) * 1_000 / (48_000 * 4),
        };
        assert_eq!(track(&path), TrackEvidence::BeyondDeclaration(expected));
    }

    /// One track and not the other is ordinary: a dead input device abandons one engine and
    /// starts the other, so saying "both tracks" would be a lie of the same family as saying
    /// "crashed".
    #[test]
    fn an_unfinished_session_reports_each_track_separately() {
        let dir = tempfile::tempdir().unwrap();
        let paths = crate::Paths::new(dir.path());
        let session = paths.session(&crate::SessionId::parse("20260809-052607").unwrap());
        std::fs::create_dir_all(session.dir()).unwrap();

        abandoned_track(&session.mic_wav(), mono_float(48_000), 0, false, 60_000);

        let unfinished = unfinished(&session);
        assert!(
            matches!(unfinished.mic, TrackEvidence::BeyondDeclaration(_)),
            "the mic track that is there has to report what it holds: {:?}",
            unfinished.mic
        );
        assert_eq!(
            unfinished.speaker,
            TrackEvidence::Absent,
            "and the one that never started has to say it never started"
        );
    }

    /// A file that exists and will not open is a state, not a silence: the report still has a
    /// row for it.
    #[test]
    fn a_file_that_cannot_be_read_is_reported_rather_than_skipped() {
        // Running as root reads anything, so the state is unreachable rather than absent.
        // SAFETY: `geteuid` takes no arguments and cannot fail.
        if unsafe { libc::geteuid() } == 0 {
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");
        std::fs::write(&path, b"RIFF\0\0\0\0WAVE").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();

        assert_eq!(track(&path), TrackEvidence::Unreadable);
    }
}
