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

use std::fs::File;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use hound::{WavSpec, WavWriter};

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

    let mut at = 12;
    while at + 8 <= wav.len() {
        let id = &wav[at..at + 4];
        let size = u32::from_le_bytes(wav[at + 4..at + 8].try_into().ok()?) as usize;
        let body = at + 8;

        if id == b"fmt " {
            // wFormatTag(2) + WAVEFORMAT(14) + wBitsPerSample(2) + cbSize(2)
            // + wValidBitsPerSample(2) puts dwChannelMask 20 bytes into the chunk.
            let fmt = wav.get(body..body + size.min(wav.len().saturating_sub(body)))?;
            if fmt.len() < 24
                || u16::from_le_bytes(fmt[0..2].try_into().ok()?) != WAVE_FORMAT_EXTENSIBLE
            {
                return None;
            }
            return Some(u32::from_le_bytes(fmt[20..24].try_into().ok()?));
        }

        // Chunk bodies are word-aligned: an odd size carries a pad byte the size excludes.
        at = body + size + (size & 1);
    }

    None
}

/// The header, asserted by layout rather than only by outcome.
///
/// A hound upgrade that moved `dwChannelMask` would make the shim quietly stop patching, so
/// these tests pin the offsets around it -- the format tag, `cbSize`, the SubFormat GUID, and
/// where `data` starts -- and fail loudly instead.
#[cfg(test)]
mod tests {
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
}
