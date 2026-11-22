//! The FLAC spelling of a 16 kHz mono raw track, and a streaming writer for it.
//!
//! A raw track (`mic.flac`, `speaker.flac`) is 16 kHz mono 16-bit FLAC: the one format every
//! downstream stage reads, small enough that a long meeting is tens of megabytes rather than
//! gigabytes, and written *as the audio arrives* so a recorder holds about one block of it in
//! memory. This module owns that layout -- the block size, the quantisation, the STREAMINFO
//! rules -- for the same reason [`crate::wav`] owns the WAV header: the session contract is
//! what these files *are*, not only where they live.
//!
//! # Crash behaviour
//!
//! Every frame is self-contained, so a file cut anywhere decodes up to its last whole frame.
//! STREAMINFO is written up front as a placeholder that already states the one block size
//! (macOS `afconvert` reads a file whose minimum and maximum block sizes disagree as zero
//! frames; MH-40.1) with total samples and MD5 left at zero, which FLAC defines as "unknown".
//! [`FlacTrackWriter::finish`] seeks back and replaces it with the real totals and the MD5 of
//! the PCM. Output is pushed to the OS about once per second of audio, so a killed recorder
//! loses roughly that much and a reader sees a plain "no declared length" file.
//!
//! The encoder is flacenc, not rusty_flac's: rusty_flac's frames carry no sample-rate code in
//! their headers, which `afconvert` reads as zero frames. rusty_flac remains the decoder.

use std::io::{self, BufWriter, Seek, SeekFrom, Write};

use flacenc::bitsink::ByteSink;
use flacenc::component::{BitRepr, StreamInfo};
use flacenc::config;
use flacenc::error::{Verified, Verify};
use flacenc::source::{Fill, FrameBuf};
use md5::{Digest, Md5};

/// The sample rate of every raw track and derived track. Not configurable: it is a property of
/// the models downstream.
pub const TRACK_RATE: u32 = 16_000;

/// 16-bit full scale: the divisor from FLAC samples to `f32`, the multiplier back.
pub const FLAC_SCALE: f32 = 32_768.0;

/// Samples per FLAC frame. One size throughout, because afconvert needs STREAMINFO's minimum
/// and maximum block sizes to agree.
pub const FLAC_BLOCK_SAMPLES: usize = 4096;

/// flacenc will not encode a block shorter than this (MH-40.1 wrote 16; the library says 32).
const MIN_BLOCK_SAMPLES: usize = 32;

/// Samples between pushes of buffered bytes to the OS: one second of audio.
const FLUSH_SAMPLES: usize = TRACK_RATE as usize;

/// STREAMINFO sits right after the 4-byte marker and the 4-byte metadata block header.
const STREAMINFO_AT: u64 = 8;

/// One sample on the 16-bit grid, as the integer FLAC stores. NaN clamps to NaN and casts to
/// 0, which is silence: the right reading of a sample that carries no signal.
pub fn quantize_sample(sample: f32) -> i32 {
    (sample.clamp(-1.0, 1.0) * FLAC_SCALE)
        .round()
        .clamp(-32768.0, 32767.0) as i32
}

fn other(e: impl std::fmt::Display) -> io::Error {
    io::Error::other(e.to_string())
}

/// Streams 16 kHz mono 16-bit samples into a FLAC file, a block at a time.
///
/// Generic over the sink so the same code writes a live file (the recorder) and the temp file
/// [`crate::write_atomic_with`] hands out (import). The sink must be at offset 0 and seekable,
/// because [`FlacTrackWriter::finish`] rewrites STREAMINFO in place.
///
/// A track whose length leaves a final block under 32 samples (flacenc's floor) is padded with
/// silence to 32, and the padding is counted in the totals and the MD5, so the file always
/// agrees with itself. That is at most 2 ms of trailing silence.
pub struct FlacTrackWriter<W: Write + Seek> {
    out: BufWriter<W>,
    config: Verified<config::Encoder>,
    info: StreamInfo,
    /// Quantised samples waiting for a whole block. Never holds more than one block.
    pending: Vec<i32>,
    frame_number: usize,
    md5: Md5,
    total_samples: u64,
    since_flush: usize,
    frame_bytes: ByteSink,
}

impl<W: Write + Seek> FlacTrackWriter<W> {
    /// Writes the marker and the placeholder STREAMINFO to `sink`.
    pub fn new(sink: W) -> io::Result<Self> {
        let config = config::Encoder::default()
            .into_verified()
            .map_err(|(_, e)| other(e))?;
        let mut info = StreamInfo::new(TRACK_RATE as usize, 1, 16).map_err(other)?;
        info.set_block_sizes(FLAC_BLOCK_SAMPLES, FLAC_BLOCK_SAMPLES)
            .map_err(other)?;

        let mut out = BufWriter::with_capacity(64 * 1024, sink);
        out.write_all(b"fLaC")?;
        // Last-metadata-block flag set, type 0 (STREAMINFO), length 34.
        out.write_all(&[0x80, 0, 0, 34])?;
        out.write_all(&streaminfo_bytes(&info)?)?;
        // The header reaches the OS now, so a recorder killed before its first second of audio
        // still leaves a file that says what it is, rather than an empty one.
        out.flush()?;

        Ok(FlacTrackWriter {
            out,
            config,
            info,
            pending: Vec::with_capacity(FLAC_BLOCK_SAMPLES),
            frame_number: 0,
            md5: Md5::new(),
            total_samples: 0,
            since_flush: 0,
            frame_bytes: ByteSink::new(),
        })
    }

    /// Quantises `samples` to 16 bits and writes every block they complete.
    pub fn push(&mut self, samples: &[f32]) -> io::Result<()> {
        for sample in samples {
            self.pending.push(quantize_sample(*sample));
            if self.pending.len() == FLAC_BLOCK_SAMPLES {
                self.write_block()?;
            }
        }
        Ok(())
    }

    /// Samples held back waiting for a whole block. Exposed so a test can state the memory
    /// bound rather than assume it.
    pub fn buffered_samples(&self) -> usize {
        self.pending.len()
    }

    /// Encodes `pending` as one frame and appends it.
    fn write_block(&mut self) -> io::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        // Padding to flacenc's minimum block; see the type's note.
        self.pending
            .resize(self.pending.len().max(MIN_BLOCK_SAMPLES), 0);
        let size = self.pending.len();

        let mut buf = FrameBuf::with_size(1, size).map_err(other)?;
        buf.fill_interleaved(&self.pending).map_err(other)?;
        let frame =
            flacenc::encode_fixed_size_frame(&self.config, &buf, self.frame_number, &self.info)
                .map_err(other)?;
        self.info.update_frame_info(&frame);

        self.frame_bytes.clear();
        frame.write(&mut self.frame_bytes).map_err(other)?;
        self.out.write_all(self.frame_bytes.as_slice())?;

        // The MD5 is over the samples as signed 16-bit little-endian, which is what a decoder
        // reproduces.
        let mut pcm = Vec::with_capacity(size * 2);
        for sample in &self.pending {
            pcm.extend_from_slice(&(*sample as i16).to_le_bytes());
        }
        self.md5.update(&pcm);

        self.frame_number += 1;
        self.total_samples += size as u64;
        self.since_flush += size;
        self.pending.clear();
        if self.since_flush >= FLUSH_SAMPLES {
            self.out.flush()?;
            self.since_flush = 0;
        }
        Ok(())
    }

    /// Writes the last block and replaces the placeholder STREAMINFO with the real one.
    /// Returns the number of samples in the file, padding included.
    pub fn finish(mut self) -> io::Result<u64> {
        self.write_block()?;

        self.info
            .set_block_sizes(FLAC_BLOCK_SAMPLES, FLAC_BLOCK_SAMPLES)
            .map_err(other)?;
        // flacenc's frame bookkeeping already counted the samples; this states it.
        self.info.set_total_samples(self.total_samples as usize);
        let digest: [u8; 16] = self.md5.finalize().into();
        self.info.set_md5_digest(&digest);

        self.out.seek(SeekFrom::Start(STREAMINFO_AT))?;
        self.out.write_all(&streaminfo_bytes(&self.info)?)?;
        self.out.flush()?;
        Ok(self.total_samples)
    }
}

fn streaminfo_bytes(info: &StreamInfo) -> io::Result<Vec<u8>> {
    let mut sink = ByteSink::new();
    info.write(&mut sink).map_err(other)?;
    let bytes = sink.as_slice().to_vec();
    debug_assert_eq!(bytes.len(), 34);
    Ok(bytes)
}

/// Total samples a FLAC header declares, or `None` when `header` is not a FLAC marker followed
/// by a STREAMINFO block. Zero is FLAC's "unknown", which a killed recorder leaves behind.
///
/// Takes the leading bytes only (42 are needed), so classifying a track never reads it.
pub fn declared_samples(header: &[u8]) -> Option<(u32, u64)> {
    if header.len() < 42 || &header[0..4] != b"fLaC" || header[4] & 0x7f != 0 {
        return None;
    }
    // 20-bit sample rate, 3-bit channels-1, 5-bit bits-1, 36-bit total samples.
    let rate =
        (u32::from(header[18]) << 12) | (u32::from(header[19]) << 4) | (u32::from(header[20]) >> 4);
    let total = (u64::from(header[21] & 0x0f) << 32)
        | (u64::from(header[22]) << 24)
        | (u64::from(header[23]) << 16)
        | (u64::from(header[24]) << 8)
        | u64::from(header[25]);
    Some((rate, total))
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::Path;

    use super::*;

    /// Samples on the 16-bit grid, so a round trip is exact.
    fn pcm(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 37) % 2001) as f32 - 1000.0) / FLAC_SCALE)
            .collect()
    }

    fn grid(samples: &[f32]) -> Vec<i32> {
        samples.iter().map(|s| quantize_sample(*s)).collect()
    }

    fn write(path: &Path, samples: &[f32]) -> u64 {
        let mut writer = FlacTrackWriter::new(File::create(path).unwrap()).unwrap();
        writer.push(samples).unwrap();
        writer.finish().unwrap()
    }

    fn decode(bytes: &[u8]) -> Vec<i32> {
        let mut decoder = rusty_flac::Decoder::new(bytes).unwrap();
        let mut samples = Vec::new();
        let mut block = Vec::new();
        loop {
            block.clear();
            match decoder.next_frame_interleaved::<i32>(&mut block) {
                Ok(Some(_)) => samples.extend_from_slice(&block),
                Ok(None) | Err(rusty_flac::DecodeError::Truncated) => break,
                Err(e) => panic!("decode failed: {e}"),
            }
        }
        samples
    }

    #[test]
    fn awkward_lengths_decode_bit_exact_and_pad_only_a_sub_32_tail() {
        let dir = tempfile::tempdir().unwrap();
        for len in [
            0usize, 1, 15, 16, 17, 31, 32, 33, 4095, 4096, 4097, 4111, 4112, 4127, 4128, 8193,
        ] {
            let path = dir.path().join(format!("{len}.flac"));
            let input = pcm(len);
            let total = write(&path, &input);

            let tail = len % FLAC_BLOCK_SAMPLES;
            let padded = if tail != 0 && tail < MIN_BLOCK_SAMPLES {
                len - tail + MIN_BLOCK_SAMPLES
            } else {
                len
            };
            assert_eq!(total as usize, padded, "reported length for {len}");

            let bytes = std::fs::read(&path).unwrap();
            let mut expected = grid(&input);
            expected.resize(padded, 0);
            assert_eq!(decode(&bytes), expected, "samples for {len}");
        }
    }

    #[test]
    fn finish_patches_total_samples_and_md5() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.flac");
        let input = pcm(10_000);
        write(&path, &input);
        let bytes = std::fs::read(&path).unwrap();

        let (rate, total) = declared_samples(&bytes).unwrap();
        assert_eq!((rate, total), (16_000, 10_000));

        let mut pcm_bytes = Vec::new();
        for s in grid(&input) {
            pcm_bytes.extend_from_slice(&(s as i16).to_le_bytes());
        }
        let expected: [u8; 16] = Md5::digest(&pcm_bytes).into();
        // STREAMINFO's MD5 is its last 16 bytes: file offset 8 + 34 - 16.
        assert_eq!(&bytes[26..42], &expected);

        // The patched header states the one block size, so afconvert reads it.
        assert_eq!(&bytes[8..12], &[0x10, 0x00, 0x10, 0x00]);

        // And an independent decoder agrees the stream verifies.
        let decoder = rusty_flac::Decoder::new(&bytes).unwrap();
        assert_eq!(decoder.streaminfo().total_samples, 10_000);
    }

    #[test]
    fn a_file_that_was_never_finished_decodes_to_its_last_whole_frame() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("killed.flac");
        let input = pcm(3 * FLAC_BLOCK_SAMPLES + 100);

        let mut writer = FlacTrackWriter::new(File::create(&path).unwrap()).unwrap();
        writer.push(&input).unwrap();
        // The whole frames are in the BufWriter; push them to the OS the way the once-a-second
        // flush would have, then die without `finish`.
        writer.out.flush().unwrap();
        std::mem::forget(writer);

        let bytes = std::fs::read(&path).unwrap();
        // Placeholder header: unknown length, but the block sizes already agree.
        assert_eq!(declared_samples(&bytes).unwrap().1, 0);
        assert_eq!(&bytes[8..12], &[0x10, 0x00, 0x10, 0x00]);
        assert_eq!(decode(&bytes), grid(&input[..3 * FLAC_BLOCK_SAMPLES]));
    }

    #[test]
    fn output_is_flushed_about_once_a_second_of_audio() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("flushed.flac");
        let mut writer = FlacTrackWriter::new(File::create(&path).unwrap()).unwrap();
        // Just over a second, in blocks: the flush interval has passed without any explicit
        // flush from the test.
        writer
            .push(&pcm(FLUSH_SAMPLES + FLAC_BLOCK_SAMPLES))
            .unwrap();
        let on_disk = std::fs::metadata(&path).unwrap().len();
        assert!(on_disk > 42 + 1000, "only {on_disk} bytes reached the OS");
        std::mem::forget(writer);
    }

    #[test]
    fn memory_stays_at_one_block_however_long_the_track() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("long.flac");
        let mut writer = FlacTrackWriter::new(File::create(&path).unwrap()).unwrap();
        let piece = pcm(1_000);
        for _ in 0..200 {
            writer.push(&piece).unwrap();
            assert!(writer.buffered_samples() < FLAC_BLOCK_SAMPLES);
        }
        assert_eq!(writer.finish().unwrap(), 200_000);
    }

    #[test]
    fn declared_samples_refuses_what_is_not_a_streaminfo() {
        assert_eq!(declared_samples(b"RIFF"), None);
        assert_eq!(declared_samples(&[0u8; 64]), None);
    }
}
