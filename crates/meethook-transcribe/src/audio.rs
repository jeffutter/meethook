//! Getting a recorded track into the one format Whisper accepts.
//!
//! whisper.cpp takes 16 kHz mono `f32` and nothing else. The recorder already writes mono
//! 32-bit float at whatever rate the device runs at, so sample rate is the only thing that
//! ever needs changing -- and the conversion is done streaming, because holding an hour of
//! 48 kHz audio *and* its 16 kHz result at once costs ~900 MB to save a few lines of code.

use std::path::Path;

#[cfg(test)]
use hound::WavSpec;
use hound::{SampleFormat, WavReader};
use meethook_session::{
    FLAC_BLOCK_SAMPLES, FLAC_SCALE, FlacTrackWriter, ResampleError, Resampler, quantize_sample,
    write_atomic_with,
};

use crate::progress::Phase;
use crate::{Error, Result};

/// The sample rate whisper.cpp requires. Not configurable: it is a property of the model.
pub const TARGET_RATE: u32 = 16_000;

/// The header of the legacy WAV raw track, which only tests still write (to stand in for a
/// session recorded before FLAC). Recorded and imported tracks are FLAC; see
/// [`write_raw_track_flac`].
#[cfg(test)]
const TRACK_SPEC: WavSpec = WavSpec {
    channels: 1,
    sample_rate: TARGET_RATE,
    bits_per_sample: 32,
    sample_format: SampleFormat::Float,
};

/// Reads a mono float WAV or a mono 16-bit FLAC and returns it as 16 kHz mono `f32`.
///
/// Which one is decided by the file's first four bytes (`fLaC`), not its extension, so a
/// session keeps working whichever format its tracks were written in. FLAC samples map to
/// `f32` as `sample / 32768`, the same grid a 16-bit WAV would use.
///
/// A file that is not mono 32-bit float (WAV) or mono 16-bit (FLAC) is rejected by name and
/// header rather than reinterpreted: silently misreading a stereo or integer WAV produces
/// audio that sounds like noise to the model and like a transcription bug to everyone else.
///
/// A FLAC cut off inside its last frame (a recorder killed mid-write) yields the samples of
/// the frames that are whole; only a file with no whole frame at all is an error.
///
/// A zero-length track is not an error. A session where nobody spoke is a real session, and
/// it still has to produce transcript files so a rerun skips it.
pub fn read_track_16k_mono(path: &Path) -> Result<Vec<f32>> {
    if is_flac(path)? {
        return read_flac_16k_mono(path);
    }
    let reader = WavReader::open(path).map_err(|e| Error::wav(path, e))?;
    let spec = reader.spec();

    if spec.channels != 1 || spec.sample_format != SampleFormat::Float || spec.bits_per_sample != 32
    {
        return Err(Error::UnsupportedAudio {
            path: path.to_path_buf(),
            detail: format!(
                "expected mono 32-bit float, found {} channel(s), {}-bit {:?}",
                spec.channels, spec.bits_per_sample, spec.sample_format
            ),
        });
    }

    let frames = reader.len() as usize;
    // Half a gigabyte of 48 kHz audio takes a minute or so to read and resample a sample at a
    // time, and this is the first thing `transcribe` does, so it is also the first stretch of
    // silence a user would have to guess about. Named after the file, because a session reads
    // two of these back to back.
    let mut phase = Phase::start(format!("reading {}", file_name(path)));

    if spec.sample_rate == TARGET_RATE {
        let mut samples = Vec::with_capacity(frames);
        for sample in reader.into_samples::<f32>() {
            phase.at(samples.len(), frames);
            samples.push(sample.map_err(|e| Error::wav(path, e))?);
        }
        phase.done();
        return Ok(samples);
    }

    let mut resample = Resample::new(spec.sample_rate, TARGET_RATE, frames)?;
    for (read, sample) in reader.into_samples::<f32>().enumerate() {
        phase.at(read, frames);
        resample.push(sample.map_err(|e| Error::wav(path, e))?)?;
    }
    let samples = resample.finish();
    phase.done();
    samples
}

/// Whether the file starts with the FLAC marker. A file too short to hold one is not FLAC;
/// the WAV reader then says what is wrong with it.
fn is_flac(path: &Path) -> Result<bool> {
    use std::io::Read;

    let mut marker = [0u8; 4];
    let mut file = std::fs::File::open(path).map_err(|e| Error::Flac {
        path: path.to_path_buf(),
        detail: e.to_string(),
    })?;
    match file.read_exact(&mut marker) {
        Ok(()) => Ok(&marker == b"fLaC"),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(Error::Flac {
            path: path.to_path_buf(),
            detail: e.to_string(),
        }),
    }
}

/// The FLAC half of [`read_track_16k_mono`].
///
/// The whole file is held (about 70 MB for a 79 minute track) because the decoder works on a
/// byte slice, but frames are decoded one at a time into the same push-shaped resampler the
/// WAV path uses, so the decoded `f32` track is never held beside its resampled twin.
fn read_flac_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let flac_err = |detail: String| Error::Flac {
        path: path.to_path_buf(),
        detail,
    };
    let bytes = std::fs::read(path).map_err(|e| Error::Flac {
        path: path.to_path_buf(),
        detail: e.to_string(),
    })?;
    let mut decoder = rusty_flac::Decoder::new(&bytes).map_err(|e| flac_err(e.to_string()))?;
    let info = decoder.streaminfo().clone();

    if info.channels != 1 || info.bits_per_sample != 16 {
        return Err(Error::UnsupportedAudio {
            path: path.to_path_buf(),
            detail: format!(
                "expected mono 16-bit FLAC, found {} channel(s), {}-bit",
                info.channels, info.bits_per_sample
            ),
        });
    }

    // Zero in STREAMINFO means "unknown", which is what a killed recorder leaves behind. It
    // only sizes the output and paces the progress line, so it is allowed to be wrong.
    let frames = info.total_samples as usize;
    let mut phase = Phase::start(format!("reading {}", file_name(path)));
    let mut resample = if info.sample_rate == TARGET_RATE {
        None
    } else {
        Some(Resample::new(info.sample_rate, TARGET_RATE, frames)?)
    };
    let mut samples = Vec::with_capacity(if resample.is_none() { frames } else { 0 });
    let mut block: Vec<i32> = Vec::new();
    let mut read = 0usize;
    loop {
        block.clear();
        match decoder.next_frame_interleaved::<i32>(&mut block) {
            Ok(Some(_)) => {}
            Ok(None) => break,
            // The frames before a cut one are intact; an empty result means the file never
            // had a whole frame, which is damage rather than a short recording.
            Err(rusty_flac::DecodeError::Truncated) if read > 0 => break,
            Err(e) => return Err(flac_err(e.to_string())),
        }
        for &sample in &block {
            phase.at(read, frames.max(read + 1));
            read += 1;
            let value = sample as f32 / FLAC_SCALE;
            match resample.as_mut() {
                Some(resample) => resample.push(value)?,
                None => samples.push(value),
            }
        }
    }
    let samples = match resample {
        Some(resample) => resample.finish()?,
        None => samples,
    };
    phase.done();
    Ok(samples)
}

/// A path's file name for a progress label, falling back to the whole path.
///
/// Lossy rather than fallible: this decides what a heartbeat line says, and a track whose name
/// is not UTF-8 is still a track worth reporting progress over.
pub(crate) fn file_name(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}

/// Converts audio already held in memory from `source_rate` to `target_rate`.
///
/// Delay-compensated, and the output holds exactly `len * target / source` samples (rounded
/// down), so a 16 kHz -> 48 kHz -> 16 kHz round trip returns the input's length and timeline.
pub(crate) fn resample_between(
    samples: &[f32],
    source_rate: u32,
    target_rate: u32,
) -> Result<Vec<f32>> {
    if source_rate == target_rate {
        return Ok(samples.to_vec());
    }
    let mut resample = Resample::new(source_rate, target_rate, samples.len())?;
    for sample in samples {
        resample.push(*sample)?;
    }
    resample.finish()
}

/// Converts audio already held in memory to [`TARGET_RATE`].
///
/// The in-memory twin of the loop in [`read_track_16k_mono`], for callers whose samples did
/// not come from a file this crate is willing to open -- see [`crate::import`]. Both go
/// through [`Resample`], so the filter length and the delay compensation are decided once.
pub(crate) fn resample_to_target(samples: &[f32], source_rate: u32) -> Result<Vec<f32>> {
    resample_between(samples, source_rate, TARGET_RATE)
}

/// Writes a legacy-format raw track: 16 kHz mono float WAV, atomically. Test-only, to build
/// the session a pre-FLAC recorder left behind and prove it still reads.
#[cfg(test)]
pub(crate) fn write_track_16k_mono(path: &Path, audio: &[f32]) -> Result<()> {
    write_atomic_with(path, |file| {
        // Buffered: hound writes each sample straight through, and an hour of audio is 57
        // million of them. Via `meethook_session::wav` rather than hound directly, so the mono
        // channel mask says front-centre instead of hound's front-left.
        let mut writer = meethook_session::wav::new(std::io::BufWriter::new(file), TRACK_SPEC)
            .map_err(|e| Error::wav(path, e))?;
        // Usually finishes inside the reporting interval and therefore usually says nothing --
        // but it is 47 million `write_sample` calls on a long session, on a disk that may not
        // be fast, and covering it costs a masked counter.
        let mut phase = Phase::start(format!("writing {}", file_name(path)));
        for (written, sample) in audio.iter().enumerate() {
            phase.at(written, audio.len());
            writer
                .write_sample(*sample)
                .map_err(|e| Error::wav(path, e))?;
        }
        let finalized = writer.finalize().map_err(|e| Error::wav(path, e));
        phase.done();
        finalized
    })
}

/// Writes a raw track (`mic.flac`, `speaker.flac`) in the format the recorder writes: 16 kHz
/// mono 16-bit FLAC, atomically.
///
/// The same writer the recorder streams through, so an imported session is indistinguishable
/// from a recorded one. Samples are quantised to the 16-bit grid; a track whose length leaves a
/// final block under 32 samples gains up to 2 ms of trailing silence (see
/// [`meethook_session::FlacTrackWriter`]).
pub(crate) fn write_raw_track_flac(path: &Path, audio: &[f32]) -> Result<()> {
    let flac_err = |e: std::io::Error| Error::Flac {
        path: path.to_path_buf(),
        detail: e.to_string(),
    };
    write_atomic_with(path, |file| {
        let mut writer = FlacTrackWriter::new(file).map_err(flac_err)?;
        let mut phase = Phase::start(format!("writing {}", file_name(path)));
        for (index, block) in audio.chunks(FLAC_BLOCK_SAMPLES).enumerate() {
            phase.at(index * FLAC_BLOCK_SAMPLES, audio.len());
            writer.push(block).map_err(flac_err)?;
        }
        writer.finish().map_err(flac_err)?;
        phase.done();
        Ok(())
    })
}

/// Snaps a float track to the 16-bit grid a FLAC holds, in place.
///
/// Applied to the in-memory cleaned track before it is written, so recognition hears exactly
/// the samples a later run reads back from `mic.cleaned.flac`, rather than a float original
/// that exists only until the process exits. Idempotent.
pub(crate) fn quantize_to_16_bit(audio: &mut [f32]) {
    for sample in audio {
        *sample = quantize_sample(*sample) as f32 / FLAC_SCALE;
    }
}

/// Writes the cleaned mic track as 16 kHz mono 16-bit FLAC, atomically.
///
/// This is where a float track is quantised to 16 bits: samples are clamped to [-1, 1],
/// scaled by 32768 and rounded, so a track that is already on the 16-bit grid (anything
/// [`read_track_16k_mono`] returned from a FLAC) round-trips exactly. Whisper's own input
/// is far coarser than that, but the cost is measured per session rather than assumed -- see
/// MH-40.2.
///
/// Atomic for the same reason as [`write_raw_track_flac`]: `mic.cleaned.flac` is what the
/// next run reads, and it must be a whole track or no file.
pub(crate) fn write_cleaned_track_flac(path: &Path, audio: &[f32]) -> Result<()> {
    use flacenc::component::BitRepr;

    let flac_err = |detail: String| Error::Flac {
        path: path.to_path_buf(),
        detail,
    };
    let config = flacenc::error::Verify::into_verified(flacenc::config::Encoder::default())
        .map_err(|(_, e)| flac_err(e.to_string()))?;

    let mut stream = flacenc::encode_with_fixed_block_size(
        &config,
        QuantizedTrack { audio, head: 0 },
        FLAC_BLOCK_SAMPLES,
    )
    .map_err(|e| flac_err(e.to_string()))?;

    // The encoder records the short final block as the minimum block size, which makes the
    // stream variable-block-size on paper. macOS `afconvert` then decodes it as zero frames
    // (MH-40.1), so the header states what the frames actually are: all one size, bar a tail.
    let block = FLAC_BLOCK_SAMPLES.min(audio.len().max(16));
    stream
        .stream_info_mut()
        .set_block_sizes(block, block)
        .map_err(|e| flac_err(e.to_string()))?;

    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| flac_err(e.to_string()))?;
    write_atomic_with(path, |file| {
        use std::io::Write;
        file.write_all(sink.as_slice()).map_err(|e| Error::Flac {
            path: path.to_path_buf(),
            detail: e.to_string(),
        })
    })
}

/// A float track presented to the FLAC encoder as 16-bit samples, a block at a time, so the
/// quantised copy is never held whole.
struct QuantizedTrack<'a> {
    audio: &'a [f32],
    head: usize,
}

impl flacenc::source::Source for QuantizedTrack<'_> {
    fn channels(&self) -> usize {
        1
    }

    fn bits_per_sample(&self) -> usize {
        16
    }

    fn sample_rate(&self) -> usize {
        TARGET_RATE as usize
    }

    fn read_samples<F: flacenc::source::Fill>(
        &mut self,
        block_size: usize,
        dest: &mut F,
    ) -> std::result::Result<usize, flacenc::error::SourceError> {
        let end = (self.head + block_size).min(self.audio.len());
        let block: Vec<i32> = self.audio[self.head..end]
            .iter()
            .map(|s| quantize_sample(*s))
            .collect();
        dest.fill_interleaved(&block)?;
        self.head = end;
        Ok(block.len())
    }

    fn len_hint(&self) -> Option<usize> {
        Some(self.audio.len())
    }
}

/// A fixed-ratio sample-rate conversion into a growing in-memory result, fed one sample at a
/// time.
///
/// A thin holder over [`meethook_session::Resampler`], which owns the filter, its delay
/// compensation and the exact output length (the recorder streams through the same one).
/// Push-shaped because the caller that matters is reading a file: holding an hour of 48 kHz
/// audio *and* its 16 kHz result at once costs ~900 MB, which is the whole reason the read
/// loop above is streaming.
struct Resample {
    inner: Resampler,
    out: Vec<f32>,
}

impl Resample {
    /// `frames` is the input length, used to size the output exactly. It is a promise about
    /// how much will be pushed, not a limit: pushing fewer simply yields a shorter track.
    fn new(source_rate: u32, target_rate: u32, frames: usize) -> Result<Self> {
        let inner = Resampler::new(source_rate, target_rate).map_err(resample_error)?;
        let expected =
            (frames as u128 * u128::from(target_rate) / u128::from(source_rate)) as usize;
        Ok(Resample {
            inner,
            out: Vec::with_capacity(expected),
        })
    }

    fn push(&mut self, sample: f32) -> Result<()> {
        self.inner
            .push(std::slice::from_ref(&sample), &mut self.out)
            .map_err(resample_error)
    }

    fn finish(mut self) -> Result<Vec<f32>> {
        self.inner.finish(&mut self.out).map_err(resample_error)?;
        Ok(self.out)
    }
}

fn resample_error(e: ResampleError) -> Error {
    Error::Resample(e.0)
}

/// Synthetic tracks, plus the two measurements needed to read them back.
///
/// Here rather than inside one test module because [`crate::mixdown`] and the crate-root tests
/// need the same shapes, and two copies of a fixture this load-bearing would drift -- the same
/// reason [`crate::loudness::fixtures`] exists. What the mixed-rate session fixture in
/// [`crate::tests`] depends on is precisely that *each* track states its own rate, so the writer
/// that takes a rate per call has to be shared rather than reimplemented beside whichever test
/// asked for it first.
#[cfg(test)]
pub(crate) mod fixtures {
    use hound::{SampleFormat, WavSpec, WavWriter};

    /// A mono 32-bit float WAV at `rate`, written from `samples`.
    ///
    /// Note the header says `rate`, not anything the caller's audio was made at: that
    /// discrepancy is the whole content of a rate-mismatch fixture.
    pub(crate) fn write_wav(path: &std::path::Path, rate: u32, samples: &[f32]) {
        write_wav_spec(
            path,
            WavSpec {
                channels: 1,
                sample_rate: rate,
                bits_per_sample: 32,
                sample_format: SampleFormat::Float,
            },
            samples,
        );
    }

    /// A WAV with an arbitrary header, for the tests that are about the header itself.
    pub(crate) fn write_wav_spec(path: &std::path::Path, spec: WavSpec, samples: &[f32]) {
        let mut writer = WavWriter::create(path, spec).unwrap();
        for sample in samples {
            match spec.sample_format {
                SampleFormat::Float => writer.write_sample(*sample).unwrap(),
                SampleFormat::Int => writer.write_sample(*sample as i16).unwrap(),
            }
        }
        writer.finalize().unwrap();
    }

    /// `seconds` of a continuous sine at `hz`, amplitude 0.5.
    pub(crate) fn tone(rate: u32, seconds: f32, hz: f32) -> Vec<f32> {
        let n = (rate as f32 * seconds) as usize;
        (0..n)
            .map(|i| (i as f32 / rate as f32 * hz * std::f32::consts::TAU).sin() * 0.5)
            .collect()
    }

    /// Quadrature projection magnitude of `hz` over `from..to`: the correlation of the buffer
    /// with a cosine and a sine at that frequency, normalised so a pure tone of amplitude `a`
    /// measures `a` and anything else measures ~0. Exact for a known frequency, insensitive to
    /// unknown phase, and unaffected by the codec's phase distortion - which is why it is used
    /// instead of an FFT.
    ///
    /// Accumulated in `f64`: the windows here run to tens of thousands of samples, and summing
    /// their products in `f32` costs a rounding error comparable to the reject bound callers
    /// assert against.
    pub(crate) fn projection(samples: &[f32], rate: u32, hz: f32, from: usize, to: usize) -> f32 {
        use std::f64::consts::TAU;

        let (mut cos_sum, mut sin_sum) = (0.0f64, 0.0f64);
        for (index, sample) in samples[from..to].iter().enumerate() {
            // Absolute index, so the window's position in the track is part of the geometry
            // rather than something each caller has to keep straight. Projection magnitude
            // does not depend on it; reproducibility across a refactor does.
            let phase = f64::from((index + from) as u32) * f64::from(hz) / f64::from(rate) * TAU;
            cos_sum += f64::from(*sample) * phase.cos();
            sin_sum += f64::from(*sample) * phase.sin();
        }
        // A pure tone of amplitude `a` puts `a * n / 2` into each quadrature, so twice the
        // per-sample mean of the pair's magnitude reads back as `a`.
        ((cos_sum * cos_sum + sin_sum * sin_sum).sqrt() * 2.0 / (to - from) as f64) as f32
    }

    /// Index of the first sample whose magnitude reaches `threshold`, or `None`.
    pub(crate) fn first_onset(samples: &[f32], threshold: f32) -> Option<usize> {
        samples.iter().position(|sample| sample.abs() >= threshold)
    }
}

#[cfg(test)]
mod tests {
    use hound::WavSpec;

    use super::fixtures::*;
    use super::*;

    #[test]
    fn a_16k_track_passes_through_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");
        let samples = tone(16_000, 0.5, 440.0);
        write_wav(&path, 16_000, &samples);

        assert_eq!(read_track_16k_mono(&path).unwrap(), samples);
    }

    #[test]
    fn a_48k_track_is_decimated_to_a_third_of_its_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");
        write_wav(&path, 48_000, &tone(48_000, 2.0, 440.0));

        let resampled = read_track_16k_mono(&path).unwrap();
        assert_eq!(resampled.len(), 32_000);
    }

    /// A phase-preserved 440 Hz tone should survive decimation. Comparing against a
    /// directly generated 16 kHz tone catches both a mangled resample and a resampler delay
    /// left uncompensated, which a length check alone would miss.
    #[test]
    fn the_resampled_signal_still_matches_the_original_waveform() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");
        write_wav(&path, 48_000, &tone(48_000, 1.0, 440.0));

        let resampled = read_track_16k_mono(&path).unwrap();
        let reference = tone(16_000, 1.0, 440.0);

        // Skip the filter's settling region at either end; the steady state is the part that
        // has to line up.
        let worst = resampled[2000..14_000]
            .iter()
            .zip(&reference[2000..14_000])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.05, "worst sample error {worst}");
    }

    #[test]
    fn an_empty_track_yields_no_samples_rather_than_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");
        write_wav(&path, 48_000, &[]);

        assert!(read_track_16k_mono(&path).unwrap().is_empty());
    }

    #[test]
    fn a_non_float_track_is_rejected_by_name_and_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");
        write_wav_spec(
            &path,
            WavSpec {
                channels: 2,
                sample_rate: 48_000,
                bits_per_sample: 16,
                sample_format: SampleFormat::Int,
            },
            &[0.0; 64],
        );

        let err = read_track_16k_mono(&path).unwrap_err().to_string();
        assert!(err.contains("2 channel(s)"), "{err}");
        assert!(err.contains("16-bit"), "{err}");
    }

    /// The writer and the reader agree, which is the only thing [`TRACK_SPEC`] exists to
    /// guarantee: every track this crate writes is one the strict reader will open again.
    #[test]
    fn a_written_track_reads_back_sample_for_sample_through_the_strict_reader() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("track.wav");
        let samples = tone(16_000, 0.5, 440.0);

        write_track_16k_mono(&path, &samples).unwrap();

        assert_eq!(read_track_16k_mono(&path).unwrap(), samples);
    }

    /// One writer covers every WAV track [`crate::import`] builds, so this one assertion is
    /// every WAV the transcribe side produces.
    #[test]
    fn a_written_track_is_tagged_mono_rather_than_front_left() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("track.wav");

        write_track_16k_mono(&path, &tone(16_000, 0.1, 440.0)).unwrap();

        let wav = std::fs::read(&path).unwrap();
        assert_eq!(
            meethook_session::wav::channel_mask_of(&wav),
            Some(meethook_session::wav::MONO_CHANNEL_MASK)
        );
    }

    /// Audio already on the 16-bit grid, which is what a FLAC holds.
    fn on_grid(samples: &[f32]) -> Vec<f32> {
        samples
            .iter()
            .map(|s| (s * FLAC_SCALE).round() / FLAC_SCALE)
            .collect()
    }

    /// Criterion: a WAV and a FLAC of the same audio read back as the same samples.
    #[test]
    fn a_wav_and_a_flac_of_the_same_audio_read_back_identically() {
        let dir = tempfile::tempdir().unwrap();
        let samples = on_grid(&tone(16_000, 1.5, 440.0));
        let wav = dir.path().join("a.wav");
        let flac = dir.path().join("a.flac");
        write_wav(&wav, 16_000, &samples);
        write_cleaned_track_flac(&flac, &samples).unwrap();

        assert_eq!(read_track_16k_mono(&flac).unwrap(), samples);
        assert_eq!(
            read_track_16k_mono(&flac).unwrap(),
            read_track_16k_mono(&wav).unwrap()
        );
    }

    #[test]
    fn a_flac_reader_goes_by_content_not_extension() {
        let dir = tempfile::tempdir().unwrap();
        let samples = on_grid(&tone(16_000, 0.3, 300.0));
        let path = dir.path().join("mic.wav");
        write_cleaned_track_flac(&path, &samples).unwrap();

        assert_eq!(read_track_16k_mono(&path).unwrap(), samples);
    }

    #[test]
    fn a_float_track_is_quantised_to_within_one_16_bit_step() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.cleaned.flac");
        let samples = tone(16_000, 1.0, 440.0);
        write_cleaned_track_flac(&path, &samples).unwrap();

        let back = read_track_16k_mono(&path).unwrap();
        assert_eq!(back.len(), samples.len());
        let worst = back
            .iter()
            .zip(&samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst <= 0.5 / FLAC_SCALE + f32::EPSILON, "worst {worst}");
    }

    #[test]
    fn out_of_range_and_nan_samples_are_clamped_not_wrapped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.flac");
        write_cleaned_track_flac(&path, &[2.0, -2.0, f32::NAN, 1.0, -1.0]).unwrap();

        assert_eq!(
            read_track_16k_mono(&path).unwrap(),
            vec![32767.0 / FLAC_SCALE, -1.0, 0.0, 32767.0 / FLAC_SCALE, -1.0]
        );
    }

    /// Lengths around the block size and the encoder's 16-sample minimum, including none.
    #[test]
    fn flac_tracks_of_awkward_lengths_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.flac");
        for len in [0usize, 1, 15, 16, 17, 4095, 4096, 4097, 8192, 8193] {
            let samples = on_grid(&tone(16_000, len as f32 / 16_000.0 + 0.0001, 220.0)[..len]);
            write_cleaned_track_flac(&path, &samples).unwrap();
            assert_eq!(read_track_16k_mono(&path).unwrap(), samples, "len {len}");
        }
    }

    #[test]
    fn a_flac_track_states_one_block_size_so_afconvert_reads_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.flac");
        write_cleaned_track_flac(&path, &on_grid(&tone(16_000, 1.0, 440.0))).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let info = rusty_flac::Decoder::new(&bytes)
            .unwrap()
            .streaminfo()
            .clone();
        assert_eq!((info.min_block_size, info.max_block_size), (4096, 4096));
        assert_eq!(info.total_samples, 16_000);
        assert_eq!((info.channels, info.bits_per_sample), (1, 16));
    }

    #[test]
    fn a_flac_cut_inside_its_last_frame_keeps_the_whole_frames() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.flac");
        let samples = on_grid(&tone(16_000, 1.0, 440.0));
        write_cleaned_track_flac(&path, &samples).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 5);
        std::fs::write(&path, &bytes).unwrap();

        let back = read_track_16k_mono(&path).unwrap();
        // 16000 = 3 whole blocks of 4096 plus a tail of 3712, which is the cut frame.
        assert_eq!(back, samples[..3 * 4096]);
    }

    #[test]
    fn a_flac_with_no_whole_frame_is_an_error_not_an_empty_track() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.flac");
        write_cleaned_track_flac(&path, &on_grid(&tone(16_000, 0.1, 440.0))).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 5);
        std::fs::write(&path, &bytes).unwrap();

        assert!(read_track_16k_mono(&path).is_err());
    }

    #[test]
    fn a_stereo_or_non_16_bit_flac_is_rejected_by_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.flac");
        write_cleaned_track_flac(&path, &on_grid(&tone(16_000, 0.1, 440.0))).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        // STREAMINFO bytes 12..: rate(20) channels-1(3) bits-1(5). Rewrite to 2 channels, 24 bit.
        let at = 8 + 10;
        let rate = ((bytes[at] as u32) << 12)
            | ((bytes[at + 1] as u32) << 4)
            | (bytes[at + 2] as u32 >> 4);
        assert_eq!(rate, 16_000);
        bytes[at + 2] = (bytes[at + 2] & 0xF0) | (1 << 1) | 1;
        bytes[at + 3] = (bytes[at + 3] & 0x0F) | (7 << 4);
        std::fs::write(&path, &bytes).unwrap();

        let err = read_track_16k_mono(&path).unwrap_err().to_string();
        assert!(
            err.contains("2 channel(s)") && err.contains("24-bit"),
            "{err}"
        );
    }

    #[test]
    fn a_non_audio_file_is_still_refused_as_a_wav() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.wav");
        std::fs::write(&path, b"no").unwrap();

        assert!(matches!(read_track_16k_mono(&path), Err(Error::Wav { .. })));
    }

    /// The in-memory conversion is the file one, so audio imported from a foreign wav is the
    /// same audio it would have been had the recorder captured it at that rate.
    #[test]
    fn resampling_in_memory_agrees_with_resampling_from_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.wav");
        let samples = tone(48_000, 1.0, 440.0);
        write_wav(&path, 48_000, &samples);

        assert_eq!(
            resample_to_target(&samples, 48_000).unwrap(),
            read_track_16k_mono(&path).unwrap()
        );
    }

    #[test]
    fn a_header_claiming_no_sample_rate_is_refused_rather_than_divided_by() {
        let err = resample_to_target(&[0.1; 64], 0).unwrap_err().to_string();
        assert!(err.contains("sample rate of 0"), "{err}");
    }
}
