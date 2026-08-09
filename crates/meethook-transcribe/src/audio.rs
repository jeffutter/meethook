//! Getting a recorded track into the one format Whisper accepts.
//!
//! whisper.cpp takes 16 kHz mono `f32` and nothing else. The recorder already writes mono
//! 32-bit float at whatever rate the device runs at, so sample rate is the only thing that
//! ever needs changing -- and the conversion is done streaming, because holding an hour of
//! 48 kHz audio *and* its 16 kHz result at once costs ~900 MB to save a few lines of code.

use std::path::Path;

use hound::{SampleFormat, WavReader, WavSpec};
use meethook_session::write_atomic_with;
use rubato::{FftFixedIn, Resampler};

use crate::{Error, Result};

/// The sample rate whisper.cpp requires. Not configurable: it is a property of the model.
pub const TARGET_RATE: u32 = 16_000;

/// The header on every track meethook writes for itself.
///
/// One spelling, because a track written at a different rate or width than
/// [`read_track_16k_mono`] insists on would be rejected by the very next thing to open it.
const TRACK_SPEC: WavSpec = WavSpec {
    channels: 1,
    sample_rate: TARGET_RATE,
    bits_per_sample: 32,
    sample_format: SampleFormat::Float,
};

/// Input frames handed to the resampler at a time.
///
/// Together with `SUB_CHUNKS` this sets the internal FFT size, and therefore both the
/// anti-aliasing filter's length and its group delay. These values give a filter of a few
/// hundred taps and a delay of a few milliseconds -- ample rejection for a 3:1 decimation,
/// and a delay far below the 10 ms granularity Whisper reports timestamps at.
const CHUNK_FRAMES: usize = 4096;
const SUB_CHUNKS: usize = 16;

/// Reads a mono float WAV and returns it as 16 kHz mono `f32`.
///
/// A file that is not mono 32-bit float is rejected by name and header rather than
/// reinterpreted: silently misreading a stereo or integer WAV produces audio that sounds
/// like noise to the model and like a transcription bug to everyone else.
///
/// A zero-length track is not an error. A session where nobody spoke is a real session, and
/// it still has to produce transcript files so a rerun skips it.
pub fn read_track_16k_mono(path: &Path) -> Result<Vec<f32>> {
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
    if spec.sample_rate == TARGET_RATE {
        let mut samples = Vec::with_capacity(frames);
        for sample in reader.into_samples::<f32>() {
            samples.push(sample.map_err(|e| Error::wav(path, e))?);
        }
        return Ok(samples);
    }

    let mut resample = Resample::new(spec.sample_rate, frames)?;
    for sample in reader.into_samples::<f32>() {
        resample.push(sample.map_err(|e| Error::wav(path, e))?)?;
    }
    resample.finish()
}

/// Converts audio already held in memory to [`TARGET_RATE`].
///
/// The in-memory twin of the loop in [`read_track_16k_mono`], for callers whose samples did
/// not come from a file this crate is willing to open -- see [`crate::import`]. Both go
/// through [`Resample`], so the filter length and the delay compensation are decided once.
pub(crate) fn resample_to_target(samples: &[f32], source_rate: u32) -> Result<Vec<f32>> {
    if source_rate == TARGET_RATE {
        return Ok(samples.to_vec());
    }

    let mut resample = Resample::new(source_rate, samples.len())?;
    for sample in samples {
        resample.push(*sample)?;
    }
    resample.finish()
}

/// Writes a track in the one format everything downstream reads: 16 kHz mono float,
/// atomically.
///
/// Atomic because every one of these files is something another command classifies a session
/// by; a reader must see a whole track or no file, never a truncated one.
pub(crate) fn write_track_16k_mono(path: &Path, audio: &[f32]) -> Result<()> {
    write_atomic_with(path, |file| {
        // Buffered: hound writes each sample straight through, and an hour of audio is 57
        // million of them.
        let mut writer = hound::WavWriter::new(std::io::BufWriter::new(file), TRACK_SPEC)
            .map_err(|e| Error::wav(path, e))?;
        for sample in audio {
            writer
                .write_sample(*sample)
                .map_err(|e| Error::wav(path, e))?;
        }
        writer.finalize().map_err(|e| Error::wav(path, e))
    })
}

/// A fixed-ratio conversion to [`TARGET_RATE`], fed one sample at a time.
///
/// Push-shaped rather than slice-shaped because the caller that matters is reading a file:
/// holding an hour of 48 kHz audio *and* its 16 kHz result at once costs ~900 MB, which is
/// the whole reason the read loop above is streaming.
struct Resample {
    resampler: FftFixedIn<f32>,
    /// Input frames still waiting for a full [`CHUNK_FRAMES`] to hand the resampler.
    chunk: Vec<f32>,
    out: Vec<f32>,
    /// Leading output frames belonging to the filter's own group delay.
    delay: usize,
    /// How many output frames the conversion should yield once that delay is dropped.
    expected: usize,
    pushed: usize,
}

impl Resample {
    /// `frames` is the input length, used to size the output exactly. It is a promise about
    /// how much will be pushed, not a limit: pushing fewer simply yields a shorter track.
    fn new(source_rate: u32, frames: usize) -> Result<Self> {
        // Checked here rather than left to rubato, both because a rate of zero would divide
        // by zero below and because "a header claiming 0 Hz" is a diagnosis, where a
        // resampler-internal message would not be.
        if source_rate == 0 {
            return Err(Error::Resample(
                "the file's header reports a sample rate of 0".to_string(),
            ));
        }

        let resampler = FftFixedIn::<f32>::new(
            source_rate as usize,
            TARGET_RATE as usize,
            CHUNK_FRAMES,
            SUB_CHUNKS,
            1,
        )
        .map_err(|e| Error::Resample(e.to_string()))?;

        // The filter is linear-phase, so its output lags its input by a known, constant
        // number of frames. Dropping exactly that many leading output frames keeps the
        // resampled track aligned with the original rather than uniformly late.
        let delay = resampler.output_delay();
        let expected =
            (frames as u128 * u128::from(TARGET_RATE) / u128::from(source_rate)) as usize;

        Ok(Resample {
            resampler,
            chunk: Vec::with_capacity(CHUNK_FRAMES),
            out: Vec::with_capacity(expected + delay + CHUNK_FRAMES),
            delay,
            expected,
            pushed: 0,
        })
    }

    fn push(&mut self, sample: f32) -> Result<()> {
        self.pushed += 1;
        self.chunk.push(sample);
        if self.chunk.len() == CHUNK_FRAMES {
            let produced = self
                .resampler
                .process(&[&self.chunk], None)
                .map_err(|e| Error::Resample(e.to_string()))?;
            self.out.extend_from_slice(&produced[0]);
            self.chunk.clear();
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<f32>> {
        if !self.chunk.is_empty() {
            let produced = self
                .resampler
                .process_partial(Some(&[&self.chunk]), None)
                .map_err(|e| Error::Resample(e.to_string()))?;
            self.out.extend_from_slice(&produced[0]);
        }

        // One flush of silence pushes the samples still inside the filter out, so the last
        // fraction of a second of speech is not lost to the resampler's own latency.
        if self.pushed > 0 {
            let flushed = self
                .resampler
                .process_partial::<Vec<f32>>(None, None)
                .map_err(|e| Error::Resample(e.to_string()))?;
            self.out.extend_from_slice(&flushed[0]);
        }

        self.out.drain(..self.delay.min(self.out.len()));
        self.out.truncate(self.expected);
        Ok(self.out)
    }
}

#[cfg(test)]
mod tests {
    use hound::{WavSpec, WavWriter};

    use super::*;

    fn write_wav(path: &Path, rate: u32, samples: &[f32]) {
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

    fn write_wav_spec(path: &Path, spec: WavSpec, samples: &[f32]) {
        let mut writer = WavWriter::create(path, spec).unwrap();
        for sample in samples {
            match spec.sample_format {
                SampleFormat::Float => writer.write_sample(*sample).unwrap(),
                SampleFormat::Int => writer.write_sample(*sample as i16).unwrap(),
            }
        }
        writer.finalize().unwrap();
    }

    fn tone(rate: u32, seconds: f32, hz: f32) -> Vec<f32> {
        let n = (rate as f32 * seconds) as usize;
        (0..n)
            .map(|i| (i as f32 / rate as f32 * hz * std::f32::consts::TAU).sin() * 0.5)
            .collect()
    }

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
        let path = dir.path().join("mic.cleaned.wav");
        let samples = tone(16_000, 0.5, 440.0);

        write_track_16k_mono(&path, &samples).unwrap();

        assert_eq!(read_track_16k_mono(&path).unwrap(), samples);
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
