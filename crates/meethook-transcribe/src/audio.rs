//! Getting a recorded track into the one format Whisper accepts.
//!
//! whisper.cpp takes 16 kHz mono `f32` and nothing else. The recorder already writes mono
//! 32-bit float at whatever rate the device runs at, so sample rate is the only thing that
//! ever needs changing -- and the conversion is done streaming, because holding an hour of
//! 48 kHz audio *and* its 16 kHz result at once costs ~900 MB to save a few lines of code.

use std::path::Path;

use hound::{SampleFormat, WavReader};
use rubato::{FftFixedIn, Resampler};

use crate::{Error, Result};

/// The sample rate whisper.cpp requires. Not configurable: it is a property of the model.
pub const TARGET_RATE: u32 = 16_000;

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

    resample_to_target(reader, path, spec.sample_rate, frames)
}

fn resample_to_target(
    reader: WavReader<std::io::BufReader<std::fs::File>>,
    path: &Path,
    source_rate: u32,
    frames: usize,
) -> Result<Vec<f32>> {
    let mut resampler = FftFixedIn::<f32>::new(
        source_rate as usize,
        TARGET_RATE as usize,
        CHUNK_FRAMES,
        SUB_CHUNKS,
        1,
    )
    .map_err(|e| Error::Resample(e.to_string()))?;

    // The filter is linear-phase, so its output lags its input by a known, constant number
    // of frames. Dropping exactly that many leading output frames keeps the resampled track
    // aligned with the original rather than uniformly late.
    let delay = resampler.output_delay();
    let expected = (frames as u128 * u128::from(TARGET_RATE) / u128::from(source_rate)) as usize;

    let mut out: Vec<f32> = Vec::with_capacity(expected + delay + CHUNK_FRAMES);
    let mut chunk: Vec<f32> = Vec::with_capacity(CHUNK_FRAMES);

    for sample in reader.into_samples::<f32>() {
        chunk.push(sample.map_err(|e| Error::wav(path, e))?);
        if chunk.len() == CHUNK_FRAMES {
            let produced = resampler
                .process(&[&chunk], None)
                .map_err(|e| Error::Resample(e.to_string()))?;
            out.extend_from_slice(&produced[0]);
            chunk.clear();
        }
    }

    if !chunk.is_empty() {
        let produced = resampler
            .process_partial(Some(&[&chunk]), None)
            .map_err(|e| Error::Resample(e.to_string()))?;
        out.extend_from_slice(&produced[0]);
    }

    // One flush of silence pushes the samples still inside the filter out, so the last
    // fraction of a second of speech is not lost to the resampler's own latency.
    if frames > 0 {
        let flushed = resampler
            .process_partial::<Vec<f32>>(None, None)
            .map_err(|e| Error::Resample(e.to_string()))?;
        out.extend_from_slice(&flushed[0]);
    }

    out.drain(..delay.min(out.len()));
    out.truncate(expected);
    Ok(out)
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
}
