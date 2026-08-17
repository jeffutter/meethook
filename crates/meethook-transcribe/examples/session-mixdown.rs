//! Renders one session's mixdown at every setting worth arguing about, so the argument can be
//! settled with headphones instead of a table.
//!
//! ```text
//! cargo run --example session-mixdown -- ~/meethook/sessions/20260809-025745 --out /tmp/mixdown
//! ```
//!
//! `transcribe` writes exactly one `meeting.opus`, at one bitrate, from one source, with the
//! two voices at one pan width. Every one of those is a judgement about how a meeting sounds
//! to a person, and nothing in this repository can make it: a test can prove the file decodes
//! and that the channels are not hard-panned, and it cannot hear comb filtering, or notice
//! that 24 kbps turns a quiet participant to gravel. So this writes the whole grid out to a
//! directory, one file per cell, named after its settings.
//!
//! What each arm is for:
//!
//! - **bitrate** (`*-16k-NNkbps-pan30.opus`) -- the headline question. Listen down from 64
//!   until a voice on the far end stops being comfortable, then take the step above it.
//! - **pan** (`cleaned-16k-32kbps-panNN.opus`) -- `pan00` is mono-in-stereo, `pan100` is one
//!   track per ear, `pan30` is what ships. The question is whether the local voice is
//!   *placeable* without either ear ever working alone.
//! - **source** (`raw-...` against `cleaned-...`) -- `transcribe` mixes the echo-cancelled mic
//!   track, since the raw one carries the far end back in through the microphone, delayed.
//!   The risk in that choice is the canceller having chewed something off the local voice, and
//!   this pair is where that would be heard.
//! - **rate** (`*-48k-*`) -- the mix is built from the 16 kHz tracks transcription already
//!   holds in memory. This arm asks whether the band above 8 kHz is worth a second pass over
//!   the originals. Written only when both source files are already at a rate Opus accepts and
//!   agree on it; when they do not, that is said and the arm is skipped rather than resampled
//!   here, because a resampler this file invented would be the thing under test.
//! - **`mix-16k.wav`** -- the encoder's own input, uncompressed. Every judgement above is
//!   against this, not against memory of the meeting.
//!
//! Both offsets come from `meethook_transcribe`'s own accessors, so every file here sits on
//! the timeline `transcript.md` describes.
//!
//! Like the other diagnostics in this directory, this takes no dependency the library does not
//! already have.

use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavReader, WavSpec};
use meethook_session::{SessionMetadata, SessionPaths};
use meethook_transcribe::mixdown::{self, Source};
use meethook_transcribe::{
    TARGET_RATE, mic_offset_seconds, read_track_16k_mono, speaker_offset_seconds,
};

/// The bitrates worth hearing, in kbps. Spans the usual speech range and a step past it in
/// both directions, so the chosen value is bracketed rather than merely plausible.
const BITRATES_KBPS: [u32; 5] = [16, 24, 32, 48, 64];

/// Pan positions, as hundredths, for the placement judgement. 0 is centre, 100 is hard.
const PANS: [u32; 3] = [0, 30, 100];

/// The bitrate the pan, source and rate arms are all rendered at, so those comparisons vary
/// one thing.
const REFERENCE_KBPS: u32 = 32;

fn main() {
    let mut args = std::env::args_os().skip(1);
    let Some(dir) = args.next().map(PathBuf::from) else {
        usage();
    };
    let out = match (args.next(), args.next()) {
        (Some(flag), Some(path)) if flag == "--out" => PathBuf::from(path),
        (None, None) => dir.join("mixdown-candidates"),
        _ => usage(),
    };

    let paths = SessionPaths::new(&dir);
    let metadata = read_metadata(&paths);
    let (mic_offset, speaker_offset) = (
        mic_offset_seconds(&metadata).unwrap(),
        speaker_offset_seconds(&metadata).unwrap(),
    );

    std::fs::create_dir_all(&out).unwrap();
    println!("{} -> {}", dir.display(), out.display());
    println!("offsets: mic {mic_offset:.3} s, speaker {speaker_offset:.3} s");
    println!(
        "pan {:.0}/100, {} kbps is what ships",
        mixdown::PAN_POSITION * 100.0,
        mixdown::BITRATE_BPS / 1000
    );

    // The cleaned track if `transcribe` has already run over this session, the raw one
    // otherwise -- and say which, because the whole source arm below is about that difference.
    let raw_mic = read_track_16k_mono(&paths.mic_wav()).unwrap();
    let cleaned_mic = match read_track_16k_mono(&paths.mic_cleaned_wav()) {
        Ok(track) => track,
        Err(e) => {
            println!("note: no mic.cleaned.wav ({e}); the `cleaned` arm repeats the raw track");
            raw_mic.clone()
        }
    };
    let speaker = read_track_16k_mono(&paths.speaker_wav()).unwrap_or_default();

    let cleaned = |pan: f32| -> Vec<f32> {
        mixdown::mix(
            &sources(&cleaned_mic, mic_offset, &speaker, speaker_offset, pan),
            TARGET_RATE,
        )
    };

    // The encoder's input, so every judgement below has something uncompressed to be a
    // judgement against.
    let reference = cleaned(mixdown::PAN_POSITION);
    write_wav(&out.join("mix-16k.wav"), &reference, TARGET_RATE);

    for kbps in BITRATES_KBPS {
        emit(&out, "cleaned", TARGET_RATE, kbps, 30, &reference);
    }

    for pan in PANS {
        if pan == 30 {
            continue; // already written by the bitrate arm
        }
        let mix = cleaned(pan as f32 / 100.0);
        emit(&out, "cleaned", TARGET_RATE, REFERENCE_KBPS, pan, &mix);
    }

    let raw = mixdown::mix(
        &sources(
            &raw_mic,
            mic_offset,
            &speaker,
            speaker_offset,
            mixdown::PAN_POSITION,
        ),
        TARGET_RATE,
    );
    emit(&out, "raw", TARGET_RATE, REFERENCE_KBPS, 30, &raw);

    match native_rate(&paths) {
        Ok(rate) => {
            let mic = read_native(&paths.mic_wav());
            let speaker = read_native(&paths.speaker_wav());
            let mix = mixdown::mix(
                &sources(
                    &mic,
                    mic_offset,
                    &speaker,
                    speaker_offset,
                    mixdown::PAN_POSITION,
                ),
                rate,
            );
            // The raw mic track, necessarily: `mic.cleaned.wav` only exists at 16 kHz, so this
            // arm answers "is the extra band worth it" with the echo still in. Read it as an
            // upper bound on what a native-rate mix would sound like, not as a candidate.
            for kbps in [REFERENCE_KBPS, 48] {
                emit(&out, "raw", rate, kbps, 30, &mix);
            }
        }
        Err(why) => println!("note: skipping the native-rate arm: {why}"),
    }
}

fn usage() -> ! {
    eprintln!("usage: session-mixdown <session-dir> [--out <dir>]");
    std::process::exit(2);
}

/// The two tracks as [`Source`]s, panned symmetrically about centre.
fn sources<'a>(
    mic: &'a [f32],
    mic_offset: f64,
    speaker: &'a [f32],
    speaker_offset: f64,
    pan: f32,
) -> [Source<'a>; 2] {
    [
        Source {
            samples: mic,
            offset_s: mic_offset,
            pan: -pan,
        },
        Source {
            samples: speaker,
            offset_s: speaker_offset,
            pan,
        },
    ]
}

/// Encodes one cell of the grid, naming the file after the settings that produced it and
/// printing what it cost.
fn emit(out: &Path, source: &str, rate: u32, kbps: u32, pan: u32, mix: &[f32]) {
    let name = format!("{source}-{}k-{kbps}kbps-pan{pan:02}.opus", rate / 1000);
    let path = out.join(&name);
    mixdown::write(&path, mix, rate, kbps * 1000).unwrap();

    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let seconds = mix.len() as f64 / 2.0 / f64::from(rate);
    println!(
        "{name:<34} {:>8.1} MB  {:>6.1} MB/hour",
        bytes as f64 / 1e6,
        bytes as f64 / 1e6 * 3600.0 / seconds.max(1e-9),
    );
}

fn read_metadata(paths: &SessionPaths) -> SessionMetadata {
    SessionMetadata::read(&paths.session_json()).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    })
}

/// The rate both source tracks are already at, when that is a rate Opus takes.
///
/// The mixdown cannot combine two tracks at different rates and nothing here resamples, so
/// this is a precondition rather than a preference.
fn native_rate(paths: &SessionPaths) -> Result<u32, String> {
    let rate = |path: PathBuf| -> Result<u32, String> {
        WavReader::open(&path)
            .map(|reader| reader.spec().sample_rate)
            .map_err(|e| format!("{}: {e}", path.display()))
    };
    let mic = rate(paths.mic_wav())?;
    let speaker = rate(paths.speaker_wav())?;
    if mic != speaker {
        return Err(format!("mic is {mic} Hz and speaker is {speaker} Hz"));
    }
    if ![8_000, 12_000, 16_000, 24_000, 48_000].contains(&mic) {
        return Err(format!("{mic} Hz is not a rate Opus encodes"));
    }
    if mic == TARGET_RATE {
        return Err("the tracks are already at 16 kHz".to_string());
    }
    Ok(mic)
}

/// A mono float track at whatever rate it was recorded at.
fn read_native(path: &Path) -> Vec<f32> {
    WavReader::open(path)
        .unwrap()
        .into_samples::<f32>()
        .map(|s| s.unwrap())
        .collect()
}

/// The interleaved mix, as a stereo WAV, so the encoder can be judged against its own input.
fn write_wav(path: &Path, stereo: &[f32], rate: u32) {
    let spec = WavSpec {
        channels: 2,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    // hound directly rather than `meethook_session::wav`: that helper exists to correct the
    // channel mask of a *mono* header, and a stereo one hound writes is already right.
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for sample in stereo {
        writer.write_sample(*sample).unwrap();
    }
    writer.finalize().unwrap();
    println!("{:<34} (the encoder's input)", "mix-16k.wav");
}
