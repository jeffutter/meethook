//! Reports whether each track of a recording actually captured audio.
//!
//! ```text
//! cargo run --example track-levels ~/meethook/sessions/20260809-025745
//! cargo run --example track-levels some-other-recording.wav
//! ```
//!
//! A session directory means **both** its tracks, because the failure worth catching is the
//! one where the microphone is alive and the system-audio capture is not; any other path is
//! used as given.
//!
//! Run this in the ten seconds after a call ends, before anyone disperses. It needs no ONNX
//! models, does no segmentation and no embedding, and touches nothing but the two wav files,
//! so it works on a fresh checkout where `fetch-onnx-models` has never run. That is the
//! whole point: `cluster-speaker-track` and `segment-speaker-track` can answer "is there
//! anything here" only after a full model download and a segmentation pass, and they answer
//! it as "0 turns", which is what a dead track and a track the model found nothing in both
//! look like.
//!
//! What to read:
//!
//! ```text
//! mic.wav
//!   format:       1 ch, 48000 Hz, 32-bit float
//!   duration:     10.0 s (481536 samples)
//!   peak:         0.573 (-4.8 dBFS)
//!   above floor:  99.8% of samples
//!   longest run:  10.0 s
//! ```
//!
//! Peak on its own does not settle it. A track carrying two UI chimes and nothing else
//! peaks around 0.57 -- the same neighbourhood as speech -- while being silent for 99% of
//! its length, so it is the fraction and the run length that separate "somebody talked"
//! from "the interface beeped twice".
//!
//! Nothing here rejects a file for having the wrong header. A track that is not the mono
//! 32-bit float the recorder writes is itself a finding, so the spec is printed first and
//! unconditionally, and integer formats are measured anyway by scaling to the same ±1.0
//! range. Only a format `hound` cannot decode at all declines to produce numbers.

use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavReader, WavSpec};
use meethook_transcribe::{LevelSummary, RUN_BRIDGE_S, SILENCE_FLOOR};

fn main() {
    let Some(target) = std::env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: track-levels <session-dir | wav-file>");
        std::process::exit(2);
    };

    println!("{}", target.display());
    println!(
        "floor {SILENCE_FLOOR:e} ({:.0} dBFS); runs bridge gaps under {:.0} ms",
        20.0 * f64::from(SILENCE_FLOOR).log10(),
        RUN_BRIDGE_S * 1000.0
    );

    let tracks: Vec<PathBuf> = if target.is_dir() {
        // Absent metadata is noted rather than fatal. This is a raw-capture inspection, and
        // an interrupted session -- which is exactly when someone reaches for it -- is the
        // case where `session.json` was never written.
        let metadata = target.join("session.json");
        if !metadata.exists() {
            println!("note: no session.json in this directory; measuring the tracks anyway");
        }
        vec![target.join("mic.wav"), target.join("speaker.wav")]
    } else {
        vec![target]
    };

    for track in &tracks {
        println!();
        report(track);
    }
}

/// Prints one track's spec and levels. Every failure short of "the process cannot continue"
/// is reported and returned from, because with two tracks the state of the other one is
/// still worth knowing -- and one missing track *is* the diagnosis.
fn report(track: &Path) {
    let name = track.file_name().unwrap_or(track.as_os_str());
    println!("{}", Path::new(name).display());

    if !track.exists() {
        println!("  missing:      no file at {}", track.display());
        return;
    }

    let reader = match WavReader::open(track) {
        Ok(reader) => reader,
        Err(e) => {
            println!("  unreadable:   {e}");
            return;
        }
    };
    let spec = reader.spec();
    println!(
        "  format:       {} ch, {} Hz, {}-bit {}",
        spec.channels,
        spec.sample_rate,
        spec.bits_per_sample,
        match spec.sample_format {
            SampleFormat::Float => "float",
            SampleFormat::Int => "int",
        }
    );

    let samples = match read_samples(reader, spec) {
        Ok(samples) => samples,
        Err(reason) => {
            println!("  no levels:    {reason}");
            return;
        }
    };

    // Interleaved samples are measured as one stream, at the rate they arrive rather than
    // the frame rate, so durations stay correct for a multi-channel file without pretending
    // the channels were separated.
    let rate = spec.sample_rate * u32::from(spec.channels.max(1));
    let summary = LevelSummary::measure(&samples, rate);

    if summary.samples == 0 {
        println!("  duration:     empty (0 samples); the data chunk holds nothing");
        return;
    }
    if spec.sample_rate == 0 {
        println!("  duration:     unknown; the header claims a sample rate of 0");
    } else {
        println!(
            "  duration:     {:.2} s ({} samples)",
            summary.duration_s(),
            summary.samples
        );
    }

    let dbfs = summary.peak_dbfs();
    if dbfs.is_infinite() {
        println!("  peak:         0.0 (digital silence)");
    } else {
        println!("  peak:         {:.4} ({dbfs:.1} dBFS)", summary.peak);
    }
    println!(
        "  above floor:  {:.1}% of samples ({} of {})",
        summary.above_fraction() * 100.0,
        summary.above_floor,
        summary.samples
    );
    println!(
        // Three decimals, unlike the duration: the figure that distinguishes a chime from a
        // sentence lands in the milliseconds, and `0.00 s` would erase the distinction
        // between a click and a syllable.
        "  longest run:  {:.3} s ({} samples)",
        summary.longest_run_s(),
        summary.longest_run
    );
}

/// Decodes a track to the ±1.0 float scale whatever its header says.
///
/// Integer formats are scaled by their own full scale rather than refused, so a recorder
/// that started writing 16-bit PCM still produces comparable numbers next to the header that
/// reveals it. The error case is a file `hound` will not decode at all, where declining is
/// better than a number derived from a guess.
fn read_samples(
    reader: WavReader<std::io::BufReader<std::fs::File>>,
    spec: WavSpec,
) -> Result<Vec<f32>, String> {
    let mut samples = Vec::with_capacity(reader.len() as usize);
    match spec.sample_format {
        SampleFormat::Float => {
            for sample in reader.into_samples::<f32>() {
                samples.push(sample.map_err(|e| e.to_string())?);
            }
        }
        SampleFormat::Int => {
            let full_scale = (1i64 << (spec.bits_per_sample.max(1) - 1)) as f32;
            for sample in reader.into_samples::<i32>() {
                samples.push(sample.map_err(|e| e.to_string())? as f32 / full_scale);
            }
        }
    }
    Ok(samples)
}
