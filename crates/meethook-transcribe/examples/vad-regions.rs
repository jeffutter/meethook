//! Measures how much of a track holds speech, and how much a gate built from that would save.
//!
//! ```text
//! cargo run --release --example vad-regions -- ~/meethook/sessions/20260810-093047
//! cargo run --release --example vad-regions -- --threshold 0.3 --min-silence 0.5 some.wav
//! ```
//!
//! `--release` matters: a 43-minute track is ~81,000 sequential graph evaluations, and a debug
//! build turns seconds into minutes.
//!
//! # Why this exists
//!
//! A gate that skips silence before recognition has two numbers nobody can guess -- the
//! probability threshold and the minimum silence that ends a speech region -- and the only
//! other way to see their effect is to re-run `meethook transcribe` over a whole session. That
//! is a multi-minute loop per attempt, and the answer it prints ("115 turns") does not say
//! whether the gate kept the user's real speech or threw half of it away. This makes both
//! questions answerable in seconds, at any setting, without a rebuild.
//!
//! # What to read
//!
//! ```text
//! mic.cleaned.wav
//!   duration:     2603.4 s
//!   regions:      64
//!   speech:       171.2 s (6.6% of the track)
//!   gated:        171.2 s in 6 window(s) of 30 s
//!   whole track:  2603.4 s in 87 window(s)
//!   saved:        2432.2 s, 81 window(s)
//! ```
//!
//! The finding is a *contrast*, which is why a session directory measures three tracks and
//! prints them side by side at the end. The microphone track of a normal meeting should come
//! back as a small fraction of its duration; the speaker track as most of the meeting. Both
//! mic tracks are measured because `transcribe_session` recognises the *cleaned* one -- so that
//! is the honest measurement of what a gate would see -- while raw `mic.wav` beside it says how
//! much of the mic track's apparent speech was echo the canceller had already removed.
//!
//! The regions themselves are printed in full, so a run can be spot-checked against the
//! session's own `transcript.md`.
//!
//! # Getting the weights
//!
//! This fetches [`SILERO_VAD_MODEL`] itself, through the same [`ensure_model`] the real command
//! uses, so it works before `transcribe` has ever run. That is a departure from
//! `fetch-onnx-models`, deliberately: that example exists because 32 MB pulled behind
//! `cargo test` is a download nobody asked for, and 885 KB inside an explicitly invoked
//! diagnostic is not that case. `fetch-onnx-models` is left alone -- it is correct by its name
//! and its contents, and bolting a ggml file onto it would make the name a lie.

#[path = "support/mod.rs"]
mod support;

use std::path::{Path, PathBuf};

use meethook_models::ensure_model;
use meethook_transcribe::{
    SILERO_VAD_MODEL, SileroVad, SpeechRegion, TARGET_RATE, VadTuning, read_track_16k_mono,
};
use support::fail;

/// The window Whisper decodes at a time. Not configurable here: it is a property of the model,
/// and it is what turns "seconds of audio saved" into "decoder passes saved".
const WHISPER_WINDOW_S: f64 = 30.0;

/// The floor whisper.cpp merges regions under whatever `--min-silence` says. Printed rather
/// than enforced, so a sweep that stops responding below it reads as documented behaviour.
const MERGE_FLOOR_S: f64 = 0.2;

fn main() {
    let args = parse().unwrap_or_else(|message| {
        eprintln!("{message}");
        eprintln!(
            "usage: vad-regions [--root <dir>] [--threshold <p>] [--min-silence <s>]\n       \
             [--min-speech <s>] [--speech-pad <s>] [--max-speech <s>] <session-dir | wav-file>"
        );
        std::process::exit(2);
    });

    let tracks = tracks_of(&args.target);

    println!("{}", args.target.display());
    println!(
        "tuning:    threshold {:.2} (speech ends below {:.2}), min speech {:.3} s, \
         min silence {:.3} s, pad {:.3} s, max speech {}",
        args.tuning.threshold,
        // Hysteresis is whisper.cpp's, not this tool's, and one knob moving two edges is
        // surprising enough to spell out at the setting actually in use.
        (args.tuning.threshold - 0.15).max(0.01),
        args.tuning.min_speech_s,
        args.tuning.min_silence_s,
        args.tuning.speech_pad_s,
        match args.tuning.max_speech_s {
            Some(seconds) => format!("{seconds:.1} s"),
            None => "unbounded".to_string(),
        }
    );
    if args.tuning.min_silence_s < MERGE_FLOOR_S {
        println!(
            "note:      whisper.cpp merges regions separated by under {MERGE_FLOOR_S:.1} s \
             regardless, so min silence below that changes little"
        );
    }

    let model = install(&args.root);
    let mut vad = SileroVad::load(&model).unwrap_or_else(|e| fail(&e.to_string()));

    let mut measured: Vec<Measurement> = Vec::new();
    for track in &tracks {
        println!();
        // Each track is read, measured and dropped before the next one: three 43-minute tracks
        // held at once is ~500 MB for nothing.
        if let Some(measurement) = report(&mut vad, track, args.tuning) {
            measured.push(measurement);
        }
    }

    summarize(&measured);
}

/// What one track came back as, kept only so the tracks can be printed next to each other.
struct Measurement {
    name: String,
    duration_s: f64,
    regions: usize,
    speech_s: f64,
}

impl Measurement {
    fn share(&self) -> f64 {
        self.speech_s / self.duration_s.max(f64::MIN_POSITIVE)
    }
}

/// Measures one track and prints everything about it, or says why it could not be measured.
///
/// Every failure short of "the process cannot continue" is reported and returned from: with
/// three tracks the state of the others is still worth knowing, and one missing track is itself
/// a diagnosis.
fn report(vad: &mut SileroVad, track: &Path, tuning: VadTuning) -> Option<Measurement> {
    let name = Path::new(track.file_name().unwrap_or(track.as_os_str()))
        .display()
        .to_string();
    println!("{name}");

    if !track.exists() {
        println!("  missing:      no file at {}", track.display());
        return None;
    }
    let audio = match read_track_16k_mono(track) {
        Ok(audio) => audio,
        Err(e) => {
            println!("  unreadable:   {e}");
            return None;
        }
    };

    let duration_s = audio.len() as f64 / f64::from(TARGET_RATE);
    println!(
        "  duration:     {duration_s:.1} s ({} samples)",
        audio.len()
    );
    if audio.is_empty() {
        println!("  regions:      none; the track holds no samples");
        return None;
    }

    let started = std::time::Instant::now();
    let regions = match vad.speech_regions(&audio, tuning) {
        Ok(regions) => regions,
        Err(e) => {
            println!("  failed:       {e}");
            return None;
        }
    };
    let elapsed = started.elapsed();

    // Folded rather than summed: `Sum` for `f64` folds from `-0.0`, so a track with no regions
    // at all -- the interesting case for this tool -- would report "-0.0 s".
    let speech_s: f64 = regions
        .iter()
        .map(SpeechRegion::duration_s)
        .fold(0.0, |total, seconds| total + seconds);
    let share = speech_s / duration_s.max(f64::MIN_POSITIVE);
    println!("  detected in:  {:.1} s", elapsed.as_secs_f64());
    println!("  regions:      {}", regions.len());
    println!(
        "  speech:       {speech_s:.1} s ({:.1}% of the track)",
        share * 100.0
    );

    // The regions are spliced into one buffer and decoded once, so the gated cost is one
    // ceiling over the total. Decoding each region separately would instead be the sum of the
    // per-region ceilings -- a different and much larger number, since a 1 s region still costs
    // a whole window.
    let gated_windows = windows(speech_s);
    let whole_windows = windows(duration_s);
    println!("  gated:        {speech_s:.1} s in {gated_windows} window(s) of 30 s");
    println!("  whole track:  {duration_s:.1} s in {whole_windows} window(s)");
    println!(
        "  saved:        {:.1} s, {} window(s) ({:.1}% less audio decoded)",
        duration_s - speech_s,
        whole_windows.saturating_sub(gated_windows),
        (1.0 - share) * 100.0
    );

    if regions.is_empty() {
        println!("  no regions at this tuning");
    } else {
        for region in &regions {
            println!(
                "  {:>9.2} -> {:>9.2}  ({:>6.2} s)",
                region.start_s,
                region.end_s,
                region.duration_s()
            );
        }
    }

    Some(Measurement {
        name,
        duration_s,
        regions: regions.len(),
        speech_s,
    })
}

/// The tracks side by side. The finding this tool exists to produce is a contrast, and a
/// contrast has to be printed next to itself to be read.
fn summarize(measured: &[Measurement]) {
    println!();
    println!("summary");
    if measured.is_empty() {
        println!("  nothing could be measured");
        return;
    }
    println!(
        "  {:<18} {:>10} {:>8} {:>10} {:>8} {:>8}",
        "track", "duration", "regions", "speech", "share", "windows"
    );
    for m in measured {
        println!(
            "  {:<18} {:>9.1}s {:>8} {:>9.1}s {:>7.1}% {:>4} / {:<3}",
            m.name,
            m.duration_s,
            m.regions,
            m.speech_s,
            m.share() * 100.0,
            windows(m.speech_s),
            windows(m.duration_s)
        );
    }
}

/// How many 30 s Whisper windows a stretch of audio costs to decode.
fn windows(seconds: f64) -> usize {
    (seconds / WHISPER_WINDOW_S).ceil().max(0.0) as usize
}

/// The tracks a target names.
///
/// A session directory means three: raw `mic.wav`, the `mic.cleaned.wav` recognition actually
/// reads, and `speaker.wav`. Any other path is used as given.
fn tracks_of(target: &Path) -> Vec<PathBuf> {
    if target.is_dir() {
        vec![
            target.join("mic.wav"),
            target.join("mic.cleaned.wav"),
            target.join("speaker.wav"),
        ]
    } else {
        vec![target.to_path_buf()]
    }
}

/// Fetches the weights if they are not installed, verified against the embedded sha256.
fn install(root: &Path) -> PathBuf {
    let models_dir = root.join("models");
    // One line when a download happens, and none on a cache hit: 885 KB does not need a
    // percentage.
    let mut announced = false;
    let mut progress = |_done: u64, _total: u64| {
        if !announced {
            announced = true;
            println!("fetching {} ...", SILERO_VAD_MODEL.file_name);
        }
    };
    ensure_model(&models_dir, &SILERO_VAD_MODEL, &mut progress)
        .unwrap_or_else(|e| fail(&format!("could not install the VAD weights: {e}")))
}

struct Args {
    root: PathBuf,
    target: PathBuf,
    tuning: VadTuning,
}

/// Hand-rolled rather than clap, matching the other examples in this crate: a diagnostic must
/// never be the reason a build breaks.
///
/// Every default comes from [`VadTuning::default`], which is whisper.cpp's own set, and the
/// resolved tuning is printed before any number derived from it -- a measurement whose settings
/// are not on the page is not evidence.
fn parse() -> Result<Args, String> {
    let mut root = std::env::var_os("MEETHOOK_ROOT").map(PathBuf::from);
    let mut target: Option<PathBuf> = None;
    let mut tuning = VadTuning::default();

    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |flag: &str| {
            args.next()
                .and_then(|v| v.into_string().ok())
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        let number = |flag: &str, raw: String| {
            raw.parse::<f64>()
                .map_err(|_| format!("{flag} needs a number, not {raw:?}"))
        };
        match arg.to_str() {
            Some("--root") => root = Some(PathBuf::from(value("--root")?)),
            Some("--threshold") => {
                tuning.threshold = number("--threshold", value("--threshold")?)? as f32;
            }
            Some("--min-silence") => {
                tuning.min_silence_s = number("--min-silence", value("--min-silence")?)?;
            }
            Some("--min-speech") => {
                tuning.min_speech_s = number("--min-speech", value("--min-speech")?)?;
            }
            Some("--speech-pad") => {
                tuning.speech_pad_s = number("--speech-pad", value("--speech-pad")?)?;
            }
            Some("--max-speech") => {
                tuning.max_speech_s = Some(number("--max-speech", value("--max-speech")?)?);
            }
            Some(flag) if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            _ if target.is_none() => target = Some(PathBuf::from(arg)),
            _ => return Err("only one session directory or wav file may be given".to_string()),
        }
    }

    Ok(Args {
        // A `~/meethook` fallback, unlike `speaker-trials`: nothing here creates a session
        // directory, and the only thing the root is used for is where to keep 885 KB of
        // weights.
        root: root
            .or_else(|| std::env::home_dir().map(|home| home.join("meethook")))
            .ok_or("could not determine the home directory; pass --root or set $MEETHOOK_ROOT")?,
        target: target.ok_or("no session directory or wav file was given")?,
        tuning,
    })
}
