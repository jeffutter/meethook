//! Builds a meethook session directory out of wav files meethook did not record.
//!
//! ```text
//! MEETHOOK_ROOT=/tmp/calibration \
//!   cargo run --release --example build-session -- alice-part1.wav alice-part2.wav
//!
//! cargo run --release --example build-session -- --root /tmp/calibration alice.wav
//! cargo run --release --example build-session -- --root /tmp/calibration --mic near.wav far.wav
//! ```
//!
//! Why this exists: the enrolled-speaker threshold has to be calibrated against one person's
//! stored reference measured on a *different* recording of that person, and against somebody
//! else. Nothing about that measurement needs meethook to have done the capturing -- but until
//! this existed, the only way to get a reference into `speakers.json` was to hold a meeting,
//! transcribe it and enroll from it, so every attempt cost a recording sitting.
//!
//! What it writes is an ordinary session directory. `meethook transcribe` and `meethook enroll`
//! then run over it unchanged, which is the point: the reference that comes out was produced by
//! the real diarization, clustering and enrollment path rather than hand-written into JSON.
//!
//! ```text
//! meethook transcribe --root /tmp/calibration
//! printf 'Alice\n' | meethook enroll --root /tmp/calibration
//! MEETHOOK_ROOT=/tmp/calibration \
//!   cargo run --release --example cluster-speaker-track /tmp/calibration/sessions/<second-id>
//! ```
//!
//! The root is `--root` or `$MEETHOOK_ROOT`, and **there is deliberately no `~/meethook`
//! fallback**, unlike `cluster-speaker-track`. A tool whose whole purpose is to create session
//! directories should not be able to create them among a real set of recordings because an
//! environment variable was unset.
//!
//! The audio goes on `speaker.wav` and `mic.wav` gets a second of digital silence, because
//! diarization and embedding only ever run on the speaker track. `--mic` is there for the
//! caller who genuinely has a two-track recording; no measurement needs it.
//!
//! Levels are printed for both tracks it wrote. Reading them takes a second and turns "the
//! measurement came back empty" into "the input was dead" *before* a whisper pass is paid for
//! rather than after.

#[path = "support/mod.rs"]
mod support;

use std::path::PathBuf;

use meethook_session::Paths;
use meethook_transcribe::{BuiltSession, MIC_SILENCE_S, SPLICE_GAP_S, build_session};
use support::session_prep::{converted, levels};

fn main() {
    let args = parse().unwrap_or_else(|message| {
        eprintln!("{message}");
        eprintln!(
            "usage: build-session [--root <dir>] [--mic <wav>]... <wav>...\n       \
             the root may also come from $MEETHOOK_ROOT; there is no default"
        );
        std::process::exit(2);
    });

    let paths = Paths::new(&args.root);
    let built = build_session(&paths, &args.speaker, &args.mic).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    report(&paths, &args, &built);
}

struct Args {
    root: PathBuf,
    speaker: Vec<PathBuf>,
    mic: Vec<PathBuf>,
}

/// Hand-rolled rather than clap: the examples in this crate take no dependency the library
/// does not, so that a diagnostic can never be the reason a build breaks.
fn parse() -> Result<Args, String> {
    let mut root = std::env::var_os("MEETHOOK_ROOT").map(PathBuf::from);
    let mut speaker = Vec::new();
    let mut mic = Vec::new();

    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |flag: &str| {
            args.next()
                .map(PathBuf::from)
                .ok_or_else(|| format!("{flag} needs a path"))
        };
        match arg.to_str() {
            Some("--root") => root = Some(value("--root")?),
            Some("--mic") => mic.push(value("--mic")?),
            Some(flag) if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            _ => speaker.push(PathBuf::from(arg)),
        }
    }

    let root = root.ok_or("no root: pass --root or set $MEETHOOK_ROOT")?;
    if speaker.is_empty() {
        return Err("no source wav files were given".to_string());
    }
    Ok(Args { root, speaker, mic })
}

fn report(paths: &Paths, args: &Args, built: &BuiltSession) {
    println!("root: {}", paths.root().display());

    println!(
        "\nspeaker.wav, from {} file(s):",
        built.speaker_sources.len()
    );
    for source in &built.speaker_sources {
        println!("  {}", converted(source));
    }
    if built.speaker_sources.len() > 1 {
        println!("  spliced with {SPLICE_GAP_S:.2} s of silence between sources");
    }

    if built.mic_sources.is_empty() {
        println!("\nmic.wav: {MIC_SILENCE_S:.1} s of digital silence (no local track supplied)");
    } else {
        println!("\nmic.wav, from {} file(s):", built.mic_sources.len());
        for source in &built.mic_sources {
            println!("  {}", converted(source));
        }
    }

    println!("\nwrote {}", built.paths.dir().display());
    levels("speaker.wav", &built.speaker);
    levels("mic.wav", &built.mic);

    // The next two commands, spelled out with this root already in them, because getting the
    // root wrong is the one mistake that reaches `~/meethook`.
    let root = paths.root().display();
    println!("\nnext:");
    println!("  meethook transcribe --root {root} {}", built.id);
    println!("  printf 'Alice\\n' | meethook enroll --root {root}");
    if !args.mic.is_empty() {
        println!(
            "  note: a supplied mic track is transcribed but never diarized, \
             so it cannot reach clustering or enrollment"
        );
    }
}
