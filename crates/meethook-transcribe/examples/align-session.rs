//! Reports what alignment and echo cancellation actually did for one session, without running
//! the rest of the transcription pipeline.
//!
//! ```text
//! cargo run --example align-session -- ~/meethook/sessions/20260809-025745
//! ```
//!
//! `measure_reference_lag`'s and `cancel_bleed`'s outcomes reach disk only as one line in
//! `transcribe`'s progress output, written once and not kept -- so "did this session's mic
//! track get cleaned, and what did the pre-pass measure" is otherwise a question that needs an
//! instrumented rebuild to answer a second time. This is that instrument, kept as its own
//! binary rather than a flag on `transcribe` so it can be pointed at a session repeatedly while
//! iterating on `align.rs` or `aec.rs` without re-running ASR.
//!
//! Like the other diagnostics in this directory, this takes no dependency the library does not
//! already have.

use std::path::PathBuf;

use meethook_session::{SessionMetadata, SessionPaths};
use meethook_transcribe::{
    Alignment, cancel_bleed, measure_reference_lag, mic_offset_seconds, read_track_16k_mono,
    speaker_offset_seconds,
};

fn main() {
    let Some(dir) = std::env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: align-session <session-dir>");
        std::process::exit(2);
    };

    let paths = SessionPaths::new(&dir);
    let metadata = SessionMetadata::read(&paths.session_json()).unwrap_or_else(|e| {
        eprintln!("{}: unreadable session.json: {e}", dir.display());
        std::process::exit(1);
    });

    // Exactly one of these is non-zero (see their doc comments); subtracting recovers the
    // signed value `measure_reference_lag` and `cancel_bleed` both want, without reaching for
    // the private helper that `mic_offset_seconds`/`speaker_offset_seconds` are themselves
    // built on.
    let mic_minus_speaker_s =
        mic_offset_seconds(&metadata).unwrap() - speaker_offset_seconds(&metadata).unwrap();

    let mic = read_track_16k_mono(&paths.mic_wav()).unwrap_or_else(|e| {
        eprintln!("{}: unreadable mic.wav: {e}", paths.mic_wav().display());
        std::process::exit(1);
    });
    let speaker = read_track_16k_mono(&paths.speaker_wav()).unwrap_or_default();

    println!("{}", dir.display());
    println!(
        "metadata offset: mic {:.3} s, speaker {:.3} s",
        mic_offset_seconds(&metadata).unwrap(),
        speaker_offset_seconds(&metadata).unwrap()
    );

    match measure_reference_lag(&mic, &speaker, mic_minus_speaker_s) {
        Alignment::Measured {
            lag_samples,
            windows_used,
            spread_samples,
            drift_ms_per_hour,
        } => println!(
            "alignment: measured lag {lag_samples} samples, {windows_used} windows, \
             spread {spread_samples} samples, drift {drift_ms_per_hour:+.1} ms/hour"
        ),
        Alignment::NotMeasurable { reason } => println!("alignment: not measurable: {reason}"),
    }

    let cleaned = cancel_bleed(&mic, &speaker, mic_minus_speaker_s);
    println!("cleaning: {}", cleaned.cleaning);
}
