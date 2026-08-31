//! The source-conversion and level lines `build-session` and `speaker-trials` both print over
//! the same [`meethook_transcribe::BuiltSession`]: what each input wav became, and whether the
//! track they were written into actually holds audio.
//!
//! Reading the levels takes a second and turns "the measurement came back empty" into "the
//! input was dead" *before* a whisper pass is paid for rather than after -- which is why both
//! tools print them, and why the line belongs here once rather than spelled twice.

use meethook_transcribe::{ImportedSource, LevelSummary, TARGET_RATE};

/// One source wav as it was imported: where it came from, what it held, and what the session
/// track ended up as.
pub fn converted(source: &ImportedSource) -> String {
    format!(
        "{}: {} Hz, {} ch -> {TARGET_RATE} Hz mono ({:.2} s)",
        source.path.display(),
        source.sample_rate,
        source.channels,
        source.samples as f64 / f64::from(TARGET_RATE)
    )
}

/// The written track's loudness profile, labelled by the caller.
///
/// Peak alone does not settle whether a track holds speech: two UI chimes peak around 0.57
/// while the track is silent for 99% of its length. The fraction and the run length are what
/// separate "somebody talked" from "something beeped".
pub fn levels(name: &str, summary: &LevelSummary) {
    let dbfs = summary.peak_dbfs();
    let peak = if dbfs.is_infinite() {
        "0.0 (digital silence)".to_string()
    } else {
        format!("{:.4} ({dbfs:.1} dBFS)", summary.peak)
    };
    println!(
        "  {name:<12} {:.2} s, peak {peak}, {:.1}% above floor, longest run {:.3} s",
        summary.duration_s(),
        summary.above_fraction() * 100.0,
        summary.longest_run_s()
    );
}
