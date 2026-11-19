//! Diarizes a speaker track with `NemotronDiarizer` and prints the clusters.
//!
//! ```text
//! cargo run --release --example nemotron-diarize ~/meethook/sessions/X/speaker.wav [seconds]
//! ```
//!
//! The optional second argument keeps only the first N seconds.

#[path = "support/mod.rs"]
mod support;

use std::path::PathBuf;

use meethook_transcribe::{Diarize, EMBEDDING_MODEL, NemotronDiarizer};
use support::fail;

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let Some(track) = args.first().map(PathBuf::from) else {
        eprintln!("usage: nemotron-diarize <wav> [seconds]");
        std::process::exit(2);
    };
    let mut audio =
        meethook_transcribe::read_track_16k_mono(&track).unwrap_or_else(|e| fail(&format!("{e}")));
    if let Some(secs) = args.get(1).and_then(|s| s.parse::<usize>().ok()) {
        audio.truncate(secs * 16_000);
    }
    let root = match std::env::var_os("MEETHOOK_ROOT") {
        Some(root) => PathBuf::from(root),
        None => std::env::home_dir()
            .unwrap_or_else(|| fail("no home directory; set MEETHOOK_ROOT"))
            .join("meethook"),
    };
    let models = root.join("models");
    let mut diarizer = NemotronDiarizer::load(&models, &models.join(EMBEDDING_MODEL.file_name))
        .unwrap_or_else(|e| fail(&format!("{e}")));
    let d = diarizer
        .diarize(&audio)
        .unwrap_or_else(|e| fail(&format!("{e}")));
    println!("{} clusters, {} turns", d.clusters.len(), d.turns.len());
    for c in &d.clusters {
        println!(
            "cluster {}: {:.0} s speech, first spoke {:.1} s, heard at once with {:?}",
            c.id, c.speech_seconds, c.first_spoke_seconds, c.heard_at_once_with
        );
    }
}
