//! Runs a whole speaker track through Nemotron and reports wall time and the probabilities.
//!
//! ```text
//! /usr/bin/time -l cargo run --release --example nemotron-track ~/meethook/sessions/X/speaker.wav
//! /usr/bin/time -l cargo run --release --example nemotron-track clip.wav 7200
//! ```
//!
//! The optional second argument tiles the track to that many seconds, for measuring a two hour
//! meeting without one. A trailing `cpu` argument forces the CPU provider.

#[path = "support/mod.rs"]
mod support;

use std::path::PathBuf;
use std::time::Instant;

use meethook_transcribe::{NemotronGraph, NemotronMel, speaker_probabilities};
use support::fail;

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let Some(track) = args.first().map(PathBuf::from) else {
        eprintln!("usage: nemotron-track <wav> [tile-to-seconds] [cpu]");
        std::process::exit(2);
    };
    let mut audio =
        meethook_transcribe::read_track_16k_mono(&track).unwrap_or_else(|e| fail(&format!("{e}")));
    if let Some(secs) = args.get(1).and_then(|s| s.parse::<usize>().ok()) {
        let want = secs * 16_000;
        let base = audio.clone();
        while audio.len() < want {
            audio.extend_from_slice(&base);
        }
        audio.truncate(want);
    }
    let root = match std::env::var_os("MEETHOOK_ROOT") {
        Some(root) => PathBuf::from(root),
        None => std::env::home_dir()
            .unwrap_or_else(|| fail("no home directory; set MEETHOOK_ROOT"))
            .join("meethook"),
    };
    let models = root.join("models");
    let cpu = args.iter().any(|a| a == "cpu");
    let mut graph = if cpu {
        NemotronGraph::open_cpu(&models)
    } else {
        NemotronGraph::open(&models)
    }
    .unwrap_or_else(|e| fail(&format!("{e}")));
    println!(
        "{:.0} s of audio, accelerated = {}",
        audio.len() as f64 / 16_000.0,
        graph.accelerated
    );
    let started = Instant::now();
    let probs = speaker_probabilities(&mut graph, &mut NemotronMel::new(), &audio)
        .unwrap_or_else(|e| fail(&format!("{e}")));
    println!(
        "{} frames in {:.1} s",
        probs.len() / 8,
        started.elapsed().as_secs_f64()
    );
}
