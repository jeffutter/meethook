//! Re-measures the turn-decoding constants of `nemotron_turns` on a real speaker track.
//!
//! ```text
//! cargo run --release --example nemotron-turns ~/meethook/sessions/X/speaker.wav [seconds]
//! ```
//!
//! Runs the track through Nemotron once, then decodes the probabilities with the shipped
//! `slot_turns` and with a local copy of the same state machine whose four constants can vary
//! (the library deliberately exposes no knobs). The local copy is checked against `slot_turns`
//! at the shipped values. The optional second argument keeps only the first N seconds.

#[path = "support/mod.rs"]
mod support;

use std::path::PathBuf;

use meethook_transcribe::{NemotronGraph, NemotronMel, slot_turns, speaker_probabilities};
use support::fail;

const SLOTS: usize = 8;

/// `(raw runs, turns, speech seconds, union speech seconds)` for one parameter set.
fn decode(
    probs: &[f32],
    enter: f32,
    stay: f32,
    gap: usize,
    min: usize,
) -> (usize, usize, f64, f64) {
    let frames = probs.len() / SLOTS;
    let mut raw = 0;
    let mut turns = 0;
    let mut speech = 0usize;
    let mut union = vec![false; frames];
    for slot in 0..SLOTS {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut open = None;
        for f in 0..=frames {
            let p = if f < frames {
                probs[f * SLOTS + slot]
            } else {
                0.0
            };
            match open {
                None if p > enter => open = Some(f),
                Some(s) if p < stay => {
                    open = None;
                    raw += 1;
                    match runs.last_mut() {
                        Some(l) if s - l.1 <= gap => l.1 = f,
                        _ => runs.push((s, f)),
                    }
                }
                _ => {}
            }
        }
        for (s, e) in runs.into_iter().filter(|(s, e)| e - s >= min) {
            turns += 1;
            speech += e - s;
            union[s..e].iter_mut().for_each(|u| *u = true);
        }
    }
    let u = union.iter().filter(|&&x| x).count();
    (raw, turns, speech as f64 / 100.0, u as f64 / 100.0)
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let Some(track) = args.first().map(PathBuf::from) else {
        eprintln!("usage: nemotron-turns <wav> [first-seconds]");
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
    let mut graph =
        NemotronGraph::open(&root.join("models")).unwrap_or_else(|e| fail(&format!("{e}")));
    let probs = speaker_probabilities(&mut graph, &mut NemotronMel::new(), &audio)
        .unwrap_or_else(|e| fail(&format!("{e}")));
    println!("{:.0} s of audio", audio.len() as f64 / 16_000.0);

    let shipped = slot_turns(&probs);
    let base = decode(&probs, 0.5, 0.25, 100, 50);
    assert_eq!(
        shipped.len(),
        base.1,
        "local decoder drifted from slot_turns"
    );
    println!("shipped slot_turns: {} turns", shipped.len());
    println!("enter stay gap_s min_s | raw turns slot_speech_s union_speech_s");
    let row = |label: &str, enter, stay, gap, min| {
        let (r, t, s, u) = decode(&probs, enter, stay, gap, min);
        println!("{label:<22} | {r} {t} {s:.0} {u:.0}");
    };
    row("baseline 0.5/0.25/1.0/0.5", 0.5, 0.25, 100, 50);
    for enter in [0.4, 0.6, 0.7] {
        row(&format!("enter {enter}"), enter, 0.25, 100, 50);
    }
    for stay in [0.15, 0.4] {
        row(&format!("stay {stay}"), 0.5, stay, 100, 50);
    }
    for gap in [0, 50, 150] {
        row(
            &format!("gap {:.1}s", gap as f64 / 100.0),
            0.5,
            0.25,
            gap,
            50,
        );
    }
    for min in [0, 30, 80] {
        row(
            &format!("min {:.1}s", min as f64 / 100.0),
            0.5,
            0.25,
            100,
            min,
        );
    }
}
