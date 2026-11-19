//! Times and compares the Nemotron graph on CoreML and on CPU for one real track.
//!
//! ```text
//! cargo run --release --example nemotron-step ~/meethook/sessions/20260809-021730
//! cargo run --release --example nemotron-step some.wav
//! ```
//!
//! Runs the first `STEPS` chunk steps of the track's `speaker.wav` (or the given wav) with the
//! speaker cache carried between them, once with the default provider (CoreML on macOS) and
//! once on CPU, then prints whether CoreML accepted the graph, the median step time of each and
//! the largest difference between their logits. Nothing on the normal path calls this; it is
//! how "does CoreML help here, and does it change the answer" gets measured again.

#[path = "support/mod.rs"]
mod support;

use std::path::PathBuf;
use std::time::Instant;

use meethook_transcribe::{NemotronGraph, NemotronMel, encoder_valid};
use support::fail;

const STEPS: usize = 10;
const CHUNK: usize = 340;
const RIGHT: usize = 40;
const STACK: usize = 8;
const BINS: usize = 128;
const HIDDEN: usize = 512;

fn main() {
    let Some(target) = std::env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: nemotron-step <session-dir | wav-file>");
        std::process::exit(2);
    };
    let track = if target.is_dir() {
        target.join("speaker.wav")
    } else {
        target
    };
    let audio =
        meethook_transcribe::read_track_16k_mono(&track).unwrap_or_else(|e| fail(&format!("{e}")));
    let root = match std::env::var_os("MEETHOOK_ROOT") {
        Some(root) => PathBuf::from(root),
        None => std::env::home_dir()
            .unwrap_or_else(|| fail("no home directory; set MEETHOOK_ROOT"))
            .join("meethook"),
    };
    let models = root.join("models");

    let mel = NemotronMel::new().compute_all(&audio);
    let frames = mel.len() / BINS;
    println!(
        "{}: {:.1} s, {frames} mel frames",
        track.display(),
        audio.len() as f64 / 16_000.0
    );

    let mut runs = Vec::new();
    for cpu in [false, true] {
        let mut graph = if cpu {
            NemotronGraph::open_cpu(&models)
        } else {
            NemotronGraph::open(&models)
        }
        .unwrap_or_else(|e| fail(&format!("{e}")));
        println!(
            "{}: accelerated = {}",
            if cpu { "cpu" } else { "default" },
            graph.accelerated
        );

        // Without the speaker cache's compression the cache would grow past what a real run
        // holds, so carry only the fixed-size tail: the previous step's own chunk embeddings.
        // That is not the real cache, but it gives non-empty cached input of a realistic size.
        let mut cached: Vec<f32> = Vec::new();
        let mut times = Vec::new();
        let mut logits = Vec::new();
        for step in 0..STEPS {
            let first = step * CHUNK * STACK;
            let last = (first + (CHUNK + RIGHT) * STACK).min(frames);
            if first >= last {
                break;
            }
            let feats = &mel[first * BINS..last * BINS];
            let valid = encoder_valid(last - first, last - first);
            let t = Instant::now();
            let out = graph
                .step(feats, &cached, &valid)
                .unwrap_or_else(|e| fail(&format!("{e}")));
            times.push(t.elapsed().as_secs_f64());
            let enc = out.chunk_embeds.len() / HIDDEN;
            cached = out.chunk_embeds[..enc.min(CHUNK) * HIDDEN].to_vec();
            logits.push(out.logits);
        }
        times.sort_by(|a, b| a.total_cmp(b));
        println!(
            "  {} steps, median {:.3} s, min {:.3} s, max {:.3} s",
            times.len(),
            times[times.len() / 2],
            times[0],
            times[times.len() - 1]
        );
        runs.push(logits);
    }
    let diff = runs[0]
        .iter()
        .zip(&runs[1])
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
        .fold(0.0f32, f32::max);
    println!("max abs logit difference, default vs cpu: {diff}");
}
