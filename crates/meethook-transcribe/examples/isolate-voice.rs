//! Times the voice isolator on CoreML and on CPU for one 48 kHz mono wav.
//!
//! ```text
//! cargo run --release --example isolate-voice crates/meethook-transcribe/testdata/voice-isolator/sample_14.wav
//! cargo run --release --example isolate-voice in.wav out.wav   # also writes the first result
//! cargo run --release --example isolate-voice in.wav --only cpu # or `--only coreml`
//! ```
//!
//! Prints whether CoreML accepted the graph, each path's wall time and throughput (seconds of
//! audio per second of wall time), and the SNR of the CPU output against the CoreML one when
//! both ran. Nothing
//! on the normal path calls this; it is how "is the isolator fast enough, and does CoreML change
//! the answer" gets measured again. The weights are downloaded on first use.

#[path = "support/mod.rs"]
mod support;

use std::path::PathBuf;
use std::time::Instant;

use meethook_models::ensure_model;
use meethook_transcribe::{VOICE_ISOLATOR_MODEL, VoiceIsolator};
use support::fail;

fn main() {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    let only = args
        .iter()
        .position(|a| a == "--only")
        .map(|at| {
            let mut taken = args.drain(at..(at + 2).min(args.len()));
            taken.nth(1).unwrap_or_default()
        })
        .and_then(|v| v.into_string().ok());
    let mut args = args.into_iter();
    let Some(input) = args.next().map(PathBuf::from) else {
        eprintln!("usage: isolate-voice <48 kHz mono wav> [output wav]");
        std::process::exit(2);
    };
    let output = args.next().map(PathBuf::from);

    let mut reader = hound::WavReader::open(&input).unwrap_or_else(|e| fail(&format!("{e}")));
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 48_000 {
        fail("the input must be 48 kHz mono");
    }
    let audio: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / scale)
                .collect()
        }
    };
    let seconds = audio.len() as f64 / 48_000.0;

    let root = match std::env::var_os("MEETHOOK_ROOT") {
        Some(root) => PathBuf::from(root),
        None => std::env::home_dir()
            .unwrap_or_else(|| fail("no home directory; set MEETHOOK_ROOT"))
            .join("meethook"),
    };
    let models = root.join("models");
    ensure_model(&models, &VOICE_ISOLATOR_MODEL, &mut |_, _| {})
        .unwrap_or_else(|e| fail(&format!("{e}")));
    println!("{}: {seconds:.2} s", input.display());

    let mut results = Vec::new();
    // CoreML first so a run that is killed for memory still says so before it dies; it is
    // also the path to be careful with (see the module docs of `voice_isolator`).
    for cpu in [false, true] {
        if only.as_deref() == Some(if cpu { "coreml" } else { "cpu" }) {
            continue;
        }
        let opened = Instant::now();
        let mut isolator = if cpu {
            VoiceIsolator::open(&models)
        } else {
            VoiceIsolator::open_coreml(&models)
        }
        .unwrap_or_else(|e| fail(&format!("{e}")));
        let label = if cpu {
            "CPU"
        } else if isolator.accelerated {
            "default (CoreML accepted the load)"
        } else {
            "default (CoreML declined, running on CPU)"
        };
        let load = opened.elapsed().as_secs_f64();
        println!("{label}: loaded in {load:.1} s");
        // One warm-up pass would hide CoreML's first-run compile, which a real run pays too, so
        // the timed pass is the first one.
        let started = Instant::now();
        let voice = isolator
            .isolate(&audio)
            .unwrap_or_else(|e| fail(&format!("{e}")));
        let wall = started.elapsed().as_secs_f64();
        println!(
            "{label}: load {load:.1} s, isolate {wall:.2} s = {:.2}x realtime",
            seconds / wall
        );
        results.push(voice);
    }

    let a = &results[0];
    if let [a, b] = &results[..] {
        report_snr(a, b);
    }

    if let Some(path) = output {
        write_wav(&path, a);
    }
}

fn report_snr(a: &[f32], b: &[f32]) {
    let signal: f64 = b.iter().map(|&s| f64::from(s).powi(2)).sum();
    let noise: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x - y).powi(2))
        .sum();
    println!(
        "CPU vs default output: SNR {:.1} dB",
        10.0 * (signal / noise.max(1e-30)).log10()
    );
}

fn write_wav(path: &std::path::Path, samples: &[f32]) {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap_or_else(|e| fail(&format!("{e}")));
    for &s in samples {
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
    println!("wrote {}", path.display());
}
