//! Times the voice isolator on CoreML, on the CPU and (with `--features mlx`) on MLX for one
//! 48 kHz mono wav.
//!
//! ```text
//! cargo run --release --example isolate-voice crates/meethook-transcribe/testdata/voice-isolator/sample_14.wav
//! cargo run --release --example isolate-voice in.wav out.wav   # also writes the first result
//! cargo run --release --example isolate-voice in.wav --only cpu # or `coreml`, or `mlx`
//! cargo run --release --example isolate-voice in.wav --stage [--minutes 60] [--only cpu]
//! cargo run --release --example isolate-voice mic.cleaned.wav --stage --track --minutes 30
//! ```
//!
//! `--stage` instead times the whole mic-isolation stage (real speech detector, 60 s span cap,
//! splicing, CPU isolator as `transcribe` uses it) over a synthetic mic track built by
//! repeating the wav until it lasts `--minutes` (default 60), and prints audio seconds per
//! wall second. Back-to-back copies of one speech sample are all speech: the worst case for a
//! stage that isolates only speech regions.
//!
//! Without `--stage`, prints whether CoreML accepted the graph, each path's wall time and throughput (seconds of
//! audio per second of wall time), and the SNR of the CPU output against the CoreML one when
//! both ran. Nothing
//! on the normal path calls this; it is how "is the isolator fast enough, and does CoreML change
//! the answer" gets measured again. The weights are downloaded on first use.

#[path = "support/mod.rs"]
mod support;

use std::path::PathBuf;
use std::time::Instant;

use meethook_models::ensure_model;
#[cfg(feature = "mlx")]
use meethook_transcribe::VOICE_ISOLATOR_MLX_MODEL;
use meethook_transcribe::{
    FIRERED_VAD_MODEL, FireRedVad, GateSpeechFinder, IsolationStage, VOICE_ISOLATOR_MODEL,
    VoiceIsolator,
};
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
    let stage = args
        .iter()
        .position(|a| a == "--stage")
        .map(|at| args.remove(at))
        .is_some();
    let minutes: f64 = args
        .iter()
        .position(|a| a == "--minutes")
        .map(|at| {
            let mut taken = args.drain(at..(at + 2).min(args.len()));
            taken.nth(1).and_then(|v| v.into_string().ok())
        })
        .and_then(|v| v.and_then(|v| v.parse().ok()))
        .unwrap_or(60.0);
    let track = args
        .iter()
        .position(|a| a == "--track")
        .map(|at| args.remove(at))
        .is_some();
    let mut args = args.into_iter();
    let Some(input) = args.next().map(PathBuf::from) else {
        eprintln!("usage: isolate-voice <48 kHz mono wav> [output wav]");
        std::process::exit(2);
    };
    let output = args.next().map(PathBuf::from);

    if stage {
        time_stage(&input, minutes, track, only.as_deref());
        return;
    }
    let audio = read_48k(&input);
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
    #[cfg(feature = "mlx")]
    ensure_model(&models, &VOICE_ISOLATOR_MLX_MODEL, &mut |_, _| {})
        .unwrap_or_else(|e| fail(&format!("{e}")));
    println!("{}: {seconds:.2} s", input.display());

    // CoreML first so a run that is killed for memory still says so before it dies; it is
    // also the path to be careful with (see the module docs of `voice_isolator`).
    let mut engines = vec!["coreml", "cpu"];
    #[cfg(feature = "mlx")]
    engines.push("mlx");
    if let Some(only) = only.as_deref() {
        engines.retain(|e| *e == only);
        if engines.is_empty() {
            fail("--only takes `coreml`, `cpu` or (with --features mlx) `mlx`");
        }
    }
    let mut results = Vec::new();
    for engine in engines {
        let opened = Instant::now();
        let opened_engine = match engine {
            "coreml" => VoiceIsolator::open_coreml(&models),
            "cpu" => VoiceIsolator::open_onnx(&models),
            #[cfg(feature = "mlx")]
            "mlx" => VoiceIsolator::open_mlx(&models),
            _ => unreachable!("engines are filtered above"),
        };
        let mut isolator = opened_engine.unwrap_or_else(|e| fail(&format!("{e}")));
        let label = match engine {
            "coreml" if isolator.accelerated => "default (CoreML accepted the load)",
            "coreml" => "default (CoreML declined, running on CPU)",
            "cpu" => "CPU",
            _ => "MLX",
        };
        let load = opened.elapsed().as_secs_f64();
        println!("{label}: loaded in {load:.1} s");
        // One warm-up pass would hide CoreML's first-run compile (and MLX's kernel compile), which
        // a real run pays too, so the timed pass is the first one.
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

fn read_48k(path: &std::path::Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path).unwrap_or_else(|e| fail(&format!("{e}")));
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != 48_000 {
        fail("the input must be 48 kHz mono");
    }
    match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / scale)
                .collect()
        }
    }
}

fn models_dir() -> PathBuf {
    let root = match std::env::var_os("MEETHOOK_ROOT") {
        Some(root) => PathBuf::from(root),
        None => std::env::home_dir()
            .unwrap_or_else(|| fail("no home directory; set MEETHOOK_ROOT"))
            .join("meethook"),
    };
    root.join("models")
}

fn time_stage(input: &std::path::Path, minutes: f64, real_track: bool, only: Option<&str>) {
    let models = models_dir();
    let wanted = if only == Some("cpu") {
        &VOICE_ISOLATOR_MODEL
    } else {
        VoiceIsolator::required_model()
    };
    ensure_model(&models, wanted, &mut |_, _| {}).unwrap_or_else(|e| fail(&format!("{e}")));
    let vad = ensure_model(&models, &FIRERED_VAD_MODEL, &mut |_, _| {})
        .unwrap_or_else(|e| fail(&format!("{e}")));
    let target = (minutes * 60.0 * 16_000.0) as usize;
    let (track, seconds) = if real_track {
        // A recorder track: 16 kHz mono float, such as a session's `mic.cleaned.wav`. Only the
        // first `minutes` are read, and nothing is written.
        let mut reader = hound::WavReader::open(input).unwrap_or_else(|e| fail(&format!("{e}")));
        let spec = reader.spec();
        if spec.channels != 1
            || spec.sample_rate != 16_000
            || spec.sample_format != hound::SampleFormat::Float
        {
            fail("--track takes a 16 kHz mono float wav");
        }
        let track: Vec<f32> = reader
            .samples::<f32>()
            .take(target)
            .map(|s| s.unwrap_or_else(|e| fail(&format!("{e}"))))
            .collect();
        let seconds = track.len() as f64 / 16_000.0;
        println!(
            "real mic track: first {seconds:.0} s of {}",
            input.display()
        );
        (track, seconds)
    } else {
        // Not `read_track_16k_mono`, which accepts only the recorder's float format; the sample
        // is 16-bit. A 3-sample average is a crude 48 -> 16 kHz decimation, fine for a timing
        // track.
        let audio = read_48k(input);
        let sample: Vec<f32> = audio
            .as_chunks::<3>()
            .0
            .iter()
            .map(|c| (c[0] + c[1] + c[2]) / 3.0)
            .collect();
        let mut track = Vec::with_capacity(target + sample.len());
        while track.len() < target {
            track.extend_from_slice(&sample);
        }
        track.truncate(target);
        let seconds = track.len() as f64 / 16_000.0;
        println!(
            "synthetic mic track: {seconds:.0} s from {} s copies",
            sample.len() / 16_000
        );
        (track, seconds)
    };

    let mut stage = IsolationStage {
        isolator: Box::new(
            match only {
                Some("cpu") => VoiceIsolator::open_onnx(&models),
                _ => VoiceIsolator::open(&models),
            }
            .unwrap_or_else(|e| fail(&format!("{e}"))),
        ),
        finder: Box::new(GateSpeechFinder(
            FireRedVad::load(&vad).unwrap_or_else(|e| fail(&format!("{e}"))),
        )),
    };
    let started = Instant::now();
    let (audio, report) = stage.run(&track).unwrap_or_else(|e| fail(&format!("{e}")));
    let wall = started.elapsed().as_secs_f64();
    assert_eq!(audio.len(), track.len());
    println!("{report}");
    println!(
        "stage: {seconds:.0} s of audio in {wall:.0} s wall = {:.2}x realtime",
        seconds / wall
    );
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
