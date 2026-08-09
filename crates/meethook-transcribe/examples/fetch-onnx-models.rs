//! Installs the two ONNX graphs diarization needs, so the smoke test has weights to load.
//!
//! ```text
//! cargo run --example fetch-onnx-models
//! MEETHOOK_ROOT=/tmp/meethook cargo run --example fetch-onnx-models
//! ```
//!
//! Until diarization is wired into `meethook transcribe`, nothing on the normal path
//! acquires these files, and the graph-contract test in `onnx.rs` skips without them. This
//! is that acquisition, on demand: the same [`ensure_model`] call the real command will
//! make, against the same [`ModelSpec`] constants, so what gets installed here is byte for
//! byte what the tool will install later -- verified against the embedded sha256, not
//! merely downloaded.
//!
//! Together they are 32 MB, which is why this is a deliberate invocation rather than
//! something `cargo test` does behind the user's back.

use std::io::Write;

use meethook_models::{ModelSpec, ensure_model};
use meethook_transcribe::{EMBEDDING_MODEL, SEGMENTATION_MODEL};

fn main() {
    let root = match std::env::var_os("MEETHOOK_ROOT") {
        Some(root) => std::path::PathBuf::from(root),
        None => std::env::home_dir()
            .expect("could not determine the home directory; set MEETHOOK_ROOT")
            .join("meethook"),
    };
    let models_dir = root.join("models");

    for spec in [&SEGMENTATION_MODEL, &EMBEDDING_MODEL] {
        install(&models_dir, spec);
    }
}

fn install(models_dir: &std::path::Path, spec: &ModelSpec) {
    // Carriage returns, not newlines: one line per model that fills up, rather than a
    // thousand lines of scrollback per download.
    let mut last_percent = u64::MAX;
    let mut progress = |done: u64, total: u64| {
        let percent = done * 100 / total.max(1);
        if percent != last_percent {
            last_percent = percent;
            print!("\r{}  {percent:>3}%", spec.file_name);
            let _ = std::io::stdout().flush();
        }
    };

    match ensure_model(models_dir, spec, &mut progress) {
        Ok(path) => println!("\r{}  installed at {}", spec.file_name, path.display()),
        Err(e) => {
            println!();
            eprintln!("{}  failed: {e}", spec.file_name);
            std::process::exit(1);
        }
    }
}
