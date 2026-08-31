//! Times the diarization graphs under each CoreML model format, with and without a compiled-model
//! cache, and reports what CoreML did with each.
//!
//! ```text
//! cargo run --release --example coreml-configurations ~/meethook/sessions/20260810-121550
//! cargo run --release --example coreml-configurations some-other-recording.wav
//! cargo run --release --example coreml-configurations ~/meethook/sessions/20260810-121550 --cache-dir ./coreml-cache
//! cargo run --release --example coreml-configurations short.wav --log
//! ```
//!
//! A session directory means its `speaker.wav`; any other path is used as given -- the same rule
//! the sibling diagnostics use. Everything is printed on stdout, and the library's progress
//! heartbeats go to stderr, so `> report.txt` keeps the two apart. `--log` mirrors every ONNX
//! Runtime line this tool captures to stderr, for the run where the coverage column comes out
//! empty and the question becomes why.
//!
//! # Why this exists
//!
//! Two CoreML options are on the table for `open_session`: the newer `MLProgram` model format,
//! and a compiled-model cache directory. Neither is obviously a win, and neither can be decided
//! by reading either runtime's source, so the answer has to come from timing both against the
//! graphs this project actually loads.
//!
//! # macOS only
//!
//! The tool measures CoreML model formats, and `ort` compiles its CoreML module in only where
//! Apple's runtime exists, so the body lives in [`inner`] behind a target gate. Off macOS this
//! file still builds -- an example crate must define `main` wherever it compiles -- and running
//! it says so and stops, instead of failing the build of everything else `--all-targets` pulls
//! in.

#[cfg(target_os = "macos")]
mod inner;

// Only compiled where `inner` is: off macOS nothing in this example reads it, and an unused
// private module would be a warning rather than dead weight.
#[cfg(target_os = "macos")]
#[path = "../support/mod.rs"]
mod support;

#[cfg(target_os = "macos")]
fn main() {
    inner::run();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!(
        "coreml-configurations times CoreML model formats and only runs on macOS, where ort's \
         CoreML execution provider exists."
    );
    std::process::exit(1);
}
