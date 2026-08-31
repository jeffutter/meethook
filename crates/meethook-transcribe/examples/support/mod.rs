//! Shared helpers for the diagnostic examples in this crate.
//!
//! An example is its own crate, so the examples cannot share a normal sibling module; this is
//! the one tree they pull in through `#[path]` instead. What lives here is the small set of
//! helpers the examples carried word for word -- a failure exit, a rate formatter, the model
//! load with its CoreML-declined note, and the trial-report lines `speaker-trials` and
//! `cluster-speaker-track` both print. The arithmetic those lines report on stays in the
//! library; only the printing is shared, because pushing presentation into the crate would
//! make the library depend on diagnostic formatting.
//!
//! Every tool that pulls from here must stay byte-identical on re-run: bodies moved in from an
//! example are verbatim, modulo an indent parameter where the two callers indented differently.

// Every example compiles this module whole but consumes only its own slice -- `vad-regions`
// takes just `fail`, `build-session` just `session_prep`. The examples are separate crates,
// so without this the unused pieces would warn in every one of them.
#![allow(dead_code)]

pub mod session_prep;

use std::path::Path;

use meethook_transcribe::{TrialReport, open_session};

/// Prints the message and exits 1. The examples' one shared failure shape: a diagnostic that
/// dies mid-run says why on stderr rather than unwinding with a backtrace.
pub fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}

/// A rate as a trailing parenthetical, with the two leading spaces baked into the return
/// value: the callers print `{count} pair(s){}` and the spaces are the column gap, so they
/// belong to the formatter rather than to every call site.
pub fn percent(rate: Option<f32>) -> String {
    match rate {
        Some(rate) => format!("  ({:.1}%)", rate * 100.0),
        None => "  (no such pairs, so no rate)".to_string(),
    }
}

/// Cosine distance between two unit-length vectors, from the crate: the same dot product the
/// library's decisions are made on, so a diagnostic measuring it cannot disagree with them.
// Not measured by every example that compiles this module.
#[allow(unused_imports)]
pub use meethook_transcribe::cosine_distance;

/// Loads one ONNX graph from `<root>/models`, failing with the fetch hint if it is not there,
/// and noting on stderr where CoreML declined it.
///
/// The note is gated to macOS: off macOS `accelerated` is always false by construction of the
/// build, so printing it there would name a component the platform never had.
pub fn load(root: &Path, file_name: &str) -> ort::session::Session {
    let model = root.join("models").join(file_name);
    let loaded = open_session(&model).unwrap_or_else(|e| {
        fail(&format!(
            "{e}\nrun `cargo run --release --example fetch-onnx-models` first"
        ))
    });
    #[cfg(target_os = "macos")]
    if !loaded.accelerated {
        eprintln!("note: CoreML declined {file_name}; running on CPU");
    }
    loaded.session
}

// ---------------------------------------------------------------------------------------
// Trial-report lines
//
// `speaker-trials` reports a whole trial list; `cluster-speaker-track` scores one session's
// adoption populations. Wherever the two print the same arithmetic they now call the same
// printers below, indented to their own section rather than re-spelling the lines. The
// headers stay at each call site: they differ (one says "separation", the other
// "  separation:"), and the callers keep the surrounding prose that only makes sense in
// context.
// ---------------------------------------------------------------------------------------

/// The false-accept / false-reject pair at one cut, in the wording both tools use for the
/// same numbers.
pub fn cost_lines(indent: &str, report: &TrialReport) {
    println!(
        "{indent}false accepts: {} different-speaker pair(s) below the cut{}",
        report.false_accepts,
        percent(report.false_accept_rate)
    );
    println!(
        "{indent}false rejects: {} same-speaker pair(s) at or above it{}",
        report.false_rejects,
        percent(report.false_reject_rate)
    );
}

/// Whether the two populations overlap, where the equal-error rate sits, and the largest
/// misattribution-free cut -- the three verdicts both tools take over one [`TrialReport`].
pub fn separation_and_rates(indent: &str, report: &TrialReport) {
    match report.overlap {
        Some((min_different, max_same)) => println!(
            "{indent}NO SINGLE THRESHOLD SEPARATES THESE TWO POPULATIONS. They overlap between \
             {min_different:.3} (the closest different-speaker pair) and {max_same:.3} (the \
             furthest-apart same-speaker pair); every cut inside that band trades one kind of \
             mistake for the other."
        ),
        None => match (report.same, report.different) {
            (Some(same), Some(different)) => println!(
                "{indent}the two populations do not overlap: every same-speaker pair is below \
                 {:.3} and every different-speaker pair is at or above {:.3}, so any cut in \
                 between makes no mistakes on this list",
                same.max, different.min
            ),
            _ => println!("{indent}not measurable: one side of the trial list is empty"),
        },
    }

    match report.equal_error {
        Some(equal_error) => println!(
            "{indent}equal error rate {:.1}% at a cut of {:.3}  (the mean of the two rates where \
             they come closest to crossing)",
            equal_error.rate * 100.0,
            equal_error.threshold
        ),
        None => println!("{indent}equal error rate: not measurable, one side of the list is empty"),
    }
    match report.zero_false_accept {
        Some(zero) => println!(
            "{indent}the largest cut that misattributes nobody is {:.3}, and it rejects {:.1}% \
             of same-speaker pairs",
            zero.threshold,
            zero.false_reject_rate * 100.0
        ),
        None => println!("{indent}no misattribution-free cut is measurable from this list"),
    }
}
