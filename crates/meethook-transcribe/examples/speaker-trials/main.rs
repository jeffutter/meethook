//! Scores a whole speaker trial list: many cross-session pairs measured at once.
//!
//! ```text
//! cargo run --release --example speaker-trials -- \
//!   --root /tmp/calibration --embeddings /tmp/calibration/embeddings.json trials.tsv
//!
//! # re-print at a different cut, from the cache, with no audio and no models
//! cargo run --release --example speaker-trials -- \
//!   --root /tmp/calibration --embeddings /tmp/calibration/embeddings.json \
//!   --threshold 0.55 trials.tsv
//! ```
//!
//! Why this exists: `cluster-speaker-track` prints every cluster-to-reference distance for
//! *one* session, which is the right arithmetic at the wrong scale. Deciding whether
//! `IDENTIFY_DISTANCE` is in the right place needs error rates over hundreds of pairs -- a
//! false-accept rate, a false-reject rate, an equal-error rate, and whether the two
//! distributions overlap at all -- and none of those are obtainable by reading a table by eye.
//!
//! # The manifest
//!
//! Tab-separated, one line per source wav. `#` comments and blank lines are ignored:
//!
//! ```text
//! speaker_id <TAB> session_id <TAB> /path/to/audio.wav
//! ```
//!
//! Lines sharing a `(speaker_id, session_id)` become one **item**, spliced in file order. TSV
//! rather than JSON because the thing that writes one is a shell pipeline over a corpus
//! directory listing, and a person deciding what to include has to be able to write one by
//! hand.
//!
//! The item -- one person as they sounded in one recording session -- is the unit here because
//! that is what an enrolled reference *is*. Each becomes exactly one voice, through the same
//! `build_session` -> `segment_speaker_track` -> `cluster_speaker_turns` path a real session
//! takes, and the voice is the dominant cluster rather than a mean over clusters: `enroll`
//! copies one cluster's embedding verbatim, so averaging here would be calibrating an
//! algorithm meethook does not run.
//!
//! # Two rules that make the numbers mean what they claim
//!
//! **No pair drawn from a single session is a trial.** That is `MERGE_DISTANCE`'s question,
//! and including such pairs would flatter the same-speaker side with exactly the within-session
//! variation `IDENTIFY_DISTANCE` does not govern. Excluded pairs are counted and printed, so
//! the rule is visible rather than trusted.
//!
//! **The identification simulation goes through the real [`identify_clusters`].** The
//! trial-list rates above it are the standard speaker-verification quantities and they are
//! *not* what meethook decides: identification is argmax over every enrolled reference and
//! *then* the cut, so a reference that clears the threshold while a nearer one wins is not a
//! match. Both are reported, because only one of them is about this code.
//!
//! No audio is written into the repository by any path here, and the only artifact worth
//! carrying away from a corpus run is `--embeddings`, which holds derived vectors and no
//! audio.
//!
//! # `--policies`: what a person's reference should be made of
//!
//! Off by default, so the report above is byte-identical for anyone re-running an earlier
//! calibration. With it, the same items are scored under the three answers TASK-027 poses to
//! "one person has now been named twice" -- newest-wins replacement, a normalized mean, and
//! keeping both and taking the nearest -- through [`meethook_transcribe::policy_sweep`]. The
//! arithmetic and every verdict live in the crate and are unit-tested there; this file prints.

mod cache;
mod identify_sim;
mod manifest;
mod policies;
mod trials;
mod voices;

#[path = "../support/mod.rs"]
mod support;

use std::path::PathBuf;

use meethook_session::Paths;
use meethook_transcribe::{IDENTIFY_DISTANCE, score_trials};

use cache::write_cache;
use identify_sim::report_identification;
use manifest::read_manifest;
use policies::report_policies;
use support::fail;
use trials::{pair_up, report_scores, report_shape};
use voices::embed_items;

/// How much of each item's audio is measured, in seconds, unless `--seconds` says otherwise.
///
/// An enrolled reference is one meeting's cluster -- minutes of speech, not half an hour -- so
/// a thirty-minute corpus track measured whole would be a more flattering reference than
/// meethook ever holds.
const DEFAULT_SECONDS: f64 = 120.0;

/// How much speech an item's dominant cluster must hold to be scored, unless `--min-speech`
/// says otherwise.
///
/// This tool's own floor, deliberately not `speakers::MIN_EMBEDDABLE_SECONDS`: that constant
/// is private, it governs whether a *turn* can be embedded at all, and a copy of it here would
/// be a number that drifts out of agreement with the one it claims to be. Printed in the
/// header for the same reason the threshold is: a population nobody can reconstruct is not
/// evidence.
const DEFAULT_MIN_SPEECH: f64 = 3.0;

fn main() {
    let args = parse().unwrap_or_else(|message| {
        eprintln!("{message}");
        eprintln!(
            "usage: speaker-trials [--root <dir>] [--seconds <n>] [--min-speech <n>]\n       \
             [--threshold <distance>] [--embeddings <file>] [--fresh] [--keep-sessions]\n       \
             [--policies] <manifest.tsv>\n       \
             the root may also come from $MEETHOOK_ROOT; there is no default"
        );
        std::process::exit(2);
    });

    let items = read_manifest(&args.manifest).unwrap_or_else(|e| fail(&e));
    let paths = Paths::new(&args.root);

    println!("root:      {}", paths.root().display());
    println!(
        "manifest:  {} ({} items)",
        args.manifest.display(),
        items.len()
    );
    println!(
        "threshold: {:.3}{}",
        args.threshold,
        if args.threshold == IDENTIFY_DISTANCE {
            "  (IDENTIFY_DISTANCE)"
        } else {
            "  (--threshold; IDENTIFY_DISTANCE is not this)"
        }
    );
    println!(
        "limits:    at most {:.0} s measured per item, dominant cluster must hold {:.1} s of \
         speech",
        args.seconds, args.min_speech
    );

    let voices = embed_items(&paths, &args, &items);
    if let Some(path) = &args.embeddings {
        write_cache(path, &voices);
        println!("\nwrote {} voice(s) to {}", voices.len(), path.display());
    }

    let trials = pair_up(&voices);
    report_shape(&voices, &trials);

    let report = score_trials(&trials.trials, args.threshold);
    report_scores(&report);
    report_identification(&voices, args.threshold);

    if args.policies {
        report_policies(&voices, args.threshold);
    }
}

pub struct Args {
    pub root: PathBuf,
    pub manifest: PathBuf,
    pub seconds: f64,
    pub min_speech: f64,
    pub threshold: f32,
    pub embeddings: Option<PathBuf>,
    /// Ignore whatever `--embeddings` already holds and re-measure every item.
    pub fresh: bool,
    pub keep_sessions: bool,
    /// Also score the three reference policies. Off by default: an extra block would change
    /// the output of every earlier calibration re-run, and this one is about TASK-027.
    pub policies: bool,
}

/// Hand-rolled rather than clap, matching the other examples in this crate: a diagnostic must
/// never be the reason a build breaks.
///
/// There is deliberately no `~/meethook` fallback for the root, unlike `cluster-speaker-track`
/// and like `build-session`: this tool *creates* session directories, and it must not be able
/// to create a hundred of them among somebody's real recordings because a variable was unset.
fn parse() -> Result<Args, String> {
    let mut root = std::env::var_os("MEETHOOK_ROOT").map(PathBuf::from);
    let mut manifest: Option<PathBuf> = None;
    let mut seconds = DEFAULT_SECONDS;
    let mut min_speech = DEFAULT_MIN_SPEECH;
    let mut threshold = IDENTIFY_DISTANCE;
    let mut embeddings = None;
    let mut fresh = false;
    let mut keep_sessions = false;
    let mut policies = false;

    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |flag: &str| {
            args.next()
                .and_then(|v| v.into_string().ok())
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        let number = |flag: &str, raw: String| {
            raw.parse::<f64>()
                .map_err(|_| format!("{flag} needs a number, not {raw:?}"))
        };
        match arg.to_str() {
            Some("--root") => root = Some(PathBuf::from(value("--root")?)),
            Some("--embeddings") => embeddings = Some(PathBuf::from(value("--embeddings")?)),
            Some("--seconds") => seconds = number("--seconds", value("--seconds")?)?,
            Some("--min-speech") => min_speech = number("--min-speech", value("--min-speech")?)?,
            Some("--threshold") => {
                threshold = number("--threshold", value("--threshold")?)? as f32;
            }
            Some("--fresh") => fresh = true,
            Some("--policies") => policies = true,
            Some("--keep-sessions") => keep_sessions = true,
            Some(flag) if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            _ if manifest.is_none() => manifest = Some(PathBuf::from(arg)),
            _ => return Err("only one manifest may be given".to_string()),
        }
    }

    Ok(Args {
        root: root.ok_or("no root: pass --root or set $MEETHOOK_ROOT")?,
        manifest: manifest.ok_or("no manifest was given")?,
        seconds,
        min_speech,
        threshold,
        embeddings,
        fresh,
        keep_sessions,
        policies,
    })
}
