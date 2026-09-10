//! What the fragment-bundling cut actually separates: distance populations over real sessions.
//!
//! ```text
//! cargo run --release -p meethook-enroll --example fragment-trials -- \
//!   ~/meethook/sessions/20260818-132033 ~/meethook/sessions/20260818-143027
//!
//! # re-print at another cut; the header says which constant this is not
//! cargo run --release -p meethook-enroll --example fragment-trials -- \
//!   --threshold 0.45 ~/meethook/sessions
//! ```
//!
//! This is the harness [`GROUP_DISTANCE`]'s documentation cites: it prices that constant the
//! way `examples/speaker-trials` prices `IDENTIFY_DISTANCE` and `TENTATIVE_DISTANCE` -- one
//! named command whose every printed number anyone can regenerate, over session directories
//! someone can name.
//!
//! It needs **no models and no audio**: `speaker_clusters.json` already holds each cluster's
//! centroid, so the whole measurement is JSON reads and dot products, done in well under a
//! second. That is what makes it affordable to re-run after an embedding-model change, which
//! is the point.
//!
//! Each positional argument is a session directory (one holding `speaker_clusters.json`) or a
//! parent whose subdirectories are sessions. There is deliberately **no `~/meethook`
//! fallback**, unlike `cluster-speaker-track`: an operator-run measurement that silently
//! picks a machine-local default prints numbers whose corpus nobody can name.
//!
//! The arithmetic is not here. Distances come from [`fragment_pairs`] (which also applies the
//! heard-at-once veto evidence and the on-disk naming labels), bundling is walked by
//! [`fragment_bundles`] -- the shipping [`fragment_groups`] itself, so the report cannot
//! describe a rule the tool does not run -- and every rate is [`score_trials`]' conventions.
//! Output is byte-identical for the same inputs, so "the output did not change" is checkable.

use std::path::{Path, PathBuf};

use meethook_enroll::{FragmentPairLabel, GROUP_DISTANCE, fragment_bundles, fragment_pairs};
use meethook_session::{SessionPaths, SpeakerClusters, SpeakerNames};
use meethook_transcribe::{
    Spread, TENTATIVE_FLOOR_SECONDS, TrialReport, score_trials, wilson_interval,
};

/// Candidate cuts the population counts are printed against, ascending. A fixed list rather
/// than a sweep because these are the values the constant's documentation argues between --
/// vacuity below, `MERGE_DISTANCE`'s territory above -- and a table whose rows depend on the
/// data would make two corpora's tables unreadable side by side.
const CANDIDATE_CUTS: [f32; 6] = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50];

fn main() {
    let args = parse().unwrap_or_else(|message| {
        eprintln!("{message}");
        eprintln!(
            "usage: fragment-trials [--threshold <distance>] <session dir or sessions parent>\n       \
             at least one directory is required; there is no default root"
        );
        std::process::exit(2);
    });

    let threshold_note = if args.threshold == GROUP_DISTANCE {
        "  (GROUP_DISTANCE)".to_string()
    } else {
        format!("  (--threshold; GROUP_DISTANCE is {GROUP_DISTANCE:.3})")
    };
    println!("threshold: {:.3}{threshold_note}", args.threshold);
    // The pool's floor is enroll's own `PROMPT_FLOOR_SECONDS`; this crate cannot see it from
    // outside, and printing a copied decimal would be exactly the drift this harness exists
    // to prevent. Transcribe's twin is public, and TASK-059.02 pins the two to one value by
    // test, so naming that one is honest.
    println!(
        "pool:      clusters under {TENTATIVE_FLOOR_SECONDS:.0} s of speech \
         (TENTATIVE_FLOOR_SECONDS; parity with enroll's prompt floor is pinned by test)"
    );

    let mut corpus = Corpus::default();
    for dir in &args.dirs {
        let sessions = expand(dir);
        if sessions.is_empty() {
            println!(
                "\n{}\n  skipped: no session directories found",
                dir.display()
            );
        }
        for session_dir in sessions {
            if let Some(session) = measure(&session_dir) {
                report_session(&session, args.threshold, &mut corpus);
            }
        }
    }
    corpus.report(args.threshold);
}

struct Args {
    threshold: f32,
    dirs: Vec<PathBuf>,
}

/// Hand-rolled rather than clap, matching the other diagnostics in this workspace: a
/// diagnostic must never be the reason a build breaks.
fn parse() -> Result<Args, String> {
    let mut threshold = GROUP_DISTANCE;
    let mut dirs = Vec::new();
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--threshold") => {
                let raw = args.next().and_then(|v| v.into_string().ok());
                let raw = raw.ok_or("--threshold needs a value")?;
                threshold = raw
                    .parse::<f32>()
                    .map_err(|_| format!("--threshold needs a number, not {raw:?}"))?;
            }
            Some(flag) if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            _ => dirs.push(PathBuf::from(arg)),
        }
    }
    if dirs.is_empty() {
        return Err("no directories given".to_string());
    }
    Ok(Args { threshold, dirs })
}

/// One session measured: the file reads happen once, everything downstream is arithmetic on
/// these vectors.
struct Session {
    path: PathBuf,
    id: String,
    cluster_count: usize,
    below_floor: usize,
    pairs: Vec<meethook_enroll::FragmentPair>,
    skipped_incomparable: usize,
    named_rows: usize,
    has_names_file: bool,
    bundles: Vec<Vec<u32>>,
}

/// Turns each argument into sessions: a directory holding `speaker_clusters.json` is one
/// session; any other directory is a parent, and its subdirectories that hold that file are
/// sessions, in filename order so two runs see the same sequence.
fn expand(dir: &Path) -> Vec<PathBuf> {
    if dir.join("speaker_clusters.json").is_file() {
        return vec![dir.to_path_buf()];
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.join("speaker_clusters.json").is_file())
        .collect();
    found.sort();
    found
}

fn report_session(session: &Session, threshold: f32, corpus: &mut Corpus) {
    println!("\n{}", session.path.display());
    println!("  session:   {}", session.id);
    println!(
        "  clusters:  {} total, {} under the floor",
        session.cluster_count, session.below_floor
    );
    println!(
        "  pairs:     {} measured, {} skipped as incomparable embeddings",
        session.pairs.len(),
        session.skipped_incomparable
    );
    println!(
        "  labels:    {}",
        if !session.has_names_file {
            "no speaker_names.json -- same-person positives unavailable from this session"
                .to_string()
        } else if session.named_rows == 0 {
            "speaker_names.json holds no named rows -- same-person positives unavailable"
                .to_string()
        } else {
            format!("{} named row(s) in speaker_names.json", session.named_rows)
        }
    );

    let by_label = split(&session.pairs);
    print_spreads(&by_label);
    print_bundles(&session.bundles);
    println!("  pairs under candidate cuts (same | different | heard-at-once | unlabelled):");
    cut_lines(
        &by_label.same,
        &by_label.different,
        &by_label.overlapped,
        &by_label.unlabelled,
        threshold,
    );
    corpus.absorb(&by_label, &session.bundles);
}

/// The four populations, distances only, in label order.
struct ByLabel {
    same: Vec<f32>,
    different: Vec<f32>,
    overlapped: Vec<f32>,
    unlabelled: Vec<f32>,
}

fn split(pairs: &[meethook_enroll::FragmentPair]) -> ByLabel {
    let mut out = ByLabel {
        same: Vec::new(),
        different: Vec::new(),
        overlapped: Vec::new(),
        unlabelled: Vec::new(),
    };
    for pair in pairs {
        match pair.label {
            FragmentPairLabel::SamePerson => out.same.push(pair.distance),
            FragmentPairLabel::DifferentPerson => out.different.push(pair.distance),
            FragmentPairLabel::Overlapped => out.overlapped.push(pair.distance),
            FragmentPairLabel::Unlabelled => out.unlabelled.push(pair.distance),
        }
    }
    out
}

/// The per-population shapes. The unlabelled side is carried beside the labelled ones rather
/// than hidden -- a threshold read off nothing would otherwise look like it had been measured
/// against the pairs it actually governs.
fn print_spreads(by_label: &ByLabel) {
    spread_line("same person     ", &by_label.same);
    spread_line("different person", &by_label.different);
    spread_line("heard at once   ", &by_label.overlapped);
    spread_line("unlabelled      ", &by_label.unlabelled);
}

fn spread_line(label: &str, distances: &[f32]) {
    match Spread::of(distances) {
        Some(s) => println!(
            "  {label}: {} pair(s)  min {:.4}  p05 {:.4}  median {:.4}  p95 {:.4}  max {:.4}",
            s.count, s.min, s.p05, s.median, s.p95, s.max
        ),
        None => println!("  {label}: no samples"),
    }
}

/// What the shipping walk makes of this file: components at GROUP_DISTANCE, whatever
/// --threshold this run printed. The bundling is the tool's rule, not a parameter of the
/// report; re-implementing it at another cut here would be measuring a rule nothing runs.
fn print_bundles(bundles: &[Vec<u32>]) {
    let multi: Vec<&Vec<u32>> = bundles.iter().filter(|b| b.len() > 1).collect();
    let members: usize = multi.iter().map(|b| b.len()).sum();
    println!(
        "  bundles:   {} component(s) at GROUP_DISTANCE; {} multi-member, bundling {members} \
         fragment(s): {}",
        bundles.len(),
        multi.len(),
        if multi.is_empty() {
            "(all singletons)".to_string()
        } else {
            multi
                .iter()
                .map(|b| {
                    format!(
                        "[{}]",
                        b.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
                    )
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
    );
}

/// Corpus totals: every session's populations pooled, then the rates the cut is actually
/// priced with -- each one printed with its denominator, because a rate over a handful of
/// pairs must not read as a distribution.
#[derive(Default)]
struct Corpus {
    same: Vec<f32>,
    different: Vec<f32>,
    overlapped: Vec<f32>,
    unlabelled: Vec<f32>,
    components: usize,
    multi: usize,
    bundled_members: usize,
    sessions: usize,
}

impl Corpus {
    fn absorb(&mut self, by_label: &ByLabel, bundles: &[Vec<u32>]) {
        self.sessions += 1;
        self.same.extend(by_label.same.iter());
        self.different.extend(by_label.different.iter());
        self.overlapped.extend(by_label.overlapped.iter());
        self.unlabelled.extend(by_label.unlabelled.iter());
        self.components += bundles.len();
        self.multi += bundles.iter().filter(|b| b.len() > 1).count();
        self.bundled_members += bundles
            .iter()
            .filter(|b| b.len() > 1)
            .map(|b| b.len())
            .sum::<usize>();
    }

    fn report(self, threshold: f32) {
        println!("\ncorpus totals across {} session(s)", self.sessions);
        spread_line("same person     ", &self.same);
        spread_line("different person", &self.different);
        spread_line("heard at once   ", &self.overlapped);
        spread_line("unlabelled      ", &self.unlabelled);
        println!(
            "  bundles:   {} component(s) at GROUP_DISTANCE; {} multi-member, bundling {} \
             fragment(s)",
            self.components, self.multi, self.bundled_members
        );

        // Trials are rebuilt from the labelled distances at print time, which keeps the
        // library's `trial()` rule the only place a label becomes a score: heard-at-once and
        // named-different pairs both enter as negatives, unlabelled enters nowhere.
        let mut trials: Vec<meethook_transcribe::Trial> = self
            .same
            .iter()
            .map(|&distance| meethook_transcribe::Trial {
                same_speaker: true,
                distance,
            })
            .chain(
                self.different
                    .iter()
                    .chain(self.overlapped.iter())
                    .map(|&distance| meethook_transcribe::Trial {
                        same_speaker: false,
                        distance,
                    }),
            )
            .collect();
        trials.sort_by(|a, b| {
            a.same_speaker
                .cmp(&b.same_speaker)
                .then(a.distance.total_cmp(&b.distance))
        });

        println!("\nat threshold {:.3}", threshold);
        let report = score_trials(&trials, threshold);
        cost_lines(&report);

        println!("\npairs under candidate cuts (same | different | heard-at-once | unlabelled):");
        cut_lines(
            &self.same,
            &self.different,
            &self.overlapped,
            &self.unlabelled,
            threshold,
        );
    }
}

/// One line per candidate cut: how many pairs of each label class sit strictly under it --
/// the denominators the constant's documentation argues between. The unlabelled column is
/// carried here too so a reader can see how much of a widening the cut would take on without
/// ever being able to price it.
fn cut_lines(same: &[f32], different: &[f32], overlapped: &[f32], unlabelled: &[f32], run: f32) {
    for cut in CANDIDATE_CUTS {
        let under = |side: &[f32]| side.iter().filter(|d| **d < cut).count();
        let note = if cut == GROUP_DISTANCE {
            "  <- GROUP_DISTANCE"
        } else if run == cut {
            "  <- this run"
        } else {
            ""
        };
        println!(
            "    {:.3}: {:>5} | {:>5} | {:>5} | {:>5}{note}",
            cut,
            under(same),
            under(different),
            under(overlapped),
            under(unlabelled),
        );
    }
}

/// One threshold's costs, every rate printed with the population it is a fraction of, and
/// "no samples" rather than a flattering zero when a side does not exist.
fn cost_lines(report: &TrialReport) {
    match report.false_accept_rate {
        Some(rate) => println!(
            "  false accepts: {} of {} provably-different pair(s) below the cut ({:.1}%){}",
            report.false_accepts,
            report.different.map_or(0, |s| s.count),
            100.0 * rate,
            interval(
                report.false_accepts,
                report.different.map_or(0, |s| s.count)
            ),
        ),
        None => println!("  false accepts: no samples (nothing provably different on file)"),
    }
    match report.false_reject_rate {
        Some(rate) => println!(
            "  false rejects: {} of {} same-person pair(s) at or above the cut ({:.1}%){}",
            report.false_rejects,
            report.same.map_or(0, |s| s.count),
            100.0 * rate,
            interval(report.false_rejects, report.same.map_or(0, |s| s.count)),
        ),
        None => println!("  false rejects: no samples (no named positives on file)"),
    }
    match report.overlap {
        Some((min_different, max_same)) => println!(
            "  overlap:       a different-person pair at {min_different:.4} sits nearer than a \
             same-person pair at {max_same:.4}"
        ),
        None => println!("  overlap:       none between the labelled populations"),
    }
    match report.equal_error {
        Some(ee) => println!(
            "  equal error:   {:.1}% at {:.4}",
            100.0 * ee.rate,
            ee.threshold
        ),
        None => println!("  equal error:   undefined (one labelled side has no samples)"),
    }
    match report.zero_false_accept {
        Some(z) => println!(
            "  never accept a wrong pairing: cut at {:.4}, costing {:.1}% of same-person pairs",
            z.threshold,
            100.0 * z.false_reject_rate
        ),
        None => println!("  never accept a wrong pairing: undefined (needs both sides)"),
    }
}

/// The 95% Wilson interval on a printed rate, when there is a population to take one of:
/// two decimals' worth of pairs is one significant figure, and a bare "0.0%" invites acting
/// on precision the corpus does not have.
fn interval(count: usize, population: usize) -> String {
    match wilson_interval(count, population) {
        Some((low, high)) => format!(", 95% interval {:.1}%-{:.1}%", 100.0 * low, 100.0 * high),
        None => String::new(),
    }
}

fn measure(session_dir: &Path) -> Option<Session> {
    let paths = SessionPaths::new(session_dir);
    let clusters = match SpeakerClusters::read(&paths.speaker_clusters_json()) {
        Ok(clusters) => clusters,
        Err(e) => {
            println!("\n{}\n  skipped: {e}", session_dir.display());
            return None;
        }
    };

    let has_names_file = paths.speaker_names_json().is_file();
    let names = match SpeakerNames::read_or_empty(&paths, &clusters.session_id) {
        Ok(names) => names,
        Err(e) => {
            println!("\n{}\n  skipped: {e}", session_dir.display());
            return None;
        }
    };

    let below_floor = clusters
        .clusters
        .iter()
        .filter(|c| c.speech_seconds < TENTATIVE_FLOOR_SECONDS)
        .count();
    let pairs = fragment_pairs(&clusters.clusters, &names);
    // Every pool pair either got a distance or was incomparable; the difference is the count
    // the report owes, without a second convention for finding it.
    let candidate_pool = below_floor * below_floor.saturating_sub(1) / 2;
    let skipped_incomparable = candidate_pool - pairs.len();

    Some(Session {
        path: session_dir.to_path_buf(),
        id: clusters.session_id.as_str().to_string(),
        cluster_count: clusters.clusters.len(),
        below_floor,
        skipped_incomparable,
        named_rows: names.names.len(),
        has_names_file,
        pairs,
        bundles: fragment_bundles(&clusters.clusters),
    })
}
