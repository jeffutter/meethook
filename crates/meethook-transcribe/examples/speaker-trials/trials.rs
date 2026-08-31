//! Pairing and reporting.

use std::collections::BTreeMap;

use meethook_transcribe::{Spread, Trial, TrialReport};

use super::support::{cosine_distance, cost_lines, separation_and_rates};
use super::voices::Voice;

pub struct TrialList {
    pub trials: Vec<Trial>,
    /// Pairs refused because both items came from one recording session.
    within_session: usize,
}

/// Every legal unordered pair of items.
///
/// The one rule with teeth: **two items from the same session are never a trial**, whoever
/// they are. For two items of one speaker that is `MERGE_DISTANCE`'s question rather than this
/// one; for two speakers recorded in a single session it is a pair that shares a microphone,
/// a room and a codec, which is exactly the variation a cross-session threshold has to survive
/// and would therefore be measured too favourably.
///
/// Unordered and counted once, because cosine distance is symmetric.
pub fn pair_up(voices: &[Voice]) -> TrialList {
    let mut trials = Vec::new();
    let mut within_session = 0;
    for (index, a) in voices.iter().enumerate() {
        for b in &voices[..index] {
            if a.session == b.session {
                within_session += 1;
                continue;
            }
            trials.push(Trial {
                same_speaker: a.speaker == b.speaker,
                distance: cosine_distance(&a.embedding, &b.embedding),
            });
        }
    }
    TrialList {
        trials,
        within_session,
    }
}

/// The shape of the trial list, printed before any statistic taken over it.
///
/// A trial list whose shape is not stated cannot be checked, and every published number of
/// this kind is quoted alongside its trial count.
pub fn report_shape(voices: &[Voice], trials: &TrialList) {
    let mut sessions: BTreeMap<&str, usize> = BTreeMap::new();
    for voice in voices {
        *sessions.entry(voice.speaker.as_str()).or_default() += 1;
    }
    let per_speaker: Vec<f32> = sessions.values().map(|count| *count as f32).collect();

    let same = trials
        .trials
        .iter()
        .filter(|trial| trial.same_speaker)
        .count();

    println!("\ntrial list");
    println!(
        "  {} item(s) over {} speaker(s)",
        voices.len(),
        sessions.len()
    );
    match Spread::of(&per_speaker) {
        Some(spread) => println!(
            "  sessions per speaker: min {:.0}  median {:.0}  max {:.0}",
            spread.min, spread.median, spread.max
        ),
        None => println!("  sessions per speaker: no items at all"),
    }
    println!("  {same} same-speaker pair(s)");
    println!("  {} different-speaker pair(s)", trials.trials.len() - same);
    println!(
        "  {} pair(s) refused for sharing one session (MERGE_DISTANCE's question, not this one)",
        trials.within_session
    );

    if same == 0 {
        println!(
            "  note: no same-speaker pairs. Every speaker in this manifest was recorded in one \
             session only, so nothing here measures whether one voice matches itself."
        );
    }
    if trials.trials.len() == same {
        println!(
            "  note: no different-speaker pairs. This manifest names one speaker, so nothing \
             here measures whether two voices are told apart."
        );
    }
}

pub fn report_scores(report: &TrialReport) {
    println!("\ndistances");
    print_spread("same speaker     ", report.same.as_ref());
    print_spread("different speaker", report.different.as_ref());

    println!("\nat threshold {:.3}", report.threshold);
    cost_lines("  ", report);

    println!("\nseparation");
    separation_and_rates("  ", report);
}

pub fn print_spread(label: &str, spread: Option<&Spread>) {
    match spread {
        Some(s) => println!(
            "  {label}: {} pair(s)  min {:.3}  p05 {:.3}  median {:.3}  p95 {:.3}  max {:.3}  \
             mean {:.3}",
            s.count, s.min, s.p05, s.median, s.p95, s.max, s.mean
        ),
        None => println!("  {label}: no pairs"),
    }
}
