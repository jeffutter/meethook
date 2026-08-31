//! What a person's reference should be made of (TASK-027).

use meethook_transcribe::{
    ArmReport, IDENTIFY_DISTANCE, PolicyItem, PolicyReport, PolicySweep, policy_sweep,
    wilson_interval,
};

use super::trials::print_spread;
use super::voices::Voice;

/// Prints the three reference policies scored over the same voices.
///
/// Printing only. Every count, distance and verdict comes from
/// [`meethook_transcribe::policy_sweep`], which is unit-tested inside the crate, because
/// `cargo test` builds examples without running the `#[test]`s in them -- so an arithmetic
/// convention that lived here would be a number to believe rather than evidence.
///
/// One run describes **one** cache. Two caches that disagree are two runs and a written
/// comparison, not an average.
pub fn report_policies(voices: &[Voice], threshold: f32) {
    let items: Vec<PolicyItem> = voices
        .iter()
        .map(|voice| PolicyItem {
            speaker: voice.speaker.clone(),
            session: voice.session.clone(),
            embedding: voice.embedding.clone(),
        })
        .collect();
    let sweep = policy_sweep(&items, threshold);

    report_policy_shape(&sweep);
    if sweep.reports.iter().all(|report| report.combinations == 0) {
        println!(
            "\n  nothing was scored: a two-reference arm needs a speaker with three sessions \
             -- two to enrol from and one to probe with -- in one comparable embedding space"
        );
        return;
    }

    println!("\n  closed set -- ARM A, the controlled comparison");
    println!(
        "  every speaker but the target holds one reference from one session, identically \
         across all three arms,"
    );
    println!(
        "  so the only thing varying between arms is the target person's own reference shape."
    );
    for report in &sweep.reports {
        report_arm(report, &report.controlled);
    }

    println!("\n  closed set -- ARM A', every impostor built under the policy too");
    println!(
        "  what a real user produces by naming several people twice. It varies two things at \
         once, so it does"
    );
    println!("  not replace ARM A; a disagreement between the two blocks is itself a result.");
    for report in &sweep.reports {
        report_arm(report, &report.policy_impostors);
    }

    println!("\n  distance populations -- ARM B, every speaker's reference built under the policy");
    println!(
        "  references are each speaker's first two sessions in cache order, fixed once per \
         policy rather than"
    );
    println!(
        "  re-derived per combination; probes are the sessions those references did not \
         consume. One trial per"
    );
    println!("  person, at the nearest of their references, which is what argmax sees.");
    for report in &sweep.reports {
        report_distances(report);
    }
}

fn report_policy_shape(sweep: &PolicySweep) {
    println!("\nreference policies: what one person's reference is made of after two answers");
    println!(
        "  population:  {} item(s) over {} speaker(s), embedding length(s) {:?}",
        sweep.items, sweep.speakers, sweep.dimensions
    );
    match sweep.sessions_per_speaker {
        Some(spread) => println!(
            "  sessions per speaker: min {:.0}  median {:.0}  max {:.0}  mean {:.2}",
            spread.min, spread.median, spread.max, spread.mean
        ),
        None => println!("  sessions per speaker: no items at all"),
    }
    println!(
        "  targets:     {} speaker(s) with the three sessions a two-reference arm needs{}",
        sweep.targets.len(),
        match sweep.targets.is_empty() {
            true => String::new(),
            false => format!(": {}", sweep.targets.join(" ")),
        }
    );
    println!(
        "  verdicts:    identify_clusters, argmax then threshold, fixed at IDENTIFY_DISTANCE \
         {IDENTIFY_DISTANCE:.3} -- never a bare distance comparison"
    );
    println!(
        "  distances:   scored at {:.3}; no trial pairs two recordings of one session, and \
         every refusal is counted below",
        sweep.threshold
    );

    println!("\n  trial shape");
    println!(
        "    {:<8}  {:>7}  {:>8}  {:>9}  {:>7}  {:>9}  {:>6}  {:>7}  {:>8}",
        "policy",
        "ordered",
        "distinct",
        "A refused",
        "dropped",
        "A' refused",
        "probes",
        "B pairs",
        "declines"
    );
    for report in &sweep.reports {
        println!(
            "    {:<8}  {:>7}  {:>8}  {:>9}  {:>7}  {:>9}  {:>6}  {:>7}  {:>8}",
            report.policy.label(),
            report.combinations,
            report.distinct_combinations,
            report.controlled.references_refused,
            report.controlled.impostors_dropped,
            report.policy_impostors.references_refused,
            report.distance_probes,
            report.impostor_pairs_refused,
            report.declines
        );
    }
    println!(
        "    ordered counts every pair both ways round, which only newest-wins is sensitive \
         to; distinct is the"
    );
    println!(
        "    denominator every interval below is taken on, halved for the two symmetric arms \
         so that scoring"
    );
    println!("    each of their measurements twice does not narrow it by about sqrt(2).");
    println!(
        "    refusals are counted once per probe (an impostor database depends on the probe, \
         not on the pair);"
    );
    println!("    `B pairs` is (probe, impostor) pairs refused in the distance populations.");
}

fn report_arm(report: &PolicyReport, arm: &ArmReport) {
    let scored = arm.closed.scored();
    println!(
        "    {:<8} {scored} combination(s), {} distinct",
        report.policy.label(),
        report.distinct_combinations
    );
    for (label, count) in [
        ("correct       ", arm.closed.correct),
        ("misattributed ", arm.closed.misattributed),
        ("rejected      ", arm.closed.rejected),
        ("open-set alarm", arm.open_false_alarms),
    ] {
        println!(
            "      {label}  {count:>4}  {}",
            rate_with_interval(count, scored, report)
        );
    }
    if report.policy.symmetric()
        && [
            arm.closed.correct,
            arm.closed.misattributed,
            arm.closed.rejected,
            arm.open_false_alarms,
        ]
        .iter()
        .any(|count| count % 2 != 0)
    {
        println!(
            "      NOTE: an odd count under an arm that cannot depend on the answer order. \
             The two orderings disagreed, so the distinct-count intervals above are wrong."
        );
    }
    for taken in arm.misattributions.iter().take(8) {
        println!(
            "      MISATTRIBUTED: {} / {} named as {} (reference built from {})",
            taken.speaker,
            taken.probe_session,
            taken.named,
            taken.built_from.join(" then ")
        );
    }
    for taken in arm.false_alarms.iter().take(8) {
        println!(
            "      OPEN-SET FALSE ALARM: {} / {} named as {} with {} not enrolled",
            taken.speaker, taken.probe_session, taken.named, taken.speaker
        );
    }
}

/// A rate over the ordered combinations, with its interval taken on the distinct ones.
///
/// The rate is the same number either way -- a symmetric arm's two orderings agree -- but the
/// interval is not, and quoting the ordered denominator would narrow it by about `sqrt(2)`.
fn rate_with_interval(count: usize, scored: usize, report: &PolicyReport) -> String {
    let (distinct_count, population) = match report.policy.symmetric() {
        true => (count / 2, report.distinct_combinations),
        false => (count, report.combinations),
    };
    let rate = match scored {
        0 => return "no combinations, so no rate".to_string(),
        _ => 100.0 * count as f32 / scored as f32,
    };
    match wilson_interval(distinct_count, population) {
        Some((low, high)) => format!(
            "{rate:>5.1}%  95% [{:.1}%, {:.1}%] over {population} distinct",
            low * 100.0,
            high * 100.0
        ),
        None => format!("{rate:>5.1}%  (no interval: nothing to take one over)"),
    }
}

fn report_distances(report: &PolicyReport) {
    println!(
        "    {:<8} {} probe(s)",
        report.policy.label(),
        report.distance_probes
    );
    print_spread("      own reference    ", report.distances.same.as_ref());
    print_spread("      nearest impostor ", report.nearest_impostor.as_ref());
    print_spread(
        "      every impostor   ",
        report.distances.different.as_ref(),
    );
    match report.distances.zero_false_accept {
        Some(zero) => println!(
            "      the largest cut that misattributes nobody is {:.3}, and it rejects {:.1}% \
             of same-speaker pairs",
            zero.threshold,
            zero.false_reject_rate * 100.0
        ),
        None => println!("      no misattribution-free cut is measurable from this population"),
    }
    match report.distances.overlap {
        Some((min_different, max_same)) => println!(
            "      the two populations overlap between {min_different:.3} and {max_same:.3}, \
             so no cut separates them"
        ),
        None => println!("      the two populations do not overlap"),
    }
}
