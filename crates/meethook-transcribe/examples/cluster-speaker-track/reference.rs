//! The enrolled-reference table and the reference-duration sweep: how far stored references sit
//! from every cluster, accepted or not, and how a reference's distance from its owner grows as
//! the speech it was built from shrinks.

use super::adoption::THIN_POPULATION;
use super::clustering::turn_span;
use super::meethook_root;
use super::support::{cosine_distance, fail};
use meethook_session::{EnrolledSpeakers, Paths, SpeakerCluster};
use meethook_transcribe::{
    Clustering, IDENTIFY_DISTANCE, LocalTurn, Sampling, Sweep, Verdict, fragment_probe,
    identify_clusters, reference_duration_sweep, stored_reference_distances,
};

/// The reference durations the sweep asks for, in seconds.
///
/// Roughly Fibonacci, because the question is an order of magnitude rather than a decimal: the
/// two references this measurement exists to explain were built from 1.0 s and 1.1 s, and the
/// smallest confirmed participant here holds 51.5 s, so the grid has to cover a factor of fifty
/// without spending most of its rows in a region no reference lands in. Turns are atomic, so a
/// requested value is a lower bound on what gets realized and several of these commonly name one
/// turn set.
const DURATION_GRID: [f64; 9] = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0];

/// How much speech a held-out remainder has to hold before it is worth measuring a reference
/// against.
///
/// Below this the distance describes the noise in a thin remainder rather than the weakness of
/// the reference, which is the exact confusion this sweep exists to remove -- and it is the same
/// confusion in the mirror, since a remainder of five seconds is itself a five-second estimate of
/// a voice. 20 s is under the 30 s floor that makes a cluster a speaker at all, so no speaker is
/// excluded outright, and it costs only the top of the grid for the smaller speakers.
const MIN_HELD_OUT_SECONDS: f64 = 20.0;

/// How many below-floor clusters get the fragment probe, largest first.
///
/// The probe is the only measurement available for a cluster too small to split into a reference
/// and a remainder, and the largest few are the ones that are plausibly a real quiet participant
/// rather than a one-second scrap. Three because that is where the interesting cases are on the
/// session this was written against and every one costs a page of output.
const FRAGMENTS_PROBED: usize = 3;

/// Every cluster measured against every enrolled reference, accepted or not.
///
/// This is the number TASK-014 has to calibrate `IDENTIFY_DISTANCE` on, and it is not
/// obtainable from anything else meethook writes: identification records a similarity only
/// where it matched, so the rejections -- the half of the evidence that says how much room
/// there is above the cut -- are exactly the half that gets dropped.
///
/// The accept/reject column comes from [`identify_clusters`] rather than from comparing each
/// distance to the threshold here. Identification is argmax *then* threshold, so a reference
/// that clears the cut while a nearer one wins is not a match, and a hand-rolled comparison
/// would print it as one -- a diagnostic that disagrees with the decision it is diagnosing is
/// worse than no diagnostic.
pub(crate) fn print_enrolled_distances(clusters: &[SpeakerCluster]) {
    let paths = Paths::new(meethook_root());
    let enrolled =
        EnrolledSpeakers::read_or_empty(&paths).unwrap_or_else(|e| fail(&format!("{e}")));

    println!(
        "\ncosine distance to enrolled references in {}",
        paths.speakers_json().display()
    );
    println!("identification accepts a cluster's nearest reference below {IDENTIFY_DISTANCE:.3}");

    // Both degenerate shapes say so out loud. A section that renders as nothing is
    // indistinguishable from the feature being broken, and both of these are the *normal*
    // state of a fresh install rather than a failure.
    if enrolled.speakers.is_empty() {
        println!("  no enrolled speakers yet; run `meethook enroll` on a session first");
        return;
    }
    if clusters.is_empty() {
        println!("  no clusters in this recording; nothing to compare against");
        return;
    }

    let identified = identify_clusters(clusters, &enrolled);
    for cluster in clusters {
        let matched = identified.get(&cluster.id);
        match matched {
            Some(id) => println!("\n  speaker {} -> {}", cluster.id, id.name),
            None => println!("\n  speaker {} -> unidentified", cluster.id),
        }
        // A person is every row bearing their name, so one name prints as several lines.
        // Numbered per person, because three identical labels in a row read as one line
        // repeated rather than as three recordings of one voice.
        let mut nth: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
        for speaker in &enrolled.speakers {
            let held = enrolled.references(&speaker.name);
            let index = nth.entry(speaker.name.as_str()).or_default();
            *index += 1;
            let label = if held == 1 {
                speaker.name.clone()
            } else {
                format!("{} #{index}/{held}", speaker.name)
            };

            // A reference of a different length came from a different embedding model, and
            // `best_match` skips it for that reason. Printing a truncated `zip` of the two as
            // a distance would invent evidence about an entry identification is ignoring.
            if speaker.embedding.len() != cluster.embedding.len() {
                println!(
                    "    {label:<20} not comparable ({} dims vs the cluster's {})",
                    speaker.embedding.len(),
                    cluster.embedding.len()
                );
                continue;
            }

            // Both sides are unit vectors by contract, so the dot product is the cosine --
            // the same arithmetic `best_match` does, so the two cannot disagree.
            let distance = cosine_distance(&speaker.embedding, &cluster.embedding);
            // Only the *nearest* of a person's references wins the argmax, so "accepted" is
            // marked on that one row rather than on every row bearing the winning name --
            // which would claim identification rested on evidence it never looked at.
            //
            // Compared as `distance == 1.0 - id.similarity`, not `id.similarity == 1.0 -
            // distance`: both sides then do exactly one subtraction on the same bit-identical
            // dot product, where the reverse comparison round-trips `distance` through two
            // subtractions and loses exactness for roughly half of all f32 inputs.
            let accepted = matched
                .is_some_and(|id| id.name == speaker.name && distance == 1.0 - id.similarity);
            println!(
                "    {label:<20} {:>7.3}  {}",
                distance,
                if accepted { "accepted" } else { "rejected" }
            );
        }
    }
}

/// How far a reference sits from its owner as a function of the speech it was built from.
///
/// The block a write floor gets chosen from. Everything above measures voices that already
/// exist; this measures the *reference* -- a vector built from part of one speaker and then
/// asked to find the rest of them, against every other participant enrolled in full.
///
/// Nothing here compares a vector with one it was derived from. The turns a reference is built
/// from are removed from the remainder it is measured against, which is what the three 0.000
/// rows in the block above are missing: those are references that *are* the cluster being
/// matched, and a tautology measures nothing.
///
/// Construction and verdicts belong to `meethook_transcribe::reference_duration_sweep`, which is
/// unit tested; what is here is the printing and the three caveats that travel with every number.
pub(crate) fn print_reference_durations(clustering: &Clustering, turns: &[LocalTurn], floor: f64) {
    let sweep = reference_duration_sweep(
        turns,
        clustering,
        floor,
        &DURATION_GRID,
        MIN_HELD_OUT_SECONDS,
    );

    println!("\nreference duration sweep: how much speech a stored reference has to rest on");
    println!(
        "  the quantity: cosine distance between a reference built from d seconds of one \
         speaker's turns and the normalized mean of that speaker's REMAINING turns. Both sides \
         built the way enroll builds one -- the unweighted mean of per-turn unit embeddings, \
         normalized after averaging -- because enroll copies cluster.embedding into speakers.json \
         verbatim."
    );
    println!(
        "  held out means held out: the reference's own turns are removed from the remainder, so \
         no number below is a vector compared with one it was derived from."
    );
    println!(
        "  the verdict columns are the shipped identify_clusters over synthetic values, not a \
         comparison against IDENTIFY_DISTANCE ({IDENTIFY_DISTANCE:.3}). Identification is argmax \
         THEN threshold, so a reference that clears the cut while a nearer one wins is not a \
         match, and 'vetoed' is a name lost to the heard-at-once constraint rather than to a \
         distance."
    );
    println!(
        "    starved: this speaker enrolled from d seconds, EVERY other speaker enrolled from \
         their whole cluster -- the deployment case, and the one a floor is chosen from."
    );
    println!(
        "    all-starved: every speaker enrolled from the same grid value, every voice a held-out \
         remainder. Secondary: how the database as a whole degrades."
    );
    println!(
        "  grid {} s, requested; turns are atomic so realized is what any curve is plotted \
         against. Remainder floor {:.1} s. Speaker floor {floor:.1} s, the same partition every \
         block above uses.",
        DURATION_GRID
            .iter()
            .map(|d| format!("{d:.0}"))
            .collect::<Vec<_>>()
            .join("/"),
        sweep.min_held_out_s
    );
    println!("  three caveats, and they belong beside every number below:");
    println!(
        "    1. the selection effect. Holding out removes the tautology but not the reason those \
         turns are in that cluster -- the merge loop put them there BECAUSE it found them close. \
         So 'own' is biased low by an amount nothing within one session can measure."
    );
    println!(
        "    2. one call, one microphone, one channel is the easiest condition a reference will \
         ever face (IDENTIFY_DISTANCE's own doc comment says so); cross-session variation is \
         strictly larger."
    );
    println!(
        "    3. six speakers times a handful of durations is tens of points, well under the {} \
         pairs this report calls a thin population elsewhere. These are readings, not rates.",
        THIN_POPULATION
    );

    if sweep.speakers.is_empty() {
        println!(
            "  no cluster clears the {floor:.1} s floor, so there is no speaker here to build a \
             reference from and hold speech out of. Normal on a short recording."
        );
        return;
    }

    for sampling in Sampling::ALL {
        println!(
            "\n  sampling: {}",
            match sampling {
                Sampling::Prefix =>
                    "prefix -- the speaker's turns in start order until cumulative \
                                 speech first reaches d, which is what enrollment would get from \
                                 a caller who left the meeting early",
                Sampling::Spread =>
                    "spread -- turns nearest to k evenly spaced times across the \
                                 speaker's span, k rising until cumulative speech first reaches \
                                 d. Same seconds, scattered across the call rather than \
                                 contiguous, because one minute of one topic is not the same \
                                 object as a minute sampled from a meeting",
            }
        );

        for &speaker in &sweep.speakers {
            let cluster = &clustering.clusters[speaker as usize];
            println!(
                "\n    speaker {speaker}: {:.1} s of speech over {} turns",
                cluster.speech_seconds,
                clustering
                    .assignment
                    .iter()
                    .filter(|assigned| **assigned == Some(speaker))
                    .count()
            );
            let mut printed = 0;
            for point in sweep.arm(speaker, sampling) {
                printed += 1;
                let rival = match point.nearest_rival() {
                    Some((other, distance)) => {
                        format!("speaker {other} at {distance:.3}")
                    }
                    None => "none -- the only speaker above the floor".to_string(),
                };
                let margin = match point.margin() {
                    Some(margin) => format!("{margin:+.3}"),
                    None => "     -".to_string(),
                };
                println!(
                    "      d {:>4.0} -> {:>6.1} s over {:>2} turns, {:>6.1} s held out over {:>2}: \
                     own {:.3}   nearest rival {rival}   margin {margin}   starved {:<14} \
                     all-starved {}",
                    point.requested_s,
                    point.realized_s,
                    point.reference_turns.len(),
                    point.held_out_s,
                    point.held_out_turns,
                    point.own,
                    point.starved_alone.label(),
                    point.all_starved.label(),
                );
                println!(
                    "        that reference to every other speaker's full centroid: {}",
                    point
                        .others
                        .iter()
                        .map(|(other, distance)| format!("{other} {distance:.3}"))
                        .collect::<Vec<_>>()
                        .join("  ")
                );
            }
            if printed == 0 {
                println!("      no measurable point at any grid value; see the declined list");
            }
        }

        print_duration_summary(&sweep, sampling);
    }

    let declined = &sweep.declined;
    println!("\n  declined, {} of them:", declined.len());
    if declined.is_empty() {
        println!("    none: every speaker supported every grid value");
    }
    for point in declined {
        let reason = match point.reason {
            meethook_transcribe::Decline::ShortOfGrid { available_s } => format!(
                "short of the grid: {available_s:.1} s of embedded speech in the whole cluster"
            ),
            meethook_transcribe::Decline::ThinRemainder { held_out_s } => format!(
                "remainder of {held_out_s:.1} s is under the {:.1} s floor, so the distance \
                 would describe the remainder rather than the reference",
                sweep.min_held_out_s
            ),
            meethook_transcribe::Decline::NoDirection => {
                "a side had no direction, which is unreachable for real voices".to_string()
            }
        };
        println!(
            "    speaker {} {} d {:.0}: {reason}",
            point.speaker,
            point.sampling.label(),
            point.requested_s
        );
    }

    print_stored_references(&sweep, clustering, turns, floor);
    print_fragment_probes(clustering, floor);
}

/// Per grid value across speakers, and the band that falls out of it.
fn print_duration_summary(sweep: &Sweep, sampling: Sampling) {
    println!("\n    across speakers, {}:", sampling.label());
    println!(
        "      {:>6}  {:>6}  {:>7}  {:>13}  {:>6}  {:>12}  {:>9}",
        "d", "points", "correct", "misattributed", "vetoed", "unidentified", "worst own"
    );
    for &requested in &DURATION_GRID {
        let mine: Vec<_> = sweep
            .points
            .iter()
            .filter(|point| point.sampling == sampling && point.requested_s == requested)
            .collect();
        if mine.is_empty() {
            continue;
        }
        let count = |wanted: fn(&Verdict) -> bool| {
            mine.iter()
                .filter(|point| wanted(&point.starved_alone))
                .count()
        };
        println!(
            "      {requested:>6.0}  {:>6}  {:>7}  {:>13}  {:>6}  {:>12}  {:>9.3}",
            mine.len(),
            count(|verdict| matches!(verdict, Verdict::Correct)),
            count(|verdict| matches!(verdict, Verdict::Misattributed(_))),
            count(|verdict| matches!(verdict, Verdict::Vetoed)),
            count(|verdict| matches!(verdict, Verdict::Unidentified)),
            mine.iter()
                .map(|point| point.own)
                .fold(f32::NEG_INFINITY, f32::max)
        );
    }

    match sweep.band(sampling) {
        Some((failing, above)) => {
            println!(
                "      band: every write floor f with {failing:.1} < f <= {above:.1} refuses \
                 exactly the references that failed here and writes exactly those that did not, \
                 so any value inside it is one decision rather than a fitted point."
            );
            println!(
                "        lower edge {failing:.1} s: the largest realized duration at which some \
                 speaker's starved reference did NOT identify its own held-out speech. Below it a \
                 stored vector loses to other people's references, which makes identification \
                 worse than having no reference for that person at all."
            );
            let sacrificed = sweep.sacrificed(sampling);
            println!(
                "        upper edge {above:.1} s: the next duration any measured reference \
                 realizes. Above the band a participant who spoke that long stops contributing a \
                 reference and has to be named again in every future meeting -- {} of the \
                 measured references here worked at or below the lower edge and a floor inside \
                 the band throws them away.",
                sacrificed.len()
            );
            for point in sacrificed {
                println!(
                    "          speaker {} at {:.1} s: own {:.3}, correct",
                    point.speaker, point.realized_s, point.own
                );
            }
        }
        None => println!(
            "      band: none. Either no measured reference failed -- this session says only \
             that the floor is below everything on the grid, not where it is -- or none sat \
             above the failures, where the grid ran out before the answer did."
        ),
    }
}

/// The references already in `speakers.json`, put on the curve above rather than left as
/// anecdotes.
///
/// Two of them were enrolled from single one-second turns, which is what produced the 0.707 and
/// 0.843 this whole section exists to explain. Adoption has since pulled those turns into the
/// clusters they came from, so the number to read is the held-out one: measuring against a
/// centroid the reference is part of is the tautology again, by a slower route.
fn print_stored_references(
    sweep: &Sweep,
    clustering: &Clustering,
    turns: &[LocalTurn],
    floor: f64,
) {
    let paths = Paths::new(meethook_root());
    let enrolled =
        EnrolledSpeakers::read_or_empty(&paths).unwrap_or_else(|e| fail(&format!("{e}")));
    let stored = stored_reference_distances(&enrolled, clustering, floor);

    println!(
        "\n  the references already in {}, with the turn each was built from held out:",
        paths.speakers_json().display()
    );
    if stored.is_empty() {
        println!(
            "    none comparable: either nothing is enrolled yet, or every entry came from a \
             different embedding model and describes a different space"
        );
        return;
    }

    for reference in &stored {
        match reference.origin {
            Some((turn, apart)) => println!(
                "\n    {}: enrolled from one turn, {} ({:.2} s), which this run's embedding of \
                 that turn sits {apart:.4} from -- so the reference IS that turn and holding it \
                 out is what stops the comparison being a tautology",
                reference.name,
                turn_span(turns, turn),
                turns[turn].end_s - turns[turn].start_s,
            ),
            None => println!(
                "\n    {}: no single turn is this vector, so it was enrolled from a cluster of \
                 several and there is nothing to hold out",
                reference.name
            ),
        }
        for &(cluster, full, held_out) in &reference.against {
            let moved = if (full - held_out).abs() < 1e-6 {
                "unchanged: that cluster does not hold the origin turn"
            } else {
                "the origin turn was in this cluster"
            };
            println!(
                "      speaker {cluster:>2}: full centroid {full:.3}   held out {held_out:.3}   \
                 ({moved})"
            );
        }

        // Where that number sits on the sweep. The owner is taken to be the speaker the
        // reference is nearest to with its origin held out, which is a measurement rather than
        // an ear judgement -- and for a reference this far from everybody it is worth reading as
        // "the least bad" rather than as an identification.
        let owner = reference
            .against
            .iter()
            .copied()
            .min_by(|a, b| a.2.total_cmp(&b.2));
        let Some((owner, _, held_out)) = owner else {
            continue;
        };
        match sweep.arm(owner, Sampling::Prefix).next() {
            Some(point) => println!(
                "      on the curve: nearest speaker is {owner} at {held_out:.3}. The smallest \
                 measured prefix reference for that speaker is {:.1} s and sits {:.3} from its \
                 own held-out speech.",
                point.realized_s, point.own
            ),
            None => println!(
                "      on the curve: nearest speaker is {owner} at {held_out:.3}, which the sweep \
                 has no measured point for"
            ),
        }
    }
}

/// The below-floor clusters, used as references, which is the case a write floor is really about.
///
/// A cluster of eight seconds cannot be split into a reference and a remainder that mean
/// anything, so it never appears in the sweep. What can be measured is whether it finds the
/// nearest other cluster -- if the two really are one voice, that is a same-speaker pair with no
/// shared turns at all, which is the same held-out discipline by another route.
fn print_fragment_probes(clustering: &Clustering, floor: f64) {
    let mut below: Vec<&SpeakerCluster> = clustering
        .clusters
        .iter()
        .filter(|cluster| cluster.speech_seconds < floor)
        .collect();
    below.sort_by(|a, b| b.speech_seconds.total_cmp(&a.speech_seconds));

    println!(
        "\n  the largest below-floor clusters used as references, which is the case a write floor \
         is really about -- a real participant who barely spoke, correctly named, whose stored \
         vector then competes with references built from minutes:"
    );
    if below.is_empty() {
        println!("    none: every cluster clears the {floor:.1} s floor");
    }
    for cluster in below.into_iter().take(FRAGMENTS_PROBED) {
        let Some(probe) = fragment_probe(clustering, floor, cluster.id) else {
            println!("    speaker {}: no direction to measure", cluster.id);
            continue;
        };
        println!(
            "\n    speaker {} ({:.1} s), enrolled as though a user had named it:",
            probe.cluster, probe.seconds
        );
        println!(
            "      nearest other clusters: {}",
            probe
                .nearest
                .iter()
                .take(3)
                .map(|(id, seconds, distance)| format!("{id} ({seconds:.1} s) {distance:.3}"))
                .collect::<Vec<_>>()
                .join("   ")
        );
        println!(
            "      to every speaker above the floor: {}",
            probe
                .against_speakers
                .iter()
                .map(|(id, distance)| format!("{id} {distance:.3}"))
                .collect::<Vec<_>>()
                .join("  ")
        );
        match probe.verdict {
            Some((subject, verdict)) => println!(
                "      with this fragment enrolled under speaker {subject}'s name and every \
                 above-floor speaker enrolled in full, identify_clusters decides speaker \
                 {subject} is {}. Whether those two clusters are one person is an ear judgement \
                 this report does not make.",
                verdict.label()
            ),
            None => println!("      no other cluster to measure against"),
        }
    }
}
