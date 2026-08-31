//! The adoption-population report: the two segmentation-labelled populations of the centroid
//! distance [`ADOPTION_DISTANCE`] thresholds, scored, with the pairs behind every number, plus
//! the stranded-cluster table that prices a further sweep over what the shipped pass declined.

use super::clustering::{plural, turn_span};
use super::support::{cost_lines, separation_and_rates};
use meethook_session::SpeakerCluster;
use meethook_transcribe::{
    ADOPTION_DISTANCE, AdoptionPopulations, CentroidPair, Clustering, LocalTurn, MERGE_DISTANCE,
    PairLabel, score_trials,
};

/// Pairs below which the adoption-population block reports a population as too thin to choose a
/// threshold from.
///
/// Not a statistical criterion and not pretending to be one. It is a count at which one pair
/// moving takes the answer with it: the largest cut that misattributes nobody *is* the minimum of
/// the different-speaker side, so its uncertainty is the uncertainty of a single observation
/// however many others sit behind it, and at a couple of dozen pairs there is no reason to think
/// the closest pair observed is near the closest pair possible. A report that printed a
/// separation point from twenty pairs without saying that would be inviting a constant to be read
/// off it.
pub(crate) const THIN_POPULATION: usize = 100;

/// Every cluster still below the talk-time floor after `adopt_below_floor` has run, its distance
/// to every cluster above it, and what a further sweep of those distances would adopt.
///
/// Two blocks in one function because they share the same scaffolding -- which clusters are
/// below the floor, which are above, each one's embeddings, and whether the constraint forbids
/// the pair -- and a second function that rebuilt any of that could sweep a different population
/// from the one it printed.
///
/// Three distance columns, because the two criteria are two numbers and the third says why.
/// `linkage` is the average of the cross-pair cosine distances, which is what `agglomerate`
/// compared against [`MERGE_DISTANCE`] and declined. `centroid` is the distance between the two
/// clusters' reference vectors, which is what [`ADOPTION_DISTANCE`] thresholds here and what
/// [`IDENTIFY_DISTANCE`] thresholds against an enrolled reference. `shrinkage` is the factor
/// between them,
/// and it is the mechanism of the bug rather than a curiosity: it is at most 1 and falls as a
/// group grows and spreads, so a fragment is charged for the spread of whatever group it is
/// offered to, and the cluster most likely to own it resists it hardest.
///
/// `blocked` is a separate column and not folded into the distances. A merge the same-window
/// constraint forbids cannot be adopted however close the two look, and a merge that is merely
/// far might be adoptable under a threshold nobody has chosen yet. One number could not tell a
/// reader which of those it was looking at.
///
/// Every distance printed here is read off [`AdoptionPopulations::offers`] rather than measured
/// again. The grid is the same one [`print_adoption_populations`] scores, and two computations of
/// it could quietly diverge -- which for a report whose whole job is to be believed instead of a
/// threshold picked by ear would be the one failure that matters.
pub(crate) fn print_stranded_clusters(
    clustering: &Clustering,
    members: &[Vec<usize>],
    populations: &AdoptionPopulations,
    floor: f64,
) {
    let clusters = &clustering.clusters;
    // Folded from 0.0 rather than `sum()`, whose identity for floats is `-0.0`: an empty row of
    // the sweep below would otherwise report "-0.0 s" adopted, which reads as a bug in the
    // instrument and costs the reader more attention than the whole line is worth.
    let seconds = |group: &[u32]| -> f64 {
        group
            .iter()
            .fold(0.0, |total, &c| total + clusters[c as usize].speech_seconds)
    };
    let below = &populations.below;
    let above = &populations.above;

    println!(
        "\nstranded clusters, and where each would go (floor {floor:.1} s of speech). These are \
         what the shipped adoption pass DECLINED -- it has already run on this clustering, so \
         every row below is residue rather than the material its constant was chosen from:"
    );

    // Every degenerate shape says so in a sentence. A block that renders as no lines is
    // indistinguishable from a block that is broken, and two of these -- a recording with one
    // speaker, a recording with no speech on the speaker track -- are ordinary rather than odd.
    if clusters.is_empty() {
        println!("  no speakers in this recording; nothing is stranded");
        return;
    }
    if below.is_empty() {
        println!(
            "  nothing is stranded: every cluster ({} of them) holds at least {floor:.1} s",
            clusters.len()
        );
        return;
    }
    if above.is_empty() {
        println!(
            "  every cluster ({} of them) is under {floor:.1} s, so there is nothing to adopt \
             into -- lower --floor, or accept that this track has no dominant speaker",
            clusters.len()
        );
        return;
    }

    println!(
        "  {} of {} clusters are below the floor, holding {:.1} s of the {:.1} s that got \
         clustered; {} are above it, holding {:.1} s",
        below.len(),
        clusters.len(),
        seconds(below),
        seconds(below) + seconds(above),
        above.len(),
        seconds(above)
    );
    println!(
        "  linkage is what agglomerate merged on and declined, against its cut of \
         {MERGE_DISTANCE:.3}; centroid is what ADOPTION_DISTANCE ({ADOPTION_DISTANCE:.3}) \
         thresholds; shrinkage is the \
         factor between them, 1 - linkage = shrinkage * (1 - centroid); blocked means the \
         same-window constraint forbids this merge whatever the distances say"
    );

    if populations.declined.offers_without_distance > 0 {
        println!(
            "  {} below/above pair(s) have no distance at all -- one side has no embedded turns \
             -- and are missing from every row below",
            populations.declined.offers_without_distance
        );
    }

    // Where each fragment would go, kept for the sweep below so it thresholds the same numbers
    // it just printed. `None` is a fragment every above-floor cluster is barred from adopting.
    let mut nearest: Vec<(u32, Option<f32>)> = Vec::with_capacity(below.len());

    for &small in below {
        println!(
            "\n  cluster {} ({:.1} s, {}, first at {:.1})",
            clusters[small as usize].id,
            clusters[small as usize].speech_seconds,
            plural(members[small as usize].len(), "turn"),
            clusters[small as usize].first_spoke_seconds
        );

        let mut best: Option<(f32, u32, f32)> = None;
        for offer in offers_from(populations, small) {
            let blocked = is_blocked(offer);
            println!(
                "     -> {:<4} linkage {:.3}   centroid {:.3}   shrinkage {:.3}   {}",
                offer.large.cluster,
                offer.distance.average_linkage,
                offer.distance.centroid,
                offer.distance.shrinkage,
                if blocked { "blocked" } else { "-" }
            );
            if !blocked && best.is_none_or(|(closest, _, _)| offer.distance.centroid < closest) {
                best = Some((
                    offer.distance.centroid,
                    offer.large.cluster,
                    offer.distance.average_linkage,
                ));
            }
        }

        match best {
            Some((centroid, id, linkage)) => println!(
                "     nearest permitted: {id} at centroid {centroid:.3} (linkage {linkage:.3})"
            ),
            None => {
                println!("     nearest permitted: none -- every cluster above the floor is blocked")
            }
        }
        nearest.push((small, best.map(|(centroid, _, _)| centroid)));
    }

    // The sweep. Centroid distance, because that is what `ADOPTION_DISTANCE` thresholds and
    // saying so is the only thing keeping the two criteria from being confused for each other.
    // Argmax among permitted targets and then the cut, matching `adopt_below_floor`; centroids
    // frozen, because that pass adopts in one pass and does not re-centroid as it goes. An
    // iterative pass would adopt more and would need its own sweep.
    //
    // Read as a *further* sweep: the shipped pass has already taken everything under its own cut,
    // so the row at that cut adopts nothing and the rows above it price widening it.
    println!(
        "\n  further adoption sweep over centroid distance, one pass, argmax among permitted \
         targets then the cut. The shipped pass has already run at {ADOPTION_DISTANCE:.3}, so \
         this prices widening it rather than choosing it:"
    );
    println!("    threshold   adopted              remaining            clusters after");
    for step in 4..=16 {
        let threshold = step as f32 * 0.05;
        let adopted: Vec<u32> = nearest
            .iter()
            .filter(|(_, centroid)| centroid.is_some_and(|centroid| centroid < threshold))
            .map(|&(cluster, _)| cluster)
            .collect();
        let remaining: Vec<u32> = below
            .iter()
            .copied()
            .filter(|cluster| !adopted.contains(cluster))
            .collect();
        println!(
            "    {threshold:>9.3}   {:>3} ({:>6.1} s)      {:>3} ({:>6.1} s)      {:>3}",
            adopted.len(),
            seconds(&adopted),
            remaining.len(),
            seconds(&remaining),
            above.len() + remaining.len()
        );
    }

    let unadoptable: Vec<u32> = nearest
        .iter()
        .filter(|(_, centroid)| centroid.is_none())
        .map(|&(cluster, _)| cluster)
        .collect();
    println!(
        "  no permitted target at any threshold: {} ({:.1} s)",
        plural(unadoptable.len(), "cluster"),
        seconds(&unadoptable)
    );

    // The ceiling on any centroid threshold, and the reason the sweep is printed with one.
    // Two above-floor clusters are two speakers a person confirmed; a cut wider than the gap
    // between the closest of them is measuring a distance two different people fit inside.
    match populations.ceiling() {
        Some((centroid, left, right)) => println!(
            "  the closest two clusters above the floor ({left} and {right}) sit {centroid:.3} \
             apart by centroid; a cut at or above that is measuring a gap two separate speakers \
             fit inside"
        ),
        None => {
            println!("  only one cluster above the floor, so there is no gap to compare against")
        }
    }
}

/// One below-floor cluster's row of the offer grid, in the grid's own order.
fn offers_from(
    populations: &AdoptionPopulations,
    small: u32,
) -> impl Iterator<Item = &CentroidPair> {
    populations
        .offers
        .iter()
        .filter(move |offer| offer.small.cluster == small)
}

/// Whether the same-window constraint bars this offer, which is also what labels it a
/// different-speaker pair. One question, so one spelling of it.
fn is_blocked(offer: &CentroidPair) -> bool {
    matches!(offer.label, PairLabel::CannotLink { .. })
}

/// The two segmentation-labelled populations of the distance [`ADOPTION_DISTANCE`] thresholds,
/// scored, with the pairs behind every number.
///
/// Measured on the clustering that ships, which is the one `adopt_below_floor` has already swept:
/// the positives are now classes inside a cluster the pass may have added fragments to, and the
/// offer grid holds only what it declined. So this verifies the constant, and the numbers it was
/// *chosen* from are the ones recorded on TASK-018.02.02 from before the pass existed.
///
/// The blocks above print the raw material and never join it: every below-floor cluster against
/// every above-floor one with a `blocked` column, and every must-link pair with its cluster ids.
/// This is the join -- **two labelled populations of one quantity**, which is what a threshold can
/// be chosen from and a table read by eye cannot be.
///
/// The quantity is centroid distance, a small group's normalized mean against a larger group's,
/// and the section says so on every line that prints a number. `MERGE_DISTANCE` thresholds the
/// other quantity, average linkage; [`ADOPTION_DISTANCE`] thresholds this one, for this decision,
/// and [`IDENTIFY_DISTANCE`] thresholds it for another. The first two were once the
/// same number, and TASK-020 is the bug that cost -- one confirmed pair of speakers landing on
/// opposite sides of the two criteria at a shared 0.45 -- which is why the insistence is worth
/// the words even now that the values differ.
///
/// Nothing here decides anything. The construction and its exclusions live in
/// `meethook_transcribe::adoption_populations`, where they are unit-tested, and the scoring lives
/// in [`score_trials`], which states and tests the boundary convention -- accept is *strictly*
/// below the cut -- so that this report cannot state a different one than the decision it informs
/// would use. Wherever this section prints the same arithmetic `speaker-trials` does, it calls
/// the same shared printers (`support::cost_lines`, `support::separation_and_rates`); what stays
/// local is the prose only this section can say.
pub(crate) fn print_adoption_populations(
    clustering: &Clustering,
    turns: &[LocalTurn],
    populations: &AdoptionPopulations,
    floor: f64,
    cut: f32,
) {
    let clusters = &clustering.clusters;
    let declined = &populations.declined;
    let negatives: Vec<&CentroidPair> = populations.negatives().collect();
    let ceiling = populations.ceiling();

    println!("\nadoption populations: the two labelled populations of one distance");
    println!(
        "  the quantity, everywhere in this section: CENTROID distance -- a small group's \
         normalized mean against a larger group's, cosine, 0 for the same direction. Not \
         turn-to-turn, and not the average linkage agglomerate merged on. MERGE_DISTANCE \
         ({MERGE_DISTANCE:.3}) governs that other number and governs nothing below; linkage and \
         shrinkage are printed beside each pair so that neither can be read as the other."
    );

    if populations.positives.is_empty() && negatives.is_empty() {
        println!(
            "  neither population exists on this recording, so there is nothing here to choose a \
             threshold from. Both need clusters on either side of the floor and segmentation \
             judgements joining them; a track with one speaker has neither."
        );
        return;
    }

    // ------------------------------------------------------------------------------------
    // What was built, and what was left out of it
    // ------------------------------------------------------------------------------------
    println!("\n  construction:");
    println!(
        "    positives, labelled same-speaker by segmentation alone: leave-one-class-out. A \
         must-link class -- two or more embedded turns sharing one (window, local speaker), which \
         segmentation heard as one person -- that landed wholly inside a cluster above the \
         {floor:.1} s floor, measured against the REST of that cluster. The class is excluded \
         from the residual; left in, it would bias the mean it is being compared to."
    );
    println!(
        "    the weak leg of that label, and it belongs beside every number derived from it: \
         segmentation says the class's own turns are one person. That the class and the residual \
         are one person is CLUSTERING's claim -- seeding forced the class together, and the merge \
         loop is what put it in that cluster. The within-class cross-check at the end of this \
         section is the population with no such leg."
    );
    println!(
        "    negatives, labelled different-speaker by the same-window cannot-link constraint: a \
         below-floor cluster the constraint bars from an above-floor cluster, because \
         segmentation heard one turn of each at once under different local speaker indices. The \
         witness turn pair is printed with each one so it can be played rather than believed."
    );
    println!(
        "    the caveat that belongs beside any cut derived from the negatives: every one of them \
         is a pair the constraint ALREADY refuses, so a cut read off them prices a distance-only \
         rule the pass does not use. What they do measure is how close two provably different \
         people's centroids come at this granularity -- which is what bounds trust in a cut on \
         the unblocked pairs, where no constraint protects anybody."
    );
    println!(
        "    excluded from the positives, counted rather than dropped, of {} (window, local \
         speaker) classes in all: {} of a single embedded turn (one turn is no pair, so no \
         must-link assertion at all -- which cluster it joined is clustering's decision), {} \
         inside a below-floor cluster (no speaker-scale residual to measure against), {} that are \
         their whole cluster (no residual left), {} split across clusters (seeding makes that \
         impossible, so this must read 0), {} with no direction. {} positives.",
        populations.classes,
        declined.single_turn,
        declined.below_floor,
        declined.whole_cluster,
        declined.split_across_clusters,
        declined.no_direction,
        populations.positives.len()
    );

    // ------------------------------------------------------------------------------------
    // The two populations
    // ------------------------------------------------------------------------------------
    let positive_distances = centroids(populations.positives.iter());
    let negative_distances = centroids(negatives.iter().copied());

    println!("\n  centroid distance, the two labelled populations:");
    print_centroid_spread(
        "same speaker      (leave-one-class-out)",
        &positive_distances,
    );
    print_centroid_spread(
        "different speakers (cannot-link)       ",
        &negative_distances,
    );
    println!(
        "    {} below-floor/above-floor offers in all, {} of them blocked and so labelled; the \
         other {} carry no label in either direction and are scored as neither",
        populations.offers.len(),
        negatives.len(),
        populations.offers.len() - negatives.len()
    );

    println!("\n  the same-speaker pairs, each class against the rest of its own cluster:");
    for pair in &populations.positives {
        let (window, local_speaker) = pair
            .label
            .class()
            .expect("a positive is labelled by its must-link class");
        println!(
            "    cluster {:<3} window {:>4} local speaker {}   centroid {:.3}   (linkage {:.3}, \
             shrinkage {:.3})   class {:.1} s over {} vs residual {:.1} s over {}",
            pair.large.cluster,
            window,
            local_speaker,
            pair.distance.centroid,
            pair.distance.average_linkage,
            pair.distance.shrinkage,
            pair.small.seconds,
            plural(pair.small.turns.len(), "turn"),
            pair.large.seconds,
            plural(pair.large.turns.len(), "turn")
        );
        println!("        class turns: {}", spans(turns, &pair.small.turns));
    }

    println!("\n  the different-speaker pairs, each fragment against a speaker it is barred from:");
    for pair in &negatives {
        let PairLabel::CannotLink { witness } = pair.label else {
            continue;
        };
        let (earlier, later) = witness;
        println!(
            "    {:>3} vs {:<3} centroid {:.3}   (linkage {:.3}, shrinkage {:.3})   fragment \
             {:.1} s over {}, {}",
            pair.small.cluster,
            pair.large.cluster,
            pair.distance.centroid,
            pair.distance.average_linkage,
            pair.distance.shrinkage,
            pair.small.seconds,
            plural(pair.small.turns.len(), "turn"),
            purity(pair.small.classes)
        );
        println!(
            "        heard at once in window {}: {} (local speaker {}) / {} (local speaker {});  \
             fragment turns: {}",
            turns[earlier].window,
            turn_span(turns, earlier),
            turns[earlier].local_speaker,
            turn_span(turns, later),
            turns[later].local_speaker,
            spans(turns, &pair.small.turns)
        );
    }

    // ------------------------------------------------------------------------------------
    // Scored, through the one place that states the conventions
    // ------------------------------------------------------------------------------------
    let trials = populations.trials();
    let report = score_trials(&trials, cut);

    println!(
        "\n  scored through score_trials, which states and tests the conventions: accept is \
         STRICTLY below the cut, a false accept is a different-speaker pair below it and a false \
         reject a same-speaker pair at or above it, percentiles are nearest-rank."
    );
    println!(
        "    at the cut {:.3} -- ADOPTION_DISTANCE unless --cut moved it, so these two counts \
         price the shipped pass on the populations it was chosen from. Read them with the \
         negatives' caveat above: every different-speaker pair here is one the constraint \
         already refuses and the pass never offers, so a false accept in this column is a cut a \
         DISTANCE-ONLY rule would have got wrong, not one this pass does:",
        report.threshold
    );
    cost_lines("      ", &report);

    println!("\n  separation:");
    separation_and_rates("    ", &report);
    if let Some(zero) = &report.zero_false_accept {
        // Priced as well as named. The asymmetry argument the adoption constant will rest on
        // -- a silent misattribution is expensive and a visible extra Unknown N is cheap --
        // is only an argument until somebody says what the cheap error costs in pairs.
        let priced = score_trials(&trials, zero.threshold);
        println!("    what refusing to misattribute anybody costs, at that cut:");
        cost_lines("      ", &priced);
    }

    // The thinness statement, immediately under the cut it is about, because the cut is the
    // minimum of one population and therefore rests on one pair however many are behind it.
    let thin = populations.positives.len() < THIN_POPULATION || negatives.len() < THIN_POPULATION;
    if thin {
        println!(
            "\n    TOO THIN TO CHOOSE A CONSTANT FROM: {} same-speaker and {} different-speaker \
             pairs, against the {THIN_POPULATION} this report treats as the point where one pair \
             stops moving the answer. The misattribution-free cut IS the minimum of the \
             different-speaker side, so it is a single observation: it moves wherever that one \
             pair moves, and nothing here says the closest pair seen is near the closest pair \
             possible. Read it as a bound to check by ear, not as a number to ship.",
            populations.positives.len(),
            negatives.len()
        );
    }
    if !negative_distances.is_empty() {
        println!("    the three closest different-speaker pairs, which are the ones it rests on:");
        let mut closest: Vec<&&CentroidPair> = negatives.iter().collect();
        closest.sort_by(|a, b| a.distance.centroid.total_cmp(&b.distance.centroid));
        for pair in closest.iter().take(3) {
            println!(
                "      {:>3} vs {:<3} centroid {:.3}   fragment {:.1} s over {}, {}   {}",
                pair.small.cluster,
                pair.large.cluster,
                pair.distance.centroid,
                pair.small.seconds,
                plural(pair.small.turns.len(), "turn"),
                purity(pair.small.classes),
                spans(turns, &pair.small.turns)
            );
        }
    }

    // ------------------------------------------------------------------------------------
    // The pairs a cut would be deciding, whether or not any label covers them
    // ------------------------------------------------------------------------------------
    print_contested_targets(clusters, turns, populations, ceiling);

    // ------------------------------------------------------------------------------------
    // The floor, which decides what counts as a speaker before any distance is looked at
    // ------------------------------------------------------------------------------------
    println!("\n  the talk-time floor, and the band of floors giving this same partition:");
    println!(
        "    the convention is speech_seconds < floor is a fragment and >= floor is a speaker, so \
         a cluster sitting exactly ON the floor is a speaker"
    );
    for cluster in clusters {
        println!(
            "    cluster {:<3} {:>7.1} s   {}",
            cluster.id,
            cluster.speech_seconds,
            if populations.above.contains(&cluster.id) {
                "speaker"
            } else {
                "fragment"
            }
        );
    }
    match populations.floor_band {
        Some((max_below, min_above)) => {
            println!(
                "    any floor f with {max_below:.1} < f <= {min_above:.1} produces exactly this \
                 partition -- {} speakers holding the talk time and {} fragments -- so the \
                 --floor {floor:.1} this ran at is insensitive across a {:.1} s band rather than \
                 tuned to the answer",
                populations.above.len(),
                populations.below.len(),
                min_above - max_below
            );
            println!(
                "    both edges are consequences, not trivia: at f <= {max_below:.1} the largest \
                 fragment becomes an adoption TARGET on {max_below:.1} s of speech, which is the \
                 failure the floor exists to prevent; above {min_above:.1} the smallest speaker \
                 stops being one and its {min_above:.1} s go looking for an owner"
            );
        }
        None => println!(
            "    no band: one side of the floor is empty on this recording, so there is no \
             partition for a range of floors to preserve"
        ),
    }

    // ------------------------------------------------------------------------------------
    // The two populations that are the right label or the right scale, but never both
    // ------------------------------------------------------------------------------------
    println!("\n  cross-check -- purest label, WRONG SHAPE: leave-one-TURN-out inside a class.");
    println!(
        "    One embedded turn of a must-link class against the rest of that class, in any \
         cluster, above the floor or below it. Nothing but segmentation stands behind the label, \
         which is what the leave-one-class-out positives cannot say. Its shape is wrong: a class \
         fits inside one ten-second window, so neither side is a speaker-scale mean and the \
         shrinkage differs. It says whether the positives LOOK like a same-speaker population. It \
         is not a substitute for them and is not in the trial list above."
    );
    print_centroid_spread(
        "within one class  (turn vs rest of class)",
        &centroids(populations.within_class.iter()),
    );

    println!("\n  auxiliary negatives -- right scale, label from a person rather than the model:");
    println!(
        "    every pair of clusters above the floor. Large against large is a third shape and \
         folding it into the trial list would flatter the separation, so it is scored on its own. \
         What it licenses is the ceiling: two clusters above the floor are two people, so a cut at \
         or above the gap between the closest of them is measuring a distance two speakers fit \
         inside."
    );
    print_centroid_spread(
        "speaker vs speaker (above the floor)   ",
        &centroids(populations.above_floor.iter()),
    );
    for pair in &populations.above_floor {
        println!(
            "      {:>3} vs {:<3} centroid {:.3}   (linkage {:.3}, shrinkage {:.3})   {:.1} s vs \
             {:.1} s",
            pair.small.cluster,
            pair.large.cluster,
            pair.distance.centroid,
            pair.distance.average_linkage,
            pair.distance.shrinkage,
            pair.small.seconds,
            pair.large.seconds
        );
    }
    match ceiling {
        Some((centroid, left, right)) => {
            println!("    the ceiling is {centroid:.3}, clusters {left} and {right}")
        }
        None => println!(
            "    no ceiling: fewer than two clusters above the floor, so nothing here bounds a cut"
        ),
    }
}

/// Every below-floor cluster a cut is actually deciding about: those closer to some speaker than
/// the two closest speakers are to each other.
///
/// The selection rule is structural rather than a margin heuristic, and rather than a hardcoded
/// cluster id, so it stays true on a session where the ids are different. Under the ceiling means
/// the fragment sits nearer a speaker than two known-different speakers sit to each other, so no
/// cut can take it without also being wide enough to have merged those two -- which is exactly
/// the set of pairs a threshold choice is a choice about.
///
/// On session `20260810-093047` cluster 12 lands here at 0.296 under column 1 with 0.550 as its
/// second choice. It is not ambiguous by margin, which is why a margin rule would have missed it:
/// what makes it the pair the ceiling turns on is external knowledge -- it may be a seventh
/// speaker whose stored reference is the old blended cluster's centroid, so the enrolled distances
/// that would settle it are circular. No report can settle that. This one only has to make it
/// impossible to miss.
fn print_contested_targets(
    clusters: &[SpeakerCluster],
    turns: &[LocalTurn],
    populations: &AdoptionPopulations,
    ceiling: Option<(f32, u32, u32)>,
) {
    println!("\n  contested targets: the fragments a cut is really deciding about.");
    match ceiling {
        Some((centroid, left, right)) => println!(
            "    Rows are every below-floor cluster whose nearest PERMITTED speaker sits under the \
             ceiling of {centroid:.3} -- the gap between clusters {left} and {right}, two \
             different people -- ordered by that distance. A fragment in this table is closer to \
             some speaker than those two are to each other, so no cut adopts it without being \
             wide enough to have merged them."
        ),
        None => println!(
            "    Fewer than two clusters above the floor, so there is no ceiling to select on and \
             every fragment with a permitted target is listed."
        ),
    }
    println!(
        "    Cells are centroid distance; 'b' marks a pair the same-window constraint bars \
         whatever the distance says, and blocked pairs are not eligible to be a nearest choice. \
         'nearest' and 'second' are the two closest permitted speakers and 'margin' the gap \
         between them."
    );

    // Rows, and the two closest permitted targets for each.
    let mut rows: Vec<(u32, f32, Option<f32>, u32)> = Vec::new();
    for &small in &populations.below {
        let mut permitted: Vec<(f32, u32)> = offers_from(populations, small)
            .filter(|offer| !is_blocked(offer))
            .map(|offer| (offer.distance.centroid, offer.large.cluster))
            .collect();
        permitted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let Some(&(nearest, target)) = permitted.first() else {
            continue;
        };
        if ceiling.is_some_and(|(ceiling, _, _)| nearest >= ceiling) {
            continue;
        }
        rows.push((small, nearest, permitted.get(1).map(|&(d, _)| d), target));
    }
    rows.sort_by(|a, b| a.1.total_cmp(&b.1));

    if rows.is_empty() {
        println!(
            "    none: every fragment's nearest permitted speaker is further away than two \
             speakers are from each other, so no cut under the ceiling adopts anything"
        );
        return;
    }

    print!(
        "    {:>7}  {:>8}  {:>8}  {:>7}  {:>7}  {:>7} |",
        "cluster", "seconds", "first at", "nearest", "second", "margin"
    );
    for &large in &populations.above {
        print!("{large:>8}");
    }
    println!();
    for &(small, nearest, second, target) in &rows {
        let cluster = &clusters[small as usize];
        print!(
            "    {:>7}  {:>8.1}  {:>8.1}  {:>7.3}  {:>7}  {:>7} |",
            small,
            cluster.speech_seconds,
            cluster.first_spoke_seconds,
            nearest,
            match second {
                Some(second) => format!("{second:.3}"),
                None => "-".to_string(),
            },
            match second {
                Some(second) => format!("{:.3}", second - nearest),
                None => "-".to_string(),
            }
        );
        for &large in &populations.above {
            let cell = offers_from(populations, small)
                .find(|offer| offer.large.cluster == large)
                .map(|offer| {
                    format!(
                        "{:.3}{}",
                        offer.distance.centroid,
                        if is_blocked(offer) { "b" } else { " " }
                    )
                })
                .unwrap_or_else(|| "-  ".to_string());
            print!("{cell:>8}");
        }
        println!();
        println!(
            "        cluster {small} nearest permitted {target} at {nearest:.3}; turns: {}",
            spans(
                turns,
                offers_from(populations, small)
                    .next()
                    .map(|offer| offer.small.turns.as_slice())
                    .unwrap_or_default()
            )
        );
    }
}

/// Every pair's centroid distance, which is the only field a spread of these is ever over.
fn centroids<'a>(pairs: impl Iterator<Item = &'a CentroidPair>) -> Vec<f32> {
    pairs.map(|pair| pair.distance.centroid).collect()
}

/// A population's shape, labelled `centroid` because that is the only quantity here.
fn print_centroid_spread(label: &str, distances: &[f32]) {
    match meethook_transcribe::Spread::of(distances) {
        Some(s) => println!(
            "    {label}: {} centroid pair(s)  min {:.3}  p05 {:.3}  median {:.3}  p95 {:.3}  \
             max {:.3}  mean {:.3}",
            s.count, s.min, s.p05, s.median, s.p95, s.max, s.mean
        ),
        None => println!("    {label}: no pairs"),
    }
}

/// Whether segmentation grouped this fragment itself, or embedding assembled it across windows.
///
/// The distinction decides what a different-speaker label on it is worth. One class means
/// segmentation heard these turns in one window under one index, so the fragment is a small group
/// of one person. Several means the clustering built it out of turns from different windows, and
/// it may be a blend belonging to nobody -- in which case its label is only as good as the
/// clustering, which is the one thing the labelled populations exist to avoid depending on.
fn purity(classes: usize) -> String {
    match classes {
        1 => "one (window, local speaker) class, so segmentation itself grouped it".to_string(),
        several => format!(
            "{several} classes across windows, assembled by embedding -- may be a blend belonging \
             to nobody"
        ),
    }
}

/// Several turns' timings on one line, so a group of numbers can be played rather than believed.
fn spans(turns: &[LocalTurn], held: &[usize]) -> String {
    held.iter()
        .map(|&turn| turn_span(turns, turn))
        .collect::<Vec<String>>()
        .join(" ")
}
