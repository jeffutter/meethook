//! The first half of the report: per-speaker within-cluster spread, the inter-speaker cosine
//! matrix, the turn-to-turn cross distances, and the known-different block -- plus the pair
//! helpers and the local [`Spread`] shape those blocks summarize.

use super::support::cosine_distance;
use meethook_transcribe::{Clustering, LocalTurn};

/// Every cluster's own turns, its within-cluster spread, and the clips to play.
pub(crate) fn print_within_cluster_spreads(
    clustering: &Clustering,
    members: &[Vec<usize>],
    turns: &[LocalTurn],
) {
    for (index, cluster) in clustering.clusters.iter().enumerate() {
        let mine = &members[index];
        let spoken: Vec<String> = mine.iter().map(|&turn| turn_span(turns, turn)).collect();
        println!(
            "\nspeaker {}: {:.1} s of speech over {} turns",
            cluster.id,
            cluster.speech_seconds,
            spoken.len()
        );
        println!("  turns: {}", spoken.join(" "));

        // How close this speaker came to being split in two. The max is the number that
        // matters: `MERGE_DISTANCE` has headroom only insofar as it sits above this.
        match spread(pairs_within(mine, &clustering.turn_embeddings)) {
            Some(within) => {
                println!(
                    "  within speaker {}: {} pairs  min {:.3}  median {:.3}  max {:.3}",
                    cluster.id, within.count, within.min, within.median, within.max
                );
                let (a, b) = within.furthest;
                println!(
                    "    furthest apart: {} vs {}  ({:.3})",
                    turn_span(turns, a),
                    turn_span(turns, b),
                    within.max
                );
            }
            None => println!(
                "  within speaker {}: {} embedded turn, no pairs",
                cluster.id,
                mine.len()
            ),
        }

        for clip in &cluster.representatives {
            println!(
                "  play:  {:>8.2} -> {:>8.2}  ({:.2} s)",
                clip.start,
                clip.end,
                clip.seconds()
            );
        }
    }
}

/// The mean-to-mean cosine distance between every pair of clusters.
pub(crate) fn print_inter_speaker_matrix(clustering: &Clustering) {
    if clustering.clusters.len() > 1 {
        println!("\ncosine distance between speakers:");
        print!("     ");
        for cluster in &clustering.clusters {
            print!("{:>7}", cluster.id);
        }
        println!();
        for row in &clustering.clusters {
            print!("{:>5}", row.id);
            for column in &clustering.clusters {
                print!(
                    "{:>7.3}",
                    cosine_distance(&row.embedding, &column.embedding)
                );
            }
            println!();
        }
    }
}

/// The closest approach of the clouds behind each pair of means, which is why both it and the
/// matrix above are printed.
pub(crate) fn print_cross_speaker_distances(
    clustering: &Clustering,
    members: &[Vec<usize>],
    turns: &[LocalTurn],
) {
    // Deliberately adjacent to the matrix above: the comparison between a mean-to-mean
    // distance and the closest approach of the two clouds behind those means is the whole
    // reason both are printed.
    println!("\nturn-to-turn distance between speakers:");
    if clustering.clusters.len() < 2 {
        // Both degenerate shapes say so out loud rather than rendering as zero lines, which
        // would be indistinguishable from the block being broken.
        println!(
            "  {} in this recording; no cluster pairs to compare",
            match clustering.clusters.len() {
                0 => "no speakers",
                _ => "only one speaker",
            }
        );
    }
    for left in 0..clustering.clusters.len() {
        for right in 0..left {
            let cross = pairs_between(&members[left], &members[right], &clustering.turn_embeddings);
            let Some(cross) = spread(cross) else {
                println!(
                    "  {} vs {}:  no pairs (one of them has no embedded turns)",
                    clustering.clusters[right].id, clustering.clusters[left].id
                );
                continue;
            };
            let (a, b) = cross.closest;
            println!(
                "  {} vs {}:  {} pairs  min {:.3}  median {:.3}   closest: {} vs {}",
                clustering.clusters[right].id,
                clustering.clusters[left].id,
                cross.count,
                cross.min,
                cross.median,
                turn_span(turns, a),
                turn_span(turns, b)
            );
        }
    }
}

/// The known-different block: same-window, different-local-speaker turn pairs, with their
/// closest approach.
pub(crate) fn print_known_different(clustering: &Clustering, turns: &[LocalTurn]) {
    // The one block no threshold stands behind. Segmentation heard these two turns at once
    // under different local speaker indices, so they are different people whatever the
    // clustering decided -- and `agglomerate` refuses to merge such a pair for that reason,
    // meaning these distances are the only different-speaker evidence here that is not
    // conditional on a grouping a reader has to trust.
    println!("\nknown-different speakers (heard in one window, different local speakers):");
    match spread(pairs_heard_at_once(turns, &clustering.turn_embeddings)) {
        Some(known) => {
            let (a, b) = known.closest;
            println!(
                "  {} pairs  min {:.3}  median {:.3}  max {:.3}   closest: {} vs {}",
                known.count,
                known.min,
                known.median,
                known.max,
                turn_span(turns, a),
                turn_span(turns, b)
            );
        }
        None => println!("  none: no two speakers were ever heard in the same window"),
    }
}

/// The shape of a set of turn-to-turn distances, and which pair sits at each end.
struct Spread {
    count: usize,
    min: f32,
    median: f32,
    max: f32,
    /// Turn indices of the pair at `min`, and of the pair at `max`. Printed as timings so
    /// the reader can play them: a distribution nobody can check against the audio is a
    /// number to believe rather than evidence.
    closest: (usize, usize),
    furthest: (usize, usize),
}

/// Summarizes distances between turn pairs, or `None` when there are no pairs at all.
///
/// The arithmetic -- and with it the conventions that matter, that the median of an even count
/// is the mean of the two middle values and that an empty population is `None` rather than
/// zeroes or NaN -- belongs to [`meethook_transcribe::Spread`], which states and tests them in
/// one place. `None` here still means what it always did: a cluster holding one turn has no
/// within-cluster distances, callers print "no pairs", and a fabricated 0.000 would read as two
/// identical turns.
///
/// What stays local is the pair of turn indices at each end, which a summary of bare numbers
/// cannot carry and should not try to: they are what turns a distance into a clip somebody can
/// play.
fn spread(pairs: Vec<(usize, usize, f32)>) -> Option<Spread> {
    let mut pairs = pairs;
    pairs.sort_by(|a, b| a.2.total_cmp(&b.2));

    let distances: Vec<f32> = pairs.iter().map(|pair| pair.2).collect();
    let summary = meethook_transcribe::Spread::of(&distances)?;

    let (first, last) = (pairs[0], pairs[pairs.len() - 1]);
    Some(Spread {
        count: summary.count,
        min: summary.min,
        median: summary.median,
        max: summary.max,
        closest: (first.0, first.1),
        furthest: (last.0, last.1),
    })
}

/// Every unordered pair of turns within one cluster, with its distance.
///
/// Turns too short to embed carry no vector and take part in nothing; they are already
/// reported as the "turns too short to embed" count on the first line.
///
/// The distance is [`cosine_distance`] -- raw, unlike the distance clustering merges on,
/// which substitutes infinity for a pair segmentation heard at once. Within a cluster the
/// difference cannot arise -- an infinite pair makes its groups' average infinite, so no
/// cluster can hold one -- but between clusters and in the known-different block it is the
/// entire point: infinity there would erase precisely the closest approaches being measured.
fn pairs_within(members: &[usize], embeddings: &[Option<Vec<f32>>]) -> Vec<(usize, usize, f32)> {
    let mut pairs = Vec::new();
    for (nth, &i) in members.iter().enumerate() {
        for &j in &members[nth + 1..] {
            if let (Some(a), Some(b)) = (&embeddings[i], &embeddings[j]) {
                pairs.push((i, j, cosine_distance(a, b)));
            }
        }
    }
    pairs
}

/// Every turn of one cluster against every turn of another.
fn pairs_between(
    left: &[usize],
    right: &[usize],
    embeddings: &[Option<Vec<f32>>],
) -> Vec<(usize, usize, f32)> {
    let mut pairs = Vec::new();
    for &i in left {
        for &j in right {
            if let (Some(a), Some(b)) = (&embeddings[i], &embeddings[j]) {
                pairs.push((i, j, cosine_distance(a, b)));
            }
        }
    }
    pairs
}

/// Pairs of turns segmentation heard in one window under different local speaker indices.
///
/// Different people by construction: the model was asked who was speaking during a single
/// ten-second window and answered with two of them. Nothing about voice embeddings or merge
/// thresholds is assumed, which is what makes these distances worth more than the rest.
fn pairs_heard_at_once(
    turns: &[LocalTurn],
    embeddings: &[Option<Vec<f32>>],
) -> Vec<(usize, usize, f32)> {
    let mut pairs = Vec::new();
    for i in 0..turns.len() {
        for j in 0..i {
            if !known_different(turns, i, j) {
                continue;
            }
            if let (Some(a), Some(b)) = (&embeddings[i], &embeddings[j]) {
                pairs.push((i, j, cosine_distance(a, b)));
            }
        }
    }
    pairs
}

/// Whether segmentation heard these two turns at once under different local speaker indices,
/// which makes them different people whatever their embeddings look like.
pub(crate) fn known_different(turns: &[LocalTurn], i: usize, j: usize) -> bool {
    turns[i].window == turns[j].window && turns[i].local_speaker != turns[j].local_speaker
}

/// "1 pair", "206 pairs".
///
/// Counts here run from zero to hundreds on the same line of the same run, and a report that
/// says "1 pairs" spends its reader's attention on wondering what else it is not being careful
/// about. Every verb near one of these is a participle for the same reason.
pub(crate) fn plural(count: usize, noun: &str) -> String {
    format!("{count} {noun}{}", if count == 1 { "" } else { "s" })
}

/// One turn's timing, so a number printed beside it can be played rather than believed.
pub(crate) fn turn_span(turns: &[LocalTurn], turn: usize) -> String {
    format!("{:.1}-{:.1}", turns[turn].start_s, turns[turn].end_s)
}
