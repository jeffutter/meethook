//! The must-link split report: turn pairs segmentation heard in one window under the *same*
//! local speaker index, and what clustering did with them -- plus the transitive cluster merge
//! that those pairs would imply if cannot-link never vetoed.

use super::clustering::{known_different, plural, turn_span};
use meethook_transcribe::{Clustering, LocalTurn};

/// Turn pairs segmentation heard in one window under the same local speaker index, and what
/// clustering did with them.
///
/// The must-link direction of the constraint. Different indices in one window are different
/// people; the *same* index in one window is one person, on exactly the same authority. Windows do
/// not overlap and segmentation closes and reopens a turn for one index whenever the silence
/// inside it runs past a quarter second, so a pair like this is two turns of one speaker rather
/// than an artefact of the decoder -- and wherever clustering put such a pair in two clusters, it
/// is wrong, with no embedding, no threshold and no model standing behind the claim.
///
/// `agglomerate` reads that direction now: it seeds its groups by `(window, local_speaker)`, so
/// every embedded pair this block finds is in one cluster by construction. That makes the block a
/// regression check rather than the opportunity report it was written as -- **the line to read is
/// "split across two", and it must be 0.** Anything else means seeding is broken, and no unit test
/// says so as directly, because this is the constraint measured on the audio it came from.
///
/// The other two counts still say something. "Already in one cluster" is the population the
/// seeding covers, and the pairs with a turn too short to embed are out of its reach in principle:
/// `constraints` only carries embedded turns, so lowering `MIN_EMBEDDABLE_SECONDS` is the only
/// thing that would reach them, and that is a separate judgement.
///
/// The last line -- how many clusters transitive cluster-level merging of these pairs would
/// leave -- is kept because it is not the same operation the library performs. It merges whole
/// clusters, which can drag turns from other windows along with the pair; seeding merges turns.
/// Now that the split count should be 0 it will normally have nothing to do, and a session where
/// it does is a session where the two operations disagree.
pub(crate) fn print_must_link_splits(
    clustering: &Clustering,
    turns: &[LocalTurn],
    members: &[Vec<usize>],
) {
    println!("\nmust-link pairs (heard in one window under the same local speaker index):");

    let mut together = 0usize;
    let mut unembedded = 0usize;
    let mut split: Vec<(usize, usize)> = Vec::new();
    for i in 0..turns.len() {
        for j in 0..i {
            if !same_local_speaker(turns, i, j) {
                continue;
            }
            match (clustering.assignment[j], clustering.assignment[i]) {
                (Some(left), Some(right)) if left == right => together += 1,
                (Some(_), Some(_)) => split.push((j, i)),
                // No cluster on at least one side, so nothing here can sweep it either way.
                // Counted rather than dropped: it is still one speaker's speech going
                // unattributed, and the first line of this report already totals those turns.
                _ => unembedded += 1,
            }
        }
    }

    let population = together + unembedded + split.len();
    if population == 0 {
        println!("  none: no local speaker was ever heard twice in one window");
        return;
    }
    println!(
        "  {}: {together} already in one cluster, {} split across two, {unembedded} with a turn \
         too short to embed",
        plural(population, "pair"),
        split.len()
    );
    if split.is_empty() {
        println!("  clustering split none of them, so there is nothing to gain here");
        return;
    }

    let mut involved: Vec<usize> = split.iter().flat_map(|&(j, i)| [j, i]).collect();
    involved.sort_unstable();
    involved.dedup();
    let speech: f64 = involved
        .iter()
        .map(|&turn| turns[turn].end_s - turns[turn].start_s)
        .sum();
    println!(
        "  the split ones, covering {} and {speech:.1} s of speech:",
        plural(involved.len(), "distinct turn")
    );
    for &(earlier, later) in &split {
        println!(
            "    window {:>4}  local speaker {}  clusters {} and {}   {} / {}",
            turns[earlier].window,
            turns[earlier].local_speaker,
            cluster_of(clustering, earlier),
            cluster_of(clustering, later),
            turn_span(turns, earlier),
            turn_span(turns, later)
        );
    }

    let mut pairs: Vec<(usize, usize)> = split
        .iter()
        .map(|&(earlier, later)| {
            let (left, right) = (
                cluster_of(clustering, earlier) as usize,
                cluster_of(clustering, later) as usize,
            );
            (left.min(right), left.max(right))
        })
        .collect();
    pairs.sort_unstable();
    pairs.dedup();

    // Applied transitively, because merging A into B and B into C says A and C are one speaker
    // too, and with the cannot-link constraint winning any conflict: a component that has come
    // to hold two turns heard at once cannot absorb another, and segmentation is the authority
    // on both sides of that. Cluster ids index `members` -- `cluster_speaker_turns` numbers them
    // by position -- so the two are the same key throughout.
    let mut parent: Vec<usize> = (0..clustering.clusters.len()).collect();
    let (mut merges, mut vetoed) = (0usize, 0usize);
    for &(left, right) in &pairs {
        let (left, right) = (root(&mut parent, left), root(&mut parent, right));
        if left == right {
            continue;
        }
        let (mine, theirs) = (
            component_turns(&mut parent, left, members),
            component_turns(&mut parent, right, members),
        );
        if heard_apart(&mine, &theirs, turns) {
            vetoed += 1;
            continue;
        }
        parent[right] = left;
        merges += 1;
    }
    let mut left_over = 0usize;
    for cluster in 0..clustering.clusters.len() {
        if root(&mut parent, cluster) == cluster {
            left_over += 1;
        }
    }

    println!(
        "  {} implicated; applied transitively with cannot-link winning any conflict, {} applied \
         and {vetoed} refused, leaving {} of {} -- before any distance is looked at",
        plural(pairs.len(), "distinct cluster pair"),
        plural(merges, "merge"),
        plural(left_over, "cluster"),
        clustering.clusters.len()
    );
}

/// The cluster a turn was assigned, which the caller has already established it has.
fn cluster_of(clustering: &Clustering, turn: usize) -> u32 {
    clustering.assignment[turn].expect("this turn was clustered")
}

/// The representative of a cluster's merge component, flattening the path walked to reach it.
fn root(parent: &mut [usize], mut cluster: usize) -> usize {
    while parent[cluster] != cluster {
        parent[cluster] = parent[parent[cluster]];
        cluster = parent[cluster];
    }
    cluster
}

/// Every turn held by every cluster currently merged into one component.
fn component_turns(parent: &mut [usize], component: usize, members: &[Vec<usize>]) -> Vec<usize> {
    let mut held = Vec::new();
    for (cluster, mine) in members.iter().enumerate() {
        if root(parent, cluster) == component {
            held.extend_from_slice(mine);
        }
    }
    held
}

/// Whether segmentation heard these two turns in one window under the *same* local speaker
/// index, which makes them one person on exactly the same authority.
fn same_local_speaker(turns: &[LocalTurn], i: usize, j: usize) -> bool {
    turns[i].window == turns[j].window && turns[i].local_speaker == turns[j].local_speaker
}

/// Whether the cannot-link constraint forbids holding these two sets of turns in one cluster.
///
/// The same test `agglomerate` encodes as an infinity in its distance matrix, asked directly
/// rather than read off a distance: once a group holds a forbidden pair every average linkage
/// through it is infinite, so clustering never needs to ask again, but a report weighing a merge
/// that clustering did not make does.
fn heard_apart(left: &[usize], right: &[usize], turns: &[LocalTurn]) -> bool {
    left.iter()
        .any(|&i| right.iter().any(|&j| known_different(turns, i, j)))
}
