//! The two labelled populations an adoption threshold has to be chosen from, and nothing else.
//!
//! Clustering leaves a long tail of one- and two-second fragments in clusters of their own, and
//! the pass that would sweep them up compares one quantity against one cut. This module builds
//! the populations that say where that cut can go, from labels nobody had to supply by ear.
//!
//! # The quantity, stated once so that no reader has to guess
//!
//! **A small group's normalized mean against an above-floor cluster's normalized mean**:
//! [`GroupDistance::centroid`], cosine, 0 for the same direction and 1 for orthogonal. Not
//! turn-to-turn, and *not* [`GroupDistance::average_linkage`] -- which is the criterion
//! clustering merged on, is a different number on the same pair, and is thresholded by a
//! different constant that happens to hold the same value. TASK-020 is a live bug caused by
//! exactly that confusion, so every pair here carries the whole [`GroupDistance`] rather than a
//! bare `f32`: the two criteria travel together with the shrinkage relating them, and no field
//! can be read as the other one.
//!
//! [`crate::MERGE_DISTANCE`] governs `average_linkage` and governs nothing in this module.
//!
//! # Where the labels come from
//!
//! Segmentation's local speaker index is free supervision in both directions, and both
//! directions are about turns rather than about voices, so neither is conditional on a
//! clustering decision a reader would have to trust:
//!
//! - **Positives** are [`AdoptionPopulations::positives`], built leave-one-class-out. Take a
//!   must-link class -- two or more embedded turns sharing one `(window, local_speaker)`, which
//!   segmentation heard as one person -- that ended up wholly inside a cluster above the
//!   talk-time floor, and measure it against **the rest of that cluster**. The class is excluded
//!   from the residual, because a class left in the group it is compared to biases the mean it
//!   is measured against; `positive_excludes_the_class_from_the_residual` asserts that rather
//!   than leaving it as a sentence here.
//!
//!   The shape is the one the pass measures: three to ten seconds of speech against a speaker
//!   estimated from minutes. **The weak leg**, and it belongs beside every number derived from
//!   these: segmentation says the class's own turns are one person, but that the *class and the
//!   residual* are one person is clustering's claim, not segmentation's. Seeding forces the
//!   class together; the merge loop is what put it in that cluster.
//!
//! - **Negatives** are [`AdoptionPopulations::negatives`], a view of
//!   [`AdoptionPopulations::offers`] rather than a second vector, so the grid a report prints
//!   and the population it scores cannot be different numbers. A below-floor cluster that the
//!   same-window cannot-link constraint bars from an above-floor cluster is a different-speaker
//!   pair at exactly the granularity the pass works at, and the witness turn pair that bars it
//!   is carried on the label so it can be played.
//!
//!   **The caveat**, which has to be printed anywhere a cut derived from these appears: every
//!   negative here is a pair the constraint *already refuses*, so a threshold read off them
//!   prices a distance-only rule the pass does not use. What they measure is how close two
//!   provably different people's centroids come at this granularity, which is what bounds trust
//!   in a cut on the unblocked pairs, where no constraint protects anybody.
//!
//! - **The cross-check** is [`AdoptionPopulations::within_class`], leave-one-*turn*-out inside a
//!   class. Nothing but segmentation stands behind that label, so it is the purest same-speaker
//!   population available -- and its shape is wrong, because a class fits inside one ten-second
//!   window and neither side is a speaker-scale mean. It says whether the positives look like a
//!   same-speaker population. It does not substitute for them.
//!
//! - **The auxiliary negatives** are [`AdoptionPopulations::above_floor`], every pair of
//!   above-floor clusters. Large against large is a third shape and folding it into the trial
//!   list would flatter the separation, so it stays out and is reported on its own. What it
//!   licenses is the ceiling: two above-floor clusters are two people, and a cut wider than the
//!   gap between the closest of them is measuring a distance two speakers fit inside.
//!
//! Everything measured but labelled by neither direction is [`PairLabel::Unlabelled`] and is
//! carried anyway, so that the offer grid is one computation. [`CentroidPair::trial`] returns
//! [`None`] for those, which is what keeps the caller from having to decide a label.
//!
//! # What is not a positive, and why
//!
//! Counted in [`Declined`] and reported, because an unstated exclusion is how a population gets
//! quietly flattered: a class of a single embedded turn (it carries no must-link assertion at
//! all -- which cluster that turn joined is a clustering decision); a class inside a below-floor
//! cluster (no speaker-scale residual to measure against); a class that is its whole cluster (no
//! residual left); a class split across clusters (which seeding makes impossible, so it is a
//! regression check that reads 0); and a group with no direction at all.
//!
//! # Order
//!
//! Every population is deterministically ordered so that two runs of a report over one session
//! are diffable, which is the only way "the output did not change" gets checked at all.
//! `positives` by enclosing cluster id then `(window, local_speaker)`; `offers` by below-floor
//! cluster id then above-floor cluster id; `within_class` by `(window, local_speaker)` then the
//! held-out turn; `above_floor` by the two cluster ids ascending.
//!
//! # What this module does not do
//!
//! It does not pick a threshold, it holds no constant, and no production path calls it. Scoring
//! belongs to [`crate::score_trials`], which states and tests the conventions -- accept is
//! strictly below the cut, nearest-rank percentiles -- so that a report built on it cannot drift
//! from the decision it informs. Presentation belongs to `examples/cluster-speaker-track.rs`.
//! The arithmetic is here rather than there because `cargo test` builds examples without running
//! the `#[test]`s inside them, and a diagnostic whose conventions nobody can test is a number to
//! believe rather than evidence.

use std::collections::BTreeMap;

use crate::segmentation::LocalTurn;
use crate::speakers::{Clustering, GroupDistance, group_distance, heard_at_once};
use crate::trials::Trial;

/// One side of a measured pair: which cluster it came from, and what it holds.
#[derive(Debug, Clone, PartialEq)]
pub struct Side {
    /// The cluster this side is part of. Ids are positions in [`Clustering::clusters`], so this
    /// indexes that slice as well as naming it.
    pub cluster: u32,

    /// Turn indices, ascending, positional against the `turns` slice handed to
    /// [`adoption_populations`]. Only embedded turns appear -- a turn with no vector takes part
    /// in no distance.
    pub turns: Vec<usize>,

    /// Speech these turns cover, in seconds. Summed from the turns rather than read off the
    /// cluster, because a side is usually part of one.
    pub seconds: f64,

    /// How many distinct `(window, local_speaker)` classes these turns span.
    ///
    /// Worth carrying for the small side of a negative, where it decides how much the label is
    /// worth. **1** means segmentation itself grouped this fragment, so it is a small group of
    /// one person. **More than 1** means embedding alone assembled it across windows, and it may
    /// be a blend belonging to nobody -- which makes its different-speaker label only as good as
    /// the clustering that built it.
    pub classes: usize,
}

/// What segmentation says about a pair, and on what evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PairLabel {
    /// Same speaker: segmentation heard the small side's turns in one window under one index.
    MustLinkClass { window: usize, local_speaker: usize },

    /// Different speakers: segmentation heard these two turns at once under different indices.
    ///
    /// `witness` is the lowest such turn pair, ascending, so a reader can play the two clips the
    /// label rests on rather than take the word "blocked" for it. One witness is enough -- a
    /// single forbidden pair bars the whole merge.
    CannotLink { witness: (usize, usize) },

    /// Neither direction says anything. Carried so that the offer grid is one computation, and
    /// excluded from every trial list by [`CentroidPair::trial`].
    Unlabelled,
}

impl PairLabel {
    /// The must-link class behind a positive, as `(window, local_speaker)`, or [`None`].
    ///
    /// Public because it is what a report prints beside a positive to say which window's
    /// judgement labelled it, and what the ordering of [`AdoptionPopulations::positives`] is by.
    pub fn class(&self) -> Option<(usize, usize)> {
        match *self {
            PairLabel::MustLinkClass {
                window,
                local_speaker,
            } => Some((window, local_speaker)),
            _ => None,
        }
    }
}

/// One measured pair: a small group against a larger one, both group distances, and the label.
///
/// [`CentroidPair::distance`] carries both criteria rather than the one being thresholded, so a
/// line printed from this can show them side by side and neither can be mistaken for the other.
#[derive(Debug, Clone, PartialEq)]
pub struct CentroidPair {
    /// The quantity is [`GroupDistance::centroid`]: `small`'s normalized mean against `large`'s.
    /// The linkage and the shrinkage travel with it because they are one identity, not three
    /// measurements -- see [`group_distance`].
    pub distance: GroupDistance,

    /// The fragment-scale side: a must-link class, one held-out turn, or a below-floor cluster.
    ///
    /// For [`AdoptionPopulations::above_floor`] neither side is small and this is merely the
    /// lower cluster id.
    pub small: Side,

    /// The speaker-scale side: an above-floor cluster, or -- for a positive -- that cluster
    /// **minus** the class on the small side, which is why both sides can name one cluster.
    pub large: Side,

    pub label: PairLabel,
}

impl CentroidPair {
    /// This pair as a scoring trial, or [`None`] when segmentation labelled it neither way.
    ///
    /// The distance handed over is the centroid one, which is the whole point of the type. The
    /// label is turned into `same_speaker` here rather than at the call site so that a report
    /// never decides what a pair means: an unlabelled pair scored either way is a fabricated
    /// error rate.
    pub fn trial(&self) -> Option<Trial> {
        let same_speaker = match self.label {
            PairLabel::MustLinkClass { .. } => true,
            PairLabel::CannotLink { .. } => false,
            PairLabel::Unlabelled => return None,
        };
        Some(Trial {
            same_speaker,
            distance: self.distance.centroid,
        })
    }
}

/// Must-link classes that yielded no positive, by reason.
///
/// Reported rather than dropped. Each count is a way the positive population is narrower than
/// "everything segmentation labelled", and a reader who cannot see them cannot tell a small
/// population from a filtered one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Declined {
    /// Classes of a single embedded turn: no must-link assertion at all, since one turn is not
    /// a pair. Which cluster it joined is clustering's decision, not segmentation's.
    pub single_turn: usize,

    /// Classes whose embedded turns landed in more than one cluster.
    ///
    /// A regression check that reads 0: `agglomerate` seeds its groups by
    /// `(window, local_speaker)`, so no sequence of merges can pull a class apart. Any other
    /// number means seeding is not doing what it claims.
    pub split_across_clusters: usize,

    /// Classes inside a below-floor cluster: no speaker-scale residual to measure against, so
    /// the pair would be the wrong shape.
    pub below_floor: usize,

    /// Classes that are their whole cluster: excluding the class leaves nothing to compare to.
    pub whole_cluster: usize,

    /// Classes where a side had no direction -- an empty group or one whose members cancel.
    /// Unreachable for real voices; counted so that it cannot be silent if it happens.
    pub no_direction: usize,

    /// Below-floor/above-floor offers with no measurable distance, for the same reason.
    /// Excluded from [`Declined::classes`], which is about must-link classes only.
    pub offers_without_distance: usize,
}

impl Declined {
    /// How many must-link classes were declined. Added to
    /// [`AdoptionPopulations::positives`]`.len()` this is
    /// [`AdoptionPopulations::classes`] exactly, which is the arithmetic
    /// `every_class_is_a_positive_or_declined_exactly_once` asserts.
    pub fn classes(&self) -> usize {
        self.single_turn
            + self.split_across_clusters
            + self.below_floor
            + self.whole_cluster
            + self.no_direction
    }
}

/// Everything one session says about where an adoption cut could go.
///
/// See the module documentation for what each population is labelled by, what shape it has, and
/// which of them may be folded into one trial list.
#[derive(Debug, Clone, PartialEq)]
pub struct AdoptionPopulations {
    /// Same-speaker pairs, leave-one-class-out. A must-link class against the rest of the
    /// above-floor cluster it landed in.
    pub positives: Vec<CentroidPair>,

    /// Every below-floor cluster against every above-floor one: what the pass would choose
    /// among. The blocked ones are the negative population; see [`AdoptionPopulations::negatives`].
    pub offers: Vec<CentroidPair>,

    /// The pure-ground-truth cross-check, leave-one-turn-out within a class. Right label, wrong
    /// shape.
    pub within_class: Vec<CentroidPair>,

    /// Every pair of above-floor clusters: the auxiliary negatives the ceiling comes from.
    pub above_floor: Vec<CentroidPair>,

    /// Cluster ids under the floor, ascending. The convention is `speech_seconds < floor`.
    pub below: Vec<u32>,

    /// Cluster ids at or above the floor, ascending, so a cluster sitting exactly at the floor
    /// is a speaker rather than a fragment.
    pub above: Vec<u32>,

    /// `Some((largest below, smallest above))`: the band of floors giving *this* partition.
    ///
    /// Half-open on the left and closed on the right -- any floor `f` with
    /// `largest_below < f <= smallest_above` partitions these clusters identically, because the
    /// test is `< floor`. [`None`] when either side is empty, since a partition with nothing on
    /// one side of it is not a partition this band describes.
    pub floor_band: Option<(f64, f64)>,

    /// How many distinct `(window, local_speaker)` classes the embedded turns span, declined
    /// ones included. The denominator [`Declined`] is read against.
    pub classes: usize,

    pub declined: Declined,
}

impl AdoptionPopulations {
    /// The offers the cannot-link constraint bars: the negative population.
    ///
    /// A view of [`AdoptionPopulations::offers`] rather than a second vector, so the grid a
    /// report prints and the population it scores are the same numbers by construction.
    pub fn negatives(&self) -> impl Iterator<Item = &CentroidPair> {
        self.offers
            .iter()
            .filter(|pair| matches!(pair.label, PairLabel::CannotLink { .. }))
    }

    /// The primary trial list: positives and negatives, and nothing else.
    ///
    /// Deliberately excludes [`AdoptionPopulations::within_class`] and
    /// [`AdoptionPopulations::above_floor`]. Both are other shapes of the same quantity and
    /// folding either in would flatter the separation between the two populations that share
    /// the pass's shape -- see the module documentation.
    pub fn trials(&self) -> Vec<Trial> {
        self.positives
            .iter()
            .chain(self.negatives())
            .filter_map(CentroidPair::trial)
            .collect()
    }

    /// The closest two above-floor clusters, as `(centroid distance, lower id, higher id)`.
    ///
    /// The ceiling on any cut. Both are speakers, so a cut at or above this is measuring a gap
    /// two separate people fit inside. [`None`] when fewer than two clusters clear the floor.
    pub fn ceiling(&self) -> Option<(f32, u32, u32)> {
        self.above_floor
            .iter()
            .fold(None, |best: Option<&CentroidPair>, pair| match best {
                Some(best) if best.distance.centroid <= pair.distance.centroid => Some(best),
                _ => Some(pair),
            })
            .map(|pair| {
                (
                    pair.distance.centroid,
                    pair.small.cluster,
                    pair.large.cluster,
                )
            })
    }
}

/// Builds every population above from one session's clustering.
///
/// `turns` is the same slice handed to [`crate::cluster_speaker_turns`], `clustering` what it
/// returned, and `floor_seconds` the talk-time below which a cluster is a fragment looking for
/// an owner rather than a speaker that could own one. The floor partition reads
/// `clustering.clusters[].speech_seconds`, which is the same number every other consumer
/// partitions on, so two blocks of one report cannot disagree about which clusters are speakers.
///
/// Reads no models and no files, and pure: the same arguments give the same populations in the
/// same order. Cost is dominated by the offer grid, one [`group_distance`] per below-floor
/// cluster per above-floor cluster.
///
/// Every degenerate shape is empty populations rather than a panic. No clusters, nothing above
/// the floor, nothing below it, turns with no embeddings, and a `clustering` whose assignment
/// names a cluster or a turn that does not exist all yield a value a report can print.
pub fn adoption_populations(
    turns: &[LocalTurn],
    clustering: &Clustering,
    floor_seconds: f64,
) -> AdoptionPopulations {
    let clusters = &clustering.clusters;
    let mut declined = Declined::default();

    // Embedded turns per cluster, ascending, and each turn's cluster. Anything the clustering
    // names that this `turns` slice does not have is dropped rather than indexed: the two come
    // from one call in production, and a report is not the place to panic over a mismatch.
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); clusters.len()];
    let mut owner: Vec<Option<usize>> = vec![None; turns.len()];
    for (turn, assigned) in clustering.assignment.iter().enumerate() {
        let Some(&id) = assigned.as_ref() else {
            continue;
        };
        if turn >= turns.len()
            || !clustering
                .turn_embeddings
                .get(turn)
                .is_some_and(Option::is_some)
        {
            continue;
        }
        if let Some(mine) = members.get_mut(id as usize) {
            mine.push(turn);
            owner[turn] = Some(id as usize);
        }
    }

    let is_above = |cluster: usize| clusters[cluster].speech_seconds >= floor_seconds;
    let below_at: Vec<usize> = (0..clusters.len()).filter(|&c| !is_above(c)).collect();
    let above_at: Vec<usize> = (0..clusters.len()).filter(|&c| is_above(c)).collect();

    let vectors = |held: &[usize]| -> Vec<&[f32]> {
        held.iter()
            .filter_map(|&turn| clustering.turn_embeddings[turn].as_deref())
            .collect()
    };
    let side = |cluster: usize, held: Vec<usize>| -> Side {
        // Folded from 0.0 rather than `sum()`, whose float identity is `-0.0`: an empty side
        // would otherwise print "-0.0 s", which reads as a broken instrument.
        let seconds = held.iter().fold(0.0, |total, &turn| {
            total + turns[turn].end_s - turns[turn].start_s
        });
        let mut classes: Vec<(usize, usize)> = held.iter().map(|&turn| key(turns, turn)).collect();
        classes.sort_unstable();
        classes.dedup();
        Side {
            cluster: clusters[cluster].id,
            turns: held,
            seconds,
            classes: classes.len(),
        }
    };

    // Every below-floor cluster against every above-floor one, blocked ones labelled. Below
    // ascends inside above ascending, which is the documented order.
    let mut offers = Vec::with_capacity(below_at.len() * above_at.len());
    for &small in &below_at {
        for &large in &above_at {
            let Some(distance) =
                group_distance(&vectors(&members[small]), &vectors(&members[large]))
            else {
                declined.offers_without_distance += 1;
                continue;
            };
            offers.push(CentroidPair {
                distance,
                small: side(small, members[small].clone()),
                large: side(large, members[large].clone()),
                label: match witness_pair(&members[small], &members[large], turns) {
                    Some(witness) => PairLabel::CannotLink { witness },
                    None => PairLabel::Unlabelled,
                },
            });
        }
    }

    // Must-link classes over embedded turns. A `BTreeMap` so the key order is the report order
    // and two runs are diffable; each class's turns ascend because `members` does.
    let mut classes: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for mine in &members {
        for &turn in mine {
            classes.entry(key(turns, turn)).or_default().push(turn);
        }
    }
    for class in classes.values_mut() {
        class.sort_unstable();
    }

    let mut positives = Vec::new();
    let mut within_class = Vec::new();
    for (&(window, local_speaker), class) in &classes {
        let label = PairLabel::MustLinkClass {
            window,
            local_speaker,
        };

        // The cross-check: every held-out turn against the rest of its own class, whatever
        // cluster the class sits in. Built before the exclusions below, because none of them is
        // a reason this pair is not ground truth -- they are reasons it is the wrong shape for a
        // threshold.
        if class.len() >= 2 {
            for (nth, &held) in class.iter().enumerate() {
                let rest: Vec<usize> = class
                    .iter()
                    .enumerate()
                    .filter(|&(other, _)| other != nth)
                    .map(|(_, &turn)| turn)
                    .collect();
                let (Some(mine), Some(theirs)) = (owner[held], owner[rest[0]]) else {
                    continue;
                };
                let Some(distance) = group_distance(&vectors(&[held]), &vectors(&rest)) else {
                    continue;
                };
                within_class.push(CentroidPair {
                    distance,
                    small: side(mine, vec![held]),
                    large: side(theirs, rest),
                    label,
                });
            }
        }

        // The positive, and the four ways there is not one. Ordered so that every class falls
        // into exactly one bucket, which `Declined::classes` is read against.
        if class.len() < 2 {
            declined.single_turn += 1;
            continue;
        }
        let Some(cluster) = one_cluster(class, &owner) else {
            declined.split_across_clusters += 1;
            continue;
        };
        if !is_above(cluster) {
            declined.below_floor += 1;
            continue;
        }
        // Excluded, not merely omitted: a class left inside the group it is compared against
        // biases the mean it is measured by, and the bias is largest exactly where the class is
        // a big share of a small cluster.
        let residual: Vec<usize> = members[cluster]
            .iter()
            .copied()
            .filter(|turn| !class.contains(turn))
            .collect();
        if residual.is_empty() {
            declined.whole_cluster += 1;
            continue;
        }
        let Some(distance) = group_distance(&vectors(class), &vectors(&residual)) else {
            declined.no_direction += 1;
            continue;
        };
        positives.push(CentroidPair {
            distance,
            small: side(cluster, class.clone()),
            large: side(cluster, residual),
            label,
        });
    }
    positives.sort_by_key(|pair| (pair.large.cluster, pair.label.class()));

    let mut above_floor = Vec::new();
    for (nth, &left) in above_at.iter().enumerate() {
        for &right in &above_at[nth + 1..] {
            let Some(distance) =
                group_distance(&vectors(&members[left]), &vectors(&members[right]))
            else {
                continue;
            };
            above_floor.push(CentroidPair {
                distance,
                small: side(left, members[left].clone()),
                large: side(right, members[right].clone()),
                label: PairLabel::Unlabelled,
            });
        }
    }

    let speech = |group: &[usize]| -> Vec<f64> {
        group.iter().map(|&c| clusters[c].speech_seconds).collect()
    };
    let floor_band = speech(&below_at)
        .into_iter()
        .max_by(f64::total_cmp)
        .zip(speech(&above_at).into_iter().min_by(f64::total_cmp));

    AdoptionPopulations {
        positives,
        offers,
        within_class,
        above_floor,
        below: below_at.iter().map(|&c| clusters[c].id).collect(),
        above: above_at.iter().map(|&c| clusters[c].id).collect(),
        floor_band,
        classes: classes.len(),
        declined,
    }
}

/// One turn's `(window, local_speaker)`: the key both directions of the constraint are about.
fn key(turns: &[LocalTurn], turn: usize) -> (usize, usize) {
    (turns[turn].window, turns[turn].local_speaker)
}

/// The cluster holding every one of these turns, or [`None`] if they are not all in one.
fn one_cluster(class: &[usize], owner: &[Option<usize>]) -> Option<usize> {
    let first = owner[*class.first()?]?;
    class
        .iter()
        .all(|&turn| owner[turn] == Some(first))
        .then_some(first)
}

/// The lowest turn pair, ascending, that segmentation heard at once under different indices.
///
/// The witness for a cannot-link label: one such pair anywhere across the two groups bars the
/// merge, so the *whole* pair is labelled different-speaker by it. Lowest rather than any, so
/// two runs name the same clips. [`None`] means nothing bars this merge.
fn witness_pair(left: &[usize], right: &[usize], turns: &[LocalTurn]) -> Option<(usize, usize)> {
    left.iter()
        .flat_map(|&i| right.iter().map(move |&j| (i.min(j), i.max(j))))
        .filter(|&(i, j)| heard_at_once(key(turns, i), key(turns, j)))
        .min()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segmentation::WINDOW_SECONDS;
    use meethook_session::SpeakerCluster;

    /// A unit vector pointing `degrees` away from the first axis, so a test can name the
    /// distance it means -- `1 - cos(difference)` -- instead of a pile of decimals. Same helper
    /// `speakers.rs` tests with, for the same reason.
    fn at(degrees: f32) -> Vec<f32> {
        let radians = degrees.to_radians();
        vec![radians.cos(), radians.sin(), 0.0, 0.0]
    }

    /// One turn, in the window its start really falls inside.
    fn turn(start_s: f64, end_s: f64, local_speaker: usize) -> LocalTurn {
        LocalTurn {
            start_s,
            end_s,
            window: (start_s / WINDOW_SECONDS) as usize,
            local_speaker,
        }
    }

    /// A clustering assembled by hand: `groups` gives each cluster's turn indices in id order,
    /// and `voices` each turn's embedding, `None` for a turn too short to have one.
    ///
    /// `speech_seconds` is summed from the turns rather than passed in, so a test that wants a
    /// cluster above the floor makes it talk for longer instead of asserting that it did.
    fn clustering(
        turns: &[LocalTurn],
        groups: &[&[usize]],
        voices: &[Option<Vec<f32>>],
    ) -> Clustering {
        let mut assignment = vec![None; turns.len()];
        let mut clusters = Vec::new();
        for (id, group) in groups.iter().enumerate() {
            for &turn in *group {
                assignment[turn] = Some(id as u32);
            }
            let held: Vec<&[f32]> = group
                .iter()
                .filter_map(|&turn| voices[turn].as_deref())
                .collect();
            clusters.push(SpeakerCluster {
                id: id as u32,
                embedding: mean(&held),
                speech_seconds: group
                    .iter()
                    .fold(0.0, |total, &t| total + turns[t].end_s - turns[t].start_s),
                first_spoke_seconds: group
                    .iter()
                    .map(|&t| turns[t].start_s)
                    .fold(f64::INFINITY, f64::min),
                // This instrument measures adoption distances and never identifies anybody,
                // so it does not carry the relation `cluster_speaker_turns` computes.
                heard_at_once_with: Vec::new(),
                representatives: Vec::new(),
            });
        }
        Clustering {
            clusters,
            assignment,
            turn_embeddings: voices.to_vec(),
        }
    }

    fn mean(members: &[&[f32]]) -> Vec<f32> {
        let mut mean = vec![0.0f32; members.first().map_or(0, |m| m.len())];
        for member in members {
            for (m, v) in mean.iter_mut().zip(*member) {
                *m += v / members.len() as f32;
            }
        }
        let norm = mean.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            mean.iter_mut().for_each(|v| *v /= norm);
        }
        mean
    }

    fn voices(degrees: &[f32]) -> Vec<Option<Vec<f32>>> {
        degrees.iter().map(|&d| Some(at(d))).collect()
    }

    /// Two turns in window 0 under one index -- a must-link class -- plus two more turns of the
    /// same cluster, and a second cluster far away and short. Cluster 0 is above a 5 s floor.
    fn class_inside_a_speaker() -> (Vec<LocalTurn>, Vec<Option<Vec<f32>>>, Clustering) {
        let turns = vec![
            turn(0.0, 3.0, 0),
            turn(5.0, 8.0, 0),
            turn(20.0, 24.0, 0),
            turn(30.0, 34.0, 0),
            turn(40.0, 41.0, 0),
        ];
        let voices = voices(&[10.0, 14.0, 40.0, 44.0, 120.0]);
        let clustering = clustering(&turns, &[&[0, 1, 2, 3], &[4]], &voices);
        (turns, voices, clustering)
    }

    #[test]
    fn positive_excludes_the_class_from_the_residual() {
        let (turns, voices, grouping) = class_inside_a_speaker();
        let populations = adoption_populations(&turns, &grouping, 5.0);

        assert_eq!(populations.positives.len(), 1);
        let positive = &populations.positives[0];
        assert_eq!(positive.small.turns, [0, 1]);
        assert_eq!(
            positive.large.turns,
            [2, 3],
            "the class stayed in the residual"
        );
        assert_eq!(
            positive.label,
            PairLabel::MustLinkClass {
                window: 0,
                local_speaker: 0
            }
        );

        let class = [voices[0].as_deref().unwrap(), voices[1].as_deref().unwrap()];
        let residual = [voices[2].as_deref().unwrap(), voices[3].as_deref().unwrap()];
        assert_eq!(
            positive.distance,
            group_distance(&class, &residual).unwrap(),
            "the positive is not the class against the rest of its cluster"
        );

        // The claim the exclusion is for: including the class pulls the mean it is measured
        // against toward it, so the same pair would read closer than it is.
        let whole = [
            voices[0].as_deref().unwrap(),
            voices[1].as_deref().unwrap(),
            voices[2].as_deref().unwrap(),
            voices[3].as_deref().unwrap(),
        ];
        let biased = group_distance(&class, &whole).unwrap();
        assert!(
            biased.centroid < positive.distance.centroid,
            "including the class did not flatter the distance, so this test proves nothing: \
             {biased:?} vs {:?}",
            positive.distance
        );
    }

    #[test]
    fn a_class_of_one_embedded_turn_is_no_positive() {
        // Turns 0 and 1 are one class but only turn 0 was embeddable, so the class carries no
        // must-link assertion at all.
        let turns = vec![
            turn(0.0, 3.0, 0),
            turn(5.0, 5.2, 0),
            turn(20.0, 24.0, 0),
            turn(30.0, 34.0, 0),
        ];
        let mut voices = voices(&[10.0, 14.0, 40.0, 44.0]);
        voices[1] = None;

        let populations =
            adoption_populations(&turns, &clustering(&turns, &[&[0, 2, 3]], &voices), 5.0);

        assert!(populations.positives.is_empty());
        assert_eq!(
            populations.declined.single_turn, 3,
            "one class per embedded turn"
        );
        assert_eq!(populations.classes, 3);
    }

    #[test]
    fn a_class_that_is_its_whole_cluster_is_no_positive() {
        let turns = vec![turn(0.0, 4.0, 0), turn(5.0, 9.0, 0), turn(40.0, 41.0, 0)];
        let voices = voices(&[10.0, 14.0, 120.0]);

        let populations =
            adoption_populations(&turns, &clustering(&turns, &[&[0, 1], &[2]], &voices), 5.0);

        assert!(populations.positives.is_empty(), "measured against nothing");
        assert_eq!(populations.declined.whole_cluster, 1);
    }

    #[test]
    fn a_class_inside_a_below_floor_cluster_is_no_positive() {
        let (turns, _, grouping) = class_inside_a_speaker();

        // A floor above every cluster: the class is still a class, but there is no speaker-scale
        // residual anywhere to measure it against.
        let populations = adoption_populations(&turns, &grouping, 100.0);

        assert!(populations.positives.is_empty());
        assert_eq!(populations.declined.below_floor, 1);
        assert!(populations.above.is_empty());
        assert!(populations.offers.is_empty(), "nothing to adopt into");
    }

    #[test]
    fn every_class_is_a_positive_or_declined_exactly_once() {
        let (turns, _, grouping) = class_inside_a_speaker();
        for floor in [0.0, 5.0, 14.0, 100.0] {
            let populations = adoption_populations(&turns, &grouping, floor);
            assert_eq!(
                populations.positives.len() + populations.declined.classes(),
                populations.classes,
                "the declined counts do not add up at floor {floor}"
            );
        }
    }

    /// Window 0 held two local speakers, so cluster 1 can never be adopted into cluster 0 --
    /// and that is a labelled different-speaker pair at the granularity the pass works at.
    #[test]
    fn a_blocked_offer_is_a_negative_and_an_unblocked_one_is_not() {
        let turns = vec![
            turn(0.0, 4.0, 0),
            turn(20.0, 24.0, 0),
            turn(4.0, 6.0, 1),
            turn(40.0, 41.0, 0),
        ];
        let voices = voices(&[10.0, 14.0, 30.0, 120.0]);

        let populations = adoption_populations(
            &turns,
            &clustering(&turns, &[&[0, 1], &[2], &[3]], &voices),
            5.0,
        );

        assert_eq!(populations.above, [0]);
        assert_eq!(populations.below, [1, 2]);
        assert_eq!(populations.offers.len(), 2);

        let negatives: Vec<&CentroidPair> = populations.negatives().collect();
        assert_eq!(negatives.len(), 1);
        assert_eq!(negatives[0].small.cluster, 1);
        assert_eq!(negatives[0].large.cluster, 0);
        assert_eq!(
            negatives[0].label,
            PairLabel::CannotLink { witness: (0, 2) },
            "the witness is the turn pair heard at once, lowest first"
        );
        assert_eq!(
            negatives[0].small.classes, 1,
            "one window and one index, so segmentation itself grouped this fragment"
        );

        assert_eq!(populations.offers[1].label, PairLabel::Unlabelled);
        assert_eq!(
            populations.offers[1].trial(),
            None,
            "an unlabelled offer must not reach a trial list"
        );
        assert_eq!(
            populations
                .trials()
                .iter()
                .filter(|t| !t.same_speaker)
                .count(),
            1
        );
    }

    #[test]
    fn a_fragment_blocked_from_two_speakers_is_two_negatives() {
        let turns = vec![
            turn(0.0, 4.0, 0),
            turn(20.0, 24.0, 0),
            turn(4.0, 6.0, 1),
            turn(24.0, 28.0, 1),
            turn(6.0, 8.0, 2),
        ];
        let voices = voices(&[10.0, 14.0, 60.0, 64.0, 120.0]);

        let populations = adoption_populations(
            &turns,
            &clustering(&turns, &[&[0, 1], &[2, 3], &[4]], &voices),
            5.0,
        );

        assert_eq!(populations.above, [0, 1]);
        assert_eq!(populations.below, [2]);
        let negatives: Vec<&CentroidPair> = populations.negatives().collect();
        assert_eq!(negatives.len(), 2);
        assert_eq!(
            negatives
                .iter()
                .map(|pair| pair.large.cluster)
                .collect::<Vec<_>>(),
            [0, 1]
        );
    }

    #[test]
    fn the_floor_band_is_the_gap_the_partition_survives() {
        let (turns, _, grouping) = class_inside_a_speaker();

        // Cluster 0 holds 14 s and cluster 1 holds 1 s.
        let populations = adoption_populations(&turns, &grouping, 5.0);
        assert_eq!(populations.floor_band, Some((1.0, 14.0)));

        // The closed right edge: a cluster sitting exactly at the floor is above it.
        let populations = adoption_populations(&turns, &grouping, 14.0);
        assert_eq!(populations.above, [0]);
        assert_eq!(populations.floor_band, Some((1.0, 14.0)));

        // A hair past it and cluster 0 stops being a speaker, which is the band's right edge
        // being the last floor that works rather than the first that does not.
        let populations = adoption_populations(&turns, &grouping, 14.001);
        assert!(populations.above.is_empty());
        assert_eq!(populations.floor_band, None, "nothing above the floor");
    }

    #[test]
    fn the_cross_check_is_leave_one_turn_out_within_a_class() {
        let (turns, voices, grouping) = class_inside_a_speaker();
        let populations = adoption_populations(&turns, &grouping, 5.0);

        // Turns 0 and 1 are the only class with two embedded turns, so two pairs: each turn
        // against the other. Its label is the class, and its shape is one turn a side.
        assert_eq!(populations.within_class.len(), 2);
        assert_eq!(populations.within_class[0].small.turns, [0]);
        assert_eq!(populations.within_class[0].large.turns, [1]);
        assert_eq!(populations.within_class[1].small.turns, [1]);
        assert_eq!(
            populations.within_class[0].distance,
            group_distance(
                &[voices[0].as_deref().unwrap()],
                &[voices[1].as_deref().unwrap()]
            )
            .unwrap()
        );
        assert!(
            populations
                .within_class
                .iter()
                .all(|pair| pair.trial().is_some_and(|trial| trial.same_speaker)),
            "segmentation labelled every one of these same-speaker"
        );
    }

    #[test]
    fn the_ceiling_is_the_closest_two_speakers() {
        let turns = vec![
            turn(0.0, 4.0, 0),
            turn(20.0, 24.0, 0),
            turn(40.0, 44.0, 0),
            turn(60.0, 64.0, 0),
            turn(80.0, 84.0, 0),
            turn(100.0, 104.0, 0),
        ];
        // Three speakers at 10, 50 and 60 degrees: the closest pair is the last two.
        let voices = voices(&[8.0, 12.0, 48.0, 52.0, 58.0, 62.0]);
        let populations = adoption_populations(
            &turns,
            &clustering(&turns, &[&[0, 1], &[2, 3], &[4, 5]], &voices),
            5.0,
        );

        assert_eq!(populations.above, [0, 1, 2]);
        assert_eq!(populations.above_floor.len(), 3);
        let (centroid, left, right) = populations.ceiling().expect("three speakers");
        assert_eq!((left, right), (1, 2));
        assert!((centroid - (1.0 - 10.0f32.to_radians().cos())).abs() < 1e-5);
        assert!(
            populations
                .above_floor
                .iter()
                .all(|pair| pair.trial().is_none()),
            "large against large is a different shape and must not reach the trial list"
        );
    }

    #[test]
    fn degenerate_shapes_are_empty_populations_rather_than_panics() {
        let empty = Clustering {
            clusters: Vec::new(),
            assignment: Vec::new(),
            turn_embeddings: Vec::new(),
        };
        let populations = adoption_populations(&[], &empty, 30.0);
        assert_eq!(populations.classes, 0);
        assert_eq!(populations.floor_band, None);
        assert!(populations.offers.is_empty() && populations.positives.is_empty());
        assert!(populations.trials().is_empty());
        assert_eq!(populations.ceiling(), None);

        // Nothing below the floor: one speaker and no fragments to adopt.
        let (turns, _, grouping) = class_inside_a_speaker();
        let populations = adoption_populations(&turns, &grouping, 0.0);
        assert!(populations.below.is_empty());
        assert!(populations.offers.is_empty());
        assert_eq!(populations.negatives().count(), 0);

        // Turns that were all too short to embed: clusters exist but hold no vectors.
        let turns = vec![turn(0.0, 0.2, 0), turn(5.0, 5.2, 0)];
        let voices = vec![None, None];
        let populations =
            adoption_populations(&turns, &clustering(&turns, &[&[0], &[1]], &voices), 5.0);
        assert_eq!(populations.classes, 0, "no embedded turn is in any class");
        assert!(populations.positives.is_empty() && populations.within_class.is_empty());
    }
}
