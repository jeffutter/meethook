//! Unit-vector algebra and the group-distance oracle.
//!
//! Pure linear algebra over slices of `f32` -- no audio, no clustering. Everything else in
//! the crate measures how far apart two groups of turns sit by going through here rather
//! than re-deriving means and distances, which is what keeps a diagnostic from quietly
//! disagreeing with the code it diagnoses.

/// Scales a vector to unit length, leaving a zero vector alone.
///
/// Every distance below is a dot product, which is only a cosine if both sides are unit
/// length; normalizing here is what lets the rest of this module stop thinking about it.
pub(crate) fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.iter_mut().for_each(|v| *v /= norm);
    }
}

/// How far apart two groups of unit-length embeddings are, under both criteria at once.
///
/// See [`group_distance`] for the identity relating the three fields, which is the whole
/// reason they are returned together rather than measured separately.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupDistance {
    /// The mean of every cross-group pairwise cosine distance: the criterion clustering merges
    /// on, against [`crate::MERGE_DISTANCE`].
    ///
    /// Computed from the embeddings alone, *without* the cannot-link substitution, so it is a
    /// distance rather than a decision and the identity below holds for it. Whether a merge is
    /// forbidden is a separate question with a separate answer, and folding the two into one
    /// number would erase both.
    pub average_linkage: f32,

    /// The cosine distance between the two groups' normalized means.
    ///
    /// This is what [`crate::ADOPTION_DISTANCE`] thresholds, and the quantity
    /// [`crate::IDENTIFY_DISTANCE`] is measured in -- the distance between a cluster's
    /// reference vector and another's.
    pub centroid: f32,

    /// `|a| * |b|`: the product of the lengths of the two *unnormalized* group means.
    ///
    /// At most 1, reached only when a group's members all point one way, and falling as either
    /// group grows or spreads out. It is the factor by which averaging distances exceeds the
    /// distance of averages.
    pub shrinkage: f32,
}

/// Both group distances and the shrinkage tying them together, or [`None`] if either group is
/// empty or has no direction.
///
/// Members must be unit length and of one dimensionality -- which is what one embedding model
/// produces, and what [`crate::Clustering::turn_embeddings`] hands over.
///
/// For unit-length members these are not two independent measurements but one. With `a` and
/// `b` the *unnormalized* group means, the mean of the cross-group pairwise cosine distances
/// is `1 - a.b`, while centroid distance is `1 - (a/|a|).(b/|b|)`. So:
///
/// ```text
/// average_linkage = 1 - |a| * |b| * (1 - centroid)
/// ```
///
/// Average linkage is centroid distance inflated by the shrinkage of the two means, and a
/// group mean shrinks as its group grows and spreads. That inflation is the size bias TASK-018
/// is about: a two-second fragment offered to sixty-seven turns of one speaker is charged for
/// the spread of the group it is being compared to, so the cluster most likely to own it is the
/// hardest one for it to join. On clusters 1 and 3 of session `20260810-093047` the two read
/// 0.604 and 0.429, putting the shrinkage at 0.693.
///
/// `average_linkage` is computed as the honest mean over pairs rather than from the identity,
/// so that `the_two_group_distances_are_one_identity` is asserting a claim about this code and
/// not rearranging its own algebra.
///
/// [`None`] rather than zeroes, matching [`crate::Spread::of`]: the distance between a group
/// and nothing does not exist, and a fabricated 0.000 reads as two identical voices. The other
/// [`None`] case -- a group whose members cancel to a zero mean -- has no direction to compare,
/// so its centroid distance would be arbitrary rather than merely uncertain.
pub fn group_distance(a: &[&[f32]], b: &[&[f32]]) -> Option<GroupDistance> {
    let (unit_a, length_a) = group_mean(a)?;
    let (unit_b, length_b) = group_mean(b)?;

    let sum: f32 = a
        .iter()
        .flat_map(|x| b.iter().map(move |y| (x, y)))
        .map(|(x, y)| 1.0 - dot(x, y))
        .sum();

    Some(GroupDistance {
        average_linkage: sum / (a.len() * b.len()) as f32,
        centroid: 1.0 - dot(&unit_a, &unit_b),
        shrinkage: length_a * length_b,
    })
}

/// A group's mean direction, and the length of its mean before normalizing.
///
/// Same order of operations as the pipeline's `reference_embedding` -- average, then
/// normalize -- so the direction returned here is the vector a cluster stores. What this adds is the length that
/// step throws away, which for unit-length members is the group's coherence: 1 when they all
/// point one way, falling toward 0 as they spread.
///
/// [`None`] for an empty group, and for one whose members cancel exactly. The second is
/// unreachable for real voices but trivially reachable in a test, and a zero vector normalizes
/// to itself, which would otherwise be reported as a confident distance of 1.0 to everything.
///
/// Visible to the crate because [`crate::reference_duration_sweep`] builds references out of
/// parts of a cluster and has to build them the way the pipeline's `reference_embedding`
/// does, or it would be
/// measuring an algorithm meethook does not run. Reaching for this rather than re-deriving a
/// mean over there is what makes that a fact about the code instead of a comment.
pub(crate) fn group_mean(members: &[&[f32]]) -> Option<(Vec<f32>, f32)> {
    let mut mean = vec![0.0f32; members.first()?.len()];
    for member in members {
        for (m, v) in mean.iter_mut().zip(*member) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= members.len() as f32;
    }

    let length = mean.iter().map(|v| v * v).sum::<f32>().sqrt();
    if length == 0.0 {
        return None;
    }
    normalize(&mut mean);
    Some((mean, length))
}

/// Cosine of two unit-length vectors, which for them is just the dot product.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// A unit vector pointing `degrees` away from the first axis, in the first two
/// dimensions. Cosine distance between two of these is `1 - cos(difference)`, so the tests
/// can name the distance they mean instead of a pile of decimals.
#[cfg(test)]
pub(crate) fn at(degrees: f32) -> Vec<f32> {
    let radians = degrees.to_radians();
    vec![radians.cos(), radians.sin(), 0.0, 0.0]
}

/// Borrowed views of the members of one group, the shape [`group_distance`] takes.
#[cfg(test)]
pub(crate) fn vectors<'a>(embeddings: &'a [Vec<f32>], group: &[usize]) -> Vec<&'a [f32]> {
    group.iter().map(|&i| embeddings[i].as_slice()).collect()
}

/// Group shapes worth measuring a distance over: two singletons, a singleton against a
/// spread group, two spread groups, and lopsided sizes -- the last because shrinkage is
/// where group size enters, so a pair of groups with very different spreads is the case
/// most likely to expose an arithmetic slip.
#[cfg(test)]
pub(crate) fn shapes() -> Vec<(Vec<usize>, Vec<usize>)> {
    vec![
        (vec![0], vec![4]),
        (vec![0], vec![4, 5, 6]),
        (vec![0, 1, 2], vec![4, 5, 6]),
        (vec![0, 1, 2, 3], vec![6]),
    ]
}

/// A cloud of one voice around 0 degrees and another around 70, spread enough that the two
/// group means are visibly shorter than their members.
#[cfg(test)]
pub(crate) fn two_clouds() -> Vec<Vec<f32>> {
    [0.0, 18.0, -14.0, 7.0, 70.0, 88.0, 55.0]
        .iter()
        .map(|d| at(*d))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AC#2 of TASK-018.01, and the reason [`GroupDistance`] returns three numbers rather than
    /// two: average linkage is centroid distance inflated by the shrinkage of the two group
    /// means. If a refactor ever computes one of the columns differently from the others, this
    /// is what says so, and it is the claim the whole stranded-cluster report rests on.
    #[test]
    fn the_two_group_distances_are_one_identity() {
        let embeddings = two_clouds();
        for (left, right) in shapes() {
            let measured =
                group_distance(&vectors(&embeddings, &left), &vectors(&embeddings, &right))
                    .expect("neither group is empty");
            let from_identity = 1.0 - measured.shrinkage * (1.0 - measured.centroid);
            assert!(
                (measured.average_linkage - from_identity).abs() < 1e-5,
                "{left:?} vs {right:?}: linkage {} but the identity says {from_identity} \
                 (centroid {}, shrinkage {})",
                measured.average_linkage,
                measured.centroid,
                measured.shrinkage
            );
        }
    }

    /// The boundary that makes the size bias legible. Two one-member groups have means equal to
    /// their members, so there is no shrinkage and the two criteria agree exactly -- which says
    /// that every gap between the two columns elsewhere is group size and spread, and nothing
    /// else.
    #[test]
    fn two_single_turn_groups_have_no_shrinkage() {
        let embeddings = two_clouds();
        let measured = group_distance(&vectors(&embeddings, &[0]), &vectors(&embeddings, &[4]))
            .expect("neither group is empty");

        assert!((measured.shrinkage - 1.0).abs() < 1e-6, "{measured:?}");
        assert!(
            (measured.average_linkage - measured.centroid).abs() < 1e-6,
            "{measured:?}"
        );
    }

    /// Two groups nothing can be said about: one with no members, and one whose members cancel
    /// to a mean of zero. Both are [`None`] rather than a fabricated distance -- a zero vector
    /// normalizes to itself and would otherwise report a confident 1.000 to everything.
    #[test]
    fn a_group_with_no_direction_has_no_distance() {
        let voice: &[f32] = &[1.0, 0.0, 0.0, 0.0];
        let opposite: &[f32] = &[-1.0, 0.0, 0.0, 0.0];

        assert_eq!(group_distance(&[voice], &[]), None);
        assert_eq!(group_distance(&[], &[voice]), None);
        assert_eq!(group_distance(&[voice, opposite], &[voice]), None);
    }
}
