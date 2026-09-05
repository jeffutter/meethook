//! Bundling the fragments a run asks about into questions.
//!
//! A default run offers every unresolved voice at or above `PROMPT_FLOOR_SECONDS`, and the
//! quiet tail of a session -- the two-second fragments diarization shed off the main
//! clusters -- comes up one question at a time, each with its own clip and its own name to
//! type. This module decides which of those fragments are worth asking about *together*:
//! fragments close enough to be the same person become one question per bundle, answered by
//! naming the bundle once rather than every member in turn.
//!
//! The bundles are computed over the queue as it stands when the session is opened, and they
//! do not move afterwards. Question numbers are fixed at that moment on purpose -- a queue
//! that re-sorts itself under the cursor mid-run would make "the next voice" a moving target
//! -- so a fragment named through some other door before its bundle comes up leaves the
//! bundle as it was built, and the fan-out skips what is already settled.
//!
//! # Where 0.40 comes from
//!
//! Measured on two real sessions (`20260818-153044-17973` and `20260818-143214-17755`), whose
//! below-floor fragments had already been resolved by hand, so the ground truth for who-is-who
//! was known while the distances were being read:
//!
//! - Same-person chains cluster tightly. The 20260818-153044 session carries two independent
//!   chains of the same speaker's fragments, and every within-chain centroid distance sits in
//!   **0.338–0.395**; the cross-chain distances between the two people's centroids sit just
//!   above the merge band.
//! - The smallest *different*-anchor pair measured was 0.324 -- one fragment genuinely closer
//!   than its neighbours' chain -- so no threshold separates the two sessions' good merges
//!   from their bad ones perfectly; 0.40 buys eleven good merges against one questionable,
//!   and the questionable one still lands where the user can see it and say otherwise.
//! - [`meethook_transcribe::MERGE_DISTANCE`] (0.45) records a measured 0.429 for a genuine
//!   different-person centroid pair, which is the ceiling this constant must stay under.
//!
//! It is deliberately the same value as [`meethook_transcribe::IDENTIFY_DISTANCE`]: both
//! answer "is this fingerprint the same voice", and both sit in the gap the measurements put
//! between 0.395 (same person) and 0.429 (a different one). They are separate constants
//! because they act on different things -- identification labels turns, grouping bundles
//! questions -- and would move independently if one of the two measurements came apart.

use std::collections::{BTreeMap, BTreeSet};

use meethook_session::SpeakerCluster;
use meethook_transcribe::{Attribution, Resemblance, cosine_distance, heard_at_once};

/// How far apart two fragments' fingerprints may be and still be bundled into one question.
///
/// Strict `<`, like every other distance gate in this toolchain: a pair exactly at the limit
/// is not merged, the way a voice exactly at the prompt floor is offered. See the module docs
/// for where the value comes from.
pub const GROUP_DISTANCE: f32 = 0.40;

/// One bundle of below-floor fragments, projected to what an interface can show across the
/// seam.
///
/// Built once when the session's queue is built and carried unmodified into every
/// [`Voice`](crate::Voice) the run offers from it, so the pane's picture of the bundles does
/// not change shape while the questions move through them.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentGroup {
    /// The stable "Unknown N" handles, in queue order. Two or more by construction: a
    /// singleton is not a bundle, and the run asks it about the ordinary way.
    pub members: Vec<String>,
    /// Total speech across the members, in seconds. What the composite row reports instead of
    /// any single fragment's duration.
    pub speech_seconds: f64,
    /// The closest resemblance to an enrolled name anywhere in the bundle, if any member has
    /// one. Re-ranked against the database as it stands when the question is offered, not
    /// frozen at build time: a person enrolled earlier in the run is somebody the bundle most
    /// like now, and the pane should say so.
    pub best: Option<Resemblance>,
}

/// A read-only walk to a component's root, for the veto check, which must not compress paths
/// while it walks.
fn find_root(parent: &[usize], mut x: usize) -> usize {
    while parent[x] != x {
        x = parent[x];
    }
    x
}

/// Which of the session's below-floor fragments get asked about together.
///
/// Returns the pool's components -- including singletons, which the caller folds back into
/// the ordinary queue -- sorted by lowest member id, each member list in queue order.
/// Clusters outside the pool never appear at all: the caller asks about them the ordinary way
/// anyway, so a voice that is above the floor or already settled simply reads as "not bundled".
/// Deterministic for a deterministic queue: the only floating-point inputs are the pairwise
/// distances, and ties break on pool position rather than on hash order or iteration luck.
///
/// `order` is the queue as [`queue`](crate::queue) built it -- first-appearance order, ids
/// breaking ties -- restricted to nothing here: the pool predicate below applies the prompt
/// floor itself, because a targeted run reaches this path with voices the floor would have held
/// back, and the bundles are about the quiet tail specifically.
///
/// `shown` and `denied` are the labelling and suppression state as the queue was built. The
/// pool is exactly "the quiet tail this offer would still ask about": below the floor,
/// unresolved against the database, and not settled -- the shared
/// [`queue::is_settled`](crate::queue::is_settled) helper, so the group-input predicate cannot
/// drift from the queue's.
pub(crate) fn fragment_groups(
    order: &[&SpeakerCluster],
    shown: &BTreeMap<u32, Attribution>,
    denied: &BTreeSet<u32>,
) -> Vec<Vec<u32>> {
    // The pool: the quiet, unsettled tail the bundling exists for. Unnamed on the queue's own
    // terms -- a named voice asks "is this right", which a bundle cannot answer -- and out of
    // settledness on the queue's own terms too, through the shared helper.
    let pool: Vec<&SpeakerCluster> = order
        .iter()
        .copied()
        .filter(|c| c.speech_seconds < crate::queue::PROMPT_FLOOR_SECONDS)
        .filter(|c| {
            shown
                .get(&c.id)
                .is_none_or(|attribution| !attribution.is_named())
        })
        .filter(|c| !crate::queue::is_settled(c, shown, denied))
        .collect();

    // Union-find over pool positions. No rank: a session sheds a handful of fragments, the
    // trees stay shallow, and a read-only root walk is all the veto check below needs.
    let mut parent: Vec<usize> = (0..pool.len()).collect();

    // Every candidate pair, nearest first: ascending distance is what makes the result
    // independent of pair order, and the position tiebreaks keep it total. Non-finite
    // distances sort last under `total_cmp` and never merge -- a NaN is not evidence of
    // likeness, whatever it sorts as.
    //
    // `saturating_sub` rather than `pool.len() - 1`: an empty pool -- no below-floor voices
    // left to bundle, the ordinary case -- must not overflow computing a capacity for zero
    // pairs.
    let mut pairs: Vec<(f32, usize, usize)> =
        Vec::with_capacity(pool.len() * pool.len().saturating_sub(1) / 2);
    for i in 0..pool.len() {
        for j in (i + 1)..pool.len() {
            let d = cosine_distance(&pool[i].embedding, &pool[j].embedding);
            pairs.push((d, i, j));
        }
    }
    pairs.sort_by(|(da, ia, ja), (db, ib, jb)| {
        da.total_cmp(db)
            .then_with(|| ia.cmp(ib))
            .then_with(|| ja.cmp(jb))
    });

    for (d, i, j) in pairs {
        if d.partial_cmp(&GROUP_DISTANCE) != Some(std::cmp::Ordering::Less) {
            // Sorted ascending, so everything after this pair is farther still. A non-finite
            // distance compares against nothing, which reads as "not close enough" and breaks
            // the same way: a NaN sorts last under `total_cmp` and never merges.
            break;
        }
        let (root_i, root_j) = (find_root(&parent, i), find_root(&parent, j));
        if root_i == root_j {
            continue;
        }

        // The heard-at-once veto, checked at merge time rather than pair level. The pair
        // itself being simultaneous is the obvious case, but single linkage can chain a voice
        // into a component before it meets another, and joining two components whose *members*
        // overlap would bundle two people who were talking at once -- the one fact
        // segmentation is sure of. So neither component may contain a member simultaneous with
        // any member of the other; O(size_a × size_b) over fragments, small by construction.
        let vetoes = |a: usize, b: usize| {
            (0..pool.len())
                .filter(|k| find_root(&parent, *k) == a)
                .any(|x| {
                    (0..pool.len())
                        .any(|y| find_root(&parent, y) == b && heard_at_once(pool[x], pool[y]))
                })
        };
        if vetoes(root_i, root_j) {
            continue;
        }

        parent[root_j] = root_i;
    }

    // Collect the components in pool order, which is queue order: the pool keeps `order`'s
    // relative sequence, so walking pool positions ascending presents each bundle the way the
    // queue will ask about it. Singletons travel along -- the caller is the one that knows a
    // one-member bundle is just the ordinary question.
    let mut by_root: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    for (k, cluster) in pool.iter().enumerate() {
        by_root
            .entry(find_root(&parent, k))
            .or_default()
            .push(cluster.id);
    }
    let mut groups: Vec<Vec<u32>> = by_root.into_values().collect();
    groups.sort_by_key(|g| g.iter().copied().min().unwrap_or(u32::MAX));
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use meethook_session::RepresentativeSegment;

    /// One fragment whose unit embedding points `angle` radians off the first axis: the
    /// cosine distance between two of these is `1 - cos(difference)`, so the tests name the
    /// separation they mean instead of a pile of decimals, and every embedding is unit length,
    /// which is the contract [`cosine_distance`] reads its dot product under. `with` is the
    /// heard-at-once exclusion list.
    fn fragment(id: u32, angle: f32, speech: f64, with: &[u32]) -> SpeakerCluster {
        let mut embedding = vec![0.0f32; 8];
        embedding[0] = angle.cos();
        embedding[1] = angle.sin();
        SpeakerCluster {
            id,
            embedding,
            speech_seconds: speech,
            first_spoke_seconds: 0.0,
            heard_at_once_with: with.to_vec(),
            representatives: vec![RepresentativeSegment {
                start: 0.0,
                end: speech,
            }],
        }
    }

    #[test]
    fn near_identical_fragments_bundle_and_their_distinct_neighbour_does_not() {
        // Three fragments within a few degrees of each other (distances ~0.001--0.003, far
        // under the limit) and one a quarter turn away (distance 1.0): the tight three share
        // one bundle -- single linkage is supposed to chain -- and the rest does not join.
        let a = fragment(1, 0.0, 2.0, &[]);
        let b = fragment(2, 0.05, 1.5, &[]);
        let c = fragment(3, 0.08, 1.0, &[]);
        let d = fragment(4, std::f32::consts::FRAC_PI_2, 2.0, &[]);
        let order = vec![&a, &b, &c, &d];
        let groups = fragment_groups(&order, &BTreeMap::new(), &BTreeSet::new());
        assert_eq!(groups, vec![vec![1, 2, 3], vec![4]]);
    }

    #[test]
    fn a_simultaneous_pair_stays_separate_however_close() {
        // Nearly identical fingerprints, heard at once: the veto outranks the distance.
        let a = fragment(1, 0.0, 2.0, &[2]);
        let b = fragment(2, 0.01, 2.0, &[1]);
        let order = vec![&a, &b];
        let groups = fragment_groups(&order, &BTreeMap::new(), &BTreeSet::new());
        assert_eq!(groups, vec![vec![1], vec![2]]);
    }

    #[test]
    fn a_simultaneous_voice_chained_into_one_component_vetoes_the_whole_merge() {
        // x is close to a, y is close to b, and a is close to y -- close enough that the a-y
        // pair alone would join the two components, and a and y do not overlap at pair level.
        // But x and y were heard at once, and single linkage already chained each of them into
        // its component: the merge-time veto must refuse the join on their account.
        let a = fragment(1, 0.0, 2.0, &[]);
        let x = fragment(2, 0.05, 1.0, &[4]);
        let b = fragment(3, -0.93, 2.0, &[]);
        let y = fragment(4, -0.88, 1.0, &[2]);
        let order = vec![&a, &x, &b, &y];
        let groups = fragment_groups(&order, &BTreeMap::new(), &BTreeSet::new());
        // a-x merged and b-y merged; the a-y join (under the limit, no pair-level overlap)
        // is refused because x and y were heard at once.
        assert_eq!(groups, vec![vec![1, 2], vec![3, 4]]);
    }

    #[test]
    fn the_limit_is_strict() {
        // At the limit: not merged, on the strict-< precedent. The dot product of a's
        // embedding with b's is the f32 just under 0.6, and `1.0 - c` is exact for such a c,
        // so the distance is the f32 one ulp *above* GROUP_DISTANCE -- the closest
        // representable reading of "a pair at the limit", which the strict gate keeps apart.
        let a = fragment(1, 0.0, 2.0, &[]);
        let mut b_emb = vec![0.0f32; 8];
        b_emb[0] = 0.59999996f32;
        b_emb[1] = 0.8f32;
        let b = SpeakerCluster {
            id: 2,
            embedding: b_emb,
            speech_seconds: 2.0,
            first_spoke_seconds: 0.0,
            heard_at_once_with: vec![],
            representatives: vec![RepresentativeSegment {
                start: 0.0,
                end: 2.0,
            }],
        };
        let d = cosine_distance(&a.embedding, &b.embedding);
        assert!(
            (d - GROUP_DISTANCE).abs() < 1e-6,
            "setup: distance is {d}, want ~0.40"
        );
        let order = vec![&a, &b];
        let groups = fragment_groups(&order, &BTreeMap::new(), &BTreeSet::new());
        assert_eq!(groups, vec![vec![1], vec![2]]);
    }

    #[test]
    fn settled_fragments_do_not_join() {
        // A named fragment is out of the pool entirely and so absent from the output: the
        // caller asks about it the ordinary way, and the bundle holds the unnamed pair alone.
        let a = fragment(1, 0.0, 2.0, &[]);
        let b = fragment(2, 0.05, 1.5, &[]);
        let named = fragment(3, 0.08, 3.0, &[]);
        let order = vec![&a, &b, &named];
        let shown = BTreeMap::from([(
            3,
            Attribution::Identified {
                name: "Ivan".into(),
                similarity: 0.9,
            },
        )]);
        let groups = fragment_groups(&order, &shown, &BTreeSet::new());
        assert_eq!(groups, vec![vec![1, 2]]);
    }

    #[test]
    fn fragments_above_the_floor_are_not_grouped_at_all() {
        // The bundling is about the quiet tail; a loud unresolved voice is out of the pool and
        // absent from the output, however close it sounds to the fragments.
        let loud = fragment(1, 0.0, 30.0, &[]);
        let quiet = fragment(2, 0.01, 2.0, &[]);
        let order = vec![&loud, &quiet];
        let groups = fragment_groups(&order, &BTreeMap::new(), &BTreeSet::new());
        assert_eq!(groups, vec![vec![2]]);
    }

    /// The ordinary run: nothing below the floor is left to bundle, so the pool is empty. This
    /// is the shape of most sessions, not an edge case -- it must not panic computing a
    /// capacity for zero pairs.
    #[test]
    fn an_empty_pool_produces_no_groups() {
        let loud = fragment(1, 0.0, 30.0, &[]);
        let order = vec![&loud];
        let groups = fragment_groups(&order, &BTreeMap::new(), &BTreeSet::new());
        assert_eq!(groups, Vec::<Vec<u32>>::new());
    }

    #[test]
    fn members_come_back_in_queue_order_and_groups_by_lowest_id() {
        // First-appearance order scrambles the ids relative to any id sort; the presentation
        // is queue order inside a bundle, lowest id between bundles.
        let big = fragment(7, 0.0, 4.0, &[]);
        let mid = fragment(3, 0.05, 3.0, &[]);
        let small = fragment(5, 0.08, 2.0, &[]);
        let apart = fragment(9, std::f32::consts::FRAC_PI_2, 1.0, &[]);
        let order = vec![&big, &mid, &small, &apart]; // the queue's own order: 7, 3, 5, 9
        let groups = fragment_groups(&order, &BTreeMap::new(), &BTreeSet::new());
        assert_eq!(groups, vec![vec![7, 3, 5], vec![9]]);
    }
}
