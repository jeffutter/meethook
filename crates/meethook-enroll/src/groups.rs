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
//! From the populations [`fragment_pairs`] measures over the two real multi-speaker sessions
//! on file, through the harness `examples/fragment-trials` prints -- no models, no audio,
//! seconds to re-run:
//!
//! ```text
//! cargo run --release -p meethook-enroll --example fragment-trials -- \
//!   ~/meethook/sessions/20260818-132033 ~/meethook/sessions/20260818-143027
//! ```
//!
//! The corpus, so the numbers have a name: `20260818-132033` (181 clusters, 145 under the
//! prompt floor, 10,440 below-floor pairs) and `20260818-143027` (21 clusters, 14 under the
//! floor, 91 pairs). Every figure below is copied from that command's output, not computed
//! beside it.
//!
//! **What was measured.** Heard-at-once negatives -- provably two people, segmentation's own
//! assertion: 23 pairs corpus-wide, min **0.2074** / p05 0.4196 / median 0.7793 / p95 0.9561
//! / max 1.0150. Exactly one of them (the 0.2074 pair) falls under this cut, and the
//! heard-at-once veto refuses it whatever the cut is -- so it bounds *trust in the veto*, not
//! the cut itself. Named-different negatives, the unguarded kind this cut actually governs:
//! **none on file**, because neither session carries a `speaker_names.json` today. Same-person
//! positives likewise: **none on file**. What the report carries instead is the unlabelled
//! shape -- 10,508 pairs, min 0.3240 / p05 0.5814 / median 0.8416 / p95 1.0294 / max 1.2573,
//! never scored into any rate.
//!
//! **The gap that has to travel with the number.** Positives need per-cluster identity that
//! neither `speaker_clusters.json` nor an unnamed transcript holds; producing them is an ear
//! sitting, filed as TASK-072, with TASK-073 re-pricing this constant from the labelled corpus
//! it yields. Until then 0.40 is the value the *bounded* window below admits, not a number
//! fitted to positives -- stated the way [`meethook_transcribe::TENTATIVE_DISTANCE`]'s
//! documentation states the same gap, and for the same reason: an assumed number presented as
//! a measurement is worse than a small one honestly bounded.
//!
//! **The window, and 0.40 inside it.** Upper edge: **0.429**, the closest ear-confirmed
//! different-person centroid pair on record -- carried secondhand from
//! [`meethook_transcribe::MERGE_DISTANCE`]'s documentation of session `20260810-093047`, a
//! directory not on this machine, cited as what that doc records rather than as something the
//! command reproduces. A cut at or past it admits the geometry that files one real person's
//! speech under another's name. Lower edge: **0.3240**, the closest below-floor pair anywhere
//! on file -- at or below it the bundling is vacuous over the whole corpus, asking nobody's
//! question twice. Inside `(0.3240, 0.429)` sit the observations that matter: 143027's
//! fourteen-fragment tail yields exactly one mergeable pair at **0.3952** (fragments 7 and
//! 18, twelve singletons -- the one bundled question TASK-060 watched the TUI ask), and
//! 132033's twelve pairs under 0.40 -- one guarded heard-at-once pair and eleven unlabelled
//! ones -- chain into six bundled questions covering seventeen fragments. Rejected
//! alternatives:
//! 0.45 (`MERGE_DISTANCE`'s value) is at or past the 0.429 ceiling and lets a second
//! provably-different pair in at 0.4196, separated only by the veto; anything at or below
//! 0.3240 asks nothing together anywhere; and widening on the unlabelled distribution cannot
//! be done at all until TASK-072 labels it, since an unlabelled pair scored either way is a
//! fabricated error rate. That 0.40 and [`meethook_transcribe::IDENTIFY_DISTANCE`] happen to
//! be the same number is a coincidence of two independent calibrations, not a derivation: they
//! answer different questions (bundle a question vs label a turn) and decision-004 rules out
//! moving one when the other is recalibrated. Each is priced on its own evidence above.
//!
//! One piece of history belongs with the window, because the numbers look familiar: this
//! constant's documentation once cited a "same-person" band of 0.338--0.395 and a 0.324
//! outlier, measured on sessions no machine holds. Those decimals are real -- they are
//! 132033's *unlabelled* below-floor pairs -- the eleven that sit under 0.40 beside the one
//! guarded pair -- but they were presented
//! as hand-confirmed ground truth from a corpus nobody can open. That is exactly the failure
//! this harness exists to make impossible, which is why every figure above carries the command
//! that prints it.
//!
//! **What would invalidate this measurement.** Any change to the embedding model --
//! [`meethook_transcribe::EMBEDDING_MODEL`] is pinned at rev `f0c48c29…` with sha256
//! `7bb2f06e…` and 256-dimensional output, and a model swap at the *same dimension* is
//! invisible to every existing guard, so the pinned hash is the thing to check against the
//! centroids stored in each `speaker_clusters.json`. Also: `PROMPT_FLOOR_SECONDS` here (it
//! selects the pool), the 0.5 s embeddable-turn minimum in transcribe's clustering (it shapes
//! what becomes a fragment), [`meethook_transcribe::MERGE_DISTANCE`] (the outer ceiling), the
//! mean-then-normalize pooling in `voice_vectors::group_mean` that produced these centroids,
//! and any weakening of the heard-at-once veto -- the negatives are guarded by it, so a weaker
//! veto means re-pricing directly against 0.2074. Re-checking costs a re-run of the command
//! above: seconds, no models, no audio. An unchanged printout after any of those changes is
//! the point of writing the command down at all.

use std::collections::{BTreeMap, BTreeSet};

use meethook_session::{SpeakerCluster, SpeakerNames};
use meethook_transcribe::{Attribution, Resemblance, Trial, cosine_distance, heard_at_once};

/// How far apart two fragments' fingerprints may be and still be bundled into one question.
///
/// Strict `<`, like every other distance gate in this toolchain: a pair exactly at the limit
/// is not merged, the way a voice exactly at the prompt floor is offered. See the module docs
/// for where the value comes from, and `examples/fragment-trials` for the command that
/// regenerates every number behind it.
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

/// How a below-floor fragment pair's identity is known, if it is.
///
/// The two negative variants are kept apart rather than folded into one "different people"
/// bit because they are different kinds of evidence: [`FragmentPairLabel::Overlapped`] is
/// segmentation's own assertion -- certain, and *guarded* by the production veto whatever the
/// cut is -- while [`FragmentPairLabel::DifferentPerson`] is a user's naming decision, which
/// is unguarded geometry. A rate computed over both together would read guarded negatives as
/// if they priced the unguarded pairs the cut actually governs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentPairLabel {
    /// Both fragments carry the same hand-given name in this session's `speaker_names.json`:
    /// one person, twice heard.
    SamePerson,
    /// The two fragments carry *different* hand-given names: two people, and nothing guards
    /// their distance from deciding anything -- this is the population a cut is priced
    /// against, when it exists.
    DifferentPerson,
    /// Segmentation heard the two talking over each other: provably two people, and the one
    /// label that needs no ears. Also the pair the shipping merge refuses regardless of the
    /// cut, so these pairs bound trust on the *unguarded* pairs rather than pricing them
    /// directly -- the caveat [`meethook_transcribe::tentative_pairs`] carries for the same
    /// signal.
    Overlapped,
    /// Nothing on disk says who either fragment is. Never scored: an unlabelled pair turned
    /// into a trial either way is a fabricated error rate.
    Unlabelled,
}

/// One measured distance between two below-floor fragments of one session, carrying the only
/// labels the on-disk contract can supply for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FragmentPair {
    /// The lower-fragment-id side of the pair; both sides always sit under the crate's
    /// `queue::PROMPT_FLOOR_SECONDS` (5 s of speech).
    pub fragment: u32,
    /// The higher-fragment-id side. Self-pairs do not exist.
    pub partner: u32,

    /// Cosine distance between the two centroids: `1.0 - dot` over unit vectors -- the same
    /// arithmetic `fragment_groups` decides with, which is what stops this report from
    /// being able to disagree with the bundling it describes.
    pub distance: f32,

    pub label: FragmentPairLabel,
}

impl FragmentPair {
    /// This pair as a scoring trial, or [`None`] when nothing labelled it.
    ///
    /// The label is turned into `same_speaker` here rather than at the call site so that a
    /// report never gets to decide what a pair means: an unlabelled pair scored either way is
    /// a fabricated error rate.
    pub fn trial(&self) -> Option<Trial> {
        let same_speaker = match self.label {
            FragmentPairLabel::SamePerson => true,
            FragmentPairLabel::DifferentPerson | FragmentPairLabel::Overlapped => false,
            FragmentPairLabel::Unlabelled => return None,
        };
        Some(Trial {
            same_speaker,
            distance: self.distance,
        })
    }
}

/// The distance populations [`GROUP_DISTANCE`] is priced from.
///
/// Every pair of this session's below-floor fragments -- both sides under the crate's
/// `queue::PROMPT_FLOOR_SECONDS` (5 s of speech), the pool the bundling applies its floor
/// to -- in ascending fragment id then ascending partner id, so two runs
/// over one file are diffable: the only way "the output did not change" gets checked at all.
///
/// **The pool is the duration floor only.** A live offer also drops settled voices
/// (`queue::is_settled`), so what is measured here is a
/// deliberate superset of any real run's pool: wider than what a run bundles, never narrower,
/// and stated rather than quietly compared against what a TUI showed.
///
/// # What the labels are, and the gap that has to travel with them
///
/// Two supervision sources, neither invented here. Segmentation's heard-at-once relation
/// labels negatives for free ([`FragmentPairLabel::Overlapped`]) -- but it is also the veto
/// the shipping rule applies, so those pairs bound trust in the guard rather than pricing the
/// unguarded pairs. Hand-given names label both ways: rows of this session's
/// `speaker_names.json` resolve to clusters by **bit-exact embedding equality** between
/// [`AssignedName::embedding`](meethook_session::AssignedName::embedding) and the cluster's
/// centroid -- never by the row's `cluster` field, because cluster ids are stable only within
/// one clustering run, which is exactly why the row carries the vector. Two fragments under
/// one name are [`FragmentPairLabel::SamePerson`]; under two names,
/// [`FragmentPairLabel::DifferentPerson`]; anything else is
/// [`FragmentPairLabel::Unlabelled`] and reaches no rate.
///
/// A heard-at-once pair that somehow also shares a name reads as [`FragmentPairLabel::Overlapped`]:
/// the production veto would refuse that assignment, so the naming cannot be trusted where the
/// two disagree, and segmentation's certainty wins.
///
/// Pairs whose embeddings differ in length, or are empty, take part in no distance: a bare
/// `zip` dot product across unequal spaces truncates into a plausible-looking number, so the
/// comparison would be fabricated. Skipping is silent the way
/// [`meethook_transcribe::tentative_pairs`] skips silently, so one bad row cannot stop the
/// rest of a session from being measured; a caller that wants the count subtracts
/// `pairs.len()` from `n(n-1)/2` over its own pool.
///
/// This function measures; it picks no threshold and holds no constant beyond the floor that
/// selects its fragments. Scoring belongs to [`meethook_transcribe::score_trials`],
/// presentation to `examples/fragment-trials`, and the arithmetic lives in the crate rather
/// than the example for the settled reason there: a diagnostic whose conventions nobody can
/// test is a number to believe rather than evidence.
pub fn fragment_pairs(clusters: &[SpeakerCluster], names: &SpeakerNames) -> Vec<FragmentPair> {
    let mut fragments: Vec<&SpeakerCluster> = clusters
        .iter()
        .filter(|cluster| cluster.speech_seconds < crate::queue::PROMPT_FLOOR_SECONDS)
        .collect();
    fragments.sort_by_key(|cluster| cluster.id);

    // Which clusters this session has named by hand, resolved through the embedding copy
    // rather than the recorded id, on the bit-exact convention `resolve_denials` uses.
    // Rows are applied in file order, so a (pathological) duplicate match lands on the last
    // row deterministically rather than by iteration luck.
    let mut named: BTreeMap<u32, &str> = BTreeMap::new();
    for row in &names.names {
        for cluster in clusters.iter() {
            if cluster.embedding == row.embedding {
                named.insert(cluster.id, &row.name);
            }
        }
    }

    let mut pairs = Vec::new();
    for (i, fragment) in fragments.iter().enumerate() {
        for partner in fragments[i + 1..].iter() {
            let Some(distance) = comparable_distance(&fragment.embedding, &partner.embedding)
            else {
                continue;
            };
            let label = if heard_at_once(fragment, partner) {
                FragmentPairLabel::Overlapped
            } else {
                match (named.get(&fragment.id), named.get(&partner.id)) {
                    (Some(one), Some(other)) if one == other => FragmentPairLabel::SamePerson,
                    (Some(_), Some(_)) => FragmentPairLabel::DifferentPerson,
                    _ => FragmentPairLabel::Unlabelled,
                }
            };
            pairs.push(FragmentPair {
                fragment: fragment.id,
                partner: partner.id,
                distance,
                label,
            });
        }
    }
    pairs
}

/// A cosine between two vectors that live in the same space, or [`None`] when they do not.
///
/// [`cosine_distance`] is a bare `zip` dot product, so unequal lengths silently truncate into
/// a plausible-looking number instead of an error. Empty inputs are excluded on the same
/// reading: a similarity across nothing is not a measurement.
fn comparable_distance(a: &[f32], b: &[f32]) -> Option<f32> {
    (!a.is_empty() && a.len() == b.len()).then(|| cosine_distance(a, b))
}

/// What the shipping cut actually bundles over a session's clusters, through the real
/// `fragment_groups` rather than a reimplementation of it.
///
/// Public for the reason [`meethook_transcribe::IDENTIFY_DISTANCE`] is: a diagnostic that
/// had to re-implement the walk would be measuring a rule the tool does not run. The pools
/// differ -- this walks the floor-only pool (empty labelling state, nothing settled), because
/// a calibration report describes the quiet tail as the file holds it, while a live run
/// further restricts the pool to what it will still ask about. Returns every component,
/// singletons included, in `fragment_groups`' order.
pub fn fragment_bundles(clusters: &[SpeakerCluster]) -> Vec<Vec<u32>> {
    let order: Vec<&SpeakerCluster> = clusters.iter().collect();
    fragment_groups(&order, &BTreeMap::new(), &BTreeSet::new())
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

    // ------------------------------------------------------------------------------
    // fragment_pairs / fragment_bundles: the measurement behind GROUP_DISTANCE
    // ------------------------------------------------------------------------------

    use meethook_session::{AssignedName, SessionId, SpeakerNames};

    fn session_id() -> SessionId {
        SessionId::parse("20260818-132033").expect("valid test id")
    }

    /// A naming file where each row carries that fragment's own centroid -- the bit-exact
    /// handle a reader resolves through, not the recorded id.
    fn named_with(rows: &[(&SpeakerCluster, &str)]) -> SpeakerNames {
        SpeakerNames::new(
            session_id(),
            rows.iter()
                .map(|(cluster, name)| AssignedName {
                    cluster: cluster.id,
                    name: (*name).to_string(),
                    embedding: cluster.embedding.clone(),
                })
                .collect(),
        )
    }

    fn unnamed() -> SpeakerNames {
        SpeakerNames::new(session_id(), Vec::new())
    }

    #[test]
    fn the_pool_is_below_the_floor_strictly_on_both_sides() {
        // One fragment just under the floor, one exactly on it, one far above: only the
        // strictly-below pair has a distance measured, matching the bundling's own `<`.
        let quiet_a = fragment(1, 0.0, crate::queue::PROMPT_FLOOR_SECONDS - 0.01, &[]);
        let quiet_b = fragment(2, 0.05, crate::queue::PROMPT_FLOOR_SECONDS - 0.02, &[]);
        let on_floor = fragment(3, 0.1, crate::queue::PROMPT_FLOOR_SECONDS, &[]);
        let loud = fragment(4, 0.2, 30.0, &[]);
        let pairs = fragment_pairs(&[quiet_a, quiet_b, on_floor, loud], &unnamed());
        assert_eq!(
            pairs
                .iter()
                .map(|p| (p.fragment, p.partner))
                .collect::<Vec<_>>(),
            vec![(1, 2)],
            "both sides must be under the floor; a voice exactly on it is offered, not bundled"
        );
    }

    #[test]
    fn shared_names_label_positives_and_distinct_names_label_negatives() {
        // Four below-floor fragments: two share "Ivan", one is "Alex", one stays unnamed.
        let a = fragment(1, 0.0, 2.0, &[]);
        let b = fragment(2, 0.1, 2.0, &[]);
        let c = fragment(3, 0.2, 2.0, &[]);
        let d = fragment(4, 0.3, 2.0, &[]);
        let names = named_with(&[(&a, "Ivan"), (&b, "Ivan"), (&c, "Alex")]);
        let pairs = fragment_pairs(&[d, c, b, a], &names); // input order scrambled
        let labelled: Vec<_> = pairs
            .iter()
            .map(|p| ((p.fragment, p.partner), p.label))
            .collect();
        use FragmentPairLabel::{DifferentPerson, SamePerson, Unlabelled};
        assert_eq!(
            labelled,
            vec![
                ((1, 2), SamePerson),
                ((1, 3), DifferentPerson),
                ((1, 4), Unlabelled),
                ((2, 3), DifferentPerson),
                ((2, 4), Unlabelled),
                ((3, 4), Unlabelled),
            ],
            "a pair is labelled only when both sides are named, by what the names say"
        );
    }

    #[test]
    fn heard_at_once_overrides_a_shared_name_because_the_veto_would_have_refused_it() {
        // Segmentation's certainty outranks a naming the production rule could not have made.
        let a = fragment(1, 0.0, 2.0, &[2]);
        let b = fragment(2, 0.05, 2.0, &[1]);
        let names = named_with(&[(&a, "Ivan"), (&b, "Ivan")]);
        let pairs = fragment_pairs(&[a, b], &names);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].label, FragmentPairLabel::Overlapped);
    }

    #[test]
    fn only_labelled_pairs_become_trials() {
        // The rule that keeps a report from inventing an error rate: same-person trials are
        // true, *both* negative variants are false, unlabelled is no trial at all.
        let pair = |label| FragmentPair {
            fragment: 1,
            partner: 2,
            distance: 0.3,
            label,
        };
        assert_eq!(
            pair(FragmentPairLabel::SamePerson).trial(),
            Some(Trial {
                same_speaker: true,
                distance: 0.3
            })
        );
        for label in [
            FragmentPairLabel::DifferentPerson,
            FragmentPairLabel::Overlapped,
        ] {
            assert_eq!(
                pair(label).trial(),
                Some(Trial {
                    same_speaker: false,
                    distance: 0.3
                }),
                "{label:?} proves two people"
            );
        }
        assert_eq!(pair(FragmentPairLabel::Unlabelled).trial(), None);
    }

    #[test]
    fn pair_order_is_total_and_stable_across_runs() {
        // Ascending fragment id then ascending partner id regardless of input order, twice
        // over the same data: the ordering is what makes "the output did not change" checkable.
        let a = fragment(9, 0.0, 2.0, &[]);
        let b = fragment(3, 0.1, 1.0, &[]);
        let c = fragment(5, 0.2, 4.0, &[]);
        let once = fragment_pairs(&[c.clone(), a.clone(), b.clone()], &unnamed());
        let twice = fragment_pairs(&[b, c, a], &unnamed());
        let keys = |pairs: &[FragmentPair]| {
            pairs
                .iter()
                .map(|p| (p.fragment, p.partner))
                .collect::<Vec<_>>()
        };
        assert_eq!(keys(&once), vec![(3, 5), (3, 9), (5, 9)]);
        assert_eq!(keys(&once), keys(&twice));
    }

    #[test]
    fn incomparable_embeddings_are_skipped_not_truncated() {
        // An 8-d and a 9-d vector: `cosine_distance` would zip to eight terms and print a
        // number. The pair must simply not exist, and its neighbours must still be measured.
        let mut short_emb = vec![0.0f32; 8];
        short_emb[0] = 1.0;
        let mk = |id: u32, embedding: Vec<f32>| SpeakerCluster {
            id,
            embedding,
            speech_seconds: 2.0,
            first_spoke_seconds: 0.0,
            heard_at_once_with: vec![],
            representatives: vec![RepresentativeSegment {
                start: 0.0,
                end: 2.0,
            }],
        };
        let short = mk(1, short_emb);
        let mut long_emb = fragment(2, 0.1, 2.0, &[]).embedding;
        long_emb.push(0.1);
        let long = mk(2, long_emb.clone());
        let third = fragment(3, 0.2, 2.0, &[]); // 8-d
        let fourth = mk(4, long_emb); // 9-d, same length as cluster 2
        let pairs = fragment_pairs(&[short, long, third, fourth], &unnamed());
        assert_eq!(
            pairs
                .iter()
                .map(|p| (p.fragment, p.partner))
                .collect::<Vec<_>>(),
            vec![(1, 3), (2, 4)],
            "every cross-length pair is skipped; same-space neighbours are still measured"
        );
    }

    #[test]
    fn no_self_pairs_and_an_empty_pool_measures_nothing() {
        let lone = fragment(1, 0.0, 2.0, &[]);
        assert!(fragment_pairs(std::slice::from_ref(&lone), &unnamed()).is_empty());
        let loud = fragment(2, 0.0, 30.0, &[]);
        assert!(fragment_pairs(&[lone, loud], &unnamed()).is_empty());
    }

    #[test]
    fn fragment_bundles_reports_what_the_shipping_cut_does() {
        // The wrapper must agree with the bundling itself: a tight pair forms one component,
        // a simultaneous pair never joins however close, above-floor voices never appear.
        let a = fragment(1, 0.0, 2.0, &[]);
        let b = fragment(2, 0.05, 1.5, &[]);
        let c = fragment(3, std::f32::consts::FRAC_PI_2, 2.0, &[4]);
        let d = fragment(4, std::f32::consts::FRAC_PI_2 + 0.01, 2.0, &[3]);
        let loud = fragment(5, 0.0, 30.0, &[]);
        let clusters = [a, b, c, d, loud];
        assert_eq!(
            fragment_bundles(&clusters),
            vec![vec![1, 2], vec![3], vec![4]],
            "the report walks the same union-find and veto the tool runs"
        );
    }
}
