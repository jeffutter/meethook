//! Putting names to the voices diarization found.
//!
//! Clustering can say "these turns are all the same person"; it cannot say who that person
//! is, because nothing in a meeting's audio carries a name. That has to come from outside
//! the session -- from `speakers.json`, which holds one reference voice per person somebody
//! has already identified through `enroll`.
//!
//! This module is the whole decision and none of the plumbing: no file reading, no models,
//! no session directory. Given the meeting's clusters and the enrolled database, it says
//! which clusters are which people. Everything it does is a dot product, so the thresholds
//! and tie-breaks below are testable in microseconds against hand-written vectors.

use std::collections::{BTreeMap, BTreeSet};

use meethook_session::{EnrolledSpeakers, SpeakerCluster};
use serde::Serialize;

/// How far apart a meeting's voice and an enrolled reference may be and still be one person.
///
/// # What this thresholds
///
/// Cosine distance between **two single vectors**: the cluster's normalized mean-pooled
/// centroid on one side, an enrolled reference on the other. 0 is the same direction, 1 is
/// orthogonal. Nothing is averaged over pairs on either side.
///
/// # It is not `speakers::MERGE_DISTANCE`, and not merely because it is calibrated separately
///
/// A reader arriving from `speakers.rs` expects two constants that are both cosine distances,
/// both about "same voice or not", to be the same measurement applied in two places. They are
/// not, and that reader is the one these paragraphs exist to stop.
///
/// `MERGE_DISTANCE` thresholds **average linkage**: the mean over every cross-group pair of
/// turn embeddings, one distance per pair of turns, averaged. This constant thresholds the
/// distance between the two sides' *means*. Averaging distances is not the distance of
/// averages, and for unit-length members [`crate::group_distance`] gives the exact relation
/// between the two, word for word as it states it there:
///
/// ```text
/// average_linkage = 1 - shrinkage * (1 - centroid)
/// ```
///
/// `shrinkage` is the product of the two *unnormalized* group-mean lengths: at most 1, and
/// falling as either group grows or spreads out. So average linkage is centroid distance
/// inflated by the shrinkage of the two means -- always the larger of the two numbers, with a
/// gap that widens as either group gets less coherent. Two constants set to one value are
/// therefore two different cuts, and a pair of voices can sit on opposite sides of both.
///
/// A pair that did. On clusters 1 and 3 of session `20260810-093047` the two group distances
/// read **0.604** linkage and **0.429** centroid, putting the shrinkage at **0.693**. Both
/// clusters were confirmed by ear to be two different people (Andrew, and Ryan). At a
/// shared 0.45, clustering correctly declined the merge at 0.604 while identification accepted
/// at 0.429 and filed 124.1 s of Ryan -- 9% of the speech on that track -- under Andrew.
/// See TASK-020.
///
/// Those three numbers are the ones the defect was found at, on the clustering that shipped
/// before TASK-018. Re-measured on the clustering that ships now -- where the same two voices
/// hold 423.7 s and 119.5 s rather than 396.7 s and 124.1 s -- that pair reads **0.656**
/// linkage, **0.429** centroid, shrinkage **0.603**, which `cluster-speaker-track` prints in
/// its speaker-vs-speaker block. Both restatements satisfy the identity above, the centroid is
/// unchanged to three figures, and the gap between the quantities has *widened* rather than
/// closed, so nothing here rests on the older grouping. Two constants, 0.45 and 0.40, now sit
/// either side of that pair rather than both above it -- by 0.029 on this side, where it was
/// 0.079 while this constant was 0.35. That margin is the tightest constraint on this value and
/// is what puts a hard ceiling below 0.429 on any further loosening.
///
/// # Where the value comes from
///
/// From the two measured populations below, not from `MERGE_DISTANCE`.
///
/// **Cross-session corpus (TASK-014.04):** LibriSpeech dev-clean, 40 speakers, 67 items,
/// grouped so every same-speaker pair spans different recording occasions. Same-speaker 36
/// pairs, min 0.037 / median 0.129 / max 0.702; different-speaker 2170 pairs, min **0.364** /
/// median 0.897. The two populations overlap across `[0.364, 0.702]`, so no cut separates them
/// and every choice here buys one mistake with the other. Re-priced from that ticket's cached
/// embeddings:
///
/// | cut   | different-speaker accepted | same-speaker rejected |
/// |-------|----------------------------|-----------------------|
/// | 0.350 | 0/2170 = 0.00%             | 2/36 = 5.56%          |
/// | 0.450 | 9/2170 = 0.41%             | 2/36 = 5.56%          |
/// | 0.550 | 17/2170 = 0.78%            | 2/36 = 5.56%          |
///
/// The false-reject column barely moves across that band because the same-speaker distribution
/// is tight, so in this region the trade is bought almost entirely with false accepts. And all
/// 9 false accepts at 0.45 are a *single* pair of speakers, whose cross-session distances sit
/// at 0.364-0.416 -- confusable at every occasion rather than occasionally.
///
/// **Real meethook audio:** the Andrew/Ryan pair above, at centroid 0.429, is an upper bound
/// measured within one session on one microphone and one call. That is the easiest condition
/// this constant will ever face, cross-session variation being strictly larger, and 0.45
/// already fails it.
///
/// # Where the value comes from, second pass: what the corpus could not see
///
/// The table above is why this constant was **0.35** -- the largest cut with zero measured
/// misattributions, since the corpus's nearest different-speaker pair sits at 0.364. What that
/// corpus cannot price is the failure the loosening below is bought to fix, because LibriSpeech
/// has no fragments in it: every item is a clean read of tens of seconds, so its same-speaker
/// distribution is tight and 0.35 rejects almost nothing.
///
/// Real sessions are not like that. On `20260818-132033` -- 181 clusters over 17.1 min of
/// speech, seven people, most of the talk time in clusters of five to ten seconds -- the cut
/// prices out like this, against a database in which all seven were already enrolled:
///
/// | cut   | clusters identified | speech identified |
/// |-------|---------------------|-------------------|
/// | 0.350 | 32/181              | 10.7 / 17.1 min   |
/// | 0.400 | 49/181              | 11.9 / 17.1 min   |
/// | 0.450 | 71/181              | 13.4 / 17.1 min   |
///
/// A short cluster's centroid is pooled over less speech and is therefore noisier, so a genuine
/// match lands further out than the corpus's tens-of-seconds reads ever do. Those are the
/// clusters between 0.35 and 0.45, and by inspection they are overwhelmingly the enrolled
/// speakers rather than strangers -- the session has seven people in it and nobody else.
///
/// So **0.40**, moved from 0.35, and both halves of that trade are real:
///
/// - It buys +17 clusters and +1.2 min on the session above: 17 `Unknown N`s a user would
///   otherwise answer by hand, on a run that asked 34 questions.
/// - It gives up the zero-misattribution property. 0.40 is past the corpus's nearest
///   different-speaker pair at 0.364, so it accepts part of that one confusable pair's
///   0.364-0.416 band. It is *not* past the ear-confirmed Andrew/Ryan pair at 0.429, which is
///   the constraint that decided the value: 0.45 is unavailable at any price, because it is
///   known to refile one real person's speech under another real person's name.
///
/// Three caveats belong with those numbers. The corpus holds the channel constant -- one
/// volunteer, room and microphone per speaker -- so its same-speaker distances are a *floor*
/// and its false-reject rate is optimistic. 36 same-speaker pairs put that rate at roughly one
/// significant figure: a 95% interval of about 1.5%-18%. And the session table is one session
/// with no ground truth beyond who was in the room, so it prices the *gain* from loosening
/// without pricing the loss. TASK-014 still owes the recording sitting that would measure both
/// against meethook's own capture channel, and it is now owed more than it was at 0.35.
///
/// The two mistakes here are not symmetric, and the bias still follows from that. A false match
/// puts one person's words under another person's name in a transcript nobody will re-read. A
/// false rejection is an `Unknown N` the user fixes in `enroll` in ten seconds. What moved is
/// the second half of that sentence: at 0.35 a session like the one above produces `Unknown N`s
/// by the dozen, and "ten seconds" was a claim about answering one of them.
///
/// Public so that the measurement can name the cut it is measuring against: the
/// `cluster-speaker-track` example prints every cluster-to-reference distance alongside this
/// value, and a calibration constant a diagnostic has to hard-code a copy of is a constant
/// that drifts out of agreement with the code it claims to describe. Exported for reading,
/// not for deciding -- [`identify_clusters`] is argmax *then* threshold, so anything that
/// compares against this on its own will call a reference that clears the cut but is not the
/// closest a match, which it is not.
pub const IDENTIFY_DISTANCE: f32 = 0.40;

/// How far apart a meeting's *fragment* and an in-session name's voice may be and still earn
/// a **marked** guess at the name, where the strict pass earns nothing.
///
/// # What this thresholds
///
/// The same quantity [`IDENTIFY_DISTANCE`] does -- cosine distance between two single
/// vectors, the fragment's normalized mean-pooled centroid against an enrolled reference --
/// compared the same way (`1.0 - similarity < TENTATIVE_DISTANCE`, strictly below), so the
/// calibration report and the shipping pass cannot differ by one `<`. What differs is where
/// the number is allowed to sit, and why it may sit wider than [`IDENTIFY_DISTANCE`]:
///
/// - **Tail-only.** Only clusters under [`TENTATIVE_FLOOR_SECONDS`] are ever scored against
///   it. A wrong guess there costs one visibly-marked line -- `Name?` in the transcript --
///   rather than one person's words filed unmarked under another's name, which is the mistake
///   [`IDENTIFY_DISTANCE`] prices itself against.
/// - **In-session pool only.** [`tentative_identifications`] scores against names the strict
///   pass already awarded elsewhere in this session, never against the whole database. The
///   guess space is the meeting's own confirmed cast, and the tool never guesses somebody who
///   was not in the room.
///
/// Those two scopes are what make a looser cut admissible at all; see the decision record
/// revising decision-004's no-middle-tier ruling for the argument.
///
/// # Where the value comes from
///
/// From the populations [`tentative_pairs`] measures over real sessions, through the trials
/// harness (`examples/speaker-trials --tentative`), not from [`IDENTIFY_DISTANCE`]. Two
/// things about that measurement have to travel with the number:
///
/// **What the on-disk contract can and cannot label.** A pair of clusters carries one free
/// supervision signal: whether segmentation heard them talking over each other. That labels
/// *negatives* -- provably two people, so a guess of one's name for the other would be wrong
/// -- and it is also the guard the production rule applies, so those pairs are vetoed there
/// whatever the cut is. *Positives* -- a fragment and the cluster of the person it really is
/// -- require per-cluster identity, which neither `speaker_clusters.json` nor an unnamed
/// transcript holds; labelling them is ear confirmation against representative clips, and
/// that sitting has not been done for this constant yet. So the value below is the
/// conservative edge of the admissible window the negative side and the demonstrated
/// confusable pair bound, and it is flagged for re-pricing when an ear-labelled population
/// exists. It is not an assumed number presented as a measurement: every number in its
/// documentation was measured, and the gap between what was measured and what the value
/// claims is stated rather than papered over.
///
/// **The populations themselves**, measured with `examples/speaker-trials --tentative` over
/// the two real multi-speaker sessions on file (sessions of the same shape the
/// [`IDENTIFY_DISTANCE`] documentation prices against):
///
/// - `20260818-132033` -- 181 clusters over 17.1 min of speech, seven people, 145 of the
///   clusters under the floor. Heard-at-once (fragment, partner) negatives: 74 pairs,
///   min **0.207** / p05 0.359 / median 0.722. Nearest *non-overlapping* partner per
///   fragment, unlabelled: min 0.231 / p05 0.302 / median 0.458 / p95 0.605.
/// - `20260818-143027` -- 21 clusters, 14 under the floor. Negatives: 7 pairs, min
///   **0.509**. Nearest non-overlapping partner: min 0.249 / median 0.526.
///
/// Three readings belong with those numbers.
///
/// The negative minimum of 0.207 says how close two *provably different* people's centroids
/// come inside one session on one microphone. It does not bound this cut directly, because
/// the heard-at-once veto refuses exactly those pairs in production whatever the cut is --
/// it bounds *trust in the veto*: a cut this wide is safe only while the veto stands, and a
/// future change that weakens it must re-price this constant against 0.207 rather than
/// inherit it.
///
/// The upper bound is the demonstrated in-session confusable pair the
/// [`IDENTIFY_DISTANCE`] documentation records: two ear-confirmed different people whose
/// centroids sit at **0.429** within one session. A cut at or past it admits the geometry
/// that misfiles one real person under another's name, even in the marked form.
///
/// The lower bound is [`IDENTIFY_DISTANCE`] itself: anything at or below 0.40 is already
/// decided by the strict pass, and a fragment the strict pass refused cannot clear a tighter
/// cut against the same reference, so the band would be vacuous.
///
/// So the admissible window is `(0.40, 0.429)`, and the value is **0.42**: inside it, below
/// the demonstrated confusable-pair distance by 0.009, and above the strict cut by 0.02. The
/// margin to the confusable pair is thinner than the one [`IDENTIFY_DISTANCE`] keeps, and
/// that is deliberate -- the marker is what buys the width -- but it is a margin, and it is
/// stated. Rejected alternatives: anything at or past 0.429, on the pair above; anything at
/// or below 0.40, on vacuity; and a wider cut justified by the unlabelled nearest-partner
/// distribution (median 0.458), which mixes correct and wrong neighbours and therefore cannot
/// price the false-guess side at all until it is labelled by ear.
///
/// Public for the same reason [`IDENTIFY_DISTANCE`] is: a calibration constant a diagnostic
/// has to hard-code a copy of is a constant that drifts out of agreement with the code it
/// claims to describe.
pub const TENTATIVE_DISTANCE: f32 = 0.42;

/// How little speech a cluster may hold and still be a *fragment* rather than a speaker.
///
/// The eligibility gate for [`tentative_identifications`]: a cluster is scored against
/// [`TENTATIVE_DISTANCE`] only when `speech_seconds < TENTATIVE_FLOOR_SECONDS`. The
/// comparison is strict the way every floor in the codebase spells it -- a cluster sitting
/// exactly on the floor is a speaker, full stop, and keeps the strict pass's honest behaviour
/// of staying unnamed until a human says otherwise.
///
/// # Not a shared constant with enroll's `PROMPT_FLOOR_SECONDS`
///
/// That one is `pub(crate)` in `meethook-enroll` and cannot be imported here, so the two
/// live in separate crates at the same value and a test pins the parity rather than a shared
/// definition importing it. They ask the same question -- "too little speech to be worth the
/// machinery aimed at speakers" -- and disagreeing about the answer would leave a band that
/// guesses at voices enroll has already decided not to ask about, or misses voices it does
/// ask about. See the three-floors block in `meethook-enroll`'s `queue.rs` for the relation
/// to the other two floors.
pub const TENTATIVE_FLOOR_SECONDS: f64 = 5.0;

/// A cluster matched to an enrolled speaker.
#[derive(Debug, Clone, PartialEq)]
pub struct Identification {
    pub name: String,

    /// Cosine similarity between the cluster's voice and that speaker's reference, in
    /// `[-1, 1]` but in practice above `1.0 - IDENTIFY_DISTANCE` or this would not exist.
    pub similarity: f32,
}

/// How much one voice sounds like one enrolled person.
///
/// Not an [`Identification`]. That is a *decision* -- argmax, then a threshold -- and its
/// existence is the claim that this cluster is this person. This is an observation with no
/// decision in it: one row of a ranked list a person is being shown so that *they* can decide.
/// See [`rank_enrolled`] for why the list is not cut anywhere.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Resemblance {
    /// The enrolled name, exactly as `speakers.json` spells it, so answering with it stores
    /// another reference against the person already there rather than creating a near-duplicate.
    pub name: String,

    /// Cosine similarity to this person's *nearest* comparable reference, in `[-1, 1]`.
    ///
    /// The number worth reading is the gap to the next entry rather than the value itself:
    /// 0.71 against 0.38 is confident, 0.52 against 0.51 means go and listen.
    pub similarity: f32,

    /// How many recordings of this person `speakers.json` holds -- **all** of them, including
    /// any that `similarity` could not be taken from.
    ///
    /// [`EnrolledSpeakers::references`], not a local count, so this is the same multiplicity
    /// [`meethook_session::MAX_REFERENCES_PER_SPEAKER`] caps and the same one `meethook
    /// speakers` prints. It distinguishes somebody with 12 stored recordings from somebody
    /// with 1, which is most of what tells a well-established name from a guess made once.
    pub references: usize,
}

/// Matches each cluster against the enrolled database, keyed by cluster id.
///
/// A cluster appears in the result only if it matched somebody: absence *is* "nobody we
/// know", which is why there is no `Option` in the value and no entry for every cluster.
///
/// One reference per person, argmax over all of them, one threshold. Deliberately no
/// "ambiguous" middle tier between match and no-match: a three-way outcome needs a UI to
/// resolve it, `transcribe` prompts for nothing on any path, and an unresolved third state
/// would just be an `Unknown N` with extra machinery behind it.
///
/// An empty database identifies nobody, which is the normal state of every install before
/// anyone has been enrolled.
///
/// # Two clusters that both match one person
///
/// There are two of these cases and they want opposite answers, so the rule turns on which
/// one it is.
///
/// When nothing says the two are different people, **both get the name**. That is the honest
/// reading of the evidence -- clustering split one voice in two -- and it renders as one
/// person speaking throughout, which is what happened.
///
/// When [`SpeakerCluster::heard_at_once_with`] says they are different people, both cannot.
/// Segmentation heard those two voices overlapping, so no similarity between their centroids
/// makes them one person; clustering already refuses to merge such a pair, and identification
/// handing them one name puts it straight back together under a name. So:
///
/// - For each enrolled name **independently**, take the clusters whose argmax is that name
///   and which cleared [`IDENTIFY_DISTANCE`] -- the contenders, decided exactly as before.
/// - Order them by similarity descending, ties by ascending cluster id, so the outcome does
///   not depend on the order `clusters` arrived in.
/// - Walk that order and award the name to a contender **iff it is not heard-at-once with any
///   contender already awarded this name**.
/// - A contender that is vetoed is simply unidentified. It does **not** fall back to its
///   second-nearest reference, and it does not go on to contest another name.
///
/// Three parts of that are decisions rather than phrasing.
///
/// **The nearest keeps the name, rather than both being rejected.** Rejecting both throws
/// away an answer that is provably right in order to avoid one that is provably wrong, and
/// invents two `Unknown N`s where one of them is correct. It is also the "ambiguous" middle
/// tier this function deliberately does not have, under another name.
///
/// **The veto is against every contender already awarded, not just the nearest one.** This is
/// the case the obvious rule gets wrong, and it is why it matters that exclusion is not
/// transitive: with contenders C1 (nearest), C2, C3 and the only exclusion between C2 and C3,
/// "drop whoever is excluded from the winner" awards the name to all three and leaves C2 and
/// C3 -- two people the segmenter heard at once -- under it. Greedy against the awarded set
/// gives it to C1 and C2 and drops C3. It is deterministic for any number of contenders, and
/// it degenerates to exactly the old behaviour when there are no exclusions.
///
/// **Greedy, not a maximum independent set.** Maximising the count awarded could hand a name
/// to two distant clusters in preference to one near one, which is the wrong bias -- and it is
/// NP-hard besides. This is not an oversight to improve on.
///
/// **No fallback to a second reference,** here or in the leftover-adoption pass. The
/// second-nearest reference is by construction the one that already lost the argmax, and
/// awarding it is the same operation that filed 124.1 s of one person under another's name in
/// session `20260810-093047` (TASK-020). It also cascades: a fallback can contest a third
/// name, whose loser can contest a fourth, turning a per-name decision into a global
/// assignment problem with no evidence behind it.
///
/// # Why this is not the shape the adoption pass uses, which is not a divergence
///
/// The leftover-adoption pass in `speakers.rs` applies the same constraint *before* its
/// argmax -- a blocked target is simply not a candidate. This applies it after. The two are
/// answers to differently-shaped questions rather than two answers to one: adoption's
/// constraint holds between a fragment and each candidate target, so it is known before the
/// choice is made; identification's holds between two clusters, and no reference is blocked
/// for a cluster a priori -- the conflict does not exist until some *other* cluster has taken
/// the name. Adoption vetoes candidates. Identification vetoes decisions. See TASK-018.02.02.
pub fn identify_clusters(
    clusters: &[SpeakerCluster],
    enrolled: &EnrolledSpeakers,
) -> BTreeMap<u32, Identification> {
    // Each cluster's own argmax-then-threshold, unchanged, grouped by the name it claimed
    // because the conflict resolved below is per name.
    let mut contenders: BTreeMap<String, Vec<(&SpeakerCluster, f32)>> = BTreeMap::new();
    for cluster in clusters {
        if let Some(best) = best_match(&cluster.embedding, enrolled) {
            contenders
                .entry(best.name)
                .or_default()
                .push((cluster, best.similarity));
        }
    }

    let mut identified = BTreeMap::new();
    for (name, mut claiming) in contenders {
        claiming.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.id.cmp(&b.0.id)));

        let mut awarded: Vec<&SpeakerCluster> = Vec::new();
        for (cluster, similarity) in claiming {
            if awarded.iter().any(|held| heard_at_once(cluster, held)) {
                continue;
            }
            awarded.push(cluster);
            identified.insert(
                cluster.id,
                Identification {
                    name: name.clone(),
                    similarity,
                },
            );
        }
    }
    identified
}

/// The second, looser band of identification: marked guesses for the fragments the strict
/// pass left unnamed.
///
/// Runs **after** [`identify_clusters`] completes and reads its finished map: the guess
/// space is the map's image -- names the strict pass already awarded on other clusters of
/// this session -- and the veto runs against those definite holders plus whatever this pass
/// awards, over the whole set as the award proceeds. The strict pass is untouched and stays
/// byte-identical; this function is additive, and vacuous when the pool is empty (nobody
/// strictly identified in-session -> no guesses, which is the normal state of a short
/// recording).
///
/// # The rule, per fragment
///
/// A fragment is a cluster under [`TENTATIVE_FLOOR_SECONDS`] that the strict pass did not
/// award. For each, in ascending cluster id:
///
/// - Score it with [`rank_enrolled`] -- the same unthresholded per-person nearest-reference
///   ranking, the same `comparable_cosine`, the same tie-break -- and restrict its entries to
///   the in-session pool. Do not re-implement the cosine arithmetic and do not re-rank:
///   `rank_enrolled` already orders descending similarity, ties by ascending name, which is
///   the walk order this pass uses.
/// - Walk that list and award the fragment the first pooled entry that clears
///   [`TENTATIVE_DISTANCE`] (strictly below, the `best_match` spelling) **and** is not
///   heard-at-once with any holder of that name -- definite or already-tentative. When a
///   name vetoes the fragment, the fragment may still take its *next* pooled name if it
///   clears the cut: that is the next entry of the ranked list, not a runner-up fallback on
///   the same person's second reference, which is the prohibited mechanism the strict pass
///   documents refusing.
/// - One name per fragment, and an awarded fragment becomes a tentative holder of its name
///   for the rest of the walk.
///
/// The veto is whole-set for the same non-transitivity reason the strict pass gives: it is
/// checked against every holder already holding the name as the walk proceeds, never against
/// the winner alone, and the fixed walk order (ascending fragment id, descending similarity
/// then ascending name within a fragment) is what makes a rerun move no fragment between
/// names with nothing having changed.
///
/// The result reuses [`Identification`] rather than a new type: the tier's meaning lives in
/// WHICH map a row came from ([`Naming`](crate::attribution::Naming)'s `identified` versus
/// its `tentative`), and a reader asking whether a tentative row can carry over into a
/// definite one is answered by the types staying separate.
pub fn tentative_identifications(
    clusters: &[SpeakerCluster],
    enrolled: &EnrolledSpeakers,
    identified: &BTreeMap<u32, Identification>,
) -> BTreeMap<u32, Identification> {
    // The pool: the finished definite map's image. Empty means nobody was confirmed in this
    // session, and there is nothing to guess from -- return before doing any scoring at all.
    let pool: BTreeSet<&str> = identified.values().map(|who| who.name.as_str()).collect();
    if pool.is_empty() {
        return BTreeMap::new();
    }

    // Definite holders per name: the veto below runs against these plus this pass's own
    // awards, and a fragment the strict pass already named is a holder, not a candidate.
    let mut definite: BTreeMap<&str, Vec<u32>> = BTreeMap::new();
    for (id, who) in identified {
        definite.entry(&who.name).or_default().push(*id);
    }
    let by_id: BTreeMap<u32, &SpeakerCluster> = clusters
        .iter()
        .map(|cluster| (cluster.id, cluster))
        .collect();

    // Fragments only, ascending id: the walk order is the determinism contract, and cluster
    // ids rank by talk time rather than arrival, so the sort is explicit.
    let mut fragments: Vec<&SpeakerCluster> = clusters
        .iter()
        .filter(|cluster| {
            cluster.speech_seconds < TENTATIVE_FLOOR_SECONDS
                && !identified.contains_key(&cluster.id)
        })
        .collect();
    fragments.sort_by_key(|cluster| cluster.id);

    let mut tentative: BTreeMap<u32, Identification> = BTreeMap::new();
    let mut tentative_holders: BTreeMap<String, Vec<u32>> = BTreeMap::new();
    for fragment in fragments {
        // Descending similarity overall, so the first entry that fails the cut ends the walk
        // for this fragment: nothing further down can clear it either.
        for entry in rank_enrolled(&fragment.embedding, enrolled) {
            if 1.0 - entry.similarity >= TENTATIVE_DISTANCE {
                break;
            }
            if !pool.contains(entry.name.as_str()) {
                continue;
            }
            // The whole-set veto: heard-at-once with ANY holder of this name, definite or
            // already-tentative. A vetoed name loses this fragment only -- the next pooled
            // name still competes.
            let blocked = [
                definite.get(entry.name.as_str()).map(Vec::as_slice),
                tentative_holders.get(&entry.name).map(Vec::as_slice),
            ]
            .into_iter()
            .flatten()
            .flatten()
            .any(|&held| {
                by_id
                    .get(&held)
                    .is_some_and(|holder| heard_at_once(fragment, holder))
            });
            if blocked {
                continue;
            }
            tentative.insert(
                fragment.id,
                Identification {
                    name: entry.name.clone(),
                    similarity: entry.similarity,
                },
            );
            tentative_holders
                .entry(entry.name)
                .or_default()
                .push(fragment.id);
            break;
        }
    }
    tentative
}

/// Whether segmentation proved these two clusters are different people.
///
/// Reads **both** directions of a relation [`SpeakerCluster::heard_at_once_with`] documents as
/// symmetric, so that a file which somehow lost one side of a pair still excludes -- one side
/// asserting it is enough -- rather than excluding or not depending on which cluster this
/// happened to be called with first. Cheaper than validating the invariant and it fails safe.
///
/// Public because `meethook-enroll` reports the same relation when the one-remote-speaker
/// assertion overrides it: one reading of the on-disk contract rather than two, since two
/// readings could disagree about which pairs count as an overlap.
/// [`attributions`](crate::attributions) applies the exclusion to hand-given names and shares this rather than restating it, since
/// two readings of one relation could disagree.
pub fn heard_at_once(a: &SpeakerCluster, b: &SpeakerCluster) -> bool {
    a.heard_at_once_with.contains(&b.id) || b.heard_at_once_with.contains(&a.id)
}

/// The closest enrolled speaker to one voice, if any is close enough.
///
/// Both sides are unit vectors by contract -- mean-pooled, then L2-normalized, on the
/// clustering side and on the enrollment side alike -- so the dot product *is* the cosine and
/// nothing here has to renormalize. The contract is documented on both embedding fields; if
/// either producer ever breaks it the symptom is that identification silently stops firing,
/// which is why it is spelled out in three places rather than one.
fn best_match(embedding: &[f32], enrolled: &EnrolledSpeakers) -> Option<Identification> {
    let mut best: Option<(&str, f32)> = None;
    for speaker in &enrolled.speakers {
        let Some(similarity) = comparable_cosine(&speaker.embedding, embedding) else {
            continue;
        };

        // Strictly greater, then the lexicographically first name, so an exact tie between
        // two references resolves the same way on every rerun rather than by iteration order.
        let better = match best {
            None => true,
            Some((held_name, held)) => {
                similarity > held || (similarity == held && speaker.name.as_str() < held_name)
            }
        };
        if better {
            best = Some((speaker.name.as_str(), similarity));
        }
    }

    best.filter(|&(_, similarity)| 1.0 - similarity < IDENTIFY_DISTANCE)
        .map(|(name, similarity)| Identification {
            name: name.to_string(),
            similarity,
        })
}

/// The cosine between two voices, or `None` when they are not two measurements of one thing.
///
/// The one place the arithmetic and the comparability rule live, shared by [`best_match`] and
/// [`rank_enrolled`] so that a threshold decision and a ranked list cannot come to disagree
/// about which pairs are even comparable.
///
/// Both sides are unit vectors by contract, so the dot product *is* the cosine and nothing
/// here renormalizes -- see [`best_match`] for where that contract is written down.
///
/// Two refusals, both silent rather than fatal, so that one bad row cannot stop the rest of a
/// database from being compared:
///
/// - **Different lengths** came from different embedding models, so the two vectors describe
///   different spaces. `zip` would happily truncate and return a plausible-looking cosine
///   between unrelated spaces; the honest answer is no opinion.
/// - **Either side empty** has no direction to compare. The arithmetic would return 0.0, which
///   reads as "orthogonal" -- a measurement -- when nothing was measured. `best_match` can
///   afford to let its threshold reject that one line later; a ranking has no threshold to hide
///   behind, so the refusal is made here and both callers get it. This is the same rule
///   [`crate::stored_reference_distances`] already applies to a pair of references.
fn comparable_cosine(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    Some(a.iter().zip(b).map(|(x, y)| x * y).sum())
}

/// Every enrolled person one voice could be, nearest first.
///
/// What [`identify_clusters`] throws away. That function computes an argmax and then discards
/// everything that failed [`IDENTIFY_DISTANCE`], so a caller that wants to *ask* who a voice is
/// has no way to offer names -- and offering names is the difference between a prompt somebody
/// can answer and one that demands a name be typed from memory.
///
/// This is also where the "ambiguous" middle tier that [`identify_clusters`] deliberately
/// refuses to have belongs. Its own reason says so: a three-way outcome needs a UI to resolve
/// it, and this is the shape that UI reads.
///
/// # Unthresholded, on purpose
///
/// Nothing here compares against [`IDENTIFY_DISTANCE`]. That cut exists to keep the
/// *automatic* pass conservative -- a false match puts one person's words under another's name
/// in a transcript nobody will re-read. A human reading a ranked list is precisely the case it
/// was biased against serving: a cut here would hide the near-miss the user is being asked to
/// adjudicate, which is the only entry the question was worth asking about.
///
/// A caller that wants the automatic answer should call [`identify_clusters`] and get the
/// decision, not re-derive one by comparing the first entry here against the constant.
///
/// # One entry per person, not per stored reference
///
/// A person is every row bearing their name (see [`EnrolledSpeakers`]), and this database runs
/// to tens of references over a couple of dozen people, so a list of rows would name somebody
/// with 12 recordings 12 times.
///
/// - One entry per distinct name, scored at the **nearest** of that person's comparable
///   references. That is the rule [`EnrolledSpeakers`] states for itself, not a new policy.
/// - `references` counts **every** row under the name, from
///   [`EnrolledSpeakers::references`] -- including rows no similarity could be taken from.
/// - So somebody holding three rows of which one is stale appears once, with
///   `references: 3` and a similarity from the two comparable ones. Somebody **all** of whose
///   rows are incomparable does not appear at all: there is no similarity to order them by, and
///   inventing one would be the fabricated comparison `comparable_cosine` refuses to make.
///
/// # Order
///
/// Descending similarity, ties by ascending name -- the same tie-break `best_match` makes, so
/// the head of this list is the person [`identify_clusters`] would award. `total_cmp` is total
/// even over NaN, so no input can panic the sort. Names are distinct after grouping, so this is
/// a total order and two runs over one database produce one order regardless of the order the
/// rows sit in the file.
///
/// An empty database ranks nobody, which is the normal state of every install before anyone has
/// been enrolled, and is not an error.
pub fn rank_enrolled(embedding: &[f32], enrolled: &EnrolledSpeakers) -> Vec<Resemblance> {
    // Best comparable similarity per name. Grouping by name is this module's knowledge; how
    // many rows a name holds is `speakers.rs`'s, and is asked for below rather than counted
    // here, because a local count would report the comparable rows and not the person's.
    let mut nearest: BTreeMap<&str, f32> = BTreeMap::new();
    for speaker in &enrolled.speakers {
        let Some(similarity) = comparable_cosine(&speaker.embedding, embedding) else {
            continue;
        };
        nearest
            .entry(&speaker.name)
            .and_modify(|held| {
                if similarity > *held {
                    *held = similarity;
                }
            })
            .or_insert(similarity);
    }

    let mut ranked: Vec<Resemblance> = nearest
        .into_iter()
        .map(|(name, similarity)| Resemblance {
            name: name.to_string(),
            similarity,
            references: enrolled.references(name),
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.similarity
            .total_cmp(&a.similarity)
            .then(a.name.cmp(&b.name))
    });
    ranked
}

/// One measured distance between a sub-floor fragment and another cluster of the same
/// session, carrying the only label the on-disk contract can supply for it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TentativePair {
    /// The fragment side: always under [`TENTATIVE_FLOOR_SECONDS`].
    pub fragment: u32,
    /// The partner side: any other cluster of the session, above or below the floor alike --
    /// the strict pass has no floor gate, so a definite holder may be a fragment too, and a
    /// population that excluded them would measure a rule the production veto does not use.
    pub partner: u32,

    /// Cosine distance between the two centroids: `1.0 - dot` over unit vectors, 0 for the
    /// same direction and 1 orthogonal. The same arithmetic `comparable_cosine` decides
    /// with, which is what stops this from being able to disagree with it.
    pub distance: f32,

    /// Segmentation heard the two talking over each other: proof of two people, and the only
    /// direction the on-disk contract labels. See [`tentative_pairs`] for what that buys and
    /// what it cannot.
    pub different_speaker: bool,
}

/// The labelled population [`TENTATIVE_DISTANCE`] is priced from, and nothing else.
///
/// Every sub-floor fragment against every other cluster of the session, in ascending fragment
/// id then ascending partner id, so two runs over one file are diffable -- the only way "the
/// output did not change" gets checked at all.
///
/// # What the labels are, and the gap that has to travel with them
///
/// The on-disk contract carries one free supervision signal per pair: whether segmentation
/// heard the two clusters talking over each other. That labels **negatives** -- provably two
/// people, so a guess of one's name for the other would be wrong -- and it is the same
/// evidence the production veto acts on, so those pairs are refused there whatever the cut
/// is. What they price is how close two provably different people's centroids come at this
/// granularity, which bounds trust in a cut on the *unguarded* pairs, where no constraint
/// protects anybody. That caveat is stated here because a threshold read off guarded
/// negatives alone would otherwise look like it had been measured against the unguarded
/// pairs it actually governs.
///
/// **Positives are not derivable from the file.** A fragment and the cluster of the person
/// it really is need per-cluster identity, and neither `speaker_clusters.json` nor an
/// unnamed transcript holds it. Labelling them is ear confirmation against representative
/// clips; until that sitting exists, the constant's documentation says so rather than
/// presenting the unlabelled nearest-partner distances as a same-speaker population. They are
/// carried anyway (`different_speaker: false`) so the report shows their shape beside the
/// labelled side instead of hiding it.
///
/// This function measures; it picks no threshold and holds no constant beyond the floor that
/// selects its fragments. Scoring belongs to [`crate::score_trials`], presentation to
/// `examples/speaker-trials`. The arithmetic is in the crate rather than the example for the
/// reason [`crate::adoption_populations`] gives: a diagnostic whose conventions nobody can
/// test is a number to believe rather than evidence.
///
/// Pairs whose embeddings differ in length, or are empty, take part in no distance: a cosine
/// between unrelated spaces or across nothing is the fabricated comparison
/// `comparable_cosine` refuses to make, and skipping is silent the way it is there, so one
/// bad row cannot stop the rest of a session from being measured.
pub fn tentative_pairs(clusters: &[SpeakerCluster]) -> Vec<TentativePair> {
    let mut fragments: Vec<&SpeakerCluster> = clusters
        .iter()
        .filter(|cluster| cluster.speech_seconds < TENTATIVE_FLOOR_SECONDS)
        .collect();
    fragments.sort_by_key(|cluster| cluster.id);

    let mut partners: Vec<&SpeakerCluster> = clusters.iter().collect();
    partners.sort_by_key(|cluster| cluster.id);

    let mut pairs = Vec::new();
    for fragment in &fragments {
        for partner in &partners {
            if partner.id == fragment.id {
                continue;
            }
            let Some(similarity) = comparable_cosine(&fragment.embedding, &partner.embedding)
            else {
                continue;
            };
            pairs.push(TentativePair {
                fragment: fragment.id,
                partner: partner.id,
                distance: 1.0 - similarity,
                different_speaker: heard_at_once(fragment, partner),
            });
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use meethook_session::{EnrolledSpeaker, RepresentativeSegment};

    use super::*;

    /// A unit vector `theta` degrees off the x axis, which is how every fixture below states
    /// a distance: cosine distance is `1 - cos(theta)`, so the angle is the readable form.
    fn voice(degrees: f32) -> Vec<f32> {
        let radians = degrees.to_radians();
        vec![radians.cos(), radians.sin()]
    }

    fn cluster(id: u32, embedding: Vec<f32>) -> SpeakerCluster {
        SpeakerCluster {
            id,
            embedding,
            speech_seconds: 10.0,
            // Distinct per cluster, and never zero, so nothing here can pass by accident on
            // a tie between two voices that supposedly began at the same instant.
            first_spoke_seconds: 5.0 + id as f64,
            heard_at_once_with: Vec::new(),
            representatives: vec![RepresentativeSegment {
                start: 0.0,
                end: 2.0,
            }],
        }
    }

    /// A cluster segmentation heard talking over the given ones. Spelled as a wrapper so that
    /// every test which does not turn on the relation reads exactly as it did before it
    /// existed, and the ones that do state it in the call.
    fn excluding(id: u32, embedding: Vec<f32>, heard_at_once_with: Vec<u32>) -> SpeakerCluster {
        SpeakerCluster {
            heard_at_once_with,
            ..cluster(id, embedding)
        }
    }

    fn enrolled(speakers: &[(&str, Vec<f32>)]) -> EnrolledSpeakers {
        EnrolledSpeakers::new(
            speakers
                .iter()
                .map(|(name, embedding)| EnrolledSpeaker {
                    name: name.to_string(),
                    embedding: embedding.clone(),
                    clip_seconds: None,
                })
                .collect(),
        )
    }

    /// Acceptance criterion #1 and #3, at the level that decides them: a reference close to
    /// the cluster's voice names it, and the similarity it was decided on comes back out.
    ///
    /// 25 degrees is a cosine distance of 0.094 -- comfortably inside the threshold, and about
    /// what two recordings of one person actually look like.
    #[test]
    fn a_reference_inside_the_threshold_names_the_cluster() {
        let identified = identify_clusters(
            &[cluster(0, voice(0.0))],
            &enrolled(&[("Alice", voice(25.0))]),
        );

        assert_eq!(identified[&0].name, "Alice");
        let similarity = identified[&0].similarity;
        assert!(
            (similarity - 25.0f32.to_radians().cos()).abs() < 1e-6,
            "{similarity}"
        );
    }

    /// Acceptance criterion #4: too far away is not a weak match, it is no match, and the
    /// caller must not be able to mistake one for the other.
    #[test]
    fn a_reference_outside_the_threshold_leaves_the_cluster_unidentified() {
        // 80 degrees is a cosine distance of 0.83.
        let identified = identify_clusters(
            &[cluster(0, voice(0.0))],
            &enrolled(&[("Alice", voice(80.0))]),
        );

        assert!(identified.is_empty(), "{identified:?}");
    }

    /// The threshold is a threshold: the same cluster and the same person land on opposite
    /// sides of it, so this fails if the comparison is ever inverted or the constant moves
    /// without the test moving with it.
    #[test]
    fn the_decision_is_a_single_cut_at_the_identify_distance() {
        let just_inside = 1.0 - IDENTIFY_DISTANCE + 0.01;
        let just_outside = 1.0 - IDENTIFY_DISTANCE - 0.01;

        for (similarity, expected) in [(just_inside, true), (just_outside, false)] {
            let identified = identify_clusters(
                &[cluster(0, vec![1.0, 0.0])],
                &enrolled(&[(
                    "Alice",
                    vec![similarity, (1.0f32 - similarity * similarity).sqrt()],
                )]),
            );
            assert_eq!(
                identified.contains_key(&0),
                expected,
                "similarity {similarity} should {} have matched",
                if expected { "" } else { "not" }
            );
        }
    }

    /// Acceptance criterion #6 at its decision point, and the state of every install before
    /// anybody has run `enroll`: no references means no names, and no error.
    #[test]
    fn an_empty_database_identifies_nobody_without_failing() {
        let identified = identify_clusters(
            &[cluster(0, voice(0.0)), cluster(1, voice(90.0))],
            &enrolled(&[]),
        );

        assert!(identified.is_empty());
    }

    /// Argmax, not first-past-the-post. Two people can both be inside the threshold of one
    /// voice -- relatives, or a bad microphone -- and the closest one has to win, not whoever
    /// happens to be earlier in the file.
    #[test]
    fn the_closest_reference_wins_rather_than_the_first_one_that_clears() {
        let identified = identify_clusters(
            &[cluster(0, voice(0.0))],
            &enrolled(&[
                ("Alice", voice(40.0)),
                ("Bob", voice(5.0)),
                ("Carol", voice(30.0)),
            ]),
        );

        assert_eq!(identified[&0].name, "Bob");
    }

    /// An exact tie has to resolve the same way every run, or a `--force` re-transcribe could
    /// swap two names in a transcript with nothing in the session having changed.
    #[test]
    fn an_exact_tie_goes_to_the_lexicographically_first_name() {
        let one_voice = voice(10.0);
        let identified = identify_clusters(
            &[cluster(0, voice(0.0))],
            &enrolled(&[("Zoe", one_voice.clone()), ("Andrew", one_voice)]),
        );

        assert_eq!(identified[&0].name, "Andrew");
    }

    /// A reference from a different embedding model is not a bad match, it is not a
    /// comparison at all. Truncating to the shorter side would return a plausible cosine from
    /// two unrelated spaces -- and would do it silently.
    #[test]
    fn a_reference_of_the_wrong_dimension_is_skipped_rather_than_compared() {
        let stale = enrolled(&[("Stale", vec![1.0, 0.0, 0.0, 0.0])]);

        let identified = identify_clusters(&[cluster(0, voice(0.0))], &stale);

        assert!(identified.is_empty(), "{identified:?}");
    }

    /// ...and skipping it must not cost the entries that are still comparable.
    #[test]
    fn a_stale_reference_does_not_stop_a_good_one_in_the_same_file_from_matching() {
        let mixed = enrolled(&[("Stale", vec![1.0, 0.0, 0.0, 0.0]), ("Alice", voice(10.0))]);

        let identified = identify_clusters(&[cluster(0, voice(0.0))], &mixed);

        assert_eq!(identified[&0].name, "Alice");
    }

    /// Clusters are keyed by their own id, not by position, because that is what `merge` looks
    /// them up by -- and cluster ids rank by talk time, so they routinely arrive out of order.
    #[test]
    fn identifications_are_keyed_by_cluster_id() {
        let identified = identify_clusters(
            &[
                cluster(7, voice(0.0)),
                cluster(2, voice(90.0)),
                cluster(4, voice(88.0)),
            ],
            &enrolled(&[("Alice", voice(0.0)), ("Bob", voice(90.0))]),
        );

        assert_eq!(identified[&7].name, "Alice");
        assert_eq!(identified[&2].name, "Bob");
        assert_eq!(identified[&4].name, "Bob");
    }

    /// One person the clusterer split in two gets their name on both halves. The alternative
    /// -- awarding the name to the better half and leaving the other Unknown -- would invent a
    /// second participant who was never in the room.
    ///
    /// The regression the exclusion rule below must not break: nothing here says these two are
    /// different people, so nothing may stop them sharing a name.
    #[test]
    fn two_clusters_matching_one_person_both_get_that_name() {
        let identified = identify_clusters(
            &[cluster(0, voice(0.0)), cluster(1, voice(20.0))],
            &enrolled(&[("Alice", voice(10.0))]),
        );

        assert_eq!(identified[&0].name, "Alice");
        assert_eq!(identified[&1].name, "Alice");
    }

    /// The defect this rule exists to close: two voices segmentation heard talking over each
    /// other are two people, so one enrolled name cannot cover both however close their
    /// centroids are. The nearer keeps it; the other is unidentified, not merely re-ranked.
    ///
    /// Cluster 1 is the nearer, so a rule that resolved by cluster id instead of by distance
    /// would pass this by luck.
    #[test]
    fn two_clusters_heard_at_once_cannot_both_take_one_name() {
        let identified = identify_clusters(
            &[
                excluding(0, voice(25.0), vec![1]),
                excluding(1, voice(5.0), vec![0]),
            ],
            &enrolled(&[("Alice", voice(10.0))]),
        );

        assert_eq!(identified[&1].name, "Alice");
        assert!(!identified.contains_key(&0), "{identified:?}");
    }

    /// The case that separates the rule from the obvious one. Three clusters claim Alice and
    /// the only exclusion is between the two *losers*: "drop whoever is excluded from the
    /// winner" excludes nobody here and files all three, leaving clusters 1 and 2 -- provably
    /// two people -- under one name. Vetoing against everything already awarded drops 2.
    ///
    /// Exclusion is not transitive, which is exactly why the winner is not a sufficient
    /// reference point.
    #[test]
    fn a_contender_is_vetoed_by_any_cluster_already_awarded_not_just_the_nearest() {
        let identified = identify_clusters(
            &[
                cluster(0, voice(5.0)),
                excluding(1, voice(15.0), vec![2]),
                excluding(2, voice(25.0), vec![1]),
            ],
            &enrolled(&[("Alice", voice(0.0))]),
        );

        assert_eq!(identified[&0].name, "Alice");
        assert_eq!(identified[&1].name, "Alice");
        assert!(!identified.contains_key(&2), "{identified:?}");
    }

    /// Losing a contested name is not a demotion to the runner-up. Cluster 1's second-nearest
    /// reference is Bob, comfortably inside the threshold, and it must still come back
    /// unidentified: the runner-up is by construction the reference that already lost the
    /// argmax, and awarding it is the misattribution this whole rule is about, one name over.
    #[test]
    fn a_vetoed_cluster_does_not_fall_back_to_its_second_nearest_reference() {
        let identified = identify_clusters(
            &[
                excluding(0, voice(0.0), vec![1]),
                excluding(1, voice(8.0), vec![0]),
            ],
            &enrolled(&[("Alice", voice(2.0)), ("Bob", voice(20.0))]),
        );

        assert_eq!(identified[&0].name, "Alice");
        assert!(!identified.contains_key(&1), "{identified:?}");
    }

    /// Two excluded clusters exactly equidistant from one reference still have to resolve the
    /// same way on every run, and independently of the order the slice happens to be in --
    /// otherwise a `--force` re-transcribe could move a name between two speakers with nothing
    /// in the session having changed. Ascending cluster id breaks it.
    #[test]
    fn an_exact_tie_between_two_excluded_clusters_goes_to_the_lower_cluster_id() {
        for order in [[0u32, 1], [1, 0]] {
            let by_id = |id: u32| match id {
                0 => excluding(0, voice(10.0), vec![1]),
                _ => excluding(1, voice(-10.0), vec![0]),
            };
            let identified = identify_clusters(
                &[by_id(order[0]), by_id(order[1])],
                &enrolled(&[("Alice", voice(0.0))]),
            );

            assert_eq!(identified[&0].name, "Alice", "input order {order:?}");
            assert!(!identified.contains_key(&1), "input order {order:?}");
        }
    }

    /// The relation is documented as symmetric, and a file that lost one side of a pair must
    /// still exclude rather than exclude or not depending on which cluster was examined first.
    /// Here only cluster 0 names cluster 1, and cluster 1 is the nearer, so the veto can only
    /// fire if the losing side's own list is read.
    #[test]
    fn a_one_sided_exclusion_still_excludes() {
        let identified = identify_clusters(
            &[excluding(0, voice(25.0), vec![1]), cluster(1, voice(5.0))],
            &enrolled(&[("Alice", voice(10.0))]),
        );

        assert_eq!(identified[&1].name, "Alice");
        assert!(!identified.contains_key(&0), "{identified:?}");
    }

    /// Both production callers hand over the whole cluster list, but the contract is total
    /// rather than resting on that: an id the slice does not contain excludes nothing, and in
    /// particular does not silently suppress the cluster naming it.
    #[test]
    fn an_exclusion_naming_a_cluster_outside_the_slice_is_ignored() {
        let identified = identify_clusters(
            &[excluding(0, voice(0.0), vec![99])],
            &enrolled(&[("Alice", voice(10.0))]),
        );

        assert_eq!(identified[&0].name, "Alice");
    }

    /// A cluster with no embedding at all cannot be compared to anything, and must not match
    /// an equally empty reference by vacuous agreement -- an empty dot product is 0.0, which
    /// is a distance of 1.0 and rejected on its own merits.
    #[test]
    fn an_empty_embedding_matches_nobody() {
        let identified = identify_clusters(
            &[cluster(0, Vec::new())],
            &enrolled(&[("Alice", Vec::new())]),
        );

        assert!(identified.is_empty(), "{identified:?}");
    }

    /// The names in a ranking, which is what a prompt would print and what every ordering
    /// assertion below is actually about.
    fn names(ranked: &[Resemblance]) -> Vec<&str> {
        ranked.iter().map(|r| r.name.as_str()).collect()
    }

    /// Acceptance criterion #1: every comparable person comes back, nearest first. The rows are
    /// deliberately in the reverse of the answer, so this fails on a function that returns the
    /// file's order.
    #[test]
    fn every_comparable_speaker_is_ranked_by_descending_similarity() {
        let ranked = rank_enrolled(
            &voice(0.0),
            &enrolled(&[
                ("Carol", voice(80.0)),
                ("Alice", voice(40.0)),
                ("Bob", voice(5.0)),
            ]),
        );

        assert_eq!(names(&ranked), ["Bob", "Alice", "Carol"]);
        assert!(
            ranked
                .windows(2)
                .all(|pair| pair[0].similarity >= pair[1].similarity),
            "{ranked:?}"
        );
    }

    /// Acceptance criterion #2, and the whole reason this exists beside [`identify_clusters`]:
    /// the person a voice failed to match is the entry the user most needs to see.
    ///
    /// 80 degrees is a cosine distance of 0.83, twice the cut. Asserted against
    /// [`IDENTIFY_DISTANCE`] by name rather than against 0.40, so this still means "outside the
    /// cut" if the constant moves.
    #[test]
    fn the_ranking_is_not_cut_at_the_identify_distance() {
        let database = enrolled(&[("Alice", voice(80.0))]);

        let ranked = rank_enrolled(&voice(0.0), &database);

        assert_eq!(names(&ranked), ["Alice"]);
        assert!(1.0 - ranked[0].similarity > IDENTIFY_DISTANCE, "{ranked:?}");
        // The same voice against the same database, decided rather than ranked, names nobody.
        assert!(
            identify_clusters(&[cluster(0, voice(0.0))], &database).is_empty(),
            "the fixture has to be outside the cut for this test to mean anything"
        );
    }

    /// Acceptance criterion #3: all three numbers a prompt prints, and the similarity is the
    /// arithmetic the fixture's angle implies rather than merely some float.
    #[test]
    fn an_entry_carries_the_name_the_similarity_and_the_reference_count() {
        let ranked = rank_enrolled(
            &voice(0.0),
            &enrolled(&[
                ("Alice", voice(25.0)),
                ("Alice", voice(60.0)),
                ("Bob", voice(70.0)),
            ]),
        );

        assert_eq!(ranked[0].name, "Alice");
        assert!(
            (ranked[0].similarity - 25.0f32.to_radians().cos()).abs() < 1e-6,
            "{ranked:?}"
        );
        assert_eq!(ranked[0].references, 2);
        assert_eq!(ranked[1].references, 1);
    }

    /// Acceptance criterion #8: entries are people, not rows. A list of rows would name
    /// somebody with three recordings three times, and this database holds tens of rows over a
    /// couple of dozen people.
    #[test]
    fn a_person_with_several_references_appears_once_scored_at_their_nearest() {
        let ranked = rank_enrolled(
            &voice(0.0),
            &enrolled(&[
                ("Alice", voice(70.0)),
                ("Alice", voice(20.0)),
                ("Alice", voice(45.0)),
            ]),
        );

        assert_eq!(names(&ranked), ["Alice"]);
        assert_eq!(ranked[0].references, 3);
        assert!(
            (ranked[0].similarity - 20.0f32.to_radians().cos()).abs() < 1e-6,
            "{ranked:?}"
        );
    }

    /// Acceptance criterion #4, and #8's second half. A four-dimensional row against a
    /// two-dimensional voice is not a distant match, it is not a comparison: truncating
    /// `Stale`'s leading 1.0 would have scored it 1.0 and put it top of the list.
    ///
    /// So `Stale` is absent entirely, while `Mixed` -- one stale row and one good one --
    /// appears, scored on the good one and reporting **both** rows, because `references` is the
    /// person's multiplicity and not a count of what happened to be comparable.
    #[test]
    fn a_reference_of_the_wrong_dimension_is_excluded_from_the_ranking() {
        let stale = vec![1.0, 0.0, 0.0, 0.0];

        let ranked = rank_enrolled(
            &voice(0.0),
            &enrolled(&[
                ("Stale", stale.clone()),
                ("Mixed", stale),
                ("Mixed", voice(30.0)),
            ]),
        );

        assert_eq!(names(&ranked), ["Mixed"]);
        assert_eq!(ranked[0].references, 2);
        assert!(
            (ranked[0].similarity - 30.0f32.to_radians().cos()).abs() < 1e-6,
            "{ranked:?}"
        );
    }

    /// Acceptance criterion #5. Two people exactly equidistant from one voice have to be listed
    /// in the same order on every run and under either file order, or a prompt would offer two
    /// names in an order that moved with nothing having changed -- and the first entry is the
    /// one a UI will default to.
    #[test]
    fn an_exact_tie_in_the_ranking_goes_to_the_lexicographically_first_name() {
        for order in [["Zoe", "Andrew"], ["Andrew", "Zoe"]] {
            let one_voice = voice(10.0);
            let ranked = rank_enrolled(
                &voice(0.0),
                &enrolled(&[(order[0], one_voice.clone()), (order[1], one_voice)]),
            );

            assert_eq!(names(&ranked), ["Andrew", "Zoe"], "file order {order:?}");
        }
    }

    /// Acceptance criterion #6, and the state of every install before anybody has run `enroll`:
    /// nobody to offer is an empty list, not an error and not a failure to prompt.
    #[test]
    fn an_empty_database_ranks_nobody_without_failing() {
        assert!(rank_enrolled(&voice(0.0), &enrolled(&[])).is_empty());
    }

    /// A voice with no embedding cannot be compared to anything, and an empty reference is not
    /// a comparison either. Both would arithmetically produce 0.0, which reads as "measured,
    /// and orthogonal" -- a claim about a comparison that never happened. [`best_match`] can
    /// let its threshold reject that; a ranking has no threshold, so the refusal is explicit.
    #[test]
    fn an_empty_embedding_on_either_side_is_excluded_rather_than_ranked_at_zero() {
        assert!(
            rank_enrolled(
                &[],
                &enrolled(&[("Alice", voice(0.0)), ("Empty", Vec::new())])
            )
            .is_empty()
        );

        let ranked = rank_enrolled(
            &voice(0.0),
            &enrolled(&[("Empty", Vec::new()), ("Alice", voice(10.0))]),
        );

        assert_eq!(names(&ranked), ["Alice"]);
    }

    /// The two must not drift: the head of the ranking is the person identification awards,
    /// with the same number beside it, or one screen would show a name and a ranking that
    /// disagree.
    ///
    /// Asserted through [`identify_clusters`] because that is the public behaviour which must
    /// not change, and with the winner holding several references -- the case where a per-row
    /// argmax and a per-person argmax could plausibly diverge.
    #[test]
    fn the_first_ranked_entry_is_the_person_identification_awards() {
        let database = enrolled(&[
            ("Bob", voice(35.0)),
            ("Alice", voice(60.0)),
            ("Alice", voice(12.0)),
            ("Carol", voice(85.0)),
        ]);

        let identified = identify_clusters(&[cluster(0, voice(0.0))], &database);
        let ranked = rank_enrolled(&voice(0.0), &database);

        assert_eq!(ranked[0].name, identified[&0].name);
        assert_eq!(ranked[0].similarity, identified[&0].similarity);
        assert_eq!(ranked[0].references, 2);
    }

    // --- the tentative band ---------------------------------------------------------------

    /// A sub-floor cluster, since the band scores fragments only and the plain helper above
    /// hardcodes ten seconds of speech.
    fn fragment(id: u32, embedding: Vec<f32>, seconds: f64) -> SpeakerCluster {
        SpeakerCluster {
            speech_seconds: seconds,
            ..cluster(id, embedding)
        }
    }

    /// The whole point of the band, at its decision point: a fragment the strict pass refused
    /// takes the name when that name was strictly awarded elsewhere in the session, carrying
    /// the similarity it was guessed at.
    #[test]
    fn a_sub_floor_fragment_takes_a_name_strictly_awarded_in_session() {
        // The fragment sits at a centroid distance of 0.41 from Alice's reference: past the
        // strict cut of 0.40, inside the tentative cut of 0.42. Built from the distance
        // rather than an angle so the margin to each cut is exact rather than trigonometric.
        let similarity: f32 = 1.0 - 0.41;
        let fragment_emb = vec![similarity, (1.0 - similarity * similarity).sqrt()];
        let database = enrolled(&[("Alice", voice(0.0))]);
        let clusters = vec![
            cluster(1, voice(0.0)),         // the definite holder, above the floor
            fragment(0, fragment_emb, 4.0), // the candidate, under the floor
        ];

        let identified = identify_clusters(&clusters, &database);
        assert!(
            !identified.contains_key(&0),
            "the fixture must sit outside the strict cut for this test to mean anything"
        );

        let tentative = tentative_identifications(&clusters, &database, &identified);

        assert_eq!(tentative[&0].name, "Alice");
        assert!((tentative[&0].similarity - similarity).abs() < 1e-6);
    }

    /// Acceptance criterion #1, negative half: a name that is enrolled but was never strictly
    /// awarded in this session is not in the pool, and the fragment gets nothing however
    /// close. The tool never guesses somebody who was not confirmed in the meeting.
    #[test]
    fn an_enrolled_name_absent_from_the_session_is_not_guessed() {
        // One cluster only: Alice sits inside the tentative cut -- close enough to tempt a
        // guess -- but no other cluster carries her name, so the pool is empty.
        let similarity: f32 = 1.0 - 0.41;
        let emb = vec![similarity, (1.0 - similarity * similarity).sqrt()];
        let database = enrolled(&[("Alice", voice(0.0))]);
        let clusters = vec![fragment(0, emb, 3.0)];

        let identified = identify_clusters(&clusters, &database);
        assert!(
            identified.is_empty(),
            "outside the strict cut by construction"
        );
        let tentative = tentative_identifications(&clusters, &database, &identified);

        assert!(tentative.is_empty(), "{tentative:?}");
    }

    /// ...and the same fragment earns the name the moment a sibling cluster is strictly
    /// awarded it: the pool is the finished definite map's image, nothing more.
    #[test]
    fn the_same_fragment_is_guessed_once_a_sibling_cluster_is_awarded_the_name() {
        let similarity: f32 = 1.0 - 0.41;
        let emb = vec![similarity, (1.0 - similarity * similarity).sqrt()];
        let database = enrolled(&[("Alice", voice(0.0))]);
        let clusters = vec![cluster(1, voice(0.0)), fragment(0, emb, 3.0)];

        let identified = identify_clusters(&clusters, &database);
        assert_eq!(identified[&1].name, "Alice");

        let tentative = tentative_identifications(&clusters, &database, &identified);

        assert_eq!(tentative[&0].name, "Alice");
    }

    /// Acceptance criterion #2: the floor is a hard gate on eligibility, not on the cut. The
    /// same fragment, one tick of speech either side of the floor, is guessed or is not.
    #[test]
    fn a_fragment_exactly_on_the_floor_is_a_speaker_and_keeps_its_honest_unknown() {
        let similarity: f32 = 1.0 - 0.41;
        let emb = vec![similarity, (1.0 - similarity * similarity).sqrt()];
        let database = enrolled(&[("Alice", voice(0.0))]);
        let make = |seconds: f64| {
            let clusters = vec![cluster(1, voice(0.0)), fragment(0, emb.clone(), seconds)];
            let identified = identify_clusters(&clusters, &database);
            tentative_identifications(&clusters, &database, &identified)
        };

        assert_eq!(make(TENTATIVE_FLOOR_SECONDS - 0.1)[&0].name, "Alice");
        assert!(
            make(TENTATIVE_FLOOR_SECONDS).is_empty(),
            "exactly on the floor is a speaker, the codebase-wide convention"
        );
    }

    /// Acceptance criterion #7's boundary, mirrored from the strict pass by name so the test
    /// moves with the constant: the same fragment and the same in-session name land on
    /// opposite sides of the cut.
    #[test]
    fn the_tentative_decision_is_a_single_cut_at_the_tentative_distance() {
        let holder = voice(0.0);
        let database = enrolled(&[("Alice", holder)]);

        for (similarity, expected) in [
            (1.0 - TENTATIVE_DISTANCE + 0.01, true),
            (1.0 - TENTATIVE_DISTANCE - 0.01, false),
        ] {
            let emb = vec![similarity, (1.0f32 - similarity * similarity).sqrt()];
            let clusters = vec![cluster(1, voice(0.0)), fragment(0, emb, 2.0)];
            let identified = identify_clusters(&clusters, &database);
            let tentative = tentative_identifications(&clusters, &database, &identified);

            assert_eq!(
                tentative.contains_key(&0),
                expected,
                "similarity {similarity} should {} have been guessed",
                if expected { "" } else { "not" }
            );
        }
    }

    /// Acceptance criterion #3, first half: a fragment heard at once with the definite holder
    /// of a name does not take that name -- whatever the distance says -- but may still take
    /// its NEXT pooled name if that one clears the cut. The runner-up on the same person's
    /// second reference is not the mechanism.
    #[test]
    fn a_vetoed_name_loses_only_that_name_to_the_fragment() {
        // Both distances are built exact: the fragment sits 0.4001 from Alice (outside the
        // strict cut, inside the tentative one) and 0.4101 from Bob (inside the tentative
        // one), so Alice ranks first and Bob is the next pooled name that clears.
        let alice = voice(0.0);
        let at_distance = |d: f32| {
            let similarity = 1.0 - d;
            vec![similarity, (1.0 - similarity * similarity).sqrt()]
        };
        let fragment_emb = at_distance(0.4001);
        // Bob's reference is the unit vector at cosine 0.5899 to the fragment, rotated off
        // the x axis so that his holder cannot also match Alice.
        let theta = fragment_emb[1].atan2(fragment_emb[0]);
        let bob = vec![
            (theta + 0.5899f32.acos()).cos(),
            (theta + 0.5899f32.acos()).sin(),
        ];
        let database = enrolled(&[("Alice", alice), ("Bob", bob.clone())]);
        let clusters = vec![
            cluster(1, voice(0.0)),
            cluster(2, bob),
            fragment(0, fragment_emb, 2.0),
        ];
        let identified = identify_clusters(&clusters, &database);
        assert_eq!(identified[&1].name, "Alice");
        assert_eq!(identified[&2].name, "Bob");
        assert!(!identified.contains_key(&0));

        // The same fragment, heard talking over Alice's holder: Alice vetoes it, Bob stands.
        let clusters = vec![
            cluster(1, voice(0.0)),
            cluster(2, database.speakers[1].embedding.clone()),
            SpeakerCluster {
                speech_seconds: 2.0,
                heard_at_once_with: vec![1],
                ..cluster(0, at_distance(0.4001))
            },
        ];
        let identified = identify_clusters(&clusters, &database);
        let tentative = tentative_identifications(&clusters, &database, &identified);

        assert_eq!(tentative[&0].name, "Bob");
    }

    /// Acceptance criterion #3, the non-transitivity case in the C1/C2/C3 shape: three
    /// fragments all claim the pooled name, and the only exclusion is between the two
    /// *losers*. "Drop whoever the winner excludes" vetoes nobody here and files all three,
    /// leaving two provably-different fragments under one marked name. Greedy against every
    /// holder already holding the name drops the third.
    #[test]
    fn a_tentative_contender_is_vetoed_by_any_holder_not_just_the_nearest() {
        let database = enrolled(&[("Alice", voice(0.0))]);
        // All three fragments sit just outside the strict cut and inside the tentative one --
        // close enough to be guessed, far enough that the strict pass leaves them unnamed.
        // The angles are arccos of the similarities, so the distances are 0.405, 0.410 and
        // 0.415: C1 nearest, all three inside the band.
        let clusters = vec![
            cluster(5, voice(0.0)),         // the definite holder
            fragment(0, voice(53.46), 2.0), // C1, nearest
            fragment(1, voice(53.85), 2.0), // C2
            SpeakerCluster {
                speech_seconds: 2.0,
                heard_at_once_with: vec![1],
                ..cluster(2, voice(54.23)) // C3
            },
        ];
        let identified = identify_clusters(&clusters, &database);
        let tentative = tentative_identifications(&clusters, &database, &identified);

        assert_eq!(tentative[&0].name, "Alice");
        assert_eq!(tentative[&1].name, "Alice");
        assert!(
            !tentative.contains_key(&2),
            "C3 is heard at once with C2, which holds the name: {tentative:?}"
        );
    }

    /// A fragment heard at once with a TENTATIVE holder is excluded too -- the veto runs
    /// against the union of definite and tentative holders -- and the outcome must not depend
    /// on the order the slice arrived in, which is what the fixed ascending-id walk exists to
    /// guarantee.
    #[test]
    fn a_fragment_heard_at_once_with_a_tentative_holder_is_vetoed_regardless_of_input_order() {
        let database = enrolled(&[("Alice", voice(0.0))]);
        let make = |lower_first: bool| {
            // Equidistant from Alice at a centroid distance of 0.41 -- just outside the
            // strict cut, inside the tentative one -- mirrored about the x axis.
            let low = SpeakerCluster {
                speech_seconds: 2.0,
                heard_at_once_with: vec![1],
                ..cluster(0, voice(53.85))
            };
            let high = SpeakerCluster {
                speech_seconds: 2.0,
                heard_at_once_with: vec![0],
                ..cluster(1, voice(-53.85))
            };
            let clusters = if lower_first {
                vec![cluster(5, voice(0.0)), low, high]
            } else {
                vec![cluster(5, voice(0.0)), high, low]
            };
            let identified = identify_clusters(&clusters, &database);
            tentative_identifications(&clusters, &database, &identified)
        };

        for lower_first in [true, false] {
            let tentative = make(lower_first);
            // Both fragments are equally close to Alice; the lower id walks first and takes
            // the name, and the other is heard at once with it.
            assert_eq!(
                tentative[&0].name, "Alice",
                "input order lower-first={lower_first}"
            );
            assert!(
                !tentative.contains_key(&1),
                "input order lower-first={lower_first}: {tentative:?}"
            );
        }
    }

    /// An exact similarity tie between two pooled names resolves to the lexicographically
    /// first, the way `rank_enrolled` orders them, so a rerun cannot flip a guess between
    /// two people with nothing having changed.
    #[test]
    fn a_tentative_tie_between_two_names_goes_to_the_lexicographically_first() {
        // Two references mirrored about the x axis and a fragment on it: exactly equidistant
        // from both, at a centroid distance of 0.41 -- outside the strict cut, inside the
        // tentative one.
        let a = 0.59f32;
        let b = (1.0 - a * a).sqrt();
        let database = enrolled(&[("Zoe", vec![a, b]), ("Andrew", vec![a, -b])]);
        let clusters = vec![
            cluster(1, vec![a, b]),
            cluster(2, vec![a, -b]),
            fragment(0, vec![1.0, 0.0], 2.0),
        ];
        let identified = identify_clusters(&clusters, &database);
        let tentative = tentative_identifications(&clusters, &database, &identified);

        assert_eq!(tentative[&0].name, "Andrew");
    }

    /// Nobody strictly identified in-session means an empty pool and an empty map, with no
    /// scoring done at all: short recordings behave exactly as they always have. This is the
    /// regression guard on the band staying vacuous where it must be.
    #[test]
    fn an_empty_pool_leaves_every_fragment_unnamed() {
        let database = enrolled(&[("Alice", voice(90.0)), ("Bob", voice(180.0))]);
        // Every cluster is far enough out that the strict pass awards nobody.
        let clusters = vec![fragment(0, voice(0.0), 3.0), fragment(1, voice(20.0), 2.0)];

        let identified = identify_clusters(&clusters, &database);
        assert!(identified.is_empty());

        let tentative = tentative_identifications(&clusters, &database, &identified);
        assert!(tentative.is_empty(), "{tentative:?}");
    }

    /// A fragment the strict pass ALREADY named is a holder, not a candidate: the band must
    /// not re-guess a voice that has a definite answer, even though it is under the floor.
    #[test]
    fn a_strictly_identified_fragment_is_not_reguessed() {
        let database = enrolled(&[("Alice", voice(0.0))]);
        let clusters = vec![
            cluster(1, voice(0.0)),
            fragment(0, voice(5.0), 2.0), // close enough for the strict pass itself
        ];
        let identified = identify_clusters(&clusters, &database);
        assert_eq!(identified[&0].name, "Alice");

        let tentative = tentative_identifications(&clusters, &database, &identified);

        assert!(
            !tentative.contains_key(&0),
            "a definite answer is not a candidate: {tentative:?}"
        );
    }

    // --- the calibration population -------------------------------------------------------

    /// The fragment side is sub-floor only, the partner side is the rest of the session
    /// above or below the floor alike, and no pair measures a cluster against itself.
    #[test]
    fn the_population_pairs_every_sub_floor_fragment_with_every_other_cluster() {
        let clusters = vec![
            cluster(0, voice(0.0)),        // above the floor: a partner, never a fragment
            fragment(1, voice(30.0), 4.0), // a fragment
            fragment(2, voice(60.0), 4.0), // a fragment, and a partner too
        ];

        let pairs = tentative_pairs(&clusters);

        let keys: Vec<(u32, u32)> = pairs.iter().map(|p| (p.fragment, p.partner)).collect();
        assert_eq!(keys, [(1, 0), (1, 2), (2, 0), (2, 1)]);
        assert!(pairs.iter().all(|p| p.fragment != p.partner));
    }

    /// The label is the only supervision the file carries, read both directions the way
    /// [`heard_at_once`] does: one side asserting the relation is enough.
    #[test]
    fn the_population_labels_a_pair_different_speaker_when_segmentation_heard_them_overlap() {
        let clusters = vec![
            excluding(0, voice(0.0), vec![1]),
            fragment(1, voice(10.0), 2.0),
        ];

        let pairs = tentative_pairs(&clusters);

        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].different_speaker, "{pairs:?}");

        // And the unlabelled side stays unlabelled rather than being guessed at.
        let clusters = vec![cluster(0, voice(0.0)), fragment(1, voice(10.0), 2.0)];
        let pairs = tentative_pairs(&clusters);
        assert!(!pairs[0].different_speaker, "{pairs:?}");
    }

    /// The distances are the same arithmetic the decision uses: `1 - dot` over unit vectors,
    /// so a pair at `theta` degrees reads `1 - cos(theta)` rather than merely some number.
    #[test]
    fn the_population_carries_centroid_distances_the_decision_arithmetic_produces() {
        let clusters = vec![cluster(0, voice(0.0)), fragment(1, voice(30.0), 2.0)];

        let pairs = tentative_pairs(&clusters);

        assert!((pairs[0].distance - (1.0 - 30.0f32.to_radians().cos())).abs() < 1e-6);
    }

    /// A partner from a different embedding model takes part in no distance, and skipping it
    /// costs nothing else: the comparable pairs of the same session are still measured.
    #[test]
    fn an_incomparable_partner_is_skipped_rather_than_measured() {
        let stale = vec![1.0, 0.0, 0.0, 0.0];
        let clusters = vec![
            SpeakerCluster {
                embedding: stale,
                ..cluster(0, voice(0.0))
            },
            fragment(1, voice(10.0), 2.0),
            cluster(2, voice(20.0)),
        ];

        let pairs = tentative_pairs(&clusters);

        let keys: Vec<(u32, u32)> = pairs.iter().map(|p| (p.fragment, p.partner)).collect();
        assert_eq!(keys, [(1, 2)], "the stale partner is no comparison at all");
    }
}
