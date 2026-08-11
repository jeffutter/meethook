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

use std::collections::BTreeMap;

use meethook_session::{EnrolledSpeakers, SpeakerCluster};

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
/// closed, so nothing here rests on the older grouping. Two constants, 0.45 and 0.35, now sit
/// either side of that pair rather than both above it.
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
/// So **0.35**: it removes every misattribution measured on the corpus, at a cost of at most
/// one extra false reject in 86, and it rejects the ear-confirmed different-speaker pair with
/// 0.079 of margin.
///
/// Two caveats belong with those numbers. The corpus holds the channel constant -- one
/// volunteer, room and microphone per speaker -- so its same-speaker distances are a *floor*
/// and its false-reject rate is optimistic. And 36 same-speaker pairs put the false-reject rate
/// at roughly one significant figure: a 95% interval of about 1.5%-18%. TASK-014 still owes the
/// recording sitting that would measure this against meethook's own capture channel.
///
/// The two mistakes here are not symmetric, and the bias follows from that. A false match puts
/// one person's words under another person's name in a transcript nobody will re-read. A false
/// rejection is an `Unknown N` the user fixes in `enroll` in ten seconds. So: below the
/// crossover, accepting that some real matches are missed.
///
/// Public so that the measurement can name the cut it is measuring against: the
/// `cluster-speaker-track` example prints every cluster-to-reference distance alongside this
/// value, and a calibration constant a diagnostic has to hard-code a copy of is a constant
/// that drifts out of agreement with the code it claims to describe. Exported for reading,
/// not for deciding -- [`identify_clusters`] is argmax *then* threshold, so anything that
/// compares against this on its own will call a reference that clears the cut but is not the
/// closest a match, which it is not.
pub const IDENTIFY_DISTANCE: f32 = 0.35;

/// A cluster matched to an enrolled speaker.
#[derive(Debug, Clone, PartialEq)]
pub struct Identification {
    pub name: String,

    /// Cosine similarity between the cluster's voice and that speaker's reference, in
    /// `[-1, 1]` but in practice above `1.0 - IDENTIFY_DISTANCE` or this would not exist.
    pub similarity: f32,
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

/// Whether segmentation proved these two clusters are different people.
///
/// Reads **both** directions of a relation [`SpeakerCluster::heard_at_once_with`] documents as
/// symmetric, so that a file which somehow lost one side of a pair still excludes -- one side
/// asserting it is enough -- rather than excluding or not depending on which cluster this
/// happened to be called with first. Cheaper than validating the invariant and it fails safe.
///
/// Crate-private because the *convention* belongs to the on-disk contract, which is where it
/// is written down; only this crate reads it, so a method over there would be interface for no
/// leverage. [`crate::attribution`] applies the same exclusion to hand-given names and shares
/// this rather than restating it, since two readings of one relation could disagree.
pub(crate) fn heard_at_once(a: &SpeakerCluster, b: &SpeakerCluster) -> bool {
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
        // A reference of a different length came from a different embedding model, so the
        // two vectors describe different spaces and are not comparable. `zip` would happily
        // truncate and return a plausible-looking cosine; the honest answer is no opinion.
        // Skipped rather than fatal, so one stale entry cannot stop the rest from matching.
        if speaker.embedding.len() != embedding.len() {
            continue;
        }
        let similarity: f32 = speaker
            .embedding
            .iter()
            .zip(embedding)
            .map(|(a, b)| a * b)
            .sum();

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
}
