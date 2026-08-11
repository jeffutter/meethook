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
/// Two clusters both matching one person -- clustering split a voice in two -- both get that
/// name. That is the honest reading of the evidence, and it renders as one person speaking
/// throughout, which is what happened.
pub fn identify_clusters(
    clusters: &[SpeakerCluster],
    enrolled: &EnrolledSpeakers,
) -> BTreeMap<u32, Identification> {
    let mut identified = BTreeMap::new();
    for cluster in clusters {
        if let Some(best) = best_match(&cluster.embedding, enrolled) {
            identified.insert(cluster.id, best);
        }
    }
    identified
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
            representatives: vec![RepresentativeSegment {
                start: 0.0,
                end: 2.0,
            }],
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
    #[test]
    fn two_clusters_matching_one_person_both_get_that_name() {
        let identified = identify_clusters(
            &[cluster(0, voice(0.0)), cluster(1, voice(20.0))],
            &enrolled(&[("Alice", voice(10.0))]),
        );

        assert_eq!(identified[&0].name, "Alice");
        assert_eq!(identified[&1].name, "Alice");
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
