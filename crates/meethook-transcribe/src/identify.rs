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
/// Cosine distance, so 0 is the same vector and 1 is orthogonal -- the same units as
/// `speakers::MERGE_DISTANCE`, which it deliberately starts equal to and is deliberately not
/// shared with. That constant answers "are these two clips from one meeting the same voice?";
/// this one answers "is this meeting's voice the person we recorded weeks ago?", across a
/// different microphone, a different room and a different call's codec. The two will diverge
/// the moment either is measured on real audio, and one shared value would silently move both
/// when only one was calibrated.
///
/// The two mistakes here are not symmetric either, and the bias is the same. A false match
/// puts one person's words under another person's name in a transcript nobody will re-read.
/// A false rejection is an `Unknown N` the user fixes in `enroll` in ten seconds. So: strictly
/// below the crossover, accepting that some real matches are missed.
///
/// TASK-014 tracks measuring this against real cross-session recordings; it is an argued
/// starting point, not a measured one.
const IDENTIFY_DISTANCE: f32 = 0.45;

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
