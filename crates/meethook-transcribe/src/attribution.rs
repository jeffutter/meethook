//! What a voice is called, derived once for every command that has to answer it.
//!
//! Two commands put labels on the same voices. `transcribe` writes them into a fresh
//! transcript; `enroll` recomputes them against the database as it stands and rewrites the
//! transcript in place. The invariant `enroll` is built against -- a transcript it rewrote is
//! exactly what `transcribe --force` would now produce -- is a claim that those two arrive at
//! the same map, cluster id to label, every time. Two implementations of one rule cannot make
//! that claim; they can only happen to agree, and the symptom of them not agreeing is a name
//! on the wrong person's turns. So the rule lives here, once, and both go through it.
//!
//! The numbering behind an "Unknown N" is deliberately not implemented here either. It lives
//! further down still, in [`meethook_session::unknown_labels`], because the on-disk contract
//! is what both `transcribe` and `enroll` have to recover the same numbers from. This module
//! decides only which of the two things a voice is: a name somebody enrolled, or the number
//! its first appearance earned it.
//!
//! # "Attribution" here means what a cluster is *called*
//!
//! `merge` uses "attributed" in a second, older sense throughout -- which cluster said a
//! recognised segment, which is what [`meethook_session::Turn::cluster`] records and what
//! `merge::speaking_cluster` decides. That question is answered before this one is asked: a
//! turn is attributed to a cluster, and then the cluster carries an [`Attribution`]. Both
//! senses are in the codebase and neither is going away, so they are stated together here
//! rather than left for a reader to collide with.

use std::collections::BTreeMap;

use crate::identify::Identification;

/// What one voice on the speaker track is called, and what that label claims.
///
/// The two variants exist because a label carries two independent facts and a bare string
/// carries neither: whether it names a person, and how sure that naming is. Reading one off
/// the other -- "it has a confidence, so it must be a name" -- is what this type is here to
/// stop, since the two questions come apart as soon as a name can be assigned by hand.
#[derive(Debug, Clone, PartialEq)]
pub enum Attribution {
    /// The "Unknown N" this voice's first appearance earned it. Names nobody.
    ///
    /// There is no confidence for a number to be the confidence *of*: the label makes no
    /// identity claim at all, so there is nothing to be more or less sure about.
    Unknown(String),

    /// Matched to an enrolled reference in `speakers.json`, at this similarity.
    Identified { name: String, similarity: f32 },
}

impl Attribution {
    /// The label as the transcript reads it.
    pub fn label(&self) -> &str {
        match self {
            Attribution::Unknown(label) => label,
            Attribution::Identified { name, .. } => name,
        }
    }

    /// How confident the identity claim in [`label`] is, or `None` when it makes none.
    ///
    /// This is what lands in [`meethook_session::Turn::speaker_id_confidence`].
    ///
    /// [`label`]: Attribution::label
    pub fn confidence(&self) -> Option<f32> {
        match self {
            Attribution::Unknown(_) => None,
            Attribution::Identified { similarity, .. } => Some(*similarity),
        }
    }

    /// Does this label name a person?
    ///
    /// The question `enroll` is actually asking everywhere it decides what to prompt about,
    /// which voices are already resolved, and whether an answer given earlier in the run has
    /// since named this one.
    ///
    /// Deliberately **not** "carries a similarity", and deliberately not spelled
    /// `confidence().is_some()`. Today the two coincide, because the only way to get a name
    /// onto a voice is for identification to have matched one. A name a user assigns to a
    /// session's cluster by hand is a name with no similarity behind it, and any caller
    /// written against the confidence half would re-ask about that voice on every run.
    pub fn is_named(&self) -> bool {
        matches!(self, Attribution::Identified { .. })
    }
}

/// What every voice is called: an enrolled name where identification found one, otherwise the
/// "Unknown N" that voice's first appearance earned it.
///
/// `unknown` is the [`meethook_session::unknown_labels`] map -- one entry per voice, and it is
/// the key set of the result. Callers build it two different ways (`transcribe` from the
/// diarized turns, `enroll` from `speaker_clusters.json`) and get the same map, which is the
/// whole reason it is a parameter rather than something derived in here.
///
/// `identified` is [`crate::identify_clusters`]'s output, which holds an entry only for a
/// cluster that matched somebody. An id in `identified` that `unknown` does not hold is
/// dropped: a voice nothing knows the existence of cannot be labelled, and the caller that
/// built `unknown` is the one that knows which voices this session has.
///
/// Taken by reference on both sides because `enroll` holds its `unknown` for the length of a
/// prompt loop and re-derives the attributions inside it.
pub fn attributions(
    unknown: &BTreeMap<u32, String>,
    identified: &BTreeMap<u32, Identification>,
) -> BTreeMap<u32, Attribution> {
    unknown
        .iter()
        .map(|(&id, label)| {
            let attribution = match identified.get(&id) {
                Some(who) => Attribution::Identified {
                    name: who.name.clone(),
                    similarity: who.similarity,
                },
                None => Attribution::Unknown(label.clone()),
            };
            (id, attribution)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unknown(ids: &[(u32, &str)]) -> BTreeMap<u32, String> {
        ids.iter()
            .map(|(id, label)| (*id, label.to_string()))
            .collect()
    }

    fn identified(entries: &[(u32, &str, f32)]) -> BTreeMap<u32, Identification> {
        entries
            .iter()
            .map(|(id, name, similarity)| {
                (
                    *id,
                    Identification {
                        name: name.to_string(),
                        similarity: *similarity,
                    },
                )
            })
            .collect()
    }

    /// The whole rule, in one case of each: a voice the database recognised reads as that
    /// person and carries the similarity it was matched at, and the voice beside it that
    /// nothing matched keeps the number its first appearance earned it.
    #[test]
    fn a_matched_voice_carries_the_name_and_an_unmatched_one_keeps_its_number() {
        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            &identified(&[(1, "Alice", 0.83)]),
        );

        assert_eq!(map[&0], Attribution::Unknown("Unknown 1".to_string()));
        assert_eq!(map[&0].label(), "Unknown 1");
        assert_eq!(map[&0].confidence(), None);
        assert_eq!(
            map[&1],
            Attribution::Identified {
                name: "Alice".to_string(),
                similarity: 0.83,
            }
        );
        assert_eq!(map[&1].label(), "Alice");
        assert_eq!(map[&1].confidence(), Some(0.83));
    }

    /// `is_named` asserted as the question it asks -- does this label name a person -- rather
    /// than through `confidence()`. A later variant that names somebody without a similarity
    /// must not be able to satisfy this test by accident.
    #[test]
    fn only_a_label_that_names_somebody_is_named() {
        assert!(
            Attribution::Identified {
                name: "Alice".to_string(),
                similarity: 0.83,
            }
            .is_named()
        );
        assert!(!Attribution::Unknown("Unknown 1".to_string()).is_named());
    }

    /// Identification runs over the clusters; the labels run over the voices the caller knows
    /// about. Where those disagree the caller wins, which is the behaviour both of the
    /// derivations this replaced had by construction and neither stated.
    #[test]
    fn an_identification_for_a_voice_the_caller_does_not_know_about_is_dropped() {
        let map = attributions(
            &unknown(&[(0, "Unknown 1")]),
            &identified(&[(0, "Alice", 0.83), (7, "Bob", 0.91)]),
        );

        assert_eq!(map.keys().copied().collect::<Vec<u32>>(), [0]);
    }

    /// Every voice gets a label, whether or not anybody was enrolled -- the normal state of
    /// an install before the first `enroll` run.
    #[test]
    fn every_voice_is_labelled_when_the_database_is_empty() {
        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2"), (2, "Unknown 3")]),
            &BTreeMap::new(),
        );

        assert_eq!(map.keys().copied().collect::<Vec<u32>>(), [0, 1, 2]);
        assert!(map.values().all(|a| !a.is_named()));
    }
}
