//! Combining two recognised tracks into one chronological, speaker-labelled transcript.
//!
//! This is the part of transcription with no audio and no model in it. Given what Whisper
//! heard on each track and what diarization made of the speaker track, everything left --
//! putting both tracks on one timeline, deciding which voice said which recognised sentence,
//! naming the voices, ordering the result -- is deterministic. Keeping it that way is what
//! makes it testable in microseconds against fixtures, which is where nearly all of the
//! behaviour a reader of `transcript.md` will actually notice is decided.

use std::collections::BTreeMap;

use meethook_session::{SPEAKER_YOU, SourceTrack, Turn, unknown_labels, unknown_speaker};

use crate::asr::AsrSegment;
use crate::diarize::SpeakerTurn;
use crate::identify::Identification;

/// A speaker-track voice's label, and how confident the *identity* claim in it is.
///
/// `None` for an "Unknown N" label, because that label makes no identity claim at all -- there
/// is nothing for a number to be the confidence of.
type Label = (String, Option<f32>);

/// Combines both tracks into one chronological, speaker-labelled transcript.
///
/// `mic` and `speaker` are what the recogniser heard on each track, timed from the start of
/// that track; the two offsets place each track on the session timeline (exactly one of them
/// is non-zero, since the timeline starts at whichever track began first). `diarized` is the
/// speaker track's attributed speech, in that same track's time. `identified` is which of
/// those voices the enrolled database recognised, keyed by cluster id and empty when nobody
/// has been enrolled yet.
///
/// Every mic-track segment becomes a turn labelled [`SPEAKER_YOU`] with no confidence: the
/// speaker there is known by construction rather than inferred, and reporting a number for it
/// would be inventing one. There is exactly one local speaker, so the mic track is never
/// diarized at all.
///
/// Every speaker-track segment becomes a turn too, whatever diarization made of it. Dropping
/// recognised words because the diarizer heard no speech under them would lose real meeting
/// content, which is strictly worse than an imperfect label a reader can see and correct. A
/// turn whose voice was identified carries the similarity the match was decided on; one that
/// was not carries no confidence, for the same reason the mic track does not.
///
/// No overlap or cross-talk handling: two people talking at once produce two turns whose
/// times overlap, ordered by where each started.
pub fn merge(
    mic: Vec<AsrSegment>,
    mic_offset_s: f64,
    speaker: Vec<AsrSegment>,
    speaker_offset_s: f64,
    diarized: &[SpeakerTurn],
    identified: &BTreeMap<u32, Identification>,
) -> Vec<Turn> {
    let labels = label_by_first_appearance(diarized, identified);

    let mut turns: Vec<Turn> = mic
        .into_iter()
        .map(|segment| Turn {
            speaker: SPEAKER_YOU.to_string(),
            start: mic_offset_s + segment.start_s,
            end: mic_offset_s + segment.end_s,
            // Whisper's own segmentation is used exactly as emitted, on both tracks: no
            // re-splitting and no merging of neighbours, so the turns in the transcript are
            // the units the recogniser was actually confident about.
            text: segment.text,
            source_track: SourceTrack::Mic,
            speaker_id_confidence: None,
        })
        .collect();

    turns.extend(speaker.into_iter().map(|segment| {
        // No cluster at all means diarization found nobody on a track Whisper still heard
        // words on. One unnamed speaker is the honest reading of that, and a far better one
        // than an empty label or a dropped sentence.
        let (speaker, confidence) = attribute(&segment, diarized)
            .and_then(|id| labels.get(&id).cloned())
            .unwrap_or_else(|| (unknown_speaker(1), None));
        Turn {
            speaker,
            start: speaker_offset_s + segment.start_s,
            end: speaker_offset_s + segment.end_s,
            text: segment.text,
            source_track: SourceTrack::Speaker,
            speaker_id_confidence: confidence,
        }
    }));

    // Stable, and that is the tie-break: `sort_by` in Rust preserves the order of equal
    // elements, and the mic turns were built first, so a mic turn and a speaker turn that
    // start at the same instant come out mic first. Deterministic rather than arbitrary,
    // which is what makes a `--force` rerun byte-identical.
    turns.sort_by(|a, b| a.start.total_cmp(&b.start));
    turns
}

/// Labels every voice on the speaker track: an enrolled speaker's name where there is one,
/// otherwise the "Unknown N" [`unknown_labels`] hands it.
///
/// The numbering rule -- first appearance, ties by cluster id, from 1 -- deliberately is not
/// implemented here. `enroll` has to work out which cluster an "Unknown 2" in an existing
/// transcript refers to, so it needs the identical numbering, and a second copy of the rule
/// would drift from this one with a misplaced name as the only symptom. See
/// [`unknown_labels`] for the rule and why the numbers a name displaces stay unused.
fn label_by_first_appearance(
    diarized: &[SpeakerTurn],
    identified: &BTreeMap<u32, Identification>,
) -> BTreeMap<u32, Label> {
    unknown_labels(diarized.iter().map(|turn| (turn.cluster, turn.start_s)))
        .into_iter()
        .map(|(id, unknown)| {
            let label = match identified.get(&id) {
                Some(who) => (who.name.clone(), Some(who.similarity)),
                None => (unknown, None),
            };
            (id, label)
        })
        .collect()
}

/// Decides whose voice a recognised segment is.
///
/// Majority *time* overlap: the cluster that was speaking for the longest while this segment
/// was being said wins it.
///
/// The alternative -- re-running the recogniser separately over each diarized turn -- was
/// rejected in this slice's design: it costs one Whisper invocation per turn and throws away
/// the accuracy Whisper gets from surrounding context. It remains the documented fallback if
/// overlap assignment turns out to be unreliable on real meetings.
///
/// A segment overlapping nothing falls back to the nearest turn in time. Whisper hearing
/// speech where segmentation heard none is common at the quiet edges of a turn, and the
/// nearest speaker is very nearly always the right answer; what matters is that the words
/// survive either way.
///
/// How much of the segment the winner actually held is deliberately not returned. It used to
/// land in `speaker_id_confidence`, which now answers a different question -- how sure the
/// *name* is -- and one field carrying two incompatible scales, told apart only by inspecting
/// the label, would be worse than not reporting the overlap at all.
fn attribute(segment: &AsrSegment, diarized: &[SpeakerTurn]) -> Option<u32> {
    let mut overlap: BTreeMap<u32, f64> = BTreeMap::new();
    for turn in diarized {
        let shared = turn.end_s.min(segment.end_s) - turn.start_s.max(segment.start_s);
        if shared > 0.0 {
            *overlap.entry(turn.cluster).or_insert(0.0) += shared;
        }
    }

    // Ascending id out of the map, and a strictly-greater test, so an exact tie goes to the
    // lower cluster id rather than to whichever entry the iterator happened to end on.
    // `max_by` would do the opposite, silently.
    let winner =
        overlap.iter().fold(
            None,
            |best: Option<(u32, f64)>, (&cluster, &seconds)| match best {
                Some((_, held)) if held >= seconds => best,
                _ => Some((cluster, seconds)),
            },
        );

    if let Some((cluster, _)) = winner {
        return Some(cluster);
    }

    diarized
        .iter()
        .min_by(|a, b| {
            gap(segment, a)
                .total_cmp(&gap(segment, b))
                .then(a.cluster.cmp(&b.cluster))
        })
        .map(|turn| turn.cluster)
}

/// How far apart a segment and a turn are in time; zero if they touch or overlap.
fn gap(segment: &AsrSegment, turn: &SpeakerTurn) -> f64 {
    (turn.start_s - segment.end_s)
        .max(segment.start_s - turn.end_s)
        .max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(start_s: f64, end_s: f64, text: &str) -> AsrSegment {
        AsrSegment {
            start_s,
            end_s,
            text: text.to_string(),
        }
    }

    fn turn(start_s: f64, end_s: f64, cluster: u32) -> SpeakerTurn {
        SpeakerTurn {
            start_s,
            end_s,
            cluster,
        }
    }

    /// The state of every session before anybody has been enrolled, which is what all the
    /// labelling and attribution tests below are about: voices, none of them with a name.
    fn nobody() -> BTreeMap<u32, Identification> {
        BTreeMap::new()
    }

    /// Enrolled speakers matched to clusters, as `identify` would have returned them.
    fn named(entries: &[(u32, &str, f32)]) -> BTreeMap<u32, Identification> {
        entries
            .iter()
            .map(|&(cluster, name, similarity)| {
                (
                    cluster,
                    Identification {
                        name: name.to_string(),
                        similarity,
                    },
                )
            })
            .collect()
    }

    /// Turns as (speaker, start, text), which is what a reader of `transcript.md` sees.
    fn said(turns: &[Turn]) -> Vec<(&str, f64, &str)> {
        turns
            .iter()
            .map(|t| (t.speaker.as_str(), t.start, t.text.as_str()))
            .collect()
    }

    /// Acceptance criterion #2, at the level that decides it: the mic track's label does not
    /// depend on diarization in any way, because it is never consulted.
    #[test]
    fn every_mic_turn_is_you_with_no_confidence_however_the_speaker_track_was_diarized() {
        let merged = merge(
            vec![segment(0.0, 1.0, "mine"), segment(5.0, 6.0, "also mine")],
            0.0,
            Vec::new(),
            0.0,
            &[turn(0.0, 10.0, 0)],
            &nobody(),
        );

        assert_eq!(merged.len(), 2);
        for turn in &merged {
            assert_eq!(turn.speaker, SPEAKER_YOU);
            assert_eq!(turn.source_track, SourceTrack::Mic);
            assert_eq!(turn.speaker_id_confidence, None);
        }
    }

    /// Acceptance criterion #1: one timeline, both tracks, strictly chronological. The two
    /// offsets are what make this more than a sort -- the speaker track here started 2 s
    /// before the mic did, so an implementation that ignored them would emit these in
    /// exactly the wrong order.
    #[test]
    fn both_tracks_interleave_in_chronological_order_on_one_timeline() {
        let merged = merge(
            vec![segment(0.0, 1.0, "hello"), segment(4.0, 5.0, "sure")],
            2.0,
            vec![segment(0.5, 1.5, "hi there"), segment(5.0, 6.0, "thanks")],
            0.0,
            &[turn(0.0, 8.0, 0)],
            &nobody(),
        );

        assert_eq!(
            said(&merged),
            [
                ("Unknown 1", 0.5, "hi there"),
                ("You", 2.0, "hello"),
                ("Unknown 1", 5.0, "thanks"),
                ("You", 6.0, "sure"),
            ]
        );
        assert!(merged.windows(2).all(|w| w[0].start <= w[1].start));
    }

    /// Acceptance criterion #3: distinct voices get distinct labels rather than one shared
    /// "Unknown" bucket, so a reader can tell how many people they could not identify.
    #[test]
    fn distinct_clusters_get_distinct_labels() {
        let merged = merge(
            Vec::new(),
            0.0,
            vec![
                segment(0.0, 1.0, "first voice"),
                segment(2.0, 3.0, "second voice"),
                segment(4.0, 5.0, "first again"),
            ],
            0.0,
            &[turn(0.0, 1.0, 0), turn(2.0, 3.0, 1), turn(4.0, 5.0, 0)],
            &nobody(),
        );

        let speakers: Vec<&str> = merged.iter().map(|t| t.speaker.as_str()).collect();
        assert_eq!(speakers, ["Unknown 1", "Unknown 2", "Unknown 1"]);
    }

    /// Numbering is by *first appearance*, not by cluster id. Cluster ids rank voices by how
    /// much they talked, so the person who spoke first is routinely not cluster 0 -- and a
    /// transcript whose first speaker is called "Unknown 2" reads as a bug.
    #[test]
    fn labels_are_numbered_by_first_appearance_rather_than_by_cluster_id() {
        let merged = merge(
            Vec::new(),
            0.0,
            vec![segment(0.0, 1.0, "early"), segment(5.0, 6.0, "late")],
            0.0,
            // Cluster 2 speaks first; cluster 0 -- the most talkative -- speaks second.
            &[turn(0.0, 1.0, 2), turn(5.0, 6.0, 0)],
            &nobody(),
        );

        assert_eq!(
            said(&merged),
            [("Unknown 1", 0.0, "early"), ("Unknown 2", 5.0, "late")]
        );
    }

    /// The case attribution exists for: one recognised sentence spanning a hand-over goes to
    /// whoever held most of it.
    #[test]
    fn a_segment_straddling_two_turns_goes_to_the_majority_holder() {
        let merged = merge(
            Vec::new(),
            0.0,
            vec![segment(0.0, 4.0, "...and then, right, yes")],
            0.0,
            &[turn(0.0, 1.0, 0), turn(1.0, 4.0, 1)],
            &nobody(),
        );

        // Three of the segment's four seconds belonged to cluster 1, which is the second
        // voice to speak.
        assert_eq!(merged[0].speaker, "Unknown 2");
    }

    /// Whisper routinely hears speech at the quiet edges of a turn that segmentation called
    /// silence. Those words are real meeting content and must not vanish, and the nearest
    /// speaker is the honest guess.
    #[test]
    fn a_segment_overlapping_nothing_lands_on_the_nearest_speaker_rather_than_vanishing() {
        let merged = merge(
            Vec::new(),
            0.0,
            vec![segment(10.0, 11.0, "mm-hm")],
            0.0,
            &[turn(0.0, 5.0, 0), turn(12.0, 20.0, 1)],
            &nobody(),
        );

        assert_eq!(said(&merged), [("Unknown 2", 10.0, "mm-hm")]);
    }

    /// Diarization finding nobody at all -- an unusually quiet or noisy track -- still has to
    /// produce a readable transcript rather than a blank speaker or no turns.
    #[test]
    fn with_no_clusters_the_whole_speaker_track_becomes_one_unknown_speaker() {
        let merged = merge(
            Vec::new(),
            0.0,
            vec![segment(0.0, 1.0, "one"), segment(2.0, 3.0, "two")],
            0.0,
            &[],
            &nobody(),
        );

        let speakers: Vec<&str> = merged.iter().map(|t| t.speaker.as_str()).collect();
        assert_eq!(speakers, ["Unknown 1", "Unknown 1"]);
        assert!(merged.iter().all(|t| t.speaker_id_confidence.is_none()));
    }

    /// Determinism at a tie, which is what makes a `--force` rerun byte-identical. Sorting
    /// is stable and the mic turns are built first, so the local speaker wins the instant.
    #[test]
    fn a_mic_turn_and_a_speaker_turn_starting_together_put_the_mic_first() {
        let merged = merge(
            vec![segment(3.0, 4.0, "mine")],
            0.0,
            vec![segment(3.0, 4.0, "theirs")],
            0.0,
            &[turn(0.0, 10.0, 0)],
            &nobody(),
        );

        assert_eq!(
            said(&merged),
            [("You", 3.0, "mine"), ("Unknown 1", 3.0, "theirs")]
        );
    }

    /// Both single-track meetings are ordinary: a call where the user never unmuted, and one
    /// where nobody else was on the line.
    #[test]
    fn an_empty_track_on_either_side_is_not_a_problem() {
        let mic_only = merge(
            vec![segment(0.0, 1.0, "just me")],
            0.0,
            Vec::new(),
            0.0,
            &[],
            &nobody(),
        );
        assert_eq!(said(&mic_only), [("You", 0.0, "just me")]);

        let speaker_only = merge(
            Vec::new(),
            0.0,
            vec![segment(0.0, 1.0, "just them")],
            0.0,
            &[turn(0.0, 1.0, 0)],
            &nobody(),
        );
        assert_eq!(said(&speaker_only), [("Unknown 1", 0.0, "just them")]);

        assert!(merge(Vec::new(), 0.0, Vec::new(), 0.0, &[], &nobody()).is_empty());
    }

    /// A voice interrupted and resuming keeps its number: identity comes from the cluster,
    /// which spans the whole meeting, not from proximity in the transcript.
    #[test]
    fn a_speaker_returning_after_someone_else_keeps_the_same_label() {
        let merged = merge(
            Vec::new(),
            0.0,
            vec![
                segment(0.0, 5.0, "opening"),
                segment(100.0, 105.0, "interjection"),
                segment(200.0, 205.0, "closing"),
            ],
            0.0,
            &[
                turn(0.0, 5.0, 0),
                turn(100.0, 105.0, 1),
                turn(200.0, 205.0, 0),
            ],
            &nobody(),
        );

        let speakers: Vec<&str> = merged.iter().map(|t| t.speaker.as_str()).collect();
        assert_eq!(speakers, ["Unknown 1", "Unknown 2", "Unknown 1"]);
    }

    /// Two voices whose first turns begin at the same instant -- they started talking over
    /// each other -- still have to be numbered reproducibly.
    #[test]
    fn voices_that_first_speak_at_the_same_instant_are_numbered_by_cluster_id() {
        let diarized = [turn(0.0, 2.0, 1), turn(0.0, 2.0, 0)];
        let labels = label_by_first_appearance(&diarized, &nobody());
        assert_eq!(labels[&0].0, "Unknown 1");
        assert_eq!(labels[&1].0, "Unknown 2");
    }

    /// Overlap is measured in seconds, not in turns: three short turns from one voice must
    /// not out-vote one long turn from another.
    #[test]
    fn attribution_counts_time_rather_than_the_number_of_turns() {
        let merged = merge(
            Vec::new(),
            0.0,
            vec![segment(0.0, 10.0, "a long sentence")],
            0.0,
            &[
                turn(0.0, 1.0, 0),
                turn(1.0, 2.0, 0),
                turn(2.0, 3.0, 0),
                turn(3.0, 10.0, 1),
            ],
            &nobody(),
        );

        assert_eq!(merged[0].speaker, "Unknown 2");
    }

    /// Acceptance criterion #1, at the level `merge` decides it: a matched cluster renders the
    /// person's name, and the neighbours who were not matched keep theirs.
    ///
    /// The number the name replaced -- "Unknown 2" here -- stays unused, which is the visible
    /// consequence of numbering over every voice and substituting afterwards. Renumbering
    /// instead would mean naming one person silently relabels everyone who spoke after them.
    #[test]
    fn a_matched_cluster_is_named_and_leaves_the_number_it_replaced_unused() {
        let merged = merge(
            Vec::new(),
            0.0,
            vec![
                segment(0.0, 1.0, "first"),
                segment(2.0, 3.0, "second"),
                segment(4.0, 5.0, "third"),
            ],
            0.0,
            &[turn(0.0, 1.0, 0), turn(2.0, 3.0, 1), turn(4.0, 5.0, 2)],
            &named(&[(1, "Alice", 0.91)]),
        );

        let speakers: Vec<&str> = merged.iter().map(|t| t.speaker.as_str()).collect();
        assert_eq!(speakers, ["Unknown 1", "Alice", "Unknown 3"]);
    }

    /// Acceptance criterion #5, both halves, on the one merge that exercises them together:
    /// the identified turn carries the similarity its name was decided on, and every turn that
    /// makes no identity claim -- unmatched speaker turns and the whole mic track -- carries
    /// nothing rather than a number about something else.
    #[test]
    fn only_named_turns_carry_a_confidence() {
        let merged = merge(
            vec![segment(1.0, 2.0, "mine")],
            0.0,
            vec![segment(0.0, 0.5, "hers"), segment(3.0, 4.0, "theirs")],
            0.0,
            &[turn(0.0, 0.5, 0), turn(3.0, 4.0, 1)],
            &named(&[(0, "Alice", 0.87)]),
        );

        let confidences: Vec<(&str, Option<f32>)> = merged
            .iter()
            .map(|t| (t.speaker.as_str(), t.speaker_id_confidence))
            .collect();
        assert_eq!(
            confidences,
            [("Alice", Some(0.87)), ("You", None), ("Unknown 2", None),]
        );
    }

    /// One name spans the meeting, exactly as one number does: identity comes from the
    /// cluster, so a named speaker interrupted and resuming is still that person.
    #[test]
    fn a_named_speaker_keeps_their_name_across_the_whole_meeting() {
        let merged = merge(
            Vec::new(),
            0.0,
            vec![
                segment(0.0, 1.0, "opening"),
                segment(10.0, 11.0, "interjection"),
                segment(20.0, 21.0, "closing"),
            ],
            0.0,
            &[turn(0.0, 1.0, 0), turn(10.0, 11.0, 1), turn(20.0, 21.0, 0)],
            &named(&[(0, "Alice", 0.8)]),
        );

        let speakers: Vec<&str> = merged.iter().map(|t| t.speaker.as_str()).collect();
        assert_eq!(speakers, ["Alice", "Unknown 2", "Alice"]);
    }

    /// An identification for a cluster diarization did not produce cannot name anything.
    /// Nothing generates that today, but `merge` must not index a stale map into a wrong
    /// label if the two ever drift apart.
    #[test]
    fn an_identification_for_an_absent_cluster_changes_nothing() {
        let merged = merge(
            Vec::new(),
            0.0,
            vec![segment(0.0, 1.0, "only voice")],
            0.0,
            &[turn(0.0, 1.0, 0)],
            &named(&[(9, "Nobody Here", 0.99)]),
        );

        assert_eq!(said(&merged), [("Unknown 1", 0.0, "only voice")]);
        assert_eq!(merged[0].speaker_id_confidence, None);
    }
}
