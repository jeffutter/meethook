//! Combining two recognised tracks into one chronological, speaker-labelled transcript.
//!
//! This is the part of transcription with no audio and no model in it. Given what Whisper
//! heard on each track and what diarization made of the speaker track, everything left --
//! putting both tracks on one timeline, deciding which voice said which recognised sentence,
//! naming the voices, ordering the result -- is deterministic. Keeping it that way is what
//! makes it testable in microseconds against fixtures, which is where nearly all of the
//! behaviour a reader of `transcript.md` will actually notice is decided.

use std::collections::BTreeMap;

use meethook_session::{SPEAKER_YOU, SourceTrack, Turn, unknown_speaker};

use crate::asr::AsrSegment;
use crate::diarize::SpeakerTurn;

/// Combines both tracks into one chronological, speaker-labelled transcript.
///
/// `mic` and `speaker` are what the recogniser heard on each track, timed from the start of
/// that track; the two offsets place each track on the session timeline (exactly one of them
/// is non-zero, since the timeline starts at whichever track began first). `diarized` is the
/// speaker track's attributed speech, in that same track's time.
///
/// Every mic-track segment becomes a turn labelled [`SPEAKER_YOU`] with no confidence: the
/// speaker there is known by construction rather than inferred, and reporting a number for it
/// would be inventing one. There is exactly one local speaker, so the mic track is never
/// diarized at all.
///
/// Every speaker-track segment becomes a turn too, whatever diarization made of it. Dropping
/// recognised words because the diarizer heard no speech under them would lose real meeting
/// content, which is strictly worse than an imperfect label a reader can see and correct.
///
/// No overlap or cross-talk handling: two people talking at once produce two turns whose
/// times overlap, ordered by where each started.
pub fn merge(
    mic: Vec<AsrSegment>,
    mic_offset_s: f64,
    speaker: Vec<AsrSegment>,
    speaker_offset_s: f64,
    diarized: &[SpeakerTurn],
) -> Vec<Turn> {
    let labels = label_by_first_appearance(diarized);

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
        let (cluster, confidence) = attribute(&segment, diarized);
        Turn {
            // No cluster at all means diarization found nobody on a track Whisper still
            // heard words on. One unnamed speaker is the honest reading of that, and a far
            // better one than an empty label or a dropped sentence.
            speaker: cluster
                .and_then(|id| labels.get(&id).cloned())
                .unwrap_or_else(|| unknown_speaker(1)),
            start: speaker_offset_s + segment.start_s,
            end: speaker_offset_s + segment.end_s,
            text: segment.text,
            source_track: SourceTrack::Speaker,
            speaker_id_confidence: Some(confidence),
        }
    }));

    // Stable, and that is the tie-break: `sort_by` in Rust preserves the order of equal
    // elements, and the mic turns were built first, so a mic turn and a speaker turn that
    // start at the same instant come out mic first. Deterministic rather than arbitrary,
    // which is what makes a `--force` rerun byte-identical.
    turns.sort_by(|a, b| a.start.total_cmp(&b.start));
    turns
}

/// Names every voice on the speaker track "Unknown N", numbered by when it first spoke.
///
/// Cluster ids are an artifact of clustering order -- they rank voices by how much they
/// talked -- and mean nothing to someone reading a transcript from the top. Numbering by
/// first appearance makes "Unknown 1" the first unidentified person to speak, which is
/// reproducible across reruns and is the only version of the label a user can act on.
fn label_by_first_appearance(diarized: &[SpeakerTurn]) -> BTreeMap<u32, String> {
    let mut first: BTreeMap<u32, f64> = BTreeMap::new();
    for turn in diarized {
        first
            .entry(turn.cluster)
            .and_modify(|earliest| *earliest = earliest.min(turn.start_s))
            .or_insert(turn.start_s);
    }

    // Ascending cluster id out of the map, then a stable sort by time: two voices whose
    // first turns begin at the same instant are numbered by cluster id, so the labels do not
    // depend on which order the turns happened to arrive in.
    let mut order: Vec<(u32, f64)> = first.into_iter().collect();
    order.sort_by(|a, b| a.1.total_cmp(&b.1));

    order
        .into_iter()
        .enumerate()
        .map(|(rank, (id, _))| (id, unknown_speaker(rank + 1)))
        .collect()
}

/// Decides whose voice a recognised segment is, and how sure that is.
///
/// Majority *time* overlap: the cluster that was speaking for the longest while this
/// segment was being said wins it. Confidence is the fraction of the segment that cluster
/// covered, so a sentence sitting squarely inside one person's turn scores 1.0 and one
/// straddling a hand-over scores what it deserves.
///
/// The alternative -- re-running the recogniser separately over each diarized turn -- was
/// rejected in this slice's design: it costs one Whisper invocation per turn and throws away
/// the accuracy Whisper gets from surrounding context. It remains the documented fallback if
/// overlap assignment turns out to be unreliable on real meetings.
///
/// A segment overlapping nothing falls back to the nearest turn in time, with confidence
/// 0.0 to say so. Whisper hearing speech where segmentation heard none is common at the
/// quiet edges of a turn, and the nearest speaker is very nearly always the right answer;
/// what matters is that the words survive either way.
fn attribute(segment: &AsrSegment, diarized: &[SpeakerTurn]) -> (Option<u32>, f32) {
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

    if let Some((cluster, best)) = winner {
        let duration = segment.end_s - segment.start_s;
        let confidence = if duration > 0.0 {
            (best / duration).clamp(0.0, 1.0)
        } else {
            0.0
        };
        return (Some(cluster), confidence as f32);
    }

    let nearest = diarized.iter().min_by(|a, b| {
        gap(segment, a)
            .total_cmp(&gap(segment, b))
            .then(a.cluster.cmp(&b.cluster))
    });
    (nearest.map(|turn| turn.cluster), 0.0)
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
        );

        assert_eq!(
            said(&merged),
            [("Unknown 1", 0.0, "early"), ("Unknown 2", 5.0, "late")]
        );
    }

    /// The case attribution exists for: one recognised sentence spanning a hand-over goes to
    /// whoever held most of it, and reports how much of it that was.
    #[test]
    fn a_segment_straddling_two_turns_goes_to_the_majority_holder() {
        let merged = merge(
            Vec::new(),
            0.0,
            vec![segment(0.0, 4.0, "...and then, right, yes")],
            0.0,
            &[turn(0.0, 1.0, 0), turn(1.0, 4.0, 1)],
        );

        assert_eq!(merged[0].speaker, "Unknown 2");
        // Three of the segment's four seconds belonged to the winner.
        let confidence = merged[0].speaker_id_confidence.unwrap();
        assert!((confidence - 0.75).abs() < 1e-6, "{confidence}");
    }

    /// Whisper routinely hears speech at the quiet edges of a turn that segmentation called
    /// silence. Those words are real meeting content and must not vanish; the nearest
    /// speaker is the honest guess, and a zero confidence is how the guess is declared.
    #[test]
    fn a_segment_overlapping_nothing_lands_on_the_nearest_speaker_rather_than_vanishing() {
        let merged = merge(
            Vec::new(),
            0.0,
            vec![segment(10.0, 11.0, "mm-hm")],
            0.0,
            &[turn(0.0, 5.0, 0), turn(12.0, 20.0, 1)],
        );

        assert_eq!(said(&merged), [("Unknown 2", 10.0, "mm-hm")]);
        assert_eq!(merged[0].speaker_id_confidence, Some(0.0));
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
        );

        let speakers: Vec<&str> = merged.iter().map(|t| t.speaker.as_str()).collect();
        assert_eq!(speakers, ["Unknown 1", "Unknown 1"]);
        assert!(merged.iter().all(|t| t.speaker_id_confidence == Some(0.0)));
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
        );
        assert_eq!(said(&mic_only), [("You", 0.0, "just me")]);

        let speaker_only = merge(
            Vec::new(),
            0.0,
            vec![segment(0.0, 1.0, "just them")],
            0.0,
            &[turn(0.0, 1.0, 0)],
        );
        assert_eq!(said(&speaker_only), [("Unknown 1", 0.0, "just them")]);

        assert!(merge(Vec::new(), 0.0, Vec::new(), 0.0, &[]).is_empty());
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
        );

        let speakers: Vec<&str> = merged.iter().map(|t| t.speaker.as_str()).collect();
        assert_eq!(speakers, ["Unknown 1", "Unknown 2", "Unknown 1"]);
    }

    /// Two voices whose first turns begin at the same instant -- they started talking over
    /// each other -- still have to be numbered reproducibly.
    #[test]
    fn voices_that_first_speak_at_the_same_instant_are_numbered_by_cluster_id() {
        let diarized = [turn(0.0, 2.0, 1), turn(0.0, 2.0, 0)];
        let labels = label_by_first_appearance(&diarized);
        assert_eq!(labels[&0], "Unknown 1");
        assert_eq!(labels[&1], "Unknown 2");
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
        );

        assert_eq!(merged[0].speaker, "Unknown 2");
    }
}
