//! Turning Nemotron's per-slot speech probabilities into speaker turns.
//!
//! [`speaker_probabilities`](crate::speaker_probabilities) yields one probability per slot per
//! 10 ms frame. Probabilities are not turns: thresholded naively the model splits one speaker
//! into many tiny segments. [`slot_turns`] decodes each slot on its own with hysteresis, merges
//! short gaps and drops short blips. Slots are decoded independently, so two slots that are
//! active at once give two overlapping turns; that overlap is kept on purpose, because it is
//! what later shows two voices belong to different people.

use crate::nemotron_cache::config::SPEAKERS;

/// A slot becomes active when its probability is strictly above this (doc-015: "enter above
/// 0.5"). The threshold barely mattered between 0.25 and 0.5 on the doc-015 session; 0.7 was
/// worse. Re-measured on the first 40 minutes of a real speaker track (nemotron-turns example):
/// 0.4 / 0.5 / 0.6 / 0.7 gave 383 / 378 / 373 / 373 turns and 1189 / 1184 / 1178 / 1175 s of
/// speech, so the value is not sensitive.
const ENTER_PROB: f32 = 0.5;

/// An active slot stays active until its probability falls strictly below this (hysteresis, so
/// a dip between the two thresholds does not split a turn). doc-015: "stay until below 0.25".
const STAY_PROB: f32 = 0.25;

/// Gaps up to and including this many seconds between two runs of one slot are merged. doc-015
/// measured, on two 5-minute windows, raw turn counts falling from 83 to 46 and from 24 to 18
/// (with the minimum length below) with no loss of agreement with pyannote's speech map
/// (F1 about 0.84-0.93). On the 40 minute real track gap 0 / 0.5 / 1.0 / 1.5 s gave 598 / 485 /
/// 378 / 308 turns: the gap is the knob that moves the count, and 1.0 s halves the 781 raw runs.
const MERGE_GAP_S: f64 = 1.0;

/// Merged turns shorter than this many seconds are dropped (a turn of exactly this length is
/// kept). Merging happens first, as in doc-015. On the 40 minute real track min 0 / 0.3 / 0.5 /
/// 0.8 s gave 464 / 427 / 378 / 331 turns, with 1244 / 1235 / 1217 / 1187 s of speech.
const MIN_TURN_S: f64 = 0.5;

/// Seconds per probability row: the mel hop of 160 samples at 16 kHz.
const FRAME_S: f64 = 0.01;

/// One speech turn of one Nemotron slot, in seconds into the speaker track.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotTurn {
    pub start_s: f64,
    pub end_s: f64,
    pub slot: usize,
}

/// Decodes `probs` (slot-minor rows of 8 slot values, one row per 10 ms frame, as
/// [`speaker_probabilities`](crate::speaker_probabilities) returns) into turns, ordered by start
/// then slot. A run still active at the last frame ends at the final frame boundary. Silent or
/// empty input gives no turns.
///
/// `probs.len()` must be a multiple of 8; a trailing partial row is ignored.
pub fn slot_turns(probs: &[f32]) -> Vec<SlotTurn> {
    debug_assert_eq!(probs.len() % SPEAKERS, 0, "probs must be whole rows");
    let frames = probs.len() / SPEAKERS;
    let merge_gap = (MERGE_GAP_S / FRAME_S).round() as usize;
    let min_len = (MIN_TURN_S / FRAME_S).round() as usize;

    let mut turns = Vec::new();
    for slot in 0..SPEAKERS {
        // Runs as half-open frame ranges, merged as they are found.
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut open: Option<usize> = None;
        for f in 0..=frames {
            let p = if f < frames {
                probs[f * SPEAKERS + slot]
            } else {
                // Sentinel past the end closes a run that is still open.
                0.0
            };
            match open {
                None if p > ENTER_PROB => open = Some(f),
                Some(start) if p < STAY_PROB => {
                    open = None;
                    match runs.last_mut() {
                        Some(last) if start - last.1 <= merge_gap => last.1 = f,
                        _ => runs.push((start, f)),
                    }
                }
                _ => {}
            }
        }
        turns.extend(
            runs.into_iter()
                .filter(|(s, e)| e - s >= min_len)
                .map(|(s, e)| SlotTurn {
                    start_s: s as f64 * FRAME_S,
                    end_s: e as f64 * FRAME_S,
                    slot,
                }),
        );
    }
    turns.sort_by(|a, b| a.start_s.total_cmp(&b.start_s).then(a.slot.cmp(&b.slot)));
    turns
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Probabilities for one slot given as `(frames, value)` segments; other slots silent.
    fn track(slot: usize, segments: &[(usize, f32)]) -> Vec<f32> {
        track_multi(&[(slot, segments)])
    }

    fn track_multi(slots: &[(usize, &[(usize, f32)])]) -> Vec<f32> {
        let frames = slots
            .iter()
            .map(|(_, s)| s.iter().map(|(n, _)| n).sum::<usize>())
            .max()
            .unwrap();
        let mut out = vec![0.0; frames * SPEAKERS];
        for (slot, segments) in slots {
            let mut f = 0;
            for &(n, v) in *segments {
                for _ in 0..n {
                    out[f * SPEAKERS + slot] = v;
                    f += 1;
                }
            }
        }
        out
    }

    fn spans(turns: &[SlotTurn]) -> Vec<(f64, f64, usize)> {
        turns
            .iter()
            .map(|t| {
                // Whole centiseconds, so the comparison does not depend on float noise.
                let cs = |x: f64| (x * 100.0).round() / 100.0;
                (cs(t.start_s), cs(t.end_s), t.slot)
            })
            .collect()
    }

    #[test]
    fn a_gap_of_exactly_the_merge_gap_merges() {
        let p = track(2, &[(100, 0.9), (100, 0.0), (100, 0.9)]);
        assert_eq!(spans(&slot_turns(&p)), [(0.0, 3.0, 2)]);
    }

    #[test]
    fn a_gap_just_over_the_merge_gap_splits() {
        let p = track(2, &[(100, 0.9), (101, 0.0), (100, 0.9)]);
        assert_eq!(spans(&slot_turns(&p)), [(0.0, 1.0, 2), (2.01, 3.01, 2)]);
    }

    #[test]
    fn a_gap_just_under_the_merge_gap_merges() {
        let p = track(0, &[(100, 0.9), (99, 0.0), (100, 0.9)]);
        assert_eq!(spans(&slot_turns(&p)), [(0.0, 2.99, 0)]);
    }

    #[test]
    fn a_dip_between_the_thresholds_does_not_split() {
        let p = track(1, &[(100, 0.9), (10, 0.3), (100, 0.9), (200, 0.0)]);
        assert_eq!(spans(&slot_turns(&p)), [(0.0, 2.1, 1)]);
    }

    #[test]
    fn a_dip_below_the_stay_threshold_ends_the_run_but_the_gap_still_merges() {
        // 0.2 is below STAY, so the run ends; the 10 frame gap then merges, giving one turn.
        let p = track(1, &[(100, 0.9), (10, 0.2), (100, 0.9)]);
        assert_eq!(spans(&slot_turns(&p)), [(0.0, 2.1, 1)]);
    }

    #[test]
    fn a_run_cannot_start_between_the_thresholds() {
        let p = track(1, &[(100, 0.3), (100, 0.5)]);
        assert!(slot_turns(&p).is_empty());
    }

    #[test]
    fn a_turn_under_the_minimum_is_dropped_and_one_at_it_is_kept() {
        let p = track(3, &[(49, 0.9), (200, 0.0), (50, 0.9)]);
        assert_eq!(spans(&slot_turns(&p)), [(2.49, 2.99, 3)]);
    }

    #[test]
    fn two_short_runs_merge_before_the_minimum_is_applied() {
        let p = track(3, &[(30, 0.9), (10, 0.0), (30, 0.9)]);
        assert_eq!(spans(&slot_turns(&p)), [(0.0, 0.7, 3)]);
    }

    #[test]
    fn a_slot_active_at_the_first_and_last_frame_spans_the_edges() {
        let p = track(0, &[(100, 0.9), (300, 0.0), (100, 0.9)]);
        assert_eq!(spans(&slot_turns(&p)), [(0.0, 1.0, 0), (4.0, 5.0, 0)]);
    }

    #[test]
    fn silence_and_empty_input_give_no_turns() {
        assert!(slot_turns(&track(0, &[(500, 0.0)])).is_empty());
        assert!(slot_turns(&[]).is_empty());
    }

    #[test]
    fn overlapping_slots_give_overlapping_turns() {
        let p = track_multi(&[
            (5, &[(0, 0.0), (300, 0.9), (200, 0.0)]),
            (1, &[(200, 0.0), (300, 0.9)]),
        ]);
        let turns = slot_turns(&p);
        assert_eq!(spans(&turns), [(0.0, 3.0, 5), (2.0, 5.0, 1)]);
        assert!(turns[0].end_s > turns[1].start_s);
    }

    /// The Hugging Face reference probabilities decoded and compared with a list computed by
    /// `testdata/nemotron/expected-turns.py`, which does not share code with `slot_turns`.
    #[test]
    fn the_fixture_probabilities_decode_to_the_independent_expectation() {
        const PROBS: &[u8] = include_bytes!("../testdata/nemotron/nemotron-probs.bin");
        let probs: Vec<f32> = PROBS[8..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let got: Vec<(u32, u32, usize)> = slot_turns(&probs)
            .iter()
            .map(|t| {
                (
                    (t.start_s * 100.0).round() as u32,
                    (t.end_s * 100.0).round() as u32,
                    t.slot,
                )
            })
            .collect();
        // (start frame, end frame, slot)
        let want: &[(u32, u32, usize)] = &[
            (50, 338, 0),
            (439, 1134, 1),
            (1231, 1409, 2),
            (1491, 2876, 0),
            (2972, 4291, 1),
            (4432, 4534, 2),
            (4612, 5081, 0),
            (5169, 6456, 1),
            (6572, 7992, 2),
            (8060, 10001, 0),
        ];
        assert_eq!(got, want);
    }
}
