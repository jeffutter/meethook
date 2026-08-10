//! The timeline policy behind the speech gate: which stretches of a track to hand the
//! recogniser, and how to read the timestamps it returns back onto the original track.
//!
//! [`crate::SileroVad`] says where speech is. This module decides what to do with that answer,
//! and it is deliberately the part with no audio, no model and no `whisper_rs` type in it.
//! A mistake here is invisible in the worst way: the transcript still reads correctly, and
//! every turn in it is timed wrongly, which [`crate::merge()`] then interleaves against the
//! speaker track in the wrong order. Keeping the policy as pure functions over region lists
//! means it is exercised in microseconds by the tests at the bottom of this file, the way
//! `segmentation::decode_powerset` and [`crate::merge()`] already are.
//!
//! # Why splice rather than decode each region
//!
//! Whisper decodes a padded 30 s window at a time whatever it is handed, so one `full()` call
//! per region charges a whole window for a one-second region. On session `20260810-093047`'s
//! mic track that is ~315 windows against the ungated track's 87. Splicing the regions into
//! one buffer instead costs `ceil(speech / 30 s)` -- about 27 -- which is the entire point.
//!
//! # What is mirrored from whisper.cpp, and what is not
//!
//! `whisper_vad()` (whisper.cpp 1.8.3, `src/whisper.cpp:6608`) is upstream's version of this
//! gate, and the splice below is deliberately the same shape: extend each region's end by a
//! small overlap, copy the pieces into one buffer, separate them by digital silence.
//!
//! Its map-back is **not** mirrored, because it has a scale error. `whisper_vad` records
//! `orig_end` as the *unextended* region end while `vad_end` measures the piece it actually
//! copied, overlap included (`whisper.cpp:6698-6711`), then interpolates between those pairs
//! (`:7954` onwards). That compresses time inside every piece by `dur / (dur + overlap)` --
//! 10% on a one-second region. Since the samples are copied unmodified, the honest map is a
//! translation with slope exactly 1, which needs no interpolation and no mapping table, and
//! which is what makes [`Splice::to_original`]'s round trip exact rather than approximate.

use crate::audio::TARGET_RATE;
use crate::vad::SpeechRegion;

/// How far past its detected end each piece is carried, so a word is not clipped at a region
/// boundary.
///
/// whisper.cpp's `samples_overlap` default (`whisper.cpp:6653`). Note that it *stacks* on
/// [`crate::VadTuning::speech_pad_s`], which the detector has already added to both ends: at
/// the shipped tuning a piece therefore runs 0.13 s past the last frame judged to be speech.
const OVERLAP_S: f64 = 0.1;

/// Digital silence inserted between pieces, so two unrelated stretches of speech do not run
/// into each other mid-word.
///
/// whisper.cpp's `silence_samples` (`whisper.cpp:6672`). It is not a barrier -- a decoder
/// segment can and does span it -- which is why [`Splice::to_original`] states what happens to
/// a segment that crosses one rather than leaving it to emerge.
const GAP_S: f64 = 0.1;

/// The floor on a mapped span's length, mirroring upstream (`whisper.cpp:7992-7997`), so no
/// turn comes back zero-length.
const MIN_MAPPED_S: f64 = 0.01;

/// The window Whisper decodes at a time. A property of the model, not a setting; it is here
/// only to turn seconds of spliced audio into the number of decoder passes they cost, which is
/// the figure that makes the gate's saving legible.
const WHISPER_WINDOW_S: f64 = 30.0;

/// One stretch of the original track, copied into the spliced buffer at a known offset.
///
/// All three fields are **sample counts, not seconds**, and [`Splice::build`] indexes with the
/// same integers [`Splice::plan`] computed. Deriving the boundaries a second time from seconds
/// is the one way a plan/copy mismatch gets in, and its failure mode is a plausible-looking
/// transcript with every timestamp shifted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Piece {
    source_start: usize,
    source_end: usize,
    spliced_start: usize,
}

impl Piece {
    fn len(&self) -> usize {
        self.source_end - self.source_start
    }
}

/// A plan for decoding only the speech in a track, plus the map back to the track's timeline.
///
/// Built by [`Splice::plan`], which returns `None` when there is nothing to decode -- so "this
/// track holds no speech" is a value the caller cannot forget to handle rather than a length
/// check it might.
pub(crate) struct Splice {
    /// Ascending, non-overlapping, and non-empty: [`Splice::plan`] returns `None` rather than
    /// an empty `Splice`.
    pieces: Vec<Piece>,
    total_samples: usize,
    spliced_samples: usize,
}

impl Splice {
    /// Plans the splice for `regions` over a track of `total_samples`, or `None` when there is
    /// nothing to decode.
    ///
    /// `regions` is expected ascending and non-overlapping, which is what
    /// [`crate::SileroVad::speech_regions`] documents and guarantees. Nothing here depends on
    /// that being true: every end is clamped to the track and to the next region's start, every
    /// start is held at or after the previous piece's end, and a piece left with no samples is
    /// dropped. So a detector that stopped honouring its contract would cost audio, not
    /// correctness.
    pub(crate) fn plan(regions: &[SpeechRegion], total_samples: usize) -> Option<Splice> {
        let gap_samples = samples(GAP_S);
        let overlap_samples = samples(OVERLAP_S);

        let mut pieces: Vec<Piece> = Vec::with_capacity(regions.len());
        let mut cursor = 0;
        let mut previous_source_end = 0;
        for (i, region) in regions.iter().enumerate() {
            let start = samples(region.start_s).max(previous_source_end);
            let mut end = samples(region.end_s);
            // The last piece is not extended: there is nothing after it that its extension
            // could be reaching into, and upstream does the same (`whisper.cpp:6698-6711`).
            if i + 1 < regions.len() {
                // The overlap yields to the next region rather than duplicating its opening
                // samples. At the shipped tuning the regions are half a second apart and this
                // never fires, but it is what keeps the pieces disjoint by construction --
                // which is what makes the reported speech total honest and the map back
                // monotonic.
                end = (end + overlap_samples).min(samples(regions[i + 1].start_s));
            }
            let end = end.min(total_samples);
            if end <= start {
                continue;
            }

            if !pieces.is_empty() {
                cursor += gap_samples;
            }
            pieces.push(Piece {
                source_start: start,
                source_end: end,
                spliced_start: cursor,
            });
            cursor += end - start;
            previous_source_end = end;
        }

        if pieces.is_empty() {
            return None;
        }
        Some(Splice {
            pieces,
            total_samples,
            spliced_samples: cursor,
        })
    }

    /// Copies the planned pieces out of `audio_16k_mono` into one buffer, separated by
    /// [`GAP_S`] of digital silence.
    ///
    /// # Panics
    ///
    /// If `audio_16k_mono` is not the track the plan was made against. A plan applied to a
    /// different buffer produces a transcript that reads correctly and is timed wrongly, which
    /// no test downstream would notice; the two calls are two lines apart in
    /// [`crate::WhisperEngine`]'s `transcribe`, so a mismatch is a programming error and is worth
    /// saying so.
    pub(crate) fn build(&self, audio_16k_mono: &[f32]) -> Vec<f32> {
        assert_eq!(
            audio_16k_mono.len(),
            self.total_samples,
            "the splice was planned against a different track"
        );

        let mut spliced = vec![0.0; self.spliced_samples];
        for piece in &self.pieces {
            spliced[piece.spliced_start..piece.spliced_start + piece.len()]
                .copy_from_slice(&audio_16k_mono[piece.source_start..piece.source_end]);
        }
        spliced
    }

    /// Maps a segment's span from the spliced timeline back onto the original track's.
    ///
    /// Both arguments and both return values are seconds. The four rules, stated because an
    /// emergent rule here is a rule nobody can review:
    ///
    /// 1. A timestamp inside a piece maps by **translation, slope 1**:
    ///    `original = source_start + (t - spliced_start)`. Exact, because the samples were
    ///    copied unmodified. This is the rule; the other three are its edges.
    /// 2. A timestamp inside an inserted gap maps to the **following** piece's start. The gap
    ///    holds no audio of its own to be timed against, and a segment that begins in inserted
    ///    silence is describing the speech that comes after it.
    /// 3. A timestamp past the last piece maps to the last piece's end. Only reachable through
    ///    the short-track padding in [`crate::WhisperEngine`]'s `transcribe`.
    /// 4. `end_s` is clamped to the end of the piece that holds `start_s`. A segment *can*
    ///    span a seam -- the seam is 0.1 s of silence, not a barrier -- and attributing its
    ///    whole span to the region it started in keeps every emitted turn inside real audio.
    ///    Letting it run to wherever `end_s` landed would emit a turn covering silence the
    ///    user never spoke through, which is the bug this gate exists to remove.
    ///
    /// Both values are then clamped into the track, and the span is floored at
    /// [`MIN_MAPPED_S`]. The floor wins over the clamp: on a segment at the very end of the
    /// track the returned end can sit 10 ms past the final sample, which is a better answer
    /// than a zero-length turn. These are timestamps, not slice indices -- nothing downstream
    /// cuts audio with them.
    ///
    /// The whole map runs in **integer samples**, and the incoming seconds are snapped to the
    /// nearest sample on the way in. That costs nothing -- whisper.cpp reports centiseconds,
    /// which are 160 samples wide -- and it is what makes rule 1 exact rather than exact to
    /// within a rounding error: `(a + b) / rate` and `a / rate + b / rate` are not the same
    /// `f64`, so a translation done in seconds misses a piece boundary by an ULP and the round
    /// trip stops being an identity.
    pub(crate) fn to_original(&self, start_s: f64, end_s: f64) -> (f64, f64) {
        let rate = f64::from(TARGET_RATE);
        let position = |seconds: f64| (seconds * rate).round() as i64;
        let start_at = position(start_s);

        // The first piece that has not already ended is the piece that holds `start_s` (rule 1)
        // or follows the gap it landed in (rule 2); `None` means it is past the last piece
        // (rule 3).
        let holding = self
            .pieces
            .iter()
            .find(|piece| start_at < (piece.spliced_start + piece.len()) as i64);

        let (start, end) = match holding {
            Some(piece) => {
                let spliced_start = piece.spliced_start as i64;
                let source_start = piece.source_start as i64;
                let start = source_start + (start_at - spliced_start).max(0);
                let end =
                    (source_start + (position(end_s) - spliced_start)).min(piece.source_end as i64);
                (start, end)
            }
            None => {
                let last = self.pieces.last().expect("a plan holds at least one piece");
                let end = last.source_end as i64;
                (end, end)
            }
        };

        let total = self.total_samples as i64;
        let start = start.clamp(0, total);
        let end = end.clamp(start, total).max(start + position(MIN_MAPPED_S));
        (start as f64 / rate, end as f64 / rate)
    }

    /// How many stretches of the track will be decoded.
    pub(crate) fn piece_count(&self) -> usize {
        self.pieces.len()
    }

    /// Seconds of the track that hold speech, gaps excluded.
    pub(crate) fn speech_duration_s(&self) -> f64 {
        let samples: usize = self.pieces.iter().map(Piece::len).sum();
        samples as f64 / f64::from(TARGET_RATE)
    }

    /// Seconds of audio the recogniser is handed, inserted silence included.
    pub(crate) fn spliced_duration_s(&self) -> f64 {
        self.spliced_samples as f64 / f64::from(TARGET_RATE)
    }

    /// Seconds of the original track.
    pub(crate) fn total_duration_s(&self) -> f64 {
        self.total_samples as f64 / f64::from(TARGET_RATE)
    }

    /// How many 30 s decoder passes the spliced buffer costs. The figure that says what the
    /// gate bought, in the unit the time is actually spent in.
    pub(crate) fn decode_windows(&self) -> usize {
        (self.spliced_duration_s() / WHISPER_WINDOW_S).ceil() as usize
    }
}

/// Seconds to samples at [`TARGET_RATE`], rounded rather than truncated so a boundary that
/// landed a float's width short of a sample does not lose one.
fn samples(seconds: f64) -> usize {
    if seconds <= 0.0 {
        return 0;
    }
    (seconds * f64::from(TARGET_RATE)).round() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = TARGET_RATE as f64;

    fn region(start_s: f64, end_s: f64) -> SpeechRegion {
        SpeechRegion { start_s, end_s }
    }

    /// Samples in a track of `seconds`, so the tests read in the unit the plan is stated in.
    fn track(seconds: f64) -> usize {
        (seconds * RATE) as usize
    }

    #[test]
    fn no_regions_is_nothing_to_decode() {
        assert!(Splice::plan(&[], track(60.0)).is_none());
        // A region list that clamps away entirely is the same answer, reached differently.
        assert!(Splice::plan(&[region(90.0, 95.0)], track(60.0)).is_none());
    }

    #[test]
    fn one_region_is_one_piece_with_no_overlap_and_no_gap() {
        let splice = Splice::plan(&[region(10.0, 20.0)], track(60.0)).unwrap();
        assert_eq!(splice.piece_count(), 1);
        // Not 10.1: the last piece is never extended.
        assert_eq!(splice.speech_duration_s(), 10.0);
        assert_eq!(splice.spliced_duration_s(), 10.0);
        assert_eq!(splice.total_duration_s(), 60.0);
        assert_eq!(splice.decode_windows(), 1);
    }

    #[test]
    fn three_regions_are_extended_except_the_last_and_separated_by_one_gap_each() {
        let regions = [region(1.0, 2.0), region(10.0, 11.0), region(20.0, 21.0)];
        let splice = Splice::plan(&regions, track(60.0)).unwrap();

        assert_eq!(splice.piece_count(), 3);
        // Two extended by 0.1 s, one not.
        assert!((splice.speech_duration_s() - 3.2).abs() < 1e-9);
        // Plus two gaps of 0.1 s.
        assert!((splice.spliced_duration_s() - 3.4).abs() < 1e-9);

        let starts: Vec<usize> = splice.pieces.iter().map(|p| p.spliced_start).collect();
        assert_eq!(
            starts,
            vec![0, samples(1.1) + samples(0.1), samples(2.2) + samples(0.2)]
        );
    }

    #[test]
    fn an_extension_past_the_track_is_clamped_back() {
        // The first region's extension would reach 60.05 s of a 60 s track.
        let regions = [region(50.0, 59.95), region(100.0, 105.0)];
        let splice = Splice::plan(&regions, track(60.0)).unwrap();
        assert_eq!(splice.piece_count(), 1);
        assert_eq!(splice.pieces[0].source_end, track(60.0));
    }

    #[test]
    fn an_extension_yields_to_the_next_region_rather_than_duplicating_it() {
        // 0.05 s apart, which is closer than the 0.1 s overlap.
        let regions = [region(1.0, 2.0), region(2.05, 3.0)];
        let splice = Splice::plan(&regions, track(60.0)).unwrap();
        assert_eq!(splice.pieces[0].source_end, samples(2.05));
        assert_eq!(splice.pieces[1].source_start, samples(2.05));
        // Disjoint, so the reported speech total is 1.0 s + 0.05 s of extension + 0.95 s: the
        // audio actually decoded, counted once rather than 0.1 s of it twice.
        assert!((splice.speech_duration_s() - 2.0).abs() < 1e-9);
    }

    /// The property the whole map-back rests on: because the samples are copied unmodified,
    /// every piece boundary is its own inverse. Exact equality, not a tolerance -- an
    /// interpolated map (which is what upstream does) fails this by design.
    #[test]
    fn every_piece_boundary_round_trips_exactly() {
        // Deliberately awkward offsets: not whole seconds, not whole centiseconds, and not
        // whole samples either.
        let regions: Vec<SpeechRegion> = (0..300)
            .map(|i| {
                let start = f64::from(i) * 3.137 + 0.0417;
                region(start, start + 0.9137)
            })
            .collect();
        let total = track(1000.0);
        let splice = Splice::plan(&regions, total).unwrap();
        assert_eq!(splice.piece_count(), 300);

        let rate = RATE;
        for piece in &splice.pieces {
            let (start, _) = splice.to_original(piece.spliced_start as f64 / rate, 0.0);
            assert_eq!(
                start,
                piece.source_start as f64 / rate,
                "start of {piece:?} did not round trip"
            );

            // The end is asked for as the end of a span starting inside the same piece, since
            // rule 4 defines `end_s` relative to the piece holding `start_s`.
            let (_, end) = splice.to_original(
                piece.spliced_start as f64 / rate,
                (piece.spliced_start + piece.len()) as f64 / rate,
            );
            assert_eq!(
                end,
                piece.source_end as f64 / rate,
                "end of {piece:?} did not round trip"
            );
        }
    }

    /// The acceptance criterion stated at the unit level: a region at a known offset comes back
    /// at that offset.
    #[test]
    fn a_region_at_a_known_offset_comes_back_at_that_offset() {
        let splice = Splice::plan(&[region(45.0, 55.0)], track(120.0)).unwrap();
        assert_eq!(splice.to_original(0.0, 10.0), (45.0, 55.0));
        // And an interior point, mid-region, with the same slope.
        assert_eq!(splice.to_original(2.5, 3.5), (47.5, 48.5));
    }

    #[test]
    fn a_timestamp_inside_a_gap_maps_to_the_next_pieces_start() {
        let regions = [region(1.0, 2.0), region(10.0, 11.0)];
        let splice = Splice::plan(&regions, track(60.0)).unwrap();
        // The first piece runs 0..1.1 s spliced; the gap is 1.1..1.2 s.
        let (start, _) = splice.to_original(1.15, 1.25);
        assert_eq!(start, 10.0);
    }

    #[test]
    fn a_timestamp_past_the_last_piece_maps_to_its_end() {
        let splice = Splice::plan(&[region(45.0, 55.0)], track(120.0)).unwrap();
        // Reachable only through the short-track pad, which decodes past the real audio.
        let (start, end) = splice.to_original(20.0, 25.0);
        assert_eq!(start, 55.0);
        assert_eq!(end, 55.0 + MIN_MAPPED_S);
    }

    /// Rule 4, asserted so it cannot drift: a span crossing a seam is attributed to the region
    /// it started in, not to wherever it happened to end.
    #[test]
    fn a_span_crossing_a_seam_is_clamped_to_the_region_holding_its_start() {
        let regions = [region(1.0, 2.0), region(30.0, 31.0)];
        let splice = Splice::plan(&regions, track(60.0)).unwrap();
        // Starts 0.5 s into the first piece and runs past the seam into the second.
        let (start, end) = splice.to_original(0.5, 1.5);
        assert_eq!(start, 1.5);
        // 2.1 s, the extended end of the first piece -- not 30-something.
        assert_eq!(end, samples(2.1) as f64 / RATE);
    }

    #[test]
    fn mapped_starts_never_go_backwards() {
        let regions: Vec<SpeechRegion> = (0..50)
            .map(|i| {
                let start = f64::from(i) * 7.5 + 0.25;
                region(start, start + 1.75)
            })
            .collect();
        let splice = Splice::plan(&regions, track(500.0)).unwrap();

        let mut previous = 0.0;
        let mut spliced_s = 0.0;
        while spliced_s < splice.spliced_duration_s() + 1.0 {
            let (start, end) = splice.to_original(spliced_s, spliced_s + 0.5);
            assert!(start >= previous, "{start} < {previous} at {spliced_s}");
            assert!(end > start);
            previous = start;
            spliced_s += 0.017;
        }
    }

    /// The test that catches a plan/copy index mismatch, which is the failure that produces a
    /// transcript reading correctly with everything shifted. A ramp rather than audio: every
    /// sample's value is its own index, so a copy from the wrong place is visible by eye.
    #[test]
    fn build_copies_the_planned_windows_and_zeroes_the_gaps() {
        let total = 1_000;
        let ramp: Vec<f32> = (0..total).map(|i| i as f32).collect();
        // Sample offsets: 0.01 s is 160 samples at 16 kHz.
        let regions = [region(0.01, 0.02), region(0.04, 0.05)];
        let splice = Splice::plan(&regions, total).unwrap();

        let spliced = splice.build(&ramp);

        // Piece one: 160..320 extended by 0.1 s, clamped to the next region's start at 640.
        // Piece two: 640..800, unextended.
        assert_eq!(
            splice.pieces[0],
            Piece {
                source_start: 160,
                source_end: 640,
                spliced_start: 0
            }
        );
        assert_eq!(
            splice.pieces[1],
            Piece {
                source_start: 640,
                source_end: 800,
                spliced_start: 480 + samples(GAP_S)
            }
        );

        assert_eq!(spliced.len(), 480 + samples(GAP_S) + 160);
        assert_eq!(&spliced[..480], &ramp[160..640]);
        assert!(spliced[480..480 + samples(GAP_S)].iter().all(|s| *s == 0.0));
        assert_eq!(&spliced[480 + samples(GAP_S)..], &ramp[640..800]);
    }

    #[test]
    #[should_panic(expected = "planned against a different track")]
    fn build_refuses_a_track_it_was_not_planned_against() {
        let splice = Splice::plan(&[region(0.0, 0.01)], 1_000).unwrap();
        splice.build(&vec![0.0; 999]);
    }

    #[test]
    fn seconds_convert_to_whole_samples() {
        assert_eq!(samples(0.1), 1_600);
        assert_eq!(samples(1.0), 16_000);
        assert_eq!(samples(0.0), 0);
        // A negative second cannot index a buffer, so it is zero rather than a wrapped usize.
        assert_eq!(samples(-1.0), 0);
    }
}
