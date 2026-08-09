//! Finding out when someone -- anyone -- is speaking on the speaker track.
//!
//! pyannote segmentation 3.0 is a ten-second model. It looks at one window of audio at a
//! time and reports, for every ~17 ms frame in that window, which of *up to three local
//! speakers* is talking. "Local" is the whole difficulty: the network never sees more than
//! ten seconds, so speaker 0 in one window has no relationship at all to speaker 0 in the
//! next. Turning those indices into people who persist across a meeting needs voice
//! embeddings and is the clustering slice's job. What this module owes it is turns that sit
//! in the right place on the session timeline and are correctly grouped *within* each
//! window.
//!
//! Nothing outside this module should ever see a logit, a frame, or a window boundary.

use ort::session::Session;
use ort::value::TensorRef;

use crate::audio::TARGET_RATE;
use crate::{Error, Result};

/// One stretch of speech by one window-local speaker, timed from the start of the track.
///
/// `local_speaker` is meaningless on its own and only comparable to another turn from the
/// *same* `window`. Two turns from one window with different `local_speaker` values are
/// definitely different people, which is a constraint worth carrying into clustering; two
/// turns from different windows carry no information about each other whatsoever.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalTurn {
    /// Seconds from the start of the track handed to [`segment_speaker_track`].
    pub start_s: f64,
    pub end_s: f64,
    /// Which inference window produced this turn. Windows do not overlap and are ten
    /// seconds long, so this is also the turn's start divided by ten.
    pub window: usize,
    /// The model's local speaker index, 0..3. Comparable only within one `window`.
    pub local_speaker: usize,
}

/// The window pyannote segmentation 3.0 was trained on. Ten seconds at 16 kHz.
///
/// The graph's sample axis is symbolic, so a different length would run -- and would be
/// out of distribution, because the receptive field and the training regime are both built
/// around this number.
const WINDOW_SAMPLES: usize = 10 * TARGET_RATE as usize;
const WINDOW_SECONDS: f64 = WINDOW_SAMPLES as f64 / TARGET_RATE as f64;

/// Windows are laid end to end: the step equals the window.
///
/// The alternative -- overlapping windows -- reads as the more careful choice and is not.
/// Two overlapping windows that disagree about an instant cannot be reconciled by a
/// tie-break, because their speaker indices are permutations of each other with no known
/// correspondence; establishing that correspondence *is* the embed-and-cluster problem of
/// the next slice. Butting the windows up against each other means every instant is decided
/// exactly once and there is no reconciliation rule to get wrong.
///
/// The price is that a turn straddling a boundary is emitted as two turns. Clustering
/// recovers from that on its own -- both halves are the same voice -- and [`LocalTurn`]'s
/// window index says which pairs abut.
const WINDOW_STEP_SAMPLES: usize = WINDOW_SAMPLES;

/// The model reasons about at most three concurrent speakers inside one window.
const LOCAL_SPEAKERS: usize = 3;

/// Every subset of the three local speakers with 0, 1 or 2 members, in the order the
/// checkpoint's output columns are in.
///
/// **This ordering is a fact about the checkpoint, not a convention**, and it is the one
/// thing here that cannot be inferred from the surrounding code. It comes from
/// pyannote.audio's `Powerset.build_mapping`, which enumerates subsets by increasing size
/// and, within a size, in lexicographic order: the empty set, then the three singletons,
/// then the three pairs.
///
/// The seven columns are therefore *classes*, not speakers and not independent
/// probabilities. A frame is decoded by taking the argmax across them and expanding the
/// winner through this table. Thresholding each column on its own, or reading the column
/// index as a speaker index, produces output that looks plausible -- roughly the right
/// amount of speech in roughly the right places -- and is wrong.
const POWERSET: [&[usize]; 7] = [
    &[],     // 0: nobody
    &[0],    // 1
    &[1],    // 2
    &[2],    // 3
    &[0, 1], // 4
    &[0, 2], // 5
    &[1, 2], // 6
];

/// A silence this short inside one speaker's run does not end their turn.
///
/// Speech is full of gaps the model dutifully reports: stop consonants, breaths, the beat
/// between words. Left alone they shatter a sentence into a dozen fragments, each too short
/// to embed. A quarter of a second sits above ordinary inter-word pauses and below the
/// pause that means someone has actually stopped talking.
///
/// Turns shorter than this are *not* dropped. Deciding that a 30 ms fleck is too short to
/// be worth a voice fingerprint is a judgement about embeddings, and it belongs to the code
/// that computes them, not to the code that reports what the model said.
const MAX_GAP_IN_TURN_S: f64 = 0.25;

/// Runs the segmentation model across a whole track and returns its speech turns.
///
/// `samples_16k` must be 16 kHz mono `f32` -- the rate [`crate::audio::read_track_16k_mono`]
/// already produces, and the rate this model expects, so there is no resampling here.
///
/// An empty track yields no turns. A track shorter than one window is zero-padded up to one
/// and run anyway, because a six-second meeting is a normal meeting for this tool; the
/// resulting turns are clipped to the real end of the audio so padding cannot invent speech
/// that was never recorded.
pub fn segment_speaker_track(samples_16k: &[f32], session: &mut Session) -> Result<Vec<LocalTurn>> {
    if samples_16k.is_empty() {
        return Ok(Vec::new());
    }
    let track_end_s = samples_16k.len() as f64 / TARGET_RATE as f64;

    // Reused across windows: one 640 KB buffer rather than one per ten seconds of meeting.
    let mut window = vec![0.0f32; WINDOW_SAMPLES];
    let mut turns = Vec::new();

    for (index, offset) in (0..samples_16k.len())
        .step_by(WINDOW_STEP_SAMPLES)
        .enumerate()
    {
        let present = &samples_16k[offset..samples_16k.len().min(offset + WINDOW_SAMPLES)];
        window[..present.len()].copy_from_slice(present);
        window[present.len()..].fill(0.0);

        let input = TensorRef::from_array_view(([1usize, 1, WINDOW_SAMPLES], &window[..]))
            .map_err(inference_failed)?;
        let outputs = session.run(ort::inputs![input]).map_err(inference_failed)?;
        let (shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(inference_failed)?;

        // Read the frame count off the tensor rather than hardcoding it. A re-export with a
        // different receptive field would otherwise shift every timestamp in the meeting,
        // silently and by a plausible-looking amount.
        let num_frames = frame_count(shape)?;
        decode_powerset(
            logits,
            num_frames,
            WINDOW_SECONDS / num_frames as f64,
            offset as f64 / TARGET_RATE as f64,
            index,
            &mut turns,
        );
    }

    // The last window was padded with silence, and silence produces no turns -- but a turn
    // that was still open when the real audio ran out would otherwise be reported as
    // running to the end of the padding.
    turns.retain(|turn| turn.start_s < track_end_s);
    for turn in &mut turns {
        turn.end_s = turn.end_s.min(track_end_s);
    }
    Ok(turns)
}

/// Validates the logits tensor against the graph contract and returns its frame count.
fn frame_count(shape: &[i64]) -> Result<usize> {
    match shape {
        [1, frames, 7] if *frames > 0 => Ok(*frames as usize),
        _ => Err(Error::Segmentation(format!(
            "the segmentation model returned logits of shape {shape:?}, not [1, num_frames, 7]"
        ))),
    }
}

fn inference_failed(source: ort::Error) -> Error {
    Error::Segmentation(source.to_string())
}

/// Turns one window's powerset logits into turns on the session timeline.
///
/// `logits` is `num_frames * 7` values in row-major order, `frame_s` the duration one frame
/// covers, and `window_start_s` where this window begins in the track. Turns are appended to
/// `into` in the order they start.
///
/// Frames are treated as abutting tiles: frame `f` covers `[f * frame_s, (f + 1) * frame_s)`.
/// The model's frames really have overlapping receptive fields centred on those positions,
/// so a boundary here can be off by up to half a frame -- under 10 ms, well below both the
/// gap threshold above and the granularity anything downstream cares about.
///
/// Pure, and deliberately so: everything that can be got wrong about powerset decoding is
/// in this function, and it can be exercised in microseconds with no model, no audio and no
/// ONNX Runtime.
fn decode_powerset(
    logits: &[f32],
    num_frames: usize,
    frame_s: f64,
    window_start_s: f64,
    window: usize,
    into: &mut Vec<LocalTurn>,
) {
    let appended_from = into.len();
    // The turn currently open for each local speaker, if any.
    let mut open: [Option<LocalTurn>; LOCAL_SPEAKERS] = [None; LOCAL_SPEAKERS];

    for frame in 0..num_frames {
        let start_s = window_start_s + frame as f64 * frame_s;
        let end_s = start_s + frame_s;
        let class = argmax(&logits[frame * POWERSET.len()..(frame + 1) * POWERSET.len()]);

        for &speaker in POWERSET[class] {
            match &mut open[speaker] {
                // Same run, or a gap short enough not to count as one. Extending covers
                // both, which is why contiguous frames collapse rather than producing one
                // turn per frame.
                Some(turn) if start_s - turn.end_s <= MAX_GAP_IN_TURN_S => turn.end_s = end_s,
                Some(turn) => {
                    into.push(*turn);
                    *turn = LocalTurn {
                        start_s,
                        end_s,
                        window,
                        local_speaker: speaker,
                    };
                }
                slot @ None => {
                    *slot = Some(LocalTurn {
                        start_s,
                        end_s,
                        window,
                        local_speaker: speaker,
                    });
                }
            }
        }
    }

    into.extend(open.into_iter().flatten());
    // Turns close in speaker order but are wanted in time order. Only this window's turns
    // are sorted -- earlier windows are all earlier in time already, and re-sorting the
    // whole meeting once per ten seconds of it would be quadratic for no gain.
    into[appended_from..].sort_by(|a, b| {
        a.start_s
            .total_cmp(&b.start_s)
            .then(a.local_speaker.cmp(&b.local_speaker))
    });
}

/// Index of the largest value, first one winning a tie.
///
/// NaN never wins, which makes a graph that has gone numerically wrong decode as silence
/// rather than as a panic or as a different answer on every run.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for (i, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds logits whose argmax per frame is the given class.
    ///
    /// The winning class gets 1.0 and the rest 0.0 -- not softmax-shaped, which is exactly
    /// the point: decoding must depend on which column is largest and on nothing else.
    fn logits(classes: &[usize]) -> Vec<f32> {
        let mut out = vec![0.0; classes.len() * 7];
        for (frame, class) in classes.iter().enumerate() {
            out[frame * 7 + class] = 1.0;
        }
        out
    }

    /// Decodes `classes` at one frame per second, from the start of the track.
    fn decode(classes: &[usize]) -> Vec<LocalTurn> {
        decode_at(classes, 1.0, 0.0, 0)
    }

    fn decode_at(
        classes: &[usize],
        frame_s: f64,
        window_start_s: f64,
        window: usize,
    ) -> Vec<LocalTurn> {
        let mut turns = Vec::new();
        decode_powerset(
            &logits(classes),
            classes.len(),
            frame_s,
            window_start_s,
            window,
            &mut turns,
        );
        turns
    }

    /// Turns as (speaker, start, end), with times rounded to the microsecond.
    ///
    /// Frame boundaries are sums of a fractional frame duration, so they land a few ulps
    /// off the round number they mean; a microsecond is four orders of magnitude finer than
    /// anything here is claiming to resolve.
    fn spans(turns: &[LocalTurn]) -> Vec<(usize, f64, f64)> {
        let round = |s: f64| (s * 1e6).round() / 1e6;
        turns
            .iter()
            .map(|t| (t.local_speaker, round(t.start_s), round(t.end_s)))
            .collect()
    }

    /// The table is the one fact in this module that cannot be re-derived from the code
    /// around it, so it is asserted against the rule that generated it rather than trusted.
    #[test]
    fn the_powerset_table_enumerates_subsets_by_size_then_lexicographically() {
        let mut expected: Vec<Vec<usize>> = vec![vec![]];
        expected.extend((0..LOCAL_SPEAKERS).map(|a| vec![a]));
        for a in 0..LOCAL_SPEAKERS {
            expected.extend((a + 1..LOCAL_SPEAKERS).map(|b| vec![a, b]));
        }

        let table: Vec<Vec<usize>> = POWERSET.iter().map(|set| set.to_vec()).collect();
        assert_eq!(table, expected);
    }

    #[test]
    fn one_active_frame_becomes_one_turn_for_that_speaker() {
        assert_eq!(spans(&decode(&[2])), [(1, 0.0, 1.0)]);
    }

    /// The failure this catches is reading the class index as a speaker index: class 4 would
    /// then decode as "speaker 4", which does not exist, instead of as speakers 0 and 1 at
    /// once.
    #[test]
    fn a_frame_with_two_simultaneous_speakers_yields_a_turn_for_each() {
        assert_eq!(spans(&decode(&[4])), [(0, 0.0, 1.0), (1, 0.0, 1.0)]);
        assert_eq!(spans(&decode(&[5])), [(0, 0.0, 1.0), (2, 0.0, 1.0)]);
        assert_eq!(spans(&decode(&[6])), [(1, 0.0, 1.0), (2, 0.0, 1.0)]);
    }

    #[test]
    fn a_frame_with_nobody_speaking_yields_no_turn() {
        assert!(decode(&[0]).is_empty());
        assert!(decode(&[0, 0, 0]).is_empty());
    }

    #[test]
    fn contiguous_frames_for_one_speaker_collapse_into_a_single_turn() {
        assert_eq!(spans(&decode(&[1, 1, 1, 1])), [(0, 0.0, 4.0)]);
    }

    /// Silence shorter than the threshold is bridged; silence longer than it is a real
    /// boundary. Both directions matter -- bridging everything would merge a whole meeting
    /// into one turn per speaker.
    #[test]
    fn a_gap_shorter_than_the_threshold_does_not_split_a_turn() {
        // 0.1 s frames: one silent frame is a 0.1 s gap, under the 0.25 s threshold.
        assert_eq!(spans(&decode_at(&[1, 0, 1], 0.1, 0.0, 0)), [(0, 0.0, 0.3)]);
        // Five silent frames is 0.5 s, over it.
        assert_eq!(
            spans(&decode_at(&[1, 0, 0, 0, 0, 0, 1], 0.1, 0.0, 0)),
            [(0, 0.0, 0.1), (0, 0.6, 0.7)]
        );
    }

    /// Two speakers alternating must not be merged into one another, however close their
    /// runs are: a turn is tracked per speaker, not as one cursor over the window.
    ///
    /// Speaker 0's two frames *do* join across speaker 1's, because 0.1 s is under the gap
    /// threshold and somebody else talking during a pause does not make the pause longer.
    /// That is the same rule as any other short gap, asserted where it looks most
    /// surprising.
    #[test]
    fn one_speaker_talking_between_another_speaker_s_frames_keeps_the_turns_apart() {
        let turns = decode_at(&[1, 2, 1], 0.1, 0.0, 0);
        assert_eq!(spans(&turns), [(0, 0.0, 0.3), (1, 0.1, 0.2)]);
    }

    /// The window offset is what puts a turn on the session timeline rather than ten
    /// seconds early; the window index is what tells clustering which turns share a
    /// permutation.
    #[test]
    fn turns_are_placed_on_the_track_timeline_and_tagged_with_their_window() {
        let turns = decode_at(&[0, 1], 0.5, 10.0, 1);
        assert_eq!(spans(&turns), [(0, 10.5, 11.0)]);
        assert_eq!(turns[0].window, 1);
    }

    /// Overlapping runs come back in start order, not in speaker order, so a caller can
    /// walk them against a transcript without re-sorting.
    #[test]
    fn turns_are_returned_in_start_order() {
        // Speaker 1 alone, then both, then speaker 0 alone: speaker 0 starts later.
        let turns = decode_at(&[2, 4, 1], 1.0, 0.0, 0);
        assert_eq!(spans(&turns), [(1, 0.0, 2.0), (0, 1.0, 3.0)]);
    }

    #[test]
    fn a_malformed_logits_tensor_is_an_error_rather_than_a_misread() {
        assert!(frame_count(&[1, 589, 7]).is_ok());
        for bad in [&[1, 589, 3][..], &[1, 0, 7][..], &[589, 7][..]] {
            let message = frame_count(bad).unwrap_err().to_string();
            assert!(message.contains("[1, num_frames, 7]"), "{message}");
        }
    }

    /// Opens the segmentation weights, or returns `None` if they are not installed.
    ///
    /// Same bargain as the graph-contract test in `onnx.rs`: `cargo test` never reaches for
    /// a 6 MB download, so the suite passes on a machine that has never run the tool.
    fn segmentation_session() -> Option<Session> {
        let root = match std::env::var_os("MEETHOOK_ROOT") {
            Some(root) => std::path::PathBuf::from(root),
            None => std::env::home_dir()?.join("meethook"),
        };
        let path = root
            .join("models")
            .join(crate::SEGMENTATION_MODEL.file_name);
        if !path.is_file() {
            eprintln!(
                "skipping: {} is not installed; \
                 run `cargo run --example fetch-onnx-models` to fetch it",
                path.display()
            );
            return None;
        }
        Some(
            crate::open_session(&path)
                .expect("an installed model must load")
                .session,
        )
    }

    /// A hummed vowel: a 130 Hz glottal buzz through three fixed formants, gated on and off
    /// in syllable-length bursts. Not speech, but harmonic and modulated in the way speech
    /// is, which is what the segmentation model responds to.
    fn vowel(seconds: f64) -> Vec<f32> {
        const F0: f64 = 130.0;
        const FORMANTS: [f64; 3] = [730.0, 1090.0, 2440.0];
        let n = (seconds * TARGET_RATE as f64) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / TARGET_RATE as f64;
                // 4 Hz on/off gating, roughly a syllable rate, with silence between bursts.
                let gate = if (t * 4.0).fract() < 0.7 { 1.0 } else { 0.0 };
                let buzz: f64 = (1..=40)
                    .map(|h| {
                        let hz = F0 * h as f64;
                        let gain: f64 = FORMANTS
                            .iter()
                            .map(|f| 1.0 / (1.0 + ((hz - f) / 100.0).powi(2)))
                            .sum();
                        gain * (t * hz * std::f64::consts::TAU).sin()
                    })
                    .sum();
                (0.2 * gate * buzz) as f32
            })
            .collect()
    }

    /// The whole sliding loop, on audio shorter than one window.
    ///
    /// A six-second meeting is the normal case for this tool, not an edge case, so the
    /// short track has to survive zero-padding *and* the turns have to stop where the
    /// recording does rather than where the padding does.
    #[test]
    fn a_track_shorter_than_one_window_yields_turns_that_end_with_the_audio() {
        let Some(mut session) = segmentation_session() else {
            return;
        };
        let seconds = 6.0;

        let turns = segment_speaker_track(&vowel(seconds), &mut session)
            .expect("a short track is padded, not rejected");

        assert!(
            !turns.is_empty(),
            "six seconds of voiced audio produced no turns"
        );
        for turn in &turns {
            assert!(turn.start_s >= 0.0, "{turn:?}");
            assert!(turn.start_s < turn.end_s, "{turn:?}");
            assert!(
                turn.end_s <= seconds,
                "{turn:?} runs past the end of a {seconds} s track"
            );
            assert_eq!(turn.window, 0, "one window covers a short track");
        }
    }

    /// Turns from later windows have to be offset onto the track timeline. An off-by-one
    /// window here is 10 seconds of error and passes every synthetic-logit test above.
    #[test]
    fn later_windows_land_later_on_the_timeline() {
        let Some(mut session) = segmentation_session() else {
            return;
        };
        // Silence, then the same voiced audio, either side of the 10 s window boundary.
        let mut audio = vec![0.0f32; 12 * TARGET_RATE as usize];
        audio.extend(vowel(4.0));

        let turns = segment_speaker_track(&audio, &mut session).expect("inference");

        let first = turns.first().expect("voiced audio produced no turns");
        assert_eq!(
            first.window, 1,
            "the speech starts inside the second window"
        );
        assert!(
            (11.0..13.0).contains(&first.start_s),
            "speech starting at 12 s was reported at {}",
            first.start_s
        );
        assert!(turns.iter().all(|t| t.end_s <= 16.0));
    }

    #[test]
    fn an_empty_track_yields_no_turns_rather_than_an_error() {
        let Some(mut session) = segmentation_session() else {
            return;
        };
        assert!(segment_speaker_track(&[], &mut session).unwrap().is_empty());
    }
}
