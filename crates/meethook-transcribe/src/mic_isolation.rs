//! Mic isolation: runs the voice isolator over the speech regions of the mic track only.
//!
//! Isolating a whole track costs time proportional to the meeting, nearly all of it spent on
//! silence ([`crate::voice_isolator`] runs at about 0.27x realtime on the CPU). Isolating only
//! the speech regions makes the cost proportional to speech.
//!
//! # What goes in and what comes out
//!
//! The input is the AEC-cleaned 16 kHz mic track (`mic.cleaned.wav`), not the raw `mic.wav`:
//! the raw track still holds the far end that [`crate::cancel_bleed`] removed, the isolator
//! keeps voices, and remote speech would come back into the transcript. Each span is upsampled
//! to the isolator's 48 kHz, isolated, and decimated back. The 16 kHz result is spliced into a
//! *copy* of the cleaned track, inside the speech regions only, so the output has the cleaned
//! track's length and timeline by construction and every sample outside a region is bit for bit
//! the input's. The result is held in memory and handed to recognition; `mic.cleaned.wav` and
//! the mixdown keep the pre-isolation track (see `write_mixdown`).
//!
//! # Context, batching and edges
//!
//! Each region is widened by [`CONTEXT_S`] on both sides before it is run, so the model sees
//! audio around the region and the region's own edge is not the edge of its input. Widened
//! regions that touch, or lie within [`MERGE_GAP_S`] of one another, are merged into one span
//! and each span costs one [`VoiceIsolation::isolate`] call (the engine windows inside that
//! call). Spans never exceed [`MAX_SPAN_S`]: a longer run is closed and a new span started, and a
//! single region longer than that is split with [`CONTEXT_S`] of overlap, the seam written back
//! with full weight (no crossfade toward the original). Memory is thereby bounded by the cap, not
//! the meeting, and the number of calls by the speech duration and gaps, not by how many regions
//! the detector found. Only the original region samples are written back, with a
//! [`FADE_S`] linear crossfade from the original into the isolated audio at each region edge
//! (the weight is 0 at the edge sample itself), so the splice does not click or step.
//!
//! Regions are found here on the cleaned track, and recognition later finds its own on the
//! isolated one; the edges can differ slightly. That is harmless: samples outside a region are
//! untouched either way.

use std::io::Write;

use crate::audio::{TARGET_RATE, resample_between};
use crate::progress::Phase;
use crate::vad::{FireRedVad, SpeechRegion};
use crate::voice_isolator::{SAMPLE_RATE as ISOLATOR_RATE, VoiceIsolator};
use crate::{Result, asr};

/// Seconds of surrounding audio given to the model on each side of a region.
const CONTEXT_S: f64 = 1.0;
/// Widened regions closer than this are merged into one span, so tiny regions do not each pay
/// for a graph invocation (which always runs a whole 8 s window).
const MERGE_GAP_S: f64 = 3.0;
/// Longest span (context included) sent to the isolator in one call. The span is held upsampled
/// to 48 kHz, as the isolator's output, and decimated, so memory is bounded by this rather than by
/// the meeting. A single region longer than this is split into several spans.
const MAX_SPAN_S: f64 = 60.0;
/// Length of the crossfade at each region edge.
const FADE_S: f64 = 0.015;

/// The isolator engine. A seam like [`crate::SpeechToText`], so the stage is tested with a fake
/// and without the 260 MB model.
pub trait VoiceIsolation {
    /// `audio_48k` is mono at 48 kHz; the result has the same length.
    fn isolate(&mut self, audio_48k: &[f32]) -> Result<Vec<f32>>;
}

impl VoiceIsolation for VoiceIsolator {
    fn isolate(&mut self, audio_48k: &[f32]) -> Result<Vec<f32>> {
        VoiceIsolator::isolate(self, audio_48k)
    }
}

/// Finds the speech in 16 kHz mono audio. A seam for the same reason as [`VoiceIsolation`].
pub trait SpeechFinder {
    fn speech_regions(&mut self, audio_16k: &[f32]) -> Result<Vec<SpeechRegion>>;
}

/// The detector speech recognition gates with, at the gate's own settings, so the regions
/// isolated are the regions recognition will look at.
pub struct GateSpeechFinder(pub FireRedVad);

impl SpeechFinder for GateSpeechFinder {
    fn speech_regions(&mut self, audio_16k: &[f32]) -> Result<Vec<SpeechRegion>> {
        self.0.speech_regions(audio_16k, asr::gate_tuning())
    }
}

/// An isolator and the speech finder that tells it where to work.
pub struct IsolationStage {
    pub isolator: Box<dyn VoiceIsolation>,
    pub finder: Box<dyn SpeechFinder>,
}

/// What one isolation pass did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsolationReport {
    /// Speech regions the finder returned.
    pub regions: usize,
    /// Calls made to the isolator.
    pub batches: usize,
    /// Seconds of audio sent through the isolator, context included.
    pub isolated_s: f64,
    /// Seconds in the whole track.
    pub total_s: f64,
}

impl std::fmt::Display for IsolationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "mic isolation: {} regions, {:.0} s of {:.0} s processed in {} batches",
            self.regions, self.isolated_s, self.total_s, self.batches
        )
    }
}

impl IsolationStage {
    /// Isolates the speech in `cleaned_16k`, returning a track of the same length.
    pub fn run(&mut self, cleaned_16k: &[f32]) -> Result<(Vec<f32>, IsolationReport)> {
        isolate_mic(
            self.isolator.as_mut(),
            self.finder.as_mut(),
            cleaned_16k,
            Phase::start("isolating mic"),
        )
    }

    /// [`Self::run`], writing the report line (prefixed with `label`) to `progress`.
    pub(crate) fn run_reporting(
        &mut self,
        cleaned_16k: &[f32],
        label: &str,
        progress: &mut dyn Write,
    ) -> Result<Vec<f32>> {
        let (audio, report) = self.run(cleaned_16k)?;
        writeln!(progress, "{label}  {report}")?;
        Ok(audio)
    }
}

fn to_samples(seconds: f64, len: usize) -> usize {
    ((seconds * f64::from(TARGET_RATE)).round().max(0.0) as usize).min(len)
}

/// A stretch of the track whose isolated audio is written back, and which of its edges are real
/// region edges (crossfaded) rather than seams inside a region that was split to fit
/// [`MAX_SPAN_S`] (written back with full weight, so the seam is not pulled toward the original).
#[derive(Debug, Clone, Copy, PartialEq)]
struct Piece {
    start: usize,
    end: usize,
    fade_start: bool,
    fade_end: bool,
}

/// One [`VoiceIsolation::isolate`] call: the context-widened `(start, end)` samples sent in and
/// the pieces written back from the result.
#[derive(Debug, Clone, PartialEq)]
struct Span {
    start: usize,
    end: usize,
    pieces: Vec<Piece>,
}

/// The spans to run, from `regions` (sorted by start) widened by [`CONTEXT_S`]. Nearby regions
/// merge into one span, but no span exceeds [`MAX_SPAN_S`]; a region too long to fit is split
/// into pieces whose context overlaps the neighbouring span.
fn plan_spans(regions: &[SpeechRegion], len: usize) -> Vec<Span> {
    let context = to_samples(CONTEXT_S, usize::MAX);
    let merge_gap = to_samples(MERGE_GAP_S, usize::MAX);
    let max_span = to_samples(MAX_SPAN_S, usize::MAX);
    let max_owned = max_span.saturating_sub(2 * context).max(1);
    let mut spans: Vec<Span> = Vec::new();
    for region in regions {
        let start = to_samples(region.start_s, len);
        let end = to_samples(region.end_s, len);
        let mut from = start;
        while from < end {
            let to = end.min(from + max_owned);
            let piece = Piece {
                start: from,
                end: to,
                fade_start: from == start,
                fade_end: to == end,
            };
            let wide_start = from.saturating_sub(context);
            let wide_end = (to + context).min(len);
            match spans.last_mut() {
                Some(last)
                    if wide_start < last.end + merge_gap
                        && last.end.max(wide_end) - last.start <= max_span =>
                {
                    last.end = last.end.max(wide_end);
                    last.pieces.push(piece);
                }
                _ => spans.push(Span {
                    start: wide_start,
                    end: wide_end,
                    pieces: vec![piece],
                }),
            }
            from = to;
        }
    }
    spans
}

fn isolate_mic(
    isolator: &mut dyn VoiceIsolation,
    finder: &mut dyn SpeechFinder,
    cleaned_16k: &[f32],
    mut phase: Phase,
) -> Result<(Vec<f32>, IsolationReport)> {
    let regions = finder.speech_regions(cleaned_16k)?;
    let spans = plan_spans(&regions, cleaned_16k.len());
    let mut out = cleaned_16k.to_vec();
    let fade = to_samples(FADE_S, usize::MAX).max(1);
    let mut isolated_samples = 0usize;

    for (index, span) in spans.iter().enumerate() {
        phase.at(index, spans.len());
        let audio = &cleaned_16k[span.start..span.end];
        let up = resample_between(audio, TARGET_RATE, ISOLATOR_RATE)?;
        let voice_48k = isolator.isolate(&up)?;
        let voice = resample_between(&voice_48k, ISOLATOR_RATE, TARGET_RATE)?;
        isolated_samples += audio.len();

        for piece in &span.pieces {
            let (start, end) = (piece.start, piece.end);
            // Half the piece at most on each side, so the two ramps of a short region meet
            // instead of overlapping.
            let ramp = fade.min((end - start).div_ceil(2)) as f32;
            for i in start..end {
                let into = if piece.fade_start {
                    (i - start) as f32 / ramp
                } else {
                    1.0
                };
                let outof = if piece.fade_end {
                    (end - i) as f32 / ramp
                } else {
                    1.0
                };
                let weight = into.min(outof).min(1.0);
                // A resampler that came up a sample short leaves the tail as it was.
                let Some(&iso) = voice.get(i - span.start) else {
                    continue;
                };
                out[i] = cleaned_16k[i] + weight * (iso - cleaned_16k[i]);
            }
        }
    }
    phase.done();

    let report = IsolationReport {
        regions: regions.len(),
        batches: spans.len(),
        isolated_s: isolated_samples as f64 / f64::from(TARGET_RATE),
        total_s: cleaned_16k.len() as f64 / f64::from(TARGET_RATE),
    };
    Ok((out, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountingIsolator {
        gain: f32,
        calls: usize,
    }

    impl VoiceIsolation for CountingIsolator {
        fn isolate(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
            self.calls += 1;
            Ok(audio.iter().map(|s| s * self.gain).collect())
        }
    }

    struct FixedRegions(Vec<SpeechRegion>);

    impl SpeechFinder for FixedRegions {
        fn speech_regions(&mut self, _: &[f32]) -> Result<Vec<SpeechRegion>> {
            Ok(self.0.clone())
        }
    }

    fn region(start_s: f64, end_s: f64) -> SpeechRegion {
        SpeechRegion { start_s, end_s }
    }

    /// A 220 Hz tone with a little deterministic noise.
    fn signal(len: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / TARGET_RATE as f32;
                0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.02 * ((i * 7919 % 101) as f32 / 101.0 - 0.5)
            })
            .collect()
    }

    fn run(
        gain: f32,
        regions: Vec<SpeechRegion>,
        audio: &[f32],
    ) -> (Vec<f32>, IsolationReport, usize) {
        let mut isolator = CountingIsolator { gain, calls: 0 };
        let mut finder = FixedRegions(regions);
        let (out, report) =
            isolate_mic(&mut isolator, &mut finder, audio, Phase::start("t")).unwrap();
        (out, report, isolator.calls)
    }

    #[test]
    fn only_samples_inside_regions_change() {
        let audio = signal(20 * 16_000);
        let (out, report, _) = run(0.5, vec![region(5.0, 8.0), region(14.0, 15.0)], &audio);
        let inside = |i: usize| {
            let s = i as f64 / 16_000.0;
            (5.0..8.0).contains(&s) || (14.0..15.0).contains(&s)
        };
        for i in 0..audio.len() {
            if !inside(i) {
                assert_eq!(out[i].to_bits(), audio[i].to_bits(), "sample {i}");
            }
        }
        let mid = 6 * 16_000;
        assert!(
            (out[mid] - 0.5 * audio[mid]).abs() < 0.01,
            "interior is isolated"
        );
        assert_eq!(report.regions, 2);
    }

    #[test]
    fn region_edges_do_not_step() {
        // Gain zero is the harshest isolator: without a crossfade the edge would jump by the
        // signal's whole amplitude.
        let audio = signal(10 * 16_000);
        let (out, _, _) = run(0.0, vec![region(3.0, 6.0)], &audio);
        let worst = out
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        let natural = audio
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst <= natural * 1.5,
            "largest step {worst} vs the signal's own {natural}"
        );
    }

    #[test]
    fn length_is_preserved_at_awkward_lengths_and_the_track_ends() {
        for len in [16_000 * 10 + 1, 16_000 * 10 + 2, 16_000 * 7 + 5] {
            let audio = signal(len);
            let end = len as f64 / 16_000.0;
            let (out, _, _) = run(0.5, vec![region(0.0, 2.0), region(end - 2.0, end)], &audio);
            assert_eq!(out.len(), audio.len());
        }
    }

    #[test]
    fn many_close_regions_cost_few_invocations() {
        let audio = signal(60 * 16_000);
        let regions: Vec<_> = (0..200)
            .map(|k| region(1.0 + k as f64 * 0.2, 1.1 + k as f64 * 0.2))
            .collect();
        let (_, report, calls) = run(1.0, regions, &audio);
        assert_eq!(report.regions, 200);
        assert_eq!(calls, 1);
        assert_eq!(report.batches, 1);
    }

    #[test]
    fn continuous_speech_is_capped_into_several_calls() {
        let audio = signal(600 * 16_000);
        let regions = vec![region(0.0, 600.0)];
        let spans = plan_spans(&regions, audio.len());
        let cap = (MAX_SPAN_S * 16_000.0) as usize;
        assert!(spans.len() > 1);
        assert!(spans.iter().all(|s| s.end - s.start <= cap));
        // Many back-to-back regions merge, but still respect the cap.
        let many: Vec<_> = (0..600).map(|k| region(k as f64, k as f64 + 0.9)).collect();
        let spans = plan_spans(&many, audio.len());
        assert!(spans.len() > 1);
        assert!(spans.iter().all(|s| s.end - s.start <= cap));
        let (_, report, calls) = run(1.0, many, &audio);
        assert!(calls > 1);
        assert_eq!(report.batches, calls);
    }

    #[test]
    fn a_forced_split_leaves_no_seam() {
        // Identity isolator: output must equal the input wherever a region covers it, including
        // across the seams of a region split to fit the cap.
        let audio = signal(150 * 16_000);
        let (out, _, calls) = run(1.0, vec![region(10.0, 140.0)], &audio);
        assert!(calls > 1);
        let worst = (10 * 16_000..140 * 16_000)
            .map(|i| (out[i] - audio[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 0.02, "worst deviation {worst}");
    }

    #[test]
    fn distant_regions_are_separate_invocations() {
        let audio = signal(60 * 16_000);
        let (_, _, calls) = run(1.0, vec![region(5.0, 6.0), region(30.0, 31.0)], &audio);
        assert_eq!(calls, 2);
    }

    #[test]
    fn no_regions_means_no_isolation_and_an_unchanged_track() {
        let audio = signal(5 * 16_000);
        let (out, report, calls) = run(0.5, vec![], &audio);
        assert_eq!(calls, 0);
        assert_eq!(out, audio);
        assert_eq!(report.isolated_s, 0.0);
    }

    #[test]
    fn the_round_trip_does_not_shift_timing() {
        // An identity isolator: whatever moves, the rate conversion moved.
        let mut audio = vec![0.0f32; 8 * 16_000];
        let onset = 4 * 16_000;
        for (k, sample) in audio[onset..onset + 8000].iter_mut().enumerate() {
            *sample = 0.8 * (k as f32 * 0.3).sin().signum();
        }
        let (out, _, _) = run(1.0, vec![region(3.0, 6.0)], &audio);
        let before = crate::audio::fixtures::first_onset(&audio, 0.4).unwrap();
        let after = crate::audio::fixtures::first_onset(&out, 0.4).unwrap();
        assert!(before.abs_diff(after) <= 1, "{before} -> {after}");
    }

    #[test]
    fn report_text_names_regions_and_seconds() {
        let audio = signal(20 * 16_000);
        let (_, report, _) = run(1.0, vec![region(5.0, 8.0)], &audio);
        let text = report.to_string();
        assert!(text.contains("1 regions"), "{text}");
        assert!(text.contains("of 20 s processed in 1 batches"), "{text}");
    }
}
