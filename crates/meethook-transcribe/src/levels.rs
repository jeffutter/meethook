//! Whether a recorded track actually contains audio, answered without any model.
//!
//! The recording sitting is the scarce resource: it needs a person, usually a second person,
//! and sometimes a scheduled call. Discovering afterwards that one track was dead wastes all
//! of it. So the question "did both tracks capture something" has to be answerable in the
//! seconds after a call ends -- before models are fetched, before anything is transcribed --
//! and that is what this measures.
//!
//! Peak amplitude alone gets the answer wrong, which is the reason this is more than a
//! `max()`. A track carrying nothing but two UI chimes peaks around 0.57, squarely in speech
//! territory, while sitting at digital silence for 99% of its length. What separates the two
//! is [`LevelSummary::above_fraction`] -- how much of the track is above the floor at all --
//! and [`LevelSummary::longest_run_s`], which says whether there is one usable stretch or a
//! scattering of clicks.

/// The amplitude a sample has to exceed to count as signal, on the ±1.0 float scale.
///
/// This is drawn to exclude a track that is *electrically* rather than *acoustically* quiet:
/// dither, DC offset, and idle-channel noise from a capture device that is running but has
/// nothing in front of it. At -80 dBFS it sits roughly 60 dB below conversational speech and
/// well below room tone, so it does not exclude quiet talking -- a whispered word still
/// crosses it comfortably. It is deliberately not a voice-activity threshold; anything that
/// tries to judge speech from loudness belongs in the segmentation model, not here.
///
/// The value separated every recorded session cleanly when the tracks under
/// `~/meethook/sessions` were surveyed by hand: dead tracks measured a fraction of exactly
/// 0.000 above it, chime-only tracks 0.001-0.022, and a real two-person conversation 0.998.
pub const SILENCE_FLOOR: f32 = 1e-4;

/// Gaps shorter than this do not end a run of signal.
///
/// Speech crosses zero constantly, so a run of strictly consecutive above-floor samples
/// measures a single glottal pulse rather than an utterance -- a 10-second conversation
/// would report a longest run of a few milliseconds, which is the same number a click
/// produces. Bridging short gaps makes the figure mean "an unbroken stretch of activity",
/// which is the thing being asked about. 50 ms is shorter than any inter-word pause and
/// longer than the sub-millisecond gaps inside voiced speech.
pub const RUN_BRIDGE_S: f64 = 0.05;

/// What a track's samples look like, in the four numbers that distinguish a live recording
/// from a dead one.
///
/// Rate-carrying rather than rate-free on purpose: every derived figure a person reads is in
/// seconds, and a summary that made the caller supply the rate again could be printed
/// against the wrong one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelSummary {
    /// Frames per second the track was captured at. Zero is tolerated -- a corrupt header is
    /// one of the failures this exists to name -- and makes every duration read as 0.0.
    pub sample_rate: u32,
    /// Total samples measured.
    pub samples: usize,
    /// Largest absolute sample. 0.0 means digital silence, not "quiet".
    pub peak: f32,
    /// Samples strictly above [`SILENCE_FLOOR`].
    pub above_floor: usize,
    /// Longest stretch of signal in samples, with gaps under [`RUN_BRIDGE_S`] bridged.
    pub longest_run: usize,
}

impl LevelSummary {
    /// Measures a track. Handles an empty slice and a zero sample rate without dividing by
    /// either, because both are shapes that occur in real recordings.
    pub fn measure(samples: &[f32], sample_rate: u32) -> Self {
        let bridge = (RUN_BRIDGE_S * f64::from(sample_rate)) as usize;

        let mut peak = 0.0f32;
        let mut above_floor = 0;
        let mut longest_run = 0usize;
        // The open run is `run_start..=last_above`; it stays open across gaps of `bridge`
        // samples or fewer, and is committed to `longest_run` when a longer gap closes it.
        let mut run_start: Option<usize> = None;
        let mut last_above = 0usize;

        for (index, sample) in samples.iter().enumerate() {
            let magnitude = sample.abs();
            peak = peak.max(magnitude);
            if magnitude <= SILENCE_FLOOR {
                continue;
            }
            above_floor += 1;
            match run_start {
                None => run_start = Some(index),
                Some(start) => {
                    if index - last_above - 1 > bridge {
                        longest_run = longest_run.max(last_above + 1 - start);
                        run_start = Some(index);
                    }
                }
            }
            last_above = index;
        }
        if let Some(start) = run_start {
            longest_run = longest_run.max(last_above + 1 - start);
        }

        Self {
            sample_rate,
            samples: samples.len(),
            peak,
            above_floor,
            longest_run,
        }
    }

    /// Track length in seconds. 0.0 for an empty track or an unreadable sample rate.
    pub fn duration_s(&self) -> f64 {
        self.seconds(self.samples)
    }

    /// Share of samples above [`SILENCE_FLOOR`], in 0.0..=1.0. An empty track is 0.0 rather
    /// than a division by zero.
    pub fn above_fraction(&self) -> f64 {
        if self.samples == 0 {
            return 0.0;
        }
        self.above_floor as f64 / self.samples as f64
    }

    /// Longest unbroken stretch of signal, in seconds.
    pub fn longest_run_s(&self) -> f64 {
        self.seconds(self.longest_run)
    }

    /// Peak in dBFS. Digital silence is `f64::NEG_INFINITY`; callers print that as a word
    /// rather than letting `-inf` reach a column of numbers.
    pub fn peak_dbfs(&self) -> f64 {
        if self.peak <= 0.0 {
            return f64::NEG_INFINITY;
        }
        20.0 * f64::from(self.peak).log10()
    }

    fn seconds(&self, samples: usize) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        samples as f64 / f64::from(self.sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pair of shapes the whole thing exists to tell apart, with the peak deliberately
    /// identical: only the fraction and the run length separate them.
    #[test]
    fn isolated_blips_and_sustained_speech_differ_in_fraction_not_peak() {
        let mut blips = vec![0.0f32; 16_000];
        blips[100] = 0.57;
        blips[8_000] = 0.57;
        let sustained: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 / 16_000.0 * 440.0 * std::f32::consts::TAU).sin() * 0.57)
            .collect();

        let blips = LevelSummary::measure(&blips, 16_000);
        let sustained = LevelSummary::measure(&sustained, 16_000);

        assert!((blips.peak - sustained.peak).abs() < 1e-6);
        assert!(blips.above_fraction() < 0.001, "{blips:?}");
        assert!(sustained.above_fraction() > 0.9, "{sustained:?}");
        assert!(blips.longest_run_s() < 0.06, "{blips:?}");
        assert!(sustained.longest_run_s() > 0.9, "{sustained:?}");
    }

    #[test]
    fn digital_silence_measures_zero_everywhere() {
        let summary = LevelSummary::measure(&[0.0; 4_800], 48_000);

        assert_eq!(summary.peak, 0.0);
        assert_eq!(summary.above_floor, 0);
        assert_eq!(summary.above_fraction(), 0.0);
        assert_eq!(summary.longest_run, 0);
        assert!((summary.duration_s() - 0.1).abs() < 1e-9);
        assert_eq!(summary.peak_dbfs(), f64::NEG_INFINITY);
    }

    #[test]
    fn an_empty_track_reports_nothing_rather_than_dividing_by_zero() {
        let summary = LevelSummary::measure(&[], 48_000);

        assert_eq!(summary.samples, 0);
        assert_eq!(summary.duration_s(), 0.0);
        assert_eq!(summary.above_fraction(), 0.0);
        assert_eq!(summary.longest_run_s(), 0.0);
        assert_eq!(summary.peak_dbfs(), f64::NEG_INFINITY);
    }

    #[test]
    fn a_zero_sample_rate_yields_zero_durations_rather_than_infinity() {
        let summary = LevelSummary::measure(&[0.5; 16], 0);

        assert_eq!(summary.samples, 16);
        assert_eq!(summary.duration_s(), 0.0);
        assert_eq!(summary.longest_run_s(), 0.0);
        assert!((summary.above_fraction() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_single_spike_is_one_sample_of_signal() {
        let mut samples = vec![0.0f32; 1_000];
        samples[500] = -0.8;

        let summary = LevelSummary::measure(&samples, 1_000);

        assert!((summary.peak - 0.8).abs() < 1e-6);
        assert_eq!(summary.above_floor, 1);
        assert_eq!(summary.longest_run, 1);
    }

    /// Two bursts a long way apart must not be reported as one run; two bursts a hair apart
    /// must. Both are the bridging rule, from either side.
    #[test]
    fn runs_bridge_short_gaps_and_break_on_long_ones() {
        // 1000 Hz, so the 50 ms bridge is 50 samples.
        let mut near = vec![0.0f32; 300];
        near[0..10].fill(0.5);
        near[50..60].fill(0.5);
        let near = LevelSummary::measure(&near, 1_000);
        assert_eq!(near.longest_run, 60);

        let mut far = vec![0.0f32; 300];
        far[0..10].fill(0.5);
        far[200..215].fill(0.5);
        let far = LevelSummary::measure(&far, 1_000);
        assert_eq!(far.longest_run, 15);
    }

    /// The floor is a threshold on magnitude, so a quiet negative sample counts and a
    /// sample at the floor does not.
    #[test]
    fn the_floor_is_applied_to_magnitude_and_is_exclusive() {
        let summary = LevelSummary::measure(&[SILENCE_FLOOR, -SILENCE_FLOOR * 2.0], 1_000);

        assert_eq!(summary.above_floor, 1);
    }
}
