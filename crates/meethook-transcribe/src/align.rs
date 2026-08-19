//! Measuring how far the speaker track has to be shifted so its content lines up with the
//! bleed the microphone actually heard.
//!
//! Reference-based echo cancellation cannot start without this number, and `session.json`
//! cannot supply it. Both capture APIs are internally honest -- each stamps timestamps that
//! stay linear in its own sample count to under 0.1 ms over 35 s -- but the stored offset is
//! missing two terms that no API exposes: a fixed ~16 ms bias originating inside the
//! ScreenCaptureKit audio pipeline, and output latency, which SCStream presentation
//! timestamps omit entirely. Measured click residuals against the stored offset ranged from
//! about -13 ms on a USB output path to +410 ms on Bluetooth A2DP, and a CoreAudio probe
//! reading device latency, stream latency and safety offset reported 186.688 ms for a path
//! measured at ~426 ms. The audio callback in `meethook-record`'s `speaker` module carries
//! the full account. So the stored offset is the coarse starting guess for the search and
//! nothing more; the residual is measured here, from the two signals.
//!
//! The sign matters. The stored offset can make the microphone appear to hear a sound
//! *before* the speaker emitted it, so the search covers negative lags as well as the
//! Bluetooth-scale positive ones.
//!
//! "Cannot measure" is a first-class outcome rather than a failure. A headphones session has
//! no bleed to correlate against, and AEC3 given a wrong reference is reported to flatten or
//! mute the near-end talker -- so a confidently wrong lag costs far more than an honest
//! refusal, which the caller turns into "skip the pre-pass and pass the mic through
//! untouched".
//!
//! # Method
//!
//! GCC-PHAT over several windows, then a median across them.
//!
//! *Phase transform, not plain cross-correlation.* Dividing the cross-spectrum by its own
//! magnitude whitens it, which keeps the correlation peak sharp under the reverberation and
//! speaker colouration a room adds to the bleed path -- the case here by construction. Plain
//! correlation of speech against speech peaks broadly and would land several milliseconds
//! off. This is settled practice for time-delay-of-arrival, not a local invention.
//!
//! *Windows, not one whole-file correlation.* A whole-track correlation on this material is
//! dominated by mic speech against speaker silence and returns noise; `scratch/click_align.py`
//! documents that failure. Windows are chosen where the speaker track actually has energy,
//! kept apart so they measure independent moments, and each yields its own lag.
//!
//! *A line, not a point.* The two clocks recording a long meeting drift against each other --
//! a real property of independent hardware, not a measurement error -- so the per-window lags
//! are fit to a line in time rather than reduced straight to a single median. Outlier
//! rejection runs against each window's residual from that line, not against the raw lag, so
//! drift across the meeting is absorbed by the fit instead of being read as disagreement.
//! `scratch/click_align.py` fits this line with plain Theil-Sen (the median of every pairwise
//! slope); it was not carried across when this module split the click take into a
//! whole-meeting search, and plain Theil-Sen turns out not to be the right carry-over. At the
//! window counts this module actually works with -- as few as `MIN_WINDOWS` -- a single stray
//! window corrupts as many as half of Theil-Sen's pairwise slopes, which is enough to move its
//! median off the true value; a 35 s click take with dozens of clicks never exercised that
//! edge. Siegel's repeated-median refinement -- take the median slope *from each point's own
//! pairs* first, then the median of those -- has a much higher breakdown point at small counts
//! and recovers the true line in exactly the cases plain Theil-Sen does not.
//!
//! *Guards.* Each rejection below was added to `click_align.py` after it produced a
//! confident wrong answer, and each still applies:
//!
//! - A peak on the edge of the search range was not found, it was clipped: the real
//!   transient is outside the range, or there is none.
//! - A microphone window with no noise floor cannot support a ratio. A Bluetooth HFP mic
//!   gates to exact digital zero between sounds, which made a take where the mic heard
//!   nothing read as fully valid.
//! - A peak that does not clear its surroundings is not a detection. Peak-to-sidelobe ratio
//!   is the GCC-PHAT analogue of that script's 3x-noise-floor rule.

use std::sync::Arc;

use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

use crate::audio::TARGET_RATE;
use crate::progress::Phase;

/// Both tracks are expected at Whisper's rate, which is the rate the rest of this crate
/// already works in.
const RATE: f64 = TARGET_RATE as f64;

/// Search range around the metadata offset.
///
/// Measured residuals span about -13 ms to +410 ms across built-in, USB and Bluetooth output
/// paths; this keeps headroom on both sides. Do not narrow it on the assumption that the
/// offset is positive -- a negative residual is exactly what the stored offset produces on a
/// low-latency output path.
const SEARCH_LO_MS: f64 = -100.0;
const SEARCH_HI_MS: f64 = 700.0;

/// One window of speaker audio to correlate. Long enough to carry several syllables of
/// content, short enough that a dozen fit inside a short meeting.
const WINDOW_MS: f64 = 3000.0;

/// At most this many windows; more costs transform time and buys nothing once the median has
/// settled.
const MAX_WINDOWS: usize = 12;

/// Fewer surviving windows than this is not a measurement. Three is the smallest count that
/// lets a median outvote a single bad detection.
const MIN_WINDOWS: usize = 3;

/// How far the correlation peak must stand above the rest of the search range, measured
/// against the RMS of the correlation outside a small exclusion zone around the peak.
/// Uncorrelated tracks produce peaks of roughly 4x by chance alone.
const MIN_PEAK_TO_SIDELOBE: f32 = 8.0;

/// Half-width of the exclusion zone around the peak, which belongs to the peak itself rather
/// than to the sidelobes it is being compared against.
const EXCLUSION_MS: f64 = 2.0;

/// A peak this close to either end of the search range is a clipped peak, not a detection.
const EDGE_SAMPLES: i64 = 2;

/// Below this the microphone window has no noise floor to speak of: it is digital silence or
/// a gated mic, and every ratio computed from it would be meaningless.
const NOISE_FLOOR_MIN: f32 = 1e-6;

/// How far surviving windows' residuals against the fitted drift line may still disagree and
/// remain one measurement. `aggregate` fits a line through the windows first and measures this
/// against what is left over -- a real, slowly drifting delay lands near zero here even across
/// an hour, because the drift itself is absorbed by the line rather than counted as spread. A
/// spread this wide in the residuals means the windows locked onto genuinely different delays
/// -- a second regime, such as a mid-meeting output device change -- not that the delay moved.
const MAX_SPREAD_MS: f64 = 20.0;

/// Only the band where the bleed carries usable content is whitened and inverted. Whitening
/// the whole spectrum promotes rumble and near-Nyquist hiss, neither of which survives a
/// loudspeaker and a room, to the same weight as speech.
const BAND_LO_HZ: f64 = 80.0;
const BAND_HI_HZ: f64 = 7000.0;

/// The measured shift between the recorded speaker track and the recorded mic track.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Alignment {
    /// The two tracks correlate, at this lag.
    ///
    /// `lag_samples` is signed and states which mic sample a reference sample belongs with:
    /// speaker-track sample `i` is the same instant in the room as mic-track sample
    /// `i + lag_samples`. So a caller pairing frames for the echo canceller reads the render
    /// frame from `speaker[i..]` and the matching capture frame from `mic[i + lag..]`,
    /// starting at whichever `i` makes both sides non-negative. It is reported at the
    /// midpoint in time of the surviving windows, not at the start of the recording, which
    /// bounds the worst-case error against the true instantaneous lag at either end of the
    /// meeting to half of `drift_ms_per_hour`'s total excursion -- `cancel_bleed` applies one
    /// constant shift to the whole track, and its own `RENDER_HEADROOM` plus AEC3's adaptive
    /// filter absorb the rest.
    ///
    /// `spread_samples` is the range between the highest and lowest surviving window's
    /// residual against the fitted drift line -- not the raw range of their lags, which drift
    /// alone can widen without the windows actually disagreeing. It is reported rather than
    /// smoothed away: a wide residual spread on an accepted measurement means the correlation
    /// was marginal, which the caller may want to say out loud.
    ///
    /// `drift_ms_per_hour` is the fitted line's slope, converted from the dimensionless ratio
    /// between the two clocks to how many milliseconds of lag that ratio accumulates over an
    /// hour of recording. Near zero on a session with two clocks that stay in step; the ~2
    /// ms/hour to ~7 ms/hour range has been observed between a MacBook's microphone clock and
    /// ScreenCaptureKit's.
    Measured {
        lag_samples: i64,
        windows_used: usize,
        spread_samples: i64,
        drift_ms_per_hour: f64,
    },
    /// No reliable correlation, so there is no lag to report -- not a lag of zero. The caller
    /// skips echo cancellation and passes the mic track through untouched.
    NotMeasurable { reason: NotMeasurable },
}

/// Why a measurement was refused. Every variant is a normal outcome of a real recording, not
/// a malfunction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotMeasurable {
    /// The tracks cannot supply even one window with the full search range inside them.
    TracksTooShort,
    /// Too few windows produced a peak that survived the guards. The headphones case, where
    /// nothing bled into the microphone, lands here.
    TooFewWindows { survived: usize, examined: usize },
    /// Windows that individually looked convincing disagreed with each other, so no single
    /// lag describes the recording -- the output device changed mid-meeting, or the peaks
    /// were coincidence.
    InconsistentWindows { windows: usize, spread_samples: i64 },
}

impl std::fmt::Display for NotMeasurable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NotMeasurable::TracksTooShort => write!(
                f,
                "the tracks are too short to search {SEARCH_LO_MS:+.0}..{SEARCH_HI_MS:+.0} ms \
                 around a {WINDOW_MS:.0} ms window"
            ),
            NotMeasurable::TooFewWindows { survived, examined } => write!(
                f,
                "only {survived} of {examined} window(s) correlated well enough to trust \
                 (need {MIN_WINDOWS}); the microphone probably never heard the speakers"
            ),
            NotMeasurable::InconsistentWindows {
                windows,
                spread_samples,
            } => write!(
                f,
                "{windows} windows disagreed by {:.0} ms, more than the {MAX_SPREAD_MS:.0} ms \
                 a single delay can explain",
                *spread_samples as f64 * 1000.0 / RATE
            ),
        }
    }
}

/// Measures the lag between the recorded speaker track and the recorded mic track.
///
/// Both tracks must already be 16 kHz mono, as [`crate::audio::read_track_16k_mono`] returns
/// them. `metadata_offset_s` is `session.json`'s account of how much later the microphone
/// track's first sample is than the speaker track's -- negative if the microphone started
/// first. It is used only to centre the search; the answer comes from the signals.
///
/// See [`Alignment`] for the sign convention on the result, which is the single likeliest
/// thing to wire in backwards.
pub fn measure_reference_lag(
    mic_16k: &[f32],
    speaker_16k: &[f32],
    metadata_offset_s: f64,
) -> Alignment {
    let window = ms_to_samples(WINDOW_MS) as usize;
    let span = (ms_to_samples(SEARCH_HI_MS) - ms_to_samples(SEARCH_LO_MS)) as usize;
    let capture = window + span;

    // The metadata's guess at the lag, in the sense Alignment documents: if the mic track
    // started later, the same instant is at a *lower* mic index. `as i64` saturates and maps
    // NaN to zero, so a corrupt offset cannot wrap the arithmetic below.
    let baseline = -((metadata_offset_s * RATE).round() as i64);
    let lo_lag = baseline.saturating_add(ms_to_samples(SEARCH_LO_MS));

    // A window starts at `start` in the speaker track; the mic slice it is searched against
    // runs from `start + lo_lag` and has to fit entirely inside the mic track.
    let first_start = lo_lag.saturating_neg().max(0);
    let last_start = (speaker_16k.len() as i64 - window as i64).min(
        (mic_16k.len() as i64)
            .saturating_sub(capture as i64)
            .saturating_sub(lo_lag),
    );
    if last_start < first_start {
        return Alignment::NotMeasurable {
            reason: NotMeasurable::TracksTooShort,
        };
    }

    let starts = select_windows(speaker_16k, first_start, last_start, window);
    let correlator = Correlator::new((capture + window).next_power_of_two());

    // Reported separately from the scan above rather than as one continuous phase: the two
    // halves are different work over different counts -- a walk over the whole track, then at
    // most `MAX_WINDOWS` FFT correlations -- and a percentage that jumped back to zero halfway
    // would read as a bug. Both are usually quick, in which case neither says anything.
    let mut phase = Phase::start("aligning: correlating windows");
    // `(window_start_sample, lag_samples)`: the time axis `aggregate` fits a drift line
    // against, paired with the lag measured at that instant.
    let mut lags: Vec<(i64, i64)> = Vec::with_capacity(starts.len());
    for (measured, start) in starts.iter().enumerate() {
        phase.at(measured, starts.len());
        let reference = &speaker_16k[*start as usize..*start as usize + window];
        let heard_from = (start + lo_lag) as usize;
        let heard = &mic_16k[heard_from..heard_from + capture];
        if let Some(offset) = window_lag(heard, reference, span, &correlator) {
            lags.push((*start, lo_lag + offset));
        }
    }
    phase.done();

    aggregate(lags, starts.len())
}

/// Picks the windows to measure: where the speaker track has energy, spread out, capped.
///
/// Ranking by energy is what keeps this off silence. Requiring separation is what makes the
/// estimates independent -- a dozen overlapping windows over one loud sentence would agree
/// with each other and say nothing about the rest of the recording.
fn select_windows(speaker: &[f32], first_start: i64, last_start: i64, window: usize) -> Vec<i64> {
    // Candidate starts advance a quarter window at a time, so a burst of speech is never
    // straddled by every candidate at once. Summing each candidate directly costs four passes
    // over the track in total, which is cheaper than it looks and needs no prefix-sum array
    // the size of an hour of audio.
    let block = window / 4;
    // Four passes over an hour of audio, summing squares, is the expensive half of alignment
    // and the half that has no natural loop counter of its own -- so the candidate count is
    // derived up front purely to have a total to report against.
    let candidates_total = ((last_start - first_start).max(0) / block as i64) as usize + 1;
    let mut phase = Phase::start("aligning: scanning for reference windows");

    let mut candidates: Vec<(f64, i64)> = Vec::new();
    let mut scanned = 0usize;
    let mut start = first_start;
    while start <= last_start {
        phase.at(scanned, candidates_total);
        scanned += 1;
        let slice = &speaker[start as usize..start as usize + window];
        let energy: f64 = slice.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
        if energy > 0.0 {
            candidates.push((energy, start));
        }
        start += block as i64;
    }
    phase.done();

    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut chosen: Vec<i64> = Vec::with_capacity(MAX_WINDOWS);
    for (_, start) in candidates {
        if chosen.len() == MAX_WINDOWS {
            break;
        }
        if chosen
            .iter()
            .all(|other| (other - start).abs() >= window as i64)
        {
            chosen.push(start);
        }
    }
    chosen.sort_unstable();
    chosen
}

/// The lag of one window as an offset into the search range, or `None` if the window failed a
/// guard.
fn window_lag(
    heard: &[f32],
    reference: &[f32],
    span: usize,
    correlator: &Correlator,
) -> Option<i64> {
    // A mic window that is digital silence, or gated to exact zero between sounds, gives
    // nothing to compare a peak against. This is the guard that a Bluetooth HFP take failed
    // silently before it existed.
    if noise_floor(heard) < NOISE_FLOOR_MIN {
        return None;
    }

    let correlation = correlator.gcc_phat(heard, reference)?;
    let searched = &correlation[..=span];

    let (peak_at, peak) = searched
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    let peak_at = peak_at as i64;
    let peak = f64::from(*peak);
    if peak <= 0.0 || peak_at < EDGE_SAMPLES || peak_at > span as i64 - EDGE_SAMPLES {
        return None;
    }

    let exclusion = ms_to_samples(EXCLUSION_MS);
    let mut sidelobe_energy = 0.0f64;
    let mut sidelobes = 0usize;
    for (at, value) in searched.iter().enumerate() {
        if (at as i64 - peak_at).abs() > exclusion {
            sidelobe_energy += f64::from(*value) * f64::from(*value);
            sidelobes += 1;
        }
    }
    if sidelobes == 0 {
        return None;
    }
    let sidelobe_rms = (sidelobe_energy / sidelobes as f64).sqrt();
    if sidelobe_rms <= 0.0 || peak / sidelobe_rms < f64::from(MIN_PEAK_TO_SIDELOBE) {
        return None;
    }

    Some(peak_at)
}

/// Turns per-window `(window_start_sample, lag_samples)` pairs into one answer, or into a
/// refusal.
///
/// The model is a line, `lag(t) = intercept + slope * t`, fit with a repeated-median
/// regression (see [`repeated_median_fit`]) so that a single stray window cannot swing it --
/// least squares was tried against exactly that failure mode in `click_align.py` and rejected
/// there for the same reason a mean is rejected below.
///
/// Outlier rejection around the median, on median absolute deviation rather than standard
/// deviation, is `click_align.py`'s rule: a single stray detection dominates a mean and skews
/// the very deviation that is supposed to catch it. It runs here against each window's
/// *residual* from the fitted line rather than against its raw lag -- the one change that
/// matters, because a residual is deviation from a delay that is allowed to move, and a real
/// drift across a long meeting no longer inflates it the way it inflated deviation from a flat
/// median.
///
/// What used to also live here -- refusing outright once as many windows were discarded as it
/// takes to make a measurement -- is gone. It fired before the spread check below ever ran, so
/// it could refuse a window set the spread check would have passed, and separately it is not a
/// reliable way to tell a real second regime (a mid-meeting output device change) from a
/// handful of strays: a true even split between two regimes does not reliably produce a large
/// enough discard count to trip it. Telling the two apart is entirely the spread check's job.
fn aggregate(mut lags: Vec<(i64, i64)>, examined: usize) -> Alignment {
    if lags.len() < MIN_WINDOWS {
        return Alignment::NotMeasurable {
            reason: NotMeasurable::TooFewWindows {
                survived: lags.len(),
                examined,
            },
        };
    }

    lags.sort_unstable_by_key(|&(start, _)| start);
    let (slope, intercept) = repeated_median_fit(&lags);

    // `(window_start, lag, residual)`. `lags` is already ordered by `start`, so this and
    // everything derived from it by filtering stays ordered too.
    let points: Vec<(i64, i64, i64)> = lags
        .iter()
        .map(|&(start, lag)| (start, lag, lag - fitted_lag(slope, intercept, start)))
        .collect();

    let mut residuals: Vec<i64> = points.iter().map(|&(_, _, r)| r).collect();
    residuals.sort_unstable();
    let centre = median(&residuals);
    let mut deviations: Vec<i64> = residuals.iter().map(|r| (r - centre).abs()).collect();
    deviations.sort_unstable();
    let tolerance = (4 * median(&deviations)).max(ms_to_samples(2.0));

    let kept: Vec<(i64, i64, i64)> = points
        .into_iter()
        .filter(|&(_, _, residual)| (residual - centre).abs() <= tolerance)
        .collect();

    if kept.len() < MIN_WINDOWS {
        return Alignment::NotMeasurable {
            reason: NotMeasurable::TooFewWindows {
                survived: kept.len(),
                examined,
            },
        };
    }

    let kept_residuals = kept.iter().map(|&(_, _, r)| r);
    let spread = kept_residuals.clone().max().unwrap() - kept_residuals.min().unwrap();
    if spread > ms_to_samples(MAX_SPREAD_MS) {
        return Alignment::NotMeasurable {
            reason: NotMeasurable::InconsistentWindows {
                windows: kept.len(),
                spread_samples: spread,
            },
        };
    }

    // The lag at the midpoint in time of the surviving windows, not at the fitted line's
    // `t = 0` intercept, which would usually sit well before the recording started and say
    // nothing about the meeting itself. See the doc comment on [`Alignment::Measured`].
    let t_min = kept.first().unwrap().0;
    let t_max = kept.last().unwrap().0;
    let midpoint = t_min as f64 + (t_max - t_min) as f64 / 2.0;

    Alignment::Measured {
        lag_samples: (intercept + slope * midpoint).round() as i64,
        windows_used: kept.len(),
        spread_samples: spread,
        drift_ms_per_hour: slope * 3_600_000.0,
    }
}

/// Fitted lag at `start`, rounded to the nearest sample.
fn fitted_lag(slope: f64, intercept: f64, start: i64) -> i64 {
    (intercept + slope * start as f64).round() as i64
}

/// Siegel's repeated-median regression: for each point, the median of its own slope to every
/// other point; the overall slope is the median of those `n` per-point medians. The intercept
/// is the ordinary companion step, the median of each point's own `lag - slope * t`.
///
/// This is a refinement of Theil-Sen (whose slope is the single median over all `O(n^2)`
/// pairwise slopes at once), chosen over plain Theil-Sen specifically for its breakdown point
/// at the small window counts this module works with. With one bad point among `n`, plain
/// Theil-Sen has that point corrupting `n - 1` of the `n * (n - 1) / 2` pairwise slopes; at
/// `n = MIN_WINDOWS + 1 = 4`, that is 3 of 6 -- half -- which is already enough to move the
/// median off the true value, confirmed by hand against the fixture this broke
/// (`a_single_stray_window_never_turns_a_measurable_recording_into_a_refusal`). Because a
/// repeated median takes a per-point median first, a single bad point can dominate only *its
/// own* row of medians, not the pooled set, which is what gives it its well-established 50%
/// breakdown point (Rousseeuw & Leroy, *Robust Regression and Outlier Detection*) versus plain
/// Theil-Sen's roughly 29%. `scratch/click_align.py` uses plain Theil-Sen, over a single 35 s
/// take with far more clicks relative to its rare stray than a window count this small ever
/// has -- an edge plain Theil-Sen never had to survive there.
///
/// `points` here is at most `MAX_WINDOWS` long, so this computes at most `MAX_WINDOWS` times
/// `MAX_WINDOWS` minus one pairwise slopes -- trivial next to the FFT work that produced the
/// points in the first place.
fn repeated_median_fit(points: &[(i64, i64)]) -> (f64, f64) {
    let mut per_point_slopes: Vec<f64> = Vec::with_capacity(points.len());
    for i in 0..points.len() {
        let (t_i, lag_i) = points[i];
        let mut slopes: Vec<f64> = Vec::with_capacity(points.len() - 1);
        for (j, &(t_j, lag_j)) in points.iter().enumerate() {
            // Window starts are chosen distinct and at least a window apart, so `t_j != t_i`
            // holds for every real call; it guards the arithmetic rather than expecting to
            // fire.
            if j != i && t_j != t_i {
                slopes.push((lag_j - lag_i) as f64 / (t_j - t_i) as f64);
            }
        }
        per_point_slopes.push(median_f64(&mut slopes));
    }
    let slope = median_f64(&mut per_point_slopes);

    let mut intercepts: Vec<f64> = points
        .iter()
        .map(|&(t, lag)| lag as f64 - slope * t as f64)
        .collect();
    let intercept = median_f64(&mut intercepts);

    (slope, intercept)
}

/// Median of a slice, sorted in place. Unlike [`median`], this is for the `f64` slopes and
/// intercepts [`repeated_median_fit`] works in, which have no total order of their own to rely
/// on.
fn median_f64(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Generalized cross-correlation with a phase transform, wrapped around the one FFT size this
/// module needs.
struct Correlator {
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    size: usize,
    band: std::ops::Range<usize>,
}

impl Correlator {
    fn new(size: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let bin = |hz: f64| (hz * size as f64 / RATE).round() as usize;
        Correlator {
            forward: planner.plan_fft_forward(size),
            inverse: planner.plan_fft_inverse(size),
            size,
            // Never bin 0: the inverse transform requires a real DC term, and a whitened DC
            // bin is a constant offset across the whole correlation anyway.
            band: bin(BAND_LO_HZ).max(1)..bin(BAND_HI_HZ).min(size / 2),
        }
    }

    /// Correlation of `heard` against `reference` at every non-negative offset.
    ///
    /// Index `k` of the result is `sum over n of heard[n + k] * reference[n]`, whitened. The
    /// transform is long enough for both inputs plus their correlation, so offsets the caller
    /// reads are free of circular wraparound.
    fn gcc_phat(&self, heard: &[f32], reference: &[f32]) -> Option<Vec<f32>> {
        let heard_spectrum = self.spectrum(heard);
        let reference_spectrum = self.spectrum(reference);

        // The phase transform: keep the cross-spectrum's phase, discard its magnitude. The
        // floor is relative to the band's own mean magnitude, so an empty bin is left near
        // zero instead of being amplified into a full-weight one.
        let mut cross: Vec<Complex32> = heard_spectrum
            .iter()
            .zip(&reference_spectrum)
            .map(|(h, r)| h * r.conj())
            .collect();
        let mean: f64 = self
            .band
            .clone()
            .map(|bin| f64::from(cross[bin].norm()))
            .sum::<f64>()
            / self.band.len().max(1) as f64;
        if mean <= 0.0 {
            return None;
        }
        let floor = (mean * 1e-6) as f32;

        for (bin, value) in cross.iter_mut().enumerate() {
            if self.band.contains(&bin) {
                *value /= value.norm() + floor;
            } else {
                *value = Complex32::default();
            }
        }

        let mut correlation = self.inverse.make_output_vec();
        self.inverse
            .process(&mut cross, &mut correlation)
            .expect("inverse transform sized by the same planner");
        Some(correlation)
    }

    fn spectrum(&self, samples: &[f32]) -> Vec<Complex32> {
        let mut padded = self.forward.make_input_vec();
        padded[..samples.len()].copy_from_slice(samples);
        padded[samples.len()..].fill(0.0);
        debug_assert!(samples.len() <= self.size);

        let mut spectrum = self.forward.make_output_vec();
        self.forward
            .process(&mut padded, &mut spectrum)
            .expect("forward transform sized by the same planner");
        spectrum
    }
}

/// The typical short-term level of a window, which is what a peak has to stand above.
///
/// Median of 10 ms block levels, not the mean: speech in the window would drag a mean up and
/// hide the fact that everything between the words is exact zero.
fn noise_floor(samples: &[f32]) -> f32 {
    let block = ms_to_samples(10.0) as usize;
    let mut levels: Vec<f32> = samples
        .chunks(block)
        .map(|chunk| {
            let energy: f64 = chunk.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
            (energy / chunk.len() as f64).sqrt() as f32
        })
        .collect();
    if levels.is_empty() {
        return 0.0;
    }
    levels.sort_by(f32::total_cmp);
    levels[levels.len() / 2]
}

/// Median of an already sorted, non-empty slice.
fn median(sorted: &[i64]) -> i64 {
    sorted[sorted.len() / 2]
}

fn ms_to_samples(ms: f64) -> i64 {
    (ms / 1000.0 * RATE).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE_HZ: usize = TARGET_RATE as usize;

    /// Deterministic noise. A fixed seed keeps every assertion below reproducible; a real
    /// RNG would make a threshold that is marginal fail once a month instead of always.
    struct Noise(u64);

    impl Noise {
        fn sample(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            // The top 24 bits, scaled to [-1, 1). Amplitudes have to be realistic, not just
            // random: the noise-floor guard is an absolute threshold, so a helper that
            // produced samples in the thousands would sail past it for the wrong reason.
            (self.0 >> 40) as f32 / 8_388_608.0 - 1.0
        }
    }

    /// Something speech-shaped: band-limited noise gated into bursts.
    ///
    /// Not a tone and not white noise. A tone correlates with itself at every period, and
    /// white noise is easier to align than anything a loudspeaker can reproduce. The bursts
    /// are what make window selection do work, since a third of the track has nothing in it.
    ///
    /// The burst schedule is derived from the seed, so two talkers built here do not share an
    /// envelope. That is not cosmetic: a shared on/off pattern correlates at zero lag no
    /// matter what the carriers do, and an earlier version of this helper made every test
    /// pass by measuring its own gating.
    fn speech_like(seed: u64, samples: usize) -> Vec<f32> {
        let mut noise = Noise(seed);
        let mut low = 0.0f32;
        let mut previous_low = 0.0f32;
        let mut high = 0.0f32;
        let cycle = RATE_HZ * (2 + seed as usize % 3);
        let voiced_for = cycle * 2 / 3;
        let phase = seed as usize * 7919 % cycle;
        (0..samples)
            .map(|i| {
                // One-pole low-pass for a speech-like tilt, then a DC blocker: a loudspeaker
                // reproduces neither the top octave nor the rumble.
                low = 0.6 * low + 0.4 * noise.sample();
                high = 0.99 * high + low - previous_low;
                previous_low = low;
                if (i + phase) % cycle < voiced_for {
                    high * 0.3
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// A mic track that heard `speaker` through `path`, `lag` samples late, over its own
    /// near-end talker and room noise.
    ///
    /// The near-end matters: a mic track that contained only the echo would be an easier
    /// problem than any real recording.
    fn mic_hearing(speaker: &[f32], lag: i64, path: &[(i64, f32)], seed: u64) -> Vec<f32> {
        let mut floor = Noise(seed);
        let near_end = speech_like(seed ^ 0x5eed, speaker.len());
        let mut mic: Vec<f32> = (0..speaker.len())
            .map(|i| near_end[i] * 0.6 + floor.sample() * 0.005)
            .collect();
        for (tap, gain) in path {
            for i in 0..speaker.len() as i64 {
                let heard_at = i + lag + tap;
                if heard_at >= 0 && (heard_at as usize) < mic.len() {
                    mic[heard_at as usize] += speaker[i as usize] * gain;
                }
            }
        }
        mic
    }

    /// The direct sound and nothing else.
    const DIRECT: &[(i64, f32)] = &[(0, 0.35)];

    /// A mic track whose lag against `speaker` drifts linearly across the recording, rather
    /// than staying fixed -- what two independent clocks running at very slightly different
    /// rates produce, not a moving microphone.
    ///
    /// `lag_at(i) = lag_start + drift_per_sample * i` is placed by rounding the speaker-track
    /// sample's landing index to the nearest mic sample rather than by resampling the speaker
    /// track at the mismatched rate. That is a coarser model of drift than a real recording
    /// has -- it steps instead of interpolating -- but it is faithful to sample accuracy,
    /// which is what GCC-PHAT and the repeated-median fit are both being asked to recover here.
    fn mic_hearing_with_drift(
        speaker: &[f32],
        lag_start: i64,
        drift_per_sample: f64,
        gain: f32,
        seed: u64,
    ) -> Vec<f32> {
        let mut floor = Noise(seed);
        let near_end = speech_like(seed ^ 0x5eed, speaker.len());
        let mut mic: Vec<f32> = (0..speaker.len())
            .map(|i| near_end[i] * 0.6 + floor.sample() * 0.005)
            .collect();
        for (i, sample) in speaker.iter().enumerate() {
            let lag = lag_start as f64 + drift_per_sample * i as f64;
            let heard_at = (i as f64 + lag).round();
            if heard_at >= 0.0 && (heard_at as usize) < mic.len() {
                mic[heard_at as usize] += sample * gain;
            }
        }
        mic
    }

    fn measured(alignment: Alignment) -> (i64, usize, i64, f64) {
        match alignment {
            Alignment::Measured {
                lag_samples,
                windows_used,
                spread_samples,
                drift_ms_per_hour,
            } => (lag_samples, windows_used, spread_samples, drift_ms_per_hour),
            Alignment::NotMeasurable { reason } => {
                panic!("expected a measurement, got: {reason}")
            }
        }
    }

    fn reason(alignment: Alignment) -> NotMeasurable {
        match alignment {
            Alignment::NotMeasurable { reason } => reason,
            Alignment::Measured { lag_samples, .. } => {
                panic!("expected a refusal, got a lag of {lag_samples} samples")
            }
        }
    }

    /// The core claim: a known delay comes back, to the sample, anywhere in the search range.
    ///
    /// The residuals are the ones the click tests actually produced -- a negative one from a
    /// low-latency output path, a Bluetooth-scale positive one -- plus both extremes of the
    /// range, since a peak that lands on the edge is supposed to be rejected rather than
    /// reported and the edges have to be reachable for that distinction to mean anything.
    #[test]
    fn a_known_delay_is_recovered_within_a_millisecond_across_the_whole_search_range() {
        let speaker = speech_like(1, RATE_HZ * 30);
        let tolerance = ms_to_samples(1.0);

        for (metadata_offset_s, residual_ms) in [
            (0.0, -90.0),
            (0.0, -13.0),
            (0.041_666, 25.0),
            (-0.020, 410.0),
            (0.150, 690.0),
        ] {
            let baseline = -((metadata_offset_s * RATE).round() as i64);
            let truth = baseline + ms_to_samples(residual_ms);
            let mic = mic_hearing(&speaker, truth, DIRECT, 7);

            let (lag, windows, spread, drift) =
                measured(measure_reference_lag(&mic, &speaker, metadata_offset_s));
            assert!(
                (lag - truth).abs() <= tolerance,
                "offset {metadata_offset_s} s, residual {residual_ms} ms: \
                 recovered {lag} samples, truth {truth}"
            );
            assert!(windows >= MIN_WINDOWS, "only {windows} windows");
            assert!(
                spread <= tolerance,
                "a single fixed delay should not spread {spread} samples"
            );
            assert!(
                drift.abs() <= 1000.0,
                "a single fixed delay should not fit a large slope, got {drift:.1} ms/hour"
            );
        }
    }

    /// The bleed path is a room, not a delay line. Reflections and a dulled top end must not
    /// move the answer off the direct path.
    #[test]
    fn a_reflective_attenuated_path_still_yields_the_direct_path_lag() {
        let speaker = speech_like(2, RATE_HZ * 30);
        let truth = ms_to_samples(180.0);
        // Direct sound, a desk bounce, an opposite-polarity wall reflection, a late one.
        let room: &[(i64, f32)] = &[
            (0, 0.30),
            (ms_to_samples(3.0), 0.18),
            (ms_to_samples(7.5), -0.12),
            (ms_to_samples(14.0), 0.08),
        ];
        let reflected = mic_hearing(&speaker, truth, room, 11);
        // A one-pole low-pass over the whole mic track stands in for the speaker's own
        // response: the bleed arrives dull, the near-end does not, and the estimator sees
        // only the sum.
        let mut dulled = Vec::with_capacity(reflected.len());
        let mut state = 0.0f32;
        for sample in &reflected {
            state = 0.5 * state + 0.5 * sample;
            dulled.push(state);
        }

        let (lag, _, _, _) = measured(measure_reference_lag(&dulled, &speaker, 0.0));
        assert!(
            (lag - truth).abs() <= ms_to_samples(1.0),
            "recovered {lag} samples against a direct path at {truth}"
        );
    }

    /// The headphones case. Nothing bled, so there is nothing to find, and inventing a lag
    /// here is the failure that costs the user their mic track.
    #[test]
    fn uncorrelated_tracks_are_not_measurable() {
        let speaker = speech_like(3, RATE_HZ * 30);
        let mut floor = Noise(29);
        let mic: Vec<f32> = speech_like(4, RATE_HZ * 30)
            .iter()
            .map(|near| near * 0.6 + floor.sample() * 0.01)
            .collect();

        let reason = reason(measure_reference_lag(&mic, &speaker, 0.0));
        assert!(
            matches!(
                reason,
                NotMeasurable::TooFewWindows { .. } | NotMeasurable::InconsistentWindows { .. }
            ),
            "{reason}"
        );
    }

    /// Digital silence has no floor to measure a peak against, and a lag from it would be
    /// whatever the whitening amplified out of nothing.
    #[test]
    fn a_silent_mic_track_is_not_measurable() {
        let speaker = speech_like(5, RATE_HZ * 30);
        let mic = vec![0.0f32; speaker.len()];

        assert!(matches!(
            reason(measure_reference_lag(&mic, &speaker, 0.0)),
            NotMeasurable::TooFewWindows { .. }
        ));
    }

    /// The take that read as fully valid while the mic had heard nothing: a Bluetooth HFP mic
    /// gates to exact digital zero, so every ratio computed against its floor was infinite.
    #[test]
    fn a_mic_gated_to_exact_zero_between_sounds_is_not_measurable() {
        let speaker = speech_like(6, RATE_HZ * 30);
        let mut noise = Noise(31);
        // The gate opens for a tenth of a second every three seconds, on the mic's own
        // schedule, and closes to exact zero. None of it is the speaker track.
        let mic: Vec<f32> = (0..speaker.len())
            .map(|i| {
                let open = i % (RATE_HZ * 3) < RATE_HZ / 10;
                if open { noise.sample() * 0.2 } else { 0.0 }
            })
            .collect();

        assert!(matches!(
            reason(measure_reference_lag(&mic, &speaker, 0.0)),
            NotMeasurable::TooFewWindows { .. }
        ));
    }

    /// A recording shorter than one window plus the search range, and an empty one. Both are
    /// real -- a session ended a second after it started -- and neither is an error.
    #[test]
    fn tracks_too_short_for_one_window_are_not_measurable() {
        let short = speech_like(7, RATE_HZ * 2);
        assert_eq!(
            reason(measure_reference_lag(&short, &short, 0.0)),
            NotMeasurable::TracksTooShort
        );
        assert_eq!(
            reason(measure_reference_lag(&[], &[], 0.0)),
            NotMeasurable::TracksTooShort
        );

        // Long enough on the speaker side, absent on the mic side.
        let long = speech_like(8, RATE_HZ * 30);
        assert_eq!(
            reason(measure_reference_lag(&[], &long, 0.0)),
            NotMeasurable::TracksTooShort
        );
    }

    /// Windows that each look convincing but disagree are not one measurement. Averaging them
    /// would produce a lag that describes neither half of the recording.
    #[test]
    fn windows_that_disagree_are_reported_rather_than_averaged() {
        let speaker = speech_like(9, RATE_HZ * 40);
        let early = mic_hearing(&speaker, ms_to_samples(60.0), DIRECT, 13);
        let late = mic_hearing(&speaker, ms_to_samples(400.0), DIRECT, 13);
        // The output device changed halfway through, which is a thing that happens when
        // someone connects a headset mid-meeting.
        let seam = early.len() / 2;
        let mut mic = early;
        mic[seam..].copy_from_slice(&late[seam..]);

        match reason(measure_reference_lag(&mic, &speaker, 0.0)) {
            NotMeasurable::InconsistentWindows { spread_samples, .. } => {
                assert!(
                    spread_samples > ms_to_samples(MAX_SPREAD_MS),
                    "spread {spread_samples} should have exceeded the limit"
                );
            }
            other => panic!("expected inconsistent windows, got: {other}"),
        }
    }

    /// The measurement is a median over several independent moments, not one correlation.
    #[test]
    fn the_estimate_comes_from_several_separated_windows() {
        let speaker = speech_like(10, RATE_HZ * 40);
        let mic = mic_hearing(&speaker, ms_to_samples(120.0), DIRECT, 17);

        let (_, windows, _, _) = measured(measure_reference_lag(&mic, &speaker, 0.0));
        assert!(windows >= MIN_WINDOWS, "only {windows} window(s) survived");

        let window = ms_to_samples(WINDOW_MS) as usize;
        let span = (ms_to_samples(SEARCH_HI_MS) - ms_to_samples(SEARCH_LO_MS)) as usize;
        let starts = select_windows(
            &speaker,
            ms_to_samples(100.0),
            (speaker.len() - window - span) as i64,
            window,
        );
        assert!(starts.len() >= MIN_WINDOWS);
        for pair in starts.windows(2) {
            assert!(
                pair[1] - pair[0] >= window as i64,
                "windows at {} and {} overlap",
                pair[0],
                pair[1]
            );
        }
    }

    /// A peak against the end of the search range is a clipped peak: the real delay is
    /// outside the range, so the honest answer is that it was not found.
    #[test]
    fn a_delay_beyond_the_search_range_is_refused_rather_than_pinned_to_the_edge() {
        let speaker = speech_like(11, RATE_HZ * 30);
        // 1.2 s, well past the +700 ms the search covers.
        let mic = mic_hearing(&speaker, ms_to_samples(1200.0), DIRECT, 19);

        assert!(matches!(
            reason(measure_reference_lag(&mic, &speaker, 0.0)),
            NotMeasurable::TooFewWindows { .. }
        ));
    }

    /// Acceptance criterion #3. Clock drift across a long meeting is a real, slowly moving
    /// delay, not disagreement between windows, and the reported lag has to stay close to the
    /// truth at both ends of an hour, not just wherever the windows happened to cluster.
    #[test]
    fn a_hand_hour_of_clock_drift_is_measured_within_tolerance_at_both_ends() {
        let speaker = speech_like(20, RATE_HZ * 3600);
        let lag_start = ms_to_samples(120.0);
        // A few ms of drift over the hour -- inside the ~2 to ~7 ms/hour range seen between a
        // MacBook's microphone clock and ScreenCaptureKit's, and, at the old raw-spread rule,
        // already more than a quarter of MAX_SPREAD_MS by itself.
        let total_drift_ms = 6.0;
        let drift_per_sample = ms_to_samples(total_drift_ms) as f64 / speaker.len() as f64;
        let mic = mic_hearing_with_drift(&speaker, lag_start, drift_per_sample, 0.35, 23);

        let (lag_samples, windows, spread, drift_ms_per_hour) =
            measured(measure_reference_lag(&mic, &speaker, 0.0));
        assert!(windows >= MIN_WINDOWS, "only {windows} windows");
        assert!(
            spread <= ms_to_samples(MAX_SPREAD_MS),
            "residual spread {spread} should fit under the limit once drift is fit out"
        );
        assert!(
            (drift_ms_per_hour - total_drift_ms).abs() <= 2.0,
            "fitted drift {drift_ms_per_hour:.2} ms/hour, true rate was {total_drift_ms} ms/hour"
        );

        // The reported lag sits at the midpoint of the surviving windows, so the worst-case
        // error against the true instantaneous lag at either end is bounded to half the total
        // drift -- see the doc comment on `Alignment::Measured`. A couple of extra ms covers
        // ordinary GCC-PHAT/repeated-median noise on top of that bound.
        let true_lag_at = |i: usize| lag_start + (drift_per_sample * i as f64).round() as i64;
        let tolerance = ms_to_samples(total_drift_ms / 2.0 + 2.0);
        let true_start = true_lag_at(0);
        let true_end = true_lag_at(speaker.len() - 1);
        assert!(
            (lag_samples - true_start).abs() <= tolerance,
            "recovered {lag_samples}, true lag at the start of the recording was {true_start}"
        );
        assert!(
            (lag_samples - true_end).abs() <= tolerance,
            "recovered {lag_samples}, true lag at the end of the recording was {true_end}"
        );
    }

    /// Acceptance criterion #4. One stray detection among otherwise consistent windows must
    /// not be able to trip a refusal, at every window count from the smallest measurable set
    /// up through a typical one -- guards the specific failure the discard-count branch this
    /// task removes used to risk: it counted how many windows disagreed rather than how far.
    ///
    /// Exercised directly against `aggregate`: the acoustics that produce a `(start, lag)`
    /// pair are somebody else's job (`window_lag`'s and `select_windows`'s own tests cover
    /// them), and folding them in here would make this test about GCC-PHAT's accuracy instead
    /// of about the aggregation rule this acceptance criterion is actually about.
    #[test]
    fn a_single_stray_window_never_turns_a_measurable_recording_into_a_refusal() {
        for consistent in MIN_WINDOWS..=9 {
            let step = ms_to_samples(WINDOW_MS);
            let truth = ms_to_samples(150.0);
            let mut lags: Vec<(i64, i64)> =
                (0..consistent as i64).map(|i| (i * step, truth)).collect();
            // One stray, far outside any tolerance the consistent windows could produce, at
            // its own distinct time.
            lags.push((consistent as i64 * step, truth + ms_to_samples(500.0)));

            let (lag, windows, _, _) = measured(aggregate(lags, consistent + 1));
            assert_eq!(
                windows, consistent,
                "at {consistent} consistent windows plus one stray, only the stray should be \
                 discarded"
            );
            assert_eq!(lag, truth, "at {consistent} consistent windows");
        }
    }

    /// The arithmetic of `repeated_median_fit`, independent of `aggregate`'s outlier rejection
    /// or of any audio at all -- so a future change to the outlier logic cannot silently break
    /// the fit itself. Every point sits exactly on the line, since the fit's robustness
    /// against a stray point is already exercised end to end above.
    #[test]
    fn repeated_median_fit_recovers_a_hand_computed_line() {
        let points = [(0, 5), (10, 25), (20, 45), (30, 65), (40, 85)];
        let (slope, intercept) = repeated_median_fit(&points);
        assert!((slope - 2.0).abs() < 1e-9, "slope {slope}");
        assert!((intercept - 5.0).abs() < 1e-9, "intercept {intercept}");
    }
}
