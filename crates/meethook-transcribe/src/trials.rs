//! Scoring a speaker trial list: how far apart one person sits from themselves, and from
//! everybody else.
//!
//! [`crate::IDENTIFY_DISTANCE`] is only as good as the population it was priced against, and
//! the thing that prices it is not another distance -- it is a *population* of them. Hundreds of pairs, split into the two that matter: two recordings of
//! one person, and two recordings of two people. `cluster-speaker-track` prints every
//! cluster-to-reference distance for one session, which is the right arithmetic at the wrong
//! scale; false-accept rate, false-reject rate, equal-error rate and "is there any cut that
//! separates these at all" are not obtainable by reading a table by eye.
//!
//! This module is that arithmetic and none of the plumbing -- no files, no models, no session
//! directories -- for the same reason [`crate::identify_clusters`] is: every claim here is
//! decidable in microseconds against hand-written numbers, and a diagnostic whose conventions
//! nobody can test is a number to believe rather than evidence. The runner that turns audio
//! into a `&[Trial]` is `examples/speaker-trials.rs`.
//!
//! # The conventions, all in one place
//!
//! Each of these is a place where two reasonable people pick differently, so each is stated
//! rather than left to the reader:
//!
//! - **Accept is strictly below the cut**, spelled the same way `best_match` spells it
//!   (`1.0 - similarity < IDENTIFY_DISTANCE`). A pair sitting exactly on the threshold is a
//!   rejection. A boundary that differs by one `<` between the diagnostic and the decision it
//!   is diagnosing is a bug that shows up only in the one place it matters.
//! - **A false accept is a different-speaker pair below the cut; a false reject is a
//!   same-speaker pair at or above it.**
//! - **The median of an even count is the mean of the two middle values**, and an empty
//!   population is [`None`] rather than zeroes or NaN. Both inherited verbatim from
//!   `cluster-speaker-track`, which is the other caller.
//! - **Percentiles are nearest-rank** on the sorted copy, so `p05` and `p95` are always values
//!   that actually occurred rather than interpolations between two that did.
//! - **The equal-error rate is the mean of the two rates at the cut minimising their
//!   difference**, swept over every distinct distance in the list. There are three conventions
//!   for this in the literature and a number whose convention is unstated is not comparable to
//!   a published one.

use std::cmp::Ordering;

/// One scored pair: two items, and whether they are in fact the same person.
///
/// Which pairs are legal is not decided here -- it is a question about items rather than about
/// numbers, and the runner enforces it. The rule that matters is that no pair drawn from a
/// single recording session is ever a trial: that is `MERGE_DISTANCE`'s question, and folding
/// those in would flatter the same-speaker side with exactly the within-session variation
/// [`crate::IDENTIFY_DISTANCE`] does not govern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trial {
    pub same_speaker: bool,

    /// Cosine distance between the two items' voices: `1.0 - dot(a, b)` over unit vectors, so
    /// 0 is the same direction and 1 is orthogonal. The same arithmetic `best_match` does,
    /// which is what stops this from being able to disagree with it.
    pub distance: f32,
}

/// The shape of a population of distances.
///
/// Deliberately more than min/median/max: a same-speaker distribution with a well-behaved
/// median and a long tail is exactly the distribution a threshold reads as safe and a user
/// experiences as `Unknown N`s, and `p95` is where that shows up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spread {
    pub count: usize,
    pub min: f32,
    /// Nearest-rank 5th percentile: a value that occurred, not an interpolation.
    pub p05: f32,
    /// The mean of the two middle values when `count` is even.
    pub median: f32,
    pub p95: f32,
    pub max: f32,
    pub mean: f32,
}

impl Spread {
    /// Summarizes a population of distances, or [`None`] when there are none.
    ///
    /// `None` rather than zeroes, because the min, median and max of an empty set do not
    /// exist and a fabricated 0.000 reads as two identical recordings. An empty side of a
    /// trial list is a real mistake -- one speaker in the manifest, or one session each --
    /// and it is worth surfacing as "there were no pairs" rather than as a divide by zero
    /// propagating into every rate downstream.
    ///
    /// The input is not modified; a sorted copy is taken.
    pub fn of(distances: &[f32]) -> Option<Spread> {
        if distances.is_empty() {
            return None;
        }
        let mut sorted = distances.to_vec();
        sorted.sort_by(f32::total_cmp);

        let count = sorted.len();
        let median = if count % 2 == 1 {
            sorted[count / 2]
        } else {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        };

        Some(Spread {
            count,
            min: sorted[0],
            p05: nearest_rank(&sorted, 0.05),
            median,
            p95: nearest_rank(&sorted, 0.95),
            max: sorted[count - 1],
            mean: sorted.iter().sum::<f32>() / count as f32,
        })
    }
}

/// The point at which the two error rates come closest to crossing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EqualError {
    /// The mean of the false-accept and false-reject rates at [`EqualError::threshold`].
    pub rate: f32,

    /// The cut it happens at, which is as interesting as the rate: an equal-error rate of 3%
    /// occurring at 0.62 says [`crate::IDENTIFY_DISTANCE`] is conservative, and the rate alone
    /// does not.
    pub threshold: f32,
}

/// The largest cut that misattributes nobody, and what refusing to misattribute costs.
///
/// This is the number `IDENTIFY_DISTANCE`'s asymmetry argument actually wants. That doc
/// comment says a false match -- one person's words under another person's name in a
/// transcript nobody re-reads -- is worse than an `Unknown N` the user fixes in ten seconds,
/// and therefore biases low. This says in one line what that bias buys and what it costs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZeroFalseAccept {
    /// The smallest different-speaker distance in the list. Because accept is *strictly*
    /// below the cut, a cut exactly here accepts no different-speaker pair, and any larger
    /// one accepts at least this pair.
    pub threshold: f32,

    /// The fraction of same-speaker pairs rejected there: the `Unknown N` rate a
    /// "never misattribute anybody" policy would have produced over this trial list.
    pub false_reject_rate: f32,
}

/// Everything a trial list says about one threshold, and about the thresholds it did not use.
///
/// The rates are [`Option`] rather than zero for an absent population, and that is the same
/// decision [`Spread::of`] makes for the same reason: a manifest with one speaker has no
/// different-speaker pairs at all, and a false-accept rate of "0.0" over nothing is the most
/// flattering possible way to report that mistake.
#[derive(Debug, Clone, PartialEq)]
pub struct TrialReport {
    /// The cut the counts below were taken at, carried in the report so that whatever prints
    /// it cannot state a different one than was used.
    pub threshold: f32,

    pub same: Option<Spread>,
    pub different: Option<Spread>,

    /// Different-speaker pairs strictly below [`TrialReport::threshold`].
    pub false_accepts: usize,
    /// `None` when there were no different-speaker pairs to accept wrongly.
    pub false_accept_rate: Option<f32>,

    /// Same-speaker pairs at or above [`TrialReport::threshold`].
    pub false_rejects: usize,
    /// `None` when there were no same-speaker pairs to reject wrongly.
    pub false_reject_rate: Option<f32>,

    /// `Some((min_different, max_same))` when the two populations interleave, and [`None`]
    /// when no different-speaker pair is closer than the furthest-apart same-speaker pair.
    ///
    /// `None` is the good outcome and names the room either side of the cut that separates
    /// them; `Some` is the null result -- no single threshold can be right for everybody --
    /// stated as a value rather than left for a reader to notice.
    pub overlap: Option<(f32, f32)>,

    /// `None` unless *both* populations are non-empty: an error rate that trades one against
    /// the other is not defined when one of them does not exist.
    pub equal_error: Option<EqualError>,

    /// `None` on the same condition as [`TrialReport::equal_error`], and for the same reason:
    /// the threshold comes from the different-speaker side and its cost from the same-speaker
    /// side.
    pub zero_false_accept: Option<ZeroFalseAccept>,
}

/// Scores a trial list at one threshold, and sweeps every other threshold the list can see.
///
/// `threshold` is a parameter rather than a read of [`crate::IDENTIFY_DISTANCE`] so that the
/// caller can print the value it passed and so that "what would 0.55 have done" costs a
/// re-run of some dot products rather than a re-run of the embedding. The runner defaults it
/// to the real constant.
///
/// Total cost is `O(n log n)`: both sides are sorted once and every candidate cut is answered
/// by a binary search rather than by another pass over the trials. A 200-item manifest is
/// 19,900 pairs, which this scores instantly -- which is the whole reason the runner caches
/// embeddings and re-scores rather than re-embedding to try a second threshold.
pub fn score_trials(trials: &[Trial], threshold: f32) -> TrialReport {
    let mut same: Vec<f32> = distances(trials, true);
    let mut different: Vec<f32> = distances(trials, false);
    same.sort_by(f32::total_cmp);
    different.sort_by(f32::total_cmp);

    // Strictly below accepts, exactly as `best_match` does, so the boundary this reports on
    // and the boundary the code decides on cannot drift apart.
    let false_accepts = different.partition_point(|d| *d < threshold);
    let false_rejects = same.len() - same.partition_point(|d| *d < threshold);

    let same_spread = Spread::of(&same);
    let different_spread = Spread::of(&different);

    let overlap = match (same_spread, different_spread) {
        (Some(s), Some(d)) if d.min < s.max => Some((d.min, s.max)),
        _ => None,
    };

    let both_sides = !same.is_empty() && !different.is_empty();
    let zero_false_accept = both_sides.then(|| ZeroFalseAccept {
        threshold: different[0],
        false_reject_rate: false_reject_rate(&same, different[0]),
    });

    TrialReport {
        threshold,
        same: same_spread,
        different: different_spread,
        false_accepts,
        false_accept_rate: rate(false_accepts, different.len()),
        false_rejects,
        false_reject_rate: rate(false_rejects, same.len()),
        overlap,
        equal_error: both_sides.then(|| equal_error(&same, &different)),
        zero_false_accept,
    }
}

fn distances(trials: &[Trial], same_speaker: bool) -> Vec<f32> {
    trials
        .iter()
        .filter(|trial| trial.same_speaker == same_speaker)
        .map(|trial| trial.distance)
        .collect()
}

/// `count / population`, or [`None`] when there is no population to take a fraction of.
fn rate(count: usize, population: usize) -> Option<f32> {
    (population > 0).then(|| count as f32 / population as f32)
}

/// Fraction of same-speaker pairs at or above `cut`. `same` must be sorted ascending.
fn false_reject_rate(same: &[f32], cut: f32) -> f32 {
    (same.len() - same.partition_point(|d| *d < cut)) as f32 / same.len() as f32
}

/// Fraction of different-speaker pairs strictly below `cut`. `different` must be sorted.
fn false_accept_rate(different: &[f32], cut: f32) -> f32 {
    different.partition_point(|d| *d < cut) as f32 / different.len() as f32
}

/// Sweeps every distinct distance in the list as a candidate cut and returns the one where
/// the two rates come closest together.
///
/// Both slices must be sorted ascending and non-empty; the only caller checks both.
///
/// Every distinct *observed* distance, rather than a fixed grid: the rates only change where a
/// trial sits, so a grid would either miss the crossing or spend most of its work on cuts
/// nothing moves at. The reported rate is the mean of the two at that cut, which is the usual
/// reading when a finite list cannot make them exactly equal.
///
/// Ties are broken towards the lower rate and then the lower threshold, so two lists that
/// differ only in the order they were built produce the same number.
fn equal_error(same: &[f32], different: &[f32]) -> EqualError {
    let mut candidates: Vec<f32> = same.iter().chain(different).copied().collect();
    candidates.sort_by(f32::total_cmp);
    candidates.dedup_by(|a, b| a.total_cmp(b) == Ordering::Equal);

    // Gap, rate, cut -- compared in that order, which is what makes the choice deterministic.
    let mut best: Option<(f32, f32, f32)> = None;
    for cut in candidates {
        let accepts = false_accept_rate(different, cut);
        let rejects = false_reject_rate(same, cut);
        let candidate = ((accepts - rejects).abs(), (accepts + rejects) / 2.0, cut);
        if best.is_none_or(|held| ranked(candidate, held) == Ordering::Less) {
            best = Some(candidate);
        }
    }

    // `candidates` is non-empty because both inputs are, so `best` is always set by here.
    let (_, rate, threshold) = best.expect("a non-empty trial list has a candidate cut");
    EqualError { rate, threshold }
}

fn ranked(a: (f32, f32, f32), b: (f32, f32, f32)) -> Ordering {
    a.0.total_cmp(&b.0)
        .then(a.1.total_cmp(&b.1))
        .then(a.2.total_cmp(&b.2))
}

/// The value at `fraction` of the way through a sorted population, by nearest rank.
///
/// `sorted` must be non-empty. Rounding up and stepping back one puts `p95` on a value that
/// occurred rather than between two that did, which is what a reader who wants to go and play
/// the recording behind a number needs.
fn nearest_rank(sorted: &[f32], fraction: f32) -> f32 {
    let rank = (fraction * sorted.len() as f32).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` trials on one side of the list, all at the same distance.
    fn trials(same_speaker: bool, distances: &[f32]) -> Vec<Trial> {
        distances
            .iter()
            .map(|&distance| Trial {
                same_speaker,
                distance,
            })
            .collect()
    }

    fn list(same: &[f32], different: &[f32]) -> Vec<Trial> {
        let mut all = trials(true, same);
        all.extend(trials(false, different));
        all
    }

    /// The shape the threshold is hoping for: every same-speaker pair well below the cut,
    /// every different-speaker pair well above it. Both rates zero, and -- the claim that
    /// costs nothing to state and is the whole point of reporting it -- a cut exists.
    #[test]
    fn a_perfectly_separated_list_has_no_errors_and_no_overlap() {
        let report = score_trials(&list(&[0.08, 0.10, 0.12], &[0.78, 0.80, 0.82]), 0.45);

        assert_eq!(report.false_accepts, 0);
        assert_eq!(report.false_rejects, 0);
        assert_eq!(report.false_accept_rate, Some(0.0));
        assert_eq!(report.false_reject_rate, Some(0.0));
        assert_eq!(
            report.overlap, None,
            "these two populations do not interleave"
        );

        let equal_error = report.equal_error.unwrap();
        assert_eq!(equal_error.rate, 0.0);
        // Anywhere in the gap would do; the sweep can only report cuts it saw, and the
        // smallest one with both rates at zero is the lowest different-speaker distance.
        assert_eq!(equal_error.threshold, 0.78);

        let zero = report.zero_false_accept.unwrap();
        assert_eq!(zero.threshold, 0.78);
        assert_eq!(
            zero.false_reject_rate, 0.0,
            "separation means never misattributing costs nothing"
        );
    }

    /// The null result, and the one this tool exists to be able to state: two populations
    /// spanning the same range, where no cut is better than a coin toss.
    #[test]
    fn a_totally_overlapping_list_says_so_and_lands_near_a_coin_toss() {
        let both = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let report = score_trials(&list(&both, &both), 0.45);

        let (min_different, max_same) = report.overlap.expect("these populations interleave");
        assert_eq!((min_different, max_same), (0.2, 0.7));

        let equal_error = report.equal_error.unwrap();
        assert!(
            (equal_error.rate - 0.5).abs() < 0.1,
            "identical populations cannot be told apart: {equal_error:?}"
        );

        // Half of each side falls the wrong way at the cut in the middle of both.
        assert_eq!(report.false_accepts, 3);
        assert_eq!(report.false_rejects, 3);
        assert_eq!(report.false_accept_rate, Some(0.5));
        assert_eq!(report.false_reject_rate, Some(0.5));
    }

    /// One pair each side, so the percentile arithmetic cannot hide behind a large `n`: with a
    /// single value every quantile is that value, and nothing may divide by `count - 1`.
    #[test]
    fn one_pair_on_each_side_still_produces_a_whole_report() {
        let report = score_trials(&list(&[0.11], &[0.66]), 0.45);

        let same = report.same.unwrap();
        assert_eq!(
            (
                same.count,
                same.min,
                same.p05,
                same.median,
                same.p95,
                same.max,
                same.mean
            ),
            (1, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11)
        );
        assert_eq!(report.different.unwrap().count, 1);
        assert_eq!(report.false_accept_rate, Some(0.0));
        assert_eq!(report.false_reject_rate, Some(0.0));
        assert_eq!(report.equal_error.unwrap().rate, 0.0);
    }

    /// An empty manifest, or one whose every item was dropped. Zero counts and `None`
    /// everywhere -- and specifically not NaN, which would propagate through every printed
    /// line as a plausible-looking blank rather than as the missing measurement it is.
    #[test]
    fn an_empty_trial_list_reports_nothing_rather_than_nan() {
        let report = score_trials(&[], 0.45);

        assert_eq!(report.same, None);
        assert_eq!(report.different, None);
        assert_eq!(report.false_accepts, 0);
        assert_eq!(report.false_rejects, 0);
        assert_eq!(report.false_accept_rate, None);
        assert_eq!(report.false_reject_rate, None);
        assert_eq!(report.overlap, None);
        assert_eq!(report.equal_error, None);
        assert_eq!(report.zero_false_accept, None);
        assert!(!report.threshold.is_nan());
    }

    /// One side empty is the manifest mistake worth naming: one speaker means no
    /// different-speaker pairs, and a false-accept rate of 0.0 over nothing would read as the
    /// best possible result rather than as a broken measurement.
    #[test]
    fn a_list_with_only_one_kind_of_pair_reports_no_rate_for_the_other() {
        let same_only = score_trials(&list(&[0.1, 0.2], &[]), 0.45);
        assert_eq!(same_only.false_accept_rate, None);
        assert_eq!(same_only.false_reject_rate, Some(0.0));
        assert_eq!(same_only.equal_error, None);
        assert_eq!(same_only.zero_false_accept, None);
        assert_eq!(same_only.overlap, None);

        let different_only = score_trials(&list(&[], &[0.1, 0.2]), 0.45);
        assert_eq!(different_only.false_accept_rate, Some(1.0));
        assert_eq!(different_only.false_reject_rate, None);
        assert_eq!(different_only.equal_error, None);
    }

    /// The boundary, which is the one convention that has to match `best_match` exactly: it
    /// accepts strictly below the cut, so a pair sitting on it is a rejection. Off by one `<`
    /// here and every rate this tool prints describes a decision rule the code does not use.
    #[test]
    fn a_pair_exactly_on_the_threshold_is_rejected_the_way_identification_rejects_it() {
        let report = score_trials(&list(&[0.45], &[0.45]), 0.45);

        assert_eq!(
            report.false_rejects, 1,
            "a same-speaker pair exactly at the cut is not accepted"
        );
        assert_eq!(
            report.false_accepts, 0,
            "a different-speaker pair exactly at the cut is not accepted either"
        );
    }

    /// Two medians, so neither is right by accident: an odd count takes the middle value and
    /// an even count takes the mean of the two middles. The same rule `cluster-speaker-track`
    /// has always used, restated here because this is now the only place it is implemented.
    #[test]
    fn the_median_takes_the_middle_value_and_the_mean_of_two_middles() {
        let odd = Spread::of(&[0.4, 0.1, 0.3, 0.2, 0.5]).unwrap();
        assert_eq!(odd.median, 0.3);
        assert_eq!((odd.min, odd.max), (0.1, 0.5));

        let even = Spread::of(&[0.4, 0.1, 0.3, 0.2]).unwrap();
        assert_eq!(even.median, 0.25);

        assert_eq!(Spread::of(&[]), None);
    }

    /// Nearest rank, on a population where the two conventions differ visibly: with 20 values
    /// `p05` is the first and `p95` is the last, and both are values that occurred.
    #[test]
    fn the_percentiles_land_on_values_that_actually_occurred() {
        let population: Vec<f32> = (0..20).map(|i| i as f32 / 100.0).collect();
        let spread = Spread::of(&population).unwrap();

        assert_eq!(spread.p05, 0.0);
        assert_eq!(spread.p95, 0.18);
        assert!(
            (spread.mean - 0.095).abs() < 1e-6,
            "mean was {}",
            spread.mean
        );
    }

    /// An asymmetric list, so the equal-error sweep has to actually find a crossing rather
    /// than being handed one by symmetry -- and so the "never misattribute" cost is a number
    /// somebody would act on rather than zero.
    #[test]
    fn the_sweep_finds_the_crossing_and_prices_the_conservative_cut() {
        // Nine of ten same-speaker pairs are tight; one is a straggler out among the
        // different-speaker pairs, which is what a bad recording of an enrolled voice looks
        // like.
        let same: Vec<f32> = (0..9).map(|i| 0.10 + i as f32 / 100.0).collect();
        let report = score_trials(
            &list(
                &[same.as_slice(), &[0.62]].concat(),
                &[0.55, 0.60, 0.70, 0.80],
            ),
            0.45,
        );

        assert_eq!(report.false_accepts, 0);
        assert_eq!(report.false_rejects, 1, "the straggler is above 0.45");
        assert_eq!(report.overlap, Some((0.55, 0.62)));

        let zero = report.zero_false_accept.unwrap();
        assert_eq!(zero.threshold, 0.55);
        assert!(
            (zero.false_reject_rate - 0.1).abs() < 1e-6,
            "one same-speaker pair in ten is above 0.55, was {}",
            zero.false_reject_rate
        );

        // Nothing can do better than that one straggler here, so the crossing sits where the
        // rates are 10% and 0%.
        let equal_error = report.equal_error.unwrap();
        assert!((equal_error.rate - 0.05).abs() < 1e-6, "{equal_error:?}");
    }
}
