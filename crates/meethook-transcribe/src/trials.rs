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
//! # Two altitudes, not one
//!
//! [`score_trials`] takes `&[Trial]` and knows nothing about who was recorded when: a
//! distance and a boolean. [`policy_sweep`] takes [`PolicyItem`]s -- one person as they
//! sounded in one session -- because the question it answers is *what a reference should be
//! built out of*, which cannot be asked of a distance. That is still not plumbing: no files,
//! no models, no session directories, and [`crate::reference_duration_sweep`] already takes
//! clusters on the same terms. Everything below the enumeration is the arithmetic above it.
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
use std::collections::BTreeMap;

use meethook_session::{EnrolledSpeaker, EnrolledSpeakers, SpeakerCluster};

use crate::identify::identify_clusters;
use crate::speakers::group_mean;

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

/// A 95% Wilson score interval for `count` of `population`, as fractions.
///
/// Here because a rate over 36 pairs is one significant figure and a report that prints it as
/// "5.6%" invites a reader to act on the second one. `IDENTIFY_DISTANCE`'s own documentation
/// quotes 2/36 as "roughly 1.5%-18%", computed by hand outside the code for TASK-014.04; this
/// is that arithmetic, tested against that published pair, so the next report does not have to
/// be believed either.
///
/// Wilson rather than the textbook normal approximation because every interesting count here is
/// near an edge: 0 false accepts in 2170 gives `0 +/- 0` under the normal approximation, which
/// reads as certainty and is the one claim the measurement cannot make. At a count of zero
/// Wilson's upper limit lands close to the rule of three (`3/n`), which is the usual quotation
/// for "none observed" and is what to compare it against.
///
/// [`None`] for an empty population, or a count larger than it: no fraction exists to bound.
/// The result is clamped to `[0, 1]`.
pub fn wilson_interval(count: usize, population: usize) -> Option<(f32, f32)> {
    if population == 0 || count > population {
        return None;
    }
    // 1.96: the two-sided 95% normal quantile, the convention every published rate of this
    // kind uses. Not a parameter, because a caller choosing 90% for one row of a table and 95%
    // for another is the failure this function exists to prevent.
    let z = 1.96f64;
    let n = population as f64;
    let p = count as f64 / n;

    let denominator = 1.0 + z * z / n;
    let centre = (p + z * z / (2.0 * n)) / denominator;
    let margin = z / denominator * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt();

    // Both edges are exact at the extremes -- Wilson's interval for 0 of n starts at 0 and for
    // n of n ends at 1 -- so they are spelled rather than left to `centre -/+ margin`, whose
    // f64 residue would print as a lower limit of 1e-19 and read as a measurement.
    Some((
        match count {
            0 => 0.0,
            _ => (centre - margin).clamp(0.0, 1.0) as f32,
        },
        match count == population {
            true => 1.0,
            false => (centre + margin).clamp(0.0, 1.0) as f32,
        },
    ))
}

// ---------------------------------------------------------------------------------------
// What a person's reference is made of, when they have been named more than once
// ---------------------------------------------------------------------------------------

/// One person as they sounded in one recording session.
///
/// The unit an enrolled reference is built from, described with nothing else attached: no
/// paths, no durations, no audio. `session` identifies the recording *occasion* -- a
/// LibriSpeech chapter, a LibriVox project, one meeting -- and the only thing done with it is
/// equality, because "these two recordings are the same occasion" is the one fact that
/// disqualifies a pair from being evidence about [`crate::IDENTIFY_DISTANCE`].
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyItem {
    pub speaker: String,
    pub session: String,

    /// Unit length, as every embedding in this crate is by contract.
    pub embedding: Vec<f32>,
}

/// What `speakers.json` holds after one person has been named from two different recordings.
///
/// The three answers TASK-027 poses, stated as arithmetic so they can be scored against each
/// other. Given the recordings in the order the user confirmed them, `a` then `b`:
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReferencePolicy {
    /// `[b]`. Newest wins, which is what ships today: `enroll` assigns
    /// `entry.embedding = cluster.embedding.clone()` over the entry already filed under that
    /// name. **The number every other arm is reported against.**
    Replace,

    /// `[normalize((a + b) / 2)]`, via `speakers::group_mean` rather than a second
    /// implementation of it. One vector, so identification stays a single argmax and
    /// `speakers.json` keeps its shape.
    Average,

    /// `[a, b]`: two rows under one name, a person's score being the *nearest* of their
    /// references.
    ///
    /// Needs no schema change and no `best_match` change to measure, which is a fact about
    /// the shipped code rather than a result of any run: `EnrolledSpeakers::speakers` is a
    /// plain `Vec` with no uniqueness constraint, `best_match` is an argmax over rows that
    /// returns the winning row's *name*, and [`crate::identify_clusters`] groups contenders by
    /// name -- so two rows called "Ryan" already behave as one contender scored at its nearest
    /// reference.
    Set,
}

impl ReferencePolicy {
    /// All three arms, in report order, with the shipped behaviour first.
    pub const ALL: [ReferencePolicy; 3] = [
        ReferencePolicy::Replace,
        ReferencePolicy::Average,
        ReferencePolicy::Set,
    ];

    /// A one-word name for a table header or a row label.
    pub fn label(self) -> &'static str {
        match self {
            ReferencePolicy::Replace => "replace",
            ReferencePolicy::Average => "average",
            ReferencePolicy::Set => "set",
        }
    }

    /// Whether the reference this policy builds is indifferent to which recording the user
    /// happened to confirm second.
    ///
    /// Not a curiosity: it is the denominator. An enumeration that scores both orderings of
    /// every pair -- which it must, because [`ReferencePolicy::Replace`] is the only arm that
    /// is not symmetric and the order is arbitrary -- measures a symmetric arm twice. Its
    /// counts are comparable with `Replace`'s over the ordered combinations, and its
    /// *interval* has to be taken on half of them or it is narrowed by about `sqrt(2)`: a bare
    /// percentage wearing an interval's clothing.
    pub fn symmetric(self) -> bool {
        !matches!(self, ReferencePolicy::Replace)
    }
}

/// The reference rows this policy would file under one name, given that person's confirmed
/// recordings in confirmation order.
///
/// Total and general: defined for any number of recordings, and for one recording all three
/// arms collapse to the same single vector -- which is what makes "every speaker enrolled from
/// one session is identical across arms" true by construction rather than by a special case.
///
/// Empty when there is nothing to build a reference from: no recordings at all, or a group
/// whose members exactly cancel, which `speakers::group_mean` declines rather than
/// normalizing a zero vector into a confident distance of 1.0 to everything. Unreachable for
/// real voices, trivially reachable in a test, and counted rather than panicked on.
pub fn policy_references(policy: ReferencePolicy, confirmed: &[&[f32]]) -> Vec<Vec<f32>> {
    match policy {
        ReferencePolicy::Replace => confirmed
            .last()
            .map(|newest| vec![newest.to_vec()])
            .unwrap_or_default(),
        ReferencePolicy::Average => group_mean(confirmed)
            .map(|(unit, _)| vec![unit])
            .unwrap_or_default(),
        ReferencePolicy::Set => confirmed.iter().map(|one| one.to_vec()).collect(),
    }
}

/// What [`crate::identify_clusters`] decided about one probe in one simulated database.
///
/// The same three outcomes [`crate::Verdict`] names, and deliberately not that type: names
/// here are the corpus's own speaker labels rather than cluster ids, and a `String` in a
/// `Misattributed` is the whole value of reading it. Its fourth outcome cannot arise here --
/// each simulation identifies exactly one cluster, so there is nobody for the heard-at-once
/// veto to award the name to first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyVerdict {
    /// Identified as its own speaker: the only outcome a reference is enrolled for.
    Correct,

    /// Identified as somebody else, naming who took it. One person's words under another
    /// person's name, which is the failure `IDENTIFY_DISTANCE` biases low to avoid.
    Misattributed(String),

    /// Nothing was awarded: the nearest reference in the database is at or beyond
    /// [`crate::IDENTIFY_DISTANCE`]. An `Unknown N` the user fixes in ten seconds.
    Rejected,
}

/// Closed-set outcomes over one arm's combinations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClosedSet {
    pub correct: usize,
    pub misattributed: usize,
    pub rejected: usize,
}

impl ClosedSet {
    /// Every combination scored, which is the denominator of all three rates.
    pub fn scored(&self) -> usize {
        self.correct + self.misattributed + self.rejected
    }
}

/// One probe that identification put under a name that is not its owner's.
#[derive(Debug, Clone, PartialEq)]
pub struct Misattribution {
    pub speaker: String,
    pub probe_session: String,

    /// The sessions the owner's reference was built from, in confirmation order -- so a
    /// suspicious row can be traced back to the two recordings that produced it.
    pub built_from: Vec<String>,

    pub named: String,
}

/// One impostor construction's outcomes: the same shape for both, so a reader comparing them
/// is comparing like with like.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ArmReport {
    pub closed: ClosedSet,

    /// Probes given *some* name when their own speaker was removed from the database. An
    /// unenrolled voice named as somebody, which is the harm the open set prices.
    pub open_false_alarms: usize,

    /// Impostor recordings refused for coming from the probe's own recording session.
    ///
    /// Counted once per probe rather than once per combination, because an impostor's
    /// reference depends only on which probe is held out and is built once for all of that
    /// probe's combinations. A count multiplied by the combinations per probe would not
    /// reconcile with the corpus.
    pub references_refused: usize,

    /// Impostors left with no usable recording by that refusal, and therefore absent from
    /// that probe's database. Also once per probe.
    pub impostors_dropped: usize,

    pub misattributions: Vec<Misattribution>,

    /// The open-set false alarms, each naming who the unenrolled voice was handed to.
    pub false_alarms: Vec<Misattribution>,
}

/// Everything one policy scored over one population.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyReport {
    pub policy: ReferencePolicy,

    /// Ordered probe-and-reference-pair combinations scored: every held-out probe, every
    /// unordered pair of that speaker's remaining sessions, and **both orderings** of it.
    pub combinations: usize,

    /// Distinct measurements among them: half of [`PolicyReport::combinations`] for a
    /// symmetric arm, all of them for [`ReferencePolicy::Replace`]. **The denominator any
    /// interval belongs on** -- see [`ReferencePolicy::symmetric`].
    pub distinct_combinations: usize,

    /// **The controlled comparison.** Every speaker other than the probe's owner holds one
    /// reference from one session, identically across all three arms, so the only thing
    /// varying between arms is the target person's own reference shape.
    ///
    /// Its open-set count is a property of the corpus rather than of the policy: removing the
    /// owner leaves a database of single-session impostors that does not depend on the arm at
    /// all, so all three arms must report the same number here and a difference is a bug.
    pub controlled: ArmReport,

    /// The same simulation with *every* impostor also built under the policy, which is what a
    /// real user produces by naming several people twice. Not a substitute for
    /// [`PolicyReport::controlled`]: it varies two things at once, and it is reported because
    /// a disagreement between the two is itself the result.
    pub policy_impostors: ArmReport,

    /// The distance populations, with every speaker's reference built under the policy from
    /// their first two sessions in cache order. Same-speaker distances are each probe to the
    /// nearest of its owner's references; different-speaker distances are each probe to the
    /// nearest of one impostor's references -- one trial per person, which is what argmax
    /// sees, rather than one per row.
    ///
    /// [`TrialReport::zero_false_accept`] on this is the largest misattribution-free cut and
    /// its false-reject cost.
    pub distances: TrialReport,

    /// Per probe, the nearest impostor of all: the quantity that prices the risk a blend of
    /// two people creates for everybody else. Its `min` is the closest impostor pair in the
    /// whole population.
    pub nearest_impostor: Option<Spread>,

    /// Probes the distance populations were taken over. Independent of the policy.
    pub distance_probes: usize,

    /// `(probe, impostor)` pairs refused because the probe's session contributed to that
    /// impostor's reference -- both of its sessions, under an arm whose single vector carries
    /// two occasions.
    pub impostor_pairs_refused: usize,

    /// References that could not be built at all: a group whose members exactly cancel.
    /// Target-side, counted per combination; impostor-side, per probe. Zero for real voices,
    /// and printed so that it cannot be zero silently.
    pub declines: usize,
}

/// The population three policies were scored over, and what each scored.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicySweep {
    pub items: usize,
    pub speakers: usize,

    /// Sessions per speaker, over every speaker in the population.
    pub sessions_per_speaker: Option<Spread>,

    /// Speakers holding the three sessions a two-reference arm needs -- two to enrol from and
    /// one to probe with -- in name order. The population the closed-set counts are over.
    pub targets: Vec<String>,

    /// Distinct embedding lengths in the population, ascending. More than one means more than
    /// one embedding model and two spaces that cannot be compared, and nothing is scored.
    pub dimensions: Vec<usize>,

    /// The cut [`PolicyReport::distances`] was scored at, carried so that whatever prints it
    /// cannot state a different one. Identification's own verdicts are always taken at
    /// [`crate::IDENTIFY_DISTANCE`] and nothing here can move them.
    pub threshold: f32,

    /// One per [`ReferencePolicy::ALL`], in that order.
    pub reports: Vec<PolicyReport>,
}

/// Scores all three reference policies over one population of items.
///
/// # The enumeration
///
/// For each speaker with three or more sessions, for each held-out probe session, for each
/// unordered pair of the remaining sessions, and for **both orderings** of that pair. Both
/// orderings because [`ReferencePolicy::Replace`] is the only arm that is not symmetric in the
/// two recordings, and which one the user answered second is arbitrary.
///
/// # No trial pairs two recordings of one occasion
///
/// The probe is held out of its own reference by construction. On the impostor side the rule
/// has teeth -- a LibriVox project is read by several volunteers, so one session id can name
/// several speakers -- and it is applied in both places, counted, and reported:
/// [`ArmReport::references_refused`] for the identification simulation and
/// [`PolicyReport::impostor_pairs_refused`] for the distance populations.
///
/// # Every verdict comes from `identify_clusters`
///
/// Never a bare distance comparison. Identification is argmax *then* threshold, so a reference
/// clearing the cut while a nearer one wins is not a match, and the shipped function is the
/// only construction that cannot disagree with the shipped decision.
///
/// Reads no models and no files, and is pure: the same items give the same sweep. Cost is dot
/// products over embeddings the caller already has.
///
/// Every degenerate population is an empty report rather than a panic: no speaker with three
/// sessions, one speaker, no items, an empty embedding, or a mixture of embedding lengths.
pub fn policy_sweep(items: &[PolicyItem], threshold: f32) -> PolicySweep {
    let mut by_speaker: BTreeMap<&str, Vec<&PolicyItem>> = BTreeMap::new();
    for item in items {
        by_speaker
            .entry(item.speaker.as_str())
            .or_default()
            .push(item);
    }

    let per_speaker: Vec<f32> = by_speaker.values().map(|its| its.len() as f32).collect();
    let mut dimensions: Vec<usize> = items.iter().map(|item| item.embedding.len()).collect();
    dimensions.sort_unstable();
    dimensions.dedup();

    // One embedding model, or the dot products below compare unrelated spaces -- the same
    // refusal `best_match` makes for one stale row, made for a whole population.
    let comparable = dimensions.len() == 1 && dimensions[0] > 0;

    PolicySweep {
        items: items.len(),
        speakers: by_speaker.len(),
        sessions_per_speaker: Spread::of(&per_speaker),
        targets: by_speaker
            .iter()
            .filter(|(_, sessions)| sessions.len() >= 3)
            .map(|(speaker, _)| (*speaker).to_string())
            .collect(),
        dimensions,
        threshold,
        reports: ReferencePolicy::ALL
            .iter()
            .map(|&policy| match comparable {
                true => score_policy(policy, &by_speaker, threshold),
                false => empty_report(policy, threshold),
            })
            .collect(),
    }
}

fn empty_report(policy: ReferencePolicy, threshold: f32) -> PolicyReport {
    PolicyReport {
        policy,
        combinations: 0,
        distinct_combinations: 0,
        controlled: ArmReport::default(),
        policy_impostors: ArmReport::default(),
        distances: score_trials(&[], threshold),
        nearest_impostor: None,
        distance_probes: 0,
        impostor_pairs_refused: 0,
        declines: 0,
    }
}

fn score_policy(
    policy: ReferencePolicy,
    by_speaker: &BTreeMap<&str, Vec<&PolicyItem>>,
    threshold: f32,
) -> PolicyReport {
    let mut report = empty_report(policy, threshold);

    for (&target, sessions) in by_speaker {
        if sessions.len() < 3 {
            continue;
        }
        for (probe_at, probe) in sessions.iter().enumerate() {
            let others: Vec<&PolicyItem> = sessions
                .iter()
                .enumerate()
                .filter(|(at, _)| *at != probe_at)
                .map(|(_, item)| *item)
                .collect();

            // Both impostor databases are built once per probe: they depend on which session
            // is held out and on the policy, and on nothing that varies within the pair loop.
            let single = single_session_impostors(by_speaker, target, &probe.session);
            let blended = policy_impostors(policy, by_speaker, target, &probe.session);
            report.controlled.references_refused += single.refused;
            report.controlled.impostors_dropped += single.dropped;
            report.policy_impostors.references_refused += blended.refused;
            report.policy_impostors.impostors_dropped += blended.dropped;
            report.declines += blended.declined;

            for first in 0..others.len() {
                for second in 0..others.len() {
                    if first == second {
                        continue;
                    }
                    report.combinations += 1;

                    let confirmed = [
                        others[first].embedding.as_slice(),
                        others[second].embedding.as_slice(),
                    ];
                    let mine = policy_references(policy, &confirmed);
                    if mine.is_empty() {
                        report.declines += 1;
                        continue;
                    }
                    let built_from = vec![
                        others[first].session.clone(),
                        others[second].session.clone(),
                    ];

                    score_arm(
                        &mut report.controlled,
                        probe,
                        target,
                        &mine,
                        &single.rows,
                        &built_from,
                    );
                    score_arm(
                        &mut report.policy_impostors,
                        probe,
                        target,
                        &mine,
                        &blended.rows,
                        &built_from,
                    );
                }
            }
        }
    }

    report.distinct_combinations = match policy.symmetric() {
        // Always even: every unordered pair contributes exactly two ordered combinations.
        true => report.combinations / 2,
        false => report.combinations,
    };
    distance_populations(policy, by_speaker, threshold, &mut report);
    report
}

/// Scores one probe against one database, closed set and then open set.
///
/// `mine` are the target's reference rows under the policy; `impostors` is everybody else's,
/// already built. The open set is the same database with **every row filed under the target's
/// name** removed -- removal is by name, so both rows of a reference set go together, which is
/// what the shipped correction path would also have to do.
fn score_arm(
    arm: &mut ArmReport,
    probe: &PolicyItem,
    owner: &str,
    mine: &[Vec<f32>],
    impostors: &[EnrolledSpeaker],
    built_from: &[String],
) {
    let mut database: Vec<EnrolledSpeaker> = mine
        .iter()
        .map(|embedding| EnrolledSpeaker {
            name: owner.to_string(),
            embedding: embedding.clone(),
        })
        .collect();
    database.extend_from_slice(impostors);

    let record = |named: String| Misattribution {
        speaker: probe.speaker.clone(),
        probe_session: probe.session.clone(),
        built_from: built_from.to_vec(),
        named,
    };

    match decide(&probe.embedding, &database, owner) {
        PolicyVerdict::Correct => arm.closed.correct += 1,
        PolicyVerdict::Misattributed(named) => {
            arm.closed.misattributed += 1;
            arm.misattributions.push(record(named));
        }
        PolicyVerdict::Rejected => arm.closed.rejected += 1,
    }

    // With the owner gone no verdict can be `Correct`, so any name at all is a false alarm.
    let strangers: Vec<EnrolledSpeaker> = database
        .into_iter()
        .filter(|row| row.name != owner)
        .collect();
    if let PolicyVerdict::Misattributed(named) = decide(&probe.embedding, &strangers, owner) {
        arm.open_false_alarms += 1;
        arm.false_alarms.push(record(named));
    }
}

/// What the shipped decision would put on this voice.
///
/// Synthetic database, real function. The cluster carries only an embedding, because that and
/// the cannot-link list are all identification reads -- and one cluster at a time means there
/// is nobody it could have been heard at once with, so no veto can fire and
/// [`PolicyVerdict`] has three outcomes rather than four.
fn decide(probe: &[f32], database: &[EnrolledSpeaker], owner: &str) -> PolicyVerdict {
    let enrolled = EnrolledSpeakers::new(database.to_vec());
    let cluster = SpeakerCluster {
        id: 0,
        embedding: probe.to_vec(),
        speech_seconds: 0.0,
        first_spoke_seconds: 0.0,
        heard_at_once_with: Vec::new(),
        representatives: Vec::new(),
    };

    match identify_clusters(std::slice::from_ref(&cluster), &enrolled).remove(&0) {
        Some(identification) if identification.name == owner => PolicyVerdict::Correct,
        Some(identification) => PolicyVerdict::Misattributed(identification.name),
        None => PolicyVerdict::Rejected,
    }
}

/// One database's impostor rows, and what building them refused.
struct Impostors {
    rows: Vec<EnrolledSpeaker>,
    refused: usize,
    dropped: usize,
    declined: usize,
}

/// Every speaker but `target`, enrolled from **one** session: the controlled comparison.
///
/// Their reference is their first session in cache order whose id differs from the probe's, so
/// no pair here shares a recording occasion. A speaker with no such session is absent from
/// this database entirely, which is counted rather than quietly reducing the competition.
///
/// Identical for all three policies by construction, which is what makes the three arms'
/// closed-set counts a comparison rather than three unrelated numbers.
fn single_session_impostors(
    by_speaker: &BTreeMap<&str, Vec<&PolicyItem>>,
    target: &str,
    probe_session: &str,
) -> Impostors {
    let mut built = Impostors {
        rows: Vec::new(),
        refused: 0,
        dropped: 0,
        declined: 0,
    };
    for (&speaker, sessions) in by_speaker {
        if speaker == target {
            continue;
        }
        built.refused += sessions
            .iter()
            .filter(|item| item.session == probe_session)
            .count();
        match sessions.iter().find(|item| item.session != probe_session) {
            Some(item) => built.rows.push(EnrolledSpeaker {
                name: speaker.to_string(),
                embedding: item.embedding.clone(),
            }),
            None => built.dropped += 1,
        }
    }
    built
}

/// Every speaker but `target`, enrolled under the policy from their first two usable sessions.
///
/// The database a real user produces: several people named twice, not one. Same
/// within-session rule, so an impostor never contributes a reference built from the probe's
/// own recording.
fn policy_impostors(
    policy: ReferencePolicy,
    by_speaker: &BTreeMap<&str, Vec<&PolicyItem>>,
    target: &str,
    probe_session: &str,
) -> Impostors {
    let mut built = Impostors {
        rows: Vec::new(),
        refused: 0,
        dropped: 0,
        declined: 0,
    };
    for (&speaker, sessions) in by_speaker {
        if speaker == target {
            continue;
        }
        let usable: Vec<&PolicyItem> = sessions
            .iter()
            .filter(|item| item.session != probe_session)
            .copied()
            .collect();
        built.refused += sessions.len() - usable.len();
        if usable.is_empty() {
            built.dropped += 1;
            continue;
        }

        let confirmed: Vec<&[f32]> = usable
            .iter()
            .take(2)
            .map(|item| item.embedding.as_slice())
            .collect();
        let references = policy_references(policy, &confirmed);
        if references.is_empty() {
            built.declined += 1;
            built.dropped += 1;
            continue;
        }
        for embedding in references {
            built.rows.push(EnrolledSpeaker {
                name: speaker.to_string(),
                embedding,
            });
        }
    }
    built
}

/// The distance populations, with *every* speaker's reference built under the policy.
///
/// The arm that prices what a blend of two people costs everybody else, rather than what one
/// person's reference shape costs them. Each speaker's reference is their first two sessions in
/// cache order -- fixed once per policy, not re-derived per combination, which is why these
/// counts do not reconcile with the closed-set ones and are labelled apart from them. Probes
/// are every session that reference did not consume, so nothing is compared with a vector it
/// contributed to.
///
/// One different-speaker trial per *person*, at the nearest of their references, because that
/// is what an argmax over rows sees. One row per reference would give a reference set twice the
/// population and a false-accept rate that is not comparable with the other arms'.
fn distance_populations(
    policy: ReferencePolicy,
    by_speaker: &BTreeMap<&str, Vec<&PolicyItem>>,
    threshold: f32,
    report: &mut PolicyReport,
) {
    struct Enrolled<'a> {
        speaker: &'a str,
        references: Vec<Vec<f32>>,
        /// The sessions behind those references: what a probe may not share.
        sessions: Vec<&'a str>,
    }

    let mut enrolled: Vec<Enrolled> = Vec::new();
    for (&speaker, sessions) in by_speaker {
        let taken: Vec<&PolicyItem> = sessions.iter().take(2).copied().collect();
        let confirmed: Vec<&[f32]> = taken.iter().map(|item| item.embedding.as_slice()).collect();
        let references = policy_references(policy, &confirmed);
        if references.is_empty() {
            report.declines += 1;
            continue;
        }
        enrolled.push(Enrolled {
            speaker,
            references,
            sessions: taken.iter().map(|item| item.session.as_str()).collect(),
        });
    }

    let mut trials: Vec<Trial> = Vec::new();
    let mut nearest_impostors: Vec<f32> = Vec::new();
    for (&speaker, sessions) in by_speaker {
        for probe in sessions.iter().skip(2) {
            let Some(mine) = enrolled.iter().find(|held| held.speaker == speaker) else {
                continue;
            };
            let Some(own) = nearest(&probe.embedding, &mine.references) else {
                continue;
            };
            report.distance_probes += 1;
            trials.push(Trial {
                same_speaker: true,
                distance: own,
            });

            let mut closest: Option<f32> = None;
            for impostor in enrolled.iter().filter(|held| held.speaker != speaker) {
                if impostor.sessions.contains(&probe.session.as_str()) {
                    report.impostor_pairs_refused += 1;
                    continue;
                }
                if let Some(apart) = nearest(&probe.embedding, &impostor.references) {
                    trials.push(Trial {
                        same_speaker: false,
                        distance: apart,
                    });
                    closest = Some(closest.map_or(apart, |held: f32| held.min(apart)));
                }
            }
            if let Some(closest) = closest {
                nearest_impostors.push(closest);
            }
        }
    }

    report.distances = score_trials(&trials, threshold);
    report.nearest_impostor = Spread::of(&nearest_impostors);
}

/// The nearest of one person's references to a voice, which is that person's score.
fn nearest(probe: &[f32], references: &[Vec<f32>]) -> Option<f32> {
    references
        .iter()
        .map(|reference| distance(probe, reference))
        .min_by(f32::total_cmp)
}

/// Cosine distance between two unit-length voices: the dot product `best_match` takes, so a
/// distance printed beside a verdict cannot disagree with it.
///
/// Different lengths or an empty vector give [`f32::INFINITY`] rather than a truncated `zip`'s
/// plausible cosine -- the same refusal `best_match` and [`crate::reference_duration_sweep`]
/// make, for the same reason: two spaces are not comparable at all.
fn distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }
    1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identify::IDENTIFY_DISTANCE;

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

    // -----------------------------------------------------------------------------------
    // Wilson intervals
    // -----------------------------------------------------------------------------------

    /// The interval `IDENTIFY_DISTANCE`'s documentation already quotes, computed by hand for
    /// TASK-014.04: 2 same-speaker rejections in 36 pairs is "roughly 1.5%-18%". If this
    /// function disagreed with the published pair, one of the two would be wrong and nobody
    /// reading either could tell which.
    #[test]
    fn the_interval_agrees_with_the_one_identify_distance_already_publishes() {
        let (low, high) = wilson_interval(2, 36).unwrap();

        assert!((low - 0.0154).abs() < 5e-4, "lower was {low}");
        assert!((high - 0.1814).abs() < 5e-4, "upper was {high}");
    }

    /// None observed is not certainty. The textbook normal approximation gives `0 +/- 0` here,
    /// which reads as "this cannot happen"; Wilson's upper limit lands near the rule of three,
    /// `3/n`, which is the honest reading of a zero count.
    #[test]
    fn a_zero_count_still_has_an_upper_limit_near_the_rule_of_three() {
        let (low, high) = wilson_interval(0, 2170).unwrap();

        assert_eq!(low, 0.0, "a zero count cannot have a positive lower limit");
        let rule_of_three = 3.0 / 2170.0;
        assert!(
            high > rule_of_three && high < 3.0 * rule_of_three,
            "upper {high} should sit near the rule of three {rule_of_three}"
        );
    }

    /// Both edges are clamped, and an interval over nothing does not exist.
    #[test]
    fn an_interval_needs_a_population_and_stays_inside_zero_and_one() {
        assert_eq!(wilson_interval(0, 0), None);
        assert_eq!(wilson_interval(5, 4), None, "no rate above 1 to bound");

        let (low, high) = wilson_interval(192, 192).unwrap();
        assert!(low > 0.97, "192 of 192 is not consistent with a low rate");
        assert_eq!(high, 1.0);
    }

    // -----------------------------------------------------------------------------------
    // Reference policies
    // -----------------------------------------------------------------------------------

    /// A unit vector `degrees` off the first axis, so a test names the distance it means --
    /// `1 - cos(degrees)` -- rather than a pile of decimals. The same helper `speakers.rs`,
    /// `adoption.rs` and `reference.rs` state their fixtures with.
    fn at(degrees: f32) -> Vec<f32> {
        let radians = degrees.to_radians();
        vec![radians.cos(), radians.sin(), 0.0, 0.0]
    }

    fn item(speaker: &str, session: &str, degrees: f32) -> PolicyItem {
        PolicyItem {
            speaker: speaker.to_string(),
            session: session.to_string(),
            embedding: at(degrees),
        }
    }

    fn report(sweep: &PolicySweep, policy: ReferencePolicy) -> &PolicyReport {
        sweep
            .reports
            .iter()
            .find(|report| report.policy == policy)
            .expect("every policy is reported")
    }

    /// One recording collapses all three arms to the same single vector, which is what makes
    /// "every speaker enrolled once is identical across arms" a fact about the code rather
    /// than a special case somebody has to remember.
    #[test]
    fn one_confirmed_recording_gives_all_three_policies_the_same_reference() {
        let only = at(17.0);

        for policy in ReferencePolicy::ALL {
            let references = policy_references(policy, &[only.as_slice()]);
            assert_eq!(references, vec![only.clone()], "{}", policy.label());
        }
    }

    /// Replacement is what ships, and what every other arm is scored against: the newest
    /// answer, and nothing of the first one.
    #[test]
    fn replacement_keeps_only_the_recording_confirmed_last() {
        let (first, second) = (at(0.0), at(40.0));

        let references = policy_references(
            ReferencePolicy::Replace,
            &[&first, &second].map(Vec::as_slice),
        );

        assert_eq!(references, vec![second]);
    }

    /// Averaging must be the arithmetic `EnrolledSpeaker::embedding` documents -- unweighted
    /// mean of unit vectors, normalized *after* averaging -- and it is asserted by calling the
    /// same `group_mean` rather than by restating it, because a second implementation is a
    /// place for the two to drift invisibly.
    #[test]
    fn averaging_is_the_group_mean_enrollment_already_stores() {
        let (first, second) = (at(0.0), at(40.0));
        let confirmed = [first.as_slice(), second.as_slice()];

        let references = policy_references(ReferencePolicy::Average, &confirmed);

        let (expected, _) = group_mean(&confirmed).unwrap();
        assert_eq!(references, vec![expected]);
        // And it is the bisector, not either input: 20 degrees from both.
        let apart = distance(&references[0], &first);
        assert!(
            (apart - (1.0 - 20.0f32.to_radians().cos())).abs() < 1e-6,
            "{apart}"
        );
    }

    /// The reference set keeps both, in the order they were confirmed.
    #[test]
    fn the_reference_set_keeps_both_recordings() {
        let (first, second) = (at(0.0), at(40.0));

        let references =
            policy_references(ReferencePolicy::Set, &[&first, &second].map(Vec::as_slice));

        assert_eq!(references, vec![first, second]);
    }

    /// The symmetry the enumeration's double count rests on, asserted rather than argued: the
    /// two symmetric arms are indifferent to which recording was confirmed second, and
    /// replacement is not. Get this wrong and every interval on the symmetric arms is quoted
    /// over twice its real population.
    #[test]
    fn only_replacement_depends_on_which_recording_was_confirmed_second() {
        let (a, b) = (at(0.0), at(40.0));
        let forwards = [a.as_slice(), b.as_slice()];
        let backwards = [b.as_slice(), a.as_slice()];

        assert_eq!(
            policy_references(ReferencePolicy::Average, &forwards),
            policy_references(ReferencePolicy::Average, &backwards),
            "an unweighted mean of two vectors cannot depend on their order"
        );

        let mut one_way = policy_references(ReferencePolicy::Set, &forwards);
        let mut other_way = policy_references(ReferencePolicy::Set, &backwards);
        assert_ne!(
            one_way, other_way,
            "the set is stored in confirmation order"
        );
        one_way.sort_by(|x, y| x[1].total_cmp(&y[1]));
        other_way.sort_by(|x, y| x[1].total_cmp(&y[1]));
        assert_eq!(
            one_way, other_way,
            "...but it is the same set, and its score is the nearest member either way"
        );

        assert_ne!(
            policy_references(ReferencePolicy::Replace, &forwards),
            policy_references(ReferencePolicy::Replace, &backwards),
            "newest-wins is the one arm the order changes"
        );

        assert!(!ReferencePolicy::Replace.symmetric());
        assert!(ReferencePolicy::Average.symmetric() && ReferencePolicy::Set.symmetric());
    }

    /// Two exactly opposed references have no direction, and a zero vector normalized is still
    /// zero -- which would score a confident distance of 1.0 to everything. Declined and
    /// counted instead. Unreachable for real voices; one line to state here.
    #[test]
    fn two_exactly_opposed_recordings_decline_rather_than_averaging_to_nothing() {
        // Spelled out rather than via `at(180.0)`, whose sine is a rounding error rather than
        // a zero -- and this test is about the exact cancellation.
        let confirmed: [&[f32]; 2] = [&[1.0, 0.0, 0.0, 0.0], &[-1.0, 0.0, 0.0, 0.0]];

        assert!(policy_references(ReferencePolicy::Average, &confirmed).is_empty());
        assert_eq!(
            policy_references(ReferencePolicy::Set, &confirmed).len(),
            2,
            "keeping both is defined even where their mean is not"
        );
    }

    // -----------------------------------------------------------------------------------
    // The sweep
    // -----------------------------------------------------------------------------------

    /// The enumeration, on a population whose shape is countable by hand: one speaker with
    /// three sessions is 3 probes x 2 ordered pairs = 6 ordered combinations, of which 3 are
    /// distinct for a symmetric arm.
    #[test]
    fn every_probe_is_scored_against_both_orderings_of_every_remaining_pair() {
        let sweep = policy_sweep(
            &[
                item("A", "a1", 0.0),
                item("A", "a2", 8.0),
                item("A", "a3", 16.0),
                item("B", "b1", 90.0),
            ],
            IDENTIFY_DISTANCE,
        );

        assert_eq!(sweep.speakers, 2);
        assert_eq!(
            sweep.targets,
            vec!["A".to_string()],
            "only A has three sessions"
        );

        for policy in ReferencePolicy::ALL {
            let scored = report(&sweep, policy);
            assert_eq!(scored.combinations, 6, "{}", policy.label());
            assert_eq!(scored.controlled.closed.scored(), 6, "{}", policy.label());
            assert_eq!(
                scored.policy_impostors.closed.scored(),
                6,
                "{}",
                policy.label()
            );
            assert_eq!(
                scored.distinct_combinations,
                if policy.symmetric() { 3 } else { 6 },
                "{}",
                policy.label()
            );
            assert_eq!(scored.declines, 0);
        }
    }

    /// A four-session speaker alongside a three-session one, which is the shape the corpus
    /// actually has: 4 x 2 x C(3,2) = 24 and 3 x 2 x C(2,2) = 6.
    #[test]
    fn a_speaker_with_four_sessions_contributes_more_combinations_than_one_with_three() {
        let sweep = policy_sweep(
            &[
                item("A", "a1", 0.0),
                item("A", "a2", 4.0),
                item("A", "a3", 8.0),
                item("A", "a4", 12.0),
                item("B", "b1", 90.0),
                item("B", "b2", 94.0),
                item("B", "b3", 98.0),
            ],
            IDENTIFY_DISTANCE,
        );

        assert_eq!(report(&sweep, ReferencePolicy::Replace).combinations, 30);
    }

    /// The claim the whole instrument rests on: the verdict is
    /// [`crate::identify_clusters`]'s argmax *then* threshold, never a bare comparison. A's own
    /// reference sits 0.06 from the probe -- comfortably inside the cut -- and B's sits nearer,
    /// so the honest answer is a misattribution and a distance test would have called it
    /// correct.
    #[test]
    fn a_reference_inside_the_cut_still_loses_to_a_nearer_one() {
        let sweep = policy_sweep(
            &[
                item("A", "a1", 20.0),
                item("A", "a2", 22.0),
                item("A", "a3", 0.0),
                item("B", "b1", 5.0),
            ],
            IDENTIFY_DISTANCE,
        );

        // A's own references sit 0.060 and 0.073 from the a3 probe: both well inside the cut,
        // which is what makes a bare comparison against it the wrong instrument.
        assert!(1.0 - 22.0f32.to_radians().cos() < IDENTIFY_DISTANCE);

        for policy in ReferencePolicy::ALL {
            let scored = report(&sweep, policy);
            let stolen: Vec<&Misattribution> = scored
                .controlled
                .misattributions
                .iter()
                .filter(|taken| taken.probe_session == "a3")
                .collect();

            assert_eq!(
                stolen.len(),
                2,
                "{}: both orderings of the a3 probe lose to B",
                policy.label()
            );
            assert_eq!(
                (stolen[0].speaker.as_str(), stolen[0].named.as_str()),
                ("A", "B")
            );
            assert_eq!(
                stolen[0].built_from.len(),
                2,
                "both confirmed sessions are named"
            );
            assert_eq!(
                scored.controlled.closed.rejected,
                0,
                "{}: nothing here is outside the cut, so nothing is rejected for distance",
                policy.label()
            );
        }
    }

    /// What this ticket exists to measure, on a population built so that the arms *must*
    /// separate: two confirmed recordings 80 degrees apart, and a probe beside the first of
    /// them. Newest-wins throws away the reference that would have recognised it half the
    /// time; keeping both never does; averaging splits the difference and clears the cut from
    /// the middle.
    #[test]
    fn a_second_reference_recognises_a_probe_that_replacement_throws_away() {
        let sweep = policy_sweep(
            &[
                item("A", "a1", 0.0),
                item("A", "a2", 80.0),
                item("A", "a3", 10.0),
                item("B", "b1", 200.0),
            ],
            IDENTIFY_DISTANCE,
        );

        // Probing with a3 (10 degrees): replacement holds a2 in one ordering, and 70 degrees
        // is 0.658 away, so that combination is rejected.
        let replace = report(&sweep, ReferencePolicy::Replace);
        let set = report(&sweep, ReferencePolicy::Set);
        let average = report(&sweep, ReferencePolicy::Average);

        assert!(
            replace.controlled.closed.rejected > set.controlled.closed.rejected,
            "replacement rejected {} and the set {}",
            replace.controlled.closed.rejected,
            set.controlled.closed.rejected
        );
        assert_eq!(set.controlled.closed.misattributed, 0);
        assert_eq!(
            average.controlled.closed, set.controlled.closed,
            "on this population the blend also clears the cut everywhere the set does"
        );
    }

    /// Open-set removal is by **name**, so both rows of a reference set go together. If only
    /// one were dropped the probe would match the row still filed under its own name and be
    /// counted as a false alarm, so a zero here is the assertion.
    #[test]
    fn removing_a_speaker_from_the_database_removes_every_row_of_their_reference() {
        let sweep = policy_sweep(
            &[
                item("A", "a1", 0.0),
                item("A", "a2", 6.0),
                item("A", "a3", 3.0),
                item("B", "b1", 120.0),
            ],
            IDENTIFY_DISTANCE,
        );

        for policy in ReferencePolicy::ALL {
            let scored = report(&sweep, policy);
            assert_eq!(scored.controlled.closed.correct, 6, "{}", policy.label());
            assert_eq!(
                scored.controlled.open_false_alarms,
                0,
                "{} left a row behind: {:?}",
                policy.label(),
                scored.controlled.false_alarms
            );
        }
    }

    /// The within-session rule, on the shape that gives it teeth: a session id shared by two
    /// speakers, which is what a LibriVox project read by several volunteers looks like.
    ///
    /// `shared` is A's third session and also B's first and C's only. So on the two
    /// combinations probing with it, both B's and C's nearest usable recording is refused --
    /// C has no other and is absent from that database entirely -- and in the distance
    /// populations A's `shared` probe may be paired with neither, because that recording is
    /// behind both of their references.
    #[test]
    fn no_pair_is_drawn_from_one_recording_session() {
        let sweep = policy_sweep(
            &[
                item("A", "a1", 0.0),
                item("A", "a2", 5.0),
                item("A", "shared", 10.0),
                item("B", "shared", 91.0),
                item("B", "b2", 95.0),
                item("C", "shared", 180.0),
                item("D", "d1", 200.0),
                item("D", "d2", 205.0),
            ],
            IDENTIFY_DISTANCE,
        );

        for policy in ReferencePolicy::ALL {
            let scored = report(&sweep, policy);
            assert_eq!(
                scored.controlled.references_refused,
                2,
                "{}: B's and C's `shared` recording, on the one probe that is it",
                policy.label()
            );
            assert_eq!(
                scored.controlled.impostors_dropped,
                1,
                "{}: C has nothing else, so C is not in that probe's database",
                policy.label()
            );
            assert_eq!(scored.policy_impostors.references_refused, 2);
            assert_eq!(
                scored.impostor_pairs_refused,
                2,
                "{}: A's `shared` probe is scored against neither B nor C",
                policy.label()
            );
            assert_eq!(
                scored
                    .distances
                    .different
                    .expect("D is still an impostor")
                    .count,
                1,
                "{}: only D's reference is free of the probe's own recording",
                policy.label()
            );
        }
    }

    /// The distance populations: probes are the sessions the reference did not consume, so
    /// nothing is compared with a vector it contributed to, and the counts are independent of
    /// the policy.
    #[test]
    fn the_distance_populations_hold_out_every_probe_from_its_own_reference() {
        let sweep = policy_sweep(
            &[
                item("A", "a1", 0.0),
                item("A", "a2", 6.0),
                item("A", "a3", 3.0),
                item("A", "a4", 4.0),
                item("B", "b1", 100.0),
                item("B", "b2", 104.0),
                item("C", "c1", 200.0),
            ],
            IDENTIFY_DISTANCE,
        );

        for policy in ReferencePolicy::ALL {
            let scored = report(&sweep, policy);
            assert_eq!(scored.distance_probes, 2, "{}: a3 and a4", policy.label());
            let same = scored.distances.same.expect("A probes against A");
            assert_eq!(same.count, 2);
            let different = scored
                .distances
                .different
                .expect("A probes against B and C");
            assert_eq!(
                different.count,
                4,
                "{}: one trial per impostor person, not per row",
                policy.label()
            );
            let nearest = scored.nearest_impostor.expect("two probes, two nearest");
            assert_eq!(nearest.count, 2);
            assert!(
                nearest.min >= different.min,
                "the nearest cannot beat the minimum"
            );
            assert!(
                same.max < different.min,
                "these voices are far apart by construction"
            );
            let zero = scored
                .distances
                .zero_false_accept
                .expect("both populations are non-empty");
            assert_eq!(zero.false_reject_rate, 0.0);
        }
    }

    /// Averaging two people's recordings into one vector pulls that vector toward whatever
    /// sits between them, which is TASK-027's blend risk as a measurable quantity rather than
    /// a prediction. Here B's two sessions straddle A's probe, so B's blended reference lands
    /// nearer to it than either of B's own recordings.
    #[test]
    fn a_blended_impostor_can_sit_nearer_a_probe_than_either_recording_it_came_from() {
        let population = [
            item("A", "a1", 0.0),
            item("A", "a2", 2.0),
            item("A", "a3", 60.0),
            item("B", "b1", 30.0),
            item("B", "b2", 90.0),
        ];
        let sweep = policy_sweep(&population, IDENTIFY_DISTANCE);

        let closest = |policy| report(&sweep, policy).nearest_impostor.unwrap().min;
        let blended = closest(ReferencePolicy::Average);
        let kept_both = closest(ReferencePolicy::Set);

        assert!(
            blended < kept_both,
            "averaging moved the closest impostor from {kept_both} to {blended}"
        );
        assert_eq!(
            kept_both,
            closest(ReferencePolicy::Replace),
            "keeping both can only be nearer than the newest, and here the newest is nearest"
        );
    }

    /// Degenerate populations produce a report a printer can print, not a panic: nobody with
    /// three sessions, one speaker, no items at all.
    #[test]
    fn a_population_with_nothing_to_measure_reports_nothing_rather_than_panicking() {
        for population in [
            Vec::new(),
            vec![item("A", "a1", 0.0)],
            vec![item("A", "a1", 0.0), item("B", "b1", 90.0)],
            vec![item("A", "a1", 0.0), item("A", "a2", 5.0)],
        ] {
            let sweep = policy_sweep(&population, IDENTIFY_DISTANCE);

            assert!(sweep.targets.is_empty(), "{population:?}");
            assert_eq!(sweep.reports.len(), 3);
            for scored in &sweep.reports {
                assert_eq!(scored.combinations, 0);
                assert_eq!(scored.controlled.closed, ClosedSet::default());
                assert_eq!(scored.policy_impostors.closed, ClosedSet::default());
                assert_eq!(scored.distances.same, None);
                assert_eq!(scored.nearest_impostor, None);
            }
        }
    }

    /// A cache written by two embedding models describes two spaces, and a truncating `zip`
    /// would return plausible cosines between unrelated ones. Nothing is scored, and the
    /// lengths are reported so that a printer can say why.
    #[test]
    fn a_population_of_two_embedding_lengths_is_not_scored_at_all() {
        let mut population = vec![
            item("A", "a1", 0.0),
            item("A", "a2", 5.0),
            item("A", "a3", 10.0),
            item("B", "b1", 90.0),
        ];
        population[3].embedding = vec![0.0, 1.0];

        let sweep = policy_sweep(&population, IDENTIFY_DISTANCE);

        assert_eq!(sweep.dimensions, vec![2, 4]);
        assert_eq!(
            sweep.targets,
            vec!["A".to_string()],
            "the shape is still reported"
        );
        for scored in &sweep.reports {
            assert_eq!(scored.combinations, 0);
        }
    }

    /// An empty embedding is not a voice, and must not match an equally empty reference by
    /// vacuous agreement -- the same claim `identify.rs` makes at its own boundary.
    #[test]
    fn an_empty_embedding_is_not_a_population() {
        let sweep = policy_sweep(
            &[
                PolicyItem {
                    speaker: "A".to_string(),
                    session: "a1".to_string(),
                    embedding: Vec::new(),
                },
                PolicyItem {
                    speaker: "A".to_string(),
                    session: "a2".to_string(),
                    embedding: Vec::new(),
                },
                PolicyItem {
                    speaker: "A".to_string(),
                    session: "a3".to_string(),
                    embedding: Vec::new(),
                },
            ],
            IDENTIFY_DISTANCE,
        );

        assert_eq!(sweep.dimensions, vec![0]);
        for scored in &sweep.reports {
            assert_eq!(scored.combinations, 0);
        }
    }
}
