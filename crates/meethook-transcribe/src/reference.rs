//! How much speech a reference has to rest on before it can find its owner again.
//!
//! `enroll` copies a cluster's centroid into `speakers.json` and that vector then has to
//! recognise the same person in every future meeting. On session `20260810-093047` two
//! references built from one-second fragments sit 0.707 and 0.843 from their own owners --
//! further than another participant's reference sits from them -- while the three references
//! that "work" read 0.000 because the reference *is* the cluster being matched. Two anecdotes
//! and three tautologies do not locate a floor. This module is the instrument that does.
//!
//! # The quantity, stated once so that no reader has to guess
//!
//! **Cosine distance between a reference built from `d` seconds of one speaker's turns and the
//! normalized mean of that speaker's *remaining* turns.** Both sides are built the way
//! `speakers::reference_embedding` builds one -- the **unweighted** mean of per-turn unit
//! embeddings, L2-normalized *after* averaging -- because `enroll` copies `cluster.embedding`
//! into `speakers.json` verbatim. A duration-weighted mean, or an embedding of `d` seconds of
//! *concatenated* audio, would be measuring an algorithm meethook does not run.
//!
//! It is the same quantity [`crate::IDENTIFY_DISTANCE`] thresholds -- two single vectors, one
//! reference against one centroid -- and not [`crate::GroupDistance::average_linkage`], which
//! is what clustering merges on and is a different number on the same pair.
//!
//! # Held out means held out
//!
//! The turns a reference is built from are removed from the remainder it is measured against, so
//! no number here is a vector compared with one it was derived from. That is the same
//! leave-a-group-out discipline `crate::adoption_populations` uses for its positives, for the
//! same reason: a group left inside the mean it is compared to biases that mean toward it, and
//! the bias is largest exactly where the group is a big share of a small cluster.
//! `the_reference_turns_are_absent_from_the_remainder` asserts it rather than leaving it as a
//! sentence here.
//!
//! # Which `d` seconds get taken, and why there are two answers
//!
//! A reference built from one contiguous minute of one topic is not the same object as one built
//! from a minute scattered across a meeting, so [`Sampling`] runs both arms over the same grid.
//! [`Sampling::Prefix`] is what enrollment would get from a caller who stopped the meeting early;
//! [`Sampling::Spread`] is what it would get from the same amount of speech sampled across the
//! call. If the two arms disagree about where the floor sits, then the floor is not a function of
//! seconds alone and a report has to say so instead of averaging them.
//!
//! Turns are atomic -- an embedding covers a whole turn or none of it -- so a grid value of 3 s
//! against a speaker whose first turn runs 12 s realizes 12 s. [`SweepPoint::realized_s`] is
//! what any curve is plotted against; [`SweepPoint::requested_s`] is only the grid value that
//! asked for it, and several grid values commonly realize one turn set.
//!
//! # The verdict comes from `identify_clusters`, never from a bare comparison
//!
//! Identification is argmax *then* threshold, so a reference that clears
//! [`crate::IDENTIFY_DISTANCE`] while a nearer one wins is not a match. Every verdict here is
//! read off the shipped [`crate::identify_clusters`] over synthetic values, which is also what
//! makes the [`SpeakerCluster::heard_at_once_with`] veto a real one:
//!
//! - **[`SweepPoint::starved_alone`]** is the deployment case. One database holding the starved
//!   speaker's `d`-second reference *and every other speaker's full-cluster reference*, against
//!   clusters that are the starved speaker's held-out remainder plus everybody else's full
//!   centroid. This is the case where a weak vector has to beat strong ones -- Alex enrolled
//!   from 8 s while Andrew is enrolled from seven minutes -- and it is the one a floor is chosen
//!   from. The other speakers' rows in that simulation are tautological by construction and are
//!   not read; they are there so the competition and the veto are the real ones.
//! - **[`SweepPoint::all_starved`]** is the secondary case: every reference built from `d`,
//!   every cluster a held-out remainder. It says how the database as a whole degrades.
//!
//! # What is not measured, which belongs beside every number that is
//!
//! **The selection effect.** Holding out removes the tautology but not the reason those turns
//! are in that cluster: the merge loop put them there *because* it found them close. So
//! [`SweepPoint::own`] is biased low by an amount nothing within one session can measure.
//!
//! **The channel.** One call, one microphone, one channel is the easiest condition a reference
//! will ever face -- [`crate::IDENTIFY_DISTANCE`]'s own documentation says so -- and
//! cross-session variation is strictly larger. A band chosen from a within-session sweep is
//! optimistic in a direction, and the direction is known.
//!
//! # What this module does not do
//!
//! It does not pick a floor, it holds no threshold, and no production path calls it. Population
//! arithmetic -- false accept, false reject, equal error, percentiles -- belongs to
//! [`crate::score_trials`], which states and tests its conventions. Presentation belongs to
//! `examples/cluster-speaker-track.rs`. The arithmetic is here rather than there because
//! `cargo test` builds examples without running the `#[test]`s inside them, and a diagnostic
//! whose conventions nobody can test is a number to believe rather than evidence.

use meethook_session::{EnrolledSpeaker, EnrolledSpeakers, SpeakerCluster};

use crate::identify::{IDENTIFY_DISTANCE, identify_clusters};
use crate::segmentation::LocalTurn;
use crate::speakers::Clustering;
use crate::voice_vectors::group_mean;

/// How close a stored reference has to sit to a single turn's embedding before
/// [`stored_reference_distances`] calls that turn the one it was built from.
///
/// Not a similarity judgement. A reference enrolled from a one-turn cluster *is* that turn's
/// embedding -- the mean of one vector, normalized, is the vector -- so the honest reading of a
/// distance this small is identity rather than resemblance, and the smallest genuine
/// same-speaker turn-to-turn distance measured anywhere in this codebase is two orders of
/// magnitude above it. The margin is for f32 arithmetic having taken a different route to the
/// same vector, nothing else.
///
/// Public because [`StoredReference::origin`] cannot be read without it: "this reference is that
/// turn" is a claim about a number, and a caller that cannot see the number has to take it on
/// trust.
pub const ORIGIN_DISTANCE: f32 = 0.01;

/// Which `d` seconds of a speaker's turns a reference gets built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Sampling {
    /// The speaker's turns in start order until cumulative embedded speech first reaches `d`.
    Prefix,

    /// Turns nearest to `k` evenly spaced target times across the speaker's span, `k` rising
    /// until cumulative embedded speech first reaches `d`.
    Spread,
}

impl Sampling {
    /// Both arms, in report order.
    pub const ALL: [Sampling; 2] = [Sampling::Prefix, Sampling::Spread];

    /// A one-word name for a table header or a row label.
    pub fn label(self) -> &'static str {
        match self {
            Sampling::Prefix => "prefix",
            Sampling::Spread => "spread",
        }
    }
}

/// What [`crate::identify_clusters`] decided about one voice in one simulated database.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Identified as its own owner, which is the only outcome a reference is enrolled for.
    Correct,

    /// Identified as somebody else, naming whose reference took it. The failure a floor exists
    /// to prevent: one person's speech under another person's name, in a transcript nobody will
    /// re-read.
    Misattributed(u32),

    /// Nothing was awarded, and nothing was close enough to be: the nearest reference in the
    /// database sits at or beyond [`crate::IDENTIFY_DISTANCE`]. The visible failure -- an
    /// `Unknown N` the user fixes in ten seconds.
    Unidentified,

    /// Nothing was awarded although the nearest reference *did* clear the cut, which
    /// [`crate::identify_clusters`] does for exactly one reason: the name this voice claimed had
    /// already been awarded to a cluster segmentation heard at once with it. Reported apart from
    /// [`Verdict::Unidentified`] because the cause is a cannot-link fact rather than a distance,
    /// and folding the two together would read as a threshold being too tight.
    Vetoed,
}

impl Verdict {
    /// Whether identification put this voice under its own owner's name.
    pub fn is_correct(self) -> bool {
        matches!(self, Verdict::Correct)
    }

    /// A short form for a table cell.
    pub fn label(self) -> String {
        match self {
            Verdict::Correct => "correct".to_string(),
            Verdict::Misattributed(owner) => format!("-> speaker {owner}"),
            Verdict::Unidentified => "unidentified".to_string(),
            Verdict::Vetoed => "vetoed".to_string(),
        }
    }
}

/// One measured point: a reference built from `d` seconds, against the speech held out of it.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepPoint {
    /// The cluster this reference and this remainder both come from. Ids are
    /// [`SpeakerCluster::id`], which is also the position in [`Clustering::clusters`].
    pub speaker: u32,

    pub sampling: Sampling,

    /// The grid value that asked for this point. Several grid values commonly produce one turn
    /// set, so this is not a curve's x axis -- [`SweepPoint::realized_s`] is.
    pub requested_s: f64,

    /// Embedded speech the reference was actually built from, in seconds. At least
    /// `requested_s`, because turns are atomic.
    pub realized_s: f64,

    /// Turn indices the reference was built from, ascending, positional against the `turns`
    /// slice handed to [`reference_duration_sweep`].
    pub reference_turns: Vec<usize>,

    /// Embedded speech left in the remainder after the reference's turns were removed.
    pub held_out_s: f64,

    pub held_out_turns: usize,

    /// **The measurement.** The `d`-second reference against the normalized mean of this
    /// speaker's remaining turns. Cosine distance, 0 for the same direction.
    pub own: f32,

    /// The `d`-second reference against every *other* above-floor speaker's full centroid,
    /// ascending by cluster id.
    ///
    /// What a stored reference has to stay away from. Not the comparison identification makes
    /// for this point -- that is [`SweepPoint::rivals`], and the two are different numbers
    /// because a reference built from `d` seconds is not the same vector as the remainder it
    /// left behind.
    pub others: Vec<(u32, f32)>,

    /// This speaker's held-out remainder against every other above-floor speaker's *full*
    /// reference, ascending by cluster id.
    ///
    /// The competition [`SweepPoint::starved_alone`] resolves: in that simulation every other
    /// speaker is enrolled from their whole cluster, so these are the distances
    /// [`SweepPoint::own`] has to beat.
    pub rivals: Vec<(u32, f32)>,

    /// The deployment case: this speaker starved, everybody else enrolled in full. See the
    /// module documentation.
    pub starved_alone: Verdict,

    /// The whole database starved to the same grid value. Secondary.
    pub all_starved: Verdict,
}

impl SweepPoint {
    /// The nearest other speaker's full reference to this remainder, as `(cluster, distance)`.
    ///
    /// [`None`] when this speaker is the only one above the floor, where there is no
    /// competition and the verdict is uninformative for that reason.
    pub fn nearest_rival(&self) -> Option<(u32, f32)> {
        self.rivals
            .iter()
            .copied()
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// How much nearer this speaker's own starved reference is than the closest rival, in
    /// cosine distance. Negative means a rival is nearer, which is a misattribution unless a
    /// cannot-link fact intervenes.
    pub fn margin(&self) -> Option<f32> {
        self.nearest_rival().map(|(_, nearest)| nearest - self.own)
    }
}

/// Why a grid value produced no point for a speaker.
///
/// Counted and printed rather than dropped: a report that silently omits rows cannot be
/// reconciled with the cluster table above it, and an unstated exclusion is how a curve gets
/// quietly flattered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Decline {
    /// The speaker does not hold `requested_s` of embedded speech at all.
    ShortOfGrid { available_s: f64 },

    /// The remainder fell below `min_held_out_s`, so the distance would describe the noise in a
    /// thin remainder rather than the weakness of the reference -- the exact confusion this
    /// module exists to remove.
    ThinRemainder { held_out_s: f64 },

    /// A side had no direction: a remainder holding no turns at all, or a group whose members
    /// cancel. The first is reachable only with `min_held_out_s` at zero, where a grid value
    /// covering every one of a speaker's turns leaves nothing to measure against; the second is
    /// unreachable for real voices and is counted so that it cannot be silent if it happens.
    NoDirection,
}

/// One grid value that yielded no point, and why.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeclinedPoint {
    pub speaker: u32,
    pub sampling: Sampling,
    pub requested_s: f64,
    pub reason: Decline,
}

/// Everything one session says about how a reference behaves as a function of its duration.
#[derive(Debug, Clone, PartialEq)]
pub struct Sweep {
    /// Ordered by speaker, then [`Sampling::ALL`]'s order, then requested duration ascending, so
    /// two runs over one session are diffable -- which is the only way "the output did not
    /// change" gets checked at all.
    pub points: Vec<SweepPoint>,

    /// Same order as [`Sweep::points`].
    pub declined: Vec<DeclinedPoint>,

    /// The above-floor cluster ids the sweep ran over, ascending.
    pub speakers: Vec<u32>,

    /// The requested grid, as handed in.
    pub grid: Vec<f64>,

    /// The remainder floor below which a point was declined, as handed in.
    pub min_held_out_s: f64,
}

impl Sweep {
    /// Every point for one speaker under one sampling arm, in requested order.
    pub fn arm(&self, speaker: u32, sampling: Sampling) -> impl Iterator<Item = &SweepPoint> {
        self.points
            .iter()
            .filter(move |point| point.speaker == speaker && point.sampling == sampling)
    }

    /// The band of reference durations this session licenses, as
    /// `(largest failing duration, smallest duration above it)`.
    ///
    /// Read off [`SweepPoint::starved_alone`] over one sampling arm, and shaped like
    /// [`crate::SPEAKER_FLOOR_SECONDS`]'s own `(12.3, 47.0]`: **the widest range of floors giving
    /// the same partition**. A write floor `f` stores a reference when the speech behind it is
    /// `>= f`, so any `f` with `largest_failing < f <= smallest_above` refuses exactly the
    /// measured references that failed and writes exactly those that did not, and every value
    /// inside the band is the same decision.
    ///
    /// Both edges are consequences rather than curiosities, which is what a caller quoting this
    /// has to price. Below the lower edge a stored vector loses to other people's references and
    /// makes identification worse than having no reference at all. Above the upper edge a
    /// participant who spoke for that long stops contributing a reference and has to be named
    /// again in every future meeting -- [`Sweep::sacrificed`] is how many measured references
    /// that already costs at the lower edge.
    ///
    /// [`None`] when no point failed -- this session cannot say where the bottom is, only that
    /// it is below everything measured -- or when no point sits above the failures, where the
    /// grid ran out before the answer did.
    pub fn band(&self, sampling: Sampling) -> Option<(f64, f64)> {
        let mine = || {
            self.points
                .iter()
                .filter(|point| point.sampling == sampling)
        };
        let failing = mine()
            .filter(|point| !point.starved_alone.is_correct())
            .map(|point| point.realized_s)
            .max_by(f64::total_cmp)?;
        let above = mine()
            .map(|point| point.realized_s)
            .filter(|&realized| realized > failing)
            .min_by(f64::total_cmp)?;
        Some((failing, above))
    }

    /// Measured references a floor inside [`Sweep::band`] would refuse although they worked.
    ///
    /// The upper edge's price, counted rather than asserted: every one of these is a duration at
    /// which some speaker's starved reference did identify its own remainder, sitting at or below
    /// a duration at which some *other* speaker's did not. A long list means the verdicts are not
    /// monotone in seconds across speakers, and that a floor in seconds alone is buying safety
    /// for one voice with a name the user has to retype for another.
    pub fn sacrificed(&self, sampling: Sampling) -> Vec<&SweepPoint> {
        let Some((failing, _)) = self.band(sampling) else {
            return Vec::new();
        };
        self.points
            .iter()
            .filter(|point| point.sampling == sampling)
            .filter(|point| point.starved_alone.is_correct() && point.realized_s <= failing)
            .collect()
    }
}

/// Builds the whole duration sweep from one session's clustering.
///
/// `turns` is the slice handed to [`crate::cluster_speaker_turns`], `clustering` what it
/// returned, `floor_seconds` the talk time above which a cluster is a speaker rather than a
/// fragment -- the same [`crate::SPEAKER_FLOOR_SECONDS`] partition
/// [`crate::adoption_populations`] uses, so two blocks of one report cannot disagree about who
/// the speakers are. `grid` is the requested durations, and `min_held_out_s` the remainder floor
/// documented on [`Decline::ThinRemainder`].
///
/// Reads no models and no files, and is pure: the same arguments give the same sweep in the same
/// order. Cost is arithmetic over embeddings the caller already computed -- one normalized mean
/// per grid point per speaker per arm, and one [`crate::identify_clusters`] call per point.
///
/// The sweep runs over above-floor clusters only. A cluster holding eight seconds cannot be
/// split into a reference and a remainder that mean anything; what a below-floor cluster does as
/// a reference is [`fragment_probe`]'s question.
///
/// Every degenerate shape is an empty sweep rather than a panic: no clusters, nothing above the
/// floor, a one-turn speaker, turns with no embeddings, and an assignment naming a cluster or a
/// turn that does not exist all yield a value a report can print.
pub fn reference_duration_sweep(
    turns: &[LocalTurn],
    clustering: &Clustering,
    floor_seconds: f64,
    grid: &[f64],
    min_held_out_s: f64,
) -> Sweep {
    let session = Session::new(turns, clustering, floor_seconds);

    let mut points = Vec::new();
    let mut declined = Vec::new();
    for sampling in Sampling::ALL {
        for &requested in grid {
            // Every speaker's selection at this grid value first, because `all_starved` is one
            // simulation over all of them and cannot be built a speaker at a time.
            let mut chosen: Vec<Selection> = Vec::new();
            for &speaker in &session.above {
                match session.select(speaker, sampling, requested, min_held_out_s) {
                    Ok(selection) => chosen.push(selection),
                    Err(reason) => declined.push(DeclinedPoint {
                        speaker: session.id(speaker),
                        sampling,
                        requested_s: requested,
                        reason,
                    }),
                }
            }

            let starved_database: Vec<(u32, &[f32])> = chosen
                .iter()
                .map(|s| (session.id(s.speaker), s.reference.as_slice()))
                .collect();
            let starved_clusters: Vec<(u32, &[f32])> = chosen
                .iter()
                .map(|s| (session.id(s.speaker), s.remainder.as_slice()))
                .collect();

            for selection in &chosen {
                points.push(session.measure(
                    selection,
                    sampling,
                    requested,
                    &starved_database,
                    &starved_clusters,
                ));
            }
        }
    }

    let order = |point: &SweepPoint| (point.speaker, point.sampling);
    points.sort_by(|a, b| {
        order(a)
            .cmp(&order(b))
            .then(a.requested_s.total_cmp(&b.requested_s))
    });
    let order = |point: &DeclinedPoint| (point.speaker, point.sampling);
    declined.sort_by(|a, b| {
        order(a)
            .cmp(&order(b))
            .then(a.requested_s.total_cmp(&b.requested_s))
    });

    Sweep {
        points,
        declined,
        speakers: session.above.iter().map(|&c| session.id(c)).collect(),
        grid: grid.to_vec(),
        min_held_out_s,
    }
}

/// One stored `speakers.json` reference measured against this session's voices.
///
/// What puts the 0.707 and 0.843 anecdotes on the sweep's curve: those two references are single
/// turns, so they are a measured point at `d` = one turn rather than a story about one.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredReference {
    pub name: String,

    /// The turn this reference *is*, as `(turn, distance)`, when one sits within
    /// [`ORIGIN_DISTANCE`] of it -- i.e. when it was enrolled from a cluster of that one turn.
    ///
    /// [`None`] for a reference built from a cluster of several turns, where no single turn is
    /// the vector and there is nothing to hold out.
    pub origin: Option<(usize, f32)>,

    /// This reference against every above-floor speaker, ascending by cluster id, as
    /// `(cluster, distance to the full centroid, distance with the origin turn held out)`.
    ///
    /// The third number is the one to read. Adoption has since pulled those one-second fragments
    /// into the clusters they came from, so measuring against the full centroid compares a
    /// vector with a mean it is part of -- the tautology this whole module is built to avoid.
    /// The two numbers are equal for every cluster not holding the origin turn.
    pub against: Vec<(u32, f32, f32)>,
}

/// Measures every stored reference against this session's above-floor speakers, holding out the
/// turn each reference was built from.
///
/// A reference of a different dimensionality came from a different embedding model and describes
/// a different space, so it is skipped entirely -- the same thing `identify::best_match` does
/// with it, and for the same reason: a truncated `zip` would return a plausible-looking cosine
/// about an entry identification is ignoring.
pub fn stored_reference_distances(
    enrolled: &EnrolledSpeakers,
    clustering: &Clustering,
    floor_seconds: f64,
) -> Vec<StoredReference> {
    let session = Session::new(&[], clustering, floor_seconds);
    let dimensions = session
        .above
        .first()
        .map(|&speaker| session.centroid[speaker].len());

    enrolled
        .speakers
        .iter()
        .filter(|speaker| dimensions == Some(speaker.embedding.len()))
        .map(|speaker| {
            let origin = clustering
                .turn_embeddings
                .iter()
                .enumerate()
                .filter_map(|(turn, held)| {
                    Some((turn, distance(held.as_deref()?, &speaker.embedding)))
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .filter(|&(_, apart)| apart < ORIGIN_DISTANCE);

            let against = session
                .above
                .iter()
                .map(|&at| {
                    let full = distance(&session.centroid[at], &speaker.embedding);
                    let rest: Vec<usize> = session.members[at]
                        .iter()
                        .copied()
                        .filter(|&turn| origin.is_none_or(|(source, _)| turn != source))
                        .collect();
                    let held_out = session
                        .mean(&rest)
                        .map_or(full, |mean| distance(&mean, &speaker.embedding));
                    (session.id(at), full, held_out)
                })
                .collect();

            StoredReference {
                name: speaker.name.clone(),
                origin,
                against,
            }
        })
        .collect()
}

/// One below-floor cluster used as though it were an enrolled reference.
///
/// The shape of the case a write floor is really about: a real participant who barely spoke,
/// correctly named, whose stored vector then has to compete with references built from minutes.
/// Below the floor there is no held-out split to make inside the cluster -- which is why
/// [`FragmentProbe::verdict`] measures it against the *nearest other cluster* instead, a
/// same-speaker pair with no shared turns at all if the two clusters really are one voice.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentProbe {
    pub cluster: u32,
    pub seconds: f64,

    /// Every other cluster in the session, ascending by distance from this one's centroid, as
    /// `(cluster, its speech seconds, centroid distance)`.
    ///
    /// Whether the nearest is the same person is an ear judgement this module does not make.
    pub nearest: Vec<(u32, f64, f32)>,

    /// This fragment's centroid against every above-floor speaker's, ascending by cluster id.
    pub against_speakers: Vec<(u32, f32)>,

    /// The nearest other cluster, and what [`crate::identify_clusters`] decides about it when
    /// this fragment is the reference enrolled under *that* cluster's name, alongside every
    /// above-floor speaker's full-cluster reference.
    ///
    /// So [`Verdict::Correct`] means the fragment found the other half of its own voice against
    /// the real competition, and anything else means a reference built from that much speech
    /// could not. [`None`] when this is the only cluster in the session.
    pub verdict: Option<(u32, Verdict)>,
}

/// Runs [`FragmentProbe`] over one cluster, or [`None`] if that cluster has no direction or does
/// not exist. Works at any size; the interesting cases are below `floor_seconds`.
pub fn fragment_probe(
    clustering: &Clustering,
    floor_seconds: f64,
    cluster: u32,
) -> Option<FragmentProbe> {
    let session = Session::new(&[], clustering, floor_seconds);
    let at = session.at(cluster)?;
    if session.centroid[at].is_empty() {
        return None;
    }

    let mut nearest: Vec<(u32, f64, f32)> = (0..clustering.clusters.len())
        .filter(|&other| other != at && !session.centroid[other].is_empty())
        .map(|other| {
            (
                session.id(other),
                clustering.clusters[other].speech_seconds,
                distance(&session.centroid[at], &session.centroid[other]),
            )
        })
        .collect();
    nearest.sort_by(|a, b| a.2.total_cmp(&b.2).then(a.0.cmp(&b.0)));

    let against_speakers = session
        .above
        .iter()
        .filter(|&&speaker| speaker != at)
        .map(|&speaker| {
            (
                session.id(speaker),
                distance(&session.centroid[at], &session.centroid[speaker]),
            )
        })
        .collect();

    // The same simulation `starved_alone` runs, with the fragment in the starved speaker's
    // place: it is enrolled under its own name, everybody else from their whole cluster, and the
    // voice being identified is the nearest other cluster rather than a held-out remainder.
    let verdict = nearest.first().map(|&(subject, _, _)| {
        let subject_at = session.at(subject).expect("nearest names a real cluster");
        let mut database: Vec<(u32, &[f32])> = vec![(subject, session.centroid[at].as_slice())];
        let mut voices: Vec<(u32, &[f32])> =
            vec![(subject, session.centroid[subject_at].as_slice())];
        for &speaker in &session.above {
            if speaker == at || speaker == subject_at {
                continue;
            }
            database.push((session.id(speaker), session.centroid[speaker].as_slice()));
            voices.push((session.id(speaker), session.centroid[speaker].as_slice()));
        }
        (subject, session.decide(subject, &database, &voices))
    });

    Some(FragmentProbe {
        cluster,
        seconds: clustering.clusters[at].speech_seconds,
        nearest,
        against_speakers,
        verdict,
    })
}

/// One speaker's chosen reference turns, and everything derived from them.
struct Selection {
    /// Position in [`Clustering::clusters`], not a cluster id -- the two agree in production and
    /// nothing here relies on it.
    speaker: usize,
    turns: Vec<usize>,
    realized_s: f64,
    held_out: Vec<usize>,
    held_out_s: f64,
    reference: Vec<f32>,
    remainder: Vec<f32>,
}

/// The session's turns, clusters and centroids, resolved once.
///
/// Exists so that the selection, the distances and the two simulations all read one grouping.
/// Two loops that each rebuilt it could disagree about which turns are in which cluster, and
/// that disagreement would be invisible in the output.
struct Session<'a> {
    turns: &'a [LocalTurn],
    clustering: &'a Clustering,
    /// Embedded turn indices per cluster, in start order.
    members: Vec<Vec<usize>>,
    /// Each cluster's full centroid, empty for a cluster with no direction.
    centroid: Vec<Vec<f32>>,
    /// Positions of the above-floor clusters, ascending.
    above: Vec<usize>,
}

impl<'a> Session<'a> {
    /// `turns` may be empty, in which case durations read 0 and only the centroid arithmetic is
    /// usable -- which is all [`stored_reference_distances`] and [`fragment_probe`] need.
    fn new(turns: &'a [LocalTurn], clustering: &'a Clustering, floor_seconds: f64) -> Session<'a> {
        let clusters = &clustering.clusters;
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); clusters.len()];
        for (turn, assigned) in clustering.assignment.iter().enumerate() {
            // Anything the clustering names that this slice does not have is dropped rather than
            // indexed: the two come from one call in production, and a diagnostic is not the
            // place to panic over a mismatch.
            let Some(&id) = assigned.as_ref() else {
                continue;
            };
            if !clustering
                .turn_embeddings
                .get(turn)
                .is_some_and(Option::is_some)
            {
                continue;
            }
            if let Some(mine) = members.get_mut(id as usize) {
                mine.push(turn);
            }
        }
        if !turns.is_empty() {
            for mine in &mut members {
                mine.retain(|&turn| turn < turns.len());
                mine.sort_by(|&a, &b| {
                    turns[a]
                        .start_s
                        .total_cmp(&turns[b].start_s)
                        .then(a.cmp(&b))
                });
            }
        }

        let mut session = Session {
            turns,
            clustering,
            centroid: Vec::new(),
            above: (0..clusters.len())
                .filter(|&at| clusters[at].speech_seconds >= floor_seconds)
                .collect(),
            members,
        };
        session.centroid = (0..clusters.len())
            .map(|at| session.mean(&session.members[at]).unwrap_or_default())
            .collect();
        session
    }

    fn id(&self, at: usize) -> u32 {
        self.clustering.clusters[at].id
    }

    fn at(&self, id: u32) -> Option<usize> {
        self.clustering.clusters.iter().position(|c| c.id == id)
    }

    /// Seconds of speech in a set of turns.
    ///
    /// Folded from 0.0 rather than summed, whose float identity is `-0.0`: an empty set would
    /// otherwise print "-0.0 s", which reads as a broken instrument.
    fn seconds(&self, held: &[usize]) -> f64 {
        held.iter().fold(0.0, |total, &turn| {
            total + self.turns[turn].end_s - self.turns[turn].start_s
        })
    }

    /// The normalized mean of these turns' embeddings: the vector `enroll` would store.
    fn mean(&self, held: &[usize]) -> Option<Vec<f32>> {
        let vectors: Vec<&[f32]> = held
            .iter()
            .filter_map(|&turn| self.clustering.turn_embeddings[turn].as_deref())
            .collect();
        group_mean(&vectors).map(|(unit, _)| unit)
    }

    /// Picks one speaker's reference turns at one grid value, or says why there are none.
    fn select(
        &self,
        speaker: usize,
        sampling: Sampling,
        requested: f64,
        min_held_out_s: f64,
    ) -> Result<Selection, Decline> {
        let mine = &self.members[speaker];
        let available_s = self.seconds(mine);
        let taken = match sampling {
            Sampling::Prefix => self.prefix(mine, requested),
            Sampling::Spread => self.spread(mine, requested),
        }
        .ok_or(Decline::ShortOfGrid { available_s })?;

        let held_out: Vec<usize> = mine
            .iter()
            .copied()
            .filter(|turn| !taken.contains(turn))
            .collect();
        let held_out_s = self.seconds(&held_out);
        if held_out_s < min_held_out_s {
            return Err(Decline::ThinRemainder { held_out_s });
        }

        let (Some(reference), Some(remainder)) = (self.mean(&taken), self.mean(&held_out)) else {
            return Err(Decline::NoDirection);
        };
        Ok(Selection {
            speaker,
            realized_s: self.seconds(&taken),
            turns: taken,
            held_out_s,
            held_out,
            reference,
            remainder,
        })
    }

    /// Turns in start order until cumulative speech first reaches `target`.
    fn prefix(&self, mine: &[usize], target: f64) -> Option<Vec<usize>> {
        let mut taken = Vec::new();
        let mut total = 0.0;
        for &turn in mine {
            if total >= target {
                break;
            }
            total += self.turns[turn].end_s - self.turns[turn].start_s;
            taken.push(turn);
        }
        (total >= target).then_some(taken)
    }

    /// Turns nearest to `k` evenly spaced times across this speaker's span, `k` rising until
    /// cumulative speech first reaches `target`.
    ///
    /// Greedy per target and deterministic: nearest midpoint, ties to the lower turn index. `k`
    /// may select fewer than `k` turns when two targets share a nearest turn, which is why the
    /// stopping rule is the duration reached rather than the count taken.
    fn spread(&self, mine: &[usize], target: f64) -> Option<Vec<usize>> {
        let midpoint = |turn: usize| (self.turns[turn].start_s + self.turns[turn].end_s) / 2.0;
        let first = midpoint(*mine.first()?);
        let last = midpoint(*mine.last()?);
        let span = (last - first).max(f64::MIN_POSITIVE);

        for k in 1..=mine.len() {
            let mut taken: Vec<usize> = Vec::new();
            let mut total = 0.0;
            for slot in 0..k {
                let want = first + span * (slot as f64 + 0.5) / k as f64;
                let nearest = mine
                    .iter()
                    .copied()
                    .filter(|turn| !taken.contains(turn))
                    .min_by(|&a, &b| {
                        (midpoint(a) - want)
                            .abs()
                            .total_cmp(&(midpoint(b) - want).abs())
                            .then(a.cmp(&b))
                    });
                let Some(turn) = nearest else { break };
                total += self.turns[turn].end_s - self.turns[turn].start_s;
                taken.push(turn);
                if total >= target {
                    break;
                }
            }
            if total >= target {
                taken.sort_unstable();
                return Some(taken);
            }
        }
        None
    }

    /// Everything derived from one selection, including both simulations.
    fn measure(
        &self,
        selection: &Selection,
        sampling: Sampling,
        requested: f64,
        starved_database: &[(u32, &[f32])],
        starved_clusters: &[(u32, &[f32])],
    ) -> SweepPoint {
        let speaker = self.id(selection.speaker);
        let others: Vec<(u32, f32)> = self
            .above
            .iter()
            .filter(|&&other| other != selection.speaker)
            .map(|&other| {
                (
                    self.id(other),
                    distance(&selection.reference, &self.centroid[other]),
                )
            })
            .collect();
        let rivals: Vec<(u32, f32)> = self
            .above
            .iter()
            .filter(|&&other| other != selection.speaker)
            .map(|&other| {
                (
                    self.id(other),
                    distance(&selection.remainder, &self.centroid[other]),
                )
            })
            .collect();

        // The deployment case: this speaker's starved reference, everybody else's full one.
        let mut database: Vec<(u32, &[f32])> = vec![(speaker, selection.reference.as_slice())];
        let mut voices: Vec<(u32, &[f32])> = vec![(speaker, selection.remainder.as_slice())];
        for &other in &self.above {
            if other == selection.speaker {
                continue;
            }
            database.push((self.id(other), self.centroid[other].as_slice()));
            voices.push((self.id(other), self.centroid[other].as_slice()));
        }

        SweepPoint {
            speaker,
            sampling,
            requested_s: requested,
            realized_s: selection.realized_s,
            reference_turns: selection.turns.clone(),
            held_out_s: selection.held_out_s,
            held_out_turns: selection.held_out.len(),
            own: distance(&selection.reference, &selection.remainder),
            others,
            rivals,
            starved_alone: self.decide(speaker, &database, &voices),
            all_starved: self.decide(speaker, starved_database, starved_clusters),
        }
    }

    /// What [`crate::identify_clusters`] decides about `subject`, given one reference and one
    /// cluster per speaker.
    ///
    /// Synthetic values, real function. Each cluster carries the *real* cluster's id and
    /// [`SpeakerCluster::heard_at_once_with`] -- a held-out remainder is the same voice, so the
    /// cannot-link facts still hold -- and no representatives, which identification does not
    /// read. Nothing here re-implements argmax or the threshold: a hand-rolled comparison
    /// against [`crate::IDENTIFY_DISTANCE`] would call a reference that clears the cut while a
    /// nearer one wins a match, which it is not.
    fn decide(
        &self,
        subject: u32,
        database: &[(u32, &[f32])],
        voices: &[(u32, &[f32])],
    ) -> Verdict {
        let enrolled = EnrolledSpeakers::new(
            database
                .iter()
                .map(|&(owner, embedding)| EnrolledSpeaker {
                    name: enrolled_name(owner),
                    embedding: embedding.to_vec(),
                    clip_seconds: None,
                })
                .collect(),
        );
        let clusters: Vec<SpeakerCluster> = voices
            .iter()
            .map(|&(id, embedding)| {
                let real = self.at(id).map(|at| &self.clustering.clusters[at]);
                SpeakerCluster {
                    id,
                    embedding: embedding.to_vec(),
                    speech_seconds: real.map_or(0.0, |c| c.speech_seconds),
                    first_spoke_seconds: real.map_or(0.0, |c| c.first_spoke_seconds),
                    heard_at_once_with: real
                        .map(|c| c.heard_at_once_with.clone())
                        .unwrap_or_default(),
                    representatives: Vec::new(),
                }
            })
            .collect();

        match identify_clusters(&clusters, &enrolled).get(&subject) {
            Some(matched) if matched.name == enrolled_name(subject) => Verdict::Correct,
            Some(matched) => database
                .iter()
                .map(|&(owner, _)| owner)
                .find(|&owner| enrolled_name(owner) == matched.name)
                .map_or(Verdict::Unidentified, Verdict::Misattributed),
            // Nothing was awarded. Which of the two reasons that is cannot be read off the
            // return value -- absence is absence -- so it is read off the distances instead:
            // if the nearest reference in the database cleared the cut and the name still did
            // not land, the only mechanism `identify_clusters` has for that is the veto.
            None => {
                let mine = voices
                    .iter()
                    .find(|&&(id, _)| id == subject)
                    .map(|&(_, embedding)| embedding);
                let nearest = mine.and_then(|mine| {
                    database
                        .iter()
                        .map(|&(_, reference)| distance(mine, reference))
                        .min_by(f32::total_cmp)
                });
                match nearest {
                    Some(nearest) if nearest < IDENTIFY_DISTANCE => Verdict::Vetoed,
                    _ => Verdict::Unidentified,
                }
            }
        }
    }
}

/// The name a simulated database files one speaker under.
///
/// Any injective function of the id would do; this one is readable in a panic message. It never
/// reaches disk -- these databases exist for the length of one [`identify_clusters`] call.
fn enrolled_name(speaker: u32) -> String {
    format!("speaker {speaker}")
}

/// Cosine distance between two unit-length vectors.
///
/// The same arithmetic `identify::best_match` does -- a dot product, because both sides are unit
/// vectors by contract -- so a distance printed beside a verdict cannot disagree with the
/// verdict. Vectors of different lengths give [`f32::INFINITY`] rather than a truncated `zip`'s
/// plausible-looking cosine: different lengths mean different embedding models, and the honest
/// answer about a pair from two spaces is that they are not comparable at all.
fn distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return f32::INFINITY;
    }
    1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segmentation::WINDOW_SECONDS;

    /// A unit vector pointing `degrees` away from the first axis, so a test can name the
    /// distance it means -- `1 - cos(difference)` -- instead of a pile of decimals. Same helper
    /// `speakers.rs` and `adoption.rs` test with, for the same reason.
    fn at(degrees: f32) -> Vec<f32> {
        let radians = degrees.to_radians();
        vec![radians.cos(), radians.sin(), 0.0, 0.0]
    }

    fn turn(start_s: f64, end_s: f64, local_speaker: usize) -> LocalTurn {
        LocalTurn {
            start_s,
            end_s,
            window: (start_s / WINDOW_SECONDS) as usize,
            local_speaker,
        }
    }

    /// A clustering assembled by hand: `groups` gives each cluster's turn indices in id order,
    /// `voices` each turn's embedding, and `heard_at_once_with` the cannot-link relation between
    /// clusters, which identification's veto reads.
    fn clustering(
        turns: &[LocalTurn],
        groups: &[&[usize]],
        voices: &[Option<Vec<f32>>],
        exclusions: &[(u32, u32)],
    ) -> Clustering {
        let mut assignment = vec![None; turns.len()];
        let mut clusters = Vec::new();
        for (id, group) in groups.iter().enumerate() {
            for &turn in *group {
                assignment[turn] = Some(id as u32);
            }
            let held: Vec<&[f32]> = group
                .iter()
                .filter_map(|&turn| voices[turn].as_deref())
                .collect();
            clusters.push(SpeakerCluster {
                id: id as u32,
                embedding: group_mean(&held).map(|(unit, _)| unit).unwrap_or_default(),
                speech_seconds: group
                    .iter()
                    .fold(0.0, |total, &t| total + turns[t].end_s - turns[t].start_s),
                first_spoke_seconds: group
                    .iter()
                    .map(|&t| turns[t].start_s)
                    .fold(f64::INFINITY, f64::min),
                heard_at_once_with: exclusions
                    .iter()
                    .filter(|&&(a, _)| a == id as u32)
                    .map(|&(_, b)| b)
                    .collect(),
                representatives: Vec::new(),
            });
        }
        Clustering {
            clusters,
            assignment,
            turn_embeddings: voices.to_vec(),
        }
    }

    fn voices(degrees: &[f32]) -> Vec<Option<Vec<f32>>> {
        degrees.iter().map(|&d| Some(at(d))).collect()
    }

    /// One speaker of five 10 s turns, all near 0 degrees, and a second speaker of five 10 s
    /// turns near 60. Both clear a 30 s floor.
    fn two_speakers() -> (Vec<LocalTurn>, Vec<Option<Vec<f32>>>, Clustering) {
        let turns: Vec<LocalTurn> = (0..10)
            .map(|n| turn(n as f64 * 20.0, n as f64 * 20.0 + 10.0, 0))
            .collect();
        let voices = voices(&[0.0, 2.0, 4.0, 6.0, 8.0, 60.0, 62.0, 64.0, 66.0, 68.0]);
        let grouping = clustering(&turns, &[&[0, 1, 2, 3, 4], &[5, 6, 7, 8, 9]], &voices, &[]);
        (turns, voices, grouping)
    }

    #[test]
    fn the_reference_turns_are_absent_from_the_remainder() {
        let (turns, voices, grouping) = two_speakers();
        let sweep = reference_duration_sweep(&turns, &grouping, 30.0, &[20.0], 10.0);

        let point = sweep
            .arm(0, Sampling::Prefix)
            .next()
            .expect("speaker 0 holds 50 s");
        assert_eq!(point.reference_turns, [0, 1]);
        assert_eq!(point.realized_s, 20.0);
        assert_eq!(point.held_out_turns, 3);
        assert_eq!(point.held_out_s, 30.0);

        let reference = group_mean(&[voices[0].as_deref().unwrap(), voices[1].as_deref().unwrap()])
            .unwrap()
            .0;
        let remainder = group_mean(&[
            voices[2].as_deref().unwrap(),
            voices[3].as_deref().unwrap(),
            voices[4].as_deref().unwrap(),
        ])
        .unwrap()
        .0;
        assert!((point.own - distance(&reference, &remainder)).abs() < 1e-6);

        // The claim the exclusion is for: leaving the reference's turns in the group it is
        // measured against pulls that mean toward it, so the same pair reads closer than it is.
        let whole = group_mean(
            &(0..5)
                .map(|n| voices[n].as_deref().unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .0;
        assert!(
            distance(&reference, &whole) < point.own,
            "including the reference's own turns did not flatter the distance, so this test \
             proves nothing"
        );
    }

    #[test]
    fn the_reference_is_the_unweighted_mean_normalized_afterwards() {
        // One long turn and one short one. The unweighted mean of two unit vectors bisects
        // them; a duration-weighted mean would sit nine tenths of the way toward the long one.
        let turns = vec![
            turn(0.0, 90.0, 0),
            turn(100.0, 110.0, 0),
            turn(200.0, 230.0, 0),
            turn(300.0, 330.0, 0),
        ];
        let voices = voices(&[0.0, 40.0, 80.0, 82.0]);
        let grouping = clustering(&turns, &[&[0, 1, 2, 3]], &voices, &[]);

        let sweep = reference_duration_sweep(&turns, &grouping, 30.0, &[95.0], 10.0);
        let point = sweep
            .points
            .first()
            .expect("95 s takes the first two turns");
        assert_eq!(point.reference_turns, [0, 1]);

        let unweighted =
            group_mean(&[voices[0].as_deref().unwrap(), voices[1].as_deref().unwrap()])
                .unwrap()
                .0;
        let remainder = group_mean(&[voices[2].as_deref().unwrap(), voices[3].as_deref().unwrap()])
            .unwrap()
            .0;
        assert!((point.own - distance(&unweighted, &remainder)).abs() < 1e-6);

        // A duration-weighted reference is 20 degrees away from the unweighted one, so the two
        // are not the same measurement and this assertion is not a tautology.
        let weighted = group_mean(&[
            voices[0].as_deref().unwrap(),
            voices[0].as_deref().unwrap(),
            voices[0].as_deref().unwrap(),
            voices[0].as_deref().unwrap(),
            voices[0].as_deref().unwrap(),
            voices[0].as_deref().unwrap(),
            voices[0].as_deref().unwrap(),
            voices[0].as_deref().unwrap(),
            voices[0].as_deref().unwrap(),
            voices[1].as_deref().unwrap(),
        ])
        .unwrap()
        .0;
        assert!((distance(&weighted, &remainder) - point.own).abs() > 0.01);
    }

    #[test]
    fn turns_are_atomic_so_realized_is_what_a_curve_is_plotted_against() {
        let (turns, _, grouping) = two_speakers();
        let sweep = reference_duration_sweep(&turns, &grouping, 30.0, &[1.0, 5.0, 10.0], 10.0);

        // Every grid value at or below one turn realizes that one turn, which is why several
        // requested values commonly name one turn set.
        for point in sweep.arm(0, Sampling::Prefix) {
            assert_eq!(point.reference_turns, [0]);
            assert_eq!(point.realized_s, 10.0);
            assert!(point.realized_s >= point.requested_s);
        }
        assert_eq!(sweep.arm(0, Sampling::Prefix).count(), 3);
    }

    #[test]
    fn a_thin_remainder_is_declined_rather_than_measured() {
        let (turns, _, grouping) = two_speakers();
        // 35 s of reference realizes 40 s and leaves 10 s behind, under a 20 s remainder floor.
        let sweep = reference_duration_sweep(&turns, &grouping, 30.0, &[35.0], 20.0);

        assert!(sweep.points.is_empty());
        assert_eq!(sweep.declined.len(), 4, "two speakers, two arms");
        assert!(
            sweep
                .declined
                .iter()
                .all(|point| point.reason == Decline::ThinRemainder { held_out_s: 10.0 })
        );
    }

    #[test]
    fn a_grid_value_beyond_the_speech_available_is_declined() {
        let (turns, _, grouping) = two_speakers();
        let sweep = reference_duration_sweep(&turns, &grouping, 30.0, &[80.0], 0.0);

        assert!(sweep.points.is_empty());
        assert!(
            sweep
                .declined
                .iter()
                .all(|point| point.reason == Decline::ShortOfGrid { available_s: 50.0 })
        );
    }

    #[test]
    fn spread_and_prefix_agree_when_nearly_every_turn_is_needed() {
        let (turns, _, grouping) = two_speakers();
        // 40 of this speaker's 50 s: there is only one four-turn set that leaves a remainder at
        // all, so the two arms have nothing left to disagree about.
        let sweep = reference_duration_sweep(&turns, &grouping, 30.0, &[40.0], 0.0);

        let prefix = sweep.arm(0, Sampling::Prefix).next().expect("40 of 50 s");
        let spread = sweep.arm(0, Sampling::Spread).next().expect("40 of 50 s");
        assert_eq!(prefix.reference_turns, [0, 1, 2, 3]);
        assert_eq!(prefix.reference_turns, spread.reference_turns);

        // And they disagree where there is a choice, or the second arm would be measuring
        // nothing: 20 s of spread takes the first and the middle turn, not the first two.
        let sweep = reference_duration_sweep(&turns, &grouping, 30.0, &[20.0], 0.0);
        assert_eq!(
            sweep
                .arm(0, Sampling::Spread)
                .next()
                .unwrap()
                .reference_turns,
            [1, 3]
        );
    }

    /// The point of running the shipped function rather than comparing against the constant: a
    /// reference that clears the cut is not a match when a nearer one wins.
    #[test]
    fn the_verdict_is_identify_clusters_argmax_and_not_a_bare_comparison() {
        // Speaker 0's first turn points at 120 degrees while the rest of him sits near 55, so
        // his starved reference is 0.577 from his own remainder -- and speaker 1, enrolled from
        // his whole cluster at 60, is 0.004 away from it.
        let turns: Vec<LocalTurn> = (0..10)
            .map(|n| turn(n as f64 * 20.0, n as f64 * 20.0 + 10.0, 0))
            .collect();
        let near = voices(&[120.0, 55.0, 55.0, 55.0, 55.0, 60.0, 60.0, 60.0, 60.0, 60.0]);
        let grouping = clustering(&turns, &[&[0, 1, 2, 3, 4], &[5, 6, 7, 8, 9]], &near, &[]);

        let sweep = reference_duration_sweep(&turns, &grouping, 30.0, &[5.0], 10.0);
        let point = sweep.arm(0, Sampling::Prefix).next().unwrap();
        assert!(point.own > IDENTIFY_DISTANCE, "own reference {}", point.own);
        assert_eq!(point.starved_alone, Verdict::Misattributed(1));
        assert!(point.margin().is_some_and(|margin| margin < 0.0));

        // The same geometry with the two clusters heard at once: speaker 1's own cluster takes
        // the name first, and the veto -- not the threshold -- is what leaves this one nameless.
        let vetoed = clustering(
            &turns,
            &[&[0, 1, 2, 3, 4], &[5, 6, 7, 8, 9]],
            &near,
            &[(0, 1), (1, 0)],
        );
        let sweep = reference_duration_sweep(&turns, &vetoed, 30.0, &[5.0], 10.0);
        assert_eq!(
            sweep.arm(0, Sampling::Prefix).next().unwrap().starved_alone,
            Verdict::Vetoed
        );

        // And a remainder far from every reference is unidentified for the ordinary reason.
        let distant = voices(&[120.0, 55.0, 55.0, 55.0, 55.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let apart = clustering(&turns, &[&[0, 1, 2, 3, 4], &[5, 6, 7, 8, 9]], &distant, &[]);
        let sweep = reference_duration_sweep(&turns, &apart, 30.0, &[5.0], 10.0);
        assert_eq!(
            sweep.arm(0, Sampling::Prefix).next().unwrap().starved_alone,
            Verdict::Unidentified
        );
    }

    #[test]
    fn a_reference_built_from_enough_speech_finds_its_own_remainder() {
        let (turns, _, grouping) = two_speakers();
        let sweep = reference_duration_sweep(&turns, &grouping, 30.0, &[20.0], 10.0);

        for point in &sweep.points {
            assert_eq!(point.starved_alone, Verdict::Correct);
            assert_eq!(point.all_starved, Verdict::Correct);
            assert!(point.margin().is_some_and(|margin| margin > 0.0));
        }
        assert_eq!(sweep.speakers, [0, 1]);
    }

    #[test]
    fn the_band_is_the_widest_range_of_floors_giving_one_partition() {
        // Speaker 0's early turns point away from him, so a short reference misattributes and a
        // longer one does not. Turns 0 and 1 are at 120 degrees; the rest sit at 55, next to
        // speaker 1 at 60.
        let turns: Vec<LocalTurn> = (0..12)
            .map(|n| turn(n as f64 * 20.0, n as f64 * 20.0 + 10.0, 0))
            .collect();
        let voices = voices(&[
            120.0, 120.0, 55.0, 55.0, 55.0, 55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]);
        let grouping = clustering(
            &turns,
            &[&[0, 1, 2, 3, 4, 5], &[6, 7, 8, 9, 10, 11]],
            &voices,
            &[],
        );

        let sweep = reference_duration_sweep(&turns, &grouping, 30.0, &[5.0, 15.0, 25.0], 20.0);
        let failing: Vec<f64> = sweep
            .points
            .iter()
            .filter(|point| point.sampling == Sampling::Prefix && point.speaker == 0)
            .filter(|point| !point.starved_alone.is_correct())
            .map(|point| point.realized_s)
            .collect();
        assert_eq!(
            failing,
            [10.0, 20.0],
            "one and two turns of the wrong voice"
        );

        // Every floor above 20 s and at or below 30 s refuses exactly those two references and
        // writes every other measured one, so the band is that gap and not a fitted point.
        assert_eq!(sweep.band(Sampling::Prefix), Some((20.0, 30.0)));

        // And its price: speaker 1's references worked at 10 s and 20 s, and a floor inside the
        // band throws both away.
        let sacrificed = sweep.sacrificed(Sampling::Prefix);
        assert_eq!(sacrificed.len(), 2);
        assert!(sacrificed.iter().all(|point| point.speaker == 1));
    }

    #[test]
    fn a_stored_reference_is_measured_with_the_turn_it_was_built_from_held_out() {
        let (_, voices, grouping) = two_speakers();
        // Enrolled from turn 4 alone -- the shape of the 1.0 s fragments this measures.
        let enrolled = EnrolledSpeakers::new(vec![
            EnrolledSpeaker {
                name: "fragment".to_string(),
                embedding: voices[4].clone().unwrap(),
                clip_seconds: None,
            },
            EnrolledSpeaker {
                name: "wrong model".to_string(),
                embedding: vec![1.0, 0.0],
                clip_seconds: None,
            },
        ]);

        let stored = stored_reference_distances(&enrolled, &grouping, 30.0);
        assert_eq!(
            stored.len(),
            1,
            "a reference of another dimensionality is skipped"
        );
        assert_eq!(stored[0].origin.map(|(turn, _)| turn), Some(4));

        let (cluster, full, held_out) = stored[0].against[0];
        assert_eq!(cluster, 0);
        assert!(
            held_out > full,
            "holding out the turn the reference is did not move the number, so the exclusion \
             is doing nothing: {full} vs {held_out}"
        );
        let rest = group_mean(
            &(0..4)
                .map(|n| voices[n].as_deref().unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .0;
        assert!((held_out - distance(&rest, voices[4].as_deref().unwrap())).abs() < 1e-6);

        // The other speaker does not hold that turn, so its two numbers are the same one.
        let (cluster, full, held_out) = stored[0].against[1];
        assert_eq!(cluster, 1);
        assert_eq!(full, held_out);
    }

    #[test]
    fn a_fragment_is_probed_against_the_nearest_other_cluster() {
        let mut turns: Vec<LocalTurn> = (0..10)
            .map(|n| turn(n as f64 * 20.0, n as f64 * 20.0 + 10.0, 0))
            .collect();
        // Two below-floor fragments of one further voice, five degrees apart from each other and
        // far from both speakers.
        turns.push(turn(300.0, 302.0, 0));
        turns.push(turn(400.0, 402.0, 0));
        let mut degrees = vec![0.0, 2.0, 4.0, 6.0, 8.0, 60.0, 62.0, 64.0, 66.0, 68.0];
        degrees.push(150.0);
        degrees.push(155.0);
        let voices = voices(&degrees);
        let grouping = clustering(
            &turns,
            &[&[0, 1, 2, 3, 4], &[5, 6, 7, 8, 9], &[10], &[11]],
            &voices,
            &[],
        );

        let probe = fragment_probe(&grouping, 30.0, 2).expect("cluster 2 exists");
        assert_eq!(probe.seconds, 2.0);
        assert_eq!(probe.nearest.first().map(|&(id, _, _)| id), Some(3));
        assert_eq!(
            probe
                .against_speakers
                .iter()
                .map(|&(id, _)| id)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        // Five degrees apart is 0.004, well inside the cut, and no above-floor reference is
        // anywhere near: the fragment identifies its own other half.
        assert_eq!(probe.verdict, Some((3, Verdict::Correct)));

        assert_eq!(fragment_probe(&grouping, 30.0, 99), None);
    }

    #[test]
    fn degenerate_shapes_are_an_empty_sweep_rather_than_a_panic() {
        let empty = Clustering {
            clusters: Vec::new(),
            assignment: Vec::new(),
            turn_embeddings: Vec::new(),
        };
        let sweep = reference_duration_sweep(&[], &empty, 30.0, &[1.0, 2.0], 20.0);
        assert!(sweep.points.is_empty() && sweep.declined.is_empty());
        assert!(sweep.speakers.is_empty());
        assert_eq!(sweep.band(Sampling::Prefix), None);
        assert!(
            stored_reference_distances(&EnrolledSpeakers::new(Vec::new()), &empty, 30.0).is_empty()
        );
        assert_eq!(fragment_probe(&empty, 30.0, 0), None);

        // Nothing above the floor.
        let (long, _, grouping) = two_speakers();
        let sweep = reference_duration_sweep(&long, &grouping, 1000.0, &[1.0], 0.0);
        assert!(sweep.points.is_empty() && sweep.declined.is_empty());

        // A speaker of one turn: every grid value it can reach leaves nothing behind.
        let turns = vec![turn(0.0, 40.0, 0)];
        let voices = voices(&[10.0]);
        let one = clustering(&turns, &[&[0]], &voices, &[]);
        let sweep = reference_duration_sweep(&turns, &one, 30.0, &[1.0], 0.0);
        assert!(sweep.points.is_empty());
        assert_eq!(
            sweep.declined[0].reason,
            Decline::NoDirection,
            "a remainder of no turns has nothing to measure against"
        );
        let sweep = reference_duration_sweep(&turns, &one, 30.0, &[1.0], 20.0);
        assert_eq!(
            sweep.declined[0].reason,
            Decline::ThinRemainder { held_out_s: 0.0 },
            "and a remainder floor catches it before the direction does"
        );

        // Turns too short to embed: a cluster exists but holds no vectors.
        let turns = vec![turn(0.0, 20.0, 0), turn(30.0, 50.0, 0)];
        let voices = vec![None, None];
        let none = clustering(&turns, &[&[0, 1]], &voices, &[]);
        let sweep = reference_duration_sweep(&turns, &none, 30.0, &[1.0], 0.0);
        assert!(sweep.points.is_empty());
        assert_eq!(
            sweep.declined[0].reason,
            Decline::ShortOfGrid { available_s: 0.0 }
        );

        // An assignment naming a cluster and a turn that do not exist.
        let (turns, voices, mut grouping) = two_speakers();
        grouping.assignment.push(Some(99));
        grouping.turn_embeddings.push(voices[0].clone());
        let sweep = reference_duration_sweep(&turns, &grouping, 30.0, &[20.0], 10.0);
        assert_eq!(sweep.points.len(), 4);
    }
}
