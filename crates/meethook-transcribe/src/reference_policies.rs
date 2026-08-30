//! What an enrolled reference is built out of when a person is named from multiple recordings:
//! the three-policy simulation scored through the shipped identifier.
//!
//! Every verdict routes through the real [`crate::identify_clusters`] -- the strictly-below-cut
//! boundary it owns is not re-implemented here -- and the distance populations each arm produces
//! are scored with [`crate::trials::score_trials`].

use std::collections::BTreeMap;

use meethook_session::{EnrolledSpeaker, EnrolledSpeakers, SpeakerCluster};

use crate::identify::identify_clusters;
use crate::trials::{Spread, Trial, TrialReport, score_trials};
use crate::voice_vectors::group_mean;

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

    /// `[normalize((a + b) / 2)]`, via `voice_vectors::group_mean` rather than a second
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
/// whose members exactly cancel, which `voice_vectors::group_mean` declines rather than
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
            clip_seconds: None,
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
                clip_seconds: None,
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
                clip_seconds: None,
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
