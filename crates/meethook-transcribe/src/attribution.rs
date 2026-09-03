//! What a voice is called, derived once for every command that has to answer it.
//!
//! Two commands put labels on the same voices. `transcribe` writes them into a fresh
//! transcript; `enroll` recomputes them against the database as it stands and rewrites the
//! transcript in place. The invariant `enroll` is built against -- a transcript it rewrote is
//! exactly what `transcribe --force` would now produce -- is a claim that those two arrive at
//! the same map, cluster id to label, every time. Two implementations of one rule cannot make
//! that claim; they can only happen to agree, and the symptom of them not agreeing is a name
//! on the wrong person's turns. So the rule lives here, once, and both go through it.
//!
//! The numbering behind an "Unknown N" is deliberately not implemented here either. It lives
//! further down still, in [`meethook_session::unknown_labels`], because the on-disk contract
//! is what both `transcribe` and `enroll` have to recover the same numbers from. This module
//! decides only which of the three things a voice is: a name the user gave this session's
//! voice by hand, a name somebody enrolled, or the number its first appearance earned it.
//!
//! # Precedence
//!
//! Above all three sits the one-remote-speaker assertion: when the user has said this
//! session's speaker track is one person, every voice on it reads as that person and the
//! rest of the rule is moot -- see [`Naming::with_one_remote_speaker`] for why the
//! heard-at-once exclusion does not apply to the asserted name.
//!
//! Below the assertion and above everything else sits the forced-label tier:
//! [`Naming::with_forced`] names specific voices, exempting them from every exclusion the
//! rule otherwise applies. A forced label is the user's declaration that *these* voices are
//! one person, given against the session's own clustering while the run is walking it --
//! stronger than a hand-given name, which the exclusions still test, because the user has
//! already been shown the segmentation evidence and overridden it deliberately. The
//! exemption is the point rather than an exception: a forced pair that the segmenter heard
//! talking over each other keeps the shared name, exactly as an asserted pair does, only
//! scoped to the voices named. Forced holders count as holders for every non-forced voice,
//! so a conflicting assignment or identification on a voice they overlap falls back to its
//! number rather than slipping past the veto; and where a forced entry conflicts with the
//! voice's own assignment, the forced name replaces it. The assertion outranks the tier
//! outright: on a track the user has called one person, every voice reads the asserted name
//! whatever was forced.
//!
//! Below it, a hand-given name beats identification, which beats the tentative guess, which
//! beats the number. The order is not a preference between two guesses: a row in
//! `speaker_names.json` is a person saying *that voice, in this recording, is this human*,
//! and identification is a cosine distance between two vectors. Where they disagree, the one
//! that saw the meeting wins. A tentative guess is a looser cosine distance still -- see
//! [`crate::identify::tentative_identifications`] -- and it loses to both of them for the
//! same reason it exists at all: it is only ever a marked question, never an answer. The
//! number is what is left when nobody has made a claim at all.
//!
//! # "Attribution" here means what a cluster is *called*
//!
//! `merge` uses "attributed" in a second, older sense throughout -- which cluster said a
//! recognised segment, which is what [`meethook_session::Turn::cluster`] records and what
//! `merge::speaking_cluster` decides. That question is answered before this one is asked: a
//! turn is attributed to a cluster, and then the cluster carries an [`Attribution`]. Both
//! senses are in the codebase and neither is going away, so they are stated together here
//! rather than left for a reader to collide with.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use meethook_session::{AssignedName, DeniedName, SpeakerCluster};

use crate::identify::{Identification, heard_at_once};

/// What one voice on the speaker track is called, and what that label claims.
///
/// The three variants exist because a label carries two independent facts and a bare string
/// carries neither: whether it names a person, and how sure that naming is. Reading one off
/// the other -- "it has a confidence, so it must be a name" -- is what this type is here to
/// stop, and [`Assigned`] is the case where the two come apart.
///
/// [`Assigned`]: Attribution::Assigned
#[derive(Debug, Clone, PartialEq)]
pub enum Attribution {
    /// The "Unknown N" this voice's first appearance earned it. Names nobody.
    ///
    /// There is no confidence for a number to be the confidence *of*: the label makes no
    /// identity claim at all, so there is nothing to be more or less sure about.
    Unknown(String),

    /// Matched to an enrolled reference in `speakers.json`, at this similarity.
    Identified { name: String, similarity: f32 },

    /// A marked guess at a name the session already confirmed elsewhere, at the similarity
    /// it was guessed at.
    ///
    /// Different from [`Self::Identified`] in exactly the way the marker is different: the cut
    /// that admitted it is the looser one documented on
    /// [`crate::identify::TENTATIVE_DISTANCE`], so it is a machine similarity -- unlike
    /// [`Self::Assigned`], which carries none -- but a similarity measured against a wider
    /// tolerance, and it says so by rendering as `Name?`. It is also deliberately NOT named,
    /// below: a guess is an open question, and that single bit is what keeps `enroll`
    /// surfacing these voices and its pass-over gate honest rather than reading a guess as a
    /// settled answer.
    Tentative { name: String, similarity: f32 },

    /// Named by the user against this session alone, from `speaker_names.json`.
    ///
    /// A name with no similarity behind it, because there is no reference to have been near
    /// to: the voice was too short a fragment to enrol as one. That is not an unconfident
    /// naming, it is a naming that confidence is not a property of -- the person listened to
    /// the clip and said who it was -- which is why this carries no number rather than a low
    /// one. A low one would render as a hedge in the transcript and would sort this below a
    /// machine match, and both would misreport what happened.
    Assigned { name: String },
}

impl Attribution {
    /// The label as the transcript reads it: the bare name, the marked guess, or the number.
    ///
    /// The `?` rides the label string itself rather than living in any renderer: `merge`
    /// projects `label()` into `Turn.speaker`, and the json/md/VTT renderers see only the
    /// string (decision-010's contract is a stable JSON shape plus whatever the template can
    /// see), so a mark made here is a mark made everywhere with zero renderer changes.
    pub fn label(&self) -> Cow<'_, str> {
        match self {
            Attribution::Unknown(label) => Cow::Borrowed(label.as_str()),
            Attribution::Identified { name, .. } => Cow::Borrowed(name.as_str()),
            Attribution::Tentative { name, .. } => Cow::Owned(format!("{name}?")),
            Attribution::Assigned { name } => Cow::Borrowed(name.as_str()),
        }
    }

    /// How confident the identity claim in [`label`] is, or `None` when it makes none.
    ///
    /// This is what lands in [`meethook_session::Turn::speaker_id_confidence`]. A tentative
    /// guess reports its real similarity, like an identification: the number is true, and the
    /// marker is what records the wider tolerance it was measured under rather than a lower
    /// number pretending to be a tighter one.
    ///
    /// [`label`]: Attribution::label
    pub fn confidence(&self) -> Option<f32> {
        match self {
            Attribution::Unknown(_) | Attribution::Assigned { .. } => None,
            Attribution::Identified { similarity, .. }
            | Attribution::Tentative { similarity, .. } => Some(*similarity),
        }
    }

    /// Does this label name a person?
    ///
    /// The question `enroll` is actually asking everywhere it decides what to prompt about,
    /// which voices are already resolved, and whether an answer given earlier in the run has
    /// since named this one.
    ///
    /// A tentative guess is deliberately NOT named, although it carries a name: it is an open
    /// question rendered as text, and counting it as settled would make `enroll` stop asking
    /// about exactly the voices the band exists to surface. The name in the label and the
    /// answer to this question are independent facts, the same way [`Self::Assigned`] shows they
    /// are independent of confidence.
    ///
    /// Deliberately **not** "carries a similarity", and deliberately not spelled
    /// `confidence().is_some()`. [`Attribution::Assigned`] is exactly the case the two answers
    /// differ on: a name with no similarity behind it, which a caller written against the
    /// confidence half would re-ask about on every run.
    pub fn is_named(&self) -> bool {
        matches!(
            self,
            Attribution::Identified { .. } | Attribution::Assigned { .. }
        )
    }
}

/// Everything known about who one session's voices are, as the naming rule needs to read it.
///
/// A bundle rather than three parameters because the three are one input: the assignments
/// cannot be resolved without the clusters they were recorded against, and the veto between an
/// assignment and an identification needs all three at once. Callers that have nothing to say
/// beyond identification -- every test of `merge`'s timeline behaviour, and any caller from
/// before hand-given names existed -- use [`Naming::nothing`] and one of the `with_` builders
/// rather than spelling out empties.
#[derive(Debug, Clone, Copy)]
pub struct Naming<'a> {
    /// This session's clusters, as `speaker_clusters.json` holds them. What an assignment is
    /// resolved through, and where the heard-at-once exclusions live.
    clusters: &'a [SpeakerCluster],
    identified: &'a BTreeMap<u32, Identification>,
    assigned: &'a [AssignedName],
    /// The user's assertion that this session's speaker track is one person, by name.
    ///
    /// `None` applies no such claim and the rule runs exactly as it always has; `Some` names
    /// every voice on the track and settles the question the exclusions below exist to keep
    /// open.
    one_remote_speaker: Option<&'a str>,
    /// The forced-label tier: the voices the user has declared one person, by cluster id and
    /// name. An empty map applies no such claim and the rule runs exactly as it always has;
    /// see the module doc for the tier's rank and why its exemption is the point.
    forced: &'a BTreeMap<u32, String>,
    /// The voice an in-progress answer is about, if any: its hand-given row is what the answer
    /// would create rather than a standing declaration, so it earns no pass in the assignment
    /// award below -- which is what leaves the vetoed demotion for the dry run's refusal to fire
    /// off. `None` on every reading that is not previewing an answer.
    pending: Option<u32>,
    /// The tentative band's answers: sub-floor fragments guessed at names the session already
    /// confirmed elsewhere, under the looser cut documented on
    /// [`crate::identify::TENTATIVE_DISTANCE`]. An empty map applies no such claims and the
    /// rule runs exactly as it always has; the tier's rank and its demotions are stated in
    /// the module doc and on the variant itself.
    tentative: &'a BTreeMap<u32, Identification>,
    /// Standing denials, resolved: the `(cluster, name)` pairs a user refused, ready to apply.
    /// Supplied by the caller through [`resolve_denials`] -- one implementation, called from
    /// both `transcribe` and `enroll` -- because bit-exact matching against the session's
    /// clusters is the removal's own rule, and a second copy of it could drift out of
    /// agreement with the write it previews.
    denied: &'a BTreeSet<(u32, String)>,
}

/// Somewhere for [`Naming::nothing`] to point its identification map at.
static NOBODY: BTreeMap<u32, Identification> = BTreeMap::new();

/// Somewhere for [`Naming::new`] and [`Naming::nothing`] to point their forced set at.
static NO_FORCED: BTreeMap<u32, String> = BTreeMap::new();

/// Somewhere for every caller before the tentative band existed to point it at.
static NO_TENTATIVE: BTreeMap<u32, Identification> = BTreeMap::new();

/// Somewhere for every caller without standing denials to point them at.
static NO_DENIED: BTreeSet<(u32, String)> = BTreeSet::new();

impl<'a> Naming<'a> {
    pub fn new(
        clusters: &'a [SpeakerCluster],
        identified: &'a BTreeMap<u32, Identification>,
        assigned: &'a [AssignedName],
    ) -> Self {
        Naming {
            clusters,
            identified,
            assigned,
            one_remote_speaker: None,
            forced: &NO_FORCED,
            pending: None,
            tentative: &NO_TENTATIVE,
            denied: &NO_DENIED,
        }
    }

    /// Nobody is identified and nobody has been named by hand: every voice keeps its number.
    pub fn nothing() -> Self {
        Naming {
            clusters: &[],
            identified: &NOBODY,
            assigned: &[],
            one_remote_speaker: None,
            forced: &NO_FORCED,
            pending: None,
            tentative: &NO_TENTATIVE,
            denied: &NO_DENIED,
        }
    }

    /// The same naming under the user's assertion that this session's speaker track is one
    /// person, `name` -- or unchanged when `name` is `None`, which is how every caller that
    /// has nothing to assert stays exactly as it was.
    ///
    /// The assertion outranks everything the rule otherwise decides: every voice on the track
    /// reads as that person, whatever an assignment, an identification, or the heard-at-once
    /// exclusion would have said. On a track the user has called one person, an overlap is
    /// echo, diarisation error, or somebody who was never on the invite -- not evidence of two
    /// people -- so the exclusion is not applied to the asserted name rather than applied and
    /// then undone. That is also why the bypass lives here, in the one place both `transcribe`
    /// and `enroll` label through, rather than in either of them: the module invariant above
    /// is that two implementations of one rule cannot agree by accident.
    pub fn with_one_remote_speaker(self, name: Option<&'a str>) -> Self {
        Naming {
            one_remote_speaker: name,
            ..self
        }
    }

    /// The same naming with the forced-label tier filled in: every id in `forced` reads its
    /// name unconditionally, exempt from the heard-at-once exclusion and from any assignment
    /// recorded against it, while every other voice treats the forced entries as additional
    /// holders of their names.
    ///
    /// The tier sits between the assertion and the hand-given names -- see the module doc for
    /// the rank and the reason the exemption is deliberate. An empty map, and every caller
    /// before this existed, leaves the rule byte-identical to how it ran.
    pub fn with_forced(self, forced: &'a BTreeMap<u32, String>) -> Self {
        Naming { forced, ..self }
    }

    /// The same naming while an answer about `pending` is being previewed: that voice's own
    /// row earns no co-declaration pass in the assignment award, because the row is what the
    /// answer would write, not a decision already made -- awarding it would hide the cost the
    /// dry run exists to show. Every other row stands as usual.
    pub fn with_pending(self, pending: u32) -> Self {
        Naming {
            pending: Some(pending),
            ..self
        }
    }

    /// The same naming with the tentative band's answers filled in: a sub-floor fragment reads
    /// its marked guess wherever the tiers above it do not speak, subject to the same
    /// heard-at-once demotion an identification gets and to the standing denials below. An
    /// empty map -- and every caller before the band existed -- leaves the rule byte-identical
    /// to how it ran.
    pub fn with_tentative(self, tentative: &'a BTreeMap<u32, Identification>) -> Self {
        Naming { tentative, ..self }
    }

    /// The same naming under standing denials: the `(cluster, name)` pairs a user refused,
    /// resolved through [`resolve_denials`]. A denied pair suppresses the tentative guess it
    /// names -- a refusal is about the voice, and the embedding is the identity that ties the
    /// fragment to the refused cluster -- leaving the voice with its number rather than a
    /// guess the user has already answered. No denial applies to a definite name here: that
    /// is the database's business, done where the reference goes.
    pub fn with_denials(self, denied: &'a BTreeSet<(u32, String)>) -> Self {
        Naming { denied, ..self }
    }

    /// The same naming with identification's answers filled in.
    pub fn with_identified(self, identified: &'a BTreeMap<u32, Identification>) -> Self {
        Naming { identified, ..self }
    }

    /// Which cluster each hand-given name actually applies to *now*, name included.
    ///
    /// Two things happen here, and both are the reason assignments are not simply read off
    /// [`AssignedName::cluster`].
    ///
    /// **Resolution.** A row is matched to the cluster whose embedding is bit-for-bit the one
    /// recorded with it. Exact equality is the whole condition: the embedding is a copy of an
    /// array through one file format, not a measurement, so anything that moved it at all came
    /// from a re-clustering that redrew this voice. Matching no cluster, or more than one,
    /// drops the row -- a name given against a clustering that no longer exists is a claim
    /// about words that no longer exist, and applying it by cluster id would put it on whoever
    /// inherited the number.
    ///
    /// **The exclusion.** One name cannot land on two voices segmentation heard talking over
    /// each other; that pair is proof of two people, so the second is somebody else however
    /// certain the user was. Resolved exactly as [`crate::identify_clusters`] resolves the same
    /// conflict between two matches -- greedy, in ascending cluster id, against every voice
    /// already holding the name rather than just the first -- and for the reasons argued there.
    /// Ascending id rather than by similarity because an assignment has no similarity to order
    /// by; it is arbitrary, and being arbitrary in a fixed way is what matters.
    ///
    /// Two exceptions to the exclusion, and both exist because one name on two overlapping
    /// voices is otherwise impossible to record at all. **Co-declaration:** two *standing*
    /// rows of one name on overlapping voices are the user's own group declaration -- made
    /// together, with the heard-at-once override reported at the moment it was made -- and
    /// both stand; demoting one of them afterwards would undo a decision the database holds
    /// and re-prompt forever. **Pending:** the row an in-progress answer would create ([`Naming`
    /// 's `pending`) earns no such pass -- it is not a decision already made, and awarding it
    /// here would hide the very cost the dry run exists to show. A forced holder outranks
    /// both: the tier replaces whatever a row says for its voice.
    fn awarded(&self) -> BTreeMap<u32, String> {
        let mut resolved: BTreeMap<u32, &str> = BTreeMap::new();
        for row in self.assigned {
            // Unique or nothing. `or_insert` settles a hand-edited file that named one
            // cluster twice by keeping the first row, so the outcome is still deterministic.
            if let Some(cluster) = resolve_by_embedding(self.clusters, &row.embedding) {
                resolved.entry(cluster.id).or_insert(&row.name);
            }
        }

        // The forced labels hold first: they are the user's declaration that those voices are
        // one person, so they stand as holders before any hand-given name is considered, and
        // an award that would put a conflicting name on a voice heard at once with one of
        // them loses here rather than later. A forced id's own assignment is never awarded --
        // the forced name replaces it, which is also what keeps the seed from being shadowed
        // by a row the tier has already overruled.
        let mut awarded: BTreeMap<u32, String> = BTreeMap::new();
        for (id, name) in self.forced {
            awarded.insert(*id, name.clone());
        }
        // Standing assignment holders only: the pending test below asks whether the row under
        // preview is contesting a decision already made, and a forced label is not one.
        let mut standing: BTreeMap<u32, String> = BTreeMap::new();
        for (id, name) in resolved {
            if self.forced.contains_key(&id) {
                continue;
            }
            let conflicts_forced = self
                .forced
                .iter()
                .any(|(held, held_name)| held_name == name && self.heard_at_once(id, *held));
            let contests_standing = self.pending == Some(id)
                && standing
                    .iter()
                    .any(|(held, held_name)| held_name == name && self.heard_at_once(id, *held));
            if conflicts_forced || contests_standing {
                continue;
            }
            awarded.insert(id, name.to_string());
            standing.insert(id, name.to_string());
        }
        awarded
    }

    /// Whether giving `id` the name `name` would put it on two voices heard at once.
    ///
    /// `holders` is who already has names; the caller decides whether that means the
    /// assignments awarded so far or all of them, which is the difference between the
    /// assignment-versus-assignment conflict and the assignment-versus-identification one.
    fn overlaps_a_holder_of(&self, id: u32, name: &str, holders: &BTreeMap<u32, String>) -> bool {
        holders
            .iter()
            .any(|(&held, held_name)| held_name == name && self.heard_at_once(id, held))
    }

    /// Did segmentation hear these two cluster ids overlapping? Unknown ids never exclude:
    /// a caller may label voices this clustering does not describe, and inventing an exclusion
    /// for one would drop a name on no evidence.
    fn heard_at_once(&self, a: u32, b: u32) -> bool {
        match (self.cluster(a), self.cluster(b)) {
            (Some(a), Some(b)) => heard_at_once(a, b),
            _ => false,
        }
    }

    fn cluster(&self, id: u32) -> Option<&SpeakerCluster> {
        self.clusters.iter().find(|cluster| cluster.id == id)
    }
}

/// What every voice is called: a name the user gave this session's voice where there is one,
/// an enrolled name where identification found one, otherwise the "Unknown N" that voice's
/// first appearance earned it.
///
/// `unknown` is the [`meethook_session::unknown_labels`] map -- one entry per voice, and it is
/// the key set of the result. Callers build it two different ways (`transcribe` from the
/// diarized turns, `enroll` from `speaker_clusters.json`) and get the same map, which is the
/// whole reason it is a parameter rather than something derived in here.
///
/// An id in `naming` that `unknown` does not hold is dropped: a voice nothing knows the
/// existence of cannot be labelled, and the caller that built `unknown` is the one that knows
/// which voices this session has.
///
/// Taken by reference because `enroll` holds its `unknown` for the length of a prompt loop and
/// re-derives the attributions inside it.
///
/// # An identification a hand-given name contradicts
///
/// One person cannot be two voices heard talking over each other, and that holds however each
/// of the two names arrived. So a match to somebody an assignment has already put on a voice
/// this one overlaps is refused, and the voice falls back to its number -- never to its
/// second-nearest reference, for the reason [`crate::identify_clusters`] gives at length: the
/// runner-up is the one that already lost the argmax, and awarding it is how a person's words
/// end up filed under somebody else's name.
///
/// The refusal is one-sided. The assignment always wins, because it is the claim a human made
/// and the identification is a distance between two vectors.
pub fn attributions(
    unknown: &BTreeMap<u32, String>,
    naming: Naming<'_>,
) -> BTreeMap<u32, Attribution> {
    // The assertion, when there is one, settles the whole question before any of it runs:
    // every voice on the track is the person it names, so the exclusions below -- which exist
    // to keep two people apart -- have no work to do. See `with_one_remote_speaker` for why
    // overriding them is the point rather than the exception.
    if let Some(name) = naming.one_remote_speaker {
        return unknown
            .iter()
            .map(|(&id, _)| {
                (
                    id,
                    Attribution::Assigned {
                        name: name.to_string(),
                    },
                )
            })
            .collect();
    }
    let assigned = naming.awarded();
    unknown
        .iter()
        .map(|(&id, label)| {
            // The forced tier, after the assertion and before everything else: the voice reads
            // the name the user declared for it, whatever an assignment or an identification
            // says, and the exclusion is not applied to it rather than applied and then undone
            // -- the user declared these voices one person, and the segmentation evidence is
            // overridden for them, reported elsewhere.
            let attribution = if let Some(name) = naming.forced.get(&id) {
                Attribution::Assigned { name: name.clone() }
            } else if let Some(name) = assigned.get(&id) {
                Attribution::Assigned { name: name.clone() }
            } else if let Some(who) = naming
                .identified
                .get(&id)
                .filter(|who| !naming.overlaps_a_holder_of(id, &who.name, &assigned))
            {
                Attribution::Identified {
                    name: who.name.clone(),
                    similarity: who.similarity,
                }
            } else if let Some(guess) = naming
                .tentative
                .get(&id)
                // A standing denial suppresses the guess it names, whatever the distance says:
                // the refusal is about the voice, and this is the voice it was recorded against.
                .filter(|guess| !naming.denied.contains(&(id, guess.name.clone())))
                // The same demotion an identification gets, for the same reason: a hand-named
                // holder on an overlapping voice is a human claim, and a human claim outranks a
                // vector distance -- a looser one at that.
                .filter(|guess| !naming.overlaps_a_holder_of(id, &guess.name, &assigned))
            {
                Attribution::Tentative {
                    name: guess.name.clone(),
                    similarity: guess.similarity,
                }
            } else {
                Attribution::Unknown(label.clone())
            };
            (id, attribution)
        })
        .collect()
}

/// Which `(cluster, name)` pairs a set of denials removes from a rewrite, resolved against
/// the session's own clusters.
///
/// The matching rule is deliberately the affirmation path's own (`Naming::awarded`):
/// bit-exact embedding equality, and a match that is not unique is DROPPED rather than
/// guessed at. Two clusters sharing one centroid -- the shape a stale model or a
/// re-clustering leaves behind -- are exactly where suppressing the wrong one is most costly,
/// so the conservative answer (suppress nothing) is the only defensible one, and it is the
/// answer the rows the user DID give get for the same input. One implementation, called from
/// both `transcribe` and `enroll`, is what keeps a preview and a write from resolving the
/// same refusal to different rows: a preview that demotes a different voice than the write
/// does is a lie with extra steps.
pub fn resolve_denials(
    clusters: &[SpeakerCluster],
    denied: &[DeniedName],
) -> BTreeSet<(u32, String)> {
    let mut removed = BTreeSet::new();
    for denial in denied {
        // The same unique-or-nothing rule `awarded` resolves its rows through, and the same
        // function: a preview and a write that resolved a stored embedding to two different
        // clusters would be a lie with extra steps.
        if let Some(cluster) = resolve_by_embedding(clusters, &denial.embedding) {
            removed.insert((cluster.id, denial.name.clone()));
        }
    }
    removed
}

/// Resolves a stored embedding to the one cluster it bit-exactly matches, or `None` when the
/// match is not unique.
///
/// The shared rule an affirmation ([`Naming::awarded`]) and a denial ([`resolve_denials`])
/// both resolve their rows through: a stored embedding is a handle into *some* clustering, and
/// re-clustering can leave it matching zero or several current clusters. Either way there is no
/// single answer to apply the row to, so the conservative one -- resolve to nothing -- is what
/// both callers get, from one place, so they cannot drift into resolving the same embedding to
/// different clusters.
fn resolve_by_embedding<'a>(
    clusters: &'a [SpeakerCluster],
    embedding: &[f32],
) -> Option<&'a SpeakerCluster> {
    let mut matching = clusters.iter().filter(|c| c.embedding == embedding);
    match (matching.next(), matching.next()) {
        (Some(cluster), None) => Some(cluster),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unknown(ids: &[(u32, &str)]) -> BTreeMap<u32, String> {
        ids.iter()
            .map(|(id, label)| (*id, label.to_string()))
            .collect()
    }

    fn identified(entries: &[(u32, &str, f32)]) -> BTreeMap<u32, Identification> {
        entries
            .iter()
            .map(|(id, name, similarity)| {
                (
                    *id,
                    Identification {
                        name: name.to_string(),
                        similarity: *similarity,
                    },
                )
            })
            .collect()
    }

    /// A cluster whose embedding is its own id, so that a test can say which cluster a name
    /// was recorded against without repeating a vector.
    fn cluster(id: u32, heard_at_once_with: Vec<u32>) -> SpeakerCluster {
        SpeakerCluster {
            id,
            embedding: vec![id as f32, 0.5],
            speech_seconds: 10.0,
            first_spoke_seconds: 5.0 + id as f64,
            heard_at_once_with,
            representatives: Vec::new(),
        }
    }

    fn clusters(ids: &[u32]) -> Vec<SpeakerCluster> {
        ids.iter().map(|&id| cluster(id, Vec::new())).collect()
    }

    /// A name recorded against the cluster `cluster(id, _)` would have had.
    fn assignment(id: u32, name: &str) -> AssignedName {
        AssignedName {
            cluster: id,
            name: name.to_string(),
            embedding: cluster(id, Vec::new()).embedding,
        }
    }

    /// The whole rule, in one case of each: a voice the database recognised reads as that
    /// person and carries the similarity it was matched at, and the voice beside it that
    /// nothing matched keeps the number its first appearance earned it.
    #[test]
    fn a_matched_voice_carries_the_name_and_an_unmatched_one_keeps_its_number() {
        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::nothing().with_identified(&identified(&[(1, "Alice", 0.83)])),
        );

        assert_eq!(map[&0], Attribution::Unknown("Unknown 1".to_string()));
        assert_eq!(map[&0].label(), "Unknown 1");
        assert_eq!(map[&0].confidence(), None);
        assert_eq!(
            map[&1],
            Attribution::Identified {
                name: "Alice".to_string(),
                similarity: 0.83,
            }
        );
        assert_eq!(map[&1].label(), "Alice");
        assert_eq!(map[&1].confidence(), Some(0.83));
    }

    /// `is_named` asserted as the question it asks -- does this label name a person -- rather
    /// than through `confidence()`. A name the user assigned has no similarity behind it and
    /// must still count as named, or `enroll` would ask about that voice again every run. And
    /// a tentative guess carries a name yet is NOT named: it is an open question rendered as
    /// text, and counting it as settled would make `enroll` stop asking about exactly the
    /// voices the band exists to surface.
    #[test]
    fn only_a_label_that_names_somebody_is_named() {
        assert!(
            Attribution::Identified {
                name: "Alice".to_string(),
                similarity: 0.83,
            }
            .is_named()
        );
        assert!(
            Attribution::Assigned {
                name: "Alice".to_string(),
            }
            .is_named()
        );
        assert!(!Attribution::Unknown("Unknown 1".to_string()).is_named());
        assert!(
            !Attribution::Tentative {
                name: "Ivan".to_string(),
                similarity: 0.59,
            }
            .is_named()
        );
    }

    /// The label a user gave reads as a name and claims no confidence, which is what makes it
    /// different from an identification rather than a weaker one.
    #[test]
    fn an_assigned_name_is_a_label_with_no_confidence() {
        let assigned = Attribution::Assigned {
            name: "Alex".to_string(),
        };

        assert_eq!(assigned.label(), "Alex");
        assert_eq!(assigned.confidence(), None);
    }

    /// Identification runs over the clusters; the labels run over the voices the caller knows
    /// about. Where those disagree the caller wins, which is the behaviour both of the
    /// derivations this replaced had by construction and neither stated.
    #[test]
    fn an_identification_for_a_voice_the_caller_does_not_know_about_is_dropped() {
        let map = attributions(
            &unknown(&[(0, "Unknown 1")]),
            Naming::nothing().with_identified(&identified(&[(0, "Alice", 0.83), (7, "Bob", 0.91)])),
        );

        assert_eq!(map.keys().copied().collect::<Vec<u32>>(), [0]);
    }

    /// Every voice gets a label, whether or not anybody was enrolled -- the normal state of
    /// an install before the first `enroll` run.
    #[test]
    fn every_voice_is_labelled_when_nobody_has_been_named() {
        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2"), (2, "Unknown 3")]),
            Naming::nothing(),
        );

        assert_eq!(map.keys().copied().collect::<Vec<u32>>(), [0, 1, 2]);
        assert!(map.values().all(|a| !a.is_named()));
    }

    /// The point of the whole file: a voice too quiet to enrol still reads as the person the
    /// user said it was, in this session's transcript, with nothing in `speakers.json`.
    #[test]
    fn a_voice_named_by_hand_reads_as_that_person() {
        let clusters = clusters(&[0, 1]);

        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(&clusters, &NOBODY, &[assignment(1, "Alex")]),
        );

        assert_eq!(
            map[&1],
            Attribution::Assigned {
                name: "Alex".to_string(),
            }
        );
        assert_eq!(map[&0], Attribution::Unknown("Unknown 1".to_string()));
    }

    /// Precedence, in the case where the two claims are about the same voice. The person who
    /// listened to the clip outranks a cosine distance, and the similarity goes with the
    /// answer it belonged to rather than being carried over onto the name that won.
    #[test]
    fn a_name_the_user_gave_beats_an_identification_of_the_same_voice() {
        let clusters = clusters(&[0]);

        let map = attributions(
            &unknown(&[(0, "Unknown 1")]),
            Naming::new(
                &clusters,
                &identified(&[(0, "Alice", 0.83)]),
                &[assignment(0, "Alex")],
            ),
        );

        assert_eq!(
            map[&0],
            Attribution::Assigned {
                name: "Alex".to_string(),
            }
        );
        assert_eq!(map[&0].confidence(), None);
    }

    /// A name is resolved through the embedding it was recorded with, not the cluster id.
    /// Re-diarization renumbers voices, so applying a stale row by id would put a name on
    /// whoever inherited the number -- silently, and on words the user never heard.
    #[test]
    fn a_name_recorded_against_a_clustering_that_no_longer_exists_is_ignored() {
        let clusters = clusters(&[0, 1]);
        let stale = AssignedName {
            embedding: vec![1.000_001, 0.5],
            ..assignment(1, "Alex")
        };

        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(&clusters, &NOBODY, &[stale]),
        );

        assert_eq!(map[&1], Attribution::Unknown("Unknown 2".to_string()));
    }

    /// The other half of resolution: an embedding that matches two clusters identifies
    /// neither, because there is no way to tell which of them the user was listening to.
    #[test]
    fn a_name_matching_more_than_one_cluster_is_ignored() {
        // Two clusters that came out of re-diarization holding the same centroid.
        let twin = SpeakerCluster {
            embedding: cluster(0, Vec::new()).embedding,
            ..cluster(1, Vec::new())
        };
        let clusters = vec![cluster(0, Vec::new()), twin];

        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(&clusters, &NOBODY, &[assignment(0, "Alex")]),
        );

        assert!(map.values().all(|a| !a.is_named()), "{map:?}");
    }

    /// Two *standing* rows of one name on overlapping voices are the user's group declaration
    /// -- the only path that writes such a pair reports the heard-at-once override at the moment
    /// it is made -- so both stand rather than one being demoted afterwards. What the
    /// exclusion still refuses is the *fresh* answer about one of those voices, which previews
    /// with itself pending and so earns no pass here.
    #[test]
    fn one_name_cannot_land_on_two_voices_heard_talking_over_each_other() {
        let clusters = vec![cluster(0, vec![1]), cluster(1, vec![0])];

        // Standing: the declaration is already made, and both halves of it stand.
        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(
                &clusters,
                &NOBODY,
                &[assignment(1, "Alex"), assignment(0, "Alex")],
            ),
        );
        assert_eq!(
            map[&0],
            Attribution::Assigned {
                name: "Alex".to_string()
            }
        );
        assert_eq!(
            map[&1],
            Attribution::Assigned {
                name: "Alex".to_string()
            }
        );

        // Fresh: an answer about cluster 1 would create its row, not stand on it, so the award
        // is withheld and the voice falls back to its number -- the cost a refusal shows off.
        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(
                &clusters,
                &NOBODY,
                &[assignment(1, "Alex"), assignment(0, "Alex")],
            )
            .with_pending(1),
        );
        assert_eq!(
            map[&0],
            Attribution::Assigned {
                name: "Alex".to_string()
            }
        );
        assert_eq!(map[&1], Attribution::Unknown("Unknown 2".to_string()));
    }

    /// Two voices that never overlapped may both be one person -- clustering split a voice in
    /// two -- exactly as identification allows, so the exclusion has to be positive evidence
    /// rather than a rule against repeating a name.
    #[test]
    fn one_name_may_land_on_two_voices_never_heard_at_once() {
        let clusters = clusters(&[0, 1]);

        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(
                &clusters,
                &NOBODY,
                &[assignment(0, "Alex"), assignment(1, "Alex")],
            ),
        );

        assert!(map.values().all(|a| a.label() == "Alex"), "{map:?}");
    }

    /// The cross-kind case: a match to somebody the user has already put on a voice this one
    /// was heard talking over is refused. It falls to its number, never to the reference that
    /// already lost its argmax.
    #[test]
    fn an_identification_a_hand_given_name_contradicts_falls_back_to_the_number() {
        let clusters = vec![cluster(0, vec![1]), cluster(1, vec![0])];

        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(
                &clusters,
                &identified(&[(1, "Alex", 0.83)]),
                &[assignment(0, "Alex")],
            ),
        );

        assert_eq!(
            map[&0],
            Attribution::Assigned {
                name: "Alex".to_string(),
            }
        );
        assert_eq!(map[&1], Attribution::Unknown("Unknown 2".to_string()));
    }

    /// And the refusal is only for the voice the exclusion is about: a match to the same
    /// person on a voice nothing says is somebody else stands, because clustering splitting
    /// one person in two is the ordinary reason for it.
    #[test]
    fn an_identification_stands_when_nothing_says_the_two_voices_are_different_people() {
        let clusters = clusters(&[0, 1]);

        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(
                &clusters,
                &identified(&[(1, "Alex", 0.83)]),
                &[assignment(0, "Alex")],
            ),
        );

        assert_eq!(
            map[&1],
            Attribution::Identified {
                name: "Alex".to_string(),
                similarity: 0.83,
            }
        );
    }

    // --- the one-remote-speaker assertion -----------------------------------------------

    /// The assertion outranks everything the rule otherwise decides: a hand-given name and an
    /// identification both stand without it, and with it every voice reads as the asserted
    /// person instead -- carrying no confidence, the way every session-assigned name does.
    #[test]
    fn an_asserted_name_beats_an_assignment_and_an_identification_on_every_voice() {
        let clusters = clusters(&[0, 1]);
        let identified = identified(&[(1, "Alice", 0.83)]);
        let assigned = vec![assignment(0, "Alex")];

        let naming =
            Naming::new(&clusters, &identified, &assigned).with_one_remote_speaker(Some("Grace"));

        let map = attributions(&unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]), naming);

        for label in map.values() {
            assert_eq!(
                label,
                &Attribution::Assigned {
                    name: "Grace".to_string()
                }
            );
            assert_eq!(label.confidence(), None);
        }
    }

    /// The hard part: voices the segmenter heard talking over each other are exactly what the
    /// heard-at-once exclusion refuses to put under one name, and the assertion overrides it
    /// rather than losing to it. Without the assertion the lower id keeps the name and the
    /// other falls back to its number; with it, both keep it.
    #[test]
    fn the_heard_at_once_exclusion_does_not_apply_to_an_asserted_name() {
        let clusters = vec![cluster(0, vec![1]), cluster(1, vec![0])];

        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(&clusters, &NOBODY, &[]).with_one_remote_speaker(Some("Grace")),
        );

        assert_eq!(
            map[&0],
            Attribution::Assigned {
                name: "Grace".to_string()
            }
        );
        assert_eq!(
            map[&1],
            Attribution::Assigned {
                name: "Grace".to_string()
            }
        );
    }

    /// A re-diarised session still honours the assertion: the bit-exact rows die when the
    /// clustering changes, but the assertion is a session-level fact, so a fresh labelling
    /// names every new voice with it. This is what makes it more durable than assignments.
    #[test]
    fn an_assertion_survives_reclustering_that_a_stale_assignment_cannot() {
        // The assignment was recorded against a clustering that no longer exists, so it drops
        // out of the rule entirely -- while the assertion, which resolves through nothing,
        // still names the voice.
        let stale = AssignedName {
            embedding: vec![1.000_001, 0.5],
            ..assignment(1, "Alex")
        };
        let clusters = clusters(&[0, 1]);

        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(&clusters, &NOBODY, &[stale]).with_one_remote_speaker(Some("Grace")),
        );

        assert_eq!(
            map[&0],
            Attribution::Assigned {
                name: "Grace".to_string()
            }
        );
        assert_eq!(
            map[&1],
            Attribution::Assigned {
                name: "Grace".to_string()
            }
        );
    }

    // --- the forced-label tier -------------------------------------------------------------

    /// A forced set built the way `enroll` builds one: the resolved members' ids mapped to the
    /// group's name.
    fn forced(entries: &[(u32, &str)]) -> BTreeMap<u32, String> {
        entries
            .iter()
            .map(|(id, name)| (*id, name.to_string()))
            .collect()
    }

    /// The tier's rank on the voice it names: an assignment and an identification both stand
    /// without the forcing, and with it the voice reads the forced name -- carrying no
    /// confidence, the way every session-assigned name does -- while the voice beside it is
    /// untouched by a declaration made about somebody else.
    #[test]
    fn a_forced_label_beats_an_assignment_and_an_identification_on_that_voice_only() {
        let clusters = clusters(&[0, 1]);
        let id = identified(&[(0, "Alice", 0.83)]);
        let rows = vec![assignment(0, "Alex")];
        let f = forced(&[(0, "Grace")]);
        let naming = Naming::new(&clusters, &id, &rows).with_forced(&f);

        let map = attributions(&unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]), naming);

        assert_eq!(
            map[&0],
            Attribution::Assigned {
                name: "Grace".to_string()
            }
        );
        assert_eq!(map[&0].confidence(), None);
        // The neighbour keeps what the rule would have given it: nothing was said about it.
        assert_eq!(map[&1], Attribution::Unknown("Unknown 2".to_string()));
    }

    /// The hard part, scoped: voices the segmenter heard talking over each other are exactly
    /// what the heard-at-once exclusion refuses to put under one name, and the forcing
    /// overrides it rather than losing to it -- for the pair it declares, only. Without the
    /// forcing the lower id keeps the name and the other falls back to its number; with it,
    /// both keep it.
    #[test]
    fn the_heard_at_once_exclusion_does_not_apply_to_a_forced_name() {
        let pair = forced(&[(0, "Grace"), (1, "Grace")]);
        let mut clusters = vec![cluster(0, Vec::new()), cluster(1, Vec::new())];
        clusters[0].heard_at_once_with = vec![1];
        clusters[1].heard_at_once_with = vec![0];

        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(&clusters, &NOBODY, &[]).with_forced(&pair),
        );

        assert_eq!(
            map[&0],
            Attribution::Assigned {
                name: "Grace".to_string()
            }
        );
        assert_eq!(
            map[&1],
            Attribution::Assigned {
                name: "Grace".to_string()
            }
        );
    }

    /// The exemption cuts one way only: a forced holder still counts as a holder for every
    /// non-forced voice, so a hand-given name that would land on two voices heard at once is
    /// refused here -- falling back to the number, never to some other name.
    #[test]
    fn a_forced_holder_excludes_a_conflicting_assignment() {
        let mut clusters = vec![cluster(0, Vec::new()), cluster(1, Vec::new())];
        clusters[0].heard_at_once_with = vec![1];
        clusters[1].heard_at_once_with = vec![0];
        let f = forced(&[(0, "Grace")]);

        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(&clusters, &NOBODY, &[assignment(1, "Grace")]).with_forced(&f),
        );

        assert_eq!(
            map[&0],
            Attribution::Assigned {
                name: "Grace".to_string()
            }
        );
        assert_eq!(map[&1], Attribution::Unknown("Unknown 2".to_string()));
    }

    /// And the same holding power against an identification: a match to the person a forced
    /// label already put on a voice this one was heard talking over is refused, and the voice
    /// falls to its number rather than to the reference that already lost.
    #[test]
    fn a_forced_holder_excludes_a_conflicting_identification() {
        let mut clusters = vec![cluster(0, Vec::new()), cluster(1, Vec::new())];
        clusters[0].heard_at_once_with = vec![1];
        clusters[1].heard_at_once_with = vec![0];
        let f = forced(&[(0, "Grace")]);

        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(&clusters, &identified(&[(1, "Grace", 0.95)]), &[]).with_forced(&f),
        );

        assert_eq!(
            map[&0],
            Attribution::Assigned {
                name: "Grace".to_string()
            }
        );
        assert_eq!(map[&1], Attribution::Unknown("Unknown 2".to_string()));
    }

    /// The top of the stack: the assertion outranks the forced tier outright, so a track the
    /// user has called one person reads the asserted name even where a different name was
    /// forced onto part of it.
    #[test]
    fn an_assertion_outranks_a_forced_label() {
        let clusters = clusters(&[0, 1]);
        let f = forced(&[(0, "Milo")]);

        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(&clusters, &NOBODY, &[])
                .with_forced(&f)
                .with_one_remote_speaker(Some("Grace")),
        );

        assert_eq!(
            map[&0],
            Attribution::Assigned {
                name: "Grace".to_string()
            }
        );
        assert_eq!(
            map[&1],
            Attribution::Assigned {
                name: "Grace".to_string()
            }
        );
    }

    // --- the tentative band ---------------------------------------------------------------

    /// The three facts that keep a guess a question rather than an answer: it reads marked,
    /// it carries its real similarity, and it is not named.
    #[test]
    fn a_tentative_guess_reads_marked_carries_its_similarity_and_is_not_named() {
        let guess = Attribution::Tentative {
            name: "Ivan".to_string(),
            similarity: 0.59,
        };

        assert_eq!(guess.label(), "Ivan?");
        assert_eq!(guess.confidence(), Some(0.59));
        assert!(!guess.is_named());
    }

    /// The whole stack in one voice: forced beats assigned, which beats identified, which
    /// beats the tentative guess, which beats the number -- each tier visible the moment the
    /// one above it steps aside.
    #[test]
    fn the_precedence_ladder_runs_forced_assigned_identified_tentative_unknown() {
        let clusters = clusters(&[0]);
        let id = identified(&[(0, "Alice", 0.83)]);
        let rows = vec![assignment(0, "Alex")];
        let tent = identified(&[(0, "Ivan", 0.59)]);
        let f = forced(&[(0, "Grace")]);
        let numbers = unknown(&[(0, "Unknown 1")]);

        let naming = Naming::new(&clusters, &id, &rows)
            .with_tentative(&tent)
            .with_forced(&f);
        let map = attributions(&numbers, naming);
        assert_eq!(map[&0].label(), "Grace");

        let naming = Naming::new(&clusters, &id, &rows).with_tentative(&tent);
        let map = attributions(&numbers, naming);
        assert_eq!(map[&0].label(), "Alex");

        let naming = Naming::new(&clusters, &id, &[]).with_tentative(&tent);
        let map = attributions(&numbers, naming);
        assert_eq!(
            map[&0],
            Attribution::Identified {
                name: "Alice".to_string(),
                similarity: 0.83,
            }
        );

        let naming = Naming::new(&clusters, &NOBODY, &[]).with_tentative(&tent);
        let map = attributions(&numbers, naming);
        assert_eq!(
            map[&0],
            Attribution::Tentative {
                name: "Ivan".to_string(),
                similarity: 0.59,
            }
        );

        let map = attributions(&numbers, Naming::new(&clusters, &NOBODY, &[]));
        assert_eq!(map[&0], Attribution::Unknown("Unknown 1".to_string()));
    }

    /// A hand-named holder on an overlapping voice beats a guess, the way it beats an
    /// identification: the human claim outranks the vector distance -- a looser one at that
    /// -- and the fragment falls to its number rather than keeping a mark the segmentation
    /// evidence contradicts.
    #[test]
    fn a_tentative_guess_an_overlapping_hand_named_holder_contradicts_falls_back_to_the_number() {
        let clusters = vec![cluster(0, vec![1]), cluster(1, vec![0])];
        let tent = identified(&[(1, "Alex", 0.59)]);

        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(&clusters, &NOBODY, &[assignment(0, "Alex")]).with_tentative(&tent),
        );

        assert_eq!(
            map[&0],
            Attribution::Assigned {
                name: "Alex".to_string()
            }
        );
        assert_eq!(map[&1], Attribution::Unknown("Unknown 2".to_string()));
    }

    /// The refusal is one-sided like the demotion's: a holder of the guessed name on a voice
    /// nothing says overlaps earns nothing against the guess, because clustering splitting one
    /// person in two is the ordinary reason a name appears on two non-overlapping voices.
    #[test]
    fn a_tentative_guess_stands_when_no_holder_of_the_name_overlaps_it() {
        let clusters = clusters(&[0, 1]);
        let tent = identified(&[(1, "Alex", 0.59)]);

        let map = attributions(
            &unknown(&[(0, "Unknown 1"), (1, "Unknown 2")]),
            Naming::new(&clusters, &NOBODY, &[assignment(0, "Alex")]).with_tentative(&tent),
        );

        assert_eq!(
            map[&1],
            Attribution::Tentative {
                name: "Alex".to_string(),
                similarity: 0.59,
            }
        );
    }

    /// A standing denial suppresses the guess it names, whatever the similarity says: the user
    /// already answered this voice, and the answer was no.
    #[test]
    fn a_denied_name_suppresses_the_tentative_guess_even_at_high_similarity() {
        let clusters = clusters(&[0]);
        let tent = identified(&[(0, "Ivan", 0.97)]);
        let denial = DeniedName {
            cluster: 0,
            name: "Ivan".to_string(),
            embedding: cluster(0, Vec::new()).embedding,
        };
        let denied = resolve_denials(&clusters, &[denial]);

        let map = attributions(
            &unknown(&[(0, "Unknown 1")]),
            Naming::new(&clusters, &NOBODY, &[])
                .with_tentative(&tent)
                .with_denials(&denied),
        );

        assert_eq!(map[&0], Attribution::Unknown("Unknown 1".to_string()));
    }

    /// Resolution is the removal's own rule: a unique bit-exact match resolves; a stale one
    /// and a twin-centroid one drop, so a denial suppresses nothing rather than the wrong
    /// voice.
    #[test]
    fn denials_resolve_unique_bit_exact_matches_only() {
        let good = cluster(0, Vec::new());
        let stale = DeniedName {
            cluster: 0,
            name: "Stale".to_string(),
            embedding: vec![9.0, 0.5],
        };
        let hit = DeniedName {
            cluster: 0,
            name: "Ivan".to_string(),
            embedding: good.embedding.clone(),
        };

        // Unique match: resolved.
        let mut expected = BTreeSet::new();
        expected.insert((0u32, "Ivan".to_string()));
        assert_eq!(
            resolve_denials(std::slice::from_ref(&good), std::slice::from_ref(&hit)),
            expected
        );

        // No match: dropped.
        assert!(resolve_denials(std::slice::from_ref(&good), &[stale]).is_empty());

        // Two clusters holding the same centroid: ambiguous, so dropped.
        let twin = SpeakerCluster {
            embedding: good.embedding.clone(),
            ..cluster(1, Vec::new())
        };
        assert!(resolve_denials(&[good, twin], &[hit]).is_empty());
    }
}
