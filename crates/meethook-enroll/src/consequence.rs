//! What answering with a name *would* do, worked out before anything is written.
//!
//! Every outcome an answer can have is decided by rules that live somewhere else -- the
//! reference floor, the cap on how many recordings one person keeps, the correction that drops
//! a reference built from a voice the user has just renamed, and the heard-at-once veto that
//! refuses to put one name on two voices the segmenter proved are different people. What none
//! of those rules can say on their own is *which* of the outcomes a particular answer lands on,
//! because two files carry a name -- `speakers.json` and one session's `speaker_names.json` --
//! and both feed the same labelling. The only way to know is to build the state the answer
//! would leave and label the session through it.
//!
//! [`crate::enroll_session`] has always done exactly that, on copies, before committing
//! anything: it is the pre-flight that lets an answer costing somebody else their name be
//! refused instead of written and then unpicked. This module is that pre-flight, addressable.
//!
//! # A dry run, not a description of one
//!
//! [`Preview::of`] clones the database, applies the answer to the clone, and labels the session
//! twice through [`crate::effective_labels`] -- the same function the transcript is written
//! with. So a [`Consequence`] is not a prediction that could disagree with the write: it *is*
//! the write, held rather than performed, and committing it is `enroll_session` taking the
//! clones it was handed. Nothing here re-derives a threshold. `REFERENCE_FLOOR_SECONDS` is
//! read, `MAX_REFERENCES_PER_SPEAKER` stays inside
//! [`EnrolledSpeakers::store_reference`](meethook_session::EnrolledSpeakers::store_reference),
//! `IDENTIFY_DISTANCE` stays inside
//! [`identify_clusters`](meethook_transcribe::identify_clusters), and the veto is reached the
//! only way this crate has ever reached it: through the labelling it produces.
//!
//! # What it costs
//!
//! One clone of `speakers.json`, one of this session's `speaker_names.json`, and two full
//! labellings of the session -- which means two passes of `identify_clusters` over every cluster
//! in it. That is what an answer costs today, and it is fine per *answer*. It is not fine per
//! keystroke: an interface offering names as the user types must resolve text with
//! [`crate::resolve()`], which is built for that, and preview only the one candidate it has
//! highlighted.
//!
//! [`Preview::group`] prices the aggregate the same way: N times one clone pair and two
//! labellings, folded forward over clones in queue order, which is acceptable once per distinct
//! candidate name and never per keystroke for the same reason.
//!
//! Nothing on the `--name` path reaches [`Preview::of`] at all, so the non-interactive command
//! pays for the preview it never asks for exactly nothing.

use std::borrow::Cow;
use std::collections::BTreeMap;

use meethook_session::{Displaced, EnrolledSpeakers, SpeakerCluster, SpeakerNames, Stored};
use meethook_transcribe::{Attribution, heard_at_once};
use serde::Serialize;

use crate::{Enrolment, REFERENCE_FLOOR_SECONDS, effective_labels};

/// One voice, and the question "what would this name do to it?".
///
/// Constructed per voice and borrowing the session it is about -- the clusters, the "Unknown N"
/// numbering, the database, and this session's hand-given names -- so it cannot outlive the
/// question. Answering is [`Preview::of`], which may be called as many times as there are
/// candidates to consider and writes nothing on any of them.
pub struct Preview<'a> {
    clusters: &'a [SpeakerCluster],
    unknown: &'a BTreeMap<u32, String>,
    speakers: &'a EnrolledSpeakers,
    assigned: &'a SpeakerNames,
    cluster: &'a SpeakerCluster,
    enrolment: Enrolment,
    /// The session's one-remote-speaker assertion, if there is one -- or `None`, which labels
    /// exactly as before. Carried rather than looked up because the dry run and the write must
    /// honour the same labelling: a preview that did not see the assertion could show a veto
    /// the commit would override, and disagree with the transcript it predicts.
    one_remote_speaker: Option<&'a str>,
    /// The clusters this run has already committed, in commit order -- which is the walk
    /// order, since a cluster is committed as the walk reaches it. Consumed only by
    /// [`Preview::one_speaker`], which must promise the commits the run will still make rather
    /// than the ones it has made; [`Preview::of`] ignores it.
    committed: &'a [&'a SpeakerCluster],
    /// The group's declared members, keyed by cluster id to the name the group commits them
    /// under -- or `None`, which labels exactly as before. Set by [`Preview::with_forced`]
    /// and threaded into every labelling the dry run does, so a group member previewed against
    /// the running state honours the same forced tier the write applies: a preview that did not
    /// see it could show a veto the commit overrides, and disagree with the transcript it
    /// predicts.
    forced: Option<&'a BTreeMap<u32, String>>,
}

impl<'a> Preview<'a> {
    /// The eight things a dry run of one answer needs, and nothing else.
    ///
    /// `speakers` and `assigned` are the two files a name can land in, borrowed rather than
    /// cloned here: the clone happens per [`Preview::of`] call, since a preview that held its
    /// own copy would go stale the moment an answer was committed.
    ///
    /// `one_remote_speaker` is the session-level fact [`effective_labels`] applies above all
    /// the rest of the rule; passing it is what keeps a preview built during an assertion run
    /// from predicting a refusal the assertion overrides.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        clusters: &'a [SpeakerCluster],
        unknown: &'a BTreeMap<u32, String>,
        speakers: &'a EnrolledSpeakers,
        assigned: &'a SpeakerNames,
        cluster: &'a SpeakerCluster,
        enrolment: Enrolment,
        one_remote_speaker: Option<&'a str>,
        committed: &'a [&'a SpeakerCluster],
    ) -> Preview<'a> {
        Preview {
            clusters,
            unknown,
            speakers,
            assigned,
            cluster,
            enrolment,
            one_remote_speaker,
            committed,
            forced: None,
        }
    }

    /// A copy of this preview whose dry run honours the group's declared members: the ids in
    /// `forced` read their name unconditionally and are exempt from the heard-at-once and
    /// argmax exclusions for the duration of the labelling -- see
    /// [`meethook_transcribe::Naming::with_forced`] for the rank and the reason. `None` back
    /// to the ordinary rule.
    pub(crate) fn with_forced(mut self, forced: Option<&'a BTreeMap<u32, String>>) -> Preview<'a> {
        self.forced = forced;
        self
    }

    /// What answering this voice with `name` would do, without writing anything.
    ///
    /// `None` for a name that is not one -- empty, or whitespace only. That is a skip rather
    /// than an answer, and the nothing it writes is exactly the nothing this returns; the
    /// trimming is the same normalisation [`crate::GivenName`] applies on the way in, so a
    /// typed answer and one supplied up front are decided identically.
    ///
    /// One call is one answer's worth of work -- a database clone and two full labellings of
    /// the session, as the module doc spells out -- so this belongs on a name the user has
    /// settled on, not on every keystroke.
    ///
    /// The result reflects the database **as it stands now**, not as it stood when the run
    /// began: a name given earlier in this same run is already in it, which is the useful
    /// behaviour rather than an accident when clustering has split one person in two.
    pub fn of(&self, name: &str) -> Option<Consequence> {
        let name = name.trim();
        if name.is_empty() {
            return None;
        }
        Some(self.commit_to(
            name,
            self.cluster,
            self.speakers.clone(),
            self.assigned.clone(),
            self.forced,
        ))
    }

    /// The mutation sequence one answer applies to a candidate state: the correction, the two
    /// labellings, the addition, the stale check, and the refusal -- the whole body of
    /// [`Preview::of`], factored out so the group fold applies the same sequence to the same
    /// kind of state rather than re-deriving it.
    ///
    /// `name` must be trimmed and non-empty -- the normalisation [`Preview::of`] applies -- and
    /// `candidate` and `candidate_assigned` are the copies the caller cloned, one per question,
    /// since a preview that held its own copy would go stale the moment an answer was
    /// committed. `forced` is the group's declared members, passed by the fold and `None`
    /// everywhere else; see [`meethook_transcribe::Naming::with_forced`] for what the tier does
    /// to the rule.
    ///
    /// One producer of the sequence rather than two is the module invariant above: a preview
    /// and a write that disagreed would be two implementations of one rule, and the symptom of
    /// those disagreeing is a name on the wrong person's turns.
    fn commit_to(
        &self,
        name: &str,
        cluster: &SpeakerCluster,
        mut candidate: EnrolledSpeakers,
        mut candidate_assigned: SpeakerNames,
        forced: Option<&BTreeMap<u32, String>>,
    ) -> Consequence {
        // Naming a voice and storing a reference built from it are two different acts, and this
        // is where they come apart. Below the floor the name is recorded against the session and
        // `speakers.json` is not touched at all -- see `REFERENCE_FLOOR_SECONDS` for what a
        // reference built from two seconds of speech does to every future meeting.
        let session_only =
            cluster.speech_seconds < REFERENCE_FLOOR_SECONDS && self.enrolment != Enrolment::Always;

        // The correction, on the above-floor path only: a reference identical to this cluster
        // was built from this voice, and the user has just told us this voice is somebody else,
        // so it is a stored claim about a person it is not of and it competes as an argmax in
        // every future meeting -- winning whenever its name sorts first
        // (`identify::best_match`'s tie-break).
        let displaced = if session_only {
            Vec::new()
        } else {
            candidate.forget_reference(&cluster.embedding, name)
        };

        // What every voice reads once the correction alone has been applied. The baseline the
        // refusal is measured against, and the reason it is two labellings rather than one: a
        // name lost *here* is the correction's documented consequence -- the user has just said
        // that reference was of somebody else -- and refusing it would undo the guarantee the
        // correction exists to keep.
        let corrected = effective_labels(
            self.clusters,
            self.unknown,
            &candidate,
            &candidate_assigned.names,
            self.one_remote_speaker,
            forced,
            None,
            &candidate_assigned.denied,
        );

        // The addition. `None` on the below-floor path, where no reference is stored at all.
        let stored = if session_only {
            candidate_assigned.assign(cluster.id, name, cluster.embedding.clone());
            None
        } else {
            let stored =
                candidate.store_reference(name, cluster.embedding.clone(), cluster.speech_seconds);
            if matches!(stored, Stored::AtCapacity { .. }) {
                // At the cap with nothing shorter than this recording to displace, so it is not
                // stored and the answer falls back to the session-only path rather than being
                // lost: the transcript still reads the right person, and nothing already stored
                // is dropped for a recording that is no better than it.
                candidate_assigned.assign(cluster.id, name, cluster.embedding.clone());
            } else if forced.is_some() {
                // The group's half of one voice, one record: the declaration stands in *both*
                // stores, and both say the same thing -- the reference says the name because
                // the user declared it, and the row says the name for exactly the same reason,
                // so there is nothing for them to disagree about. Keeping the row is what makes
                // the decision durable against the heard-at-once exclusion on every later run:
                // two standing rows of one name on overlapping voices are co-declaration, and
                // both stand -- demoting one afterwards would undo the declaration and re-prompt
                // forever. A plain naming still forgets its row below; only a declaration made
                // with the overlap reported keeps it.
                candidate_assigned.assign(cluster.id, name, cluster.embedding.clone());
            } else if stored == Stored::AlreadyHeld && !cluster.heard_at_once_with.is_empty() {
                // The stranded declaration stands up in both stores. `AlreadyHeld` means the
                // database already holds a bit-identical reference built from this exact
                // fragment under this exact name -- a commit interrupted between its
                // `speakers.json` write and its `speaker_names.json` write left exactly that:
                // the declaration survived in the first store and never reached the second.
                // Standing it up here records the existing declaration in the second store,
                // which is what the heard-at-once exclusion needs to stand on: two standing
                // rows of one name on overlapping voices are co-declaration, and both stand.
                // Without the row the exclusion demotes the voice on every later pass, so the
                // transcript reads `Unknown N` forever while the database says the name. The
                // overlap condition is the boundary: on a solo voice the confirmation is pure
                // redundancy, where the row risks the stale-pin disagreement the forget below
                // guards against.
                candidate_assigned.assign(cluster.id, name, cluster.embedding.clone());
            } else {
                // One voice, one record. A voice named for this session only and then enrolled
                // properly -- the same fragment reached again with `--force-reference`, or a
                // later clustering that gave it enough speech -- must stop also being an
                // assignment, or the two could be made to disagree about who it is.
                candidate_assigned.forget(cluster.id);
            }
            Some(stored)
        };

        // The answer is about `self.cluster`, whose own row is what this answer would create
        // rather than a standing declaration: pending keeps it out of the assignment award's
        // co-declaration pass, which is what leaves the vetoed demotion for the refusal below
        // to fire off.
        let after = effective_labels(
            self.clusters,
            self.unknown,
            &candidate,
            &candidate_assigned.names,
            self.one_remote_speaker,
            forced,
            Some(self.cluster.id),
            &candidate_assigned.denied,
        );

        // A legacy reference that *is* this exact fragment, still standing under somebody
        // else's name. Only the below-floor path can leave one -- above the floor
        // `forget_reference` has just dropped every such row -- so this is empty by
        // construction on every other path rather than by a special case here.
        let stale: Vec<String> = candidate
            .speakers
            .iter()
            .filter(|s| s.name != name && s.embedding == cluster.embedding)
            .map(|s| s.name.clone())
            .collect();

        Consequence {
            refused: refusal_of(cluster.id, name, self.unknown, &corrected, &after),
            stored,
            displaced,
            stale,
            demoted: None,
            speakers: candidate,
            assigned: candidate_assigned,
        }
    }

    /// What refusing this voice's standing guess would do, without writing anything.
    ///
    /// The complement of [`Preview::of`] on a guessed fragment. Where naming adds a claim --
    /// to the database or to this session's rows -- denial removes one: the tentative "Name?"
    /// the band wrote into the transcript is refused, and the row that suppresses it from here
    /// on goes through exactly the candidate state the commit will apply, so a preview and a
    /// write cannot disagree about what the fragment reads afterwards.
    ///
    /// There is no refusal path for a denial: it takes nothing off any other voice -- the
    /// label it removes is this cluster's own -- so there is nothing to refuse and no `None`
    /// to return. And `name` is the guess as displayed, non-empty by construction, so the
    /// trimming [`Preview::of`] applies is the only normalisation this needs too.
    pub fn deny_to(&self, name: &str) -> Consequence {
        let name = name.trim();
        let mut candidate_assigned = self.assigned.clone();
        candidate_assigned.deny(self.cluster.id, name, &self.cluster.embedding);

        // What every voice read before the answer, and once the denial alone has been applied:
        // the demotion is measured between these two, the way a refusal is measured against
        // its baseline in [`Self::commit_to`]. Two labellings rather than one is the same cost
        // model the module doc spells out, and the same reason: the baseline is what says what
        // moved.
        let before = effective_labels(
            self.clusters,
            self.unknown,
            self.speakers,
            &self.assigned.names,
            self.one_remote_speaker,
            self.forced,
            None,
            &self.assigned.denied,
        );
        let after = effective_labels(
            self.clusters,
            self.unknown,
            self.speakers,
            &candidate_assigned.names,
            self.one_remote_speaker,
            self.forced,
            None,
            &candidate_assigned.denied,
        );

        Consequence {
            refused: None,
            stored: None,
            displaced: Vec::new(),
            stale: Vec::new(),
            demoted: Some(Demotion {
                from: before[&self.cluster.id].label().to_string(),
                to: after[&self.cluster.id].label().to_string(),
            }),
            speakers: self.speakers.clone(),
            assigned: candidate_assigned,
        }
    }

    /// What asserting that the session's speaker track is one person called `name` would do to
    /// the voices this run has not named yet, without writing anything.
    ///
    /// Session-level rather than voice-level on purpose: the assertion outranks the queue and
    /// its gates alike, so it is about every cluster in the session at once, and this ignores
    /// the voice the question happens to be about entirely.
    ///
    /// `None` for a name that is not one -- empty, or whitespace only -- the same
    /// normalisation [`Preview::of`] applies, so a stray-keystroke press previews nothing and
    /// answers nothing.
    ///
    /// Cheap enough to ask per highlighted candidate: no clone, no labelling -- one sort and
    /// an overlap check per uncommitted pair. Deliberately no reference prediction: predicting
    /// what the database would hold means simulating the store-and-cap trajectory across the
    /// walk, which is the store's logic reached a second time, and the existing summary note
    /// reports the references post-hoc in the same log pane seconds later.
    pub fn one_speaker(&self, name: &str) -> Option<Assertion> {
        if name.trim().is_empty() {
            return None;
        }

        // The summary line reports how many voices the assertion names, which is exactly the
        // commits the run makes under it: every cluster not already committed, below the
        // prompt floor included, each through its own commit. A mid-run assertion must not
        // promise to re-name the voices whose rows already stand.
        let voices = self.clusters.len() - self.committed.len();

        // The override report, simulated rather than counted statically: the run walks the
        // clusters in first-appearance order starting from whatever it has already committed,
        // and reports a veto for a cluster iff it was heard at once with one already holding
        // the name. Walking the same order from the same starting set, with the same
        // predicate, is the counter itself -- a static "has an overlap partner" count would
        // drift from it in asymmetric orders and after mid-run naming.
        let mut order: Vec<&SpeakerCluster> = self.clusters.iter().collect();
        order.sort_by(|a, b| {
            a.first_spoke_seconds
                .total_cmp(&b.first_spoke_seconds)
                .then(a.id.cmp(&b.id))
        });
        let mut seen: Vec<&SpeakerCluster> = self.committed.to_vec();
        let mut vetoes_overridden = 0;
        for cluster in order {
            if self.committed.iter().any(|done| done.id == cluster.id) {
                continue;
            }
            if seen.iter().any(|seen| heard_at_once(cluster, seen)) {
                vetoes_overridden += 1;
            }
            seen.push(cluster);
        }

        Some(Assertion {
            voices,
            vetoes_overridden,
        })
    }

    /// What naming a chosen group of voices -- the stable "Unknown N" handles the interface
    /// shows in its queue pane -- with one name would do to the session, without writing
    /// anything.
    ///
    /// The aggregate dry run behind a group commit: each member is applied through the same
    /// `commit_to` core [`Preview::of`] uses, in queue order, over clones
    /// that grow as members land. That is what makes this the sequential application of the
    /// members' individual previews rather than a re-derivation of them, and it is why the
    /// cost is N times one clone pair and two labellings -- acceptable once per distinct
    /// candidate name, never per keystroke, for the reason the module doc gives.
    ///
    /// `None` for a name that is not one -- empty, or whitespace only, the same normalisation
    /// [`Preview::of`] applies -- and for any handle that does not resolve to a cluster of
    /// this session: a group that cannot say who its members are goes unanswered rather than
    /// partially answered, which is the blank-name precedent reached with a different input.
    /// Duplicate handles dedupe to first appearance; the walk itself is in queue order, the
    /// order the commit walks in, so a preview and a write cannot disagree about sequence.
    ///
    /// The group carries veto authority at two or more resolved members: a member heard at
    /// once with a holder of the name is named anyway, and counted in
    /// [`GroupConsequence::vetoes_overridden`] the way the commit counts it. A one-member
    /// group has none -- no forcing at all, exactly today's plain naming of that member --
    /// which is the threshold the commit enforces too, so the two cannot see the group
    /// differently.
    pub fn group(&self, name: &str, members: &[&str]) -> Option<GroupConsequence> {
        let name = name.trim();
        if name.is_empty() || members.is_empty() {
            return None;
        }
        let resolved = self.resolve_members(members)?;
        // Veto authority iff the group names two or more voices: one member is a plain
        // naming, and the veto refuses it exactly as today.
        Some(self.fold(name, &resolved[..], resolved.len() >= 2))
    }

    /// What naming a library-formed bundle of below-floor fragments with one name would do:
    /// [`group`](Self::group) without its veto authority.
    ///
    /// The authority is the staged group's act -- the user chose those rows as one person, and
    /// two or more of them may override the heard-at-once veto on that claim. A bundle the
    /// bundling proposed is not that act, so a member heard at once with somebody already
    /// holding the name is refused here rather than overridden, and the rest of the fold still
    /// applies. The write path makes the same choice, which is why both run over this one
    /// fold rather than two computations that could drift.
    pub fn fragments(&self, name: &str, members: &[&str]) -> Option<GroupConsequence> {
        let name = name.trim();
        if name.is_empty() || members.is_empty() {
            return None;
        }
        let resolved = self.resolve_members(members)?;
        Some(self.fold(name, &resolved[..], false))
    }

    /// Handles to clusters, deduplicated and re-sorted into queue order.
    ///
    /// Resolution and deduplication against the session's own numbering: the handles are the
    /// values of `unknown`, built over every cluster in the session, so quiet members below
    /// the offer floor resolve alike. First-appearance order keeps the deduplication
    /// deterministic; the fold re-sorts into queue order regardless.
    fn resolve_members(&self, members: &[&str]) -> Option<Vec<&SpeakerCluster>> {
        let mut ids: Vec<u32> = Vec::new();
        for handle in members {
            let id = self
                .unknown
                .iter()
                .find(|(_, label)| label == handle)
                .map(|(&id, _)| id)?;
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        let mut resolved: Vec<&SpeakerCluster> = ids
            .iter()
            .filter_map(|id| self.clusters.iter().find(|c| c.id == *id))
            .collect();
        resolved.sort_by(|a, b| {
            a.first_spoke_seconds
                .total_cmp(&b.first_spoke_seconds)
                .then(a.id.cmp(&b.id))
        });
        Some(resolved)
    }

    /// The fold itself: one clone pair per member, applied forward over the running state,
    /// with the forced set grown as the walk commits, under `authority` only.
    fn fold(&self, name: &str, resolved: &[&SpeakerCluster], authority: bool) -> GroupConsequence {
        // The fold: one clone pair per member, applied forward over the running state. A
        // refused member leaves the running state untouched, so the members after it are
        // previewed against the state the run will actually reach.
        let mut running_speakers = self.speakers.clone();
        let mut running_assigned = self.assigned.clone();
        let mut committed: Vec<&SpeakerCluster> = Vec::new();
        let mut result = GroupConsequence {
            name: name.to_string(),
            applied: Vec::new(),
            refused: Vec::new(),
            vetoes_overridden: 0,
            references_after: 0,
            displaced: Vec::new(),
            stale: Vec::new(),
        };

        for member in resolved.iter().copied() {
            // The members committed so far, by id and name: the previous forced set for the
            // veto count, and minus this member the seed of the current one.
            let mut previous_forced: BTreeMap<u32, String> = committed
                .iter()
                .map(|done| (done.id, name.to_string()))
                .collect();

            // The veto count, measured before the member lands: the holders of the name in
            // the running pre-state labelling under the previous forced set, excluding the
            // member itself, filtered to the ones segmentation heard at once with it. One
            // overridden veto per member however many holders it overlaps -- the run's own
            // counter, which the summary line reports.
            if authority {
                let pre = effective_labels(
                    self.clusters,
                    self.unknown,
                    &running_speakers,
                    &running_assigned.names,
                    self.one_remote_speaker,
                    Some(&previous_forced),
                    None,
                    &running_assigned.denied,
                );
                let overlapped = pre.iter().any(|(&id, label)| {
                    id != member.id
                        && label.label() == name
                        && self
                            .clusters
                            .iter()
                            .any(|c| c.id == id && heard_at_once(member, c))
                });
                if overlapped {
                    result.vetoes_overridden += 1;
                }
            }

            // The member's individual consequence through the shared core, with the growing
            // forced set -- the declared members committed so far plus this one, only while
            // the group has authority. Under forcing a `Vetoed` refusal cannot arise, because
            // the member always holds the name; a `Taken` refusal can, and it is honoured
            // here the way the commit honours it: refused, state left unchanged, the walk
            // carries on.
            let forced_now = if authority {
                previous_forced.insert(member.id, name.to_string());
                Some(&previous_forced)
            } else {
                None
            };
            let consequence = self.commit_to(
                name,
                member,
                running_speakers.clone(),
                running_assigned.clone(),
                forced_now,
            );

            match consequence.refused {
                Some(refusal) => {
                    result
                        .refused
                        .push((handle(member.id, self.unknown), refusal));
                }
                None => {
                    result.applied.push(handle(member.id, self.unknown));
                    // Merged by name rather than concatenated: one person can lose references
                    // to two members, and concatenation would report them twice with two
                    // different remainings. Last `remaining` wins, like the run's own writes.
                    for displaced in consequence.displaced {
                        if let Some(existing) = result
                            .displaced
                            .iter_mut()
                            .find(|d| d.name == displaced.name)
                        {
                            *existing = displaced;
                        } else {
                            result.displaced.push(displaced);
                        }
                    }
                    for stale in consequence.stale {
                        if !result.stale.contains(&stale) {
                            result.stale.push(stale);
                        }
                    }
                    // Applied by taking the copies the dry run produced, exactly as the
                    // commit takes them out of its own consequence.
                    running_speakers = consequence.speakers;
                    running_assigned = consequence.assigned;
                    committed.push(member);
                }
            }
        }

        // The group's total reference count after applying, read off the final database
        // rather than re-derived from the members: the cap does the bounding, so the count
        // it holds is the answer.
        result.references_after = running_speakers.references(name);
        result
    }
}

/// What naming a chosen group of voices with one name would do to the session, as the
/// aggregate the commit reports.
///
/// Read [`GroupConsequence::refused`] first: a refused member is one the group could not
/// name -- the rest still apply, which is the difference between a group and a single answer,
/// where a refusal writes nothing at all.
///
/// Every field is what the run's own report says once the group has walked -- the members it
/// named, the vetoes it overrode, the references the database now holds -- computed the way
/// the run computes them rather than re-derived, which is what keeps a preview and a write
/// from drifting apart.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupConsequence {
    /// The name, trimmed the same way [`Preview::of`] trims one.
    pub name: String,

    /// The members that would be committed, by their "Unknown N", in queue order.
    pub applied: Vec<String>,

    /// The members whose naming would be refused, with the refusal, in queue order.
    pub refused: Vec<(String, Refusal)>,

    /// How many heard-at-once vetoes the group would override on the way, counting a member
    /// once however many holders it overlaps -- the run's own counter.
    pub vetoes_overridden: usize,

    /// The group's total reference count after applying, off the final database.
    pub references_after: usize,

    /// The people who would lose a reference to the group's corrections, merged by name.
    pub displaced: Vec<Displaced>,

    /// Names that would still hold a reference built from an applied member's exact voice
    /// afterwards, unioned across the members.
    pub stale: Vec<String>,
}

impl GroupConsequence {
    /// What committing the group would do, as the sentences an interface shows before the
    /// group is committed -- or after it, for the ones that print outcomes rather than
    /// preview them.
    ///
    /// Stated here once beside [`Consequence::would_do`] so no caller restates it: the frame's
    /// "would" pane and whatever prints the group's report are readers of the same fact, and a
    /// second copy of this mapping is exactly what this module's doc forbids. The lines run
    /// displaced first, then stale, then the refused members -- the load-bearing counts before
    /// the exceptions, so a pane that clips still shows what the commit did to people.
    ///
    /// The stale line drops the single voice's "of this voice": the union carries no
    /// per-member attribution, and naming one member would be a claim the aggregate does not
    /// make.
    pub fn would_do(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for Displaced { name, remaining } in &self.displaced {
            lines.push(format!(
                "takes a recording off {name}, leaving them {remaining}"
            ));
        }
        for name in &self.stale {
            lines.push(format!("leaves a recording standing under {name}"));
        }
        for (handle, refusal) in &self.refused {
            lines.push(format!("{handle}: {}", refusal.sentence()));
        }
        lines
    }
}

/// What asserting one remote speaker would do to the session, as the two numbers the commit
/// reports.
///
/// Fully public and owned rather than borrowed, for the reason [`Refusal`] made the frame's
/// cost type testable across the seam: the state machine carries this into its view, and a
/// test there must be able to construct it.
///
/// Both fields are what the run's own report says once the assertion has run -- the voices it
/// named and the vetoes it overrode -- computed the way the run computes them rather than
/// re-derived, which is what keeps a preview and a write from drifting apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Assertion {
    /// How many voices the assertion would name: every cluster this run has not committed,
    /// below the prompt floor included.
    pub voices: usize,
    /// How many heard-at-once vetoes the assertion would override on the way, each of which
    /// the run reports as overridden rather than refused.
    pub vetoes_overridden: usize,
}

/// Everything answering one voice with one name would do.
///
/// Read [`Consequence::refused`] first: when it is `Some` nothing would be written at all, and
/// the other fields describe the state that was rejected.
///
/// # The five outcomes, which are `stored` plus [`Consequence::session_only`]
///
/// Stated here once, so no caller has to re-derive which of them it is holding:
///
/// - `Some(`[`Stored::Enrolled`]`)` -- somebody who was not in the database now is.
/// - `Some(`[`Stored::Added`]`)` -- another recording of somebody already here.
/// - `Some(`[`Stored::Replaced`]`)` -- the shortest recording they held is dropped for this
///   one, **and that dropped row may have been the only thing naming a voice in another
///   session**.
/// - `Some(`[`Stored::AtCapacity`]`)` -- they already hold as many recordings as meethook keeps
///   and none is shorter than this one, so nothing is stored and the name lands in this session
///   only.
/// - `Some(`[`Stored::AlreadyHeld`]`)` -- bit-identical to a recording they have, so the file is
///   not rewritten and the transcript still reads their name.
/// - `None` -- under `REFERENCE_FLOOR_SECONDS`, so likewise this session only.
///
/// There is deliberately no second enum restating that mapping. A parallel outcome type is
/// precisely the duplicated set of thresholds this module exists to avoid.
/// A label moving back rather than forward: the shape of a denial's effect on the transcript.
///
/// Owned like [`Displaced`], for the same reason: the frame carries it into its view and a
/// test must be able to construct it across the seam.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Demotion {
    /// What the voice read before the answer -- the marked guess, e.g. "Ivan?".
    pub from: String,
    /// What it reads after -- the "Unknown N" its turns were written with, e.g. "Unknown 3".
    pub to: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Consequence {
    /// Why this answer would not be honoured, if it would not be.
    ///
    /// `Some` means nothing is written: not the reference, not the assignment, not the
    /// transcript. See [`Refusal`] for the three ways an answer can take a name off a voice the
    /// user was not asked about, and why one check covers all of them.
    pub refused: Option<Refusal>,

    /// What `speakers.json` would record, or `None` for an answer that goes no further than
    /// this session. The five outcomes above.
    pub stored: Option<Stored>,

    /// The *people* who would lose a reference to this answer's correction.
    ///
    /// People rather than rows, and with a count left, because "Nate no longer has a reference"
    /// is a lie when Nate has three and lost one -- which under a reference set is the usual
    /// case rather than the rare one.
    pub displaced: Vec<Displaced>,

    /// Names that would still hold a reference built from this exact voice afterwards.
    ///
    /// Reachable only below the reference floor, where nothing in `speakers.json` is touched
    /// and a legacy row built from this fragment therefore survives to go on competing as an
    /// argmax under the wrong name. Empty on every other path, because the correction has
    /// already dropped every such row.
    pub stale: Vec<String>,

    /// One label moving back rather than forward, or `None` when the answer moves labels the
    /// usual way.
    ///
    /// `Some` only for a denial, where the demotion is the whole of the write: a denial stores
    /// nothing and displaces nothing, so what it does to the transcript is all there is to
    /// preview and all there is to report. `None` on every naming path, where the transcript
    /// change rides the new label the user already sees.
    pub demoted: Option<Demotion>,

    /// The database this answer would leave. Crate-visible on purpose: an [`crate::Interviewer`]
    /// able to read it could write it, behind the back of the one loop that decides what lands
    /// on disk.
    pub(crate) speakers: EnrolledSpeakers,

    /// This session's hand-given names as this answer would leave them. Crate-visible for the
    /// same reason as [`Consequence::speakers`].
    pub(crate) assigned: SpeakerNames,
}

impl Consequence {
    /// Whether this answer would name the voice for this session alone, storing nothing that
    /// helps recognise the person in the next meeting.
    ///
    /// Two different sentences reach this: under the reference floor, and at the cap with
    /// nothing shorter to displace. Derived from [`Consequence::stored`] rather than carried
    /// beside it, so the two cannot disagree.
    pub fn session_only(&self) -> bool {
        self.stored.is_none() || matches!(self.stored, Some(Stored::AtCapacity { .. }))
    }

    /// What answering would do, as the sentences an interface shows before the answer is
    /// given -- or after it, for the ones that print outcomes rather than preview them.
    ///
    /// The mapping from [`Consequence::stored`] plus [`session_only`](Self::session_only)
    /// to a sentence, stated here once so no caller restates it: the frame's "would" pane
    /// and the headless dry-run print are two readers of the same fact, and a second copy of
    /// this mapping is exactly what this module's doc forbids.
    pub fn would_do(&self) -> Vec<String> {
        let mut lines = Vec::new();
        match &self.stored {
            Some(Stored::Enrolled) => lines.push("enrols them, from this voice".to_string()),
            Some(Stored::Added { held }) => {
                lines.push(format!("stores another recording of them, {held} in all"));
            }
            Some(Stored::AlreadyHeld) => {
                lines.push("stores nothing new: they already hold this recording".to_string());
            }
            Some(Stored::Replaced {
                held,
                evicted_seconds,
            }) => lines.push(format!(
                "stores this recording in place of their shortest, {}, {held} in all",
                crate::speech(*evicted_seconds)
            )),
            Some(Stored::AtCapacity { held, .. }) => lines.push(format!(
                "stores nothing: they hold {held} recordings and none is shorter than this voice"
            )),
            None => {}
        }
        if self.session_only() {
            lines.push("names this voice in this session only, storing no reference".to_string());
        }
        for Displaced { name, remaining } in &self.displaced {
            lines.push(format!(
                "takes a recording off {name}, leaving them {remaining}"
            ));
        }
        for name in &self.stale {
            lines.push(format!(
                "leaves a recording of this voice standing under {name}"
            ));
        }
        if let Some(Demotion { from, to }) = &self.demoted {
            lines.push(format!("moves {from} back to {to}"));
        }
        lines
    }

    /// What an interface tells somebody who proposed this answer: the one sentence saying why
    /// it will not be honoured, or -- when it will be -- the [`would_do`](Self::would_do)
    /// lines.
    ///
    /// The refusal wins over the would-do lines because a refused answer writes nothing, so
    /// those lines would describe a world that does not come to pass: the frame shows the
    /// refusal alone for the same reason, and the headless dry-run print must not disagree
    /// with it. Stated here rather than at each caller, beside the mapping it guards.
    pub fn outcome_lines(&self) -> Vec<String> {
        if let Some(refusal) = &self.refused {
            return vec![refusal.sentence()];
        }
        self.would_do()
    }
}

/// What honouring an answer would have taken away from a voice the user was not asked about.
///
/// Both variants name that voice by its "Unknown N" rather than by what it currently reads,
/// because that is the one handle which reaches a voice whatever it is called and is exactly
/// what [`crate::VoiceSelector`] accepts -- so a refusal is a line the user can act on rather
/// than only read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum Refusal {
    /// The answered voice would not have ended up with the name at all: the heard-at-once veto
    /// refuses to put one name on two voices segmentation proved are different people, and
    /// `holder` is the voice it left the name on instead.
    ///
    /// `None` is "the answer simply did not take, and nobody else has that name" -- which the
    /// veto makes unreachable in practice, since something has to have won the name for the
    /// answered voice to have lost it. Refused just as firmly, because writing a reference that
    /// then names nobody is not an outcome to accept silently.
    Vetoed { holder: Option<String> },

    /// The answer would have moved a name off another voice: `voice` is the voice, `losing` is
    /// the name it reads now and would not read afterwards.
    Taken { voice: String, losing: String },
}

impl Refusal {
    /// Why this candidate cannot be chosen, as the one sentence an interface shows instead of
    /// a consequence.
    ///
    /// Stated here beside [`Consequence::would_do`] for the same reason: the frame's "cannot"
    /// pane and the headless dry-run print read the same fact through the same words.
    pub fn sentence(&self) -> String {
        match self {
            Refusal::Vetoed {
                holder: Some(voice),
            } => format!(
                "unavailable: {voice} was heard at the same time as this voice and would keep \
                 the name"
            ),
            Refusal::Vetoed { holder: None } => {
                "unavailable: the name would not end up on this voice".to_string()
            }
            Refusal::Taken { voice, losing } => {
                format!("unavailable: {voice} would stop reading {losing}")
            }
        }
    }
}

/// How a voice is named in a refusal line: the "Unknown N" its first appearance earned it.
///
/// Every id reaching here is a key of `unknown`, which is built over every cluster in the
/// session -- so the fallback is unreachable and exists only so this cannot panic on a
/// hand-edited clusters file.
pub(crate) fn handle(id: u32, unknown: &BTreeMap<u32, String>) -> String {
    unknown
        .get(&id)
        .cloned()
        .unwrap_or_else(|| "that voice".to_string())
}

/// Whether honouring an answer would cost some *other* voice its name, and how.
///
/// `corrected` is what the session reads once the correction the answer implies has been
/// applied and nothing else; `after` is what it reads once the whole answer has been. Both are
/// full labellings, produced by the same [`effective_labels`] the transcript is written
/// through, which is the point: a guard reading anything else could disagree with what the
/// transcript will say.
///
/// # Why the check is here rather than inside identification
///
/// Three different paths can take a name off a voice the user never mentioned, and all three
/// resolve into one labelling before anything is written:
///
/// 1. **The heard-at-once veto.** Name two voices the segmenter heard overlapping with one
///    name and the veto must refuse one -- by design. Which one it refuses is decided by
///    similarity then cluster id, so it can be the *earlier* answer that loses.
/// 2. **Theft by argmax.** A reference stored for one person can be nearer to some third voice
///    than that voice's current name's references are, moving a name the user never asked
///    about.
/// 3. **An assignment beating an identification.** A hand-given name always wins over a match
///    on a voice it overlaps, so naming a quiet fragment can drop that name off the voice that
///    had it.
///
/// A check at this level covers all three at once, and cannot be inconsistent with the outcome
/// the way three checks inside the three mechanisms could be. It is also why a preview of the
/// veto is not a fourth reading of `heard_at_once_with`: it is this same labelling, computed
/// one call earlier.
///
/// # What is *not* a cost
///
/// A name lost between the labels shown before the answer and `corrected` is the correction's
/// documented consequence: the user has just said that reference was of somebody else, and it
/// goes with a line of its own. Collapsing the two labellings into one would refuse exactly the
/// corrections the tool exists to accept.
///
/// A voice that *gains* a name is never a cost either -- that is one person's clustering split
/// in two being named by one answer, which is the behaviour the split-voice guard relies on.
fn refusal_of(
    answered: u32,
    name: &str,
    unknown: &BTreeMap<u32, String>,
    corrected: &BTreeMap<u32, Attribution>,
    after: &BTreeMap<u32, Attribution>,
) -> Option<Refusal> {
    if after.get(&answered).map(Attribution::label) != Some(Cow::Borrowed(name)) {
        return Some(Refusal::Vetoed {
            holder: after
                .iter()
                .find(|&(&id, label)| id != answered && label.label() == name)
                .map(|(&id, _)| handle(id, unknown)),
        });
    }
    corrected
        .iter()
        .find(|&(&id, label)| {
            id != answered
                && label.is_named()
                && after.get(&id).map(Attribution::label) != Some(label.label())
        })
        .map(|(&id, label)| Refusal::Taken {
            voice: handle(id, unknown),
            losing: label.label().to_string(),
        })
}

#[cfg(test)]
mod tests {
    use meethook_session::{EnrolledSpeaker, MAX_REFERENCES_PER_SPEAKER, RepresentativeSegment};
    use meethook_session::{SessionId, unknown_labels, unknown_speaker};

    use super::*;
    use crate::tests::{nearly, voice};

    /// A cluster with a given voice and a given amount of speech, which is all any of these
    /// fixtures turns on: no audio is read and no transcript is written.
    fn cluster(id: u32, embedding: Vec<f32>, speech_seconds: f64) -> SpeakerCluster {
        SpeakerCluster {
            id,
            embedding,
            speech_seconds,
            first_spoke_seconds: f64::from(id),
            heard_at_once_with: Vec::new(),
            representatives: vec![RepresentativeSegment {
                start: 0.0,
                end: 1.0,
            }],
        }
    }

    /// A unit vector in `dimensions` dimensions pointing along one axis, for the fixtures that
    /// need more mutually-distant voices than [`voice`] has axes -- ten references under one
    /// name, none of them near the voice being answered.
    fn axis(dimensions: usize, index: usize) -> Vec<f32> {
        let mut embedding = vec![0.0f32; dimensions];
        embedding[index] = 1.0;
        embedding
    }

    fn enrolled(entries: &[(&str, Vec<f32>, Option<f64>)]) -> EnrolledSpeakers {
        EnrolledSpeakers::new(
            entries
                .iter()
                .map(|(name, embedding, clip_seconds)| EnrolledSpeaker {
                    name: name.to_string(),
                    embedding: embedding.clone(),
                    clip_seconds: *clip_seconds,
                })
                .collect(),
        )
    }

    fn no_names() -> SpeakerNames {
        SpeakerNames::new(SessionId::parse("20260809-052600").unwrap(), Vec::new())
    }

    /// The whole fixture in one call: the clusters, the numbering derived from them exactly as
    /// `enroll` derives it, and a preview aimed at the first of them.
    fn preview_of<'a>(
        clusters: &'a [SpeakerCluster],
        unknown: &'a BTreeMap<u32, String>,
        speakers: &'a EnrolledSpeakers,
        assigned: &'a SpeakerNames,
        enrolment: Enrolment,
    ) -> Preview<'a> {
        preview_with_assertion(clusters, unknown, speakers, assigned, enrolment, None)
    }

    /// The whole fixture aimed at `cluster` under a one-remote-speaker assertion, for the tests
    /// that pin what the dry run says when the assertion outranks the veto.
    fn preview_with_assertion<'a>(
        clusters: &'a [SpeakerCluster],
        unknown: &'a BTreeMap<u32, String>,
        speakers: &'a EnrolledSpeakers,
        assigned: &'a SpeakerNames,
        enrolment: Enrolment,
        one_remote_speaker: Option<&'a str>,
    ) -> Preview<'a> {
        Preview::new(
            clusters,
            unknown,
            speakers,
            assigned,
            &clusters[0],
            enrolment,
            one_remote_speaker,
            &[],
        )
    }

    fn numbering(clusters: &[SpeakerCluster]) -> BTreeMap<u32, String> {
        unknown_labels(clusters.iter().map(|c| (c.id, c.first_spoke_seconds)))
    }

    /// Acceptance criterion #2, first outcome: nobody enrolled, so the answer is an enrollment.
    #[test]
    fn naming_a_voice_nobody_is_enrolled_against_enrols_them() {
        let clusters = vec![cluster(0, voice(0), 40.0)];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[]);
        let assigned = no_names();

        let consequence = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        )
        .of("Alice")
        .unwrap();

        assert_eq!(consequence.refused, None);
        assert_eq!(consequence.stored, Some(Stored::Enrolled));
        assert!(!consequence.session_only());
        assert_eq!(consequence.displaced, []);
        assert_eq!(consequence.stale, [] as [String; 0]);
    }

    /// Second outcome: somebody already here, and this is another recording of them.
    #[test]
    fn naming_a_voice_an_enrolled_person_already_has_a_recording_of_adds_one() {
        let clusters = vec![cluster(0, voice(0), 40.0)];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[("Alice", nearly(80.0), Some(30.0))]);
        let assigned = no_names();

        let consequence = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        )
        .of("Alice")
        .unwrap();

        assert_eq!(consequence.stored, Some(Stored::Added { held: 2 }));
        assert!(!consequence.session_only());
    }

    /// Third outcome: at the cap with something shorter to give up, so the trade is made -- and
    /// the length that goes is reported, because it may have been the only thing naming a voice
    /// in another session.
    #[test]
    fn naming_a_voice_at_the_cap_with_a_shorter_recording_held_replaces_the_shortest() {
        let dimensions = MAX_REFERENCES_PER_SPEAKER + 1;
        let clusters = vec![cluster(0, axis(dimensions, 0), 40.0)];
        let unknown = numbering(&clusters);
        let held: Vec<(&str, Vec<f32>, Option<f64>)> = (1..=MAX_REFERENCES_PER_SPEAKER)
            .map(|i| ("Alice", axis(dimensions, i), Some(10.0 + i as f64)))
            .collect();
        let speakers = enrolled(&held);
        let assigned = no_names();

        let consequence = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        )
        .of("Alice")
        .unwrap();

        assert_eq!(
            consequence.stored,
            Some(Stored::Replaced {
                held: MAX_REFERENCES_PER_SPEAKER,
                evicted_seconds: 11.0,
            })
        );
        assert!(
            !consequence.session_only(),
            "a replacement is stored, so it is not a session-only name"
        );
    }

    /// Fourth outcome: at the cap with nothing shorter, so nothing is stored -- and the name
    /// still lands, for this session. "Refused at capacity" and "below the floor" are different
    /// sentences, and this is the one that still has a [`Stored`] to show.
    #[test]
    fn naming_a_voice_at_the_cap_with_nothing_shorter_held_names_the_session_only() {
        let dimensions = MAX_REFERENCES_PER_SPEAKER + 1;
        let clusters = vec![cluster(0, axis(dimensions, 0), 6.0)];
        let unknown = numbering(&clusters);
        let held: Vec<(&str, Vec<f32>, Option<f64>)> = (1..=MAX_REFERENCES_PER_SPEAKER)
            .map(|i| ("Alice", axis(dimensions, i), Some(100.0 + i as f64)))
            .collect();
        let speakers = enrolled(&held);
        let assigned = no_names();

        let consequence = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        )
        .of("Alice")
        .unwrap();

        assert_eq!(
            consequence.stored,
            Some(Stored::AtCapacity {
                held: MAX_REFERENCES_PER_SPEAKER,
                shortest: Some(101.0),
            })
        );
        assert!(consequence.session_only());
    }

    /// Fifth outcome: two seconds of speech is under the reference floor, so the name is
    /// recorded against the session and `speakers.json` is not touched at all.
    #[test]
    fn naming_a_voice_under_the_reference_floor_stores_nothing() {
        let clusters = vec![cluster(0, voice(0), 2.0)];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[]);
        let assigned = no_names();

        let consequence = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        )
        .of("Alice")
        .unwrap();

        assert_eq!(consequence.refused, None);
        assert_eq!(consequence.stored, None);
        assert!(consequence.session_only());
    }

    /// The floor is a rule this preview honours rather than owns: `--force-reference` lifts it,
    /// and the same two-second voice then enrols.
    #[test]
    fn forcing_a_reference_stores_one_under_the_floor() {
        let clusters = vec![cluster(0, voice(0), 2.0)];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[]);
        let assigned = no_names();

        let consequence = preview_of(&clusters, &unknown, &speakers, &assigned, Enrolment::Always)
            .of("Alice")
            .unwrap();

        assert_eq!(consequence.stored, Some(Stored::Enrolled));
        assert!(!consequence.session_only());
    }

    /// Acceptance criterion #3: the reference this voice was wrongly stored under is dropped,
    /// and the person who loses it is named.
    #[test]
    fn an_answer_that_takes_a_reference_off_somebody_names_them() {
        let clusters = vec![cluster(0, voice(0), 40.0)];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[("Milo", voice(0), Some(30.0))]);
        let assigned = no_names();

        let consequence = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        )
        .of("Ryan")
        .unwrap();

        assert_eq!(consequence.refused, None);
        assert_eq!(
            consequence.displaced,
            [Displaced {
                name: "Milo".to_string(),
                remaining: 0
            }]
        );
        assert_eq!(consequence.stored, Some(Stored::Enrolled));
    }

    /// The count matters as much as the name: somebody holding three recordings who loses one
    /// still has two, and a line saying they have none would be false.
    #[test]
    fn somebody_who_keeps_other_references_is_reported_with_what_is_left() {
        let clusters = vec![cluster(0, voice(0), 40.0)];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[
            ("Milo", voice(0), Some(30.0)),
            ("Milo", nearly(70.0), Some(30.0)),
            ("Milo", nearly(85.0), Some(30.0)),
        ]);
        let assigned = no_names();

        let consequence = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        )
        .of("Ryan")
        .unwrap();

        assert_eq!(
            consequence.displaced,
            [Displaced {
                name: "Milo".to_string(),
                remaining: 2
            }]
        );
    }

    /// Acceptance criterion #4: two voices the segmenter heard at once cannot be one person, so
    /// the name stays where it is and the answer is refused -- naming the voice that holds it,
    /// by the "Unknown N" that `--voice` accepts.
    #[test]
    fn an_answer_the_heard_at_once_veto_would_refuse_names_the_voice_that_holds_it() {
        let mut clusters = vec![
            cluster(0, nearly(0.0), 40.0),
            cluster(1, nearly(20.0), 40.0),
        ];
        clusters[0].heard_at_once_with = vec![1];
        clusters[1].heard_at_once_with = vec![0];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[("Alice", nearly(0.0), Some(30.0))]);
        let assigned = no_names();

        let consequence = Preview::new(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            &clusters[1],
            Enrolment::default(),
            None,
            &[],
        )
        .of("Alice")
        .unwrap();

        assert_eq!(
            consequence.refused,
            Some(Refusal::Vetoed {
                holder: Some("Unknown 1".to_string())
            })
        );
    }

    /// The same pair under the one-remote-speaker assertion: the dry run honours the assertion
    /// the write will honour, so it predicts no veto at all -- a preview that still refused
    /// would show a cost the commit overrides, which is the drift the assertion parameter
    /// exists to keep out.
    #[test]
    fn a_preview_built_under_an_assertion_predicts_no_veto_for_the_asserted_name() {
        let mut clusters = vec![
            cluster(0, nearly(0.0), 40.0),
            cluster(1, nearly(20.0), 40.0),
        ];
        clusters[0].heard_at_once_with = vec![1];
        clusters[1].heard_at_once_with = vec![0];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[("Alice", nearly(0.0), Some(30.0))]);
        let assigned = no_names();

        let preview = Preview::new(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            &clusters[1],
            Enrolment::default(),
            Some("Alice"),
            &[],
        );

        let consequence = preview.of("Alice").unwrap();

        assert_eq!(consequence.refused, None);
        // Both voices read the asserted name in the labelling the write commits.
        let after = crate::effective_labels(
            &clusters,
            &unknown,
            &consequence.speakers,
            &consequence.assigned.names,
            Some("Alice"),
            None,
            None,
            &[],
        );
        assert_eq!(after[&0].label(), "Alice");
        assert_eq!(after[&1].label(), "Alice");
    }

    /// A name of nothing but spaces previews nothing, the same normalisation `of` applies --
    /// which is what keeps a stray-keystroke press from previewing an assertion it will not
    /// answer.
    #[test]
    fn an_assertion_of_nothing_but_whitespace_previews_nothing() {
        let clusters = vec![cluster(0, voice(0), 40.0)];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[]);
        let assigned = no_names();
        let preview = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        );

        for blank in ["", "   ", "\t\n"] {
            assert!(preview.one_speaker(blank).is_none(), "{blank:?}");
        }
        assert_eq!(
            preview.one_speaker("Grace"),
            Some(Assertion {
                voices: 1,
                vetoes_overridden: 0
            })
        );
    }

    /// The counts on a fragmented fixture with a heard-at-once pair: every voice, below the
    /// prompt floor included, is counted in `voices`, and the pair contributes exactly the one
    /// override the run's own counter reports -- the earlier partner holds the name, the later
    /// one is the one overridden.
    #[test]
    fn an_assertion_counts_every_uncommitted_voice_and_the_pair_contributes_one_override() {
        let mut clusters = vec![
            cluster(0, nearly(0.0), 40.0),
            cluster(1, nearly(20.0), 1.5),
            cluster(2, nearly(40.0), 0.9),
            cluster(3, nearly(60.0), 2.0),
        ];
        clusters[0].heard_at_once_with = vec![1];
        clusters[1].heard_at_once_with = vec![0];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[]);
        let assigned = no_names();
        let preview = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        );

        // All four voices, the two below-floor fragments among them: the assertion reaches the
        // quiet ones alike, and each of them gets its own commit.
        assert_eq!(
            preview.one_speaker("Grace"),
            Some(Assertion {
                voices: 4,
                vetoes_overridden: 1
            })
        );
    }

    /// A mid-run assertion: the voice this run has already committed is out of both counts, and
    /// the committed set seeds the override walk -- the pair's uncommitted partner is still
    /// reported as overridden against a holder whose row already stands.
    #[test]
    fn an_assertion_mid_run_excludes_committed_voices_from_both_counts() {
        let mut clusters = vec![
            cluster(0, nearly(0.0), 40.0),
            cluster(1, nearly(20.0), 40.0),
            cluster(2, nearly(40.0), 40.0),
        ];
        clusters[0].heard_at_once_with = vec![1];
        clusters[1].heard_at_once_with = vec![0];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[]);
        let assigned = no_names();
        let done = &clusters[0];
        let preview = Preview::new(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            &clusters[1],
            Enrolment::default(),
            None,
            std::slice::from_ref(&done),
        );

        assert_eq!(
            preview.one_speaker("Grace"),
            Some(Assertion {
                voices: 2,
                vetoes_overridden: 1
            })
        );
    }

    /// The committed set is not walk-order-dependent: a voice committed out of first-appearance
    /// order -- a targeted answer landed on a later voice first -- still overrides the veto of
    /// an earlier partner, because the run checks every committed holder rather than only the
    /// ones walked so far.
    #[test]
    fn an_assertion_checks_every_committed_holder_not_only_the_earlier_walked_ones() {
        let mut clusters = vec![
            cluster(0, nearly(0.0), 40.0),
            cluster(1, nearly(20.0), 40.0),
            cluster(2, nearly(40.0), 40.0),
        ];
        clusters[0].heard_at_once_with = vec![2];
        clusters[2].heard_at_once_with = vec![0];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[]);
        let assigned = no_names();
        // Cluster 2 speaks last but was committed first: the walk reaches cluster 0 before it,
        // and the override must still be counted.
        let done = &clusters[2];
        let preview = Preview::new(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            &clusters[0],
            Enrolment::default(),
            None,
            std::slice::from_ref(&done),
        );

        assert_eq!(
            preview.one_speaker("Grace"),
            Some(Assertion {
                voices: 2,
                vetoes_overridden: 1
            })
        );
    }

    // --- the group fold -------------------------------------------------------------------

    /// The guards, reached through the group door: a blank name previews nothing, a handle
    /// nothing resolves goes unanswered rather than partially answered, and an empty group is
    /// no group at all.
    #[test]
    fn a_group_that_cannot_name_its_members_previews_nothing() {
        let clusters = vec![cluster(0, voice(0), 40.0)];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[]);
        let assigned = no_names();
        let preview = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        );

        assert!(preview.group("", &["Unknown 1"]).is_none());
        assert!(preview.group("   ", &["Unknown 1"]).is_none());
        assert!(preview.group("Alice", &[]).is_none());
        assert!(preview.group("Alice", &["Unknown 9"]).is_none());
        assert!(
            preview
                .group("Alice", &["Unknown 1", "Unknown 9"])
                .is_none()
        );
    }

    /// A group of one member behaves exactly like today's plain naming of that member: the
    /// same displaced, stale, and final reference count as [`Preview::of`] for the same voice,
    /// which is the threshold below which the group carries no veto authority at all.
    #[test]
    fn a_one_member_group_is_the_plain_naming_of_that_member() {
        let clusters = vec![cluster(0, voice(0), 40.0)];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[("Milo", voice(0), Some(30.0))]);
        let assigned = no_names();
        let preview = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        );

        let solo = preview.of("Ryan").unwrap();
        let group = preview.group("Ryan", &["Unknown 1"]).unwrap();

        assert_eq!(group.name, "Ryan");
        assert_eq!(group.applied, ["Unknown 1".to_string()]);
        assert!(group.refused.is_empty());
        assert_eq!(group.vetoes_overridden, 0);
        assert_eq!(group.displaced, solo.displaced);
        assert_eq!(group.stale, solo.stale);
        // The total read off the final database, and the same number `of`'s own clone holds.
        assert_eq!(group.references_after, solo.speakers.references("Ryan"));
    }

    /// Duplicate handles dedupe to first appearance: the member is committed once, not twice,
    /// and the second mention neither adds a reference nor reports a second application.
    #[test]
    fn duplicate_member_handles_dedupe_to_one_commit() {
        let clusters = vec![cluster(0, voice(0), 40.0), cluster(1, voice(1), 40.0)];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[]);
        let assigned = no_names();
        let preview = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        );

        let group = preview
            .group("Alice", &["Unknown 2", "Unknown 1", "Unknown 1"])
            .unwrap();

        assert_eq!(
            group.applied,
            ["Unknown 1".to_string(), "Unknown 2".to_string()]
        );
        assert!(group.refused.is_empty());
        assert_eq!(group.references_after, 2);
    }

    /// The queue order, not the input order: the walk commits in first-appearance order, so a
    /// preview and a write cannot disagree about sequence whatever order the interface listed
    /// the marks in.
    #[test]
    fn the_group_walks_in_queue_order_not_input_order() {
        let clusters = vec![cluster(0, voice(0), 40.0), cluster(1, voice(1), 40.0)];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[]);
        let assigned = no_names();
        let preview = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        );

        let group = preview.group("Alice", &["Unknown 2", "Unknown 1"]).unwrap();

        assert_eq!(
            group.applied,
            ["Unknown 1".to_string(), "Unknown 2".to_string()]
        );
    }

    /// A refused member leaves the running state untouched while the members after it still
    /// apply: the refusal is per member, not per group, and the walk carries on against the
    /// state the run will actually reach.
    #[test]
    fn a_taken_refused_member_leaves_state_unchanged_while_later_members_apply() {
        let clusters = vec![
            cluster(0, nearly(0.0), 40.0),
            cluster(1, nearly(20.0), 40.0),
            cluster(2, voice(2), 40.0),
        ];
        let unknown = numbering(&clusters);
        // Bob is 40 degrees from cluster 1 and 60 from cluster 0: naming cluster 0 Alice
        // stores a reference nearer to cluster 1 than Bob's is, which takes Bob off it -- a
        // `Taken` the group honours rather than overrides.
        let speakers = enrolled(&[("Bob", nearly(60.0), Some(30.0))]);
        let assigned = no_names();
        let preview = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        );

        let group = preview.group("Alice", &["Unknown 1", "Unknown 3"]).unwrap();

        assert_eq!(
            group.refused,
            [(
                "Unknown 1".to_string(),
                Refusal::Taken {
                    voice: "Unknown 2".to_string(),
                    losing: "Bob".to_string()
                }
            )]
        );
        // The later member still applies, against the state the refusal left behind: Bob's
        // reference is still there, so it is displaced by the member that did commit rather
        // than lost to the one that did not.
        assert_eq!(group.applied, ["Unknown 3".to_string()]);
        assert_eq!(group.references_after, 1);
    }

    /// The aggregate the frame pane shows before any member commits: on a heard-at-once pair
    /// plus a pre-enrolled colliding name, the fold reports the veto it will override, the
    /// displacement it will make, and the group's total reference count off the final clone.
    #[test]
    fn the_group_fold_reports_the_veto_the_displacement_and_the_final_reference_count() {
        let mut clusters = vec![
            cluster(0, nearly(0.0), 40.0),
            cluster(1, nearly(20.0), 40.0),
            cluster(2, voice(2), 40.0),
        ];
        clusters[0].heard_at_once_with = vec![1];
        clusters[1].heard_at_once_with = vec![0];
        let unknown = numbering(&clusters);
        // Milo holds a reference built from cluster 0's exact voice: naming cluster 0 Grace
        // displaces it, and cluster 1 is the member the veto would have refused singly.
        let speakers = enrolled(&[("Milo", nearly(0.0), Some(30.0))]);
        let assigned = no_names();
        let preview = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        );

        let group = preview
            .group("Grace", &["Unknown 1", "Unknown 2", "Unknown 3"])
            .unwrap();

        assert_eq!(
            group.applied,
            [
                "Unknown 1".to_string(),
                "Unknown 2".to_string(),
                "Unknown 3".to_string()
            ]
        );
        assert!(group.refused.is_empty());
        // Cluster 2 overlaps nobody; cluster 1 was heard at once with cluster 0, which holds
        // the name when cluster 1 lands -- exactly one overridden veto, counted once.
        assert_eq!(group.vetoes_overridden, 1);
        assert_eq!(
            group.displaced,
            [Displaced {
                name: "Milo".to_string(),
                remaining: 0
            }]
        );
        // Three references stored, one displaced, none held before: the total is what the
        // final database holds, not the sum of the members' individual counts.
        assert_eq!(group.references_after, 3);
    }

    /// The group's report as sentences, pinned verbatim: the same register
    /// [`Consequence::would_do`] keeps, so the frame's pane and whatever prints the commit read
    /// the same words off the same struct.
    #[test]
    fn the_groups_report_reads_as_its_own_lines() {
        let empty = GroupConsequence {
            name: "Grace".to_string(),
            applied: vec!["Unknown 1".to_string()],
            refused: Vec::new(),
            vetoes_overridden: 0,
            references_after: 1,
            displaced: Vec::new(),
            stale: Vec::new(),
        };
        assert_eq!(empty.would_do(), Vec::<String>::new());

        let displaced_only = GroupConsequence {
            displaced: vec![Displaced {
                name: "Milo".to_string(),
                remaining: 2,
            }],
            ..empty.clone()
        };
        assert_eq!(
            displaced_only.would_do(),
            ["takes a recording off Milo, leaving them 2"]
        );

        let stale_only = GroupConsequence {
            stale: vec!["Bob".to_string()],
            ..empty.clone()
        };
        // The single voice's "of this voice" is gone: the union carries no per-member
        // attribution, so the line names only the person the legacy reference stands under.
        assert_eq!(
            stale_only.would_do(),
            ["leaves a recording standing under Bob"]
        );

        let refused_only = GroupConsequence {
            refused: vec![(
                "Unknown 3".to_string(),
                Refusal::Taken {
                    voice: "Unknown 2".to_string(),
                    losing: "Bob".to_string(),
                },
            )],
            ..empty.clone()
        };
        assert_eq!(
            refused_only.would_do(),
            ["Unknown 3: unavailable: Unknown 2 would stop reading Bob"]
        );

        let combined = GroupConsequence {
            displaced: vec![Displaced {
                name: "Milo".to_string(),
                remaining: 0,
            }],
            stale: vec!["Ivan".to_string()],
            refused: vec![(
                "Unknown 2".to_string(),
                Refusal::Vetoed {
                    holder: Some("Unknown 1".to_string()),
                },
            )],
            ..empty
        };
        // Displaced first, then stale, then the exceptions: the load-bearing counts before the
        // lines a clipping pane may lose.
        assert_eq!(
            combined.would_do(),
            [
                "takes a recording off Milo, leaving them 0",
                "leaves a recording standing under Ivan",
                "Unknown 2: unavailable: Unknown 1 was heard at the same time as this voice \
                 and would keep the name",
            ]
        );
    }

    /// The other refusal: the answer would take a name off a voice the user was not asked
    /// about, because the reference it stores is nearer to that voice than its current name's
    /// is.
    #[test]
    fn an_answer_that_would_move_another_voices_name_is_refused_with_both_names() {
        let clusters = vec![
            cluster(0, nearly(0.0), 40.0),
            cluster(1, nearly(20.0), 40.0),
        ];
        let unknown = numbering(&clusters);
        // Bob is 40 degrees from cluster 1 and 60 from cluster 0: inside the cut for one voice
        // and outside it for the other, which is the state a theft starts from.
        let speakers = enrolled(&[("Bob", nearly(60.0), Some(30.0))]);
        let assigned = no_names();

        let consequence = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        )
        .of("Alice")
        .unwrap();

        assert_eq!(
            consequence.refused,
            Some(Refusal::Taken {
                voice: "Unknown 2".to_string(),
                losing: "Bob".to_string()
            })
        );
    }

    /// A blank answer is a skip rather than an answer, so there is no consequence to report --
    /// which is what keeps "what an empty name means" in one place rather than two.
    #[test]
    fn a_name_that_is_only_whitespace_has_no_consequence() {
        let clusters = vec![cluster(0, voice(0), 40.0)];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[]);
        let assigned = no_names();
        let preview = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        );

        for blank in ["", "   ", "\t\n"] {
            assert!(preview.of(blank).is_none(), "{blank:?}");
        }
        assert!(preview.of("  Alice  ").is_some());
    }

    /// The stale list, from both sides. Below the floor `speakers.json` is untouched, so a
    /// legacy reference built from this exact voice survives under the wrong name and is
    /// reported; above the floor the correction has already dropped it, so there is nothing to
    /// report and somebody has been displaced instead.
    #[test]
    fn a_legacy_reference_built_from_this_voice_is_reported_only_below_the_floor() {
        let unknown_below = numbering(&[cluster(0, voice(0), 2.0)]);
        let speakers = enrolled(&[("Milo", voice(0), None)]);
        let assigned = no_names();

        let below = vec![cluster(0, voice(0), 2.0)];
        let consequence = preview_of(
            &below,
            &unknown_below,
            &speakers,
            &assigned,
            Enrolment::default(),
        )
        .of("Ryan")
        .unwrap();
        assert_eq!(consequence.stored, None);
        assert_eq!(consequence.stale, ["Milo".to_string()]);
        assert_eq!(consequence.displaced, []);

        let above = vec![cluster(0, voice(0), 40.0)];
        let consequence = preview_of(
            &above,
            &numbering(&above),
            &speakers,
            &assigned,
            Enrolment::default(),
        )
        .of("Ryan")
        .unwrap();
        assert_eq!(consequence.stale, [] as [String; 0]);
        assert_eq!(
            consequence.displaced,
            [Displaced {
                name: "Milo".to_string(),
                remaining: 0
            }]
        );
    }

    /// A preview must not be able to change what the next preview says: asking about two names
    /// and then about the first again gives the first answer back.
    #[test]
    fn previewing_one_name_does_not_change_what_another_preview_says() {
        let clusters = vec![cluster(0, voice(0), 40.0)];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[("Milo", voice(0), Some(30.0))]);
        let assigned = no_names();
        let preview = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        );

        let first = preview.of("Alice").unwrap();
        let _ = preview.of("Bob").unwrap();
        assert_eq!(preview.of("Alice").unwrap(), first);
    }

    /// A commit interrupted between its `speakers.json` write and its `speaker_names.json`
    /// write leaves a reference with no names-file row. Confirming that stranded voice plainly,
    /// on a voice heard at once with another, must stand the declaration up in *both* stores:
    /// the row the heard-at-once exclusion needs to stand on is written by the very
    /// confirmation that today forgets it.
    #[test]
    fn a_stranded_confirmation_on_a_heard_at_once_voice_keeps_its_row() {
        let mut clusters = vec![cluster(0, voice(0), 40.0), cluster(1, voice(1), 40.0)];
        clusters[0].heard_at_once_with = vec![1];
        clusters[1].heard_at_once_with = vec![0];
        let unknown = numbering(&clusters);
        // The post-SIGKILL state: the database holds a bit-identical reference built from this
        // exact fragment under the name, but no names-file row exists yet.
        let speakers = enrolled(&[("Grace", voice(0), Some(30.0))]);
        let assigned = no_names();

        let consequence = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        )
        .of("Grace")
        .unwrap();

        assert_eq!(consequence.stored, Some(Stored::AlreadyHeld));
        // The declaration stands up in the names file rather than being forgotten.
        assert!(
            consequence
                .assigned
                .names
                .iter()
                .any(|row| row.cluster == 0 && row.name == "Grace")
        );
        // A later pass -- no pending, no assertion -- reads the name, not a demotion: the row
        // resolves the voice by embedding equality and awards it, so the veto never runs.
        let later = crate::effective_labels(
            &clusters,
            &unknown,
            &consequence.speakers,
            &consequence.assigned.names,
            None,
            None,
            None,
            &[],
        );
        assert_eq!(later[&0].label(), "Grace");
    }

    /// The mirror of the stranded case with no overlap: an `AlreadyHeld` confirmation on a solo
    /// voice is pure redundancy, so the one-voice-one-record forget still applies and no row is
    /// left behind -- byte-identical behaviour to before the fix.
    #[test]
    fn a_stranded_confirmation_on_a_solo_voice_still_forgets_its_row() {
        let clusters = vec![cluster(0, voice(0), 40.0), cluster(1, voice(1), 40.0)];
        let unknown = numbering(&clusters);
        let speakers = enrolled(&[("Grace", voice(0), Some(30.0))]);
        let assigned = no_names();

        let consequence = preview_of(
            &clusters,
            &unknown,
            &speakers,
            &assigned,
            Enrolment::default(),
        )
        .of("Grace")
        .unwrap();

        assert_eq!(consequence.stored, Some(Stored::AlreadyHeld));
        assert!(
            !consequence
                .assigned
                .names
                .iter()
                .any(|row| row.cluster == 0)
        );
    }

    /// The refusal rule, exercised as the pure comparison it is: two labellings in, and either
    /// nothing or the voice that would have paid. Cheaper and more direct than reaching each
    /// branch through a whole answer, which the tests above do for the paths that produce
    /// these maps.
    mod refusal {
        use super::*;

        fn identified(name: &str) -> Attribution {
            Attribution::Identified {
                name: name.to_string(),
                similarity: 0.9,
            }
        }

        fn numbers(ids: &[u32]) -> BTreeMap<u32, String> {
            ids.iter()
                .enumerate()
                .map(|(nth, &id)| (id, unknown_speaker(nth + 1)))
                .collect()
        }

        #[test]
        fn an_answer_that_costs_nobody_anything_is_free() {
            let labels = |zero: Attribution| BTreeMap::from([(0, zero), (1, identified("Bob"))]);

            assert_eq!(
                refusal_of(
                    0,
                    "Alice",
                    &numbers(&[0, 1]),
                    &labels(Attribution::Unknown("Unknown 1".to_string())),
                    &labels(identified("Alice")),
                ),
                None
            );
        }

        /// The answered voice did not get the name, and another voice has it: the veto, which
        /// is the one loss the reference set cannot design away.
        #[test]
        fn an_answer_the_veto_took_names_the_voice_that_kept_the_name() {
            let corrected = BTreeMap::from([
                (0, identified("Alice")),
                (1, Attribution::Unknown("Unknown 2".to_string())),
            ]);
            let after = BTreeMap::from([
                (0, identified("Alice")),
                (1, Attribution::Unknown("Unknown 2".to_string())),
            ]);

            assert_eq!(
                refusal_of(1, "Alice", &numbers(&[0, 1]), &corrected, &after),
                Some(Refusal::Vetoed {
                    holder: Some("Unknown 1".to_string())
                })
            );
        }

        /// An answer that simply did not take, with nobody else holding the name. Unreachable
        /// through the veto -- something has to have won the name for this voice to have lost
        /// it -- and refused anyway, because a reference that then names nobody is not a state
        /// to write silently.
        #[test]
        fn an_answer_that_did_not_take_at_all_is_refused_with_nobody_to_name() {
            let labels = BTreeMap::from([(0, Attribution::Unknown("Unknown 1".to_string()))]);

            assert_eq!(
                refusal_of(0, "Alice", &numbers(&[0]), &labels, &labels),
                Some(Refusal::Vetoed { holder: None })
            );
        }

        /// Theft: the answered voice gets the name, and another voice's name goes with it.
        #[test]
        fn an_answer_that_moves_another_voices_name_reports_that_voice_and_the_name() {
            let corrected = BTreeMap::from([
                (0, Attribution::Unknown("Unknown 1".to_string())),
                (1, identified("Bob")),
            ]);
            let after = BTreeMap::from([(0, identified("Alice")), (1, identified("Alice"))]);

            assert_eq!(
                refusal_of(0, "Alice", &numbers(&[0, 1]), &corrected, &after),
                Some(Refusal::Taken {
                    voice: "Unknown 2".to_string(),
                    losing: "Bob".to_string()
                })
            );
        }

        /// A voice that *gains* a name is not a cost: that is one person whose clustering split
        /// in two being named by one answer, which is behaviour the split-voice guard depends
        /// on rather than something to refuse.
        #[test]
        fn a_voice_that_gains_a_name_costs_nothing() {
            let corrected = BTreeMap::from([
                (0, Attribution::Unknown("Unknown 1".to_string())),
                (1, Attribution::Unknown("Unknown 2".to_string())),
            ]);
            let after = BTreeMap::from([(0, identified("Alice")), (1, identified("Alice"))]);

            assert_eq!(
                refusal_of(0, "Alice", &numbers(&[0, 1]), &corrected, &after),
                None
            );
        }

        /// The distinction the two labellings exist for. A name the *correction* removed is
        /// already gone in `corrected`, so it is not a refusal -- the user has just said that
        /// reference was of somebody else, and it gets a line of its own instead.
        #[test]
        fn a_name_the_correction_itself_removed_is_not_a_refusal() {
            // Nate held cluster 1 before the answer; the correction dropped the reference that
            // did it, so `corrected` already reads "Unknown 2" there.
            let corrected = BTreeMap::from([
                (0, Attribution::Unknown("Unknown 1".to_string())),
                (1, Attribution::Unknown("Unknown 2".to_string())),
            ]);
            let after = BTreeMap::from([
                (0, identified("Ryan")),
                (1, Attribution::Unknown("Unknown 2".to_string())),
            ]);

            assert_eq!(
                refusal_of(0, "Ryan", &numbers(&[0, 1]), &corrected, &after),
                None
            );
        }
    }
}
