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
//! Nothing on the `--name` path reaches [`Preview::of`] at all, so the non-interactive command
//! pays for the preview it never asks for exactly nothing.

use std::collections::BTreeMap;

use meethook_session::{Displaced, EnrolledSpeakers, SpeakerCluster, SpeakerNames, Stored};
use meethook_transcribe::Attribution;

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
}

impl<'a> Preview<'a> {
    /// The six things a dry run of one answer needs, and nothing else.
    ///
    /// `speakers` and `assigned` are the two files a name can land in, borrowed rather than
    /// cloned here: the clone happens per [`Preview::of`] call, since a preview that held its
    /// own copy would go stale the moment an answer was committed.
    pub(crate) fn new(
        clusters: &'a [SpeakerCluster],
        unknown: &'a BTreeMap<u32, String>,
        speakers: &'a EnrolledSpeakers,
        assigned: &'a SpeakerNames,
        cluster: &'a SpeakerCluster,
        enrolment: Enrolment,
    ) -> Preview<'a> {
        Preview {
            clusters,
            unknown,
            speakers,
            assigned,
            cluster,
            enrolment,
        }
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

        // Naming a voice and storing a reference built from it are two different acts, and this
        // is where they come apart. Below the floor the name is recorded against the session and
        // `speakers.json` is not touched at all -- see `REFERENCE_FLOOR_SECONDS` for what a
        // reference built from two seconds of speech does to every future meeting.
        let session_only = self.cluster.speech_seconds < REFERENCE_FLOOR_SECONDS
            && self.enrolment != Enrolment::Always;

        // Everything this answer would write, applied to copies.
        let mut candidate = self.speakers.clone();
        let mut candidate_assigned = self.assigned.clone();

        // The correction, on the above-floor path only: a reference identical to this cluster
        // was built from this voice, and the user has just told us this voice is somebody else,
        // so it is a stored claim about a person it is not of and it competes as an argmax in
        // every future meeting -- winning whenever its name sorts first
        // (`identify::best_match`'s tie-break).
        let displaced = if session_only {
            Vec::new()
        } else {
            candidate.forget_reference(&self.cluster.embedding, name)
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
        );

        // The addition. `None` on the below-floor path, where no reference is stored at all.
        let stored = if session_only {
            candidate_assigned.assign(self.cluster.id, name, self.cluster.embedding.clone());
            None
        } else {
            let stored = candidate.store_reference(
                name,
                self.cluster.embedding.clone(),
                self.cluster.speech_seconds,
            );
            if matches!(stored, Stored::AtCapacity { .. }) {
                // At the cap with nothing shorter than this recording to displace, so it is not
                // stored and the answer falls back to the session-only path rather than being
                // lost: the transcript still reads the right person, and nothing already stored
                // is dropped for a recording that is no better than it.
                candidate_assigned.assign(self.cluster.id, name, self.cluster.embedding.clone());
            } else {
                // One voice, one record. A voice named for this session only and then enrolled
                // properly -- the same fragment reached again with `--force-reference`, or a
                // later clustering that gave it enough speech -- must stop also being an
                // assignment, or the two could be made to disagree about who it is.
                candidate_assigned.forget(self.cluster.id);
            }
            Some(stored)
        };

        let after = effective_labels(
            self.clusters,
            self.unknown,
            &candidate,
            &candidate_assigned.names,
        );

        // A legacy reference that *is* this exact fragment, still standing under somebody
        // else's name. Only the below-floor path can leave one -- above the floor
        // `forget_reference` has just dropped every such row -- so this is empty by
        // construction on every other path rather than by a special case here.
        let stale: Vec<String> = candidate
            .speakers
            .iter()
            .filter(|s| s.name != name && s.embedding == self.cluster.embedding)
            .map(|s| s.name.clone())
            .collect();

        Some(Consequence {
            refused: refusal_of(self.cluster.id, name, self.unknown, &corrected, &after),
            stored,
            displaced,
            stale,
            speakers: candidate,
            assigned: candidate_assigned,
        })
    }
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
}

/// What honouring an answer would have taken away from a voice the user was not asked about.
///
/// Both variants name that voice by its "Unknown N" rather than by what it currently reads,
/// because that is the one handle which reaches a voice whatever it is called and is exactly
/// what [`crate::VoiceSelector`] accepts -- so a refusal is a line the user can act on rather
/// than only read.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    if after.get(&answered).map(Attribution::label) != Some(name) {
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
        Preview::new(
            clusters,
            unknown,
            speakers,
            assigned,
            &clusters[0],
            enrolment,
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
