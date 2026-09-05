//! Correcting what an automatic pass got wrong: the voices it could not name, and the meeting
//! it guessed at.
//!
//! Naming voices is the larger half and the one everything below is about; the calendar
//! correction is at the bottom, on the same premise and the same seam-shaped design.
//!
//! This is the only interactive path in meethook, and it is built so that almost none of it
//! is interactive. Which sessions get visited, which voices get asked about, in what order,
//! and what each answer writes to disk are all decided here, against the one-method
//! [`Interviewer`] seam -- with no terminal and no audio device on this side of it. The live
//! implementation, which prints, plays and reads a line, lives in the CLI crate; the test one
//! answers from a script, which is what makes the sequencing decidable in `cargo test` the
//! way the record loop's already is.
//!
//! Diarization never runs. Everything a prompt needs -- the voice's embedding and the bounds
//! of a clip to play -- was written to `speaker_clusters.json` when the session was
//! transcribed, which is the whole reason that file is on disk.
//!
//! Two rules are worth stating before the code:
//!
//! *Unresolved* is decided against the database as it stands right now, not against the text
//! of the transcript. Name someone in the first session and their voice in the third is
//! matched and passed over, with no cross-session comparison of unnamed voices anywhere: the
//! deduplication is enrollment itself. The one exception is [`Offer::named`], which asks about
//! resolved voices too so that an identification the database got wrong can be answered --
//! without it a false accept would be permanent short of hand-editing `speakers.json`. A
//! [`VoiceSelector`] is the same exception aimed at one voice: it overrides both [`Offer`]
//! filters for the voice it names, on the reading that somebody naming a specific voice has
//! already made the judgement those filters make on their behalf.
//!
//! A rewritten transcript is exactly what `transcribe --force` would now produce. That is the
//! invariant everything below is implemented against, because it is what stops `enroll` and
//! `transcribe` from becoming two sources of truth about a transcript. It applies to every
//! session this reads, not only to the one an answer was given in: a transcript written
//! before its speaker was enrolled is brought up to date on the way past, since a session
//! with nothing left to ask about would otherwise keep calling a named colleague "Unknown 2"
//! for good. Files that already agree are left alone, byte for byte.
//!
//! # This crate also *reports on* and *removes from* the database it writes
//!
//! [`run_speakers`] answers the question the file cannot: who is enrolled, and what is each
//! stored recording of them actually naming. It lives here rather than beside `speakers.json`
//! because the answer is not a fact about that file -- it is derived by labelling every session
//! on disk twice, once with the database as it stands and once with one row removed, which is
//! the two-labelling diff `enroll_session` already performs over a single session before it
//! honours an answer. See the `references` module for the derivation and its cost. Nothing on
//! that path writes anything.
//!
//! [`run_forget`] is the removal that report exists to inform: it takes the same derivation, uses
//! it to say what dropping a reference -- or a whole person -- would cost, and writes only once the
//! user has confirmed. See the `forget` module for the ordering and the wording. It is the last
//! thing in the tool that used to require a text editor.
//!
//! # And it decides what a *typed name* means
//!
//! [`resolve()`] is the counterpart to those two aimed at the moment before a write: a name matches
//! exactly and case-sensitively everywhere in this tool, so a typo silently enrols a second
//! person rather than adding a recording to the first. It turns typed text into an enrolled
//! person, a shortlist of the people it might mean, or a genuinely new name -- and it never picks
//! between two candidates, because on a real database `Ivan` and `Owen` are two colleagues.
//! Showing the shortlist and taking the confirmation belongs to the interface; the decision is
//! here because it is what lands on disk. See the `resolve` module for the fold and the ranking.
//!
//! # And it says what an answer *would* do, before it does it
//!
//! [`Voice::preview`] carries the other half of that moment across the seam: given a candidate
//! name, what accepting it would write -- an enrollment, another recording, a replacement of the
//! shortest one somebody held, a refusal at the cap, a name for this session alone -- and what
//! it would cost, including a reference taken off somebody else and an answer the heard-at-once
//! veto will not honour. Nothing is written by asking. It is the pre-flight `enroll_session` has
//! always run on copies before committing, made addressable rather than reimplemented, so a
//! preview and a write cannot drift apart. See the `consequence` module for the dry run and its
//! cost, which is per *answer* and emphatically not per keystroke.
//!
//! # And it corrects the *meeting* a session was labelled with
//!
//! [`run_meeting`] is this crate's premise -- an automatic guess, a human correction, a
//! transcript rewritten in place -- aimed at the calendar match rather than at a voice. It is
//! here because it needs the second and third of those and nothing from `meethook-record`: the
//! events it offers arrive through the one-method [`MeetingSource`] seam, exactly as an answer
//! arrives through [`Interviewer`], so its whole wording is decidable in `cargo test` on a
//! machine with no calendar grant. See the `meeting` module for what a correction may print,
//! which is deliberately less than it holds.

mod consequence;
mod forget;
mod groups;
mod interview;
mod meeting;
mod narration;
mod prompt;
mod queue;
mod references;
mod resolve;
mod session;

pub use consequence::{Assertion, Consequence, Demotion, GroupConsequence, Preview, Refusal};
pub use forget::{Confirm, Forgotten, Removal, Target, run_forget};
pub use groups::{FragmentGroup, GROUP_DISTANCE};
pub use interview::{Answer, GivenName, Interviewer, MeetingLabel};
pub use meethook_session::Stored;
pub use meeting::{Labelled, MeetingChoice, MeetingOffer, MeetingSource, Relabelling, run_meeting};
pub use narration::{
    AnswerNote, Lines, Narrator, Nearest, NotSelected, Note, PassedOver, RunNote, SessionFile,
    SessionNote, VoiceDescription,
};
pub use prompt::{Snippet, Voice, speech, write_clip};
pub use queue::{Offer, Position, Queued, Selection, Sessions, VoiceSelector};
pub use references::{
    Enrolled, Reference, Scan, Unreadable, VoiceChange, incomplete, run_speakers, scan,
};
pub use resolve::{Likeness, Match, Resolution, resolve};

pub(crate) use session::{effective_labels, enroll_session, relabel};

use std::path::PathBuf;

use meethook_session::{
    DiscoveredSession, EnrolledSpeakers, Paths, SessionId, TranscriptTemplate, discover_sessions,
};

#[cfg(doc)]
use queue::PROMPT_FLOOR_SECONDS;

/// How much a voice has to have spoken before an answer about it becomes a *reference* in
/// `speakers.json`.
///
/// A rule about what an accepted name writes, and nothing else. Below this a name is recorded
/// against the session in `speaker_names.json` -- the voice still reads as that person in this
/// transcript -- and `speakers.json` is not touched; [`Enrolment::Always`] overrides that.
///
/// # Where 5 s comes from
///
/// TASK-019.01 measured 104 reference-versus-turn similarities on real sessions and put the
/// usable floor in the band **(2.4 s, 5.2 s]**, recommending **5.0 s**, with `>=` meaning
/// above it. Both edges are priced.
///
/// Below 2.4 s, four measured references failed outright: 0.95 s scored 0.702 against its own
/// owner and was rejected, 1.05 s scored 0.807 with the *wrong* argmax and was rejected, 1.3 s
/// scored 0.441, 2.4 s scored 0.392. A reference built from a fragment that short describes a
/// phoneme and a prosody rather than a person, so it neither matches its owner nor stays out
/// of everybody else's way.
///
/// Above 5.2 s it starts refusing references that demonstrably worked: a 5.2 s prefix matched
/// its own speaker at 0.160, and one voice's references spread over 2.9 / 3.2 / 4.6 s all held.
/// Past ~7.8 s the cost has a name on it -- Alex's three ~8 s fragments identify
/// their owner at 0.321 / 0.173 / 0.295, and a floor above them would refuse a reference that
/// works.
///
/// Why 5.0 rather than the middle of the band: everything inside it is selection-biased low by
/// the measured 0.26-0.37, while the only references measured free of that bias succeeded at
/// 7.8 s and failed at 1.05 s -- so the honest reading of the evidence sits at the top of the
/// band rather than in the middle of it. Three caveats travel with the number: that selection
/// effect; one call on one microphone is the easiest condition a reference will ever face; and
/// 104 points are readings rather than rates.
///
/// # Three floors, same units, three questions
///
/// None of them implies another, and they are deliberately not derived from each other:
///
/// - [`meethook_transcribe::SPEAKER_FLOOR_SECONDS`] (30 s) decides **which clusters are solid
///   enough to adopt fragments into**. Necessarily the largest: it is about a centroid being
///   allowed to claim somebody else's turns.
/// - [`PROMPT_FLOOR_SECONDS`] (5 s) decides **which voices are worth asking about**. Getting it
///   wrong costs a question in one direction or the other, and nothing on disk.
/// - This one decides **which answers become references**. Naming somebody who spoke 8 s is
///   right; storing a reference built from 8 s of audio is what TASK-019 measured going wrong.
///
/// # Why it sits on the same value as [`PROMPT_FLOOR_SECONDS`]
///
/// Not a coincidence to be tidied away into one constant, and not a duplication: they answer
/// different questions and would move independently -- a better prompt heuristic does not
/// change what a reference needs, and vice versa. That they coincide today has a consequence
/// worth stating, because it is what bounds this path's exposure: a default run offers only
/// voices at or above 5.0 s, so every answer it collects clears this floor and the
/// session-scoped branch never fires. What reaches it is `enroll --all`, plus any small voice
/// reached through `--correct`.
///
/// Two floors on one value must also agree about their own boundary, or a cluster of exactly
/// 5.0 s would be asked about and then have its answer refused. The comparison here is
/// `speech_seconds >= REFERENCE_FLOOR_SECONDS`, matching both neighbours.
pub(crate) const REFERENCE_FLOOR_SECONDS: f64 = 5.0;
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Session(#[from] meethook_session::Error),

    #[error("could not write the clip to {path}: {source}")]
    Wav {
        path: PathBuf,
        #[source]
        source: hound::Error,
    },

    #[error("could not write output: {0}")]
    Output(#[from] std::io::Error),
}
pub type Result<T> = std::result::Result<T, Error>;
/// When an accepted name is allowed to become a reference in `speakers.json`.
///
/// A separate axis from [`Offer`] rather than a third field on it, because the two answer
/// different questions: `Offer` decides *which voices a run asks about*, and this decides
/// *what an answer writes*. Folding them together would make `--all` -- a request to be shown
/// the quiet voices -- also a request to store references built from two seconds of audio,
/// which is the failure `REFERENCE_FLOOR_SECONDS` exists to prevent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Enrolment {
    /// The default: a voice with at least `REFERENCE_FLOOR_SECONDS` of speech becomes a
    /// reference, and a quieter one is named for its own session only.
    #[default]
    AboveTheFloor,

    /// `--force-reference`: every accepted name becomes a reference, whatever the duration.
    ///
    /// The explicit override TASK-019 requires before a short reference is stored. Worth
    /// having because the floor is a rule about the average case and the user may know better
    /// in a particular one -- the clip is the only recording of somebody, or a legacy
    /// reference built from that same fragment needs replacing, which the default path can
    /// only report and not fix.
    Always,
}
/// How a run is configured: which voice or voices it offers, and what an answer to one writes.
///
/// A bundle rather than four parameters, because it is threaded through the walk over sessions
/// unchanged and every function on that path would otherwise carry all of them. It is the
/// caller's parameter too: four axes named at the call site read better than four in a row of
/// eight positional arguments, and a caller adding one of them cannot then transpose the rest.
///
/// There is no `Default`: [`template`](Self::template) has no sensible one. What a caller with
/// no opinion wants is [`TranscriptTemplate::resolve`] with no explicit path, which is a
/// fallible read of the root rather than a constant.
#[derive(Debug, Clone)]
pub struct EnrollRules<'a> {
    /// `Some` replaces the queue with one voice, however the user pointed at it; `None` is the
    /// queue. Not a fourth flag on [`Offer`], because it does not widen the queue -- it stands
    /// in for it.
    pub selector: Option<Selection>,

    /// Which voices get asked about. Changes the questions and nothing else: the same answers
    /// write the same files however a voice came to be offered.
    pub offer: Offer,

    /// Which sessions get visited -- the separate question [`Sessions`] describes. Ignored when
    /// `selector` is `Some`, which stands in for the queue and its gates alike.
    pub sessions: Sessions,

    /// Whether a stale `transcript.md` is brought in line before the first question.
    ///
    /// Every run that may write answers does it: a label left stale by an earlier session's
    /// answer would otherwise survive every later pass-over, since such a session is opened
    /// only to be passed over. The read-only faces (`--list`, `--dry-run`) set it off: they
    /// promise the root exactly as found, and a query must not be a writer however small the
    /// write.
    pub relabel_transcript: bool,

    /// What an accepted name writes -- the other axis, and the only one that changes that.
    pub enrolment: Enrolment,

    /// What a rewritten `transcript.md` is rendered through, handed in already compiled.
    ///
    /// Naming a voice must not be able to change the shape of a transcript it did not write,
    /// so this belongs to the run rather than to a session: it is resolved once, from the same
    /// root `transcribe` resolved it from, and every session rewritten here goes through it.
    pub template: &'a TranscriptTemplate,

    /// The user's assertion that a session's speaker track is one person, by name -- or `None`
    /// for a run that asks about voices the ordinary way.
    ///
    /// Where present, it stands in for the queue and its gates alike, the way a [`Selection`]
    /// does: every voice in the session is named with it, below the prompt floor included, and
    /// nothing is asked about any of them. A selector passed beside it is ignored for the same
    /// reason: pointing at one voice and asserting the whole track are two different requests,
    /// and the CLI refuses to take both at once.
    ///
    /// Trimmed and non-empty by the time it gets here -- [`run_enroll`] trims it and reports an
    /// empty one as a request it could not serve, which is what keeps "what an empty name
    /// means" in one place rather than two.
    pub one_speaker: Option<&'a str>,
}
/// What a run did, so the caller can pick an exit status without re-deriving it.
///
/// `named`, `skipped`, `kept` and `held_back` count *voices*; `passed_over` counts *sessions*
/// that were never asked about at all; `failed` counts requests that could not be served.
///
/// `failed` is every request this run could not serve: a session it could not read, an id that
/// is not on disk, and a [`VoiceSelector`] that matched no voice or more than one. They are one
/// count because the caller does one thing with them -- exit non-zero -- and each has already
/// printed its own line saying which of them it was.
///
/// `held_back` is unresolved voices that sat under `PROMPT_FLOOR_SECONDS` and so were never
/// asked about. Reported rather than merely not-counted, because a run that asked seven
/// questions about a meeting of fifty-six voices should say what it did not ask about. A
/// targeted run holds nothing back: it was aimed at one voice rather than filtered down to it.
///
/// `kept` is already-named voices the user left as they were -- an answer, and the common one
/// under [`Offer::named`]. Counted apart from `skipped` because they write the same nothing
/// but mean opposite things: a kept voice *has* a name, and folding it into the skipped count
/// would have the summary report a named voice as unnamed.
///
/// `session_only` is a **sub-count of `named`**, not a category beside it: those voices were
/// named, and the name is in their session's `speaker_names.json` rather than in
/// `speakers.json` -- because they spoke less than `REFERENCE_FLOOR_SECONDS`, or because that
/// person already holds [`meethook_session::MAX_REFERENCES_PER_SPEAKER`] recordings. A caller
/// adding the fields up would double-count them; a caller checking "did this run name anybody"
/// keeps using `named`, which is what makes it a sub-count rather than an eighth bucket.
///
/// `refused` is answers that were not honoured because honouring them would have taken a name
/// off another voice -- see [`Refusal`]. Not a `skipped`: the user answered, and the answer was
/// declined rather than absent. Not a `failed` either, since nothing went wrong and the run
/// carries on; it is counted so the summary can say a question was answered and came to
/// nothing, which is the one outcome a silent revert used to hide.
///
/// `denied` counts tentative guesses the user refused. A denial commits -- the suppression row
/// lands in `speaker_names.json` and the transcript moves the guess back to its "Unknown N" --
/// so it is neither a `refused` (the answer was honoured) nor a `skipped` (it wrote something),
/// and it is counted so the summary can say how many guesses were turned down rather than
/// leaving a run that denied three of them reporting only what it named.
///
/// `asserted` is a **sub-count of `named`**, like `session_only`: those voices were named, and
/// the naming came from the one-remote-speaker assertion rather than from an answer given per
/// voice. Counted apart because the summary says what the assertion did in its own sentence,
/// and folding it into `named` alone would leave a run that named forty-one voices reporting
/// only that it asked and answered zero questions.
///
/// `vetoes_overridden` counts voices the heard-at-once veto would have refused to put under the
/// asserted name, and that the assertion named anyway. Every one of them has printed its own
/// line saying which voices it overlapped, which is what makes overriding reported rather than
/// swallowed; the count is what the summary needs to say how many there were.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EnrollReport {
    pub named: usize,
    pub session_only: usize,
    pub skipped: usize,
    pub kept: usize,
    pub held_back: usize,
    pub refused: usize,
    pub denied: usize,
    pub passed_over: usize,
    pub failed: usize,
    pub asserted: usize,
    pub vetoes_overridden: usize,
}
/// Whether the queue should carry on to the next session.
enum Outcome {
    Finished,
    Quit,
}
/// Works through every unresolved voice in a selection of sessions, asking who each one is.
///
/// With no `requested` ids every discovered session is considered, in session-id order;
/// naming ids scopes the run to those, and one that is not on disk is reported individually
/// rather than ignored -- enrolling three of four requested sessions and exiting 0 would look
/// like success.
///
/// The enrolled database is read once and carried through the run, updated in memory by each
/// accepted name and written before anything else. That is what makes the second session's
/// copy of a person somebody was just named in the first one a match rather than a second
/// prompt.
///
/// [`EnrollRules::selector`] replaces the queue with exactly one voice of one session, and needs
/// exactly one session id to be meaningful: a voice number says nothing across sessions, and a
/// name would fan out over every recording on disk. It overrides both [`Offer`] filters for
/// that voice, so passing `--all` or `--correct` beside it changes nothing rather than
/// conflicting with it.
///
/// [`Offer`] widens which voices get asked about -- the quiet ones, the already-named ones, or
/// both. It changes which questions get asked and nothing else: the same answers write the
/// same files however a voice came to be offered. [`Enrolment`] is the other axis, and the
/// only one that changes *what an answer writes* -- which is exactly why the override is a
/// field of its own instead of a third flag on `Offer`. There are three files an answer
/// can land in now (`speakers.json`, a session's `speaker_names.json`, and its transcript),
/// and which of the first two it is depends on the voice's duration and on this.
pub fn run_enroll(
    paths: &Paths,
    requested: &[SessionId],
    rules: EnrollRules<'_>,
    interviewer: &mut dyn Interviewer,
    notes: &mut dyn Narrator,
) -> Result<EnrollReport> {
    let mut report = EnrollReport::default();

    // Enforced here rather than in the CLI's argument parser, because this is where the sibling
    // rule already lives -- a requested id that is not on disk is printed and counted below --
    // and because one enforcement point cannot disagree with itself. Refused before anything is
    // discovered: a run that cannot say which session it is about has nothing to read.
    if requested.len() != 1
        && let Some(selection) = rules.selector.clone()
    {
        notes.note(Note::Run(RunNote::SelectionNeedsOneSession { selection }))?;
        report.failed += 1;
        return Ok(report);
    }

    // The assertion is a fact about *one* session's speaker track, so it needs the same
    // guarantee, enforced beside the selector's for the same reason. It stands in for the queue
    // and its gates alike, so a selector passed beside it is ignored rather than composed with:
    // pointing at one voice and asserting the whole track are two different requests, and the
    // CLI refuses to take both at once.
    if rules.one_speaker.is_some() && requested.len() != 1 {
        notes.note(Note::Run(RunNote::OneSpeakerNeedsOneSession))?;
        report.failed += 1;
        return Ok(report);
    }

    // Trimmed and checked here rather than in the CLI, so a library caller wiring up the rules
    // by hand gets the same protection: an empty assertion would name every voice "", and there
    // is no such thing. Reported as a request not served rather than dropped silently, for the
    // reason `OneSpeakerIsEmpty` gives.
    let mut rules = rules;
    if let Some(name) = rules.one_speaker {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            notes.note(Note::Run(RunNote::OneSpeakerIsEmpty))?;
            report.failed += 1;
            return Ok(report);
        }
        rules.one_speaker = Some(trimmed);
    }

    // An answer supplied up front is never shown the voice it lands on, so a queue would put one
    // name on everybody in it. Refused here, beside the guard above, for the same reason: this is
    // the one place that can see both the answerer and the selection, and a library caller
    // wiring up a [`GivenName`] gets the same protection the CLI does.
    //
    // Not refused under an assertion: the assertion selects every voice in the session itself,
    // so a name waiting for a voice has all of them at once, and the answerer is never
    // consulted at all -- nothing is asked about any voice, which is the mode's whole point.
    if rules.selector.is_none() && rules.one_speaker.is_none() && interviewer.needs_one_voice() {
        notes.note(Note::Run(RunNote::NameNeedsAVoice))?;
        report.failed += 1;
        return Ok(report);
    }

    let discovered = discover_sessions(paths)?;

    for id in requested {
        if !discovered.iter().any(|session| &session.id == id) {
            notes.note(Note::Run(RunNote::SessionNotFound { id }))?;
            report.failed += 1;
        }
    }

    let selected: Vec<&DiscoveredSession> = if requested.is_empty() {
        discovered.iter().collect()
    } else {
        discovered
            .iter()
            .filter(|session| requested.contains(&session.id))
            .collect()
    };

    if selected.is_empty() && requested.is_empty() {
        notes.note(Note::Run(RunNote::NoSessionsFound {
            dir: &paths.sessions_dir(),
        }))?;
        return Ok(report);
    }

    let mut speakers = EnrolledSpeakers::read_or_empty(paths)?;

    for session in selected {
        match enroll_session(
            paths,
            session,
            rules.clone(),
            &mut speakers,
            interviewer,
            notes,
            &mut report,
        )? {
            Outcome::Finished => {}
            Outcome::Quit => break,
        }
    }

    Ok(report)
}
/// The sequencing and the writes, exercised without a terminal and without an audio device.
///
/// Every test below drives [`run_enroll`] against a scripted answerer over real session
/// directories on a temporary disk. What is *not* decidable here is whether a human can name
/// a colleague from what a prompt shows -- the audio, the snippet length, the wording -- which
/// needs a real recording and a real person.
#[cfg(test)]
mod tests;
