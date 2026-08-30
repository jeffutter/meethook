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
mod interview;
mod meeting;
mod narration;
mod prompt;
mod queue;
mod references;
mod resolve;
mod session;

pub use consequence::{Assertion, Consequence, Preview, Refusal};
pub use forget::{Confirm, Forgotten, Removal, Target, run_forget};
pub use interview::{Answer, GivenName, Interviewer, MeetingLabel};
pub use meethook_session::Stored;
pub use meeting::{Labelled, MeetingChoice, MeetingSource, Relabelling, run_meeting};
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
mod tests {
    use std::cell::Cell;
    use std::collections::VecDeque;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;

    use meethook_session::{
        Attendee, AttendeeStatus, EnrolledSpeaker, MAX_REFERENCES_PER_SPEAKER, Meeting, MeetingFit,
        RepresentativeSegment, SPEAKER_YOU, SessionMetadata, SessionPaths, SourceTrack,
        SpeakerCluster, SpeakerClusters, SpeakerNames, Stored, TRANSCRIPT_SCHEMA_VERSION,
        TrackSync, Transcript, TranscriptContext, Turn, unknown_labels, unknown_speaker,
    };
    // The cut the ranking is deliberately *not* made at, named rather than spelled 0.40, so
    // the fixtures below still mean "outside identification's reach" if it moves.
    use meethook_transcribe::{Attribution, IDENTIFY_DISTANCE, Resemblance, identify_clusters};

    use super::*;

    /// One row of the queue a prompt was shown, owned so it can outlive the call.
    ///
    /// The whole [`Attribution`] rather than its label, because what a queue pane needs is the
    /// basis as well as the name -- "identified at 0.91" and "named for this session" are two
    /// different rows however identically they read.
    #[derive(Debug, PartialEq)]
    struct Row {
        number: String,
        attribution: Attribution,
        speech_seconds: f64,
        below_floor: bool,
    }

    /// A voice recorded exactly as it was shown, so a test can assert on what the user would
    /// have been looking at rather than only on what they answered.
    #[derive(Debug, PartialEq)]
    struct Shown {
        session: String,
        /// The meeting the prompt was told this session was recorded during -- or that it was
        /// not labelled with one at all. The only way a test can check that the value crosses
        /// the Interviewer seam rather than being re-read from `session.json` behind it.
        meeting: Option<MeetingLabel>,
        /// Which of this session's questions this was, and how many there were, exactly as the
        /// prompt was handed it.
        position: Position,
        /// What the prompt was told this voice is called and on what basis -- which is the only
        /// way a test can check that a correction prompt asked "is this right" rather than
        /// "who is this", and that a voice named for one session says so.
        attribution: Attribution,
        /// The handle the prompt was given, which is the only way a test can check that it does
        /// not move when the voice is named.
        number: String,
        speech_seconds: f64,
        /// Every voice of the session as the prompt was shown them, so a test can check both
        /// what a queue pane would hold and that it is current.
        queue: Vec<Row>,
        snippets: Vec<String>,
        /// Each snippet's `(start, duration)`, so a test can prove the prompt was handed track
        /// time rather than timeline time -- the one failure no assertion about text can see.
        snippet_times: Vec<(f64, f64)>,
        /// How many samples each snippet carried, which is what says the audio was cut from
        /// the stretch those times name.
        snippet_samples: Vec<usize>,
        clip_samples: usize,
        /// Who the prompt was told this voice resembles, in the order it was handed them --
        /// which is the only way a test can check that an [`Interviewer`] can offer names
        /// without ever reading `speakers.json`.
        resembles: Vec<Resemblance>,
        /// Every enrolled name the prompt was handed -- the universe [`resolve()`] requires,
        /// which is not the same list as `resembles`.
        enrolled: Vec<String>,
    }

    impl Shown {
        fn label(&self) -> &str {
            self.attribution.label()
        }

        fn confidence(&self) -> Option<f32> {
            self.attribution.confidence()
        }

        /// The queue as a pane would list it: the handle, what the row reads as, and whether
        /// the floor held it back. For the assertions that are about the shape of the queue
        /// rather than about the basis of one row.
        fn rows(&self) -> Vec<(&str, &str, bool)> {
            self.queue
                .iter()
                .map(|row| {
                    (
                        row.number.as_str(),
                        row.attribution.label(),
                        row.below_floor,
                    )
                })
                .collect()
        }

        /// The ranking as a prompt would list it: who, and how many recordings of them.
        fn offered(&self) -> Vec<(&str, usize)> {
            self.resembles
                .iter()
                .map(|r| (r.name.as_str(), r.references))
                .collect()
        }
    }

    /// An interviewer that answers from a queue and remembers every voice it was asked about.
    /// Answers past the end of the script are skips, so a test that expects no prompt at all
    /// fails on `seen` rather than on a panic somewhere else.
    #[derive(Default)]
    struct Scripted {
        answers: VecDeque<Answer>,
        seen: Vec<Shown>,
        /// How many more stalled passes this answerer claims to still be working through. A
        /// countdown rather than a flag so that a test which gets the arithmetic wrong fails
        /// instead of hanging: once it reaches zero the session ends however the script reads.
        working_passes: Cell<usize>,
    }

    impl Scripted {
        fn answering(answers: Vec<Answer>) -> Scripted {
            Scripted {
                answers: answers.into(),
                seen: Vec::new(),
                working_passes: Cell::new(0),
            }
        }

        /// Say "still working" for the next `passes` stalled passes and finished after that,
        /// standing in for the cursor of a full-screen frame.
        fn working_for(self, passes: usize) -> Scripted {
            Scripted {
                working_passes: Cell::new(passes),
                ..self
            }
        }

        fn labels(&self) -> Vec<&str> {
            self.seen.iter().map(Shown::label).collect()
        }

        /// The positions as the user reads them, through [`Display`] rather than as a pair, so
        /// an assertion covers the form on the screen and not only the two numbers.
        fn positions(&self) -> Vec<String> {
            self.seen.iter().map(|v| v.position.to_string()).collect()
        }
    }

    impl Interviewer for Scripted {
        fn identify(&mut self, voice: &Voice<'_>) -> Answer {
            self.seen.push(Shown {
                session: voice.session.to_string(),
                meeting: voice.meeting.cloned(),
                position: voice.position,
                attribution: voice.attribution.clone(),
                number: voice.number.to_string(),
                speech_seconds: voice.speech_seconds,
                queue: voice
                    .queue
                    .iter()
                    .map(|row| Row {
                        number: row.number.to_string(),
                        attribution: row.attribution.clone(),
                        speech_seconds: row.speech_seconds,
                        below_floor: row.below_floor,
                    })
                    .collect(),
                snippets: voice.snippets.iter().map(|s| s.text.to_string()).collect(),
                snippet_times: voice
                    .snippets
                    .iter()
                    .map(|s| (s.start, s.duration))
                    .collect(),
                snippet_samples: voice.snippets.iter().map(|s| s.audio.len()).collect(),
                clip_samples: voice.clip.len(),
                resembles: voice.resembles.clone(),
                enrolled: voice.enrolled.iter().map(|n| n.to_string()).collect(),
            });
            self.answers.pop_front().unwrap_or(Answer::Skip)
        }

        fn still_working(&self) -> bool {
            let left = self.working_passes.get();
            self.working_passes.set(left.saturating_sub(1));
            left > 0
        }
    }

    fn named(name: &str) -> Answer {
        Answer::Named {
            name: name.to_string(),
            anyway: false,
        }
    }

    /// The same answer, insisted on: honour it even where it takes a name off a voice the user
    /// was not asked about. Only [`Refusal::Taken`] is in reach, which is what the veto tests
    /// below use this to pin.
    fn named_anyway(name: &str) -> Answer {
        Answer::Named {
            name: name.to_string(),
            anyway: true,
        }
    }

    /// A distinct unit vector per cluster id, so enrolling one of these voices matches that
    /// cluster and nobody else's.
    pub(crate) fn voice(id: u32) -> Vec<f32> {
        let mut embedding = vec![0.0f32; 4];
        embedding[id as usize % 4] = 1.0;
        embedding
    }

    /// A unit vector `degrees` away from cluster 0's, for the fixtures that are about how
    /// close two voices are: one person clustering split in two, or one reference that matches
    /// both halves. 0.35 of cosine distance is `IDENTIFY_DISTANCE`, so 49 degrees is the edge.
    pub(crate) fn nearly(degrees: f32) -> Vec<f32> {
        let radians = degrees.to_radians();
        vec![radians.cos(), radians.sin(), 0.0, 0.0]
    }

    fn cluster(id: u32, first_spoke: f64, representative: (f64, f64)) -> SpeakerCluster {
        SpeakerCluster {
            id,
            embedding: voice(id),
            speech_seconds: 10.0 + f64::from(id),
            first_spoke_seconds: first_spoke,
            heard_at_once_with: Vec::new(),
            representatives: vec![RepresentativeSegment {
                start: representative.0,
                end: representative.1,
            }],
        }
    }

    /// `cluster` is the voice the turn came from, exactly as `merge` would have recorded it,
    /// and `speaker` is what that voice was called when the transcript was written. The two
    /// have to agree for a fixture to mean anything: the tests below read a label off the
    /// file and expect the cluster underneath it to be the one they named.
    pub(crate) fn speaker_turn(start: f64, cluster: u32, speaker: &str, text: &str) -> Turn {
        Turn {
            speaker: speaker.to_string(),
            start,
            end: start + 1.0,
            text: text.to_string(),
            source_track: SourceTrack::Speaker,
            cluster: Some(cluster),
            speaker_id_confidence: None,
        }
    }

    pub(crate) fn mic_turn(start: f64, text: &str) -> Turn {
        Turn {
            speaker: SPEAKER_YOU.to_string(),
            start,
            end: start + 1.0,
            text: text.to_string(),
            source_track: SourceTrack::Mic,
            cluster: None,
            speaker_id_confidence: None,
        }
    }

    /// Six seconds of 16 kHz mono tone: real audio, so a clip sliced out of it has the
    /// samples a test can count.
    fn write_speaker_wav(path: &Path) {
        let samples: Vec<f32> = (0..16_000 * 6)
            .map(|i| (i as f32 / 16_000.0 * 440.0 * std::f32::consts::TAU).sin() * 0.3)
            .collect();
        write_clip(path, &samples).unwrap();
    }

    /// A transcribed two-voice session: cluster 0 speaks first, cluster 1 answers, and the
    /// local speaker is in there too so tests can prove the mic track is never touched.
    ///
    /// The transcript is written with the labels `transcribe` would have given it against an
    /// empty database, which is the state `enroll` is for.
    /// The `session.json` a fixture session carries.
    ///
    /// A real one rather than the `{}` placeholder this used to be: classification still only
    /// checks the file's presence, but re-rendering a `transcript.md` reads the session's start
    /// time and its meeting out of it.
    pub(crate) fn session_metadata(id: &SessionId) -> SessionMetadata {
        let sync = TrackSync {
            host_ticks: 1,
            timebase_numer: 125,
            timebase_denom: 3,
        };
        SessionMetadata::new(
            id.clone(),
            "2026-08-09T05:26:00Z".parse().unwrap(),
            sync,
            sync,
        )
    }

    /// Writes both transcript files the way `transcribe` does: through whatever template the
    /// root resolves to.
    ///
    /// Going through [`TranscriptTemplate::resolve`] rather than always taking the built-in is
    /// what lets a test drop a `transcript.md.jinja` into the root and have the fixture itself
    /// honour it, exactly as the CLI does.
    pub(crate) fn write_transcript(
        transcript: &Transcript,
        paths: &Paths,
        session: &SessionPaths,
        metadata: &SessionMetadata,
    ) {
        transcript
            .write(
                session,
                &TranscriptTemplate::resolve(paths, None).unwrap(),
                &TranscriptContext::now(metadata),
            )
            .unwrap();
    }

    pub(crate) fn make_session(paths: &Paths, id: &str) -> SessionPaths {
        let id = SessionId::parse(id).unwrap();
        let session = paths.session(&id);
        std::fs::create_dir_all(session.dir()).unwrap();
        let metadata = session_metadata(&id);
        metadata.write(&session.session_json()).unwrap();
        write_speaker_wav(&session.speaker_wav());

        SpeakerClusters::new(
            id.clone(),
            vec![cluster(0, 0.0, (0.5, 2.5)), cluster(1, 3.0, (3.0, 5.0))],
        )
        .write(&session)
        .unwrap();

        write_transcript(
            &Transcript::new(
                id,
                vec![
                    speaker_turn(0.0, 0, "Unknown 1", "  hi there  "),
                    mic_turn(1.0, "morning"),
                    speaker_turn(3.0, 1, "Unknown 2", "and from me"),
                    speaker_turn(4.0, 0, "Unknown 1", "let us start"),
                ],
            ),
            paths,
            &session,
            &metadata,
        );

        session
    }

    /// One voice worth naming and three fragments under the floor, which is the shape real
    /// clustering leaves a meeting in: a handful of speakers and a tail of turns too short
    /// for any distance rule to place.
    fn make_fragmented_session(paths: &Paths, id: &str) -> SessionPaths {
        let session = make_session(paths, id);
        let parsed = SessionId::parse(id).unwrap();

        let mut clusters = vec![
            cluster(0, 0.0, (0.5, 2.5)),
            cluster(1, 3.0, (3.0, 5.0)),
            cluster(2, 3.5, (1.0, 2.0)),
            cluster(3, 4.5, (2.0, 3.0)),
        ];
        for (cluster, seconds) in clusters.iter_mut().zip([40.0, 1.5, 0.9, 2.0]) {
            cluster.speech_seconds = seconds;
        }
        SpeakerClusters::new(parsed.clone(), clusters)
            .write(&session)
            .unwrap();

        write_transcript(
            &Transcript::new(
                parsed.clone(),
                vec![
                    speaker_turn(0.0, 0, "Unknown 1", "hi there"),
                    mic_turn(1.0, "morning"),
                    speaker_turn(3.0, 1, "Unknown 2", "and from me"),
                    speaker_turn(3.5, 2, "Unknown 3", "mm"),
                    speaker_turn(4.5, 3, "Unknown 4", "yes"),
                ],
            ),
            paths,
            &session,
            &session_metadata(&parsed),
        );

        session
    }

    fn run(paths: &Paths, ids: &[&str], interviewer: &mut Scripted) -> (EnrollReport, String) {
        run_asking(paths, ids, Offer::default(), interviewer)
    }

    /// `run`, with the widening flags exposed. Separate so that the dozen tests that have
    /// nothing to do with the floor or with corrections do not carry an [`Offer`] each.
    fn run_asking(
        paths: &Paths,
        ids: &[&str],
        offer: Offer,
        interviewer: &mut Scripted,
    ) -> (EnrollReport, String) {
        run_enrolling(paths, ids, offer, Enrolment::default(), interviewer)
    }

    /// `run_asking`, with the write-side override exposed too. Separate again for the same
    /// reason: only the tests about what an answer *writes* care which of the two it is.
    fn run_enrolling(
        paths: &Paths,
        ids: &[&str],
        offer: Offer,
        enrolment: Enrolment,
        interviewer: &mut Scripted,
    ) -> (EnrollReport, String) {
        run_over(
            paths,
            ids,
            None,
            offer,
            visits(offer),
            enrolment,
            interviewer,
        )
    }

    /// Which sessions the CLI visits for a given [`Offer`], so the helpers above stay the plain
    /// command: both halves come off `--correct` there, and a test that wants them apart -- the
    /// one about [`Sessions`] being separately decidable -- goes through [`run_over`] directly.
    fn visits(offer: Offer) -> Sessions {
        if offer.named {
            Sessions::Every
        } else {
            Sessions::Unresolved
        }
    }

    /// `run`, aimed at one voice. One helper per axis, like the two above, so that the tests
    /// that do not target a voice keep their short signature -- and a default [`Offer`], since
    /// the point of a selector is that it needs no flags to reach a voice.
    fn run_targeting(
        paths: &Paths,
        ids: &[&str],
        voice: &str,
        interviewer: &mut Scripted,
    ) -> (EnrollReport, String) {
        let selector = VoiceSelector::from(voice);
        run_over(
            paths,
            ids,
            Some(Selection::Voice(selector)),
            Offer::default(),
            // Irrelevant beside a selector, which stands in for the queue and its gates alike.
            Sessions::default(),
            Enrolment::default(),
            interviewer,
        )
    }

    /// `run_targeting`'s sibling, aimed at whoever was speaking at one moment. `at` is written
    /// exactly as a user would copy it off `transcript.md`, so the tests exercise the spelling
    /// as well as the lookup.
    fn run_at(
        paths: &Paths,
        ids: &[&str],
        at: &str,
        interviewer: &mut dyn Interviewer,
    ) -> (EnrollReport, String) {
        run_over(
            paths,
            ids,
            Some(Selection::At(at.parse().unwrap())),
            Offer::default(),
            Sessions::default(),
            Enrolment::default(),
            interviewer,
        )
    }

    /// The whole non-interactive command: a moment, and the name of whoever was speaking then.
    fn run_naming_at(paths: &Paths, ids: &[&str], at: &str, name: &str) -> (EnrollReport, String) {
        run_at(paths, ids, at, &mut GivenName::new(name))
    }

    #[allow(clippy::too_many_arguments)]
    fn run_over(
        paths: &Paths,
        ids: &[&str],
        selection: Option<Selection>,
        offer: Offer,
        sessions: Sessions,
        enrolment: Enrolment,
        interviewer: &mut dyn Interviewer,
    ) -> (EnrollReport, String) {
        let requested: Vec<SessionId> =
            ids.iter().map(|id| SessionId::parse(id).unwrap()).collect();
        let mut out = Vec::new();
        let report = run_enroll(
            paths,
            &requested,
            EnrollRules {
                selector: selection,
                offer,
                sessions,
                enrolment,
                relabel_transcript: true,
                one_speaker: None,
                // Resolved from the root, exactly as the CLI does, so a test that puts a
                // template there is testing the path a user takes.
                template: &TranscriptTemplate::resolve(paths, None).unwrap(),
            },
            interviewer,
            &mut Lines::new(&mut out),
        )
        .unwrap();
        (report, String::from_utf8(out).unwrap())
    }

    /// The database `enroll` would have written by naming these clusters, so a test can start
    /// from "the wrong person is already on this voice" without running a first pass.
    pub(crate) fn enrolled(entries: &[(&str, Vec<f32>)], paths: &Paths) {
        EnrolledSpeakers::new(
            entries
                .iter()
                .map(|(name, embedding)| EnrolledSpeaker {
                    name: name.to_string(),
                    embedding: embedding.clone(),
                    clip_seconds: None,
                })
                .collect(),
        )
        .write(paths)
        .unwrap();
    }

    /// `--correct` on its own: reach the already-named voices, leave the floor where it is.
    const CORRECT: Offer = Offer {
        quiet: false,
        named: true,
    };

    /// `--all` on its own: reach the quiet voices. Since [`PROMPT_FLOOR_SECONDS`] and
    /// [`REFERENCE_FLOOR_SECONDS`] are the same number, this is also the only flag that
    /// reaches a voice quiet enough for an answer to be recorded against the session alone.
    const ALL: Offer = Offer {
        quiet: true,
        named: false,
    };

    /// `--all --correct`: the only way back to a voice already named for its session, which is
    /// by construction both named *and* under the prompt floor, so either flag alone misses it.
    const ALL_AND_CORRECT: Offer = Offer {
        quiet: true,
        named: true,
    };

    /// Rewrites this session's clusters with the talk times given, ids in order, leaving
    /// first appearances and representatives as [`make_session`] wrote them.
    ///
    /// The fixture's default is `10.0 + id`, which clears the floor for every voice; the
    /// floor tests are the ones that need to say otherwise.
    fn with_speech_seconds(session: &SessionPaths, seconds: &[f64]) {
        let mut clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
        for (cluster, seconds) in clusters.clusters.iter_mut().zip(seconds) {
            cluster.speech_seconds = *seconds;
        }
        clusters.write(session).unwrap();
    }

    /// Rewrites this session's transcript, leaving its clusters and its metadata as
    /// [`make_session`] wrote them. Both files, through the same template `transcribe` uses, so
    /// the timestamps a test then points at are the ones `transcript.md` actually prints.
    ///
    /// The timestamp tests are the ones that need a timeline other than the fixture's four turns
    /// in its first five seconds.
    fn with_turns(paths: &Paths, session: &SessionPaths, id: &str, turns: Vec<Turn>) {
        let parsed = SessionId::parse(id).unwrap();
        write_transcript(
            &Transcript::new(parsed.clone(), turns),
            paths,
            session,
            &session_metadata(&parsed),
        );
    }

    /// Rewrites this session's cluster embeddings, ids in order, leaving everything else as
    /// [`make_session`] wrote it. The fixture's default is one orthogonal vector per cluster;
    /// the tests about near voices are the ones that need to say otherwise.
    pub(crate) fn with_embeddings(session: &SessionPaths, embeddings: &[Vec<f32>]) {
        let mut clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
        for (cluster, embedding) in clusters.clusters.iter_mut().zip(embeddings) {
            cluster.embedding = embedding.clone();
        }
        clusters.write(session).unwrap();
    }

    /// Records that segmentation heard these two voices speaking at once.
    ///
    /// That relation is the one piece of evidence proving two clusters are different people,
    /// and it is what the heard-at-once veto acts on -- so it is also the one way an answer can
    /// still cost another voice its name once references accumulate rather than replace.
    /// Written on both sides, as `speaker_clusters.json` documents it.
    pub(crate) fn heard_at_once(session: &SessionPaths, a: u32, b: u32) {
        let mut clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
        for cluster in &mut clusters.clusters {
            if cluster.id == a {
                cluster.heard_at_once_with.push(b);
            } else if cluster.id == b {
                cluster.heard_at_once_with.push(a);
            }
        }
        clusters.write(session).unwrap();
    }

    /// A unit vector on one axis of an `axes`-wide space: every pair of these is orthogonal, so
    /// no two of them can ever be matched to one another however many references pile up.
    /// [`voice`] is the same idea fixed at four dimensions.
    pub(crate) fn axis(which: usize, axes: usize) -> Vec<f32> {
        let mut embedding = vec![0.0f32; axes];
        embedding[which] = 1.0;
        embedding
    }

    pub(crate) fn transcript_of(session: &SessionPaths) -> Transcript {
        Transcript::read(&session.transcript_json()).unwrap()
    }

    /// This session's hand-given names as they stand on disk, which is where an answer to a
    /// voice too quiet for a reference goes instead of into `speakers.json`.
    pub(crate) fn assigned_in(session: &SessionPaths, id: &str) -> SpeakerNames {
        SpeakerNames::read_or_empty(session, &SessionId::parse(id).unwrap()).unwrap()
    }

    /// Every file under a directory, by path and by contents, so a comparison covers a file
    /// created or removed as well as one rewritten.
    ///
    /// Here rather than in one test module because "wrote nothing" is a claim two commands make,
    /// and it has to mean the same thing in both: byte-for-byte over the whole root rather than
    /// over the files each was expected to touch.
    pub(crate) fn files_under(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
        fn walk(dir: &Path, into: &mut Vec<(PathBuf, Vec<u8>)>) {
            let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
                .unwrap()
                .map(|e| e.unwrap().path())
                .collect();
            entries.sort();
            for path in entries {
                if path.is_dir() {
                    walk(&path, into);
                } else {
                    into.push((path.clone(), std::fs::read(&path).unwrap()));
                }
            }
        }
        let mut files = Vec::new();
        walk(root, &mut files);
        files
    }

    /// Turns as (speaker, text, confidence), which is what a reader of the transcript sees.
    pub(crate) fn said(transcript: &Transcript) -> Vec<(&str, &str, Option<f32>)> {
        transcript
            .turns
            .iter()
            .map(|t| (t.speaker.as_str(), t.text.as_str(), t.speaker_id_confidence))
            .collect()
    }

    /// Acceptance criteria #5 and #6, at the level a user meets them: one answer puts a
    /// person in the database and their name on their own turns, and on nobody else's.
    ///
    /// It also pins the thing a rename must never do, which is change the *shape* of a
    /// transcript it did not write. The root carries a template here, in place before the
    /// session is, so the rewrite below has something other than the built-in default to revert
    /// to if it ever resolved the template from anywhere but the root.
    #[test]
    fn naming_a_voice_enrolls_them_and_rewrites_that_sessions_transcript() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        std::fs::create_dir_all(paths.root()).unwrap();
        std::fs::write(paths.transcript_template(), USER_TEMPLATE).unwrap();
        let session = make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        assert_eq!(speakers.speakers[0].name, "Alice");
        assert_eq!(speakers.speakers[0].embedding, voice(0));

        assert_eq!(
            said(&transcript_of(&session)),
            [
                ("Alice", "  hi there  ", Some(1.0)),
                ("You", "morning", None),
                ("Unknown 2", "and from me", None),
                ("Alice", "let us start", Some(1.0)),
            ]
        );
        // The rendering is rewritten from the turns, not patched line by line.
        let markdown = std::fs::read_to_string(session.transcript_md()).unwrap();
        assert_eq!(
            markdown,
            transcript_of(&session)
                .render_markdown(
                    &TranscriptTemplate::resolve(&paths, None).unwrap(),
                    &TranscriptContext::now(&session_metadata(
                        &SessionId::parse("20260809-052600").unwrap()
                    )),
                )
                .unwrap()
        );
        assert!(markdown.contains("Alice"), "{markdown}");
        assert!(!markdown.contains("Unknown 1"), "{markdown}");
        // Acceptance criterion #5: the rewrite went through the root's template, not the
        // built-in default. Both marks, because either alone would pass on a default rendering
        // that happened to be a prefix or a suffix of this one.
        assert!(
            markdown.starts_with("---\nvault: mine\n---\n"),
            "{markdown}"
        );
        assert!(markdown.contains("Alice> let us start\n"), "{markdown}");
        assert!(!markdown.contains("**["), "{markdown}");
        // The captions are rewritten by the same call, so they cannot be left naming a voice
        // the transcript beside them no longer calls a stranger. The user's template has no
        // say here: `transcript.vtt` is a machine format.
        let vtt = std::fs::read_to_string(session.transcript_vtt()).unwrap();
        assert_eq!(vtt, transcript_of(&session).render_vtt());
        assert!(vtt.contains("<v Alice>let us start\n"), "{vtt}");
        assert!(!vtt.contains("Unknown 1"), "{vtt}");
    }

    /// A template that is nothing like the built-in default in either half: different
    /// frontmatter, and a body line no default rendering could produce.
    const USER_TEMPLATE: &str = "---\nvault: mine\n---\n\
        {% for turn in turns %}{{ turn.speaker }}> {{ turn.text }}\n{% endfor %}";

    /// Acceptance criterion #6's actual claim, which the assertion above only illustrates:
    /// the rewritten transcript is what `transcribe --force` would now produce. Checked by
    /// deriving the labels the way `merge` does -- `unknown_labels` over the clusters,
    /// `identify_clusters` against the database -- rather than by restating the expected
    /// strings, so the two paths cannot drift without this failing.
    #[test]
    fn the_rewritten_transcript_is_what_a_force_re_transcribe_would_produce() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("Alice"), named("Bob")]);
        run(&paths, &[], &mut interviewer);

        let clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        let identified = identify_clusters(&clusters.clusters, &speakers);
        let unknown = unknown_labels(
            clusters
                .clusters
                .iter()
                .map(|c| (c.id, c.first_spoke_seconds)),
        );
        // The transcript's speaker turns, in order, are cluster 0, 1, 0.
        let expected: Vec<(String, Option<f32>)> = [0u32, 1, 0]
            .iter()
            .map(|id| match identified.get(id) {
                Some(who) => (who.name.clone(), Some(who.similarity)),
                None => (unknown[id].clone(), None),
            })
            .collect();

        let written: Vec<(String, Option<f32>)> = transcript_of(&session)
            .turns
            .iter()
            .filter(|t| t.source_track == SourceTrack::Speaker)
            .map(|t| (t.speaker.clone(), t.speaker_id_confidence))
            .collect();
        assert_eq!(written, expected);
    }

    /// The guard on `merge` staying the sole producer of a turn's provenance: `enroll` changes
    /// what a cluster is called and never which cluster a turn came from. That is what keeps
    /// a rewritten transcript identical to a `--force` re-transcribe, since the field would
    /// otherwise be one `enroll` could drift.
    #[test]
    fn a_rewrite_leaves_every_turns_cluster_exactly_as_it_was() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        let before: Vec<Option<u32>> = transcript_of(&session)
            .turns
            .iter()
            .map(|t| t.cluster)
            .collect();

        let mut interviewer = Scripted::answering(vec![named("Alice"), named("Bob")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 2, "{output}");
        let after: Vec<Option<u32>> = transcript_of(&session)
            .turns
            .iter()
            .map(|t| t.cluster)
            .collect();
        assert_eq!(after, before);
        assert_eq!(before, [Some(0), None, Some(1), Some(0)]);
    }

    /// The compatibility decision on `TRANSCRIPT_SCHEMA_VERSION`, at the level a user meets
    /// it: a transcript written before turns recorded their cluster is refused rather than
    /// read with that provenance fabricated, it says how to fix it, and the session after it
    /// is still asked about.
    #[test]
    fn a_transcript_without_clusters_fails_its_session_without_ending_the_queue() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let stale = make_session(&paths, "20260809-052600");
        make_session(&paths, "20260809-052700");
        std::fs::write(
            stale.transcript_json(),
            br#"{
              "schema_version": 1,
              "session_id": "20260809-052600",
              "turns": [
                {
                  "speaker": "Unknown 1",
                  "start": 0.0,
                  "end": 1.0,
                  "text": "hi there",
                  "source_track": "speaker",
                  "speaker_id_confidence": null
                }
              ]
            }"#,
        )
        .unwrap();

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.failed, 1, "{output}");
        assert!(output.contains("--force"), "{output}");
        assert_eq!(report.named, 1, "{output}");
        for voice in &interviewer.seen {
            assert_eq!(voice.session, "20260809-052700", "{voice:?}");
        }
    }

    /// Acceptance criterion #7: a skip changes nothing, and "nothing" is byte-for-byte. A
    /// rewrite that happened to produce equivalent turns would still churn the files.
    #[test]
    fn skipping_every_voice_leaves_the_files_byte_identical() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        let before = (
            std::fs::read(session.transcript_json()).unwrap(),
            std::fs::read(session.transcript_md()).unwrap(),
            std::fs::read(session.speaker_clusters_json()).unwrap(),
        );

        let mut interviewer = Scripted::answering(vec![Answer::Skip, Answer::Skip]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.skipped, 2, "{output}");
        assert_eq!(report.named, 0);
        assert_eq!(
            (
                std::fs::read(session.transcript_json()).unwrap(),
                std::fs::read(session.transcript_md()).unwrap(),
                std::fs::read(session.speaker_clusters_json()).unwrap(),
            ),
            before
        );
        assert!(
            !paths.speakers_json().exists(),
            "a run that named nobody must not create a database"
        );
        assert!(
            !session.speaker_names_json().exists(),
            "a run that named nobody must not create a names file either"
        );
    }

    /// Acceptance criterion #4, and the boundary the clusters file exists to defend: enroll
    /// reads it and never writes it, so nothing here can start depending on a name being in
    /// there.
    #[test]
    fn a_run_that_names_everybody_still_leaves_the_clusters_file_untouched() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        let before = std::fs::read(session.speaker_clusters_json()).unwrap();

        let mut interviewer = Scripted::answering(vec![named("Alice"), named("Bob")]);
        run(&paths, &[], &mut interviewer);

        assert_eq!(
            std::fs::read(session.speaker_clusters_json()).unwrap(),
            before
        );
    }

    /// Acceptance criterion #1, and the deduplication rule: the same person in two sessions is
    /// asked about once, because the second session identifies them from the answer given in
    /// the first. Sessions are worked through in id order.
    #[test]
    fn a_person_named_in_one_session_is_matched_rather_than_asked_about_again() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let first = make_session(&paths, "20260809-052600");
        let second = make_session(&paths, "20260809-052700");

        // One name, then skips: whoever is asked about after Alice is somebody else.
        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        let sessions: Vec<&str> = interviewer
            .seen
            .iter()
            .map(|v| v.session.as_str())
            .collect();
        assert_eq!(
            sessions,
            ["20260809-052600", "20260809-052600", "20260809-052700"],
            "expected both voices of the first session, then the second session's other voice"
        );
        assert_eq!(
            interviewer.labels(),
            ["Unknown 1", "Unknown 2", "Unknown 2"],
            "the second session's Alice must not be asked about again"
        );

        // ...and her name reaches the second session's transcript anyway, on the way past.
        for session in [&first, &second] {
            assert_eq!(
                transcript_of(session).turns[0].speaker,
                "Alice",
                "in {}",
                session.dir().display()
            );
        }
    }

    /// TASK-026 acceptance criteria #1 and #2: every prompt says which voice it is of how
    /// many, and that total is the number the session line printed just above the questions.
    /// Asserted together, because the whole value of the number is that it agrees with what
    /// the user was told a moment ago.
    #[test]
    fn every_prompt_says_which_voice_it_is_of_how_many() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::default();
        let (_, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.positions(), ["1/2", "2/2"], "{output}");
        assert!(output.contains("2 unresolved voice(s)"), "{output}");
    }

    /// TASK-026 acceptance criteria #4 and #6: a run over several sessions counts each session
    /// separately, and the session on the same prompt says which one a position belongs to.
    ///
    /// The second session's total is 1 rather than 2 because Alice is identified out of its
    /// queue before any question is asked -- which is acceptance criterion #2 from the other
    /// direction: the total is whatever that session actually offered.
    #[test]
    fn positions_restart_in_each_session_of_a_run() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        make_session(&paths, "20260809-052700");

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (_, output) = run(&paths, &[], &mut interviewer);

        let sessions: Vec<&str> = interviewer
            .seen
            .iter()
            .map(|v| v.session.as_str())
            .collect();
        assert_eq!(
            sessions,
            ["20260809-052600", "20260809-052600", "20260809-052700"],
            "{output}"
        );
        assert_eq!(interviewer.positions(), ["1/2", "2/2", "1/1"], "{output}");
        assert!(
            output.contains("20260809-052700  1 unresolved voice(s)"),
            "{output}"
        );
    }

    /// TASK-026 acceptance criterion #3, and the decision behind it made assertable: a voice an
    /// earlier answer in the same run named is passed over, and its number goes with it. The
    /// positions read 1/4, 2/4, 4/4 -- a gap in the middle and a total that does not shrink --
    /// because the total is what the session line promised and the gap is a question that
    /// answered itself.
    #[test]
    fn a_voice_an_earlier_answer_named_leaves_a_gap_in_the_positions() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_fragmented_session(&paths, "20260809-052600");
        // Clusters 0 and 2 are one person that clustering split in two, so naming the first
        // names the third on the way past.
        with_embeddings(&session, &[nearly(0.0), voice(1), nearly(20.0), voice(3)]);

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (_, output) = run_asking(&paths, &[], ALL, &mut interviewer);

        assert_eq!(
            interviewer.labels(),
            ["Unknown 1", "Unknown 2", "Unknown 4"],
            "Unknown 3 is Alice, already named by the first answer: {output}"
        );
        assert_eq!(interviewer.positions(), ["1/4", "2/4", "4/4"], "{output}");
        assert!(output.contains("4 unresolved voice(s)"), "{output}");
    }

    /// Acceptance criterion #8: nothing to ask about is passed over silently rather than
    /// prompting, and so is a session nobody has transcribed yet.
    #[test]
    fn sessions_with_nothing_to_ask_about_are_passed_over_without_prompting() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());

        // Already fully identified.
        let resolved = make_session(&paths, "20260809-052600");
        EnrolledSpeakers::new(vec![
            EnrolledSpeaker {
                name: "Alice".to_string(),
                embedding: voice(0),
                clip_seconds: None,
            },
            EnrolledSpeaker {
                name: "Bob".to_string(),
                embedding: voice(1),
                clip_seconds: None,
            },
        ])
        .write(&paths)
        .unwrap();

        // Recorded but never transcribed.
        let untranscribed = paths.session(&SessionId::parse("20260809-052700").unwrap());
        std::fs::create_dir_all(untranscribed.dir()).unwrap();
        std::fs::write(untranscribed.session_json(), b"{}").unwrap();

        // The recorder died mid-session.
        let orphan = paths.session(&SessionId::parse("20260809-052800").unwrap());
        std::fs::create_dir_all(orphan.dir()).unwrap();

        let mut interviewer = Scripted::default();
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert!(interviewer.seen.is_empty(), "{:?}", interviewer.seen);
        assert_eq!(report.passed_over, 3, "{output}");
        assert_eq!(report.failed, 0, "{output}");
        assert!(output.contains("nothing unresolved"), "{output}");
        // A session where everybody is already named is the one somebody is looking at when
        // one of those names is wrong, and this line is all it prints.
        assert!(
            output.contains("2 named voice(s) -- meethook enroll --correct"),
            "a correction nobody is told how to reach is not reachable: {output}"
        );
        assert!(output.contains("not transcribed yet"), "{output}");
        assert!(output.contains("no session.json"), "{output}");
        // Nobody was asked, and the transcript still caught up with the database: a session
        // where everyone is already known is exactly the one that would otherwise be passed
        // over on every future run, keeping its stale labels for good.
        assert_eq!(
            said(&transcript_of(&resolved))
                .iter()
                .map(|(speaker, _, _)| *speaker)
                .collect::<Vec<_>>(),
            ["Alice", "You", "Bob", "Alice"]
        );
        assert!(output.contains("brought up to date"), "{output}");
    }

    /// Gives a fixture session the meeting label the recorder's lookup would have written,
    /// with the fit given: `make_session` writes sessions without meetings, so the label is
    /// attached by rewriting `session.json`, the way the `meeting.rs` fixtures do.
    fn labelled_meeting(paths: &Paths, id: &str, fit: MeetingFit) {
        let session = paths.session(&SessionId::parse(id).unwrap());
        let metadata = session_metadata(&SessionId::parse(id).unwrap()).with_meeting(Some(
            Meeting::new(
                "EVENT-1".to_owned(),
                "Incident review".to_owned(),
                "Work".to_owned(),
                "2026-08-09T05:20:00Z".parse().unwrap(),
                "2026-08-09T06:20:00Z".parse().unwrap(),
            )
            .with_fit(fit),
        ));
        metadata.write(&session.session_json()).unwrap();
    }

    /// The one display shape, pinned byte for byte over every fit: the title alone when the
    /// fit states it plainly, the title plus the fit's own caveat otherwise.
    #[test]
    fn a_meeting_label_states_a_strong_fit_plainly_and_qualifies_the_rest() {
        for fit in [
            MeetingFit::Started,
            MeetingFit::StartedEarly,
            MeetingFit::Confirmed,
        ] {
            let label = MeetingLabel {
                title: "Standup".to_owned(),
                fit,
            };
            assert_eq!(label.clause(), "Standup", "{fit:?}");
        }
        assert_eq!(
            MeetingLabel {
                title: "Standup".to_owned(),
                fit: MeetingFit::JoinedLate,
            }
            .clause(),
            "Standup  (uncertain: the recording began after this meeting had started)"
        );
        assert_eq!(
            MeetingLabel {
                title: "Standup".to_owned(),
                fit: MeetingFit::AfterEnd,
            }
            .clause(),
            "Standup  (uncertain: the recording began after this meeting had ended)"
        );
        assert_eq!(
            MeetingLabel {
                title: "Standup".to_owned(),
                fit: MeetingFit::Unknown,
            }
            .clause(),
            "Standup  (unverified: this session was recorded before meethook scored the match)"
        );
    }

    /// Acceptance criteria #1, #2 and #5, over every fit: the queue announcement names the
    /// meeting once, under the count line, unqualified when the fit is strong and with the
    /// same caveat `meethook record` prints when it is not -- and the two voices then
    /// prompted about do not repeat it.
    #[test]
    fn the_queue_line_names_the_meeting_once_and_qualifies_it_as_record_does() {
        for fit in MeetingFit::ALL {
            let root = tempfile::tempdir().unwrap();
            let paths = Paths::new(root.path());
            make_session(&paths, "20260809-052600");
            labelled_meeting(&paths, "20260809-052600", fit);

            let mut interviewer = Scripted::answering(vec![named("Alice"), named("Aaron")]);
            let (_, output) = run(&paths, &[], &mut interviewer);

            let head = "20260809-052600  2 unresolved voice(s)\n    meeting   Incident review";
            if let Some(caveat) = fit.caveat() {
                assert!(
                    output.contains(&format!("{head}  ({caveat})\n")),
                    "the weak fit must carry its caveat: {fit:?}: {output}"
                );
            } else {
                assert!(
                    output.contains(&format!("{head}\n")),
                    "the strong fit must be stated plainly: {fit:?}: {output}"
                );
            }
            // Once, with the session, not once per voice: both answers land afterwards and
            // neither may name the meeting again.
            assert_eq!(
                output.matches("Incident review").count(),
                1,
                "the meeting is named once, where the session is announced: {fit:?}: {output}"
            );
        }
    }

    /// The sub-line sits under the held-back clause too, which is the other shape the count
    /// line takes: a meeting is not only named on the plain path.
    #[test]
    fn the_queue_line_sits_under_the_held_back_clause_as_well() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_fragmented_session(&paths, "20260809-052600");
        labelled_meeting(&paths, "20260809-052600", MeetingFit::Started);

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (_, output) = run(&paths, &[], &mut interviewer);

        assert!(
            output.contains(
                "20260809-052600  1 unresolved voice(s), 3 quieter voice(s) not offered -- \
                 meethook enroll --all\n    meeting   Incident review\n"
            ),
            "{output}"
        );
    }

    /// Acceptance criterion #3: a meeting carrying everything that must not reach a terminal
    /// leaks none of it -- the queue line is the title and the fit, and nothing off the
    /// roster.
    #[test]
    fn the_queue_line_leaks_nothing_off_the_roster() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        let session = paths.session(&SessionId::parse("20260809-052600").unwrap());
        let metadata = session_metadata(&SessionId::parse("20260809-052600").unwrap())
            .with_meeting(Some(
                Meeting::new(
                    "EVENT-1".to_owned(),
                    "Incident review".to_owned(),
                    "Work".to_owned(),
                    "2026-08-09T05:20:00Z".parse().unwrap(),
                    "2026-08-09T06:20:00Z".parse().unwrap(),
                )
                .with_people(
                    Some(Attendee {
                        name: Some("Alan Turing".to_owned()),
                        email: Some("alan@example.com".to_owned()),
                        status: AttendeeStatus::Accepted,
                        is_you: false,
                    }),
                    vec![Attendee {
                        name: Some("Grace Hopper".to_owned()),
                        email: Some("grace@example.com".to_owned()),
                        status: AttendeeStatus::Accepted,
                        is_you: true,
                    }],
                )
                .with_invite(
                    Some("https://example.com/j/12345".to_owned()),
                    Some("Babbage Room, 12 Ada Street".to_owned()),
                    Some("Dial-in 555-0100, passcode 481516".to_owned()),
                )
                .with_fit(MeetingFit::JoinedLate),
            ));
        metadata.write(&session.session_json()).unwrap();

        let mut interviewer = Scripted::answering(vec![named("Alice"), named("Aaron")]);
        let (_, output) = run(&paths, &[], &mut interviewer);

        assert!(output.contains("Incident review"), "{output}");
        for secret in [
            "Turing",
            "Hopper",
            "@",
            "Babbage",
            "Ada Street",
            "example.com",
            "Dial-in",
            "555-0100",
            "passcode",
            "481516",
        ] {
            assert!(
                !output.contains(secret),
                "the queue line leaks {secret:?}: {output}"
            );
        }
    }

    /// Acceptance criterion #4, the absence half: a session with no meeting says nothing
    /// about meetings -- no reserved row, no empty label. The word does not appear anywhere
    /// else in enroll's output, so its absence is the whole claim; the byte-identity itself
    /// is pinned by `one_runs_narration_reads_as_these_lines_in_this_order`, whose fixtures
    /// carry no meetings.
    #[test]
    fn a_session_without_a_meeting_says_nothing_about_meetings() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("Alice"), named("Aaron")]);
        let (_, output) = run(&paths, &[], &mut interviewer);

        assert!(!output.contains("meeting"), "{output}");
    }

    /// TASK-051.02 acceptance criterion #6: the meeting reaches an interface across the
    /// Interviewer seam -- every voice of a labelled session is handed the same title and fit,
    /// and a session without one is handed `None` rather than a value an interface could only
    /// have gotten by reading `session.json` behind the seam's back.
    #[test]
    fn the_seam_hands_every_voice_the_meeting_it_was_recorded_during() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        labelled_meeting(&paths, "20260809-052600", MeetingFit::JoinedLate);

        let mut interviewer = Scripted::answering(vec![named("Alice"), named("Aaron")]);
        run(&paths, &[], &mut interviewer);

        let expected = Some(MeetingLabel {
            title: "Incident review".to_owned(),
            fit: MeetingFit::JoinedLate,
        });
        assert_eq!(interviewer.seen.len(), 2, "both voices were asked about");
        for seen in &interviewer.seen {
            assert_eq!(
                seen.meeting, expected,
                "the seam carries the label, per voice"
            );
        }

        // The absent half: the common case hands `None`, which is what lets a surface reserve
        // nothing for a title that is not there.
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("Alice"), named("Aaron")]);
        run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.seen.len(), 2, "both voices were asked about");
        for seen in &interviewer.seen {
            assert_eq!(
                seen.meeting, None,
                "no meeting means no label, not an empty one"
            );
        }
    }

    /// Acceptance criterion #2: ids scope the run, and one that is not on disk is named
    /// rather than quietly doing less than was asked.
    #[test]
    fn ids_scope_the_run_and_an_unknown_id_is_reported() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        let untouched = make_session(&paths, "20260809-052700");

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(
            &paths,
            &["20260809-052600", "20260809-999999"],
            &mut interviewer,
        );

        assert!(output.contains("20260809-999999  not found"), "{output}");
        assert_eq!(report.failed, 1);
        assert_eq!(report.named, 1);
        for voice in &interviewer.seen {
            assert_eq!(voice.session, "20260809-052600", "{voice:?}");
        }
        assert_eq!(transcript_of(&untouched).turns[0].speaker, "Unknown 1");
    }

    /// Acceptance criterion #9: ending the run early keeps everything already answered. The
    /// name given before the quit is on disk in both files, and nothing after it was asked.
    #[test]
    fn quitting_keeps_every_name_accepted_so_far() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let first = make_session(&paths, "20260809-052600");
        let second = make_session(&paths, "20260809-052700");

        let mut interviewer = Scripted::answering(vec![named("Alice"), Answer::Quit]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(interviewer.seen.len(), 2, "{:?}", interviewer.seen);

        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        assert_eq!(transcript_of(&first).turns[0].speaker, "Alice");
        assert!(
            std::fs::read_to_string(first.transcript_md())
                .unwrap()
                .contains("Alice")
        );
        // The queue stopped where it was told to, rather than carrying on to the next session.
        assert_eq!(transcript_of(&second).turns[0].speaker, "Unknown 1");
    }

    /// Replaces `naming_someone_already_enrolled_replaces_their_reference`, which asserted the
    /// v1 rule this ticket retires: one row per name, the second recording overwriting the
    /// first. Overwriting is what made naming a second voice cost the first one its name, and
    /// TASK-027.01 measured it as the *worst* of the three candidate policies on both corpora.
    ///
    /// Typing a name already in the database now adds another recording of that person: both
    /// rows survive, in enrollment order, and the line says how many they hold.
    #[test]
    fn naming_someone_already_enrolled_adds_a_reference_rather_than_replacing_one() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        // Alice, enrolled from a voice that matches neither cluster here.
        EnrolledSpeakers::new(vec![EnrolledSpeaker {
            name: "Alice".to_string(),
            embedding: voice(3),
            clip_seconds: None,
        }])
        .write(&paths)
        .unwrap();

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert!(
            output.contains("enrolled another recording of Alice: 2 reference(s) now"),
            "{output}"
        );
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        let stored: Vec<(&str, &[f32])> = speakers
            .speakers
            .iter()
            .map(|s| (s.name.as_str(), s.embedding.as_slice()))
            .collect();
        assert_eq!(
            stored,
            [
                ("Alice", voice(3).as_slice()),
                ("Alice", voice(0).as_slice())
            ],
            "the first recording must survive the second"
        );
    }

    /// Re-answering the same voice with the same name must not spend a capped reference slot on
    /// information already held -- the common way to reach this being a second `--correct` pass
    /// over a session that was enrolled from it in the first place.
    #[test]
    fn re_answering_a_voice_with_the_name_it_already_gave_stores_nothing_new() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        enrolled(&[("Alice", voice(0))], &paths);

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert!(
            output.contains("Alice already has a reference built from this voice"),
            "{output}"
        );
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.references("Alice"), 1, "{:?}", speakers.speakers);
    }

    /// Acceptance criterion #3 and the queue order: each prompt carries that voice's own
    /// lines and its own clip, and they arrive in "Unknown N" order rather than in talk-time
    /// order.
    ///
    /// Cluster 0 is the first to speak and cluster 1 the second, so the labels below are also
    /// the assertion that first-appearance order is what the queue follows.
    #[test]
    fn each_prompt_carries_that_voices_snippets_and_clip_in_unknown_order() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::default();
        run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"]);
        assert_eq!(
            interviewer.seen[0].snippets,
            ["hi there", "let us start"],
            "only this voice's lines, whitespace trimmed"
        );
        assert_eq!(interviewer.seen[1].snippets, ["and from me"]);
        assert_eq!(interviewer.seen[0].speech_seconds, 10.0);
        // The representative spans 0.5 s to 2.5 s of a 16 kHz track.
        assert_eq!(interviewer.seen[0].clip_samples, 32_000);
        assert_eq!(interviewer.seen[1].clip_samples, 32_000);
    }

    /// Acceptance criterion #11: no audio is not a failed session. The prompt still happens,
    /// still carries the snippets, and an answer still lands on disk.
    #[test]
    fn a_session_with_no_speaker_wav_is_still_asked_about_with_an_empty_clip() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        std::fs::remove_file(session.speaker_wav()).unwrap();

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.failed, 0, "{output}");
        assert_eq!(interviewer.seen[0].clip_samples, 0);
        assert_eq!(interviewer.seen[0].snippets, ["hi there", "let us start"]);
        assert_eq!(
            interviewer.seen[0].snippet_samples,
            [0, 0],
            "and no audio under any line either, with the times still saying when they were said"
        );
        assert_eq!(interviewer.seen[0].snippet_times, [(0.0, 1.0), (4.0, 1.0)]);
        assert_eq!(transcript_of(&session).turns[0].speaker, "Alice");
    }

    /// A representative that runs off the end of the track -- a truncated `speaker.wav` -- is
    /// clipped to what is there rather than refused, for the same reason as above.
    #[test]
    fn a_representative_past_the_end_of_the_track_plays_what_is_there() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        let id = SessionId::parse("20260809-052600").unwrap();
        SpeakerClusters::new(
            id,
            vec![
                cluster(0, 0.0, (5.0, 90.0)),
                cluster(1, 3.0, (600.0, 620.0)),
            ],
        )
        .write(&session)
        .unwrap();

        let mut interviewer = Scripted::default();
        run(&paths, &[], &mut interviewer);

        // The track is six seconds long: one second of the first clip survives, none of the
        // second.
        assert_eq!(interviewer.seen[0].clip_samples, 16_000);
        assert_eq!(interviewer.seen[1].clip_samples, 0);
    }

    /// Acceptance criterion #2 end to end, which is the half no unit test can see: that the run
    /// actually reads the offset out of `session.json` rather than defaulting it to zero.
    ///
    /// The fixture's speaker track starts a second after the microphone's, so every snippet's
    /// track time is its turn's timeline second less one -- and the audio under it is a second
    /// earlier in `speaker.wav` than a run that ignored the offset would have cut.
    #[test]
    fn a_session_whose_speaker_track_started_late_still_lines_the_audio_up() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        let metadata = with_speaker_offset(&session, "20260809-052600", 1.0);
        // Every turn after the speaker track's own start, so that what this test measures is
        // the offset and not the clamp that catches a turn from before it.
        write_transcript(
            &Transcript::new(
                SessionId::parse("20260809-052600").unwrap(),
                vec![
                    speaker_turn(2.0, 0, "Unknown 1", "hi there"),
                    mic_turn(2.5, "morning"),
                    speaker_turn(3.0, 1, "Unknown 2", "and from me"),
                    speaker_turn(5.0, 0, "Unknown 1", "let us start"),
                ],
            ),
            &paths,
            &session,
            &metadata,
        );

        let mut interviewer = Scripted::default();
        let (_, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
        assert_eq!(
            interviewer.seen[0].snippet_times,
            [(1.0, 1.0), (4.0, 1.0)],
            "the turns are at 2 s and 5 s on the timeline, and the speaker track began at 1 s"
        );
        assert_eq!(interviewer.seen[1].snippet_times, [(2.0, 1.0)]);
        assert_eq!(interviewer.seen[0].snippet_samples, [16_000, 16_000]);
        assert_eq!(interviewer.seen[1].snippet_samples, [16_000]);
        // The clip is untouched by the offset: a representative's seconds are already track
        // time, which is exactly the confusion this ticket exists to keep apart.
        assert_eq!(interviewer.seen[0].clip_samples, 32_000);
    }

    /// Rewrites a fixture's `session.json` so that its speaker track begins `seconds` after its
    /// microphone track, which is what `speaker_offset_seconds` reads.
    ///
    /// Separate from [`session_metadata`] so that the default fixture -- both tracks starting
    /// together, which is what every other test here assumes -- does not move.
    fn with_speaker_offset(session: &SessionPaths, id: &str, seconds: f64) -> SessionMetadata {
        let id = SessionId::parse(id).unwrap();
        let base = session_metadata(&id);
        let mut speaker = base.speaker;
        // Ticks, not nanoseconds: `session.json` records the machine's rational timebase and
        // the arithmetic that reads it back is exact, so the fixture does the same conversion
        // in reverse rather than guessing at a tick.
        speaker.host_ticks += (seconds * 1e9 * f64::from(speaker.timebase_denom)
            / f64::from(speaker.timebase_numer)) as u64;
        let metadata = SessionMetadata::new(id, base.start_time, base.mic, speaker);
        metadata.write(&session.session_json()).unwrap();
        metadata
    }

    /// A session transcribed by a build that did not record first appearances cannot be
    /// mapped from "Unknown 2" back to a voice, so it is reported and counted -- and the
    /// session after it is still asked about.
    #[test]
    fn a_stale_clusters_file_fails_its_session_without_ending_the_queue() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let stale = make_session(&paths, "20260809-052600");
        make_session(&paths, "20260809-052700");
        std::fs::write(
            stale.speaker_clusters_json(),
            br#"{
              "schema_version": 1,
              "session_id": "20260809-052600",
              "clusters": [
                {
                  "id": 0,
                  "embedding": [1.0, 0.0, 0.0, 0.0],
                  "speech_seconds": 42.5,
                  "representatives": [{ "start": 1.0, "end": 3.0 }]
                }
              ]
            }"#,
        )
        .unwrap();

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.failed, 1, "{output}");
        assert!(output.contains("--force"), "{output}");
        assert_eq!(report.named, 1, "{output}");
        for voice in &interviewer.seen {
            assert_eq!(voice.session, "20260809-052700", "{voice:?}");
        }
    }

    /// A blank answer is a skip, not an entry called "". Somebody pressing Enter with a stray
    /// space in the buffer must not end up in the database.
    #[test]
    fn a_blank_name_is_a_skip_rather_than_an_empty_entry() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("   "), named("  Bob  ")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.skipped, 1, "{output}");
        assert_eq!(report.named, 1, "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        // Trimmed, so the transcript does not read "**[00:03]   Bob  :**".
        assert_eq!(speakers.speakers[0].name, "Bob");
    }

    /// One person clustering split in two is named once and lands on both halves, because
    /// that is what a `--force` re-transcribe would do with the reference this answer just
    /// stored.
    #[test]
    fn naming_a_split_voice_names_its_other_half_without_asking_twice() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        // Two clusters a few degrees apart: one voice the clusterer did not join up.
        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(
            interviewer.labels(),
            ["Unknown 1"],
            "the second half of one voice must not be asked about"
        );
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        assert_eq!(
            said(&transcript_of(&session))
                .iter()
                .map(|(speaker, _, _)| *speaker)
                .collect::<Vec<_>>(),
            ["Alice", "You", "Alice", "Alice"]
        );
    }

    /// The transcript body a re-render left on disk, below its frontmatter.
    ///
    /// Compared rather than the whole file because `updated` is the render instant, and two
    /// renderings a few microseconds apart can straddle a second boundary. The body is the
    /// half TASK-038 is about.
    fn markdown_body(session: &SessionPaths) -> String {
        let markdown = std::fs::read_to_string(session.transcript_md()).unwrap();
        markdown.split_once("\n---\n").unwrap().1.to_string()
    }

    /// TASK-038 acceptance criterion #6, the half where the rename does *not* merge anything:
    /// naming the voice between two runs of another leaves the blocks where they were, and the
    /// re-rendered file is what a fresh rendering of the relabelled turns produces.
    #[test]
    fn a_rename_that_merges_nothing_re_renders_the_same_blocks() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        let (report, output) = run_naming_at(&paths, &["20260809-052600"], "00:03", "Alice");
        assert_eq!(report.named, 1, "{output}");

        let transcript = transcript_of(&session);
        assert_eq!(
            said(&transcript)
                .iter()
                .map(|(speaker, _, _)| *speaker)
                .collect::<Vec<_>>(),
            ["Unknown 1", "You", "Alice", "Unknown 1"]
        );
        let body = markdown_body(&session);
        assert_eq!(
            body,
            transcript
                .render_markdown(
                    &TranscriptTemplate::resolve(&paths, None).unwrap(),
                    &TranscriptContext::now(&session_metadata(
                        &SessionId::parse("20260809-052600").unwrap()
                    )),
                )
                .unwrap()
                .split_once("\n---\n")
                .unwrap()
                .1
        );
        // Four speakers in a row, none of them repeating: four lines, as before collapsing.
        assert_eq!(body.trim_start().lines().count(), 4, "{body}");
        assert!(body.contains("**[00:03] Alice:** and from me\n"), "{body}");
    }

    /// TASK-038 acceptance criterion #6, the half that only collapsing can get wrong: naming a
    /// voice clustering had split in two puts one name on both halves, and where those halves
    /// are adjacent the re-render must print them as one block rather than the same name twice
    /// in a row -- which is what a fresh `transcribe` of the relabelled turns now produces.
    #[test]
    fn a_rename_that_makes_two_blocks_adjacent_merges_them_on_re_render() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        // One voice the clusterer did not join up, so naming it once names both halves.
        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);
        assert_eq!(report.named, 1, "{output}");

        let transcript = transcript_of(&session);
        assert_eq!(
            said(&transcript)
                .iter()
                .map(|(speaker, _, _)| *speaker)
                .collect::<Vec<_>>(),
            ["Alice", "You", "Alice", "Alice"]
        );
        let body = markdown_body(&session);
        assert_eq!(
            body,
            transcript
                .render_markdown(
                    &TranscriptTemplate::resolve(&paths, None).unwrap(),
                    &TranscriptContext::now(&session_metadata(
                        &SessionId::parse("20260809-052600").unwrap()
                    )),
                )
                .unwrap()
                .split_once("\n---\n")
                .unwrap()
                .1
        );
        // The last two turns were two blocks under two names before the rename and are one
        // block under one timestamp after it.
        assert_eq!(body.trim_start().lines().count(), 3, "{body}");
        assert!(
            body.contains("**[00:03] Alice:** and from me let us start\n"),
            "{body}"
        );
        assert!(!body.contains("Unknown"), "{body}");
    }

    /// TASK-019.03 acceptance criteria #1 and #2, which is the whole ticket in one test: a
    /// voice the database has named the wrong person is reached, corrected, and lands in both
    /// files -- and a later default run does not ask about it again.
    #[test]
    fn correcting_a_named_voice_updates_the_database_and_this_transcript() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        // Cluster 0 is enrolled under the wrong name.
        let mut first = Scripted::answering(vec![named("Alice"), named("Carol")]);
        run(&paths, &[], &mut first);

        let mut interviewer = Scripted::answering(vec![named("Bob")]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        // The question that was asked: a name, and how confident the claim behind it was.
        assert_eq!(interviewer.labels(), ["Alice", "Carol"], "{output}");
        assert_eq!(interviewer.seen[0].confidence(), Some(1.0), "{output}");
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.kept, 1, "{output}");

        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        let stored: Vec<(&str, &[f32])> = speakers
            .speakers
            .iter()
            .map(|s| (s.name.as_str(), s.embedding.as_slice()))
            .collect();
        assert_eq!(
            stored,
            [("Carol", voice(1).as_slice()), ("Bob", voice(0).as_slice())],
            "the corrected name owns this voice, and the wrong one no longer claims it"
        );
        assert_eq!(
            said(&transcript_of(&session))
                .iter()
                .map(|(speaker, _, _)| *speaker)
                .collect::<Vec<_>>(),
            ["Bob", "You", "Carol", "Bob"]
        );

        // ...and the correction sticks: a later default run has nothing to ask about.
        let mut again = Scripted::default();
        let (report, output) = run(&paths, &[], &mut again);
        assert!(again.seen.is_empty(), "{:?}", again.seen);
        assert_eq!(report.passed_over, 1, "{output}");
    }

    /// Acceptance criterion #3: reaching an already-named voice takes an explicit request. A
    /// default run over a half-identified session offers only the half nothing matched.
    #[test]
    fn a_default_run_still_asks_only_about_unresolved_voices() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        enrolled(&[("Alice", voice(0))], &paths);

        let mut default = Scripted::default();
        let (_, output) = run(&paths, &[], &mut default);
        assert_eq!(default.labels(), ["Unknown 2"], "{output}");
        assert!(output.contains("1 unresolved voice(s)"), "{output}");

        let mut correcting = Scripted::default();
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut correcting);
        assert_eq!(correcting.labels(), ["Alice", "Unknown 2"], "{output}");
        assert_eq!(correcting.seen[0].confidence(), Some(1.0), "{output}");
        assert_eq!(correcting.seen[1].confidence(), None, "{output}");
        assert!(
            output.contains("2 voice(s) to review, 1 of them already named"),
            "{output}"
        );
        assert_eq!(report.kept, 1, "{output}");
        assert_eq!(report.skipped, 1, "{output}");
    }

    /// Acceptance criterion #4's other half: pressing Enter on an already-named voice keeps
    /// that identification. The same nothing a skip writes -- byte for byte -- and counted
    /// apart from it, because a kept voice has a name and a skipped one does not.
    #[test]
    fn keeping_an_identification_writes_nothing_and_is_not_counted_as_a_skip() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        enrolled(&[("Alice", voice(0)), ("Bob", voice(1))], &paths);

        // A default run first, so the snapshot below is of a transcript already in step with
        // the database and any difference is the correcting run's doing.
        run(&paths, &[], &mut Scripted::default());
        let before = (
            std::fs::read(session.transcript_json()).unwrap(),
            std::fs::read(session.transcript_md()).unwrap(),
            std::fs::read(session.speaker_clusters_json()).unwrap(),
            std::fs::read(paths.speakers_json()).unwrap(),
        );

        // Enter, then Enter with a stray space in the buffer.
        let mut interviewer = Scripted::answering(vec![Answer::Skip, named("   ")]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(interviewer.labels(), ["Alice", "Bob"], "{output}");
        assert_eq!(report.kept, 2, "{output}");
        assert_eq!(report.skipped, 0, "{output}");
        assert_eq!(report.named, 0, "{output}");
        assert_eq!(
            (
                std::fs::read(session.transcript_json()).unwrap(),
                std::fs::read(session.transcript_md()).unwrap(),
                std::fs::read(session.speaker_clusters_json()).unwrap(),
                std::fs::read(paths.speakers_json()).unwrap(),
            ),
            before
        );
    }

    /// Acceptance criterion #5 under `--correct`, which is where it could regress: the in-run
    /// guard no longer looks at "is this named" alone, so the split-voice case has to be
    /// checked with the flag on as well as off.
    #[test]
    fn correcting_still_asks_once_about_one_voice_clustering_split_in_two() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(
            interviewer.labels(),
            ["Unknown 1"],
            "the second half of one voice must not be asked about: {output}"
        );
        assert_eq!(report.named, 1, "{output}");
    }

    /// Replaces `a_voice_an_answer_unnamed_is_still_asked_about`, which encoded the un-naming as
    /// *intended*: re-affirming cluster 0 re-anchored Alice's only reference onto it, cluster 1
    /// fell out of range, and the old test asserted it was re-prompted about as a question the
    /// run had created. Under a reference set the situation stops existing, and this one fixture
    /// flip -- the same clusters at 0 and 80 degrees, the same Alice at 40 -- is the clearest
    /// statement of what this ticket does.
    ///
    /// Adding a reference removes none, so Alice's 40-degree reference is still there and still
    /// names cluster 1. Nothing is un-named, so there is no second question.
    #[test]
    fn an_answer_no_longer_takes_the_name_off_the_other_half_of_a_voice() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        // 80 degrees apart, with Alice's reference sitting between them: inside
        // `IDENTIFY_DISTANCE` of both, and it stays that way now that answering cluster 0
        // appends a reference instead of moving the one that named cluster 1.
        with_embeddings(&session, &[nearly(0.0), nearly(80.0)]);
        enrolled(&[("Alice", nearly(40.0))], &paths);

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        // `--correct` offers named voices, so cluster 1 is still asked about -- but it is asked
        // about as *Alice*, with the confidence behind that, rather than as the "Unknown 2" the
        // answer used to have turned it into. That is the whole difference.
        assert_eq!(interviewer.labels(), ["Alice", "Alice"], "{output}");
        assert!(interviewer.seen[1].confidence().is_some(), "{output}");
        assert_eq!(
            report.skipped, 0,
            "no voice was left unnamed, so nothing was skipped: {output}"
        );
        assert_eq!(report.kept, 1, "{output}");
        assert_eq!(report.refused, 0, "{output}");
        // Both halves still read Alice in the transcript, and the database holds both
        // recordings of her rather than only the newer one.
        let said = transcript_of(&session);
        assert_eq!(said.turns[0].speaker, "Alice", "{output}");
        assert_eq!(said.turns[2].speaker, "Alice", "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.references("Alice"), 2, "{:?}", speakers.speakers);
    }

    /// One voice cannot be two people's stored reference. Correcting a voice enrolled under
    /// the wrong name leaves that name holding a reference built from somebody else's audio,
    /// which then competes as an exact tie in every future meeting -- and wins whenever it
    /// sorts first. Both orderings are checked, so the fix cannot be about the alphabet.
    #[test]
    fn correcting_a_voice_removes_the_reference_the_wrong_name_kept_of_it() {
        for correction in ["Ryan", "Aaron"] {
            let root = tempfile::tempdir().unwrap();
            let paths = Paths::new(root.path());
            let session = make_session(&paths, "20260809-052600");
            enrolled(&[("Nate", voice(0))], &paths);

            let mut interviewer = Scripted::answering(vec![named(correction)]);
            let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

            assert_eq!(report.named, 1, "{output}");
            assert!(
                output.contains(&format!(
                    "Nate no longer has a reference: that voice is {correction}"
                )),
                "an enrollment must not vanish without a line about it: {output}"
            );
            let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
            let stored: Vec<&str> = speakers.speakers.iter().map(|s| s.name.as_str()).collect();
            assert_eq!(stored, [correction], "{output}");
            assert_eq!(
                transcript_of(&session).turns[0].speaker,
                correction,
                "{output}"
            );
        }
    }

    /// A reference built from a *different* recording of the same person is a legitimate one
    /// and is left alone: only a reference identical to this cluster is a claim about a voice
    /// the user has just said is somebody else.
    #[test]
    fn correcting_a_voice_leaves_the_wrong_names_other_reference_alone() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        // Nate, enrolled from audio that is not either cluster here, matched to cluster 0 by
        // being merely close to it -- which is the false accept this ticket opens with.
        with_embeddings(
            &paths.session(&SessionId::parse("20260809-052600").unwrap()),
            &[nearly(0.0), nearly(80.0)],
        );
        enrolled(&[("Nate", nearly(20.0))], &paths);

        let mut interviewer = Scripted::answering(vec![named("Ryan")]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert!(!output.contains("no longer has a reference"), "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        let stored: Vec<(&str, &[f32])> = speakers
            .speakers
            .iter()
            .map(|s| (s.name.as_str(), s.embedding.as_slice()))
            .collect();
        assert_eq!(
            stored,
            [
                ("Nate", nearly(20.0).as_slice()),
                ("Ryan", nearly(0.0).as_slice())
            ],
            "Nate's own enrollment must survive somebody else's correction"
        );
    }

    /// The correction guarantee under a reference set, which is where it could have quietly
    /// stopped working: the wrong name loses the reference built from *this* voice and keeps the
    /// ones built from its own recordings, and the line says how many it has left rather than
    /// claiming it has none.
    #[test]
    fn correcting_one_of_several_references_leaves_the_others_and_says_how_many_remain() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        // Nate, with two recordings: one of them *is* cluster 0, the other is somebody's real
        // second meeting with him.
        enrolled(&[("Nate", voice(0)), ("Nate", voice(3))], &paths);

        let mut interviewer = Scripted::answering(vec![named("Ryan")]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert!(
            output.contains(
                "Nate no longer has that reference: that voice is Ryan -- Nate keeps 1 other(s)"
            ),
            "a person who lost one of three recordings has not lost their enrollment: {output}"
        );
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        let stored: Vec<(&str, &[f32])> = speakers
            .speakers
            .iter()
            .map(|s| (s.name.as_str(), s.embedding.as_slice()))
            .collect();
        assert_eq!(
            stored,
            [("Nate", voice(3).as_slice()), ("Ryan", voice(0).as_slice())],
            "only the reference built from the corrected voice goes"
        );
        assert_eq!(transcript_of(&session).turns[0].speaker, "Ryan", "{output}");
    }

    /// The defect TASK-027 was raised for, stated as the smallest case that showed it: two
    /// voices in one session given the same name. Under the old replacement rule the second
    /// answer overwrote the reference that had named the first, so cluster 0 dropped back to
    /// "Unknown 1" and its transcript was rewritten to say so -- silently, because the in-run
    /// guard then declined to ask about the voice it had just un-named.
    #[test]
    fn two_voices_in_one_session_given_one_name_both_keep_it() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("Alice"), named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 2, "{output}");
        assert_eq!(report.refused, 0, "{output}");
        let said = transcript_of(&session);
        assert_eq!(
            (
                said.turns[0].speaker.as_str(),
                said.turns[2].speaker.as_str()
            ),
            ("Alice", "Alice"),
            "neither answer may cost the other: {output}"
        );
        // The rendering the user actually reads, checked separately: the defect's visible
        // symptom was an "Unknown N" line in transcript.md about somebody already named, and
        // this session has only these two voices, so neither file may mention a stranger.
        let markdown = std::fs::read_to_string(session.transcript_md()).unwrap();
        assert!(
            !markdown.contains("Unknown"),
            "transcript.md still calls a named voice a stranger: {markdown}"
        );
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.references("Alice"), 2, "{:?}", speakers.speakers);
    }

    /// The gain the reference set was measured for, and the reason it is worth a schema bump:
    /// one person named in two meetings is recognised in a third that neither recording alone
    /// would have reached.
    ///
    /// Discriminating by construction. The third voice sits 10 degrees off the first recording
    /// and 50 off the second; `IDENTIFY_DISTANCE` is 0.35, and 50 degrees is 0.357 -- outside
    /// it. So under the old rule, where the second answer replaced the first, this voice would
    /// read "Unknown 1" and be asked about instead.
    #[test]
    fn a_person_named_in_two_sessions_is_recognised_in_a_third() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let first = make_session(&paths, "20260809-052600");
        let second = make_session(&paths, "20260809-052700");
        let third = make_session(&paths, "20260809-052800");
        // Each session's second voice is orthogonal to every reference, so nothing but Alice is
        // ever in play. The two recordings of Alice are 60 degrees apart -- far enough that the
        // second session asks about her rather than matching her to the first.
        with_embeddings(&first, &[nearly(0.0), voice(3)]);
        with_embeddings(&second, &[nearly(60.0), voice(3)]);
        with_embeddings(&third, &[nearly(10.0), voice(3)]);

        let mut interviewer = Scripted::answering(vec![
            named("Alice"),
            Answer::Skip,
            named("Alice"),
            Answer::Skip,
        ]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 2, "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.references("Alice"), 2, "{:?}", speakers.speakers);
        assert_eq!(
            transcript_of(&third).turns[0].speaker,
            "Alice",
            "the third session's voice is only within reach of the first recording: {output}"
        );
        assert!(
            !interviewer
                .seen
                .iter()
                .any(|v| v.session == "20260809-052800" && v.label() == "Unknown 1"),
            "a voice the database can already name must not be asked about: {:?}",
            interviewer.seen
        );
    }

    /// The walkthrough TASK-027 closes on, and the sharpest statement of the defect: name a
    /// voice, name the same person in another session, then go back and run `enroll` over the
    /// first session again. Its voice still reads her name, and its transcript is byte-identical
    /// -- not merely equivalent, because the bug's visible symptom was a transcript rewritten to
    /// say "Unknown 1" about somebody the user had already named.
    #[test]
    fn naming_a_person_again_elsewhere_leaves_the_first_sessions_transcript_untouched() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let first = make_session(&paths, "20260809-052600");
        let second = make_session(&paths, "20260809-052700");
        with_embeddings(&first, &[nearly(0.0), voice(3)]);
        with_embeddings(&second, &[nearly(60.0), voice(3)]);

        let mut interviewer = Scripted::answering(vec![
            named("Alice"),
            Answer::Skip,
            named("Alice"),
            Answer::Skip,
        ]);
        run(&paths, &[], &mut interviewer);
        let before = (
            std::fs::read(first.transcript_json()).unwrap(),
            std::fs::read(first.transcript_md()).unwrap(),
        );
        assert_eq!(transcript_of(&first).turns[0].speaker, "Alice");

        let mut again = Scripted::default();
        let (report, output) = run(&paths, &["20260809-052600"], &mut again);

        assert_eq!(
            (
                std::fs::read(first.transcript_json()).unwrap(),
                std::fs::read(first.transcript_md()).unwrap()
            ),
            before,
            "a second naming of Alice elsewhere must not rewrite this transcript: {output}"
        );
        assert_eq!(transcript_of(&first).turns[0].speaker, "Alice", "{output}");
        assert_eq!(report.refused, 0, "{output}");
        assert!(
            !output.contains("brought up to date"),
            "nothing changed, so nothing should have been rewritten: {output}"
        );
    }

    /// The growth cap. A person met in more rooms than meethook keeps recordings of gets the
    /// name in that transcript and no new reference -- and, crucially, loses none of the ones
    /// they have, because this recording is no better than any of them. Dropping the oldest
    /// would un-name a voice in some earlier session, which is the defect this whole ticket
    /// exists to end; only a *longer* recording displaces anything, which is the companion test.
    ///
    /// Every session here holds the same 10.0 s in the answered cluster, so the offer past the
    /// cap ties with the shortest held rather than beating it. Every voice is on its own axis,
    /// so no two are ever within reach of each other and each session really does have to ask.
    #[test]
    fn at_the_reference_cap_the_name_is_recorded_against_the_session_instead() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let axes = MAX_REFERENCES_PER_SPEAKER + 2;
        let sessions: Vec<SessionPaths> = (0..=MAX_REFERENCES_PER_SPEAKER)
            .map(|i| {
                let session = make_session(&paths, &format!("20260809-0526{i:02}"));
                with_embeddings(&session, &[axis(i, axes), axis(axes - 1, axes)]);
                session
            })
            .collect();

        let mut interviewer = Scripted::answering(
            sessions
                .iter()
                .flat_map(|_| [named("Alice"), Answer::Skip])
                .collect(),
        );
        let (report, output) = run(&paths, &[], &mut interviewer);

        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(
            speakers.references("Alice"),
            MAX_REFERENCES_PER_SPEAKER,
            "nothing already stored may be dropped to make room: {output}"
        );
        // The answer past the cap is still an answer: the transcript reads Alice, the name is
        // in that session's own file, and the line says why no reference was stored.
        let last = sessions.last().unwrap();
        assert_eq!(transcript_of(last).turns[0].speaker, "Alice", "{output}");
        let assigned = assigned_in(
            last,
            &format!("20260809-0526{MAX_REFERENCES_PER_SPEAKER:02}"),
        );
        assert_eq!(assigned.names.len(), 1, "{:?}", assigned.names);
        assert_eq!(assigned.names[0].name, "Alice");
        assert_eq!(report.named, sessions.len(), "{output}");
        assert_eq!(report.session_only, 1, "{output}");
        assert!(
            output.contains(&format!(
                "Alice already holds {MAX_REFERENCES_PER_SPEAKER} reference(s)"
            )),
            "{output}"
        );
        // The remedy is two commands rather than a file path: this is the line that used to send
        // people to a text editor, and both halves have to be on it -- `speakers` because the
        // line cannot know which reference should go, `forget` because that is what removes it.
        assert!(
            output.contains("meethook speakers shows what each of them is naming"),
            "the line has to say how to see what each recording is naming: {output}"
        );
        assert!(
            output.contains("meethook forget Alice --reference N removes the one you pick"),
            "the line has to name the command that makes room: {output}"
        );
        assert!(
            !output.contains(&paths.speakers_json().display().to_string()),
            "no remedy in this tool is a hand-edit of speakers.json any more: {output}"
        );
    }

    /// The companion to the cap: a *longer* recording of somebody full displaces the shortest
    /// one they hold, and says so. This is what makes a person's references get better with use
    /// rather than merely being whichever ten meethook happened to meet first.
    ///
    /// The last session carries 90.0 s where the ten before it carried 10.0 s, so the offer past
    /// the cap beats the shortest held rather than tying with it. The line has to name both
    /// lengths: something stored was dropped, and an enrollment that vanishes without a line
    /// about it is worse than the bug.
    #[test]
    fn past_the_cap_a_longer_recording_displaces_the_shortest_and_says_what_went() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let axes = MAX_REFERENCES_PER_SPEAKER + 2;
        let sessions: Vec<SessionPaths> = (0..=MAX_REFERENCES_PER_SPEAKER)
            .map(|i| {
                let session = make_session(&paths, &format!("20260809-0526{i:02}"));
                with_embeddings(&session, &[axis(i, axes), axis(axes - 1, axes)]);
                with_speech_seconds(
                    &session,
                    &[
                        if i == MAX_REFERENCES_PER_SPEAKER {
                            90.0
                        } else {
                            10.0
                        },
                        10.0,
                    ],
                );
                session
            })
            .collect();

        let mut interviewer = Scripted::answering(
            sessions
                .iter()
                .flat_map(|_| [named("Alice"), Answer::Skip])
                .collect(),
        );
        let (report, output) = run(&paths, &[], &mut interviewer);

        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(
            speakers.references("Alice"),
            MAX_REFERENCES_PER_SPEAKER,
            "the cap still holds: {output}"
        );
        assert!(
            speakers
                .speakers
                .iter()
                .any(|s| s.clip_seconds == Some(90.0)),
            "the longer recording is what is now stored: {:?}",
            speakers.speakers
        );
        assert_eq!(
            speakers
                .speakers
                .iter()
                .filter(|s| s.clip_seconds == Some(10.0))
                .count(),
            MAX_REFERENCES_PER_SPEAKER - 1,
            "exactly one of the ten should have gone: {:?}",
            speakers.speakers
        );
        assert_eq!(
            report.session_only, 0,
            "the answer stored a reference, so it is not a session-only name: {output}"
        );
        assert!(
            output.contains("enrolled a better recording of Alice: 90.0 s replaces the shortest"),
            "{output}"
        );
        assert!(
            output.contains("which was 10.0 s"),
            "the line has to say what was dropped, not just what was kept: {output}"
        );
    }

    /// The heard-at-once veto is the one way an answer can still cost an earlier name once
    /// references accumulate instead of replacing: segmentation heard these two voices at
    /// once, so they are not one person however certain the user is, and the veto has to refuse
    /// one of the two answers.
    ///
    /// What this ticket changes is that it is refused *out loud* and the earlier name is what
    /// survives. Before, the veto could take the earlier answer instead, and said nothing.
    #[test]
    fn an_answer_the_heard_at_once_veto_would_take_from_an_earlier_voice_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        heard_at_once(&session, 0, 1);

        let mut interviewer = Scripted::answering(vec![named("Alice"), named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.refused, 1, "{output}");
        assert!(
            output.contains(
                "refused Alice for Unknown 2: Unknown 1 already has that name and the two \
                 were heard speaking at once"
            ),
            "a refusal the user cannot read is a silent revert: {output}"
        );
        let said = transcript_of(&session);
        assert_eq!(
            (
                said.turns[0].speaker.as_str(),
                said.turns[2].speaker.as_str()
            ),
            ("Alice", "Unknown 2"),
            "the first answer is what survives: {output}"
        );
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.references("Alice"), 1, "{:?}", speakers.speakers);
    }

    /// The heard-at-once veto is unchanged in effect by references accumulating: a person is one
    /// contender for one name however many recordings back it, so two references of one person
    /// can never be awarded to two voices that overlap in time.
    ///
    /// Alice ends up holding two recordings, and the third session contains a voice matching
    /// each of them *exactly* -- at distance 0, so nothing but the veto can separate them --
    /// which segmentation heard talking over each other. One of the two gets the name.
    #[test]
    fn two_references_of_one_person_are_never_awarded_to_two_voices_heard_at_once() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let first = make_session(&paths, "20260809-052600");
        let second = make_session(&paths, "20260809-052700");
        let third = make_session(&paths, "20260809-052800");
        with_embeddings(&first, &[nearly(0.0), voice(3)]);
        with_embeddings(&second, &[nearly(60.0), voice(3)]);
        with_embeddings(&third, &[nearly(0.0), nearly(60.0)]);
        heard_at_once(&third, 0, 1);

        let mut interviewer = Scripted::answering(vec![
            named("Alice"),
            Answer::Skip,
            named("Alice"),
            Answer::Skip,
        ]);
        let (_, output) = run(&paths, &[], &mut interviewer);

        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.references("Alice"), 2, "{:?}", speakers.speakers);
        let said = transcript_of(&third);
        assert_eq!(
            (
                said.turns[0].speaker.as_str(),
                said.turns[2].speaker.as_str()
            ),
            ("Alice", "Unknown 2"),
            "one name cannot land on two voices heard at once, whatever backs it: {output}"
        );
    }

    /// Theft by argmax: a reference stored for one person can sit nearer to a third voice than
    /// that voice's own name's reference does, moving a name the user never asked about. Bob is
    /// 40 degrees from cluster 1 and holds it; Alice's new reference would be 20 degrees away
    /// and would win it. Refused, and nothing is written at all.
    #[test]
    fn an_answer_that_would_move_another_voices_name_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
        enrolled(&[("Bob", nearly(60.0))], &paths);
        let before = std::fs::read(paths.speakers_json()).unwrap();

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.refused, 1, "{output}");
        assert_eq!(report.named, 0, "{output}");
        assert!(
            output.contains("refused Alice for Unknown 1: it would take Bob off Unknown 2"),
            "{output}"
        );
        assert_eq!(
            std::fs::read(paths.speakers_json()).unwrap(),
            before,
            "a refused answer writes nothing"
        );
        let said = transcript_of(&session);
        assert_eq!(
            (
                said.turns[0].speaker.as_str(),
                said.turns[2].speaker.as_str()
            ),
            ("Unknown 1", "Bob"),
            "{output}"
        );
    }

    /// The third path to the same loss, and the one neither TASK-027 nor its plan noticed: a
    /// hand-given name beats an identification on a voice it overlaps, so naming a quiet
    /// fragment can drop that name off the voice that had it -- without any reference being
    /// stored or removed. Refused by the same check, which is why the check is at the label
    /// level rather than inside identification.
    #[test]
    fn naming_a_quiet_fragment_is_refused_when_it_would_unname_the_voice_that_has_that_name() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);
        heard_at_once(&session, 0, 1);
        enrolled(&[("Alice", voice(0))], &paths);

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

        assert_eq!(report.refused, 1, "{output}");
        assert_eq!(report.named, 0, "{output}");
        assert!(
            output.contains("refused Alice for Unknown 2: it would take Alice off Unknown 1"),
            "{output}"
        );
        assert!(
            assigned_in(&session, "20260809-052600").names.is_empty(),
            "a refused answer writes nothing"
        );
        assert_eq!(
            transcript_of(&session).turns[0].speaker,
            "Alice",
            "{output}"
        );
    }

    /// The other side of theft by argmax: the same answer, insisted on. An interface that showed
    /// the user which voice pays and what it loses before a key was pressed has removed the
    /// surprise the refusal exists to prevent, so the answer is honoured -- and everything a
    /// name ordinarily writes is written, this session's transcript included.
    #[test]
    fn naming_a_voice_anyway_takes_the_name_off_the_voice_that_had_it() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
        enrolled(&[("Bob", nearly(60.0))], &paths);
        let before = std::fs::read(paths.speakers_json()).unwrap();

        let mut interviewer = Scripted::answering(vec![named_anyway("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.refused, 0, "{output}");
        assert_eq!(report.named, 1, "{output}");
        assert!(
            output.contains(
                "named Alice for Unknown 1 anyway: Unknown 2 no longer reads Bob -- \
                 meethook enroll --correct --voice Unknown 2 to give it a name again"
            ),
            "the voice that paid has to be named where the run is read afterwards, not only in \
             the pane that warned about it: {output}"
        );
        assert_ne!(
            std::fs::read(paths.speakers_json()).unwrap(),
            before,
            "an honoured answer writes the name it was given"
        );
        let said = transcript_of(&session);
        assert_eq!(said.turns[0].speaker, "Alice", "{output}");
        assert_ne!(
            said.turns[2].speaker, "Bob",
            "the transcript has to agree with the cost that was accepted: {output}"
        );
    }

    /// The heard-at-once veto is not reachable from here however insistent the answer is.
    /// Segmentation *heard* these two voices at once and so proved they are different people;
    /// overriding that is the claim that several voices are one person, which is a different
    /// question with a ticket of its own. Byte for byte the refusal an ordinary answer gets.
    #[test]
    fn the_heard_at_once_veto_is_refused_however_insistent_the_answer_is() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        heard_at_once(&session, 0, 1);

        let mut interviewer =
            Scripted::answering(vec![named_anyway("Alice"), named_anyway("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.refused, 1, "{output}");
        assert!(
            output.contains(
                "refused Alice for Unknown 2: Unknown 1 already has that name and the two \
                 were heard speaking at once"
            ),
            "insisting must not change the sentence, let alone the outcome: {output}"
        );
        let said = transcript_of(&session);
        assert_eq!(
            (
                said.turns[0].speaker.as_str(),
                said.turns[2].speaker.as_str()
            ),
            ("Alice", "Unknown 2"),
            "{output}"
        );
    }

    /// The override is at the label level, like the check it overrides: it does not depend on
    /// which of the three mechanisms produced the loss. Here no reference is stored or removed
    /// at all -- a hand-given name on a quiet fragment simply beats the identification on the
    /// voice it overlaps -- and insisting takes Alice off the voice that had her all the same.
    #[test]
    fn the_quiet_fragment_path_can_be_overridden_too() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);
        heard_at_once(&session, 0, 1);
        enrolled(&[("Alice", voice(0))], &paths);

        let mut interviewer = Scripted::answering(vec![named_anyway("Alice")]);
        let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

        assert_eq!(report.refused, 0, "{output}");
        assert_eq!(report.named, 1, "{output}");
        assert!(
            output.contains("named Alice for Unknown 2 anyway: Unknown 1 no longer reads Alice"),
            "{output}"
        );
        assert_eq!(
            assigned_in(&session, "20260809-052600").names.len(),
            1,
            "an honoured answer records the name against the session: {output}"
        );
        assert_ne!(
            transcript_of(&session).turns[0].speaker,
            "Alice",
            "the voice that lost the name keeps it in the transcript otherwise: {output}"
        );
    }

    /// A name supplied up front never overrides anything. `--name` is never shown the voice it
    /// lands on -- which is why it needs a selector at all -- so it has certainly not been shown
    /// the third voice an override would cost, and the premise the override rests on does not
    /// hold for it.
    #[test]
    fn a_name_given_up_front_cannot_override_a_refusal() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
        enrolled(&[("Bob", nearly(60.0))], &paths);
        let before = std::fs::read(paths.speakers_json()).unwrap();

        let selector = VoiceSelector::from("Unknown 1");
        let (report, output) = run_over(
            &paths,
            // `--voice` needs the one session it is about, exactly as the CLI insists.
            &["20260809-052600"],
            Some(Selection::Voice(selector)),
            Offer::default(),
            Sessions::default(),
            Enrolment::default(),
            &mut GivenName::new("Alice"),
        );

        assert_eq!(report.refused, 1, "{output}");
        assert_eq!(report.named, 0, "{output}");
        assert!(
            output.contains("refused Alice for Unknown 1: it would take Bob off Unknown 2"),
            "{output}"
        );
        assert_eq!(
            std::fs::read(paths.speakers_json()).unwrap(),
            before,
            "a refused answer writes nothing"
        );
        assert_eq!(transcript_of(&session).turns[2].speaker, "Bob", "{output}");
    }

    /// A database written before this schema bump. References cannot be regenerated -- the audio
    /// they were built from may be long deleted -- so a v1 file must be migrated rather than
    /// refused: the names in it still identify their voices, and the file is upgraded by the
    /// next write rather than left claiming a version its contents no longer match.
    #[test]
    fn a_v1_database_still_names_its_voices_and_is_upgraded_by_the_next_write() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        // Written by hand at version 1, which is the only way to produce one now: the raw bytes
        // rather than a serialized struct, so that bumping the constant cannot quietly turn this
        // fixture into a current-version file and stop testing the migration.
        std::fs::write(
            paths.speakers_json(),
            b"{\n  \"schema_version\": 1,\n  \"speakers\": [\
              {\"name\": \"Alice\", \"embedding\": [1.0, 0.0, 0.0, 0.0]}]\n}\n"
                .as_slice(),
        )
        .unwrap();
        assert_eq!(voice(0), [1.0, 0.0, 0.0, 0.0], "the fixture is cluster 0");

        let mut interviewer = Scripted::answering(vec![named("Bob")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(
            transcript_of(&session).turns[0].speaker,
            "Alice",
            "a v1 name must survive the upgrade: {output}"
        );
        let on_disk = std::fs::read_to_string(paths.speakers_json()).unwrap();
        assert!(
            on_disk.contains(&format!(
                "\"schema_version\": {}",
                meethook_session::ENROLLED_SPEAKERS_SCHEMA_VERSION
            )),
            "{on_disk}"
        );
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.references("Alice"), 1, "{:?}", speakers.speakers);
        assert_eq!(speakers.references("Bob"), 1, "{:?}", speakers.speakers);
    }

    /// The v2 -> v3 half of the same guarantee. A v2 row carries no clip length, and the
    /// migration must leave it that way rather than inventing one: an unmeasured reference is
    /// never the row an eviction picks, and a zero written here would make it the first to go.
    #[test]
    fn a_v2_reference_keeps_its_name_and_gains_no_invented_clip_length() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        std::fs::write(
            paths.speakers_json(),
            b"{\n  \"schema_version\": 2,\n  \"speakers\": [\
              {\"name\": \"Alice\", \"embedding\": [1.0, 0.0, 0.0, 0.0]}]\n}\n"
                .as_slice(),
        )
        .unwrap();

        let mut interviewer = Scripted::answering(vec![named("Bob")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(
            transcript_of(&session).turns[0].speaker,
            "Alice",
            "a v2 name must survive the upgrade: {output}"
        );
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        let alice = speakers
            .speakers
            .iter()
            .find(|s| s.name == "Alice")
            .expect("Alice survives the migration");
        assert_eq!(alice.clip_seconds, None, "{:?}", speakers.speakers);
        let bob = speakers
            .speakers
            .iter()
            .find(|s| s.name == "Bob")
            .expect("Bob was just enrolled");
        assert!(
            bob.clip_seconds.is_some(),
            "a reference written now records what it was built from: {bob:?}"
        );
    }

    /// A database from a *newer* meethook cannot be read as though it were this one, and a run
    /// that ignored it would silently un-name everybody. Reported by name against the session,
    /// like every other unreadable file on this path, and the queue does not carry on into a
    /// second session naming nobody.
    #[test]
    fn a_database_from_a_newer_meethook_fails_the_run_by_name() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        std::fs::write(
            paths.speakers_json(),
            b"{\n  \"schema_version\": 99,\n  \"speakers\": []\n}\n",
        )
        .unwrap();

        let mut interviewer = Scripted::default();
        let mut out = Vec::new();
        let error = run_enroll(
            &paths,
            &[],
            EnrollRules {
                selector: None,
                offer: Offer::default(),
                sessions: Sessions::default(),
                enrolment: Enrolment::default(),
                relabel_transcript: true,
                one_speaker: None,
                template: &TranscriptTemplate::builtin(),
            },
            &mut interviewer,
            &mut Lines::new(&mut out),
        )
        .unwrap_err();

        assert!(error.to_string().contains("speakers.json"), "{error}");
        assert!(error.to_string().contains("upgrade meethook"), "{error}");
        assert!(
            interviewer.seen.is_empty(),
            "nothing may be asked against a database that could not be read"
        );
    }

    /// The prompt finds its lines by the cluster the turns came from, not by what they read.
    /// Two voices under one enrolled name is exactly the case a correction is for, and keyed
    /// on the label text both prompts would show the same person's words.
    #[test]
    fn each_correction_prompt_carries_only_its_own_voices_lines() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        // One reference matching both clusters: two voices, one name in the transcript.
        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
        enrolled(&[("Andrew", nearly(10.0))], &paths);

        let mut interviewer = Scripted::default();
        let (_, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(interviewer.labels(), ["Andrew", "Andrew"], "{output}");
        assert_eq!(interviewer.seen[0].snippets, ["hi there", "let us start"]);
        assert_eq!(interviewer.seen[1].snippets, ["and from me"]);
    }

    /// The two flags stay orthogonal: `--correct` reaches the named voices, the floor still
    /// decides which are worth a question, and only `--all` lifts it.
    #[test]
    fn correcting_does_not_lift_the_floor_and_all_does_not_reach_named_voices() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);
        enrolled(&[("Bob", voice(1))], &paths);

        let mut correcting = Scripted::default();
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut correcting);
        assert_eq!(correcting.labels(), ["Unknown 1"], "{output}");
        assert_eq!(report.held_back, 1, "{output}");
        assert!(output.contains("meethook enroll --all"), "{output}");

        let mut both = Scripted::default();
        let (report, output) = run_asking(
            &paths,
            &[],
            Offer {
                quiet: true,
                named: true,
            },
            &mut both,
        );
        assert_eq!(both.labels(), ["Unknown 1", "Bob"], "{output}");
        assert_eq!(report.held_back, 0, "{output}");
    }

    /// TASK-021 acceptance criterion #1, at the scale a unit test can hold it: a voice under
    /// [`PROMPT_FLOOR_SECONDS`] is not asked about, and the run says both how many it held
    /// back and how to get at them.
    #[test]
    fn a_voice_too_quiet_to_be_worth_a_question_is_not_asked_about() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1"], "{output}");
        assert_eq!(report.held_back, 1, "{output}");
        assert_eq!(report.named, 1, "{output}");
        assert!(
            output.contains("1 unresolved voice(s), 1 quieter voice(s) not offered"),
            "{output}"
        );
        assert!(
            output.contains("meethook enroll --all"),
            "a held-back voice nobody is told how to reach is not reachable: {output}"
        );
    }

    /// The escape the line above advertises actually reaches them, in the same
    /// first-appearance order the queue always follows.
    #[test]
    fn all_asks_about_the_voices_the_floor_holds_back() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);

        let mut interviewer = Scripted::default();
        let (report, output) = run_asking(
            &paths,
            &[],
            Offer {
                quiet: true,
                ..Offer::default()
            },
            &mut interviewer,
        );

        assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
        assert_eq!(report.held_back, 0, "{output}");
        assert!(!output.contains("not offered"), "{output}");
    }

    /// TASK-021 acceptance criterion #2, which is the one that matters: the floor filters
    /// *questions*. Nothing is merged, deleted, renumbered or re-attributed, so the clusters
    /// file is byte-identical and every held-back voice still reads the "Unknown N" it was
    /// written with -- while the voice that was named reads their name.
    #[test]
    fn holding_a_voice_back_changes_no_cluster_and_no_unknown_numbering() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_fragmented_session(&paths, "20260809-052600");
        let before = std::fs::read(session.speaker_clusters_json()).unwrap();

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1"], "{output}");
        assert_eq!(report.held_back, 3, "{output}");
        assert_eq!(
            std::fs::read(session.speaker_clusters_json()).unwrap(),
            before,
            "the floor must not touch the clustering"
        );
        assert_eq!(
            said(&transcript_of(&session))
                .iter()
                .map(|(speaker, _, _)| *speaker)
                .collect::<Vec<_>>(),
            ["Alice", "You", "Unknown 2", "Unknown 3", "Unknown 4"],
            "held-back voices keep the labels transcribe gave them"
        );
    }

    /// The proof that the floor is a filter on questions and not on labelling: one person
    /// clustering split into a large half and a fragment is named once, from the half that
    /// was offered, and the held-back half is relabelled with them -- exactly as a `--force`
    /// re-transcribe would do it.
    #[test]
    fn naming_an_offered_voice_still_relabels_its_held_back_half() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
        with_speech_seconds(&session, &[40.0, 1.5]);

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1"], "{output}");
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(
            said(&transcript_of(&session))
                .iter()
                .map(|(speaker, _, _)| *speaker)
                .collect::<Vec<_>>(),
            ["Alice", "You", "Alice", "Alice"],
            "the floor decides which voices are asked about, not which turns are labelled"
        );
    }

    /// A floor that hides every voice in a session would be a command that does nothing, so
    /// a recording where nobody clears it offers everybody. This is what keeps the
    /// end-to-end tests -- three seconds of synthesised audio apiece -- meaningful.
    #[test]
    fn a_session_where_nobody_clears_the_floor_offers_everybody() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[1.0, 2.0]);

        let mut interviewer = Scripted::default();
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
        assert_eq!(report.held_back, 0, "{output}");
        assert!(output.contains("2 unresolved voice(s)"), "{output}");
        assert!(!output.contains("not offered"), "{output}");
    }

    /// A session whose second voice is under both floors and has already been named for this
    /// session alone -- the state the tests below start from. Cluster 0 is left unresolved on
    /// purpose, so each of them can also show what happens to a voice nobody named.
    pub(crate) fn named_for_its_session(paths: &Paths, id: &str) -> SessionPaths {
        let session = make_session(paths, id);
        with_speech_seconds(&session, &[40.0, 1.5]);

        let mut interviewer = Scripted::answering(vec![Answer::Skip, named("Silas")]);
        let (report, output) = run_asking(paths, &[], ALL, &mut interviewer);
        assert_eq!(report.session_only, 1, "{output}");
        session
    }

    /// TASK-019 acceptance criteria #1 and #2: an answer about a voice with 1.5 s of speech is
    /// kept, and kept *here* -- the transcript reads as the person the user named, and the
    /// database that every future meeting is matched against is byte-for-byte what it was.
    ///
    /// The two acts the floor separates. Before it, this answer wrote a reference built from a
    /// fragment; now it writes a row in this session's own file and says so.
    #[test]
    fn naming_a_voice_under_the_reference_floor_names_the_session_and_not_the_database() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);
        // Somebody unrelated is already enrolled, so "unchanged" is a real claim about a real
        // file rather than about one that was never created.
        enrolled(&[("Bob", voice(3))], &paths);
        let before = std::fs::read(paths.speakers_json()).unwrap();

        let mut interviewer = Scripted::answering(vec![Answer::Skip, named("Silas")]);
        let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(
            report.session_only, 1,
            "a name given to a voice under the floor is a session-scoped one: {output}"
        );
        assert_eq!(
            std::fs::read(paths.speakers_json()).unwrap(),
            before,
            "a voice this quiet must not change the enrolled database at all"
        );

        let assigned = assigned_in(&session, "20260809-052600");
        assert_eq!(
            assigned
                .names
                .iter()
                .map(|row| (row.cluster, row.name.as_str(), &row.embedding))
                .collect::<Vec<_>>(),
            [(1, "Silas", &voice(1))]
        );

        // Only that voice's turns move, and they carry no confidence: nothing was matched.
        assert_eq!(
            said(&transcript_of(&session)),
            [
                ("Unknown 1", "  hi there  ", None),
                ("You", "morning", None),
                ("Silas", "and from me", None),
                ("Unknown 1", "let us start", None),
            ]
        );

        // Which of the two it did is not something a user should have to infer from a file.
        assert!(
            output.contains("named Silas in this session only"),
            "{output}"
        );
        assert!(output.contains("1.5 s of speech"), "{output}");
        assert!(output.contains("--force-reference"), "{output}");
        assert!(!output.contains("enrolled Silas"), "{output}");
    }

    /// The override the line above advertises: `--force-reference` writes the reference the
    /// floor would have withheld, and then there is nothing session-scoped to record.
    #[test]
    fn force_reference_stores_the_reference_the_floor_would_have_withheld() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);

        let mut interviewer = Scripted::answering(vec![Answer::Skip, named("Silas")]);
        let (report, output) = run_enrolling(&paths, &[], ALL, Enrolment::Always, &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.session_only, 0, "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        assert_eq!(speakers.speakers[0].name, "Silas");
        assert_eq!(speakers.speakers[0].embedding, voice(1));
        assert!(
            !session.speaker_names_json().exists(),
            "an enrolled voice is not also a session-scoped name: {output}"
        );
        assert!(output.contains("enrolled Silas"), "{output}");

        // And the turns now carry a similarity, because this is an identification.
        assert_eq!(
            said(&transcript_of(&session))[2],
            ("Silas", "and from me", Some(1.0))
        );
    }

    /// TASK-019 acceptance criterion #5: an answer is an answer. A voice named for its session
    /// is not asked about again -- not even by `--all`, which is what reached it in the first
    /// place -- and `--correct` is the way back to it, with the prompt saying what it knows.
    #[test]
    fn a_voice_named_for_its_session_is_asked_about_again_only_by_correct() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = named_for_its_session(&paths, "20260809-052600");

        let mut again = Scripted::default();
        let (report, output) = run_asking(&paths, &[], ALL, &mut again);
        assert_eq!(
            again.labels(),
            ["Unknown 1"],
            "only the voice nobody named should still be asked about: {output}"
        );
        assert_eq!(report.skipped, 1, "{output}");

        let mut correcting = Scripted::default();
        let (_, output) = run_asking(&paths, &[], ALL_AND_CORRECT, &mut correcting);
        assert_eq!(correcting.labels(), ["Unknown 1", "Silas"], "{output}");
        assert_eq!(
            correcting.seen[1].attribution,
            Attribution::Assigned {
                name: "Silas".to_string()
            },
            "the prompt has to say this name was given to this session, not matched"
        );
        assert_eq!(correcting.seen[1].confidence(), None, "{output}");
        assert_eq!(
            transcript_of(&session).turns[2].speaker,
            "Silas",
            "a run that answered nothing must leave the name where it was"
        );
    }

    /// Correcting one: the row is replaced rather than appended to, so a voice answered twice
    /// is one claim about one voice and not two rows racing to label it.
    #[test]
    fn re_answering_a_voice_named_for_its_session_replaces_its_row() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = named_for_its_session(&paths, "20260809-052600");

        let mut correcting = Scripted::answering(vec![Answer::Skip, named("Alex")]);
        let (report, output) = run_asking(&paths, &[], ALL_AND_CORRECT, &mut correcting);

        assert_eq!(report.session_only, 1, "{output}");
        assert_eq!(
            assigned_in(&session, "20260809-052600")
                .names
                .iter()
                .map(|row| (row.cluster, row.name.as_str()))
                .collect::<Vec<_>>(),
            [(1, "Alex")]
        );
        assert_eq!(transcript_of(&session).turns[2].speaker, "Alex");
    }

    /// One voice, one record. The same fragment reached again with `--force-reference` is a
    /// promotion: the reference is written and the session-scoped row it replaces is dropped,
    /// so the two can never be made to disagree about who this voice is.
    #[test]
    fn enrolling_a_voice_that_was_named_for_its_session_drops_its_row() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = named_for_its_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![Answer::Skip, named("Silas")]);
        let (report, output) = run_enrolling(
            &paths,
            &[],
            ALL_AND_CORRECT,
            Enrolment::Always,
            &mut interviewer,
        );

        assert_eq!(report.session_only, 0, "{output}");
        assert!(
            assigned_in(&session, "20260809-052600").names.is_empty(),
            "an enrolled voice must stop being an assignment too: {output}"
        );
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        assert_eq!(speakers.speakers[0].embedding, voice(1));
        assert_eq!(
            said(&transcript_of(&session))[2],
            ("Silas", "and from me", Some(1.0)),
            "the same name, now on the basis of a match"
        );
    }

    /// The transcript's schema version survives a rewrite: `enroll` edits turns, it does not
    /// re-stamp the file as something it is not.
    #[test]
    fn a_rewritten_transcript_keeps_its_schema_version_and_session_id() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        run(&paths, &[], &mut interviewer);

        let transcript = transcript_of(&session);
        assert_eq!(transcript.schema_version, TRANSCRIPT_SCHEMA_VERSION);
        assert_eq!(transcript.session_id.as_str(), "20260809-052600");
    }

    /// An empty meethook directory is a first run, not an error.
    #[test]
    fn no_sessions_at_all_is_reported_rather_than_failing() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());

        let mut interviewer = Scripted::default();
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report, EnrollReport::default());
        assert!(output.contains("No sessions found"), "{output}");
    }

    /// TASK-025 acceptance criterion #1: `--voice` asks about the voice it names and about
    /// nobody else, in both the forms the number can be written in.
    #[test]
    fn a_voice_selected_by_number_is_the_only_one_asked_about() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("Bob")]);
        let (report, output) = run_targeting(&paths, &["20260809-052600"], "2", &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 2"], "{output}");
        assert_eq!(report.named, 1, "{output}");
        // TASK-026: a targeted run says `1/1` rather than suppressing the position. It is true,
        // and it says the useful thing -- this is the only question, the run ends after this
        // answer. Suppressing it would put a rule about when a position is worth showing inside
        // the terminal, where no test can see what the user was shown.
        assert_eq!(interviewer.positions(), ["1/1"], "{output}");
        assert_eq!(
            said(&transcript_of(&session))
                .iter()
                .map(|(speaker, _, _)| *speaker)
                .collect::<Vec<_>>(),
            ["Unknown 1", "You", "Bob", "Unknown 1"],
            "the voice that was not asked about must be left exactly as it was"
        );

        // The written-out label is the same selector: a user reading "Unknown 1" off a prompt
        // header should not have to work out which part of it to type.
        let mut spelled_out = Scripted::default();
        let (_, output) =
            run_targeting(&paths, &["20260809-052600"], "Unknown 1", &mut spelled_out);
        assert_eq!(spelled_out.labels(), ["Unknown 1"], "{output}");
    }

    /// Acceptance criteria #2 and #3: a voice the database has already named is reachable by
    /// that name, with no `--correct` -- which is the state somebody is in when the name is
    /// the thing that is wrong.
    #[test]
    fn a_voice_can_be_selected_by_the_name_it_currently_reads_as() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        enrolled(&[("Bob", voice(1))], &paths);

        let mut interviewer = Scripted::answering(vec![named("Robert Chen")]);
        let (report, output) = run_targeting(&paths, &["20260809-052600"], "Bob", &mut interviewer);

        assert_eq!(interviewer.labels(), ["Bob"], "{output}");
        assert_eq!(
            interviewer.seen[0].attribution,
            Attribution::Identified {
                name: "Bob".to_string(),
                similarity: 1.0
            },
            "the prompt has to ask whether this identification is right, not who this is"
        );
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(transcript_of(&session).turns[2].speaker, "Robert Chen");
    }

    /// Acceptance criterion #3 for the other filter, and the reason `held_back` stays at zero:
    /// a run aimed at one voice is not holding anything back, so it must not end on a line
    /// offering `--all`.
    #[test]
    fn a_targeted_voice_under_the_prompt_floor_is_asked_about_without_all() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);

        let mut interviewer = Scripted::default();
        let (report, output) = run_targeting(&paths, &["20260809-052600"], "2", &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 2"], "{output}");
        assert_eq!(report.held_back, 0, "{output}");
        assert!(!output.contains("not offered"), "{output}");
    }

    /// TASK-025 acceptance criterion #4, written as the comparison it actually is rather than
    /// as literal expectations: the prompt a targeted voice gets is the prompt that voice gets
    /// in a full run -- same header, same snippets, same clip -- because it is produced by the
    /// same code from the same cluster.
    ///
    /// Everything but the position, which is a fact about the *run* rather than about the
    /// voice, and a run aimed at one voice genuinely is a different run: it has one question in
    /// it. Destructured exhaustively, no `..`, so that a field added to [`Voice`] later cannot
    /// quietly fall out of this comparison -- the compiler makes the author name it and say
    /// which side of the line it is on.
    #[test]
    fn a_targeted_prompt_is_what_the_full_run_would_have_shown() {
        let id = "20260809-052600";

        let queued_root = tempfile::tempdir().unwrap();
        let queued_paths = Paths::new(queued_root.path());
        make_session(&queued_paths, id);
        let mut queued = Scripted::default();
        let (_, output) = run_asking(&queued_paths, &[], ALL_AND_CORRECT, &mut queued);
        assert_eq!(queued.labels(), ["Unknown 1", "Unknown 2"], "{output}");

        let targeted_root = tempfile::tempdir().unwrap();
        let targeted_paths = Paths::new(targeted_root.path());
        make_session(&targeted_paths, id);
        let mut aimed = Scripted::default();
        let (_, output) = run_targeting(&targeted_paths, &[id], "2", &mut aimed);

        assert_eq!(aimed.seen.len(), 1, "{output}");
        let Shown {
            session,
            meeting,
            position,
            attribution,
            number,
            speech_seconds,
            queue,
            snippets,
            snippet_times,
            snippet_samples,
            clip_samples,
            resembles,
            enrolled,
        } = &aimed.seen[0];
        let queued = &queued.seen[1];
        assert_eq!(session, &queued.session);
        assert_eq!(meeting, &queued.meeting);
        assert_eq!(attribution, &queued.attribution);
        assert_eq!(number, &queued.number);
        assert_eq!(speech_seconds, &queued.speech_seconds);
        // A targeted prompt sees the whole session too: narrowing decides which voices are
        // *asked about*, and the queue is what the session holds.
        assert_eq!(queue, &queued.queue);
        assert_eq!(snippets, &queued.snippets);
        assert_eq!(snippet_times, &queued.snippet_times);
        assert_eq!(snippet_samples, &queued.snippet_samples);
        assert_eq!(clip_samples, &queued.clip_samples);
        assert_eq!(resembles, &queued.resembles);
        assert_eq!(enrolled, &queued.enrolled);
        assert_eq!(
            (position.to_string(), queued.position.to_string()),
            ("1/1".to_string(), "2/2".to_string()),
            "the position is the one thing that differs, because it counts the run's questions \
             and the targeted run has one"
        );
    }

    /// Acceptance criterion #5: reaching a voice differently does not write differently. A
    /// targeted answer about a 1.5 s voice takes the same session-only path, and the same
    /// `--force-reference` override lifts it.
    #[test]
    fn naming_a_targeted_quiet_voice_still_writes_only_a_session_name() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);
        // Somebody unrelated is already enrolled, so "unchanged" is a claim about a real file.
        enrolled(&[("Bob", voice(3))], &paths);
        let before = std::fs::read(paths.speakers_json()).unwrap();

        let mut interviewer = Scripted::answering(vec![named("Silas")]);
        let (report, output) = run_targeting(&paths, &["20260809-052600"], "2", &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.session_only, 1, "{output}");
        assert_eq!(
            std::fs::read(paths.speakers_json()).unwrap(),
            before,
            "a targeted answer about a voice this quiet must not touch the database either"
        );
        assert_eq!(
            assigned_in(&session, "20260809-052600")
                .names
                .iter()
                .map(|row| (row.cluster, row.name.as_str()))
                .collect::<Vec<_>>(),
            [(1, "Silas")]
        );
        assert!(
            output.contains("named Silas in this session only"),
            "{output}"
        );

        // And the override composes with a selector exactly as it does with the queue: it is
        // the other axis, and the targeted path never touches it.
        let forced_root = tempfile::tempdir().unwrap();
        let forced_paths = Paths::new(forced_root.path());
        let forced = make_session(&forced_paths, "20260809-052600");
        with_speech_seconds(&forced, &[40.0, 1.5]);

        let mut forcing = Scripted::answering(vec![named("Silas")]);
        let second = VoiceSelector::from("2");
        let (report, output) = run_over(
            &forced_paths,
            &["20260809-052600"],
            Some(Selection::Voice(second)),
            Offer::default(),
            Sessions::default(),
            Enrolment::Always,
            &mut forcing,
        );

        assert_eq!(report.session_only, 0, "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&forced_paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        assert_eq!(speakers.speakers[0].name, "Silas");
        assert_eq!(speakers.speakers[0].embedding, voice(1));
    }

    /// Acceptance criterion #6, the miss half: a selector that names nobody asks nothing, says
    /// so, and lists what the session does have -- quiet voices included, since those are what
    /// somebody is reaching for when they miss. `failed` is what turns that into a non-zero
    /// exit at the CLI.
    #[test]
    fn a_selector_matching_nothing_reports_what_the_session_has_and_fails() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_fragmented_session(&paths, "20260809-052600");

        for missed in ["Nobody", "9"] {
            let mut interviewer = Scripted::default();
            let (report, output) =
                run_targeting(&paths, &["20260809-052600"], missed, &mut interviewer);

            assert!(interviewer.seen.is_empty(), "{missed}: {output}");
            assert_eq!(report.failed, 1, "{missed}: {output}");
            assert!(output.contains("no voice matched"), "{missed}: {output}");
            for label in ["Unknown 1", "Unknown 2", "Unknown 3", "Unknown 4"] {
                assert!(
                    output.contains(label),
                    "a miss has to say what the session contains, including the voices under \
                     the floor -- {label} missing from: {output}"
                );
            }
        }
    }

    /// Acceptance criterion #6, the ambiguous half. Two clusters under one enrolled name is
    /// exactly the false accept `--correct` exists to fix, so the message has to hand back the
    /// thing that tells them apart rather than picking one of them.
    #[test]
    fn an_ambiguous_selector_names_both_voices_and_the_numbers_that_split_them() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
        enrolled(&[("Alice", nearly(0.0))], &paths);

        let mut interviewer = Scripted::answering(vec![named("Someone")]);
        let (report, output) =
            run_targeting(&paths, &["20260809-052600"], "Alice", &mut interviewer);

        assert!(interviewer.seen.is_empty(), "{output}");
        assert_eq!(report.failed, 1, "{output}");
        assert!(output.contains("matches 2 voices"), "{output}");
        assert!(output.contains("Unknown 1"), "{output}");
        assert!(output.contains("Unknown 2"), "{output}");

        // ...and the number it handed back does reach one of them.
        let mut disambiguated = Scripted::answering(vec![named("Someone")]);
        let (report, output) = run_targeting(&paths, &["20260809-052600"], "2", &mut disambiguated);
        assert_eq!(disambiguated.labels(), ["Alice"], "{output}");
        assert_eq!(report.named, 1, "{output}");
    }

    /// A voice number means nothing across sessions and a name would fan out over every
    /// recording on disk, so a selector without exactly one session id is refused before
    /// anything is read -- and refused loudly, since the alternative is a run that asks about
    /// somebody else's Unknown 2.
    #[test]
    fn a_selector_without_exactly_one_session_id_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        make_session(&paths, "20260809-052700");

        for ids in [&[][..], &["20260809-052600", "20260809-052700"][..]] {
            let mut interviewer = Scripted::default();
            let (report, output) = run_targeting(&paths, ids, "2", &mut interviewer);

            assert!(interviewer.seen.is_empty(), "{ids:?}: {output}");
            assert_eq!(report.failed, 1, "{ids:?}: {output}");
            assert!(
                output.contains("--voice needs exactly one session id"),
                "{ids:?}: {output}"
            );
        }
    }

    /// Why the number is the "Unknown N" and not the cluster id, at the level a user meets it:
    /// naming a voice does not renumber anybody, so the number that reached it still reaches
    /// it afterwards -- and the second visit is a correction.
    #[test]
    fn a_number_keeps_pointing_at_a_voice_after_it_has_been_named() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        let mut first = Scripted::answering(vec![named("Bob")]);
        let (report, output) = run_targeting(&paths, &["20260809-052600"], "2", &mut first);
        assert_eq!(first.labels(), ["Unknown 2"], "{output}");
        assert_eq!(report.named, 1, "{output}");

        let mut again = Scripted::answering(vec![named("Robert Chen")]);
        let (report, output) = run_targeting(&paths, &["20260809-052600"], "2", &mut again);

        assert_eq!(
            again.labels(),
            ["Bob"],
            "the same number must reach the same voice, now under its name: {output}"
        );
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(transcript_of(&session).turns[2].speaker, "Robert Chen");
    }

    /// TASK-033 acceptance criteria #1 and #7: a session id, a timestamp and a name are the
    /// whole command. Nothing is prompted -- [`GivenName`] has no terminal to prompt with --
    /// and both transcript files come out of it reading the new name.
    #[test]
    fn a_timestamp_and_a_name_name_the_voice_speaking_then() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        let (report, output) = run_naming_at(&paths, &["20260809-052600"], "00:03", "Alice");

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.failed, 0, "{output}");
        assert!(output.contains("1 voice selected at 00:03"), "{output}");

        // 00:03 is cluster 1's turn, and it is that whole voice that gets enrolled.
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        assert_eq!(speakers.speakers[0].name, "Alice");
        assert_eq!(speakers.speakers[0].embedding, voice(1));

        assert_eq!(
            said(&transcript_of(&session)),
            [
                ("Unknown 1", "  hi there  ", None),
                ("You", "morning", None),
                ("Alice", "and from me", Some(1.0)),
                ("Unknown 1", "let us start", None),
            ]
        );
        let md = std::fs::read_to_string(session.transcript_md()).unwrap();
        assert!(md.contains("**[00:03] Alice:** and from me"), "{md}");
        assert!(!md.contains("Unknown 2"), "{md}");
    }

    /// Acceptance criterion #2. Minutes are not wrapped at 60 on the way out, so `90:05` is what
    /// the user has in front of them for a turn an hour and a half in -- and it has to be what
    /// reaches that turn.
    #[test]
    fn a_timestamp_past_fifty_nine_minutes_reaches_its_turn() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_turns(
            &paths,
            &session,
            "20260809-052600",
            vec![
                speaker_turn(0.0, 0, "Unknown 1", "hi there"),
                speaker_turn(5405.0, 1, "Unknown 2", "still here"),
            ],
        );
        // The label that turn prints, which is what the user copies.
        let md = std::fs::read_to_string(session.transcript_md()).unwrap();
        assert!(md.contains("**[90:05] Unknown 2:**"), "{md}");

        let (report, output) = run_naming_at(&paths, &["20260809-052600"], "90:05", "Alice");

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(said(&transcript_of(&session))[1].0, "Alice", "{output}");
    }

    /// Acceptance criterion #3. Naming a voice renames every turn it spoke, which is what naming
    /// a voice means everywhere else in this tool -- so the command says how far that reached
    /// rather than leaving a user who pointed at one line to infer it.
    #[test]
    fn renaming_through_a_timestamp_reports_the_turns_and_the_speech_it_covered() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        // Cluster 0 speaks twice, a second each, and the moment pointed at is only one of them.
        let (report, output) = run_naming_at(&paths, &["20260809-052600"], "00:00", "Alice");
        assert_eq!(report.named, 1, "{output}");
        assert!(
            output.contains("renamed 2 turn(s), 2s of speech, to Alice"),
            "{output}"
        );

        // And when clustering split one person in two, the count covers both halves: the claim
        // is about what changed, not about the voice that was selected.
        let split_root = tempfile::tempdir().unwrap();
        let split_paths = Paths::new(split_root.path());
        let split = make_session(&split_paths, "20260809-052600");
        with_embeddings(&split, &[nearly(0.0), nearly(20.0)]);

        let (report, output) = run_naming_at(&split_paths, &["20260809-052600"], "00:00", "Alice");
        assert_eq!(report.named, 1, "{output}");
        assert!(
            output.contains("renamed 3 turn(s), 3s of speech, to Alice"),
            "naming one half of a split voice renames both: {output}"
        );

        // Answering a voice with the name it already reads as changes nothing, and says that
        // rather than reporting zero turns.
        let (report, output) = run_naming_at(&split_paths, &["20260809-052600"], "00:00", "Alice");
        assert_eq!(report.failed, 0, "{output}");
        assert!(
            output.contains("no turns changed: that voice already read as Alice"),
            "{output}"
        );
    }

    /// Acceptance criterion #4. Four ways a timestamp lands on nothing nameable, and each one
    /// says which it was: only one of them is the user's mistake, and the others each suggest a
    /// different next move.
    #[test]
    fn a_timestamp_that_lands_on_nothing_nameable_says_which_of_the_four_it_was() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        // The fixture's turns are 0-1 and 3-5 on the speaker track with the mic at 1-2, so it
        // already has a hole and an end.
        for (at, expected) in [
            ("00:01", "is on the microphone track"),
            ("00:02", "nobody was speaking at 00:02"),
            (
                "00:30",
                "is past the end of this session, which ends at 00:05",
            ),
        ] {
            let (report, output) = run_naming_at(&paths, &["20260809-052600"], at, "Alice");
            assert_eq!(report.failed, 1, "{at}: {output}");
            assert_eq!(report.named, 0, "{at}: {output}");
            assert!(output.contains(expected), "{at}: {output}");
        }
        // The silence line hands back the nearest turn, because a miss here is usually a second
        // or two off and the right timestamp is on the page the user is reading.
        let (_, output) = run_naming_at(&paths, &["20260809-052600"], "00:02", "Alice");
        assert!(
            output.contains("the nearest turn is You at 00:01"),
            "{output}"
        );

        // The fourth: a transcript whose speech belongs to no cluster at all, which is what
        // diarization finding no voices leaves behind.
        let bare_root = tempfile::tempdir().unwrap();
        let bare_paths = Paths::new(bare_root.path());
        let bare = make_session(&bare_paths, "20260809-052600");
        with_turns(
            &bare_paths,
            &bare,
            "20260809-052600",
            vec![Turn {
                speaker: unknown_speaker(1),
                start: 0.0,
                end: 4.0,
                text: "hi there".to_string(),
                source_track: SourceTrack::Speaker,
                cluster: None,
                speaker_id_confidence: None,
            }],
        );

        let (report, output) = run_naming_at(&bare_paths, &["20260809-052600"], "00:00", "Alice");
        assert_eq!(report.failed, 1, "{output}");
        assert!(
            output.contains("the turn at 00:00 records no voice"),
            "{output}"
        );
    }

    /// Acceptance criterion #5. What an answer writes is the other axis entirely, and pointing
    /// at a timestamp does not touch it: the reference floor applies exactly as it does to the
    /// queue, and `--force-reference` overrides it exactly as it does there.
    #[test]
    fn a_timestamp_follows_the_same_reference_floor_and_the_same_override() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_speech_seconds(&session, &[40.0, 1.5]);

        let (report, output) = run_naming_at(&paths, &["20260809-052600"], "00:03", "Silas");
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.session_only, 1, "{output}");
        assert!(
            EnrolledSpeakers::read_or_empty(&paths)
                .unwrap()
                .speakers
                .is_empty(),
            "a voice this quiet must not reach the database however it was selected: {output}"
        );
        assert_eq!(
            assigned_in(&session, "20260809-052600")
                .names
                .iter()
                .map(|row| (row.cluster, row.name.as_str()))
                .collect::<Vec<_>>(),
            [(1, "Silas")]
        );

        // ... and the override the line above advertises writes the reference the floor withheld.
        let forced_root = tempfile::tempdir().unwrap();
        let forced_paths = Paths::new(forced_root.path());
        let forced = make_session(&forced_paths, "20260809-052600");
        with_speech_seconds(&forced, &[40.0, 1.5]);

        let (report, output) = run_over(
            &forced_paths,
            &["20260809-052600"],
            Some(Selection::At("00:03".parse().unwrap())),
            Offer::default(),
            Sessions::default(),
            Enrolment::Always,
            &mut GivenName::new("Silas"),
        );

        assert_eq!(report.session_only, 0, "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&forced_paths).unwrap();
        assert_eq!(speakers.speakers.len(), 1);
        assert_eq!(speakers.speakers[0].name, "Silas");
        assert_eq!(speakers.speakers[0].embedding, voice(1));
        assert!(!forced.speaker_names_json().exists(), "{output}");
    }

    /// Acceptance criterion #6, both halves: naming somebody already enrolled adds a recording
    /// of them rather than replacing one, and an answer that would take a name off a voice the
    /// user was not pointing at is refused. The safeguards are downstream of the selection, so a
    /// timestamp reaches exactly the same ones.
    #[test]
    fn a_name_that_already_exists_is_reused_and_never_taken_off_another_voice() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        // Alice, enrolled from a voice that matches neither cluster here.
        enrolled(&[("Alice", voice(3))], &paths);

        let (report, output) = run_naming_at(&paths, &["20260809-052600"], "00:03", "Alice");

        assert_eq!(report.named, 1, "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(
            speakers
                .speakers
                .iter()
                .map(|s| (s.name.as_str(), s.embedding.as_slice()))
                .collect::<Vec<_>>(),
            [
                ("Alice", voice(3).as_slice()),
                ("Alice", voice(1).as_slice())
            ],
            "the first recording must survive the second: {output}"
        );

        // The refusal: cluster 1 reads Bob, and naming its near neighbour Alice would move that
        // name off it. Nothing is written and the voice keeps what it read.
        let taken_root = tempfile::tempdir().unwrap();
        let taken_paths = Paths::new(taken_root.path());
        let taken = make_session(&taken_paths, "20260809-052600");
        with_embeddings(&taken, &[nearly(0.0), nearly(20.0)]);
        enrolled(&[("Bob", nearly(60.0))], &taken_paths);
        let before = std::fs::read(taken_paths.speakers_json()).unwrap();

        let (report, output) = run_naming_at(&taken_paths, &["20260809-052600"], "00:00", "Alice");

        assert_eq!(report.refused, 1, "{output}");
        assert_eq!(report.named, 0, "{output}");
        assert!(
            output.contains("refused Alice for Unknown 1: it would take Bob off Unknown 2"),
            "{output}"
        );
        assert_eq!(
            std::fs::read(taken_paths.speakers_json()).unwrap(),
            before,
            "a refused answer writes nothing"
        );
        assert_eq!(said(&transcript_of(&taken))[2].0, "Bob", "{output}");
    }

    /// Two turns a fraction of a second apart print the same label, and then the timestamp names
    /// neither voice on its own. That is a question this command cannot answer for the user, so
    /// it hands back what tells them apart -- exactly as an ambiguous `--voice` does.
    #[test]
    fn a_label_two_voices_share_is_refused_with_the_numbers_that_split_them() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_turns(
            &paths,
            &session,
            "20260809-052600",
            vec![
                speaker_turn(10.1, 0, "Unknown 1", "one word"),
                speaker_turn(10.6, 1, "Unknown 2", "another"),
            ],
        );

        let (report, output) = run_naming_at(&paths, &["20260809-052600"], "00:10", "Alice");

        assert_eq!(report.failed, 1, "{output}");
        assert_eq!(report.named, 0, "{output}");
        assert!(output.contains("is the label of 2 turns"), "{output}");
        assert!(output.contains("--voice \"Unknown 1\""), "{output}");
        assert!(output.contains("--voice \"Unknown 2\""), "{output}");
    }

    /// A timestamp is an offset into one recording, so it lands somewhere different in each of
    /// several -- refused before anything is read, like `--voice`, and with the reason that
    /// belongs to the flag that was passed.
    #[test]
    fn a_timestamp_without_exactly_one_session_id_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        make_session(&paths, "20260809-052700");

        for ids in [&[][..], &["20260809-052600", "20260809-052700"][..]] {
            let (report, output) = run_naming_at(&paths, ids, "00:03", "Alice");
            assert_eq!(report.failed, 1, "{ids:?}: {output}");
            assert!(
                output.contains("--at needs exactly one session id"),
                "{ids:?}: {output}"
            );
            assert!(
                output.contains("offset into one recording"),
                "the reason has to be the one that belongs to --at: {ids:?}: {output}"
            );
        }
    }

    /// A name supplied up front is never shown the voice it lands on, so a queue would put one
    /// name on everybody in it. Refused in the library, which is the only place that can see both
    /// the answerer and the selection.
    #[test]
    fn a_name_given_up_front_without_a_selector_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        let before = files_under(root.path());

        let (report, output) = run_over(
            &paths,
            &["20260809-052600"],
            None,
            Offer::default(),
            Sessions::default(),
            Enrolment::default(),
            &mut GivenName::new("Alice"),
        );

        assert_eq!(report.failed, 1, "{output}");
        assert_eq!(report.named, 0, "{output}");
        assert!(output.contains("--name needs a voice"), "{output}");
        assert_eq!(files_under(root.path()), before, "{output}");
    }

    /// Acceptance criterion #7: the prompt is handed everybody enrolled, nearest first, so an
    /// [`Interviewer`] can offer names without ever opening `speakers.json`.
    ///
    /// All three references are outside `IDENTIFY_DISTANCE` of cluster 0 -- 60, 75 and 85
    /// degrees, against a cut at 0.40 of cosine distance -- which is exactly the voice
    /// identification gave up on and the one whose prompt has something to offer. The names run
    /// against the ranking alphabetically, so a list in name order would fail this.
    #[test]
    fn a_prompt_is_handed_every_enrolled_person_nearest_first() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_embeddings(&session, &[voice(0), voice(1)]);
        enrolled(
            &[
                ("Zoe", nearly(60.0)),
                ("Alice", nearly(75.0)),
                ("Mona", nearly(85.0)),
            ],
            &paths,
        );

        let mut interviewer = Scripted::answering(vec![Answer::Skip]);
        let (_, output) = run(&paths, &[], &mut interviewer);

        // Cluster 1 sits close to the 60-degree reference, so it is identified and not asked
        // about; cluster 0 is the one question this run has.
        assert_eq!(interviewer.labels(), ["Unknown 1"], "{output}");
        let shown = &interviewer.seen[0];
        assert_eq!(
            shown.offered(),
            [("Zoe", 1), ("Alice", 1), ("Mona", 1)],
            "{output}"
        );
        assert!(
            (shown.resembles[0].similarity - 60.0f32.to_radians().cos()).abs() < 1e-6,
            "{:?}",
            shown.resembles
        );
        // Every one of them is past the cut identification applies, and still offered.
        for candidate in &shown.resembles {
            assert!(
                1.0 - candidate.similarity > IDENTIFY_DISTANCE,
                "{candidate:?} should be outside the cut for this test to mean anything"
            );
        }
    }

    /// The ranking reflects the database as it stands at the prompt, not as it stood when the
    /// run began -- so a name given a moment ago is offered for the next voice, which is the
    /// case that matters when clustering has split one person in two.
    #[test]
    fn a_name_given_earlier_in_the_run_is_offered_for_a_later_voice() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("Alice"), Answer::Skip]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        // Nobody was enrolled when the first question was asked.
        assert_eq!(interviewer.seen[0].offered(), [], "{output}");
        assert_eq!(interviewer.seen[1].offered(), [("Alice", 1)], "{output}");
    }

    /// Acceptance criterion #6 at the seam, and the state of every install before anybody has
    /// been enrolled: nobody to offer is an empty list, and the question is still asked.
    #[test]
    fn an_empty_database_offers_nobody_and_still_prompts() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![Answer::Skip, Answer::Skip]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(report.skipped, 2, "{output}");
        assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
        assert!(
            interviewer.seen.iter().all(|v| v.resembles.is_empty()),
            "{output}"
        );
    }

    /// A correction prompt shows a name and a ranking on one screen, and the two must not
    /// disagree: the first entry is the person the identification already named, carrying the
    /// same number the label does.
    #[test]
    fn an_identified_voices_ranking_leads_with_the_name_it_was_given() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_embeddings(&session, &[voice(0), voice(1)]);
        enrolled(&[("Alice", nearly(10.0)), ("Zoe", nearly(70.0))], &paths);

        let mut interviewer = Scripted::answering(vec![Answer::Skip, Answer::Skip]);
        let (_, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(interviewer.labels(), ["Alice", "Zoe"], "{output}");
        for shown in &interviewer.seen {
            assert_eq!(shown.resembles[0].name, shown.label(), "{output}");
            assert_eq!(
                Some(shown.resembles[0].similarity),
                shown.confidence(),
                "{output}"
            );
        }
        // Both people are offered for both voices; only the order differs.
        assert_eq!(
            interviewer.seen[0].offered(),
            [("Alice", 1), ("Zoe", 1)],
            "{output}"
        );
        assert_eq!(
            interviewer.seen[1].offered(),
            [("Zoe", 1), ("Alice", 1)],
            "{output}"
        );
    }

    /// An [`Interviewer`] that asks what one name would do before deciding what to answer, and
    /// keeps every answer it got back.
    ///
    /// The type is the point. It holds no [`EnrolledSpeakers`], no [`Paths`], no session
    /// directory and no `&mut` anything -- a [`Voice`] is the whole of what it is handed -- so a
    /// test that reads a [`Consequence`] out of it has shown that the seam carries the preview
    /// rather than that this module went and computed one.
    ///
    /// It answers the first voice it is shown and skips the rest, because these tests are about
    /// one answer landing and the fixture session has two voices.
    struct Previewing {
        asking: String,
        answer: Answer,
        previews: Vec<Option<Consequence>>,
    }

    impl Previewing {
        fn asking(name: &str, answer: Answer) -> Previewing {
            Previewing {
                asking: name.to_string(),
                answer,
                previews: Vec::new(),
            }
        }

        /// What the first voice's preview said, which is the one every test here asserts on.
        fn first(&self) -> &Consequence {
            self.previews[0]
                .as_ref()
                .expect("a name that is not blank has a consequence")
        }
    }

    impl Interviewer for Previewing {
        fn identify(&mut self, voice: &Voice<'_>) -> Answer {
            self.previews.push(voice.preview.of(&self.asking));
            if self.previews.len() == 1 {
                self.answer.clone()
            } else {
                Answer::Skip
            }
        }
    }

    fn run_previewing(paths: &Paths, interviewer: &mut Previewing) -> (EnrollReport, String) {
        run_over(
            paths,
            &[],
            None,
            Offer::default(),
            Sessions::default(),
            Enrolment::default(),
            interviewer,
        )
    }

    /// Acceptance criterion #1, at the strongest available reading of "writes nothing": not
    /// "`speakers.json` is unchanged" but "no file under the root changed by one byte", over a
    /// run that previewed a name for every voice it was shown and then answered none of them.
    #[test]
    fn asking_what_a_name_would_do_writes_nothing() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        let before = files_under(root.path());

        let mut interviewer = Previewing::asking("Alice", Answer::Skip);
        let (report, output) = run_previewing(&paths, &mut interviewer);

        assert_eq!(report.named, 0, "{output}");
        assert_eq!(report.skipped, 2, "{output}");
        assert_eq!(interviewer.previews.len(), 2, "{output}");
        assert_eq!(
            interviewer.first().stored,
            Some(Stored::Enrolled),
            "{output}"
        );
        assert_eq!(
            files_under(root.path()),
            before,
            "asking what an answer would do may not write one byte: {output}"
        );
    }

    /// Acceptance criterion #5: the outcome a preview reported is the outcome the write
    /// produced. Agreement is structural -- the commit takes the copies the dry run built -- so
    /// what this pins is that a later refactor cannot go back to deriving the two separately.
    #[test]
    fn a_preview_of_an_enrollment_is_what_the_answer_then_writes() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let mut interviewer = Previewing::asking("Alice", named("Alice"));
        let (report, output) = run_previewing(&paths, &mut interviewer);

        assert_eq!(
            interviewer.first().stored,
            Some(Stored::Enrolled),
            "{output}"
        );
        assert!(!interviewer.first().session_only(), "{output}");
        assert!(output.contains("enrolled Alice"), "{output}");
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.session_only, 0, "{output}");
        assert_eq!(
            EnrolledSpeakers::read_or_empty(&paths)
                .unwrap()
                .references("Alice"),
            1,
            "{output}"
        );
    }

    /// The same agreement over the outcome that is easiest to get wrong, because the name still
    /// lands while nothing is stored: at the cap, the preview must say so *before* the user
    /// commits to a name that will not help recognise anybody next time.
    #[test]
    fn a_preview_at_the_reference_cap_is_the_session_only_name_that_follows() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let axes = MAX_REFERENCES_PER_SPEAKER + 2;
        let session = make_session(&paths, "20260809-052600");
        // Every voice on its own axis, so nothing Alice holds is within reach of the voice being
        // asked about and the question really is asked.
        with_embeddings(&session, &[axis(axes - 2, axes), axis(axes - 1, axes)]);
        let held: Vec<(&str, Vec<f32>)> = (0..MAX_REFERENCES_PER_SPEAKER)
            .map(|i| ("Alice", axis(i, axes)))
            .collect();
        enrolled(&held, &paths);

        let mut interviewer = Previewing::asking("Alice", named("Alice"));
        let (report, output) = run_previewing(&paths, &mut interviewer);

        assert_eq!(
            interviewer.first().stored,
            Some(Stored::AtCapacity {
                held: MAX_REFERENCES_PER_SPEAKER,
                shortest: None,
            }),
            "{output}"
        );
        assert!(interviewer.first().session_only(), "{output}");
        assert!(
            output.contains(&format!(
                "named Alice in this session only: Alice already holds \
                 {MAX_REFERENCES_PER_SPEAKER} reference(s)"
            )),
            "{output}"
        );
        assert_eq!(report.session_only, 1, "{output}");
        assert_eq!(
            EnrolledSpeakers::read_or_empty(&paths)
                .unwrap()
                .references("Alice"),
            MAX_REFERENCES_PER_SPEAKER,
            "{output}"
        );
    }

    /// The override crosses the seam on the answer, with no interface anywhere in the test.
    ///
    /// [`Previewing`] holds no [`Paths`], no database and no session directory, so a refusal it
    /// can read came through [`Voice::preview`] and an answer the library honoured came back
    /// through [`Interviewer::identify`]. That is the whole claim: the answerer saw the cost and
    /// said to pay it, and the library needed to know nothing about who was asking. Which is why
    /// the line prompt and any scripted driver reach the same behaviour as the frame does --
    /// nothing about it is decided in the frame.
    #[test]
    fn an_override_crosses_the_seam_on_the_answer() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
        enrolled(&[("Bob", nearly(60.0))], &paths);

        let mut interviewer = Previewing::asking("Alice", named_anyway("Alice"));
        let (report, output) = run_previewing(&paths, &mut interviewer);

        assert_eq!(
            interviewer.first().refused,
            Some(Refusal::Taken {
                voice: "Unknown 2".to_string(),
                losing: "Bob".to_string(),
            }),
            "the answerer has to be able to see the cost, or insisting is uninformed: {output}"
        );
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.refused, 0, "{output}");
    }

    /// One run's narration, whole and in order, as [`Lines`] renders it.
    ///
    /// Every other test here asserts a substring, which cannot see a line that moved, a blank
    /// line that appeared, or a pair that swapped -- and the notes the run now emits are placed
    /// by a renderer rather than by the statement that computed them, so line order is exactly
    /// what wants pinning. The fixture is built to reach one of each tier in one run: a session
    /// passed over before anything is read, a queue header with its held-back clause, a
    /// transcript brought into line before a question is asked, an enrollment, a reference taken
    /// off somebody else, and an answer refused.
    ///
    /// `--correct` for the whole run, so the second session's already-named voice is asked
    /// about; that is also why both headers read "to review" rather than "unresolved", which
    /// the tests above cover.
    #[test]
    fn one_runs_narration_reads_as_these_lines_in_this_order() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());

        // The recorder died mid-session: nothing to read, and the first line of the run.
        let orphan = paths.session(&SessionId::parse("20260809-052500").unwrap());
        std::fs::create_dir_all(orphan.dir()).unwrap();

        // One voice worth a question and three fragments under the floor. Its voices sit on the
        // two axes Nate and the second session's voices do not, so enrolling here changes
        // nothing there and the two sessions' lines stay independent.
        let fragmented = make_fragmented_session(&paths, "20260809-052600");
        with_embeddings(&fragmented, &[voice(2), voice(3), voice(3), voice(3)]);

        let session = make_session(&paths, "20260809-052700");
        heard_at_once(&session, 0, 1);
        enrolled(&[("Nate", voice(0))], &paths);

        let mut interviewer =
            Scripted::answering(vec![named("Alice"), named("Aaron"), named("Aaron")]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(
            output,
            "20260809-052500  passed over: no session.json (the recorder crashed mid-session)\n\
             20260809-052600  1 voice(s) to review, 0 of them already named, 3 quieter voice(s) \
             not offered -- meethook enroll --all\n\
             20260809-052600  enrolled Alice\n\
             20260809-052700  transcript brought up to date\n\
             20260809-052700  2 voice(s) to review, 1 of them already named\n\
             20260809-052700  Nate no longer has a reference: that voice is Aaron\n\
             20260809-052700  enrolled Aaron\n\
             20260809-052700  refused Aaron for Unknown 2: Unknown 1 already has that name and \
             the two were heard speaking at once, so they are not one person -- meethook enroll \
             --correct --voice Unknown 1 if that is the wrong one\n"
        );
        assert_eq!(
            report,
            EnrollReport {
                named: 2,
                session_only: 0,
                skipped: 0,
                kept: 0,
                held_back: 3,
                refused: 1,
                passed_over: 1,
                failed: 0,
                asserted: 0,
                vetoes_overridden: 0,
            }
        );
    }
    /// TASK-046.06.01 acceptance criterion #1: a prompt is handed the whole session, not only
    /// the voice it is about -- which is what a queue pane is drawn from.
    #[test]
    fn a_prompt_carries_every_voice_of_its_session() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::default();
        let (_, output) = run(&paths, &[], &mut interviewer);

        // Both voices, in first-appearance order -- the order the transcript reads in -- with
        // the basis and not only the label, and neither of them under the floor.
        assert_eq!(
            interviewer.seen[0].queue,
            vec![
                Row {
                    number: "Unknown 1".to_string(),
                    attribution: Attribution::Unknown("Unknown 1".to_string()),
                    speech_seconds: 10.0,
                    below_floor: false,
                },
                Row {
                    number: "Unknown 2".to_string(),
                    attribution: Attribution::Unknown("Unknown 2".to_string()),
                    speech_seconds: 11.0,
                    below_floor: false,
                },
            ],
            "{output}"
        );
    }

    /// TASK-046.06.01 acceptance criterion #1, the half a queue pane needs to explain itself:
    /// the voices this run did *not* offer are in the queue, and say why.
    #[test]
    fn the_queue_holds_the_voices_the_floor_held_back_and_marks_them() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_fragmented_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::default();
        let (report, output) = run(&paths, &[], &mut interviewer);

        // One question, three voices held back -- and all four rows on the prompt.
        assert_eq!(interviewer.seen.len(), 1, "{output}");
        assert_eq!(report.held_back, 3, "{output}");
        assert_eq!(
            interviewer.seen[0].rows(),
            [
                ("Unknown 1", "Unknown 1", false),
                ("Unknown 2", "Unknown 2", true),
                ("Unknown 3", "Unknown 3", true),
                ("Unknown 4", "Unknown 4", true),
            ],
            "{output}"
        );
    }

    /// TASK-046.06.01 acceptance criterion #2: the queue is rebuilt per question, so it shows
    /// what this run has already done rather than what the session looked like when it opened.
    #[test]
    fn the_queue_shows_a_voice_an_earlier_answer_named() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("Alice")]);
        let (_, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(
            interviewer.seen[0].queue[0].attribution,
            Attribution::Unknown("Unknown 1".to_string()),
            "{output}"
        );
        assert_eq!(
            interviewer.seen[1].queue[0].attribution,
            Attribution::Identified {
                name: "Alice".to_string(),
                similarity: 1.0
            },
            "the second question must see the first one's answer: {output}"
        );
        // And the handle did not move with the name -- acceptance criterion #3 in its in-run
        // form, which is the one a cursor depends on.
        assert_eq!(interviewer.seen[1].queue[0].number, "Unknown 1", "{output}");
    }

    /// TASK-046.06.01 acceptance criterion #3: the handle a state machine keys on is the
    /// "Unknown N", and it stays put when the label does not.
    #[test]
    fn a_voice_carries_a_number_that_a_name_does_not_move() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        enrolled(&[("Alice", voice(0))], &paths);

        let mut interviewer = Scripted::default();
        let (_, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(interviewer.seen[0].label(), "Alice", "{output}");
        assert_eq!(
            interviewer.seen[0].number, "Unknown 1",
            "the label is the name and the number is the handle: {output}"
        );
    }

    /// TASK-046.06.01 acceptance criteria #4 and #6: a deferred voice is asked about again in
    /// the same session, with the number it was first offered with.
    #[test]
    fn a_deferred_voice_comes_back_with_the_position_it_had() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        // Defer the first voice, answer the second, then answer the first on the second pass.
        let mut interviewer =
            Scripted::answering(vec![Answer::Later, named("Bob"), named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(
            interviewer.labels(),
            ["Unknown 1", "Unknown 2", "Unknown 1"],
            "{output}"
        );
        assert_eq!(
            interviewer.positions(),
            ["1/2", "2/2", "1/2"],
            "a deferred voice is the same question, so it keeps its number: {output}"
        );
        assert_eq!(report.named, 2, "{output}");

        // And the second pass's answer landed on the voice it was asked about.
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        let stored: Vec<(&str, &[f32])> = speakers
            .speakers
            .iter()
            .map(|s| (s.name.as_str(), s.embedding.as_slice()))
            .collect();
        assert_eq!(
            stored,
            [("Bob", voice(1).as_slice()), ("Alice", voice(0).as_slice())],
            "{output}"
        );
    }

    /// TASK-046.06.01 acceptance criterion #5: a pass that produces no answer at all is where
    /// a session ends, and the voices still deferred are the skips they turned out to be.
    #[test]
    fn deferring_every_voice_ends_the_session_and_counts_skips() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        let before = files_under(root.path());

        let mut interviewer = Scripted::answering(vec![Answer::Later, Answer::Later]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(
            interviewer.labels(),
            ["Unknown 1", "Unknown 2"],
            "one pass, and no second one: nothing moved: {output}"
        );
        assert_eq!(report.skipped, 2, "{output}");
        assert_eq!(report.kept, 0, "{output}");
        assert_eq!(report.named, 0, "{output}");
        assert_eq!(
            files_under(root.path()),
            before,
            "deferring writes nothing, however many times it is answered"
        );
    }

    /// TASK-046.06.02.01 acceptance criterion #1: an answerer that says it is still working
    /// keeps the session open across a pass that produced no answer, and is asked about the
    /// same voices again with the same numbers.
    ///
    /// This is the hole a full-screen frame falls into and a line prompt cannot. A frame with a
    /// cursor defers a voice in order to *reach* another one, so moving the cursor backwards is
    /// a pass in which nothing was answered -- and before this method existed that ended the
    /// run, which from the user's side is the frame closing because they pressed Up.
    #[test]
    fn a_still_working_answerer_is_offered_the_deferred_voices_again() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        // Pass one defers both voices, which on its own is where a session ends. The answerer
        // says otherwise for that one pass, so pass two happens and lands a name.
        let mut interviewer = Scripted::answering(vec![
            Answer::Later,
            Answer::Later,
            named("Alice"),
            Answer::Skip,
        ])
        .working_for(1);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(
            interviewer.labels(),
            ["Unknown 1", "Unknown 2", "Unknown 1", "Unknown 2"],
            "the stalled pass kept the session open, so both voices come back: {output}"
        );
        assert_eq!(
            interviewer.positions(),
            ["1/2", "2/2", "1/2", "2/2"],
            "a re-offered voice is the same question, so it keeps its number: {output}"
        );
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.skipped, 1, "{output}");

        // And the second pass's answer landed on the voice it was asked about.
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        let stored: Vec<(&str, &[f32])> = speakers
            .speakers
            .iter()
            .map(|s| (s.name.as_str(), s.embedding.as_slice()))
            .collect();
        assert_eq!(stored, [("Alice", voice(0).as_slice())], "{output}");
    }

    /// TASK-046.06.02.01 acceptance criterion #4, the half of the termination contract the loop
    /// keeps for itself: a pass with nothing left to offer ends the session without asking the
    /// answerer, because there is no next prompt through which it could change its mind or
    /// reach [`Answer::Quit`].
    #[test]
    fn an_empty_queue_ends_the_session_without_consulting_the_answerer() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        // Both voices answered on the first pass, so the second pass has nothing to ask about.
        let mut interviewer =
            Scripted::answering(vec![named("Alice"), named("Bob")]).working_for(1);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
        assert_eq!(report.named, 2, "{output}");
        assert_eq!(
            interviewer.working_passes.get(),
            1,
            "the countdown is untouched, so the empty pass never asked: {output}"
        );
    }

    /// TASK-046.06.01 acceptance criterion #5, the other bucket: a deferred voice that already
    /// had a name is a kept identification, exactly as pressing Enter on it would be.
    ///
    /// And TASK-046.06.02.01 acceptance criterion #2: a session that stayed open over a
    /// still-working pass ends by the same counting when the answerer does say it is finished.
    #[test]
    fn a_deferred_voice_that_was_named_is_kept_rather_than_skipped() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        // Only the first voice is enrolled, so `--correct` offers one named and one unnamed.
        enrolled(&[("Alice", voice(0))], &paths);

        let mut interviewer = Scripted::answering(vec![Answer::Later, Answer::Later]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(interviewer.labels(), ["Alice", "Unknown 2"], "{output}");
        assert_eq!(report.kept, 1, "{output}");
        assert_eq!(report.skipped, 1, "{output}");

        // The same run, but the answerer works through one stalled pass before it agrees it is
        // finished. Deferring writes nothing, so the second run is offered exactly what the
        // first one was.
        let mut later = Scripted::answering(vec![
            Answer::Later,
            Answer::Later,
            Answer::Later,
            Answer::Later,
        ])
        .working_for(1);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut later);

        assert_eq!(
            later.labels(),
            ["Alice", "Unknown 2", "Alice", "Unknown 2"],
            "{output}"
        );
        assert_eq!(
            (report.kept, report.skipped, report.named),
            (1, 1, 0),
            "the terminal deferred set is counted once, into the same buckets: {output}"
        );
    }

    /// TASK-046.06.01 acceptance criterion #5, against the in-run guard: a voice somebody
    /// else's answer named while it sat deferred is passed over on the next pass rather than
    /// asked twice -- and counted in neither bucket, because it was answered.
    #[test]
    fn a_deferred_voice_another_answer_named_is_not_asked_again() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        // One person clustering split in two: naming either half names the other.
        with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);

        let mut interviewer = Scripted::answering(vec![Answer::Later, named("Alice")]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(
            interviewer.labels(),
            ["Unknown 1", "Unknown 2"],
            "the deferred voice was named by the other answer, so there is nothing to ask: \
             {output}"
        );
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(report.skipped, 0, "{output}");
        assert_eq!(report.kept, 0, "{output}");
    }

    /// TASK-049 acceptance criteria #1 and #2: one answer ends the session's questions, and the
    /// voices left behind are still counted -- so the report accounts for the whole queue
    /// without a keypress per voice in it.
    #[test]
    fn leaving_a_session_ends_it_without_asking_about_the_rest() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_fragmented_session(&paths, "20260809-052600");

        // Four offered under `--all`: one voice worth naming and a tail of fragments.
        let mut interviewer = Scripted::answering(vec![named("Alice"), Answer::Leave]);
        let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

        assert_eq!(
            interviewer.labels(),
            ["Unknown 1", "Unknown 2"],
            "two questions for four voices: the rest are left without being asked: {output}"
        );
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(
            report.skipped, 3,
            "the voice on the screen and the two behind it are all left: {output}"
        );
        assert_eq!(report.kept, 0, "{output}");
        assert!(
            output.contains("left early, 3 voice(s) left as they were"),
            "the run says why the skips outnumber the answers: {output}"
        );
    }

    /// TASK-049 acceptance criterion #2, the case where the arithmetic can go quietly wrong: a
    /// voice this same pass has already named by naming its other half is not also reported as
    /// one the run left alone.
    #[test]
    fn a_left_voice_named_earlier_in_the_pass_is_not_counted_twice() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_fragmented_session(&paths, "20260809-052600");
        // The first and third voices are one person clustering split in two, so the first
        // answer names a voice still sitting in the queue behind the one being asked about.
        with_embeddings(&session, &[nearly(0.0), voice(1), nearly(20.0), voice(3)]);

        let mut interviewer = Scripted::answering(vec![named("Alice"), Answer::Leave]);
        let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert_eq!(
            report.kept, 0,
            "the split half was named by this run, not left as it was found: {output}"
        );
        assert_eq!(
            report.skipped, 2,
            "the two genuinely unanswered voices, and not the one already counted: {output}"
        );
    }

    /// TASK-049 acceptance criteria #1 and #4: leaving one session opens the next, which is the
    /// whole difference between this and quitting.
    #[test]
    fn leaving_one_session_opens_the_next() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        make_session(&paths, "20260810-052600");

        let mut interviewer = Scripted::answering(vec![Answer::Leave, named("Bob"), Answer::Skip]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        let asked: Vec<(&str, &str)> = interviewer
            .seen
            .iter()
            .map(|shown| (shown.session.as_str(), shown.number.as_str()))
            .collect();
        assert_eq!(
            asked,
            [
                ("20260809-052600", "Unknown 1"),
                ("20260810-052600", "Unknown 1"),
                ("20260810-052600", "Unknown 2"),
            ],
            "one question in the session that was left, and the next session ran in full: \
             {output}"
        );
        assert_eq!(report.named, 1, "{output}");
        assert_eq!(
            report.skipped, 3,
            "both voices of the first session and the one skipped in the second: {output}"
        );
    }

    /// TASK-049 acceptance criterion #3: leaving writes nothing of its own, and what was
    /// answered before it is already on disk.
    #[test]
    fn a_name_accepted_before_leaving_stays_on_disk() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![named("Alice"), Answer::Leave]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!((report.named, report.skipped), (1, 1), "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(
            speakers
                .speakers
                .iter()
                .map(|s| s.name.as_str())
                .collect::<Vec<_>>(),
            ["Alice"],
            "the accepted name survives the session being left: {output}"
        );
        assert_eq!(
            said(&transcript_of(&session))
                .iter()
                .map(|(speaker, _, _)| *speaker)
                .collect::<Vec<_>>(),
            ["Alice", "You", "Unknown 2", "Alice"],
            "the voice left behind is untouched, and the named one is written: {output}"
        );
    }

    /// TASK-049 acceptance criterion #4: leaving the last session on disk ends the run rather
    /// than looping over it again or erroring.
    #[test]
    fn leaving_the_last_session_ends_the_run() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        let before = files_under(root.path());

        let mut interviewer = Scripted::answering(vec![Answer::Leave]);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(
            interviewer.labels(),
            ["Unknown 1"],
            "one question, and the run returned rather than coming round again: {output}"
        );
        assert_eq!(report.skipped, 2, "{output}");
        assert_eq!(report.named, 0, "{output}");
        assert_eq!(
            files_under(root.path()),
            before,
            "leaving writes nothing at all"
        );
    }

    /// TASK-049 acceptance criterion #5: leaving is an answer, not a stalled pass, so
    /// [`Interviewer::still_working`] is never consulted on this path -- it can neither
    /// suppress the exit nor be defeated by it.
    #[test]
    fn leaving_is_not_a_stalled_pass() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        // An answerer that would keep five further stalled passes open. The session ends anyway.
        let mut interviewer = Scripted::answering(vec![Answer::Leave]).working_for(5);
        let (report, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(interviewer.labels(), ["Unknown 1"], "{output}");
        assert_eq!(report.skipped, 2, "{output}");
        assert_eq!(
            interviewer.working_passes.get(),
            5,
            "the countdown is untouched, so the exit never went through the fixed point: \
             {output}"
        );
    }

    /// TASK-046.06.01 acceptance criterion #7: which voices a session offers and whether a
    /// session with nothing unresolved is visited are two decisions, decidable apart.
    #[test]
    fn visiting_a_resolved_session_is_decided_apart_from_which_voices_it_offers() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        // Nothing is unresolved: both voices are identified before the run starts.
        enrolled(&[("Alice", voice(0)), ("Bob", voice(1))], &paths);

        // Offer the named voices, but do not visit a session that has nothing unresolved.
        let mut skipping = Scripted::default();
        let (report, output) = run_over(
            &paths,
            &[],
            None,
            CORRECT,
            Sessions::Unresolved,
            Enrolment::default(),
            &mut skipping,
        );
        assert!(skipping.seen.is_empty(), "{output}");
        assert_eq!(report.passed_over, 1, "{output}");

        // Same offer, visited anyway, which is what `--correct` asks for. The pair above and
        // below is the split itself: the frame takes the first combination and `--correct` the
        // second, off the same `Offer`.
        let mut asking = Scripted::default();
        let (report, output) = run_over(
            &paths,
            &[],
            None,
            CORRECT,
            Sessions::Every,
            Enrolment::default(),
            &mut asking,
        );
        assert_eq!(asking.labels(), ["Alice", "Bob"], "{output}");
        assert_eq!(report.passed_over, 0, "{output}");

        // And visiting cannot manufacture a question: with the named voices left out there are
        // no candidates at all, so the session is still passed over.
        let mut empty_handed = Scripted::default();
        let (report, output) = run_over(
            &paths,
            &[],
            None,
            Offer::default(),
            Sessions::Every,
            Enrolment::default(),
            &mut empty_handed,
        );
        assert!(empty_handed.seen.is_empty(), "{output}");
        assert_eq!(report.passed_over, 1, "{output}");
    }

    /// TASK-046.06.01 acceptance criterion #9: every snippet crosses the seam, so the cap
    /// belongs to whatever is displaying them.
    #[test]
    fn a_voice_carries_every_snippet_it_has() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        with_turns(
            &paths,
            &session,
            "20260809-052600",
            (0..5)
                .map(|i| speaker_turn(f64::from(i), 0, "Unknown 1", &format!("line {i}")))
                .collect(),
        );

        let mut interviewer = Scripted::default();
        let (_, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(
            interviewer.seen[0].snippets,
            ["line 0", "line 1", "line 2", "line 3", "line 4"],
            "{output}"
        );
    }

    /// TASK-046.06.01 acceptance criterion #10: the universe `resolve()` requires is carried
    /// across the seam, and it is not the ranking -- which is exactly the failure that doc
    /// names, reproduced here.
    #[test]
    fn a_voice_carries_every_enrolled_name_and_not_only_the_ranked_ones() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        // "Stale" holds one reference of a dimension nothing in this session can be compared
        // to, so the ranking drops them -- and a typed "Stale" must still find them.
        enrolled(&[("Alice", voice(0)), ("Stale", vec![1.0; 8])], &paths);

        let mut interviewer = Scripted::default();
        let (_, output) = run(&paths, &[], &mut interviewer);

        assert_eq!(
            interviewer.seen[0].offered(),
            [("Alice", 1)],
            "an incomparable reference cannot be ranked: {output}"
        );
        assert_eq!(
            interviewer.seen[0].enrolled,
            ["Alice", "Stale"],
            "but both people are enrolled, and resolving a name is about who is there: {output}"
        );
    }

    // --- The one-remote-speaker assertion ----------------------------------------------------

    /// A session with `n` voices, each on its own orthogonal axis and first speaking in id
    /// order, so "Unknown N" is the cluster with id N - 1. All but the last eleven clear the
    /// reference floor at distinct lengths; those sit below it. The shape real clustering leaves
    /// when one person is split into many fragments.
    fn make_many_cluster_session(paths: &Paths, id: &str, n: usize) -> SessionPaths {
        let parsed = SessionId::parse(id).unwrap();
        let session = paths.session(&parsed);
        std::fs::create_dir_all(session.dir()).unwrap();
        let metadata = session_metadata(&parsed);
        metadata.write(&session.session_json()).unwrap();
        write_speaker_wav(&session.speaker_wav());

        let clusters: Vec<SpeakerCluster> = (0..n as u32)
            .map(|i| {
                let mut cluster = cluster(i, i as f64 * 0.1, (0.5, 2.5));
                cluster.embedding = axis(i as usize, n);
                cluster.speech_seconds = if (i as usize) < n - 11 {
                    5.0 + i as f64 * 0.5
                } else {
                    0.5 + (n - i as usize) as f64 * 0.1
                };
                cluster
            })
            .collect();
        SpeakerClusters::new(parsed.clone(), clusters)
            .write(&session)
            .unwrap();

        let turns: Vec<Turn> = (0..n as u32)
            .map(|i| speaker_turn(i as f64, i, &format!("Unknown {}", i + 1), "one word"))
            .collect();
        write_transcript(
            &Transcript::new(parsed.clone(), turns),
            paths,
            &session,
            &metadata,
        );
        session
    }

    /// `run_enroll` with the assertion half of the rules filled in, returning the result rather
    /// than unwrapping it -- the interrupt test needs to see the failure.
    fn run_asserting_raw(
        paths: &Paths,
        ids: &[&str],
        name: Option<&str>,
        interviewer: &mut dyn Interviewer,
    ) -> Result<(EnrollReport, String)> {
        let requested: Vec<SessionId> =
            ids.iter().map(|id| SessionId::parse(id).unwrap()).collect();
        let mut out = Vec::new();
        let report = run_enroll(
            paths,
            &requested,
            EnrollRules {
                selector: None,
                offer: Offer::default(),
                sessions: Sessions::Unresolved,
                enrolment: Enrolment::default(),
                one_speaker: name,
                relabel_transcript: true,
                template: &TranscriptTemplate::resolve(paths, None).unwrap(),
            },
            interviewer,
            &mut Lines::new(&mut out),
        )?;
        Ok((report, String::from_utf8(out).unwrap()))
    }

    /// `run_asserting_raw`, for the tests where the run is expected to come back whole.
    fn run_asserting(
        paths: &Paths,
        ids: &[&str],
        name: Option<&str>,
        interviewer: &mut Scripted,
    ) -> (EnrollReport, String) {
        run_asserting_raw(paths, ids, name, interviewer).unwrap()
    }

    /// Acceptance criterion #1 and #2: the user asserts one remote speaker and gives that
    /// person a name, and every voice on the track reads it afterwards -- the quiet ones
    /// included, which no queue offers by default -- without anything being asked about any of
    /// them.
    #[test]
    fn an_asserted_name_reaches_every_voice_including_the_quiet_ones() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_fragmented_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(Vec::new());
        let (report, output) = run_asserting(
            &paths,
            &["20260809-052600"],
            Some("Grace"),
            &mut interviewer,
        );

        assert!(
            interviewer.seen.is_empty(),
            "nothing may be asked under an assertion: {output}"
        );
        assert_eq!(report.asserted, 4, "{output}");
        assert_eq!(report.named, 4, "{output}");
        assert_eq!(report.session_only, 3, "{output}");

        // Every voice reads the asserted name, and the mic track is untouched.
        let transcript = transcript_of(&session);
        let said = said(&transcript);
        assert_eq!(
            said.iter().filter(|(who, _, _)| *who == "Grace").count(),
            4,
            "every speaker-track turn should read as the asserted person: {said:?}"
        );
        assert!(
            said.iter().any(|(who, _, _)| *who == SPEAKER_YOU),
            "the local speaker keeps their own label: {said:?}"
        );

        // The three quiet voices are named against the session alone; the loud one holds the
        // only reference.
        let assigned = assigned_in(&session, "20260809-052600");
        assert_eq!(assigned.names.len(), 3, "{:?}", assigned.names);
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.references("Grace"), 1, "{output}");
        assert!(output.contains("one remote speaker settled"), "{output}");
    }

    /// Acceptance criterion #3: the voices the heard-at-once veto would have refused are named
    /// anyway, and each one is reported -- naming the voice it was heard at once with, which is
    /// the evidence the veto acted on -- rather than silently overridden.
    #[test]
    fn the_heard_at_once_veto_is_overridden_and_reported_for_each_voice_it_reached() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        heard_at_once(&session, 0, 1);

        let mut interviewer = Scripted::answering(Vec::new());
        let (report, output) = run_asserting(
            &paths,
            &["20260809-052600"],
            Some("Grace"),
            &mut interviewer,
        );

        // The second voice was heard at once with the first, and that pair is what the veto
        // would have refused; the first was committed before the second, so exactly one veto
        // is overridden and it is the second voice that reports it.
        assert_eq!(report.vetoes_overridden, 1, "{output}");
        assert!(output.contains("named Grace for Unknown 2"), "{output}");
        assert!(output.contains("heard at once with Unknown 1"), "{output}");
        assert!(
            output.contains("the one-remote-speaker assertion says this track is one person"),
            "{output}"
        );
        // Both keep the name regardless.
        let transcript = transcript_of(&session);
        let said = said(&transcript);
        assert!(
            said.iter().filter(|(who, _, _)| *who == "Grace").count() == 3,
            "both voices keep the asserted name: {said:?}"
        );
        assert!(output.contains("1 veto(s) overridden"), "{output}");
    }

    /// Acceptance criterion #4, the plan's D4 rule made mechanical: a hundred and one above-
    /// and-below-floor clusters do not become a hundred and one references. The existing cap
    /// does the bounding, and the ten held are the ten longest above-floor clips -- the
    /// selection is a stated rule, not a property of how many clusters the session happens to
    /// hold.
    #[test]
    fn a_hundred_and_one_voice_session_stores_ten_references_the_ten_longest_above_the_floor() {
        const VOICES: usize = 101;
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_many_cluster_session(&paths, "20260820-140414", VOICES);

        let mut interviewer = Scripted::answering(Vec::new());
        let (report, output) = run_asserting(
            &paths,
            &["20260820-140414"],
            Some("Grace"),
            &mut interviewer,
        );

        assert_eq!(report.asserted, VOICES, "{output}");
        assert_eq!(report.session_only, 11, "{output}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(
            speakers.references("Grace"),
            MAX_REFERENCES_PER_SPEAKER,
            "{output}"
        );

        // The ten longest above-floor clips are ids 80..=89, at 45.0 s up to 49.5 s: everything
        // else held is shorter, so nothing else survives the cap.
        let held: Vec<Vec<f32>> = speakers
            .speakers
            .iter()
            .map(|s| s.embedding.clone())
            .collect();
        for i in 0..VOICES {
            let expected = (80..=89).contains(&(i as u32));
            assert_eq!(
                held.contains(&axis(i, VOICES)),
                expected,
                "voice {i} should be held iff it is among the ten longest: {output}"
            );
        }

        // And every voice, quiet included, reads the name in the transcript.
        let transcript = transcript_of(&session);
        let said = said(&transcript);
        assert_eq!(said.len(), VOICES);
        assert!(said.iter().all(|(who, _, _)| *who == "Grace"), "{output}");
        assert!(
            output.contains(&format!("{VOICES} voice(s) read as Grace")),
            "{output}"
        );
    }

    /// Acceptance criterion #7, first half: the fact lands on disk before the first per-voice
    /// commit, so an interrupt between the two leaves a state that explains itself -- the
    /// assertion present, nothing derived from it yet -- and a re-run converges onto the whole.
    #[test]
    fn an_interrupt_before_the_first_commit_leaves_the_assertion_on_disk_and_a_rerun_converges() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        // Make the database unwritable: the assertion itself lives in the session directory,
        // which stays writable, so it survives while the first commit cannot reach
        // `speakers.json`.
        std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o555)).unwrap();
        let interrupted = run_asserting_raw(
            &paths,
            &["20260809-052600"],
            Some("Grace"),
            &mut Scripted::default(),
        );
        std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(
            interrupted.is_err(),
            "the first commit must fail while the database is unwritable"
        );

        // What survived is self-consistent: the fact is on disk, and nothing derived from it
        // is.
        let metadata = SessionMetadata::read(&session.session_json()).unwrap();
        assert_eq!(metadata.one_remote_speaker.as_deref(), Some("Grace"));
        assert!(!session.speaker_names_json().exists());
        assert!(!paths.speakers_json().exists());
        let transcript = transcript_of(&session);
        let before = said(&transcript);
        assert!(
            before.iter().any(|(who, _, _)| *who == "Unknown 1"),
            "no label may have moved before the first commit: {before:?}"
        );

        // And a re-run converges onto the complete state.
        let (_, output) = run_asserting(
            &paths,
            &["20260809-052600"],
            Some("Grace"),
            &mut Scripted::default(),
        );
        let transcript = transcript_of(&session);
        let after = said(&transcript);
        assert!(
            after
                .iter()
                .all(|(who, _, _)| *who == "Grace" || *who == SPEAKER_YOU),
            "the re-run must complete what the interrupt left behind: {after:?}\n{output}"
        );
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.references("Grace"), 2, "{output}");
    }

    /// Acceptance criteria #6 and #7, second half: a re-run over the state a killed run would
    /// have left -- the fact on disk, some voices already named, the transcript still carrying
    /// the old labels -- converges onto the same state a fresh run produces, and a further
    /// pass writes nothing at all.
    #[test]
    fn a_rerun_converges_from_a_partial_state_and_then_writes_nothing_new() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_fragmented_session(&paths, "20260809-052600");
        let id = SessionId::parse("20260809-052600").unwrap();

        // Two of four voices named, one reference stored, the transcript still unlabelled: a
        // run interrupted after its second commit.
        let mut metadata = SessionMetadata::read(&session.session_json()).unwrap();
        metadata.assert_one_remote_speaker("Grace".to_string());
        metadata.write(&session.session_json()).unwrap();
        let clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
        let mut names = SpeakerNames::read_or_empty(&session, &id).unwrap();
        names.assign(0, "Grace", clusters.clusters[0].embedding.clone());
        names.assign(1, "Grace", clusters.clusters[1].embedding.clone());
        names.write(&session).unwrap();
        enrolled(&[("Grace", clusters.clusters[0].embedding.clone())], &paths);

        let (report, output) = run_asserting(
            &paths,
            &["20260809-052600"],
            Some("Grace"),
            &mut Scripted::default(),
        );
        assert_eq!(report.asserted, 4, "{output}");
        let transcript = transcript_of(&session);
        let said = said(&transcript);
        assert!(
            said.iter()
                .all(|(who, _, _)| *who == "Grace" || *who == SPEAKER_YOU),
            "the re-run must complete the transcript: {said:?}\n{output}"
        );

        // A further pass is a no-op on disk: converged means byte-identical, not merely
        // equivalent.
        let before = files_under(root.path());
        let (_, output) = run_asserting(
            &paths,
            &["20260809-052600"],
            Some("Grace"),
            &mut Scripted::default(),
        );
        assert_eq!(
            files_under(root.path()),
            before,
            "a converged assertion rewrote a file: {output}"
        );
    }

    /// The displacement D4 states: references another name built from this very track are
    /// withdrawn when the assertion names the track's one person, because the user has just
    /// said the evidence belongs to somebody else.
    #[test]
    fn an_assertion_displaces_references_that_another_name_built_from_this_track() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        let clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
        // Both voices were previously enrolled as Bob from this track.
        enrolled(
            &[
                ("Bob", clusters.clusters[0].embedding.clone()),
                ("Bob", clusters.clusters[1].embedding.clone()),
            ],
            &paths,
        );

        let (report, output) = run_asserting(
            &paths,
            &["20260809-052600"],
            Some("Grace"),
            &mut Scripted::default(),
        );
        assert_eq!(report.asserted, 2, "{output}");

        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.references("Grace"), 2, "{output}");
        assert!(
            speakers.speakers.iter().all(|s| s.name == "Grace"),
            "Bob's evidence from this track is withdrawn: {:?}",
            speakers.speakers
        );
    }

    /// Acceptance criterion #9, across sessions: asserting one session's track leaves every
    /// other session's files byte-identical.
    #[test]
    fn asserting_one_session_leaves_the_other_sessions_byte_identical() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let asserted = make_session(&paths, "20260809-052600");
        let bystander = make_session(&paths, "20260810-052600");

        let before = files_under(bystander.dir());
        let (_report, output) = run_asserting(
            &paths,
            &["20260809-052600"],
            Some("Grace"),
            &mut Scripted::default(),
        );
        assert_eq!(
            files_under(bystander.dir()),
            before,
            "an assertion about one session must not touch another: {output}"
        );
        let _ = asserted;
    }

    /// The frame's half of acceptance criterion #5, at the seam: answering one voice with the
    /// assertion switches the rest of the session to it -- the quiet voices included, which the
    /// queue never offered -- and the headless flag and this answer land the same state.
    #[test]
    fn answering_a_voice_with_the_assertion_switches_the_rest_of_the_run_to_it() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_fragmented_session(&paths, "20260809-052600");

        let mut interviewer = Scripted::answering(vec![Answer::OneSpeaker("Grace".to_string())]);
        let (report, output) = run(&paths, &["20260809-052600"], &mut interviewer);

        assert_eq!(
            interviewer.seen.len(),
            1,
            "only the voice the key was pressed on may be asked: {output}"
        );
        assert_eq!(
            report.asserted, 4,
            "the assertion reaches the quiet voices too: {output}"
        );

        let metadata = SessionMetadata::read(&session.session_json()).unwrap();
        assert_eq!(metadata.one_remote_speaker.as_deref(), Some("Grace"));
        let transcript = transcript_of(&session);
        let said = said(&transcript);
        assert!(
            said.iter()
                .all(|(who, _, _)| *who == "Grace" || *who == SPEAKER_YOU),
            "every voice reads the asserted name: {said:?}\n{output}"
        );
        assert!(output.contains("one remote speaker asserted"), "{output}");
        assert!(output.contains("one remote speaker settled"), "{output}");
    }

    /// The guards at the edge of the mode: a name of nothing but spaces is a request not
    /// served rather than a silent no-op, and the assertion needs exactly one session id.
    #[test]
    fn the_assertion_guards_refuse_without_writing_anything() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        make_session(&paths, "20260810-052600");
        let before = files_under(root.path());

        let (_, output) = run_asserting(
            &paths,
            &["20260809-052600"],
            Some("   "),
            &mut Scripted::default(),
        );
        assert!(output.contains("nothing but spaces"), "{output}");

        let (_, output) = run_asserting(
            &paths,
            &["20260809-052600", "20260810-052600"],
            Some("Grace"),
            &mut Scripted::default(),
        );
        assert!(output.contains("exactly one session id"), "{output}");

        assert_eq!(
            files_under(root.path()),
            before,
            "a refused guard writes nothing"
        );
    }

    /// An up-front name beside the assertion is not refused: the assertion selects every voice
    /// in the session itself, so a name waiting for a voice has all of them at once, and the
    /// answerer is never consulted at all.
    #[test]
    fn an_upfront_name_is_not_refused_beside_an_assertion() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let mut given = GivenName::new("Unused");
        let (report, output) =
            run_asserting_raw(&paths, &["20260809-052600"], Some("Grace"), &mut given).unwrap();
        assert_eq!(
            report.asserted, 2,
            "the assertion outranks the up-front name: {output}"
        );
        assert!(!output.contains("needs a voice"), "{output}");
    }

    /// TASK-050.01 acceptance criterion #4: the same assertion triggered from the full-screen
    /// frame (`Answer::OneSpeaker`) and from the headless flag leaves byte-identical on-disk
    /// state. One commit loop, two doors into it -- the frame contributes exactly one value,
    /// the answer, and everything else is shared.
    #[test]
    fn the_frame_door_and_the_headless_door_leave_byte_identical_state() {
        // Two identically seeded fresh roots: the same fixture builder, the same session id,
        // and a heard-at-once pair so the veto evidence is present for both doors.
        let headless_root = tempfile::tempdir().unwrap();
        let headless = Paths::new(headless_root.path());
        let headless_session = make_fragmented_session(&headless, "20260809-052600");
        heard_at_once(&headless_session, 0, 1);

        let frame_root = tempfile::tempdir().unwrap();
        let frame = Paths::new(frame_root.path());
        let frame_session = make_fragmented_session(&frame, "20260809-052600");
        heard_at_once(&frame_session, 0, 1);

        // The headless door: the flag. The frame door: the answer the key produces.
        let mut headless_interviewer = Scripted::answering(Vec::new());
        let (headless_report, _) = run_asserting(
            &headless,
            &["20260809-052600"],
            Some("Grace"),
            &mut headless_interviewer,
        );
        let mut frame_interviewer =
            Scripted::answering(vec![Answer::OneSpeaker("Grace".to_string())]);
        let (frame_report, _) = run(&frame, &["20260809-052600"], &mut frame_interviewer);

        // The write-relevant counts agree. The full reports are not compared field by field:
        // the frame door builds the queue before the assertion arrives mid-run, so it counts
        // the below-floor voices as held back, while the headless door never reaches the queue
        // at all. Prompting bookkeeping differs; what the runs leave behind must not.
        assert_eq!(headless_report.named, frame_report.named);
        assert_eq!(headless_report.session_only, frame_report.session_only);
        assert_eq!(headless_report.asserted, frame_report.asserted);
        assert_eq!(
            headless_report.vetoes_overridden,
            frame_report.vetoes_overridden
        );
        // The trees each run left are identical file by file. `files_under` returns absolute
        // paths, so strip the root first -- the claim is about the tree, not about where the
        // tempdir happened to live. And the transcript header carries the wall clock of the
        // run that rewrote it (`updated:`), which sits outside either door's control: two runs
        // straddling a second boundary would differ there and nowhere else, so that one line
        // is normalised and every other byte compared as written.
        let normalise = |path: &Path, bytes: &[u8]| -> Vec<u8> {
            // The transcript alone carries the clock; every other file compares byte for byte
            // as written.
            if path.file_name().is_some_and(|name| name == "transcript.md")
                && let Ok(text) = std::str::from_utf8(bytes)
            {
                text.lines()
                    .map(|line| {
                        if line.starts_with("updated:") {
                            "updated: <the clock>".to_string()
                        } else {
                            line.to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
                    .into_bytes()
            } else {
                bytes.to_vec()
            }
        };
        let tree = |root: &Path| {
            files_under(root)
                .into_iter()
                .map(|(path, bytes)| {
                    (
                        path.strip_prefix(root).unwrap().to_path_buf(),
                        normalise(&path, &bytes),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(tree(headless_root.path()), tree(frame_root.path()));
    }

    /// TASK-050.01: the preview's counts are the run's own numbers, not a re-derivation of them
    /// -- on the fragmented fixture with its heard-at-once pair, what `Preview::one_speaker`
    /// promises is what the run reports once it has run.
    #[test]
    fn the_assertion_preview_counts_match_the_run_s_override_report() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_fragmented_session(&paths, "20260809-052600");
        heard_at_once(&session, 0, 1);

        let clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
        let unknown = unknown_labels(
            clusters
                .clusters
                .iter()
                .map(|c| (c.id, c.first_spoke_seconds)),
        );
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        let assigned =
            SpeakerNames::read_or_empty(&session, &SessionId::parse("20260809-052600").unwrap())
                .unwrap();
        let preview = Preview::new(
            &clusters.clusters,
            &unknown,
            &speakers,
            &assigned,
            &clusters.clusters[0],
            Enrolment::default(),
            None,
            &[],
        );
        let assertion = preview.one_speaker("Grace").unwrap();

        let mut interviewer = Scripted::answering(Vec::new());
        let (report, output) = run_asserting(
            &paths,
            &["20260809-052600"],
            Some("Grace"),
            &mut interviewer,
        );

        assert_eq!(assertion.voices, report.asserted);
        assert_eq!(assertion.vetoes_overridden, report.vetoes_overridden);
        assert!(output.contains("4 voice(s) will read as Grace"));
        assert!(output.contains("1 veto(s) overridden"));
    }

    /// The frame door's interrupt rule, TASK-050.01 acceptance criterion #4: the fact lands in
    /// `session.json` before the first commit on this door too, so a failure between the two
    /// leaves a state that explains itself -- the assertion present, no partial labels -- and
    /// a re-run converges.
    #[test]
    fn an_interrupt_after_the_frame_door_fact_leaves_the_assertion_and_a_rerun_converges() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");

        // The frame door writes the fact after the answer comes back, so the database goes
        // unwritable at the moment the answer is given: the fact (an existing file rewritten in
        // place) still lands, and the first commit, which creates `speakers.json`, cannot.
        let mut interviewer = UnwritableAfterFact(root.path().to_path_buf());
        let mut out = Vec::new();
        let interrupted = run_enroll(
            &paths,
            &[SessionId::parse("20260809-052600").unwrap()],
            EnrollRules {
                selector: None,
                offer: Offer::default(),
                sessions: Sessions::Unresolved,
                enrolment: Enrolment::default(),
                one_speaker: None,
                relabel_transcript: true,
                template: &TranscriptTemplate::resolve(&paths, None).unwrap(),
            },
            &mut interviewer,
            &mut Lines::new(&mut out),
        );
        assert!(
            interrupted.is_err(),
            "the failed commit must surface as an error"
        );

        // Survivors: the fact is on disk; nothing was written into the label stores; the
        // transcript still shows the unknowns.
        let metadata = SessionMetadata::read(&session.session_json()).unwrap();
        assert_eq!(metadata.one_remote_speaker.as_deref(), Some("Grace"));
        assert!(!session.speaker_names_json().exists());
        assert!(!paths.speakers_json().exists());
        let first_transcript = transcript_of(&session);
        let before = said(&first_transcript);
        assert!(
            before.iter().any(|(who, _, _)| *who == "Unknown 1"),
            "no label may have moved before the first commit: {before:?}"
        );

        // A re-run through the same door converges: the fact is already there, so the switch
        // skips the write and commits every voice against it.
        std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        let mut rerun = Scripted::answering(vec![Answer::OneSpeaker("Grace".to_string())]);
        let (report, output) = run(&paths, &["20260809-052600"], &mut rerun);
        assert_eq!(report.asserted, 2, "{output}");
        let second_transcript = transcript_of(&session);
        let after = said(&second_transcript);
        assert!(
            after
                .iter()
                .all(|(who, _, _)| *who == "Grace" || *who == SPEAKER_YOU),
            "the re-run must complete what the interrupt left behind: {after:?}\n{output}"
        );
    }

    /// The frame door's answer, with the database made unwritable the moment it is given: the
    /// fact lands and the first commit fails, exactly as the headless flag's interrupt test
    /// arranges it for its own door.
    struct UnwritableAfterFact(PathBuf);

    impl Interviewer for UnwritableAfterFact {
        fn identify(&mut self, _voice: &Voice<'_>) -> Answer {
            std::fs::set_permissions(self.0.as_path(), std::fs::Permissions::from_mode(0o555))
                .unwrap();
            Answer::OneSpeaker("Grace".to_string())
        }
    }
}
