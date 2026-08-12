//! Naming the voices transcription could not identify.
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
//! # This crate also *reports* on the database it writes
//!
//! [`run_speakers`] answers the question the file cannot: who is enrolled, and what is each
//! stored recording of them actually naming. It lives here rather than beside `speakers.json`
//! because the answer is not a fact about that file -- it is derived by labelling every session
//! on disk twice, once with the database as it stands and once with one row removed, which is
//! the two-labelling diff `enroll_session` already performs over a single session before it
//! honours an answer. See the `references` module for the derivation and its cost. Nothing on
//! that path writes anything.

mod references;

pub use references::{Enrolled, Reference, Scan, Unreadable, VoiceChange, run_speakers, scan};

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use meethook_session::{
    AssignedName, Classification, DiscoveredSession, EnrolledSpeakers, Paths, SessionId,
    SourceTrack, SpeakerCluster, SpeakerClusters, SpeakerNames, Stored, Transcript,
    discover_sessions, unknown_labels, unknown_speaker,
};
use meethook_transcribe::{
    Attribution, Naming, TARGET_RATE, attributions, identify_clusters, read_track_16k_mono,
};

/// How many of a voice's lines to show before asking who it is.
///
/// Enough to hear a person in the words -- what they said, what they were asked -- without
/// turning a prompt into a page of transcript that hides the question at the bottom of it.
const SNIPPETS: usize = 3;

/// How much of one line to show. Long enough for a sentence, short enough to stay on a line.
const SNIPPET_CHARS: usize = 100;

/// How much a voice has to have spoken before it is worth a question.
///
/// A rule about the prompt queue and nothing else. A cluster below this still keeps its
/// "Unknown N", still holds its turns, and is still relabelled when somebody else's answer
/// turns out to name it -- it is only not *asked about* unless `enroll --all` is passed.
/// Nothing on disk depends on it.
///
/// # Where 5 s comes from
///
/// Clustering emits one cluster per voice it is sure of plus a long tail of fragments it
/// cannot place, and the tail is not a tuning failure: a one-second embedding describes a
/// phoneme and a prosody rather than a person, so no distance rule puts it anywhere. On
/// session `20260810-093047` -- seven people, 1368.7 s of speech -- the shipped clustering
/// leaves 56 clusters, 8 of which identification resolves, so without a floor `enroll` asks
/// 48 questions about a meeting with seven people in it.
///
/// Sorted by talk time those 56 clusters run 426.8 / 423.7 / 124.8 / 119.5 / 96.0 / 51.5 --
/// the six voices the user confirms are the six main speakers, 1242.2 s between them -- and
/// then fall off a cliff to 8.6 / 8.5 / 7.8 / 7.5 / 6.0 / 5.6 / 5.4 / 4.9 / 4.2 / 3.9 / ...
/// into a tail where 29 of the 56 hold under two seconds and 126.5 s covers all fifty of them.
/// Of the 48 clusters left unresolved after identification, **every floor `f` with
/// `4.9 < f <= 5.9` offers the same seven voices and holds back the same 41**; over all 56
/// clusters, ignoring which happen to be enrolled, the partition is fixed across
/// `4.9 < f <= 5.4`. 5 s is the round number in that band rather than a value fitted to this
/// recording.
///
/// Both edges are consequences. Above 7.8 s Alex -- a real seventh participant,
/// 9.8 s of speech split across clusters of 7.83 s and 1.99 s -- stops being offered and can
/// only be reached through `--all`, which is the failure TASK-021 AC #3 names; he happens to
/// be enrolled already in this session, so the cost lands on the next participant like him.
/// Below it the tail arrives fast: 9 voices at a 4 s floor, 15 at 3 s, 21 at 2 s, which is the
/// 48-question prompt again with a smaller number on it.
///
/// # Not [`meethook_transcribe::SPEAKER_FLOOR_SECONDS`], and not [`REFERENCE_FLOOR_SECONDS`]
///
/// Same units, three different questions, and they do not imply one another:
///
/// - `SPEAKER_FLOOR_SECONDS` (30 s) decides **which clusters are solid enough to adopt
///   fragments into** -- how much evidence a centroid rests on before it is allowed to claim
///   somebody else's turns. It is necessarily the larger: at 30 s the seventh participant
///   would not be asked about at all.
/// - This one decides **which voices are worth asking about**. Getting it wrong costs a
///   question, in one direction or the other, and nothing else.
/// - [`REFERENCE_FLOOR_SECONDS`] decides **which answers become references in
///   `speakers.json`**. Naming somebody who spoke 8 s is right; storing a reference built from
///   8 s of audio is what TASK-019 measured going wrong. It landed on the same 5.0 s this one
///   sits on, which is why both state the same boundary convention below: a value offered here
///   and then refused there would be a question asked for nothing.
///
/// The comparison is `speech_seconds >= PROMPT_FLOOR_SECONDS`, the same convention
/// `SPEAKER_FLOOR_SECONDS` states: a cluster sitting exactly on the floor is offered. Two
/// floors in one codebase disagreeing about their own boundary is a bug waiting to happen.
const PROMPT_FLOOR_SECONDS: f64 = 5.0;

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
const REFERENCE_FLOOR_SECONDS: f64 = 5.0;

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

/// Where a voice sits in the questions this run has for one session: "the 2nd of 9".
///
/// Reads as `2/9`. The point of it is that an interview otherwise has no visible end -- the
/// count printed on the session line has scrolled away behind the snippets and the clips by
/// the second or third question -- so every prompt carries the same number back.
///
/// Two things it deliberately is not:
///
/// - `of` counts the voices this run *offered for this session*, which is the number the
///   session line already printed. It is not a run-wide total, because that would mean reading
///   every session up front, and it does not include the voices held back under
///   `PROMPT_FLOOR_SECONDS`, which are reported on their own clause and are not questions this
///   run will ask.
/// - `nth` is the voice's place in that queue, not a tally of the questions actually asked. An
///   answer can name a voice further down the queue -- clustering splitting one person in two
///   -- and that voice is then passed over, so a number can be skipped: 1/4, 2/4, 4/4. The gap
///   is the honest reading, because it means `of - nth` is a true upper bound on the questions
///   left rather than a promise of more questions than the run will ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    /// 1-based place in the queue this run offered for this session. Never greater than `of`.
    pub nth: usize,
    /// How many voices this run offered for this session.
    pub of: usize,
}

impl std::fmt::Display for Position {
    /// One place decides the form, so no two [`Interviewer`]s can disagree about it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.nth, self.of)
    }
}

/// One voice being asked about, and everything needed to ask.
///
/// Usually a voice nothing in the database matched, which is what `enroll` exists for. Under
/// [`Offer::named`] it can also be one the database has already put a name to, and then the
/// question is a different one -- not "who is this" but "is this right" -- which is what
/// `confidence` tells the caller.
///
/// Deliberately one value rather than a play-then-ask pair of calls: the order those two
/// would have to be made in is exactly the sort of thing a seam should not be leaking.
pub struct Voice<'a> {
    pub session: &'a SessionId,

    /// Which of this session's questions this is, and how many there are.
    ///
    /// Carried across the seam rather than counted on the far side, because an [`Interviewer`]
    /// counting its own calls would be counting a different thing -- the questions asked, not
    /// the queue -- and could not know the total at all. Restarts at 1 for each session of a
    /// run; `session` above says which one.
    pub position: Position,

    /// What this voice is currently called, and on what basis.
    ///
    /// One field rather than a label plus a confidence, because the prompt turns on *which
    /// kind* of label this is and only one of the three carries a number. Reading "has a
    /// confidence" as "already has an answer" is right for exactly as long as an identification
    /// is the only way to get a name onto a voice; [`Attribution::Assigned`] is a name with no
    /// similarity behind it, and a prompt written against the number would ask "who is this?"
    /// about a voice this command named an hour ago.
    ///
    /// [`Attribution::label`] is exactly as the transcript reads -- "Unknown 2", or the name --
    /// so the user can find the voice in the file in front of them.
    pub attribution: &'a Attribution,

    /// Total speech attributed to this voice, in seconds. How the user tells a participant
    /// from someone who coughed once.
    pub speech_seconds: f64,

    /// Up to `SNIPPETS` of what this voice said, whitespace-trimmed and cut to
    /// `SNIPPET_CHARS` characters. Empty if the recogniser heard nothing over it.
    pub snippets: Vec<&'a str>,

    /// The longest representative clip: 16 kHz mono, the same rate everything else in
    /// meethook works in.
    ///
    /// Empty when `speaker.wav` is missing or unreadable, which is a voice that can still be
    /// named from its snippets rather than a session that has to fail.
    pub clip: &'a [f32],
}

/// What the user said when asked who a voice is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answer {
    Named(String),
    Skip,
    /// End the run here. A variant rather than an error because stopping early is an
    /// ordinary outcome -- everything accepted so far is already on disk.
    Quit,
}

/// Asks a user who one voice is.
///
/// Infallible on purpose. A terminal that cannot play audio still has an answer, and one
/// that cannot be read has `Quit`; making this fallible would push terminal errors into the
/// sequencing, which is the one place this design keeps them out of.
pub trait Interviewer {
    fn identify(&mut self, voice: &Voice<'_>) -> Answer;
}

/// Which one voice a run is about, when it is about one voice.
///
/// `--voice`. The queue is the right shape for "I have not named anybody here yet" and the
/// wrong one for the commonest follow-up -- one voice the user can now place, or one name
/// that is wrong -- where reaching it means pressing Enter past everybody else, and every one
/// of those presses is a chance to type a name onto the wrong person.
///
/// # What it selects
///
/// One selector matched two ways, so the user does not have to know which kind of thing they
/// are holding:
///
/// - **A number** is the number in "Unknown 3", not the cluster id. The cluster id appears in
///   `transcript.json` and nowhere a person reads, while the "Unknown N" is on every prompt
///   header and every unnamed line of the transcript -- so accepting both would be two
///   numbering systems on one flag, silently targeting the wrong voice whenever they disagree.
///   The number comes from [`unknown_labels`], which ranks *every* voice by first appearance
///   whether or not it has a name, so it is defined for named voices too and does not move
///   when one of them is named.
/// - **A name** is what the voice currently reads as: the enrolled name that matched it, the
///   name somebody gave it for this session, or its own "Unknown 3" written out.
///
/// Matching is exact after trimming -- `alice` and `Alice` are two people here as everywhere
/// else in this file. A miss costs one retry, because it prints what the session does contain.
///
/// # What it overrides
///
/// Both [`Offer`] filters, for its one voice: a targeted voice is asked about whether it is
/// under `PROMPT_FLOOR_SECONDS` and whether the database has already named it. Naming somebody
/// specific is exactly the judgement those two gates make on the user's behalf when they have
/// not made it themselves. It does not touch [`Enrolment`], which is the other axis: what an
/// answer *writes* is the same however the question came to be asked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceSelector(String);

impl VoiceSelector {
    /// Whether this selector means this voice, given the "Unknown N" it was transcribed with
    /// and what it currently reads as.
    ///
    /// Both arms, so that a number keeps pointing at the same voice after that voice has been
    /// named, and a name reaches a voice whose number the user never saw.
    fn matches(&self, unknown: &str, shown: &Attribution) -> bool {
        self.0 == unknown || self.0 == shown.label()
    }
}

impl From<&str> for VoiceSelector {
    /// Normalises to a label, so `3` and `Unknown 3` are the same selector from here on.
    ///
    /// Infallible: everything that is not a number is a name, and a name that matches nothing
    /// is reported against the session's actual voices rather than refused at the edge, where
    /// there is nothing to compare it to yet.
    fn from(raw: &str) -> VoiceSelector {
        let trimmed = raw.trim();
        match trimmed.parse::<usize>() {
            Ok(number) => VoiceSelector(unknown_speaker(number)),
            Err(_) => VoiceSelector(trimmed.to_string()),
        }
    }
}

impl std::fmt::Display for VoiceSelector {
    /// The normalised form, which is what was matched against: a user who passed `3` and
    /// missed is told that "Unknown 3" is what was looked for, beside the labels that exist.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Which voices a run offers beyond the ones it offers by default.
///
/// Two orthogonal questions -- how quiet a voice may be, and whether the database has already
/// named it -- deliberately not one flag, because `--all` already answers the first and a user
/// who wants to correct one identification is not asking to be shown the two-second fragments
/// as well. The two filters compose: the floor decides whether a voice is worth a question
/// whatever put it in the list.
///
/// Both filters are overridden, for one voice, by a [`VoiceSelector`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Offer {
    /// `--all`: voices below `PROMPT_FLOOR_SECONDS`, which are normally held back.
    pub quiet: bool,

    /// `--correct`: voices the database has already put a name to, so a wrong identification
    /// can be answered instead of being permanent.
    pub named: bool,
}

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
/// Private, and a bundle rather than three parameters, because it is threaded through the walk
/// over sessions unchanged and every function on that path would otherwise carry all of them.
/// The axes stay separate in the public signature, where a caller has to say which is which.
#[derive(Debug, Clone, Copy, Default)]
struct Rules<'a> {
    /// `Some` replaces the queue with one voice; `None` is the queue. Not a fourth flag on
    /// [`Offer`], because it does not widen the queue -- it stands in for it.
    selector: Option<&'a VoiceSelector>,
    offer: Offer,
    enrolment: Enrolment,
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
/// off another voice -- see `cost_of`. Not a `skipped`: the user answered, and the answer was
/// declined rather than absent. Not a `failed` either, since nothing went wrong and the run
/// carries on; it is counted so the summary can say a question was answered and came to
/// nothing, which is the one outcome a silent revert used to hide.
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
/// A [`VoiceSelector`] replaces the queue with exactly one voice of one session, and needs
/// exactly one session id to be meaningful: a voice number says nothing across sessions, and a
/// name would fan out over every recording on disk. It overrides both [`Offer`] filters for
/// that voice, so passing `--all` or `--correct` beside it changes nothing rather than
/// conflicting with it.
///
/// [`Offer`] widens which voices get asked about -- the quiet ones, the already-named ones, or
/// both. It changes which questions get asked and nothing else: the same answers write the
/// same files however a voice came to be offered. [`Enrolment`] is the other axis, and the
/// only one that changes *what an answer writes* -- which is exactly why the override is a
/// parameter of its own instead of a third field on `Offer`. There are three files an answer
/// can land in now (`speakers.json`, a session's `speaker_names.json`, and its transcript),
/// and which of the first two it is depends on the voice's duration and on this.
pub fn run_enroll(
    paths: &Paths,
    requested: &[SessionId],
    voice: Option<&VoiceSelector>,
    offer: Offer,
    enrolment: Enrolment,
    interviewer: &mut dyn Interviewer,
    out: &mut dyn Write,
) -> Result<EnrollReport> {
    let mut report = EnrollReport::default();

    // Enforced here rather than in the CLI's argument parser, because this is where the sibling
    // rule already lives -- a requested id that is not on disk is printed and counted below --
    // and because one enforcement point cannot disagree with itself. Refused before anything is
    // discovered: a run that cannot say which session it is about has nothing to read.
    if voice.is_some() && requested.len() != 1 {
        writeln!(
            out,
            "--voice needs exactly one session id: a voice belongs to one session, so its \
             number and its name mean nothing across several"
        )?;
        report.failed += 1;
        return Ok(report);
    }

    let discovered = discover_sessions(paths)?;

    for id in requested {
        if !discovered.iter().any(|session| &session.id == id) {
            writeln!(out, "{id}  not found")?;
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
        writeln!(
            out,
            "No sessions found in {}",
            paths.sessions_dir().display()
        )?;
        return Ok(report);
    }

    let mut speakers = EnrolledSpeakers::read_or_empty(paths)?;
    let rules = Rules {
        selector: voice,
        offer,
        enrolment,
    };

    for session in selected {
        match enroll_session(
            paths,
            session,
            rules,
            &mut speakers,
            interviewer,
            out,
            &mut report,
        )? {
            Outcome::Finished => {}
            Outcome::Quit => break,
        }
    }

    Ok(report)
}

/// Asks about every unresolved voice in one session, writing after each accepted name.
///
/// The files are written in a fixed order -- whichever of `speakers.json` and this session's
/// `speaker_names.json` the answer belongs in, then the transcript -- and after every single
/// name rather than once at the end. The name file is what the next labelling reads, so an
/// interrupt between the two writes leaves a name the next run simply re-applies, rather than
/// a transcript naming somebody nothing on disk records. It is also what makes ending a run
/// early cost nothing that was already answered.
///
/// A session this cannot read is reported and counted, and the queue carries on: one session
/// transcribed by a build too old to have recorded first appearances must not end the run.
fn enroll_session(
    paths: &Paths,
    session: &DiscoveredSession,
    rules: Rules<'_>,
    speakers: &mut EnrolledSpeakers,
    interviewer: &mut dyn Interviewer,
    out: &mut dyn Write,
    report: &mut EnrollReport,
) -> Result<Outcome> {
    match session.classification {
        Classification::Orphaned => {
            writeln!(
                out,
                "{}  passed over: no session.json (the recorder crashed mid-session)",
                session.id
            )?;
            report.passed_over += 1;
            return Ok(Outcome::Finished);
        }
        Classification::Valid => {
            writeln!(out, "{}  passed over: not transcribed yet", session.id)?;
            report.passed_over += 1;
            return Ok(Outcome::Finished);
        }
        Classification::Transcribed => {}
    }

    let clusters = match SpeakerClusters::read(&session.paths.speaker_clusters_json()) {
        Ok(clusters) => clusters,
        // The expected instance of this is a `speaker_clusters.json` from before first
        // appearances were recorded: without them an "Unknown 2" cannot be mapped back to a
        // voice at all, so the file is refused rather than read with a defaulted zero.
        Err(e) => {
            writeln!(
                out,
                "{}  failed: {e} -- re-transcribe this session with --force",
                session.id
            )?;
            report.failed += 1;
            return Ok(Outcome::Finished);
        }
    };
    let mut transcript = match Transcript::read(&session.paths.transcript_json()) {
        Ok(transcript) => transcript,
        // As above, and with the same remedy: the expected instance is a `transcript.json`
        // from before turns recorded which cluster they came from. A user told only "missing
        // field `cluster`" has been given a diagnosis with no next step.
        Err(e) => {
            writeln!(
                out,
                "{}  failed: {e} -- re-transcribe this session with --force",
                session.id
            )?;
            report.failed += 1;
            return Ok(Outcome::Finished);
        }
    };

    // Voices somebody named in this session without enrolling them. Read here, beside the
    // clusters, so the relabel below already honours them -- a name given in an earlier run is
    // part of what this session's transcript should say, exactly as an enrolled name is.
    let mut assigned = match SpeakerNames::read_or_empty(&session.paths, &session.id) {
        Ok(assigned) => assigned,
        // Unlike the two failures above, no re-transcribe recovers this one: this file holds
        // names a person typed and nothing else can regenerate them, so the only honest
        // instruction is to go and look at it.
        Err(e) => {
            writeln!(
                out,
                "{}  failed: {e} -- fix or delete that file",
                session.id
            )?;
            report.failed += 1;
            return Ok(Outcome::Finished);
        }
    };

    // The "Unknown N" numbering the transcript was written with, recovered from the clusters
    // file by the one function `transcribe` labels with. Fixed for the whole session: it is a
    // fact about when each voice first spoke, which no answer below changes.
    let unknown = unknown_labels(
        clusters
            .clusters
            .iter()
            .map(|c| (c.id, c.first_spoke_seconds)),
    );
    // What each voice should be called given the database as it stands.
    let mut shown = effective_labels(&clusters.clusters, &unknown, speakers, &assigned.names);

    // The transcript may predate an answer given in an earlier session -- name somebody in
    // January's meeting and February's transcript still calls them Unknown 2 -- so it is
    // brought in line before anything is asked. Doing it here rather than only after a name
    // is what stops a session with nothing left to ask about from keeping a stale label
    // forever, since it would be passed over on every later run too. Nothing is written when
    // nothing differs.
    if relabel(&mut transcript, &shown) {
        transcript.write(&session.paths)?;
        writeln!(out, "{}  transcript brought up to date", session.id)?;
    }

    // First-appearance order, which is "Unknown 1, Unknown 2, ..." -- the order the user
    // reads the transcript in. Talk-time order would put the most-worth-naming voice first
    // and jump around relative to the file they are looking at.
    let mut order: Vec<&SpeakerCluster> = clusters.clusters.iter().collect();
    order.sort_by(|a, b| {
        a.first_spoke_seconds
            .total_cmp(&b.first_spoke_seconds)
            .then(a.id.cmp(&b.id))
    });

    // Which voices this run is about: one the user named, or the queue. The only thing a
    // selector changes -- everything from here down runs on whichever list comes back, so a
    // targeted prompt is not a second implementation of a prompt, it is the same one asked
    // about a shorter list. `None` is a session that is finished and has said why.
    let offered = match rules.selector {
        Some(selector) => targeted(selector, &order, &unknown, &shown, session, out, report)?,
        None => queue(&order, &shown, rules.offer, session, out, report)?,
    };
    let Some(offered) = offered else {
        return Ok(Outcome::Finished);
    };

    // Read after that check, so a session with nothing to ask about never resamples an hour
    // of audio in order to then ask nothing. Unreadable is empty rather than fatal: a voice
    // with no clip can still be named from its snippets.
    let track = read_track_16k_mono(&session.paths.speaker_wav()).unwrap_or_default();

    // What each voice was called when this queue was built. The guard below compares against
    // *this* rather than against the live labels, because under `--correct` a queued voice may
    // legitimately be one the database had already named.
    let baseline = shown.clone();

    // The total every prompt below carries. Read off the same list the session line counted
    // one call ago, so the two cannot drift apart.
    let of = offered.len();

    for (index, cluster) in offered.into_iter().enumerate() {
        // A voice an answer given earlier in this run has already put a name to: clustering
        // that split one person in two must not ask about them twice. Only an in-run answer
        // can have moved a label since `baseline` was taken, so "named, and not the name it
        // had when we queued it" is exactly that case and nothing else. The `is_named()`
        // half matters as much: an in-run answer can also *un*-name a voice -- re-anchoring a
        // reference to another cluster drops this one back to its "Unknown N" -- and that is a
        // question this run created and has not answered.
        //
        // This `continue` is the one place a [`Position`] skips a number, and it is skipped
        // rather than compressed on purpose: the voice really was in the queue and really is
        // now answered, so the gap says work disappeared and the end came closer.
        if shown[&cluster.id].is_named() && shown[&cluster.id] != baseline[&cluster.id] {
            continue;
        }

        // Scoped so the borrows of `transcript` and `shown` inside the voice end before the
        // answer is acted on.
        let answer = {
            let attribution = &shown[&cluster.id];
            // Keyed on the cluster, not on the label text: under `--correct` two voices can
            // sit under one enrolled name -- which is the false accept being corrected -- and
            // a prompt showing the other person's lines cannot be answered.
            let snippets: Vec<&str> = transcript
                .turns
                .iter()
                .filter(|turn| {
                    turn.source_track == SourceTrack::Speaker && turn.cluster == Some(cluster.id)
                })
                .map(|turn| snippet(&turn.text))
                .filter(|text| !text.is_empty())
                .take(SNIPPETS)
                .collect();

            interviewer.identify(&Voice {
                session: &session.id,
                position: Position { nth: index + 1, of },
                attribution,
                speech_seconds: cluster.speech_seconds,
                snippets,
                clip: clip_for(&track, cluster),
            })
        };

        // Leaving an already-named voice alone is keeping that identification, which is an
        // answer; leaving an unnamed one alone is the question going unanswered. Same write --
        // none -- and different enough that the summary must not conflate them.
        let left_alone = if shown[&cluster.id].is_named() {
            &mut report.kept
        } else {
            &mut report.skipped
        };

        let name = match answer {
            Answer::Quit => return Ok(Outcome::Quit),
            Answer::Skip => {
                *left_alone += 1;
                continue;
            }
            Answer::Named(name) => name,
        };
        // A name of nothing but spaces is somebody pressing Enter with a stray keystroke in
        // the buffer, not a request for an entry called "".
        let name = name.trim();
        if name.is_empty() {
            *left_alone += 1;
            continue;
        }

        // Naming a voice and storing a reference built from it are two different acts, and
        // this is where they come apart. Below the floor the name is recorded against the
        // session and `speakers.json` is not touched at all -- see `REFERENCE_FLOOR_SECONDS`
        // for what a reference built from two seconds of speech does to every future meeting.
        let session_only = cluster.speech_seconds < REFERENCE_FLOOR_SECONDS
            && rules.enrolment != Enrolment::Always;

        // Everything this answer would write, applied to copies first.
        //
        // Two files can carry a name -- `speakers.json` and this session's
        // `speaker_names.json` -- and both feed the same labelling, so the only way to know
        // what an answer *does* is to build the state it would leave and label the session
        // through it. That is what these copies are for, and the pre-flight below is why the
        // answer is not simply written and inspected afterwards: undoing a write that turned
        // out to cost somebody their name means writing three files back, and a run
        // interrupted mid-undo would leave exactly the mess this is here to prevent.
        let mut candidate = speakers.clone();
        let mut candidate_assigned = assigned.clone();

        // The correction, on the above-floor path only: a reference identical to this cluster
        // was built from this voice, and the user has just told us this voice is somebody
        // else, so it is a stored claim about a person it is not of and it competes as an
        // argmax in every future meeting -- winning whenever its name sorts first
        // (`identify::best_match`'s tie-break).
        let displaced = if session_only {
            Vec::new()
        } else {
            candidate.forget_reference(&cluster.embedding, name)
        };

        // What every voice reads once the correction alone has been applied. The baseline the
        // pre-flight measures against, and the reason it is two labellings rather than one: a
        // name lost *here* is the correction's documented consequence -- the user has just
        // said that reference was of somebody else -- and refusing it would undo the guarantee
        // the correction exists to keep.
        let corrected = effective_labels(
            &clusters.clusters,
            &unknown,
            &candidate,
            &candidate_assigned.names,
        );

        // The addition. `None` on the below-floor path, where no reference is stored at all.
        let stored = if session_only {
            candidate_assigned.assign(cluster.id, name, cluster.embedding.clone());
            None
        } else {
            let stored = candidate.store_reference(name, cluster.embedding.clone());
            if matches!(stored, Stored::AtCapacity { .. }) {
                // At the cap the recording is not stored, so the answer falls back to the
                // session-only path rather than being lost: the transcript still reads the
                // right person, and nothing already stored is dropped to make room.
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

        let after = effective_labels(
            &clusters.clusters,
            &unknown,
            &candidate,
            &candidate_assigned.names,
        );

        // The refusal. An answer that would take a name off a voice the user is not answering
        // about is not honoured at all -- see `cost_of` for the three ways that can happen and
        // why one check covers them. Nothing is written, the voice keeps whatever it read, and
        // the user gets a line naming the voice that would have paid.
        if let Some(cost) = cost_of(cluster.id, name, &unknown, &corrected, &after) {
            let answered = handle(cluster.id, &unknown);
            match cost {
                Cost::Vetoed {
                    holder: Some(holder),
                } => writeln!(
                    out,
                    "{}  refused {name} for {answered}: {holder} already has that name and the \
                     two were heard speaking at once, so they are not one person -- \
                     meethook enroll --correct --voice {holder} if that is the wrong one",
                    session.id
                )?,
                Cost::Vetoed { holder: None } => writeln!(
                    out,
                    "{}  refused {name} for {answered}: that name will not apply to this voice",
                    session.id
                )?,
                Cost::Taken { voice, losing } => writeln!(
                    out,
                    "{}  refused {name} for {answered}: it would take {losing} off {voice} -- \
                     meethook enroll --correct --voice {voice} if {voice} is not {losing}",
                    session.id
                )?,
            }
            report.refused += 1;
            continue;
        }

        // Committed by taking the copies the pre-flight ran against, so what lands on disk is
        // the state that was checked rather than a second construction of it.
        let speakers_changed = *speakers != candidate;
        let assignments_changed = assigned.names != candidate_assigned.names;
        *speakers = candidate;
        assigned = candidate_assigned;

        for who in &displaced {
            // An enrollment that vanishes without a line about it is worse than the bug. Two
            // wordings, because "Nate no longer has a reference" is a lie when Nate has three
            // recordings and lost one.
            if who.remaining == 0 {
                writeln!(
                    out,
                    "{}  {} no longer has a reference: that voice is {name}",
                    session.id, who.name
                )?;
            } else {
                writeln!(
                    out,
                    "{}  {} no longer has that reference: that voice is {name} -- {} keeps {} \
                     other(s)",
                    session.id, who.name, who.name, who.remaining
                )?;
            }
        }

        match stored {
            None => {
                // The case being given up by not touching the database here: a legacy
                // reference that *is* this exact fragment (built before the floor existed)
                // stays, and goes on competing as an argmax under somebody else's name.
                // Reported rather than silently left, with the override that fixes it, because
                // an enrollment that is wrong and unmentioned is worse than one that is wrong
                // and named.
                let stale: Vec<&str> = speakers
                    .speakers
                    .iter()
                    .filter(|s| s.name != name && s.embedding == cluster.embedding)
                    .map(|s| s.name.as_str())
                    .collect();
                for who in stale {
                    writeln!(
                        out,
                        "{}  {who} still has a reference built from this voice -- \
                         meethook enroll --force-reference to replace it with {name}",
                        session.id
                    )?;
                }
                writeln!(
                    out,
                    "{}  named {name} in this session only: {:.1} s of speech is under the \
                     {REFERENCE_FLOOR_SECONDS} s reference floor -- \
                     meethook enroll --force-reference to store a reference anyway",
                    session.id, cluster.speech_seconds
                )?;
                report.session_only += 1;
            }
            Some(Stored::Enrolled) => writeln!(out, "{}  enrolled {name}", session.id)?,
            Some(Stored::Added { held }) => writeln!(
                out,
                "{}  enrolled another recording of {name}: {held} reference(s) now",
                session.id
            )?,
            Some(Stored::AlreadyHeld) => writeln!(
                out,
                "{}  {name} already has a reference built from this voice",
                session.id
            )?,
            Some(Stored::AtCapacity { held }) => {
                writeln!(
                    out,
                    "{}  named {name} in this session only: {name} already holds {held} \
                     reference(s), the most meethook keeps for one person, so this recording \
                     is not stored and does not help recognise them -- remove one from {} to \
                     make room",
                    session.id,
                    paths.speakers_json().display()
                )?;
                report.session_only += 1;
            }
        }
        report.named += 1;

        // Written in a fixed order -- the database, then this session's names, then the
        // transcript -- and only where something changed, so a skipped write leaves a file
        // byte-identical rather than merely equivalent.
        if speakers_changed {
            speakers.write(paths)?;
        }
        if assignments_changed {
            assigned.write(&session.paths)?;
        }

        // Re-identified against the updated database rather than assumed: naming one voice
        // can also name a second cluster in this session, if clustering split that person in
        // two, and a `--force` re-transcribe would name both.
        let now = effective_labels(&clusters.clusters, &unknown, speakers, &assigned.names);
        if relabel(&mut transcript, &now) {
            transcript.write(&session.paths)?;
        }
        shown = now;
    }

    Ok(Outcome::Finished)
}

/// The voices one session's run will ask about, in first-appearance order, and the line
/// saying so -- or `None` for a session with nothing to ask about, which has been reported
/// and counted.
///
/// Separated from the asking so that the one decision a [`VoiceSelector`] changes is made in
/// one place: [`targeted`] is the sibling of this, and everything downstream of both is shared.
fn queue<'c>(
    order: &[&'c SpeakerCluster],
    shown: &BTreeMap<u32, Attribution>,
    offer: Offer,
    session: &DiscoveredSession,
    out: &mut dyn Write,
    report: &mut EnrollReport,
) -> Result<Option<Vec<&'c SpeakerCluster>>> {
    // The one place "already named" is decided. Everything below -- the floor, the in-run
    // guard, the prompt -- treats a voice the same however it got into this list, which is
    // what lets `--all` and `--correct` compose without either knowing about the other.
    let candidates: Vec<&SpeakerCluster> = order
        .iter()
        .copied()
        .filter(|c| offer.named || !shown[&c.id].is_named())
        .collect();
    if candidates.is_empty() {
        // A session whose voices are all identified is exactly where somebody stands when one
        // of those identifications is wrong, and this line is the only thing it prints -- so
        // it names the escape, the way the held-back line already names `--all`.
        let named = shown.values().filter(|label| label.is_named()).count();
        if named == 0 {
            writeln!(out, "{}  passed over: nothing unresolved", session.id)?;
        } else {
            writeln!(
                out,
                "{}  passed over: nothing unresolved ({named} named voice(s) -- \
                 meethook enroll --correct)",
                session.id
            )?;
        }
        report.passed_over += 1;
        return Ok(None);
    }

    let queued = candidates.len();

    // Only the voices worth a question, unless the user asked for the rest. Clustering emits a
    // long tail of one- and two-second fragments it cannot place -- 48 unresolved clusters for
    // a meeting of seven people, measured on `20260810-093047` -- and asking about each of
    // them is how a five-minute job becomes an hour. Filtering preserves first-appearance
    // order, which is what the user reads the transcript in.
    let mut offered: Vec<&SpeakerCluster> = if offer.quiet {
        candidates.clone()
    } else {
        candidates
            .iter()
            .copied()
            .filter(|c| c.speech_seconds >= PROMPT_FLOOR_SECONDS)
            .collect()
    };
    // A floor that hides every voice in a session is not a floor, it is a command that does
    // nothing. A short recording where nobody clears it -- the three-second fixtures the
    // end-to-end tests are built on, and any real meeting that ran for a minute -- offers
    // everybody instead. Decided here rather than defended against, because the alternative is
    // `enroll` reporting "nothing to do" on a session with unnamed people in it.
    if offered.is_empty() {
        offered = candidates;
    }
    let held_back = queued - offered.len();
    report.held_back += held_back;

    // "Unresolved" is false under `--correct`, where most of the queue is resolved and the
    // point is to review it. The default wording is left exactly as it was.
    //
    // `offered.len()` here is the same number every prompt below carries as its [`Position`]
    // total, because both read this list. Anything that computes this count independently
    // breaks that.
    let counted = if offer.named {
        let already = offered.iter().filter(|c| shown[&c.id].is_named()).count();
        format!(
            "{} voice(s) to review, {already} of them already named",
            offered.len()
        )
    } else {
        format!("{} unresolved voice(s)", offered.len())
    };
    if held_back == 0 {
        writeln!(out, "{}  {counted}", session.id)?;
    } else {
        // Naming the escape rather than only the count: a voice nobody is told about is not
        // reachable, which is what AC #3 asks for.
        writeln!(
            out,
            "{}  {counted}, {held_back} quieter voice(s) not offered -- meethook enroll --all",
            session.id
        )?;
    }

    Ok(Some(offered))
}

/// The one voice a [`VoiceSelector`] names, or `None` when it named none or several -- which
/// is reported and counted as a request that could not be served.
///
/// No floor, no `--correct` gate and no "nothing unresolved" pass-over: a user who named a
/// voice has already decided it is worth a question, and a session where everybody is already
/// named is exactly where `--voice "Alice"` gets used. Nothing is counted as held back either;
/// this run was aimed at one voice rather than filtered down to it, so a summary line offering
/// `--all` would be answering a question nobody asked.
fn targeted<'c>(
    selector: &VoiceSelector,
    order: &[&'c SpeakerCluster],
    unknown: &BTreeMap<u32, String>,
    shown: &BTreeMap<u32, Attribution>,
    session: &DiscoveredSession,
    out: &mut dyn Write,
    report: &mut EnrollReport,
) -> Result<Option<Vec<&'c SpeakerCluster>>> {
    let matched: Vec<&SpeakerCluster> = order
        .iter()
        .copied()
        .filter(|c| selector.matches(&unknown[&c.id], &shown[&c.id]))
        .collect();

    // How one voice reads in a message about several: the number it is reachable by, plus the
    // name it currently carries when that is not the number itself.
    let describe = |c: &SpeakerCluster| {
        let number = &unknown[&c.id];
        let label = shown[&c.id].label();
        if label == number {
            format!("{number} ({:.1} s)", c.speech_seconds)
        } else {
            format!("{number} -- {label} ({:.1} s)", c.speech_seconds)
        }
    };

    match matched.len() {
        1 => {
            // The literal 1 is this run's whole queue, so the one prompt below reads `1/1`.
            writeln!(
                out,
                "{}  1 voice selected: {}",
                session.id,
                describe(matched[0])
            )?;
            Ok(Some(matched))
        }
        0 => {
            // The voices are listed rather than merely counted, quiet ones included, because a
            // miss is usually a number off by one or a name spelled as the user remembers it
            // rather than as the transcript has it -- and the quiet voices are exactly what
            // somebody is reaching for when they miss. Fifty-odd lines on a real session is
            // still far cheaper than fifty-odd prompts.
            writeln!(
                out,
                "{}  no voice matched {selector} -- this session has {}:",
                session.id,
                order.len()
            )?;
            for cluster in order {
                writeln!(out, "    {}", describe(cluster))?;
            }
            report.failed += 1;
            Ok(None)
        }
        _ => {
            // Two voices under one enrolled name is the false accept `--correct` exists to
            // fix, so the message has to hand back the thing that tells them apart, which is
            // the number. Quoted as a whole label rather than as a bare digit so it can be
            // pasted straight back: both forms are accepted, and only one of them survives
            // being read off a line that also contains a name.
            let voices: Vec<String> = matched.iter().map(|c| describe(c)).collect();
            let numbers: Vec<String> = matched
                .iter()
                .map(|c| format!("--voice \"{}\"", unknown[&c.id]))
                .collect();
            writeln!(
                out,
                "{}  {selector} matches {} voices: {} -- pass one of {}",
                session.id,
                matched.len(),
                voices.join(", "),
                numbers.join(" or ")
            )?;
            report.failed += 1;
            Ok(None)
        }
    }
}

/// What each voice is called given the database and this session's hand-given names as they
/// stand: a name the user assigned, else an enrolled name where one matched, else the
/// "Unknown N" its first appearance earned it.
///
/// This is the labelling `merge` performs when it writes a transcript, reached through the
/// same [`attributions`], which is what makes a rewrite here and a `--force` re-transcribe
/// agree on the answer rather than merely be written to. The precedence between the three is
/// stated there and nowhere else.
///
/// `clusters` is what identification runs over and what `assigned` is resolved against;
/// `unknown` is what the transcript was written with, and is the key set of the result. Those
/// two are built from the same file, so every voice gets an entry.
///
/// Visible to the crate rather than to this file because [`references`] labels sessions through
/// exactly this too: the claim that a reference is naming some voice is only as good as its
/// being the same labelling the transcript is written with.
pub(crate) fn effective_labels(
    clusters: &[SpeakerCluster],
    unknown: &BTreeMap<u32, String>,
    speakers: &EnrolledSpeakers,
    assigned: &[AssignedName],
) -> BTreeMap<u32, Attribution> {
    attributions(
        unknown,
        Naming::new(clusters, &identify_clusters(clusters, speakers), assigned),
    )
}

/// What honouring an answer would have taken away from a voice the user was not asked about.
///
/// Both variants name that voice by its "Unknown N" rather than by what it currently reads,
/// because that is the one handle which reaches a voice whatever it is called and is exactly
/// what [`VoiceSelector`] accepts -- so a refusal is a line the user can act on rather than
/// only read.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Cost {
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
fn handle(id: u32, unknown: &BTreeMap<u32, String>) -> String {
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
/// the way three checks inside the three mechanisms could be.
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
fn cost_of(
    answered: u32,
    name: &str,
    unknown: &BTreeMap<u32, String>,
    corrected: &BTreeMap<u32, Attribution>,
    after: &BTreeMap<u32, Attribution>,
) -> Option<Cost> {
    if after.get(&answered).map(Attribution::label) != Some(name) {
        return Some(Cost::Vetoed {
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
        .map(|(&id, label)| Cost::Taken {
            voice: handle(id, unknown),
            losing: label.label().to_string(),
        })
}

/// Rewrites every speaker-track turn to what `labels` says its voice should now be called,
/// reporting whether anything changed.
///
/// Turns are found by the cluster they were attributed to, which `transcript.json` records for
/// exactly this: it is an exact handle on one voice's turns, so what a turn currently *reads*
/// never enters into it. That matters most in the case a label lookup cannot survive -- two
/// voices both matched to one enrolled person, then corrected so they belong to different
/// people. Keyed on text those turns are indistinguishable and the only safe answer is to
/// rewrite neither; keyed on the cluster there is no ambiguity to resolve, and correcting one
/// voice leaves the other's turns exactly where they were.
///
/// The cluster is never written back. `merge` is the sole producer of that field and `enroll`
/// only ever changes what a cluster is *called*, which is what keeps a transcript rewritten
/// here identical to what `transcribe --force` would now produce.
///
/// A turn with no cluster is left alone: on the mic track that is the local speaker, whose
/// name is not `enroll`'s to change, and on the speaker track it only arises in a session
/// where diarization found no clusters -- which has no labels to map and nothing to ask about.
/// A cluster absent from `labels` is left alone for the same reason `merge` ignores an
/// identification for a cluster diarization did not produce.
///
/// Nothing is written when nothing changed, which is what makes a skipped session leave its
/// files byte-identical rather than merely equivalent.
fn relabel(transcript: &mut Transcript, labels: &BTreeMap<u32, Attribution>) -> bool {
    let mut changed = false;
    for turn in &mut transcript.turns {
        if turn.source_track != SourceTrack::Speaker {
            continue;
        }
        let Some(label) = turn.cluster.and_then(|id| labels.get(&id)) else {
            continue;
        };
        if turn.speaker != label.label() || turn.speaker_id_confidence != label.confidence() {
            turn.speaker = label.label().to_string();
            turn.speaker_id_confidence = label.confidence();
            changed = true;
        }
    }
    changed
}

/// One line of transcript, trimmed and cut to something that fits a prompt.
fn snippet(text: &str) -> &str {
    let trimmed = text.trim();
    match trimmed.char_indices().nth(SNIPPET_CHARS) {
        Some((cut, _)) => &trimmed[..cut],
        None => trimmed,
    }
}

/// The audio to play for one voice: its longest representative, cut out of the speaker track.
///
/// The clip is sliced rather than seeked to because `afplay` cannot seek -- it has no start
/// offset at all -- so somebody has to extract it either way. Slicing the 16 kHz track
/// diarization itself ran on is what makes the seconds in a [`meethook_session::RepresentativeSegment`]
/// impossible to misinterpret: they are offsets into exactly this buffer.
///
/// A range running off the end of the track is clipped to what is there, and anything left
/// empty is a voice asked about without audio rather than a session that fails.
fn clip_for<'a>(track: &'a [f32], cluster: &SpeakerCluster) -> &'a [f32] {
    let Some(segment) = cluster.representatives.first() else {
        return &[];
    };
    let start = sample_at(segment.start).min(track.len());
    let end = sample_at(segment.end).min(track.len());
    if end <= start {
        return &[];
    }
    &track[start..end]
}

fn sample_at(seconds: f64) -> usize {
    (seconds.max(0.0) * f64::from(TARGET_RATE)).round() as usize
}

/// Writes a clip where an external player can reach it: mono, 16 kHz, 32-bit float.
///
/// Here rather than in the caller because the format is this crate's knowledge -- the clip in
/// a [`Voice`] is 16 kHz mono because that is the track it was cut from -- and a player that
/// had to be told the rate could be told the wrong one.
pub fn write_clip(path: &Path, clip: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let wav = |source| Error::Wav {
        path: path.to_path_buf(),
        source,
    };

    // Not `hound::WavWriter::create`: it tags a mono stream `SPEAKER_FRONT_LEFT`, and a clip
    // that exists so a human can recognise a voice is the last place to send it to one ear.
    let mut writer = meethook_session::wav::create(path, spec).map_err(wav)?;
    for sample in clip {
        writer.write_sample(*sample).map_err(wav)?;
    }
    writer.finalize().map_err(wav)
}

/// The sequencing and the writes, exercised without a terminal and without an audio device.
///
/// Every test below drives [`run_enroll`] against a scripted answerer over real session
/// directories on a temporary disk. What is *not* decidable here is whether a human can name
/// a colleague from what a prompt shows -- the audio, the snippet length, the wording -- which
/// needs a real recording and a real person.
#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use meethook_session::{
        EnrolledSpeaker, MAX_REFERENCES_PER_SPEAKER, RepresentativeSegment, SPEAKER_YOU,
        SessionPaths, TRANSCRIPT_SCHEMA_VERSION, Turn,
    };

    use super::*;

    /// A voice recorded exactly as it was shown, so a test can assert on what the user would
    /// have been looking at rather than only on what they answered.
    #[derive(Debug, PartialEq)]
    struct Shown {
        session: String,
        /// Which of this session's questions this was, and how many there were, exactly as the
        /// prompt was handed it.
        position: Position,
        /// What the prompt was told this voice is called and on what basis -- which is the only
        /// way a test can check that a correction prompt asked "is this right" rather than
        /// "who is this", and that a voice named for one session says so.
        attribution: Attribution,
        speech_seconds: f64,
        snippets: Vec<String>,
        clip_samples: usize,
    }

    impl Shown {
        fn label(&self) -> &str {
            self.attribution.label()
        }

        fn confidence(&self) -> Option<f32> {
            self.attribution.confidence()
        }
    }

    /// An interviewer that answers from a queue and remembers every voice it was asked about.
    /// Answers past the end of the script are skips, so a test that expects no prompt at all
    /// fails on `seen` rather than on a panic somewhere else.
    #[derive(Default)]
    struct Scripted {
        answers: VecDeque<Answer>,
        seen: Vec<Shown>,
    }

    impl Scripted {
        fn answering(answers: Vec<Answer>) -> Scripted {
            Scripted {
                answers: answers.into(),
                seen: Vec::new(),
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
                position: voice.position,
                attribution: voice.attribution.clone(),
                speech_seconds: voice.speech_seconds,
                snippets: voice.snippets.iter().map(|s| s.to_string()).collect(),
                clip_samples: voice.clip.len(),
            });
            self.answers.pop_front().unwrap_or(Answer::Skip)
        }
    }

    fn named(name: &str) -> Answer {
        Answer::Named(name.to_string())
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
    fn speaker_turn(start: f64, cluster: u32, speaker: &str, text: &str) -> Turn {
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

    fn mic_turn(start: f64, text: &str) -> Turn {
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
    pub(crate) fn make_session(paths: &Paths, id: &str) -> SessionPaths {
        let id = SessionId::parse(id).unwrap();
        let session = paths.session(&id);
        std::fs::create_dir_all(session.dir()).unwrap();
        // Only its presence is read here; classification never parses it.
        std::fs::write(session.session_json(), b"{}").unwrap();
        write_speaker_wav(&session.speaker_wav());

        SpeakerClusters::new(
            id.clone(),
            vec![cluster(0, 0.0, (0.5, 2.5)), cluster(1, 3.0, (3.0, 5.0))],
        )
        .write(&session)
        .unwrap();

        Transcript::new(
            id,
            vec![
                speaker_turn(0.0, 0, "Unknown 1", "  hi there  "),
                mic_turn(1.0, "morning"),
                speaker_turn(3.0, 1, "Unknown 2", "and from me"),
                speaker_turn(4.0, 0, "Unknown 1", "let us start"),
            ],
        )
        .write(&session)
        .unwrap();

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

        Transcript::new(
            parsed,
            vec![
                speaker_turn(0.0, 0, "Unknown 1", "hi there"),
                mic_turn(1.0, "morning"),
                speaker_turn(3.0, 1, "Unknown 2", "and from me"),
                speaker_turn(3.5, 2, "Unknown 3", "mm"),
                speaker_turn(4.5, 3, "Unknown 4", "yes"),
            ],
        )
        .write(&session)
        .unwrap();

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
        run_over(paths, ids, None, offer, enrolment, interviewer)
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
        run_over(
            paths,
            ids,
            Some(VoiceSelector::from(voice)),
            Offer::default(),
            Enrolment::default(),
            interviewer,
        )
    }

    fn run_over(
        paths: &Paths,
        ids: &[&str],
        voice: Option<VoiceSelector>,
        offer: Offer,
        enrolment: Enrolment,
        interviewer: &mut Scripted,
    ) -> (EnrollReport, String) {
        let requested: Vec<SessionId> =
            ids.iter().map(|id| SessionId::parse(id).unwrap()).collect();
        let mut out = Vec::new();
        let report = run_enroll(
            paths,
            &requested,
            voice.as_ref(),
            offer,
            enrolment,
            interviewer,
            &mut out,
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

    fn transcript_of(session: &SessionPaths) -> Transcript {
        Transcript::read(&session.transcript_json()).unwrap()
    }

    /// This session's hand-given names as they stand on disk, which is where an answer to a
    /// voice too quiet for a reference goes instead of into `speakers.json`.
    pub(crate) fn assigned_in(session: &SessionPaths, id: &str) -> SpeakerNames {
        SpeakerNames::read_or_empty(session, &SessionId::parse(id).unwrap()).unwrap()
    }

    /// Turns as (speaker, text, confidence), which is what a reader of the transcript sees.
    fn said(transcript: &Transcript) -> Vec<(&str, &str, Option<f32>)> {
        transcript
            .turns
            .iter()
            .map(|t| (t.speaker.as_str(), t.text.as_str(), t.speaker_id_confidence))
            .collect()
    }

    /// A clip exists to be handed to `afplay`, so its header is part of what it is for: a
    /// mono stream tagged `SPEAKER_FRONT_LEFT` reaches the listener in one ear.
    #[test]
    fn a_clip_is_tagged_mono_so_a_player_does_not_put_it_in_one_ear() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("clip.wav");
        write_clip(&path, &[0.0, 0.25, -0.25, 0.5]).unwrap();

        let wav = std::fs::read(&path).unwrap();
        assert_eq!(
            meethook_session::wav::channel_mask_of(&wav),
            Some(meethook_session::wav::MONO_CHANNEL_MASK)
        );
    }

    /// Acceptance criteria #5 and #6, at the level a user meets them: one answer puts a
    /// person in the database and their name on their own turns, and on nobody else's.
    #[test]
    fn naming_a_voice_enrolls_them_and_rewrites_that_sessions_transcript() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
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
        assert_eq!(markdown, transcript_of(&session).render_markdown());
        assert!(markdown.contains("Alice"), "{markdown}");
        assert!(!markdown.contains("Unknown 1"), "{markdown}");
    }

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

    /// The case this whole handle exists for, and the one a label lookup cannot survive: a
    /// false accept has filed cluster 3's voice under the name of the person who is really
    /// cluster 1, so two clusters read "Andrew", and the correction sends them to
    /// different names. Keyed on the label text both turn-groups are one indistinguishable
    /// bucket and the only safe answer is to rewrite neither -- silently leaving the user
    /// looking at an uncorrected transcript. Keyed on the cluster the two are simply
    /// different turns.
    #[test]
    fn correcting_one_of_two_voices_sharing_a_label_leaves_the_other_alone() {
        let mut transcript = Transcript::new(
            SessionId::parse("20260809-052600").unwrap(),
            vec![
                speaker_turn(0.0, 1, "Andrew", "the real one"),
                mic_turn(1.0, "morning"),
                speaker_turn(2.0, 3, "Andrew", "actually Ryan"),
                speaker_turn(3.0, 1, "Andrew", "the real one again"),
            ],
        );
        for turn in &mut transcript.turns {
            if turn.source_track == SourceTrack::Speaker {
                turn.speaker_id_confidence = Some(0.71);
            }
        }

        // The database after the correction: cluster 3 is Ryan, cluster 1 is still Andrew.
        let labels: BTreeMap<u32, Attribution> = [
            (
                1,
                Attribution::Identified {
                    name: "Andrew".to_string(),
                    similarity: 0.71,
                },
            ),
            (
                3,
                Attribution::Identified {
                    name: "Ryan".to_string(),
                    similarity: 0.88,
                },
            ),
        ]
        .into_iter()
        .collect();

        assert!(
            relabel(&mut transcript, &labels),
            "the correction must be reported as a change, not silently declined"
        );
        assert_eq!(
            said(&transcript),
            [
                ("Andrew", "the real one", Some(0.71)),
                ("You", "morning", None),
                ("Ryan", "actually Ryan", Some(0.88)),
                ("Andrew", "the real one again", Some(0.71)),
            ]
        );
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
            },
            EnrolledSpeaker {
                name: "Bob".to_string(),
                embedding: voice(1),
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
    /// name in that transcript and no new reference -- and, crucially, loses none of the five
    /// they have. Dropping the oldest would un-name a voice in some earlier session, which is
    /// the defect this whole ticket exists to end.
    ///
    /// Every voice here is on its own axis, so no two are ever within reach of each other and
    /// each session really does have to ask.
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
        assert!(
            output.contains(&paths.speakers_json().display().to_string()),
            "the line has to say where to make room: {output}"
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
        assert!(on_disk.contains("\"schema_version\": 2"), "{on_disk}");
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        assert_eq!(speakers.references("Alice"), 1, "{:?}", speakers.speakers);
        assert_eq!(speakers.references("Bob"), 1, "{:?}", speakers.speakers);
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
            None,
            Offer::default(),
            Enrolment::default(),
            &mut interviewer,
            &mut out,
        )
        .unwrap_err();

        assert!(error.to_string().contains("speakers.json"), "{error}");
        assert!(error.to_string().contains("upgrade meethook"), "{error}");
        assert!(
            interviewer.seen.is_empty(),
            "nothing may be asked against a database that could not be read"
        );
    }

    /// The refusal rule, exercised as the pure comparison it is: two labellings in, and either
    /// nothing or the voice that would have paid. Cheaper and more direct than reaching each
    /// branch through a session on disk, which the tests above do for the paths that produce
    /// these maps.
    mod cost {
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
                cost_of(
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
                cost_of(1, "Alice", &numbers(&[0, 1]), &corrected, &after),
                Some(Cost::Vetoed {
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
                cost_of(0, "Alice", &numbers(&[0]), &labels, &labels),
                Some(Cost::Vetoed { holder: None })
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
                cost_of(0, "Alice", &numbers(&[0, 1]), &corrected, &after),
                Some(Cost::Taken {
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
                cost_of(0, "Alice", &numbers(&[0, 1]), &corrected, &after),
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
                cost_of(0, "Ryan", &numbers(&[0, 1]), &corrected, &after),
                None
            );
        }
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

        assert_eq!(
            interviewer.labels(),
            ["Andrew", "Andrew"],
            "{output}"
        );
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
            position,
            attribution,
            speech_seconds,
            snippets,
            clip_samples,
        } = &aimed.seen[0];
        let queued = &queued.seen[1];
        assert_eq!(session, &queued.session);
        assert_eq!(attribution, &queued.attribution);
        assert_eq!(speech_seconds, &queued.speech_seconds);
        assert_eq!(snippets, &queued.snippets);
        assert_eq!(clip_samples, &queued.clip_samples);
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
        let (report, output) = run_over(
            &forced_paths,
            &["20260809-052600"],
            Some(VoiceSelector::from("2")),
            Offer::default(),
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

    /// A long line is cut to something that fits a prompt, on a character boundary rather
    /// than a byte one.
    #[test]
    fn a_long_snippet_is_cut_to_a_readable_length() {
        let long = "é".repeat(SNIPPET_CHARS * 2);
        assert_eq!(snippet(&long).chars().count(), SNIPPET_CHARS);
        assert_eq!(snippet("  short  "), "short");
    }
}
