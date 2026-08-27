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
mod meeting;
mod narration;
mod references;
mod resolve;

use consequence::handle;
pub use consequence::{Consequence, Preview, Refusal};
pub use forget::{Confirm, Forgotten, Removal, Target, run_forget};
pub use meeting::{Labelled, MeetingChoice, MeetingSource, Relabelling, run_meeting};
pub use narration::{
    AnswerNote, Lines, Narrator, Nearest, NotSelected, Note, PassedOver, RunNote, SessionFile,
    SessionNote, VoiceDescription,
};
use narration::{about, after};
pub use references::{
    Enrolled, Reference, Scan, Unreadable, VoiceChange, incomplete, run_speakers, scan,
};
pub use resolve::{Likeness, Match, Resolution, resolve};

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use meethook_session::{
    AssignedName, Classification, DiscoveredSession, EnrolledSpeakers, Meeting, MeetingFit, Paths,
    SessionId, SourceTrack, SpeakerCluster, SpeakerClusters, SpeakerNames, Transcript,
    TranscriptContext, TranscriptTemplate, TranscriptTime, VoiceAt, discover_sessions,
    unknown_labels, unknown_speaker,
};
use meethook_transcribe::{
    Attribution, Naming, Resemblance, TARGET_RATE, attributions, identify_clusters, rank_enrolled,
    read_track_16k_mono, speaker_offset_seconds,
};

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
///
/// `nth` is fixed when the queue is built, so [`Answer::Later`] does not renumber anything: a
/// voice deferred at 3/9 comes back at 3/9, however many passes it takes. The alternative --
/// numbering the passes -- would have the same voice arrive as 3/9 and then 1/2, which reads as
/// a different voice in a different session rather than as the one question still open.
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

/// One voice of a session as a queue pane lists it, which is not the same thing as a question.
///
/// Every voice the session has, including the ones this run is not asking about: the quiet
/// fragments held back under the prompt floor, and the ones the database has already named.
/// An interface that draws a queue needs all of them at once -- a pane showing only the voices
/// currently being asked about would leave the user unable to see that the two-second fragment
/// they are looking for exists at all -- whereas [`Voice`] is one question.
///
/// Four fields and no methods on purpose: it is a row, and what a row *reads like* belongs to
/// whatever is drawing it.
pub struct Queued<'a> {
    /// The "Unknown N" this voice was transcribed with, which does not move when it is named.
    /// The same handle [`Voice::number`] carries, so an interface can match a row against the
    /// question it is being asked.
    pub number: &'a str,

    /// What this voice currently reads as and on what basis, exactly as
    /// [`Voice::attribution`] means it -- and as the database and this run's answers stand
    /// right now, not as they stood when the session was opened.
    pub attribution: &'a Attribution,

    /// Total speech attributed to this voice, in seconds. What tells a participant from
    /// somebody who coughed once, and so what a queue is worth sorting or dimming by.
    pub speech_seconds: f64,

    /// Whether the prompt floor would have held this voice back -- so a queue can say why a
    /// row is not among this run's questions, and offer `--all` by name.
    ///
    /// A boolean rather than `PROMPT_FLOOR_SECONDS` made public: where the floor sits, and that
    /// a voice sitting exactly on it is offered, stay this library's decisions. An interface
    /// comparing its own copy of the number would be a second answer to the same question.
    ///
    /// True even under `--all`, which changes which voices are *asked about* and not which ones
    /// are quiet.
    pub below_floor: bool,
}

/// One line a voice said, and the audio under it.
///
/// Four fields and no methods, for the reason [`Queued`]'s doc gives: it is a row, and what a
/// row reads like belongs to whatever is drawing it. Four things a reader cannot get from the
/// type, each of them a place this goes wrong quietly rather than loudly:
///
/// - **`start` is track time, not timeline time.** A [`meethook_session::Turn`]'s seconds are
///   on the session timeline, which begins at whichever track started first;
///   `start` here is an offset into `speaker.wav`, the same space a
///   [`meethook_session::RepresentativeSegment`] is in and the same space [`Voice::clip`] was
///   cut from. The difference between the two is
///   [`meethook_transcribe::speaker_offset_seconds`]. Nothing fails if it is not applied --
///   the words and the sound simply drift apart by however long the microphone led by, and
///   only a listener would notice.
///
/// - **`duration` is what the transcript says, `audio.len()` is what there is to play, and
///   they can disagree.** A truncated `speaker.wav` gives a full `duration` and short `audio`;
///   a missing one gives a full `duration` and empty `audio`. That split is deliberate: one
///   says how long the line took, the other says how much of it survives on disk, so anything
///   sizing a progress bar wants `audio.len() / TARGET_RATE` and not `duration`.
///
/// - **Empty `audio` is normal**, exactly as an empty [`Voice::clip`] is: a voice that can
///   still be named from its lines rather than a session that has to fail.
///
/// - **`text` is the same text a prompt always had** -- whitespace-trimmed, cut to
///   `SNIPPET_CHARS` characters, and never empty, because a line the recogniser heard nothing
///   over is dropped before a snippet is built at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Snippet<'a> {
    /// What was said, trimmed and cut. A borrow of the transcript, so a voice that talked for
    /// ten minutes costs pointers rather than text.
    pub text: &'a str,

    /// When it was said, in seconds from the first sample of `speaker.wav`.
    pub start: f64,

    /// How long the line lasted, in seconds, as the transcript has it.
    pub duration: f64,

    /// The samples for exactly that stretch: 16 kHz mono, the rate everything else in meethook
    /// works in. A borrow of the resampled track, not a copy.
    pub audio: &'a [f32],
}

/// The meeting a session was recorded during, as far as a terminal may see it.
///
/// Only the title and how strongly the session's start supports the match. [`Meeting`] holds
/// more -- organizer, attendees, location, URL, invite body -- and none of that may reach a
/// terminal or a log line: attendee names and addresses exist in `session.json` for speaker
/// identification and are deliberately never printed, and an invite body routinely carries a
/// dial-in PIN. Projecting to these two fields makes "nothing sensitive crosses" a property
/// of the type rather than a rule every consumer must remember.
///
/// It also owns the one display shape every surface derives: [`clause`](Self::clause) is what
/// `meethook record`'s meeting line and the enroll queue announcement both print, so they
/// cannot drift into two wordings of the same meeting. The caveat wording itself stays on
/// [`MeetingFit::caveat`], where it is defined and tested; this crate owns the placement, the
/// library owns the sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeetingLabel {
    /// The invite's title: the handle for "which call was this".
    pub title: String,
    /// How strongly the session's start supports this being the meeting.
    pub fit: MeetingFit,
}

impl MeetingLabel {
    /// The title alone when the fit states it plainly, the title followed by `  ({caveat})`
    /// otherwise -- exactly the clause `meethook record` prints after its `  meeting   `
    /// prefix. Half the meetings on disk are not a strong match, so a bare title would assert
    /// a match the tool does not have; the caveat is what keeps a guess from reading as a
    /// fact.
    pub fn clause(&self) -> String {
        match self.fit.caveat() {
            Some(caveat) => format!("{}  ({caveat})", self.title),
            None => self.title.clone(),
        }
    }
}

impl From<&Meeting> for MeetingLabel {
    fn from(meeting: &Meeting) -> Self {
        Self {
            title: meeting.title.clone(),
            fit: meeting.fit,
        }
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

    /// The meeting this session was recorded during, as far as a terminal may see it -- or
    /// `None`, which is the common case rather than the exception.
    ///
    /// The same value the queue announcement was handed, built from the same
    /// `session.json` load: an interface that shows it does so across this seam rather than
    /// reaching back for the file itself, and nothing off the roster crosses with it.
    /// Absent costs nothing downstream -- no reserved row, no empty label -- so a run over
    /// sessions without meetings behaves exactly as before.
    pub meeting: Option<&'a MeetingLabel>,

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

    /// The "Unknown N" this voice was transcribed with -- a handle that does not move.
    ///
    /// [`attribution`](Voice::attribution)'s label is what the voice reads as *now*, so it
    /// changes the moment the voice is named. This does not: it comes from [`unknown_labels`],
    /// which ranks every voice by first appearance whether or not it has a name, and is fixed
    /// for the session. An [`Interviewer`] with state of its own -- a cursor, a row it has
    /// marked, a name half-typed -- has to key that state on something stable across
    /// [`identify`](Interviewer::identify) calls, and this is the only field that qualifies.
    ///
    /// Not the cluster id, for the reason [`VoiceSelector`] gives: the id appears in
    /// `transcript.json` and nowhere a person reads, and two numbering systems reachable from
    /// one interface is how a cursor lands on the wrong voice.
    pub number: &'a str,

    /// Total speech attributed to this voice, in seconds. How the user tells a participant
    /// from someone who coughed once.
    pub speech_seconds: f64,

    /// Every voice in this session, in first-appearance order -- the order the transcript
    /// reads in -- whether or not this run is asking about it.
    ///
    /// What lets an interface draw the whole session beside the one question, which is what
    /// makes the quiet voices and the already-named ones visible rather than merely reachable.
    /// It includes the voice being asked about, so a queue pane needs nothing stitched onto it.
    ///
    /// Rebuilt for every call rather than handed over once per session, which is what makes it
    /// current: an answer accepted a moment ago has already been written and the session
    /// relabelled through it, so a voice named by the previous question arrives here under its
    /// new name.
    pub queue: &'a [Queued<'a>],

    /// What this voice said, whitespace-trimmed and cut to `SNIPPET_CHARS` characters each,
    /// with the lines the recogniser heard nothing over dropped -- and the audio under each.
    /// Empty if it heard nothing at all.
    ///
    /// Every snippet, uncapped. How many will fit is a fact about the thing displaying them --
    /// a line prompt has one screenful of scrollback and takes three; a pane can scroll -- so
    /// capping here would decide it for both. Both the text and the samples are borrows, of
    /// the transcript and of the resampled track, so a voice that talked for ten minutes costs
    /// pointers rather than text or audio.
    ///
    /// See [`Snippet`] for what "when it was said" means here, which is not the same clock a
    /// transcript reads in.
    pub snippets: Vec<Snippet<'a>>,

    /// The longest representative clip: 16 kHz mono, the same rate everything else in
    /// meethook works in.
    ///
    /// Empty when `speaker.wav` is missing or unreadable, which is a voice that can still be
    /// named from its snippets rather than a session that has to fail.
    pub clip: &'a [f32],

    /// Who this voice sounds like, nearest first, so a prompt can offer names instead of
    /// demanding one be typed.
    ///
    /// This is what makes an [`Interviewer`] able to answer without any access to
    /// `speakers.json`: the names and the numbers are already here, owned, and the database
    /// stays on this side of the seam where the writes are.
    ///
    /// Four things a reader cannot get from the type, all of them [`rank_enrolled`]'s
    /// decisions rather than this field's:
    ///
    /// - **Unthresholded.** Every comparable enrolled person is here, including ones far
    ///   outside `IDENTIFY_DISTANCE`. That cut keeps `transcribe`'s automatic pass
    ///   conservative; a person reading a list is the case it was biased against serving, and
    ///   cutting here would hide the near-miss they are being asked to adjudicate.
    /// - **Entries are people, not references.** Somebody holding several recordings appears
    ///   once, scored at their nearest, with `references` saying how many they hold.
    /// - **Order** is descending similarity, ties by ascending name, so the first entry is the
    ///   person `identify_clusters` would have awarded had it cleared the cut. Empty for an
    ///   install where nobody is enrolled yet.
    /// - **As the database stands now**, not as it stood when the run began. A name given
    ///   earlier in this same run is in here, which is the useful behaviour rather than an
    ///   accident: clustering splits one person in two, and the second half should offer the
    ///   name the first half was just given.
    pub resembles: Vec<Resemblance>,

    /// Every enrolled name, deduplicated, in enrolment order -- the universe [`resolve()`]
    /// requires.
    ///
    /// Not [`resembles`](Voice::resembles) with the numbers stripped off, and the difference is
    /// the whole reason this field exists: ranking a voice against the database drops a person
    /// whose every stored recording is a stale embedding dimension, and that person is still
    /// real, so resolving a typed name against the ranked list would silently enrol a second
    /// copy of them. `resembles` answers "who does this sound like"; this answers "who is
    /// there", which is the question a name being typed is about.
    ///
    /// As the database stands now, like `resembles`: a name given earlier in this same run is
    /// in here.
    pub enrolled: Vec<&'a str>,

    /// What answering with a given name *would* do, asked without writing anything.
    ///
    /// [`resembles`](Voice::resembles) says who this voice sounds like; this says what happens
    /// if you say so. An [`Interviewer`] holding it can show the consequence of the name under
    /// the cursor -- that it enrols somebody new, that it takes a reference off Milo, that the
    /// veto will refuse it -- before the user commits to it, which is the whole difference
    /// between a prompt that reports what it did and one that can be answered.
    ///
    /// The three things the type cannot say:
    ///
    /// - **Asking writes nothing.** Not `speakers.json`, not this session's
    ///   `speaker_names.json`, not the transcript. It runs on copies, which is why an
    ///   `Interviewer` may ask about as many candidates as it likes.
    /// - **One call is one answer's worth of work**, not one keystroke's: a database clone and
    ///   two full labellings of the session. Preview the candidate the user has settled on;
    ///   [`resolve()`] is the cheap thing to run per keystroke.
    /// - **An answerer that never asks pays nothing.** [`GivenName`] does not, and neither does
    ///   a line prompt that only reports outcomes after the fact.
    pub preview: Preview<'a>,
}

/// What the user said when asked who a voice is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Answer {
    /// Who this voice is.
    Named {
        /// The name, trimmed the same way [`GivenName`] trims one supplied up front.
        name: String,

        /// Honour the name even though it takes a name off a voice the user was not asked
        /// about -- [`Refusal::Taken`], and that refusal only.
        ///
        /// The refusal exists because a third party silently losing their name is a surprise,
        /// and an answerer that has already *shown* the user which voice pays and what it loses
        /// has removed the surprise. That is `forget --yes`'s argument reached from the other
        /// side: see `forget.rs`'s "Nothing is ever refused". So this is not a way to answer
        /// harder; it is an answer given by somebody who was shown the cost first, which is why
        /// the only interface that sets it is the one with a pane for the cost.
        ///
        /// Carried on the answer rather than decided by the interface because the interface is
        /// not on every path: a line prompt and any scripted answerer reach the library's guard
        /// without passing through the frame's state machine, so an override the frame merely
        /// *knew about* would be refused for them. The answer is the one thing every path has.
        ///
        /// [`Refusal::Vetoed`] is out of reach whatever this says. That refusal is a different
        /// claim -- segmentation heard the two voices at once and so proved they are different
        /// people -- and overriding it means asserting several voices are one person, which is
        /// TASK-046.09's question and not this field's.
        anyway: bool,
    },
    Skip,
    /// Not this voice, not yet: put it back in the queue and ask again later in this session.
    ///
    /// Distinct from [`Skip`](Self::Skip), which is a decision -- the question was asked and
    /// went unanswered -- where this is a request to be asked again. It exists because a queue
    /// is walked in first-appearance order and the voice somebody can actually place is often
    /// not the one at the top: without it, reaching the four-minute voice at 7/9 means pressing
    /// Enter past six people, and every one of those presses is a chance to type a name onto
    /// the wrong person. Only an interface that can show the whole queue at once has any use
    /// for it; a line prompt has nowhere to move a cursor to.
    ///
    /// Deferring costs nothing and writes nothing. The voice comes back with the [`Position`]
    /// it had, and a session ends when a pass over the deferred voices produces no answer at
    /// all -- at which point they are counted exactly as the skips and kept identifications
    /// they have turned out to be. So deferring every voice and then stopping is the same
    /// outcome as skipping every voice, which is what makes "not yet" safe to answer with when
    /// there turns out to be no later.
    Later,
    /// End this session here and open the next one.
    ///
    /// Three answers and three scopes: [`Skip`](Self::Skip) is one voice, this is the rest of
    /// this session, [`Quit`](Self::Quit) is the run. Saying so here is what stops the middle
    /// one from being read as either of its neighbours -- the run carries on to the next
    /// session on disk, and the last session being left this way ends the run exactly as
    /// finishing it would.
    ///
    /// It exists because the queue's tail is usually clustering fragments and passers-by, and
    /// the user who has named the colleagues wants out of the session rather than out of the
    /// program: without it, leaving eight voices behind is eight more keypresses on the one
    /// screen where a stray Enter types a name onto the wrong person.
    ///
    /// Answering it writes nothing, and everything accepted in this session is already on
    /// disk -- writes happen per accepted name, which is what makes both early exits cost
    /// nothing that was answered.
    ///
    /// The voices left behind are counted as the skips -- or kept identifications -- they have
    /// turned out to be, by the rule [`Later`](Self::Later) already describes. So "leave the
    /// rest" and "defer everything and stop" report identically, and the summary still
    /// accounts for every voice the queue offered.
    ///
    /// Not the fixed point: this is an answer, given while a voice was on the screen, so it
    /// returns before a pass can stall and [`Interviewer::still_working`] is never consulted
    /// on this path. That method can neither suppress this exit nor be defeated by it.
    Leave,
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

    /// Whether this answerer requires the run to have been narrowed to one voice already.
    ///
    /// `false` for anyone a person is behind: a terminal is shown each voice before it answers
    /// about it, so a queue of nine is nine questions and not nine copies of one answer. It is
    /// [`GivenName`] that needs the guarantee -- an answer supplied up front never sees the
    /// voice it lands on -- and [`run_enroll`] refuses to start such a run without a
    /// [`Selection`], which is the only thing that makes "the voice this answer is about" a
    /// voice the user picked.
    ///
    /// A method on this trait rather than a flag beside it because the requirement belongs to
    /// the answerer: the caller cannot be trusted to remember which of the two it passed, and
    /// [`run_enroll`] is where the answerer and the selection are both in hand.
    fn needs_one_voice(&self) -> bool {
        false
    }

    /// Whether this answerer still has work left after a pass over the queue that produced no
    /// answer at all.
    ///
    /// The session loop cannot decide this for itself. It knows how many voices a pass deferred
    /// and not why any of them was deferred, and for an answerer with a cursor those are
    /// different facts: such an interface defers a voice in order to *reach* another one, so
    /// "this pass produced no answer" is what moving the cursor backwards looks like, not what
    /// finishing looks like. Answering `true` there keeps the session open and offers the same
    /// voices again, with the same numbers.
    ///
    /// This is the contract, and it is the answerer's: this method is what bounds the loop. An
    /// answerer that defers every voice and always returns `true` is never finished and the
    /// session never ends, so anything that returns `true` must be able to reach
    /// [`Answer::Quit`] -- which every interface has, and which is the exit a user reaches for.
    /// The one case the loop still decides alone is an empty queue: with nothing left to offer
    /// there is no next prompt to change the answer or carry a `Quit`, so a pass with nothing
    /// to ask about ends the session whatever this returns.
    ///
    /// `false` for an answerer that never defers, which is both of the ones in this crate:
    /// [`GivenName`] answers once, and a line prompt has no cursor to move, so for them the
    /// question never arises.
    fn still_working(&self) -> bool {
        false
    }
}

/// A name decided before the run started, for the one voice a [`Selection`] picked out.
///
/// The other half of naming a voice by pointing at a timestamp: `--at` says *which* voice and
/// this says *who*, and together they make the whole operation one non-interactive command --
/// which is the point, since a user who can already see who spoke at 12:34 has nothing to be
/// asked.
///
/// In the library rather than in the CLI, unlike [`Interviewer`]'s terminal implementation,
/// because there is nothing here that needs a person in front of it: what it answers, and that
/// it is only ever asked once, are decidable in `cargo test`.
pub struct GivenName(String);

impl GivenName {
    /// Trimmed on the way in, so this and a typed answer are normalised the same way -- a name
    /// of nothing but spaces is a skip on both paths rather than an entry called "".
    pub fn new(name: &str) -> GivenName {
        GivenName(name.trim().to_string())
    }
}

impl Interviewer for GivenName {
    fn identify(&mut self, _voice: &Voice<'_>) -> Answer {
        // Never insists. A name supplied up front is never shown the voice it lands on -- which
        // is the whole reason `needs_one_voice` exists below -- so it has certainly not been
        // shown the third voice an override would cost, and the premise the override rests on
        // does not hold here.
        Answer::Named {
            name: self.0.clone(),
            anyway: false,
        }
    }

    fn needs_one_voice(&self) -> bool {
        true
    }
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

/// How a run that is about one voice arrived at that voice.
///
/// Two ways in, because there are two things a user is looking at when they want to name
/// somebody. [`Voice`](Self::Voice) is the prompt queue's own vocabulary -- "Unknown 3", or the
/// name a voice currently reads as -- and is right while the queue is on screen.
/// [`At`](Self::At) is the transcript's: a moment in the session, for the far commoner case of
/// somebody reading `transcript.md`, seeing that whoever spoke at 12:34 is Alice, and neither
/// knowing nor caring which Unknown number that voice ended up as.
///
/// One enum rather than two fields beside each other, so that "one voice, selected one way" is a
/// property of the type instead of a rule two `Option`s have to be checked against. What each
/// arm resolves *through* is different -- a label is compared against the session's voices, a
/// timestamp is looked up in its transcript -- but everything downstream of the resolution is
/// the same one voice, which is why this changes nothing about what an answer writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection<'a> {
    /// `--voice`: the label the voice reads as. See [`VoiceSelector`].
    Voice(&'a VoiceSelector),

    /// `--at`: the moment the voice was speaking at, in the `MM:SS` spelling `transcript.md`
    /// prints. Resolved through [`meethook_session::Transcript::voice_at`], which owns the rule
    /// for turning a printed label back into a turn.
    At(TranscriptTime),
}

impl Selection<'_> {
    /// The flag this arrived on, so a message about the request names what the user typed.
    fn flag(&self) -> &'static str {
        match self {
            Selection::Voice(_) => "--voice",
            Selection::At(_) => "--at",
        }
    }

    /// Why one session id and not several. Two different reasons, and a user who passed one flag
    /// is not helped by the other's.
    fn why_one_session(&self) -> &'static str {
        match self {
            Selection::Voice(_) => {
                "a voice belongs to one session, so its number and its name mean nothing across \
                 several"
            }
            Selection::At(_) => {
                "a timestamp is an offset into one recording, so it lands somewhere different in \
                 each of several"
            }
        }
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

/// Whether a session with nothing left unresolved is opened at all.
///
/// The other half of what [`Offer::named`] used to decide on its own, pulled out because they
/// are two questions: *which voices does a session offer*, which is `Offer`'s own subject, and
/// *is this session worth visiting*. They coincide for the two combinations the CLI shipped, and
/// come apart the moment an interface wants every voice in the queue pane -- widening `Offer`
/// for that would also, silently, have `meethook enroll` over a directory of finished sessions
/// open one on each of them.
///
/// An enum rather than a bool, following [`Enrolment`] and [`Confirm`]: at the call site
/// `Sessions::Every` says what it does, where `true` would need the parameter name to be read.
///
/// Nothing here applies to a run with a [`Selection`]: pointing at a voice or a moment has
/// already made this judgement, so neither of those paths has ever had this gate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Sessions {
    /// The default: pass over a session where every voice already carries a name, because there
    /// is no question left to ask about it.
    #[default]
    Unresolved,

    /// `--correct`: opened anyway. A session where nothing is unresolved is exactly where a
    /// wrong identification sits, so it is the one a user correcting an identification is
    /// reaching for.
    ///
    /// Not what the full-screen frame asks for, which is the whole point of this enum existing
    /// separately from [`Offer::named`]. The frame widens `Offer` so its queue pane can reach
    /// every voice in a session it *did* open; widening this as well would open one on every
    /// finished meeting on disk.
    Every,
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
/// A bundle rather than four parameters, because it is threaded through the walk over sessions
/// unchanged and every function on that path would otherwise carry all of them. It is the
/// caller's parameter too: four axes named at the call site read better than four in a row of
/// eight positional arguments, and a caller adding one of them cannot then transpose the rest.
///
/// There is no `Default`: [`template`](Self::template) has no sensible one. What a caller with
/// no opinion wants is [`TranscriptTemplate::resolve`] with no explicit path, which is a
/// fallible read of the root rather than a constant.
#[derive(Debug, Clone, Copy)]
pub struct EnrollRules<'a> {
    /// `Some` replaces the queue with one voice, however the user pointed at it; `None` is the
    /// queue. Not a fourth flag on [`Offer`], because it does not widen the queue -- it stands
    /// in for it.
    pub selector: Option<Selection<'a>>,

    /// Which voices get asked about. Changes the questions and nothing else: the same answers
    /// write the same files however a voice came to be offered.
    pub offer: Offer,

    /// Which sessions get visited -- the separate question [`Sessions`] describes. Ignored when
    /// `selector` is `Some`, which stands in for the queue and its gates alike.
    pub sessions: Sessions,

    /// What an accepted name writes -- the other axis, and the only one that changes that.
    pub enrolment: Enrolment,

    /// What a rewritten `transcript.md` is rendered through, handed in already compiled.
    ///
    /// Naming a voice must not be able to change the shape of a transcript it did not write,
    /// so this belongs to the run rather than to a session: it is resolved once, from the same
    /// root `transcribe` resolved it from, and every session rewritten here goes through it.
    pub template: &'a TranscriptTemplate,
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
    if let Some(selection) = rules.selector
        && requested.len() != 1
    {
        notes.note(Note::Run(RunNote::SelectionNeedsOneSession { selection }))?;
        report.failed += 1;
        return Ok(report);
    }

    // An answer supplied up front is never shown the voice it lands on, so a queue would put one
    // name on everybody in it. Refused here, beside the guard above, for the same reason: this is
    // the one place that can see both the answerer and the selection, and a library caller
    // wiring up a [`GivenName`] gets the same protection the CLI does.
    if rules.selector.is_none() && interviewer.needs_one_voice() {
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
            rules,
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
    rules: EnrollRules<'_>,
    speakers: &mut EnrolledSpeakers,
    interviewer: &mut dyn Interviewer,
    notes: &mut dyn Narrator,
    report: &mut EnrollReport,
) -> Result<Outcome> {
    match session.classification {
        Classification::Orphaned => {
            about(
                notes,
                &session.id,
                SessionNote::PassedOver(PassedOver::Orphaned),
            )?;
            report.passed_over += 1;
            return Ok(Outcome::Finished);
        }
        Classification::Valid => {
            about(
                notes,
                &session.id,
                SessionNote::PassedOver(PassedOver::NotTranscribed),
            )?;
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
            unreadable(notes, session, SessionFile::Clusters, &e)?;
            report.failed += 1;
            return Ok(Outcome::Finished);
        }
    };
    // What a re-rendered `transcript.md` needs beyond the turns: the session's start time and
    // the meeting it was recorded during. Read here, beside the clusters, so a session whose
    // `session.json` has gone bad is reported and skipped like every other unreadable one
    // rather than ending the queue -- and so nothing is read inside the naming loop below,
    // where a failure would arrive after names had already been written.
    let metadata = match session.load_metadata() {
        Ok(metadata) => metadata,
        // No re-transcribe recovers this: `session.json` is the recorder's own output and the
        // marker that this directory is a session at all, so the only honest instruction is to
        // go and look at it.
        Err(e) => {
            unreadable(notes, session, SessionFile::Metadata, &e)?;
            report.failed += 1;
            return Ok(Outcome::Finished);
        }
    };
    // The meeting, projected to what a terminal may see and built here, beside the load: the
    // queue announcement below gets it from this value rather than any surface reaching back
    // for `session.json`, and nothing off the roster ever crosses with it. It is handed to the
    // voices too -- the frame shows it across the Interviewer seam -- so the announcement takes
    // a clone and the original outlives the asking loop.
    let meeting = metadata.meeting.as_ref().map(MeetingLabel::from);
    let mut transcript = match Transcript::read(&session.paths.transcript_json()) {
        Ok(transcript) => transcript,
        // As above, and with the same remedy: the expected instance is a `transcript.json`
        // from before turns recorded which cluster they came from. A user told only "missing
        // field `cluster`" has been given a diagnosis with no next step.
        Err(e) => {
            unreadable(notes, session, SessionFile::Transcript, &e)?;
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
            unreadable(notes, session, SessionFile::Names, &e)?;
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
        transcript.write(
            &session.paths,
            rules.template,
            &TranscriptContext::now(&metadata),
        )?;
        about(notes, &session.id, SessionNote::BroughtUpToDate)?;
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
        Some(Selection::Voice(selector)) => {
            targeted(selector, &order, &unknown, &shown, session, notes, report)?
        }
        Some(Selection::At(at)) => at_timestamp(
            at,
            &transcript,
            &order,
            &unknown,
            &shown,
            session,
            notes,
            report,
        )?,
        None => queue(
            &order,
            &shown,
            rules.offer,
            rules.sessions,
            meeting.clone(),
            session,
            notes,
            report,
        )?,
    };
    let Some(offered) = offered else {
        return Ok(Outcome::Finished);
    };

    // Read after that check, so a session with nothing to ask about never resamples an hour
    // of audio in order to then ask nothing. Unreadable is empty rather than fatal: a voice
    // with no clip can still be named from its snippets.
    let track = read_track_16k_mono(&session.paths.speaker_wav()).unwrap_or_default();

    // Turn times are on the session timeline; a snippet's are offsets into `speaker.wav`.
    // `Err` is a degenerate timebase in `session.json`: the two clocks cannot be related at
    // all, so the snippets get no audio rather than audio a second out -- the same tolerance
    // an unreadable `speaker.wav` already gets a line above, and for the same reason. `clip`
    // is unaffected either way, because a representative's seconds are already track time.
    let offset = speaker_offset_seconds(&metadata);
    let snippet_track: &[f32] = if offset.is_ok() { &track } else { &[] };
    let offset = offset.unwrap_or(0.0);

    // What each voice was called when this queue was built. The guard below compares against
    // *this* rather than against the live labels, because under `--correct` a queued voice may
    // legitimately be one the database had already named.
    let baseline = shown.clone();

    // The total every prompt below carries. Read off the same list the session line counted
    // one call ago, so the two cannot drift apart.
    let of = offered.len();

    // The queue, numbered once. `nth` travels with the voice from here on rather than being
    // counted per pass, which is what makes a deferred voice come back as the same question --
    // see [`Position`].
    let mut pending: Vec<(usize, &SpeakerCluster)> = offered
        .into_iter()
        .enumerate()
        .map(|(index, cluster)| (index + 1, cluster))
        .collect();

    // Passes over a shrinking list rather than one walk, so that [`Answer::Later`] can put a
    // voice back. Each pass asks about whatever is still pending; anything deferred is asked
    // again on the next one.
    loop {
        let asked = pending.len();
        let mut deferred: Vec<(usize, &SpeakerCluster)> = Vec::new();

        // Iterated by reference rather than consumed, because [`Answer::Leave`] has to reach the
        // tail of the queue -- the voices this pass has not asked about yet -- from inside the
        // body, and a consuming `for` has thrown them away by then. `(usize, &SpeakerCluster)` is
        // `Copy`, so the pattern costs nothing, and nothing in the body mutates `pending`: it is
        // reassigned only below the loop.
        for (index, &(nth, cluster)) in pending.iter().enumerate() {
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
                let snippets = snippets_for(&transcript, cluster.id, snippet_track, offset);

                // Every voice in the session, built here and now rather than once above the
                // loop, for the reason `Voice::queue` gives and because the borrow checker
                // insists: `shown` is *reassigned* at the end of an accepted answer, so rows
                // borrowing it cannot outlive one question. That is the same thing as the rows
                // being current, which is why no separate refresh exists.
                //
                // `order` and not `pending`: a queue pane is the session, so the quiet voices
                // and the already-named ones are in it whether or not this run asks about them.
                let rows: Vec<Queued<'_>> = order
                    .iter()
                    .map(|c| Queued {
                        number: &unknown[&c.id],
                        attribution: &shown[&c.id],
                        speech_seconds: c.speech_seconds,
                        // Strictly less than the floor: a cluster sitting exactly on it is
                        // offered, which is the convention every floor in this codebase states.
                        below_floor: c.speech_seconds < PROMPT_FLOOR_SECONDS,
                    })
                    .collect();

                // Computed eagerly for every voice, including on the `--name` path where nothing
                // reads it, and deliberately not deferred behind a closure: two dozen people at
                // 256 dimensions is a few thousand multiply-adds, against a run that has already
                // read and resampled the whole speaker track a few lines above. An owned `Vec`
                // rather than a borrow of the database, so the reborrow ends here and nothing
                // downstream -- least of all an `Interviewer` -- has to reason about the write
                // that replaces `speakers` once this answer is accepted.
                interviewer.identify(&Voice {
                    session: &session.id,
                    meeting: meeting.as_ref(),
                    position: Position { nth, of },
                    attribution,
                    number: &unknown[&cluster.id],
                    speech_seconds: cluster.speech_seconds,
                    queue: &rows,
                    snippets,
                    clip: clip_for(&track, cluster),
                    resembles: rank_enrolled(&cluster.embedding, speakers),
                    // The universe `resolve()` requires, and not the ranking above -- see
                    // `Voice::enrolled`. Owned borrows, like `resembles`, so the reborrow of
                    // the database ends with this block.
                    enrolled: speakers.enrolled_names(),
                    // Six borrows and no work: what an answer would do is computed only if the
                    // answerer asks. Nothing is written by asking, so this is safe to hand out
                    // even though it holds the database -- see `Voice::preview`.
                    preview: Preview::new(
                        &clusters.clusters,
                        &unknown,
                        speakers,
                        &assigned,
                        cluster,
                        rules.enrolment,
                    ),
                })
            };

            let (name, anyway) = match answer {
                Answer::Quit => return Ok(Outcome::Quit),
                // The rest of this session, in the three groups it comes in and no fourth: the
                // voice that was on the screen when the key was pressed -- asked about, and
                // decided against, which is the same thing a deferral with no later turns out to
                // be -- then the ones this pass has not reached, then the ones it has already
                // deferred. Voices the guard above took out of the pass are in none of them,
                // which is what makes the counts add up.
                //
                // Returning from inside the loop is the whole implementation of leaving: every
                // write already happened per accepted name, and there is nothing between here and
                // the end of the function, so nothing is skipped by going early.
                Answer::Leave => {
                    let rest = std::iter::once(cluster)
                        .chain(pending[index + 1..].iter().map(|&(_, c)| c))
                        .chain(deferred.iter().map(|&(_, c)| c));
                    let left = left_unanswered(rest, &shown, &baseline, report);
                    about(notes, &session.id, SessionNote::Left { left })?;
                    return Ok(Outcome::Finished);
                }
                Answer::Skip => {
                    left_unanswered(std::iter::once(cluster), &shown, &baseline, report);
                    continue;
                }
                // Back into the queue with the number it already has, and counted as nothing:
                // it has not been answered yet. The pass that finds nobody willing to answer is
                // where these turn into skips -- see the fixed point below.
                Answer::Later => {
                    deferred.push((nth, cluster));
                    continue;
                }
                Answer::Named { name, anyway } => (name, anyway),
            };
            // Everything this answer would write, worked out on copies first -- the dry run the
            // `consequence` module holds, and the same one an [`Interviewer`] may have already run
            // through `Voice::preview`. Two files can carry a name and both feed the same
            // labelling, so the only way to know what an answer *does* is to build the state it
            // would leave and label the session through it; and the answer is not simply written
            // and inspected afterwards because undoing a write that turned out to cost somebody
            // their name means writing three files back, with a run interrupted mid-undo leaving
            // exactly the mess this prevents.
            //
            // Built here rather than held from the prompt above because the commit below needs
            // `speakers` mutably and a live `Preview` would keep it borrowed. It is the same six
            // references and the same `of`, so the preview an answerer saw and the write cannot
            // disagree.
            //
            // `None` is a name of nothing but spaces: somebody pressing Enter with a stray
            // keystroke in the buffer, not a request for an entry called "". Where that is decided
            // is `of`, so this path and a preview agree about it too.
            let Some(consequence) = Preview::new(
                &clusters.clusters,
                &unknown,
                speakers,
                &assigned,
                cluster,
                rules.enrolment,
            )
            .of(&name) else {
                left_unanswered(std::iter::once(cluster), &shown, &baseline, report);
                continue;
            };
            let name = name.trim();

            // The refusal. An answer that would take a name off a voice the user is not answering
            // about is not honoured -- see `Refusal` for the three ways that can happen and why
            // one check covers them -- unless the answer itself says otherwise. Written as one
            // total match rather than as a guard plus an exception, because the rule is which of
            // the three cases an answer falls into and reading it should not require holding a
            // negation.
            match &consequence.refused {
                // Shown what it costs and asked for it anyway. `Answer::anyway` is only ever set
                // by an interface that displayed the paying voice and what it loses before a key
                // was pressed, which makes this `forget --yes`'s argument reached from the other
                // side: see `forget.rs`'s "Nothing is ever refused". Everything below runs
                // exactly as it does for an answer nothing refused -- honouring an override is
                // skipping this guard, not a second write path.
                Some(Refusal::Taken { voice, losing }) if anyway => {
                    after(
                        notes,
                        &session.id,
                        AnswerNote::Overrode {
                            name,
                            answered: &handle(cluster.id, &unknown),
                            voice,
                            losing,
                        },
                    )?;
                }
                // Every other refusal: a `Taken` nobody insisted on, and a `Vetoed` however
                // insistent the answer was. Nothing is written, the voice keeps whatever it
                // read, and the note names the voice that would have paid.
                Some(refusal) => {
                    let answered = handle(cluster.id, &unknown);
                    after(
                        notes,
                        &session.id,
                        AnswerNote::Refused {
                            name,
                            voice: &answered,
                            refusal,
                        },
                    )?;
                    report.refused += 1;
                    continue;
                }
                None => {}
            }

            // Everything this answer wrote, as one note rather than as the four to six lines it
            // used to print, because that is the block an interface lays out together.
            //
            // Narrated *before* the copies are taken out of the consequence below: nothing between
            // here and there writes a byte, so the order the user sees is unchanged, and a
            // partially moved `Consequence` can no longer be borrowed.
            after(
                notes,
                &session.id,
                AnswerNote::Committed {
                    name,
                    speech_seconds: cluster.speech_seconds,
                    consequence: &consequence,
                },
            )?;
            // A sub-count of `named`, and now read off the type rather than off the two arms of
            // the match that used to print those two sentences -- which is what
            // `Consequence::session_only` is documented to be.
            if consequence.session_only() {
                report.session_only += 1;
            }
            report.named += 1;

            // Committed by taking the copies the dry run produced, so what lands on disk is the
            // state that was checked rather than a second construction of it.
            let speakers_changed = *speakers != consequence.speakers;
            let assignments_changed = assigned.names != consequence.assigned.names;
            *speakers = consequence.speakers;
            assigned = consequence.assigned;

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
                transcript.write(
                    &session.paths,
                    rules.template,
                    &TranscriptContext::now(&metadata),
                )?;
            }
            // Only on the timestamp path. A user who pointed at a moment did not choose the voice,
            // so how far the rename reached is the one thing they cannot infer -- whereas the queue
            // and `--voice` both showed them the voice first, and several tests pin their output
            // exactly as it is.
            if matches!(rules.selector, Some(Selection::At(_))) {
                report_rename(&transcript, &shown, &now, name, session, notes)?;
            }
            shown = now;
        }

        // The fixed point: a pass that moved nobody out of the deferred set, *and* an answerer
        // that says it has nothing left to do. Every other pass leaves `deferred.len() < asked`,
        // and the set can only shrink, so the first half terminates on its own; the second is
        // [`Interviewer::still_working`]'s contract to keep bounded.
        //
        // The size of the set and not "no answer other than `Later` came back", because the
        // in-run guard above takes a voice out of a pass without any answer being given -- an
        // earlier answer named it -- and that is progress too. Counting answers would end a
        // session while there were still questions the user had not been asked.
        //
        // And the answerer as well as the set, because a stalled pass is not the same fact as a
        // finished session for an interface with a cursor: it defers voices in order to reach
        // another one, so a pass where the user only moved around produces no answer and is not
        // the user being done. An empty queue is decided here rather than there -- nothing is
        // left to offer, so no further prompt could change the answer or carry an
        // [`Answer::Quit`], and consulting the answerer could only spin.
        //
        // Still only about a pass that produced no answer: [`Answer::Leave`] is an answer and
        // returns above this, so leaving a session never reaches the question this asks.
        if deferred.len() == asked && (asked == 0 || !interviewer.still_working()) {
            // Deferred with no later left is the skip -- or the kept identification -- it has
            // turned out to be, counted through the same rule every other unanswered voice goes
            // through so no two of them can disagree about which bucket a named voice is in.
            left_unanswered(
                deferred.iter().map(|&(_, cluster)| cluster),
                &shown,
                &baseline,
                report,
            );
            break;
        }
        pending = deferred;
    }

    Ok(Outcome::Finished)
}

/// Counts voices that were offered and not answered into the buckets they have turned out to
/// belong in, and says how many that was.
///
/// Leaving an already-named voice alone is keeping that identification, which is an answer;
/// leaving an unnamed one alone is the question going unanswered. Same write -- none -- and
/// different enough that the summary must not conflate them. One function rather than the rule
/// written out at each of the four places that needs it -- a skip, a name of nothing but spaces,
/// [`Answer::Leave`]'s tail, and the pass loop's fixed point -- so none of them can disagree
/// with the others about which bucket a voice belongs in.
///
/// `shown` is what each voice reads now and `baseline` what it read when the queue was built. A
/// voice that is named and has *moved* since the baseline was taken was named by an answer given
/// earlier in this run -- clustering split one person in two, so naming one half named the other
/// -- and has already been counted under `named`. It is counted here as nothing at all, because
/// reporting it as an identification this run left alone would put one voice in two buckets.
///
/// That guard is load-bearing only for [`Answer::Leave`], whose tail can hold a voice this same
/// pass has just named. Everywhere else it cannot fire -- the pass loop's own guard takes such a
/// voice out before it can be asked about or deferred -- which is what keeps every existing
/// count byte-identical.
fn left_unanswered<'c>(
    voices: impl IntoIterator<Item = &'c SpeakerCluster>,
    shown: &BTreeMap<u32, Attribution>,
    baseline: &BTreeMap<u32, Attribution>,
    report: &mut EnrollReport,
) -> usize {
    let mut counted = 0;
    for cluster in voices {
        let named = shown[&cluster.id].is_named();
        if named && shown[&cluster.id] != baseline[&cluster.id] {
            continue;
        }
        counted += 1;
        if named {
            report.kept += 1;
        } else {
            report.skipped += 1;
        }
    }
    counted
}

/// The voices one session's run will ask about, in first-appearance order, and the line
/// saying so -- or `None` for a session with nothing to ask about, which has been reported
/// and counted.
///
/// Separated from the asking so that the one decision a [`VoiceSelector`] changes is made in
/// one place: [`targeted`] is the sibling of this, and everything downstream of both is shared.
#[allow(clippy::too_many_arguments)]
fn queue<'c>(
    order: &[&'c SpeakerCluster],
    shown: &BTreeMap<u32, Attribution>,
    offer: Offer,
    sessions: Sessions,
    meeting: Option<MeetingLabel>,
    session: &DiscoveredSession,
    notes: &mut dyn Narrator,
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
    // The two halves of the pass-over, which used to be one: a session with no candidates at
    // all, and a session whose candidates are all already named. Only the second is
    // [`Sessions`]'s to overrule -- a session with no clusters in it has nothing to draw
    // however hard a caller asks -- which is why the emptiness test stays rather than folding
    // into the count.
    //
    // Behaviour is unchanged for the two combinations that predate the split, and the third is
    // the reason for it. With `offer.named` false -- the plain path -- every candidate is
    // unresolved, so `unresolved == 0` holds exactly when the list is empty and this is the
    // `candidates.is_empty()` gate it replaces. With `--correct` it is true alongside
    // `Sessions::Every`, so the count is not consulted. The full-screen frame is the third: it
    // sets `offer.named` so its queue pane can reach every voice, but leaves
    // `Sessions::Unresolved`, and this count is what then keeps it from opening on every
    // finished meeting on disk.
    let unresolved = candidates
        .iter()
        .filter(|c| !shown[&c.id].is_named())
        .count();
    if candidates.is_empty() || (unresolved == 0 && sessions == Sessions::Unresolved) {
        // A session whose voices are all identified is exactly where somebody stands when one
        // of those identifications is wrong, and this note is the only thing it produces -- so
        // it carries the count that reaches the escape, the way the held-back one names `--all`.
        let named = shown.values().filter(|label| label.is_named()).count();
        about(
            notes,
            &session.id,
            SessionNote::PassedOver(PassedOver::NothingUnresolved { named }),
        )?;
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

    // `offered.len()` here is the same number every prompt below carries as its [`Position`]
    // total, because both read this list. Anything that computes this count independently
    // breaks that.
    //
    // `already_named` is `Some` exactly under `--correct`, which is what makes the queue a
    // review rather than a list of unknowns -- and so is what picks between the two wordings.
    about(
        notes,
        &session.id,
        SessionNote::Queue {
            offered: offered.len(),
            already_named: offer
                .named
                .then(|| offered.iter().filter(|c| shown[&c.id].is_named()).count()),
            held_back,
            meeting,
        },
    )?;

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
    notes: &mut dyn Narrator,
    report: &mut EnrollReport,
) -> Result<Option<Vec<&'c SpeakerCluster>>> {
    let matched: Vec<&SpeakerCluster> = order
        .iter()
        .copied()
        .filter(|c| selector.matches(&unknown[&c.id], &shown[&c.id]))
        .collect();

    let describe = |c: &SpeakerCluster| describe(c, unknown, shown);

    match matched.len() {
        1 => {
            about(
                notes,
                &session.id,
                SessionNote::Selected {
                    at: None,
                    voice: describe(matched[0]),
                },
            )?;
            Ok(Some(matched))
        }
        0 => {
            // Every voice, quiet ones included: a miss is usually a number off by one or a name
            // spelled as the user remembers it rather than as the transcript has it -- and the
            // quiet voices are exactly what somebody is reaching for when they miss. Fifty-odd
            // lines on a real session is still far cheaper than fifty-odd prompts.
            about(
                notes,
                &session.id,
                SessionNote::NotSelected(NotSelected::NoVoiceMatched {
                    selector,
                    voices: order.iter().copied().map(describe).collect(),
                }),
            )?;
            report.failed += 1;
            Ok(None)
        }
        _ => {
            about(
                notes,
                &session.id,
                SessionNote::NotSelected(NotSelected::SeveralVoicesMatched {
                    selector,
                    voices: matched.iter().copied().map(describe).collect(),
                }),
            )?;
            report.failed += 1;
            Ok(None)
        }
    }
}

/// How one voice reads in a message about several: the number it is reachable by, the name it
/// currently carries, and how much it spoke.
///
/// Shared by both selectors so that a list of candidates reads the same however the user missed:
/// the number is what the message hands back, and it has to be the same number in both. Which of
/// the three fields end up in a sentence, and in what order, belongs to the `narration` module.
fn describe(
    cluster: &SpeakerCluster,
    unknown: &BTreeMap<u32, String>,
    shown: &BTreeMap<u32, Attribution>,
) -> VoiceDescription {
    VoiceDescription {
        number: unknown[&cluster.id].clone(),
        label: shown[&cluster.id].label().to_string(),
        speech_seconds: cluster.speech_seconds,
    }
}

/// The one voice speaking at a moment of this session, or `None` when that moment names no voice
/// -- which is reported and counted as a request that could not be served.
///
/// The third sibling of [`queue`] and [`targeted`], and deliberately nothing more than that:
/// it produces the same one-element list they do, so a timestamp is a way of *arriving* at a
/// voice rather than a second way of enrolling one. Everything downstream -- the prompt, the
/// pre-flight, the refusal, the three writes -- is shared, which is what makes the reference
/// floor, the already-enrolled safeguards and the two transcript files behave here exactly as
/// they do everywhere else.
///
/// No floor, no `--correct` gate and no pass-over, for the reason [`targeted`] gives: pointing at
/// a moment is already the judgement those gates make on the user's behalf.
///
/// Every refusal names the timestamp back in the spelling it was given, so the line can be read
/// beside the transcript the user copied it from.
#[allow(clippy::too_many_arguments)]
fn at_timestamp<'c>(
    at: TranscriptTime,
    transcript: &Transcript,
    order: &[&'c SpeakerCluster],
    unknown: &BTreeMap<u32, String>,
    shown: &BTreeMap<u32, Attribution>,
    session: &DiscoveredSession,
    notes: &mut dyn Narrator,
    report: &mut EnrollReport,
) -> Result<Option<Vec<&'c SpeakerCluster>>> {
    // Each non-answer says which of them it was and what to do about it: they are four
    // different situations, and only one of them is the user's mistake.
    let voice = match transcript.voice_at(at) {
        VoiceAt::Cluster(id) => id,
        VoiceAt::LocalSpeaker => {
            missed(notes, session, NotSelected::OnTheMicrophone { at }, report)?;
            return Ok(None);
        }
        VoiceAt::NoCluster => {
            missed(notes, session, NotSelected::NoClusters { at }, report)?;
            return Ok(None);
        }
        VoiceAt::Silence => {
            // A miss here is usually a second or two off, and the user is holding the file with
            // the right timestamp in it, so the nearest turn is worth more than the refusal.
            let nearest = transcript
                .turns
                .iter()
                .min_by(|a, b| gap_to(a, at).total_cmp(&gap_to(b, at)))
                .map(|turn| Nearest {
                    speaker: turn.speaker.clone(),
                    at: TranscriptTime::of(turn.start),
                });
            missed(notes, session, NotSelected::Silence { at, nearest }, report)?;
            return Ok(None);
        }
        VoiceAt::PastEnd { last } => {
            missed(
                notes,
                session,
                NotSelected::PastEnd {
                    at,
                    last: TranscriptTime::of(last),
                },
                report,
            )?;
            return Ok(None);
        }
    };

    // Two voices can print the same label -- turns a fraction of a second apart round to the
    // same second -- and then the timestamp names neither of them on its own. That is a question
    // this command cannot answer for the user, so it hands back the thing that tells them apart,
    // exactly as an ambiguous `--voice` does.
    let candidates = transcript.clusters_at(at);
    if candidates.len() > 1 {
        // The count comes off the transcript rather than off the voices below, which are looked
        // up in `speaker_clusters.json`: a transcript naming a cluster that file no longer has
        // would otherwise be reported as fewer turns than it has.
        missed(
            notes,
            session,
            NotSelected::SeveralVoicesAt {
                at,
                count: candidates.len(),
                voices: candidates
                    .iter()
                    .filter_map(|id| order.iter().find(|c| c.id == *id))
                    .map(|c| describe(c, unknown, shown))
                    .collect(),
            },
            report,
        )?;
        return Ok(None);
    }

    // A voice the transcript names and the clusters file does not is the stale-file failure the
    // rest of this crate already has wording for, reached from the other side.
    let Some(cluster) = order.iter().copied().find(|c| c.id == voice) else {
        missed(
            notes,
            session,
            NotSelected::VoiceNotInClusters { at },
            report,
        )?;
        return Ok(None);
    };

    // The same note [`targeted`] produces, plus the moment it was reached by: the user named a
    // timestamp and gets told which voice that turned out to be, which is the one thing they
    // did not already know.
    about(
        notes,
        &session.id,
        SessionNote::Selected {
            at: Some(at),
            voice: describe(cluster, unknown, shown),
        },
    )?;
    Ok(Some(vec![cluster]))
}

/// A request that could not be served: the reason, and the one counter every one of them lands
/// in. Together, because the two have never come apart -- see [`EnrollReport::failed`].
fn missed(
    notes: &mut dyn Narrator,
    session: &DiscoveredSession,
    why: NotSelected<'_>,
    report: &mut EnrollReport,
) -> Result<()> {
    about(notes, &session.id, SessionNote::NotSelected(why))?;
    report.failed += 1;
    Ok(())
}

/// A session file that would not read, reported against the remedy its kind has.
fn unreadable(
    notes: &mut dyn Narrator,
    session: &DiscoveredSession,
    file: SessionFile,
    error: &meethook_session::Error,
) -> Result<()> {
    about(notes, &session.id, SessionNote::Unreadable { file, error })
}

/// How far a turn is from an instant: zero while the instant is inside it.
fn gap_to(turn: &meethook_session::Turn, at: TranscriptTime) -> f64 {
    let instant = at.seconds();
    if instant < turn.start {
        turn.start - instant
    } else {
        (instant - turn.end).max(0.0)
    }
}

/// Says how much of the transcript naming one voice just rewrote.
///
/// Only the timestamp path prints this, and the reason is what it is measured from: the
/// difference between what every voice read *before* the answer and what it reads *after*, not
/// the voice that was selected. Naming one cluster can name a second when clustering split that
/// person in two, so the selection is not the blast radius -- the label diff is.
///
/// The turns are counted and their durations summed rather than the clusters'
/// `speech_seconds` taken: the claim is about the lines this command rewrote in the file the
/// user is reading, and those are two different quantities.
fn report_rename(
    transcript: &Transcript,
    before: &BTreeMap<u32, Attribution>,
    after: &BTreeMap<u32, Attribution>,
    name: &str,
    session: &DiscoveredSession,
    notes: &mut dyn Narrator,
) -> Result<()> {
    let mut renamed: Vec<u32> = Vec::new();
    for (id, label) in after {
        if before.get(id) != Some(label) {
            renamed.push(*id);
        }
    }
    let (turns, seconds) = transcript
        .turns
        .iter()
        .filter(|turn| {
            turn.source_track == SourceTrack::Speaker
                && turn.cluster.is_some_and(|id| renamed.contains(&id))
        })
        .fold((0usize, 0.0f64), |(count, total), turn| {
            (count + 1, total + (turn.end - turn.start))
        });

    // Spelled out because `after` is also the name of this function's label map.
    narration::after(
        notes,
        &session.id,
        AnswerNote::Renamed {
            name,
            turns,
            seconds,
        },
    )
}

/// How much somebody said, in the units a person would say it in.
///
/// Public because the prompt in the CLI and the rename line above both print a duration, and one
/// tool printing a duration two ways is a defect rather than a style.
pub fn speech(seconds: f64) -> String {
    let seconds = seconds.round() as u64;
    match seconds / 60 {
        0 => format!("{seconds}s"),
        minutes => format!("{minutes}m {:02}s", seconds % 60),
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
///
/// Visible to the crate rather than to this file because [`forget`] brings a transcript in line
/// through exactly this too: a removal that rewrote transcripts any other way would be a second
/// producer of the labels `merge` writes.
pub(crate) fn relabel(transcript: &mut Transcript, labels: &BTreeMap<u32, Attribution>) -> bool {
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

/// The samples between two offsets into the 16 kHz speaker track.
///
/// The one place the clamping rule lives, because both things that cut audio out of a track --
/// a voice's clip and a line's snippet -- have to obey the same one: a range running off the
/// end of the track yields what is there, and anything left empty is audio that is missing
/// rather than a session that fails.
fn samples_between(track: &[f32], start: f64, end: f64) -> &[f32] {
    let start = sample_at(start).min(track.len());
    let end = sample_at(end).min(track.len());
    if end <= start {
        return &[];
    }
    &track[start..end]
}

/// The audio to play for one voice: its longest representative, cut out of the speaker track.
///
/// The clip is sliced rather than seeked to because the plainest players in the search list
/// (`afplay`, `paplay`, `aplay`) take no start offset at all -- so somebody has to extract it
/// either way. Slicing the 16 kHz track
/// diarization itself ran on is what makes the seconds in a [`meethook_session::RepresentativeSegment`]
/// impossible to misinterpret: they are offsets into exactly this buffer.
fn clip_for<'a>(track: &'a [f32], cluster: &SpeakerCluster) -> &'a [f32] {
    let Some(segment) = cluster.representatives.first() else {
        return &[];
    };
    samples_between(track, segment.start, segment.end)
}

/// Every line one voice said, as a prompt needs them: the text, when it was said, and the
/// samples for it.
///
/// `offset` is [`meethook_transcribe::speaker_offset_seconds`] -- what turns a
/// session-timeline second into an offset into `track`. Same filter, same trim, same order as
/// the text-only list this replaced, so the lines a prompt shows do not move.
fn snippets_for<'a>(
    transcript: &'a Transcript,
    cluster: u32,
    track: &'a [f32],
    offset: f64,
) -> Vec<Snippet<'a>> {
    transcript
        .turns
        .iter()
        .filter(|turn| turn.source_track == SourceTrack::Speaker && turn.cluster == Some(cluster))
        .filter_map(|turn| {
            let text = snippet(&turn.text);
            if text.is_empty() {
                return None;
            }
            // Both clamps are the habit `sample_at` and `samples_between` already have of
            // defining the edge away rather than checking for it. A speaker-track turn cannot
            // begin before the speaker track by construction, so neither is a real case.
            let start = (turn.start - offset).max(0.0);
            let end = (turn.end - offset).max(start);
            Some(Snippet {
                text,
                start,
                duration: end - start,
                audio: samples_between(track, start, end),
            })
        })
        .collect()
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
    use std::cell::Cell;
    use std::collections::VecDeque;

    use meethook_session::{
        Attendee, AttendeeStatus, EnrolledSpeaker, MAX_REFERENCES_PER_SPEAKER, Meeting, MeetingFit,
        RepresentativeSegment, SPEAKER_YOU, SessionMetadata, SessionPaths, Stored,
        TRANSCRIPT_SCHEMA_VERSION, TrackSync, Turn,
    };
    // The cut the ranking is deliberately *not* made at, named rather than spelled 0.40, so
    // the fixtures below still mean "outside identification's reach" if it moves.
    use meethook_transcribe::IDENTIFY_DISTANCE;

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
            Some(Selection::Voice(&selector)),
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
        selection: Option<Selection<'_>>,
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

    /// A clip exists to be handed to an audio player, so its header is part of what it is for: a
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

    /// Six seconds of a 16 kHz track whose every sample says which sample it is, so a slice
    /// out of it can be checked for *where* it came from and not merely how long it is.
    fn counted_track() -> Vec<f32> {
        (0..16_000 * 6).map(|i| i as f32).collect()
    }

    /// A transcript with one speaker turn per `(start, text)`, all from cluster 0.
    fn transcript_of_lines(lines: &[(f64, &str)]) -> Transcript {
        Transcript::new(
            SessionId::parse("20260809-052600").unwrap(),
            lines
                .iter()
                .map(|(start, text)| speaker_turn(*start, 0, "Unknown 1", text))
                .collect(),
        )
    }

    /// Acceptance criterion #2: a snippet's seconds are offsets into `speaker.wav`, not into the
    /// session timeline, and its samples come from where those seconds point.
    ///
    /// The first-sample assertion is the point of the counted track: a length-only check passes
    /// just as happily when the offset is applied with the wrong sign.
    #[test]
    fn snippet_times_are_track_time_and_the_audio_starts_there() {
        let track = counted_track();
        let transcript = transcript_of_lines(&[(3.0, "and from me")]);

        let snippets = snippets_for(&transcript, 0, &track, 1.0);

        assert_eq!(snippets.len(), 1);
        assert_eq!((snippets[0].start, snippets[0].duration), (2.0, 1.0));
        assert_eq!(snippets[0].audio.len(), 16_000);
        assert_eq!(
            snippets[0].audio[0], 32_000.0,
            "two seconds into the track, which is the turn's three seconds less the offset"
        );
    }

    /// Acceptance criterion #3: a line running off the end of a truncated track keeps the
    /// duration the transcript gave it and plays what survives -- nothing at all when the line
    /// is entirely past the end.
    #[test]
    fn a_snippet_past_the_end_of_the_track_carries_what_is_there() {
        let track = counted_track();
        let transcript = transcript_of_lines(&[(5.5, "trailing off"), (30.0, "long gone")]);

        let snippets = snippets_for(&transcript, 0, &track, 0.0);

        assert_eq!((snippets[0].start, snippets[0].duration), (5.5, 1.0));
        assert_eq!(
            snippets[0].audio.len(),
            8_000,
            "half the line is on disk; the duration still says how long it took"
        );
        assert_eq!((snippets[1].start, snippets[1].duration), (30.0, 1.0));
        assert!(snippets[1].audio.is_empty());
    }

    /// Acceptance criterion #3, the other half: no track is no audio and not a lost line. The
    /// times are still there, which is what leaves a later run able to say what could not be
    /// played.
    #[test]
    fn an_empty_track_leaves_every_snippet_with_its_times_and_no_audio() {
        let transcript = transcript_of_lines(&[(0.0, "hi there"), (4.0, "let us start")]);

        let snippets = snippets_for(&transcript, 0, &[], 0.0);

        assert_eq!(
            snippets
                .iter()
                .map(|s| (s.start, s.duration))
                .collect::<Vec<_>>(),
            [(0.0, 1.0), (4.0, 1.0)]
        );
        assert!(snippets.iter().all(|s| s.audio.is_empty()));
    }

    /// Acceptance criterion #4: the same lines, the same text and the same order as the
    /// text-only list this replaced -- this voice's turns only, trimmed, cut, and with the ones
    /// the recogniser heard nothing over dropped rather than carried as empty rows.
    #[test]
    fn the_snippets_are_the_same_lines_in_the_same_order_as_before() {
        let long = "x".repeat(SNIPPET_CHARS + 20);
        let transcript = Transcript::new(
            SessionId::parse("20260809-052600").unwrap(),
            vec![
                speaker_turn(0.0, 0, "Unknown 1", "  hi there  "),
                mic_turn(1.0, "morning"),
                speaker_turn(2.0, 1, "Unknown 2", "and from me"),
                speaker_turn(3.0, 0, "Unknown 1", "   "),
                speaker_turn(4.0, 0, "Unknown 1", &long),
            ],
        );

        let snippets = snippets_for(&transcript, 0, &[], 0.0);

        assert_eq!(
            snippets.iter().map(|s| s.text).collect::<Vec<_>>(),
            ["hi there", &"x".repeat(SNIPPET_CHARS)],
            "the mic turn, the other voice and the line with nothing in it are all out"
        );
    }

    /// The clamping rule both a clip and a snippet obey, stated once where it lives.
    #[test]
    fn samples_between_clamps_to_what_the_track_holds() {
        let track = counted_track();

        assert_eq!(samples_between(&track, 1.0, 2.0).len(), 16_000);
        assert_eq!(samples_between(&track, 1.0, 2.0)[0], 16_000.0);
        assert_eq!(samples_between(&track, 5.0, 90.0).len(), 16_000);
        assert!(samples_between(&track, 600.0, 620.0).is_empty());
        assert!(samples_between(&track, 2.0, 2.0).is_empty());
        assert!(samples_between(&[], 0.0, 1.0).is_empty());
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
            Some(Selection::Voice(&selector)),
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
            Some(Selection::Voice(&second)),
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

    /// A long line is cut to something that fits a prompt, on a character boundary rather
    /// than a byte one.
    #[test]
    fn a_long_snippet_is_cut_to_a_readable_length() {
        let long = "é".repeat(SNIPPET_CHARS * 2);
        assert_eq!(snippet(&long).chars().count(), SNIPPET_CHARS);
        assert_eq!(snippet("  short  "), "short");
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
}
