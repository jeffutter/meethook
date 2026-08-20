//! What a run *said* happened, as values rather than as lines.
//!
//! `run_enroll` used to narrate straight into a `&mut dyn Write`, which made stdout part of its
//! signature. A full-screen interface cannot take that: written to a terminal that has been
//! swapped for the alternate screen the lines are either invisible or they tear the frame, and a
//! pre-formatted sentence cannot be wrapped into a pane, styled, or placed beside the voice it
//! is about. So the run emits [`Note`]s through the one-method [`Narrator`] seam, in the shape of
//! [`Interviewer`](crate::Interviewer) next door, and the rendering is somebody else's job.
//!
//! Three things about this module are worth stating, because none of them can be read off the
//! types:
//!
//! **[`Lines`] is the definition of the CLI's output.** Every string the enroll path prints is
//! here and nowhere else, character for character, and the enroll tests in `lib.rs` assert on
//! those strings verbatim through the `run_over` harness. That suite is the gate: a note whose
//! rendering drifts fails a test rather than reaching a user.
//!
//! **The wording stays library-side on purpose.** Moving a sentence into the interface that
//! displays it would move it out of `cargo test`, where the phrasing of a refusal -- which names
//! the command that undoes it -- is currently assertable. An interface implementing [`Narrator`]
//! and inventing wording of its own has undone that, and should be reaching for a note it can
//! lay out instead.
//!
//! **The three tiers exist so a pane can place them differently.** [`Note::Run`] is the run
//! refusing to start or finding nothing to visit; [`Note::Session`] is an event about one
//! session, whether or not this run asked anything about it; [`Note::Answer`] is the consequence
//! of one answer about one voice. A log pane wants the middle tier in arrival order and the last
//! tier beside the voice it was about, which it cannot do if both arrive as sentences.
//!
//! # Why the seam is fallible
//!
//! [`Narrator::note`] returns [`Result`], unlike `Interviewer::identify`. A failed `writeln!`
//! became [`Error::Output`](crate::Error) before this module existed, which is what makes
//! `meethook enroll | head` exit non-zero rather than pretend it printed. An infallible seam
//! would swallow that. An implementation with nowhere for a write to fail -- a full-screen one
//! pushing into a `Vec` -- returns `Ok(())` and pays nothing.

use std::io::Write;
use std::path::Path;

use meethook_session::{SessionId, Stored, TranscriptTime};

use crate::{
    Consequence, REFERENCE_FLOOR_SECONDS, Refusal, Result, Selection, VoiceSelector, speech,
};

/// Where a run's narration goes.
///
/// One method, so that an implementation is a `match` rather than a checklist of fourteen
/// callbacks with an order to get wrong. [`Lines`] is the one every command in this tool uses;
/// an interface writes its own.
pub trait Narrator {
    /// Report one thing that happened. Called between prompts, never during one.
    fn note(&mut self, note: Note<'_>) -> Result<()>;
}

/// One thing a run has to say, at the granularity an interface needs to place it.
pub enum Note<'a> {
    /// Before or across sessions: the run refused to start, or had nothing to visit.
    Run(RunNote<'a>),

    /// One session, whatever this run did or did not ask about it.
    Session {
        /// The session it is about.
        session: &'a SessionId,
        /// What happened to it.
        note: SessionNote<'a>,
    },

    /// The consequence of one answer about one voice.
    Answer {
        /// The session the voice belongs to.
        session: &'a SessionId,
        /// What the answer did.
        note: AnswerNote<'a>,
    },
}

/// Something about the run as a whole, said before any session has been opened.
pub enum RunNote<'a> {
    /// `--voice` or `--at` was passed with anything other than exactly one session id.
    SelectionNeedsOneSession {
        /// Which of the two flags it arrived on, which is also why one id and not several.
        selection: Selection<'a>,
    },

    /// `--name` was passed with no `--at` and no `--voice`, so the name has no voice to land on.
    NameNeedsAVoice,

    /// A requested session id that is not on disk. One note per missing id.
    SessionNotFound {
        /// The id as it was requested.
        id: &'a SessionId,
    },

    /// No sessions at all, on a run that asked for no particular one.
    NoSessionsFound {
        /// The directory that was looked in.
        dir: &'a Path,
    },
}

/// Something about one session: a reason it was passed over, a file that would not read, or the
/// queue of voices this run is about to ask about.
pub enum SessionNote<'a> {
    /// Nothing was asked about this session, for one of three reasons.
    PassedOver(PassedOver),

    /// A file this session needs could not be read. The run carries on with the next session.
    Unreadable {
        /// Which file, which is what decides the remedy.
        file: SessionFile,
        /// What went wrong reading it.
        error: &'a meethook_session::Error,
    },

    /// The transcript disagreed with the database and was rewritten before anything was asked.
    BroughtUpToDate,

    /// The queue this run built for the session.
    Queue {
        /// How many voices will be asked about, which is every prompt's [`Position`] total.
        ///
        /// [`Position`]: crate::Position
        offered: usize,

        /// How many of those already carry a name -- `Some` only under `--correct`, which is
        /// what makes the queue a review rather than a list of unknowns.
        already_named: Option<usize>,

        /// Unresolved voices under the prompt floor, which this run will not ask about.
        held_back: usize,
    },

    /// The user left the session before its queue ran out, and the run moved on to the next one.
    ///
    /// Its own line because the summary would otherwise report a skip count larger than the
    /// questions the user answered with nothing behind it, and because every other way a
    /// session can end already says so.
    Left {
        /// How many voices were left as they were, which is exactly what was added to the skips
        /// and the kept identifications.
        left: usize,
    },

    /// A selector or a timestamp arrived at exactly one voice.
    Selected {
        /// The moment it was reached by, on the `--at` path only.
        at: Option<TranscriptTime>,
        /// The voice it turned out to be.
        voice: VoiceDescription,
    },

    /// A selector or a timestamp arrived at no one voice. Counted as a request not served.
    NotSelected(NotSelected<'a>),
}

/// Why a session was never asked about.
pub enum PassedOver {
    /// No `session.json`: the recorder crashed mid-session.
    Orphaned,

    /// Recorded but not transcribed, so there are no voices to ask about yet.
    NotTranscribed,

    /// Every voice is already resolved.
    NothingUnresolved {
        /// How many of them carry a name, which is the escape `--correct` reaches.
        named: usize,
    },
}

/// Which of a session's files would not read. Each has its own remedy, and only one of them is
/// recoverable by re-transcribing.
pub enum SessionFile {
    /// `speaker_clusters.json`.
    Clusters,
    /// `session.json`.
    Metadata,
    /// `transcript.json`.
    Transcript,
    /// `speaker_names.json`.
    Names,
}

/// The eight ways a selector or a timestamp names no one voice.
pub enum NotSelected<'a> {
    /// `--voice` matched nothing in this session.
    NoVoiceMatched {
        /// What was looked for, in its normalised spelling.
        selector: &'a VoiceSelector,
        /// Every voice the session has, so the user can see what they missed.
        voices: Vec<VoiceDescription>,
    },

    /// `--voice` matched more than one voice, which happens when two share an enrolled name.
    SeveralVoicesMatched {
        /// What was looked for.
        selector: &'a VoiceSelector,
        /// The voices it matched.
        voices: Vec<VoiceDescription>,
    },

    /// `--at` is the printed label of turns by more than one voice.
    SeveralVoicesAt {
        /// The moment, in the spelling it was given.
        at: TranscriptTime,

        /// How many turns carry that label.
        ///
        /// Counted from the transcript rather than from `voices`, which is built by looking
        /// each one up in `speaker_clusters.json`: a transcript naming a cluster that file no
        /// longer has would otherwise be reported as fewer turns than it has.
        count: usize,

        /// The voices behind those turns, as far as the clusters file knows them.
        voices: Vec<VoiceDescription>,
    },

    /// `--at` landed on the microphone track, which is the person holding the machine.
    OnTheMicrophone {
        /// The moment.
        at: TranscriptTime,
    },

    /// `--at` landed on a turn that records no voice, because diarization found none.
    NoClusters {
        /// The moment.
        at: TranscriptTime,
    },

    /// `--at` landed between turns.
    Silence {
        /// The moment.
        at: TranscriptTime,
        /// The closest turn there is, which is usually what the user meant.
        nearest: Option<Nearest>,
    },

    /// `--at` is past the end of the recording.
    PastEnd {
        /// The moment.
        at: TranscriptTime,
        /// When the session actually ends.
        last: TranscriptTime,
    },

    /// The turn at `--at` came from a voice `speaker_clusters.json` does not have.
    VoiceNotInClusters {
        /// The moment.
        at: TranscriptTime,
    },
}

/// The turn nearest a moment nobody was speaking at.
pub struct Nearest {
    /// What that turn reads as.
    pub speaker: String,
    /// When it starts.
    pub at: TranscriptTime,
}

/// How one voice reads in a message about several.
///
/// Three fields rather than a joined sentence, so an interface can put them in three columns --
/// and so the number a message hands back and the number in the `--voice "..."` beside it are
/// the same value rather than two lookups that could disagree.
pub struct VoiceDescription {
    /// The "Unknown N" its first appearance earned it, which is the handle that reaches it
    /// whatever it currently reads as.
    pub number: String,

    /// What it reads as now. Equal to `number` for a voice nothing has named.
    pub label: String,

    /// How much it spoke.
    pub speech_seconds: f64,
}

/// What one answer about one voice did.
pub enum AnswerNote<'a> {
    /// The answer was not honoured, because honouring it would have taken a name off a voice
    /// the user was not asked about.
    Refused {
        /// The name that was given.
        name: &'a str,
        /// The voice it was given for, by its "Unknown N".
        voice: &'a str,
        /// Which of the three ways it would have cost somebody else.
        refusal: &'a Refusal,
    },

    /// The answer was honoured even though it took a name off a voice the user was not asked
    /// about, because the answer said to.
    ///
    /// The mirror of [`Refused`](Self::Refused): same cost, same voice paying for it, opposite
    /// outcome. It exists as its own note rather than as a field on
    /// [`Committed`](Self::Committed) so that every path that does not override prints exactly
    /// the bytes it printed before -- and so that the one new sentence lives in one place.
    ///
    /// Only [`Refusal::Taken`] can be overridden, so this carries that variant's two halves
    /// rather than a `Refusal`: the type cannot describe a veto that was overridden, which is
    /// the guarantee the enum would leave to a comment.
    Overrode {
        /// The name that was given, trimmed.
        name: &'a str,
        /// The voice it was given for, by its "Unknown N". Named separately from `voice`
        /// because the two are different voices and one word must not mean both.
        answered: &'a str,
        /// The voice that pays, by its "Unknown N" -- [`Refusal::Taken::voice`].
        voice: &'a str,
        /// What that voice reads now and will not read afterwards --
        /// [`Refusal::Taken::losing`].
        losing: &'a str,
    },

    /// The answer was honoured, and this is everything it wrote.
    ///
    /// One note for the whole outcome rather than one per line: the displacements, the surviving
    /// stale references and where the name landed are one block an interface lays out together,
    /// and [`Consequence`] is already the documented shape of them. Restating it here would be
    /// the second mapping that module's doc exists to forbid.
    Committed {
        /// The name that was given, trimmed.
        name: &'a str,
        /// How much the answered voice spoke, which is what the floor and the cap turn on.
        speech_seconds: f64,
        /// What it wrote.
        consequence: &'a Consequence,
    },

    /// How much of the transcript the answer rewrote. Only the `--at` path says this, because
    /// only there did the user not see the voice before naming it.
    Renamed {
        /// The name that was given.
        name: &'a str,
        /// How many turns changed.
        turns: usize,
        /// How much speech those turns cover.
        seconds: f64,
    },
}

/// The narration as the command-line tool prints it: one line per note, to a writer.
///
/// This is where every string lives. Nothing above holds a sentence, and nothing outside this
/// module writes one.
pub struct Lines<'w> {
    out: &'w mut dyn Write,
}

impl<'w> Lines<'w> {
    /// Narrate to `out`. `std::io::sink()` is a silent run; `Vec<u8>` is what the tests assert
    /// against.
    pub fn new(out: &'w mut dyn Write) -> Lines<'w> {
        Lines { out }
    }
}

impl Narrator for Lines<'_> {
    fn note(&mut self, note: Note<'_>) -> Result<()> {
        match note {
            Note::Run(note) => self.run(note),
            Note::Session { session, note } => self.session(session, note),
            Note::Answer { session, note } => self.answer(session, note),
        }
    }
}

impl Lines<'_> {
    fn run(&mut self, note: RunNote<'_>) -> Result<()> {
        let out = &mut self.out;
        match note {
            RunNote::SelectionNeedsOneSession { selection } => writeln!(
                out,
                "{} needs exactly one session id: {}",
                selection.flag(),
                selection.why_one_session()
            )?,
            RunNote::NameNeedsAVoice => writeln!(
                out,
                "--name needs a voice to put that name on: pass --at <MM:SS> or --voice <VOICE>, \
                 since a name given up front is never shown the voice it is answering about"
            )?,
            RunNote::SessionNotFound { id } => writeln!(out, "{id}  not found")?,
            RunNote::NoSessionsFound { dir } => {
                writeln!(out, "No sessions found in {}", dir.display())?
            }
        }
        Ok(())
    }

    fn session(&mut self, session: &SessionId, note: SessionNote<'_>) -> Result<()> {
        let out = &mut self.out;
        match note {
            SessionNote::PassedOver(PassedOver::Orphaned) => writeln!(
                out,
                "{session}  passed over: no session.json (the recorder crashed mid-session)"
            )?,
            SessionNote::PassedOver(PassedOver::NotTranscribed) => {
                writeln!(out, "{session}  passed over: not transcribed yet")?
            }
            // A session whose voices are all identified is exactly where somebody stands when
            // one of those identifications is wrong, and this line is the only thing it prints
            // -- so it names the escape, the way the held-back line already names `--all`.
            SessionNote::PassedOver(PassedOver::NothingUnresolved { named: 0 }) => {
                writeln!(out, "{session}  passed over: nothing unresolved")?
            }
            SessionNote::PassedOver(PassedOver::NothingUnresolved { named }) => writeln!(
                out,
                "{session}  passed over: nothing unresolved ({named} named voice(s) -- \
                 meethook enroll --correct)"
            )?,
            SessionNote::Unreadable { file, error } => {
                writeln!(out, "{session}  failed: {error} -- {}", file.remedy())?
            }
            SessionNote::BroughtUpToDate => {
                writeln!(out, "{session}  transcript brought up to date")?
            }
            SessionNote::Queue {
                offered,
                already_named,
                held_back,
            } => {
                // "Unresolved" is false under `--correct`, where most of the queue is resolved
                // and the point is to review it. The default wording is left exactly as it was.
                let counted = match already_named {
                    Some(already) => {
                        format!("{offered} voice(s) to review, {already} of them already named")
                    }
                    None => format!("{offered} unresolved voice(s)"),
                };
                if held_back == 0 {
                    writeln!(out, "{session}  {counted}")?;
                } else {
                    // Naming the escape rather than only the count: a voice nobody is told
                    // about is not reachable.
                    writeln!(
                        out,
                        "{session}  {counted}, {held_back} quieter voice(s) not offered -- \
                         meethook enroll --all"
                    )?;
                }
            }
            // "Left as they were" rather than "unanswered", because under `--correct` some of
            // them are kept identifications -- and that is already the wording the summary uses
            // for those.
            SessionNote::Left { left } => writeln!(
                out,
                "{session}  left early, {left} voice(s) left as they were"
            )?,
            // The literal 1 is this run's whole queue, so the one prompt below reads `1/1`. On
            // the timestamp path the moment comes too: the user named a time and gets told
            // which voice that turned out to be, which is the one thing they did not know.
            SessionNote::Selected { at: None, voice } => {
                writeln!(out, "{session}  1 voice selected: {voice}")?
            }
            SessionNote::Selected {
                at: Some(at),
                voice,
            } => writeln!(out, "{session}  1 voice selected at {at}: {voice}")?,
            SessionNote::NotSelected(note) => self.not_selected(session, note)?,
        }
        Ok(())
    }

    fn not_selected(&mut self, session: &SessionId, note: NotSelected<'_>) -> Result<()> {
        let out = &mut self.out;
        match note {
            // The voices are listed rather than merely counted, quiet ones included, because a
            // miss is usually a number off by one or a name spelled as the user remembers it
            // rather than as the transcript has it -- and the quiet voices are exactly what
            // somebody is reaching for when they miss.
            NotSelected::NoVoiceMatched { selector, voices } => {
                writeln!(
                    out,
                    "{session}  no voice matched {selector} -- this session has {}:",
                    voices.len()
                )?;
                for voice in &voices {
                    writeln!(out, "    {voice}")?;
                }
            }
            // Two voices under one enrolled name is the false accept `--correct` exists to fix,
            // so the message has to hand back the thing that tells them apart, which is the
            // number. Quoted as a whole label rather than as a bare digit so it can be pasted
            // straight back: both forms are accepted, and only one of them survives being read
            // off a line that also contains a name.
            NotSelected::SeveralVoicesMatched { selector, voices } => writeln!(
                out,
                "{session}  {selector} matches {} voices: {} -- pass one of {}",
                voices.len(),
                joined(&voices),
                numbers(&voices)
            )?,
            NotSelected::SeveralVoicesAt { at, count, voices } => writeln!(
                out,
                "{session}  {at} is the label of {count} turns, by different voices: {} -- \
                 pass one of {}",
                joined(&voices),
                numbers(&voices)
            )?,
            NotSelected::OnTheMicrophone { at } => writeln!(
                out,
                "{session}  {at} is on the microphone track: that is you, and enroll names the \
                 voices it heard rather than the person holding the machine"
            )?,
            NotSelected::NoClusters { at } => writeln!(
                out,
                "{session}  the turn at {at} records no voice: diarization found no speakers in \
                 this session, so its turns have nothing to hang a name on -- re-transcribe \
                 this session with --force"
            )?,
            // A miss here is usually a second or two off, and the user is holding the file with
            // the right timestamp in it, so the nearest turn is worth more than the refusal.
            NotSelected::Silence {
                at,
                nearest: Some(nearest),
            } => writeln!(
                out,
                "{session}  nobody was speaking at {at}: the nearest turn is {} at {}",
                nearest.speaker, nearest.at
            )?,
            NotSelected::Silence { at, nearest: None } => {
                writeln!(out, "{session}  nobody was speaking at {at}")?
            }
            NotSelected::PastEnd { at, last } => writeln!(
                out,
                "{session}  {at} is past the end of this session, which ends at {last}"
            )?,
            // A voice the transcript names and the clusters file does not is the stale-file
            // failure the rest of this crate already has wording for, reached from the other
            // side.
            NotSelected::VoiceNotInClusters { at } => writeln!(
                out,
                "{session}  failed: the turn at {at} came from a voice speaker_clusters.json \
                 does not have -- re-transcribe this session with --force"
            )?,
        }
        Ok(())
    }

    fn answer(&mut self, session: &SessionId, note: AnswerNote<'_>) -> Result<()> {
        match note {
            AnswerNote::Refused {
                name,
                voice,
                refusal,
            } => self.refused(session, name, voice, refusal),
            AnswerNote::Overrode {
                name,
                answered,
                voice,
                losing,
            } => self.overrode(session, name, answered, voice, losing),
            AnswerNote::Committed {
                name,
                speech_seconds,
                consequence,
            } => self.committed(session, name, speech_seconds, consequence),
            AnswerNote::Renamed {
                name,
                turns,
                seconds,
            } => self.renamed(session, name, turns, seconds),
        }
    }

    /// The user gets a line naming the voice that would have paid.
    fn refused(
        &mut self,
        session: &SessionId,
        name: &str,
        answered: &str,
        refusal: &Refusal,
    ) -> Result<()> {
        let out = &mut self.out;
        match refusal {
            Refusal::Vetoed {
                holder: Some(holder),
            } => writeln!(
                out,
                "{session}  refused {name} for {answered}: {holder} already has that name and \
                 the two were heard speaking at once, so they are not one person -- \
                 meethook enroll --correct --voice {holder} if that is the wrong one"
            )?,
            Refusal::Vetoed { holder: None } => writeln!(
                out,
                "{session}  refused {name} for {answered}: that name will not apply to this voice"
            )?,
            Refusal::Taken { voice, losing } => writeln!(
                out,
                "{session}  refused {name} for {answered}: it would take {losing} off {voice} -- \
                 meethook enroll --correct --voice {voice} if {voice} is not {losing}"
            )?,
        }
        Ok(())
    }

    /// The same cost the refusal line names, in a run where it was paid rather than declined.
    ///
    /// Printed before the [`AnswerNote::Committed`] block, which is the order
    /// [`committed`](Self::committed) already prints in: what the answer took off other people
    /// first, then where the name landed. It names the voice that paid, so a user reading the
    /// scrollback afterwards -- rather than the pane that warned them -- can still see who lost
    /// a name and reach for the command that gives it back.
    fn overrode(
        &mut self,
        session: &SessionId,
        name: &str,
        answered: &str,
        voice: &str,
        losing: &str,
    ) -> Result<()> {
        writeln!(
            self.out,
            "{session}  named {name} for {answered} anyway: {voice} no longer reads {losing} -- \
             meethook enroll --correct --voice {voice} to give it a name again"
        )?;
        Ok(())
    }

    /// Everything one honoured answer wrote, in the order it wrote it: what it took off other
    /// people first, then where the name itself landed.
    fn committed(
        &mut self,
        session: &SessionId,
        name: &str,
        speech_seconds: f64,
        consequence: &Consequence,
    ) -> Result<()> {
        let out = &mut self.out;

        for who in &consequence.displaced {
            // An enrollment that vanishes without a line about it is worse than the bug. Two
            // wordings, because "Nate no longer has a reference" is a lie when Nate has three
            // recordings and lost one.
            if who.remaining == 0 {
                writeln!(
                    out,
                    "{session}  {} no longer has a reference: that voice is {name}",
                    who.name
                )?;
            } else {
                writeln!(
                    out,
                    "{session}  {} no longer has that reference: that voice is {name} -- {} \
                     keeps {} other(s)",
                    who.name, who.name, who.remaining
                )?;
            }
        }

        match &consequence.stored {
            None => {
                // The case being given up by not touching the database: a legacy reference that
                // *is* this exact fragment (built before the floor existed) stays, and goes on
                // competing as an argmax under somebody else's name. Reported rather than
                // silently left, with the override that fixes it, because an enrollment that is
                // wrong and unmentioned is worse than one that is wrong and named.
                for who in &consequence.stale {
                    writeln!(
                        out,
                        "{session}  {who} still has a reference built from this voice -- \
                         meethook enroll --force-reference to replace it with {name}"
                    )?;
                }
                writeln!(
                    out,
                    "{session}  named {name} in this session only: {speech_seconds:.1} s of \
                     speech is under the {REFERENCE_FLOOR_SECONDS} s reference floor -- \
                     meethook enroll --force-reference to store a reference anyway"
                )?;
            }
            Some(Stored::Enrolled) => writeln!(out, "{session}  enrolled {name}")?,
            Some(Stored::Added { held }) => writeln!(
                out,
                "{session}  enrolled another recording of {name}: {held} reference(s) now"
            )?,
            Some(Stored::AlreadyHeld) => writeln!(
                out,
                "{session}  {name} already has a reference built from this voice"
            )?,
            Some(Stored::Replaced {
                held,
                evicted_seconds,
            }) => writeln!(
                out,
                "{session}  enrolled a better recording of {name}: {speech_seconds:.1} s \
                 replaces the shortest of their {held}, which was {evicted_seconds:.1} s -- that \
                 one may have been naming a voice in another session, and meethook enroll over \
                 it says so"
            )?,
            Some(Stored::AtCapacity { held, shortest }) => {
                // Why it was refused, not just that it was: at the cap the answer turns on this
                // clip's length, and a user who is not told that cannot tell "come back with a
                // longer recording" from "this person is full forever".
                let why = match shortest {
                    Some(shortest) => format!(
                        "{speech_seconds:.1} s of speech is no longer than the shortest of them, at {shortest:.1} s"
                    ),
                    None => "none of them records how long a recording it was built from, so \
                             there is nothing to say this one is better"
                        .to_string(),
                };
                // Both commands, and in this order: the line cannot know which reference should
                // go -- that is the question `speakers` exists to answer -- and naming only
                // `forget` would send the user to pick a number blind.
                writeln!(
                    out,
                    "{session}  named {name} in this session only: {name} already holds {held} \
                     reference(s), the most meethook keeps for one person, and {why}, so this \
                     recording is not stored and does not help recognise them -- meethook \
                     speakers shows what each of them is naming, meethook forget {name} \
                     --reference N removes the one you pick"
                )?;
            }
        }
        Ok(())
    }

    fn renamed(
        &mut self,
        session: &SessionId,
        name: &str,
        turns: usize,
        seconds: f64,
    ) -> Result<()> {
        let out = &mut self.out;
        if turns == 0 {
            // Not "0 turn(s)", which reads as a failure. The ordinary way to get here is
            // answering a voice with the name it already reads as.
            writeln!(
                out,
                "{session}  no turns changed: that voice already read as {name}"
            )?;
        } else {
            writeln!(
                out,
                "{session}  renamed {turns} turn(s), {} of speech, to {name}",
                speech(seconds)
            )?;
        }
        Ok(())
    }
}

impl SessionFile {
    /// What to do about a file that would not read. Only the two files a transcribe regenerates
    /// can be recovered that way; the other two hold something nothing else can reproduce.
    fn remedy(&self) -> &'static str {
        match self {
            // The expected instance of this is a `speaker_clusters.json` from before first
            // appearances were recorded, and a `transcript.json` from before turns recorded
            // which cluster they came from. A user told only "missing field `cluster`" has been
            // given a diagnosis with no next step.
            SessionFile::Clusters | SessionFile::Transcript => {
                "re-transcribe this session with --force"
            }
            // No re-transcribe recovers this: `session.json` is the recorder's own output and
            // the marker that this directory is a session at all.
            SessionFile::Metadata => "fix that file",
            // Nor this: it holds names a person typed and nothing else can regenerate them.
            SessionFile::Names => "fix or delete that file",
        }
    }
}

impl std::fmt::Display for VoiceDescription {
    /// The number it is reachable by, plus the name it currently carries when that is not the
    /// number itself. One place decides the form, so a list of candidates reads the same
    /// however the user missed.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.label == self.number {
            write!(f, "{} ({:.1} s)", self.number, self.speech_seconds)
        } else {
            write!(
                f,
                "{} -- {} ({:.1} s)",
                self.number, self.label, self.speech_seconds
            )
        }
    }
}

/// The voices of an ambiguous match, as one comma-separated clause.
fn joined(voices: &[VoiceDescription]) -> String {
    voices
        .iter()
        .map(VoiceDescription::to_string)
        .collect::<Vec<String>>()
        .join(", ")
}

/// The `--voice` arguments that split an ambiguous match, read off the same descriptions the
/// clause above was, so a message and the handle it hands back cannot disagree.
fn numbers(voices: &[VoiceDescription]) -> String {
    voices
        .iter()
        .map(|voice| format!("--voice \"{}\"", voice.number))
        .collect::<Vec<String>>()
        .join(" or ")
}

/// Narrate something about a session. A free function rather than a provided trait method,
/// which an implementor could override and so change the wording from outside this crate.
pub(crate) fn about(
    notes: &mut dyn Narrator,
    session: &SessionId,
    note: SessionNote<'_>,
) -> Result<()> {
    notes.note(Note::Session { session, note })
}

/// Narrate the consequence of one answer. See [`about`] for why it is a free function.
pub(crate) fn after(
    notes: &mut dyn Narrator,
    session: &SessionId,
    note: AnswerNote<'_>,
) -> Result<()> {
    notes.note(Note::Answer { session, note })
}
