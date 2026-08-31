//! The interactive seam: what an answerer sees and says, and the answers it gives.
//!
//! [`MeetingLabel`] projects a meeting to what a terminal may print; [`Answer`] is what comes
//! back when a voice is asked about; [`Interviewer`] is the one-method seam every path --
//! terminal, script, test -- reaches the run through, and [`GivenName`] is the scripted half of
//! it.

use meethook_session::{Meeting, MeetingFit};

use crate::prompt::Voice;

#[cfg(doc)]
use crate::queue::{Position, Selection};
#[cfg(doc)]
use crate::session::enroll_session;
#[cfg(doc)]
use crate::{EnrollRules, Refusal, run_enroll};

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
    /// This session's speaker track is one person, called `name`: name every voice in it with
    /// that name, including the one this answer was given about.
    ///
    /// The frame-side half of the one-remote-speaker assertion; the headless half is
    /// [`EnrollRules::one_speaker`], and both reach the same mode inside
    /// [`enroll_session`], which is what keeps the writes and the report identical however the
    /// assertion arrived. Answering it switches the session to assertion mode for the rest of
    /// the run over it: the remaining voices are committed through the fixed-order write path
    /// without being asked, each veto the heard-at-once rule would have raised is reported as
    /// overridden rather than refused, and the fact itself is recorded in `session.json`
    /// before the first of those commits, so an interrupt leaves a state that explains every
    /// label on disk and a re-run converges onto it.
    ///
    /// Only an interface that can show the whole session at once has any use for it -- the
    /// assertion is about every voice, and committing it from one question on screen is how a
    /// user who sees ten "Unknown N"s that are all their colleague says so without answering
    /// ten questions. A line prompt has no surface for the cost, and neither does a scripted
    /// answerer, which is why the flag rather than this answer is the way a script reaches the
    /// same mode.
    ///
    /// Trimmed the same way every other name in this file is; a name of nothing but spaces is
    /// treated as the question going unanswered rather than as an entry called "".
    OneSpeaker(String),
    /// These voices are all one person, named together with `name`: the generalisation of
    /// [`OneSpeaker`](Self::OneSpeaker) from the whole track to a user-chosen group of it.
    ///
    /// `members` are the stable "Unknown N" handles the interface shows in its queue pane --
    /// the same keys [`Voice::number`] carries across the seam -- so the answer names voices by
    /// the handle a person reads rather than by a cluster id nobody sees. The commit walks them
    /// in queue order, the order the transcript reads in, and each member goes through exactly
    /// the dry run a single naming would: a member whose naming would take a name off a voice
    /// the user was not answering about is refused for that member only, and the rest still
    /// apply, which is the difference between a group and a single answer where a refusal writes
    /// nothing at all.
    ///
    /// The group carries veto authority at two or more resolved members: a member heard at once
    /// with one already holding the name is named anyway and reported as overridden, the way
    /// the assertion reports its overrides. A one-member group has none -- no forcing at all,
    /// exactly today's plain naming of that member -- which is the threshold the commit enforces
    /// too, so a preview and a write cannot see the group differently.
    ///
    /// An unresolvable handle goes unanswered rather than partially answered: a group that
    /// cannot say who its members are is not an answer, and the caller decides that before
    /// consulting the answerer at all, on the same blank-name precedent a name of nothing but
    /// spaces reaches.
    Group {
        /// Who the group is, trimmed the same way every other name in this file is.
        name: String,

        /// The "Unknown N" handles the user chose, in whatever order the interface listed them.
        members: Vec<String>,
    },
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
