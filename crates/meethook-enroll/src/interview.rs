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
use crate::{EnrollRules, Refusal, run_enroll};

/// The meeting a session was recorded during, as far as a terminal may see it.
///
/// The title, how strongly the session's start supports the match, and the event's own
/// identifier. [`Meeting`] holds more -- organizer, attendees, location, URL, invite body --
/// and none of that may reach a terminal or a log line: attendee names and addresses exist in
/// `session.json` for speaker identification and are deliberately never printed, and an invite
/// body routinely carries a dial-in PIN. Projecting to these fields makes "nothing sensitive
/// crosses" a property of the type rather than a rule every consumer must remember; the
/// identifier is structural (it is already in `session.json`) and is carried for addressing,
/// never rendered.
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
    /// The event's own identifier, carried rather than printed: it is how a surface that
    /// shows the label addresses the meeting again -- the record interface marks the offer a
    /// guess points at and sends a hand pick back by it. Structural rather than sensitive:
    /// it is already written to `session.json`, and nothing here renders it.
    pub event_id: String,
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
            event_id: meeting.event_id.clone(),
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
        /// people -- and overriding it means asserting several voices are one person. That
        /// assertion has its own answer since, [`Answer::Group`]: it commits a user-chosen set
        /// of voices under one name and reports the vetoes it overrides. The authority still
        /// sits with the group rather than this field.
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
    /// `enroll_session`, which is what keeps the writes and the report identical however the
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
    /// These below-floor fragments are all one person, named together with `name`: the answer
    /// to a *bundled* question, where the library rather than the user decided which fragments
    /// travel together.
    ///
    /// The sibling of [`Group`](Self::Group) aimed at the questions the bundling forms, and
    /// different from it in exactly the one way that matters: it carries **no veto authority**.
    /// A staged group is the user's explicit act of saying "these voices are one person", and
    /// two or more of them may override the heard-at-once veto on that claim; a bundle is a
    /// convenience the library proposed, and honouring it must respect the veto per member --
    /// a fragment segmentation heard at once with somebody already holding the name stays
    /// unnamed while the rest of the bundle commits, which is the same per-member refusal the
    /// staged walk reports, without the override.
    ///
    /// Only formed under [`crate::Enrolment::AboveTheFloor`], because only there does a sub-floor
    /// answer store no reference: naming nine fragments as one person writes nine session rows
    /// and nothing into `speakers.json`, so a wrong bundle costs a relabel, not a poisoned
    /// reference. The commit enforces the same gate the preview does.
    ///
    /// `members` are the stable "Unknown N" handles, and an unresolvable handle goes
    /// unanswered rather than partially answered, on [`Group`](Self::Group)'s precedent.
    FragmentGroup {
        /// Who the bundle is, trimmed the same way every other name in this file is.
        name: String,

        /// The "Unknown N" handles the bundle was built from, in queue order.
        members: Vec<String>,
    },
    /// This voice is not `name`: refuse the tentative guess its turns currently read as.
    ///
    /// The complement of [`Named`](Self::Named) for a guessed fragment. Naming adds a claim --
    /// to the database or to this session's rows -- and denial removes one: the row that lands
    /// in `speaker_names.json` suppresses the guess everywhere it would otherwise appear, in
    /// this run's relabel and in every later `transcribe --force`, because both resolve their
    /// denials through the same rule. That is what makes the answer durable rather than a
    /// cosmetic edit to one transcript: the band will keep finding the resemblance on every
    /// re-run, and only a standing row says the user has already decided about it.
    ///
    /// Unlike [`Skip`](Self::Skip) and [`Later`](Self::Later), answering it commits: the
    /// cluster goes into the run's committed set and is not offered again this session, and it
    /// counts as settled for the pass-over gate the queue applies to the rest of the tail.
    /// Refusing a guess is a decision, and a decision the transcript can show -- which is also
    /// why refusing writes something where skipping writes nothing.
    ///
    /// Only an interface that shows the guess on screen has any use for it: a line prompt has
    /// no surface for the cost, and neither do the scripted answerers, which is why none of
    /// them ever returns this -- the seam stays open for it exactly the way it stayed open for
    /// [`Group`](Self::Group) until the frame needed it.
    ///
    /// `name` is the guess being refused -- what the voice reads with its mark stripped --
    /// trimmed the same way every other name in this file is.
    Deny {
        name: String,
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

    /// Whether this answerer wants below-floor fragments asked about as bundles rather than
    /// one question per fragment.
    ///
    /// `false` by default, and `false` for everyone but the full-screen frame today: a line
    /// prompt has no surface for a composite row, and a scripted answerer answers per voice, so
    /// headless runs keep asking one question per fragment and their printed output stays
    /// byte for byte what it was before the bundling existed. The frame answers `true`, which
    /// is also what makes its queue pane able to show the bundles at all -- the field carrying
    /// them across the seam is populated only for an answerer that asks for them.
    ///
    /// A method on this trait beside `needs_one_voice` for the same reason that one is: the
    /// preference belongs to the answerer, and the run is where the answerer and the queue are
    /// both in hand.
    fn accepts_fragment_groups(&self) -> bool {
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
