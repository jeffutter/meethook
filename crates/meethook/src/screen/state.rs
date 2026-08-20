//! The enrolment frame's state machine, with no terminal anywhere in it.
//!
//! Everything the full-screen interface *decides* is here: which voice the cursor is on, which
//! voice the user is steering toward, what has been typed into the filter, which candidate is
//! highlighted, which transcript line is selected, and what this run has already done to each
//! voice. It takes typed [`Event`]s and returns either "still going" or an
//! [`Answer`].
//!
//! There is deliberately no `ratatui` path in this file, no [`std::io::Write`], no clock, no
//! [`Preview`](meethook_enroll::Preview) and no I/O. That is not tidiness: the sibling module
//! `render` and the shell in `super` both need a person in front of them, and this is the part
//! that does not, so this is where the tests are. The absence of a `ratatui` import is what keeps
//! that honest.
//!
//! The cross-session [`Scan`] is the one thing here that came off a disk, and it arrives as data
//! that has already been gathered -- so the no-I/O claim stays literally true: what this file does
//! with a scan is derive [`Who`] from it, which is arithmetic over a value somebody else read.
//!
//! # The invariant the whole design rests on
//!
//! [`Answer::Later`] is only ever returned in order to *reach* another voice. A deferral is safe
//! when the target is ahead of the current voice in this pass, because the answer that follows
//! keeps the pass alive. A deferral toward a target *behind* the current voice cannot be served
//! this pass and needs [`Screen::still_working`] to be true, or the session ends on the fixed
//! point in `enroll_session`'s pass loop.
//!
//! There is deliberately **no** store-a-decision-for-another-voice path. It would be unusable:
//! the frame has no candidate list, no snippets and no consequence for a voice it was not asked
//! about, so there would be nothing for the user to decide *with*. Steering is the whole
//! mechanism, and saying so is what stops a later reader from adding the map.
//!
//! [`Answer::Leave`] is the one answer outside all of that: it ends the session outright rather
//! than deferring, so it neither sets a target nor depends on [`Screen::still_working`], and the
//! fixed point that bounds a steer does not bound it.

use std::collections::{BTreeMap, BTreeSet};

use meethook_enroll::{Answer, Position, Queued, Refusal, Resolution, Scan, Snippet, resolve};
use meethook_session::SessionId;
use meethook_transcribe::{Attribution, Resemblance};

/// One voice as the frame needs it, projected off
/// [`Voice`](meethook_enroll::Voice) by the shell.
///
/// It exists so that the tests below can build a question without a session on disk: `Voice`
/// carries a [`Preview`](meethook_enroll::Preview) whose constructor is crate-private to
/// `meethook-enroll`, and a `Vec` of snippets and resemblances this module has no use for owning.
/// Borrowing throughout, so building one costs nothing per redraw.
pub struct VoiceView<'a> {
    pub session: &'a SessionId,
    pub position: Position,
    /// The "Unknown N" this voice was transcribed with -- the one handle that does not move when
    /// the voice is named, and so the only thing this module's state may be keyed on.
    pub number: &'a str,
    pub speech_seconds: f64,
    pub attribution: &'a Attribution,
    pub queue: &'a [Queued<'a>],
    pub snippets: &'a [Snippet<'a>],
    pub resembles: &'a [Resemblance],
    pub enrolled: &'a [&'a str],
    /// Whether there is audio to play. The clip itself is the shell's business, and a state
    /// machine holding a quarter of a megabyte of samples per redraw would be paying for nothing.
    ///
    /// The samples a [`Snippet`] carries are not the same trade: they are borrowed with the
    /// snippet's text rather than copied, so a slice of them costs the same pointers per
    /// redraw whether or not anything reads the audio.
    pub clip_is_empty: bool,
}

/// What one key did: nothing the caller has to act on, or the answer to this voice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    Waiting,
    Answered(Answer),
}

/// Every key this frame binds, as the thing it means rather than as the key that produced it.
///
/// Mapped from a `KeyEvent` in `super`, which is where the one part of that mapping no test can
/// decide -- the terminal -- lives. The mapping itself is unit-tested there, because a
/// `KeyEvent` is constructible without a terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// Move the queue cursor.
    Up,
    Down,
    /// Work on the voice the queue cursor is on. A no-op when that is the voice being asked
    /// about, and otherwise the only thing that returns [`Answer::Later`].
    Select,
    /// A character typed into the candidate filter.
    Filter(char),
    Backspace,
    ClearFilter,
    /// Move the candidate highlight, which is a separate pair of keys because typing goes to the
    /// filter.
    CandidateUp,
    CandidateDown,
    /// Answer with the highlighted candidate.
    Choose,
    /// Answer with the highlighted candidate even though it takes a name off another voice.
    ///
    /// Live only where that candidate's refusal is [`Refusal::Taken`], which is the only refusal
    /// an answer can override -- [`Answer::Named::anyway`] says why, and why a
    /// [`Refusal::Vetoed`] stays refused however the key is pressed.
    ///
    /// Its own key rather than [`Event::Choose`] doing double duty on a refused row. The two
    /// mean different things -- "that one" and "that one, and I know what it costs" -- and a
    /// single key would make the more consequential of them the one nobody chose, reachable by
    /// the same reflex as the harmless one. It is what makes the cost pane load-bearing: the
    /// frame prints which voice pays and what it loses before this key exists to be pressed.
    Anyway,
    /// Answer with the typed text as somebody new. Its own key, never the fallback for
    /// unrecognised text.
    NewPerson,
    /// Move the selected transcript line, which is also the row the pane scrolls to: one index,
    /// not a cursor and an offset.
    SnippetUp,
    SnippetDown,
    /// Play the clip. Handled by the shell, which is the only thing holding the samples; here for
    /// the sake of one total `match`.
    Play,
    /// Play the selected transcript line -- the footer's "line", and the row `SnippetUp` and
    /// `SnippetDown` move to. Handled by the shell for the same reason [`Event::Play`] is: the
    /// samples are the shell's, and this is here so the `match` over events stays total.
    PlaySnippet,
    Skip,
    /// Leave the rest of this session's voices and open the next meeting.
    ///
    /// The middle of the three scopes -- [`Skip`](Self::Skip) is one voice, this is the session,
    /// [`Quit`](Self::Quit) is the run -- and the one event that ends a session without
    /// deferring anything: [`Answer::Leave`] returns out of the pass loop rather than putting
    /// the voice back, so nothing here waits to be asked again.
    Leave,
    Quit,
}

/// What answering one candidate would cost, as much of it as this module is allowed to see.
///
/// The seam that keeps [`Preview`](meethook_enroll::Preview) out of the state machine.
pub struct Cost {
    /// Why this candidate cannot be chosen, if it cannot. `Some` makes the row unavailable and
    /// unchoosable, and the reason is rendered beside it.
    pub refusal: Option<Refusal>,
    /// What choosing it would write, as the lines to show under the candidate list.
    pub summary: Vec<String>,
}

/// What answering with a name would do -- asked, never written.
///
/// A trait rather than a `Preview` because `Consequence`'s two state fields are crate-visible to
/// `meethook-enroll`: a `Consequence` cannot be constructed from this crate at all, so anything
/// taking one would be untestable here. `Refusal` is fully public, which is what makes [`Cost`]
/// constructible in a test.
pub trait Costs {
    fn of(&self, name: &str) -> Cost;
}

/// What the cross-session scan has to say about the enrolled speakers, or why it has nothing to
/// say yet.
///
/// The seam that keeps the scan's I/O out of this module the way [`Costs`] keeps
/// [`Preview`](meethook_enroll::Preview)'s out: the shell gathers a [`Scan`] on a thread of its
/// own and hands it over as a borrow. Unlike `Costs` this needs no trait, because `Scan` and
/// everything under it is fully public with public fields, so a test can build one.
///
/// [`Context::Reading`] is the first fraction of a second of a run rather than an error, and
/// [`Context::Failed`] is a database that could not be read at all -- which is a pane that says so
/// and a frame that carries on answering, because every answer already given is on disk.
#[derive(Debug, Clone, Copy)]
pub enum Context<'a> {
    Reading,
    Read(&'a Scan),
    Failed(&'a str),
}

/// Who the highlighted candidate turns out to be, across every session the scan could read.
///
/// The answer to "who is Ivan again?": how many recordings of them the database holds, and which
/// voices in which meetings read that name because of those recordings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Who {
    /// Nothing is highlighted, so there is nobody to be.
    Nobody,
    Reading,
    Failed(String),
    /// Enrolled since the scan ran -- named during this very run, and so absent from a snapshot
    /// taken before that answer. Its own case because the one wrong answer here that would look
    /// like a fact is "0 references, naming nothing".
    Unrecorded,
    Known {
        references: usize,
        /// Distinct voices those references are naming, across every session read.
        voices: usize,
        /// Most recent session first.
        sessions: Vec<Named>,
        /// Transcribed sessions the scan has no opinion about, so a reader knows the answer above
        /// is over less than everything on disk.
        unreadable: usize,
    },
}

/// One session a person's references are naming voices in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Named {
    pub session: String,
    /// The "Unknown N" of each voice reading this name here, deduped, in the scan's own order.
    pub voices: Vec<String>,
}

/// Who a candidate is, derived from one scan.
///
/// Matched on the name exactly and never fuzzily: [`Candidate::name`] is documented as the
/// `speakers.json` spelling, which is the same spelling [`Enrolled::name`](meethook_enroll::Enrolled)
/// carries, so anything looser could only turn one person into another. A name the scan does not
/// have is [`Who::Unrecorded`] rather than an empty `Known`, because `Scan::people` holds every
/// enrolled name at scan time even when it names nothing -- so absent means "enrolled since",
/// not "names nothing".
///
/// Only `depends` is read, never `elsewhere`: what a reference *names* is the question, and
/// `elsewhere` is what a removal would *also* move, which is a different one nobody asked here.
///
/// Deliberately not memoised, unlike [`Costs::of`]. This is a linear pass over a few dozen people
/// and a few hundred [`VoiceChange`](meethook_enroll::VoiceChange)s with no I/O behind it, so a
/// memo would buy nothing and would add an invalidation rule to get wrong -- and the scan under it
/// moves after every accepted answer.
///
/// One limitation, inherited from the scan rather than papered over: a voice named by hand in some
/// session's `speaker_names.json` never appears here, because no change to the database can move a
/// hand-given name and so the diff never reports one. `meethook speakers` says exactly as much,
/// which is the point -- diverging from the scan to look more complete would make the two disagree.
fn who(context: Context<'_>, highlighted: Option<&str>) -> Who {
    let Some(name) = highlighted else {
        return Who::Nobody;
    };
    let scan = match context {
        Context::Reading => return Who::Reading,
        Context::Failed(why) => return Who::Failed(why.to_string()),
        Context::Read(scan) => scan,
    };
    let Some(person) = scan.people.iter().find(|person| person.name == name) else {
        return Who::Unrecorded;
    };

    // Keyed by `SessionId`, which orders as `YYYYMMDD-HHMMSS` does, so the reverse of the map's
    // own order is chronological with the most recent meeting first.
    let mut by_session: BTreeMap<&SessionId, Vec<String>> = BTreeMap::new();
    for reference in &person.references {
        for change in &reference.depends {
            let voices = by_session.entry(&change.session).or_default();
            // Two of somebody's recordings can both be naming one voice. It is one voice.
            if !voices.contains(&change.voice) {
                voices.push(change.voice.clone());
            }
        }
    }

    Who::Known {
        references: person.references.len(),
        voices: by_session.values().map(Vec::len).sum(),
        sessions: by_session
            .into_iter()
            .rev()
            .map(|(session, voices)| Named {
                session: session.to_string(),
                voices,
            })
            .collect(),
        unreadable: scan.unreadable.len(),
    }
}

/// What this run did to a voice, which the session's own labels cannot say.
///
/// A voice named a moment ago already arrives with its new name on it, so this is not "what is
/// this voice called" -- it is "did the user deal with this row", which is the thing a queue pane
/// has to mark and nothing on the far side of the seam records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    Answered,
    Skipped,
    /// Put back to reach another voice. Not a decision, and cleared the moment the voice comes
    /// round again.
    Deferred,
}

/// One queue row as a pane lists it.
pub struct Row {
    pub number: String,
    /// What it reads as now. Equal to `number` for a voice nothing has named.
    pub label: String,
    pub speech_seconds: f64,
    /// The similarity behind the label, for a voice the database identified. `None` for a voice
    /// nobody has named and for one named by hand, which has no similarity to report rather than
    /// a low one.
    pub similarity: Option<f32>,
    /// Whether the prompt floor would have held this voice back. Rendered under a separator
    /// rather than summarised as a count, because a user looking for their two-second fragment
    /// needs to see that it exists.
    pub below_floor: bool,
    pub mark: Option<Mark>,
    /// Whether this is the voice the question is about.
    pub current: bool,
}

/// One enrolled person the user could answer with.
pub struct Candidate {
    /// The name **as `speakers.json` spells it**, which is what makes answering with it land on
    /// the person already there rather than beside them.
    pub name: String,
    /// How much this voice sounds like them, and how many recordings they hold. `None` for
    /// somebody reachable by typing but absent from the ranking -- a person all of whose stored
    /// recordings are a stale embedding dimension is exactly that, and is still real.
    pub similarity: Option<f32>,
    pub references: Option<usize>,
    /// Why this one cannot be chosen, if it cannot.
    pub refusal: Option<Refusal>,
}

/// Everything the frame draws, derived rather than stored.
pub struct View<'a> {
    pub session: &'a SessionId,
    pub position: Position,
    pub number: &'a str,
    pub label: &'a str,
    pub speech_seconds: f64,
    pub rows: Vec<Row>,
    /// Index into [`View::rows`] of the queue cursor.
    pub cursor: usize,
    pub filter: &'a str,
    /// What the filter turned the typed text into, so a pane can say "nobody enrolled matches"
    /// rather than leaving an empty list unexplained.
    pub resolution: Resolution,
    pub candidates: Vec<Candidate>,
    /// Index into [`View::candidates`] of the highlight, or `None` when there are none.
    pub candidate: Option<usize>,
    /// What choosing the highlighted candidate would do.
    pub consequence: Vec<String>,
    /// Who the highlighted candidate already is, across the sessions the scan could read. Owned
    /// like [`View::rows`] and [`View::candidates`] are, so the pane borrows nothing from the
    /// snapshot the shell is free to replace between frames.
    pub who: Who,
    /// Every snippet, with the pane scrolled to [`View::snippet`].
    pub snippets: &'a [Snippet<'a>],
    /// Index into [`View::snippets`] of the selected line -- the row the pane marks, the row at
    /// the top of it, and the line [`Event::PlaySnippet`] would play. Always the same index
    /// [`Screen::selected`] returns, because both come off `selected_index`.
    pub snippet: usize,
    pub clip_is_empty: bool,
    /// One line about what just happened -- a clip that would not play, a voice that turned out
    /// not to be reachable. Cleared by the next key.
    pub status: Option<&'a str>,
}

impl View<'_> {
    /// The candidate under the highlight, or `None` where there are none.
    ///
    /// [`View::candidate`] is an index into [`View::candidates`] and every pane that cares about
    /// the highlighted row -- its consequence, who it already is, whether the footer can offer
    /// the override -- wants the candidate rather than the index. One lookup so those panes
    /// cannot come to disagree about which row is highlighted.
    pub fn highlighted(&self) -> Option<&Candidate> {
        self.candidates.get(self.candidate?)
    }
}

/// The frame's whole state, keyed on "Unknown N" throughout.
///
/// Keyed on the number and never on the label, because a label moves the moment its voice is
/// named, and never on the cluster id, which is deliberately unreachable from this side of the
/// seam.
#[derive(Default)]
pub struct Screen {
    /// Which session the state below is about. Load-bearing: a multi-session run reuses one
    /// `Screen` across sessions, and a cursor left on session A's row 7 must not select session
    /// B's row 7.
    session: Option<SessionId>,
    cursor: usize,
    /// The "Unknown N" the user is steering toward, if any. `Some` is exactly the condition
    /// under which a pass producing no answer is not a finished session.
    target: Option<String>,
    /// Which voices have been offered since `target` was set. What bounds the steering: a target
    /// that is not in this pass at all -- a voice `--voice` narrowed away, or one an earlier
    /// answer in this run has already named -- would otherwise be waited for forever.
    awaited: BTreeSet<String>,
    filter: String,
    candidate: usize,
    /// Which transcript line is selected: the row the pane marks and the line the play-the-line
    /// key hands over. Unclamped here and clamped on the way out by
    /// [`Screen::selected_index`], which is the only place that knows how many lines this voice
    /// actually has.
    snippet: usize,
    decided: BTreeMap<String, Mark>,
    /// Memo for [`Costs::of`], keyed by name and **cleared on every arrival**. One `Costs::of` is
    /// a database clone and two full labellings of the session, and the database moves after every
    /// accepted answer, so a memo outliving one question would be stale as well as expensive.
    /// That clearing is the whole reason this is keyed by name rather than by (voice, name).
    costs: BTreeMap<String, Cost>,
    status: Option<String>,
}

impl Screen {
    /// What to do about a voice the run has just offered, before any event loop runs.
    ///
    /// `Some` is the interface deferring the question on its own: the user is steering toward
    /// another voice, so this one goes back in the queue untouched. `None` means the caller
    /// should draw and take keys, with the cursor snapped to this voice's row.
    ///
    /// Resets everything when the session changes, and clears the cost memo every time, for the
    /// reasons those two fields give.
    pub fn arrive(&mut self, view: &VoiceView<'_>) -> Option<Answer> {
        if self.session.as_ref() != Some(view.session) {
            let session = view.session.clone();
            *self = Screen {
                session: Some(session),
                ..Screen::default()
            };
        }
        self.costs.clear();

        match self.target.as_deref() {
            // Reached. Whatever the user was steering toward is now the question, so the steering
            // is over and a pass that produces no answer from here really is a finished session.
            Some(number) if number == view.number => {
                self.target = None;
                self.awaited.clear();
            }
            // Not reached, and the queue has come round to a voice already offered since the
            // target was set -- so the target is not in this pass at all and no number of further
            // passes will produce it. Abandon the steering rather than defer forever; the loop's
            // fixed point is only bounded by `still_working` going false.
            Some(_) if self.awaited.contains(view.number) => {
                let lost = self.target.take().unwrap_or_default();
                self.awaited.clear();
                self.status = Some(format!(
                    "{lost} is not among this session's questions, so the cursor stayed here"
                ));
            }
            Some(_) => {
                self.awaited.insert(view.number.to_string());
                self.decided.insert(view.number.to_string(), Mark::Deferred);
                return Some(Answer::Later);
            }
            None => {}
        }

        // A filter, a candidate highlight and a line selection are all about the voice that was on
        // the screen, so none of them survives the question changing.
        self.filter.clear();
        self.candidate = 0;
        self.snippet = 0;
        self.decided.remove(view.number);
        self.cursor = view
            .queue
            .iter()
            .position(|row| row.number == view.number)
            .unwrap_or(0);
        None
    }

    /// Whether the user is mid-steer, and so whether a pass that produced no answer is a finished
    /// session.
    ///
    /// This is [`Interviewer::still_working`](meethook_enroll::Interviewer::still_working) from
    /// this side. It goes false the moment the target is reached or abandoned, which is what
    /// bounds the loop.
    pub fn still_working(&self) -> bool {
        self.target.is_some()
    }

    /// Something the frame has to say about what just happened, shown until the next key.
    pub fn say(&mut self, message: String) {
        self.status = Some(message);
    }

    /// Takes back whatever was last [`said`](Screen::say), for a key that has now succeeded where
    /// it previously failed.
    ///
    /// The counterpart to `say` because the status line is otherwise cleared only by
    /// [`answer`](Screen::answer), and the keys handled outside the state machine -- playback --
    /// never reach one. Without this, a failed play leaves its sentence on the footer underneath a
    /// subsequent successful one.
    pub fn hush(&mut self) {
        self.status = None;
    }

    /// Acts on one key, and says whether that answered the question.
    pub fn answer(&mut self, view: &VoiceView<'_>, event: Event, costs: &dyn Costs) -> Step {
        self.status = None;
        match event {
            Event::Up => {
                self.cursor = self.cursor.saturating_sub(1);
            }
            Event::Down => {
                let last = view.queue.len().saturating_sub(1);
                self.cursor = (self.cursor + 1).min(last);
            }
            Event::Select => {
                let Some(row) = view.queue.get(self.cursor) else {
                    return Step::Waiting;
                };
                if row.number == view.number {
                    // Already the question. Nothing to defer, and returning `Later` here would
                    // put the voice back only to be asked about it again.
                    return Step::Waiting;
                }
                self.target = Some(row.number.to_string());
                self.awaited.clear();
                self.awaited.insert(view.number.to_string());
                self.decided.insert(view.number.to_string(), Mark::Deferred);
                return Step::Answered(Answer::Later);
            }
            Event::Filter(c) => {
                self.filter.push(c);
                self.candidate = 0;
            }
            Event::Backspace => {
                self.filter.pop();
                self.candidate = 0;
            }
            Event::ClearFilter => {
                self.filter.clear();
                self.candidate = 0;
            }
            Event::CandidateUp => {
                self.candidate = self.candidate.saturating_sub(1);
            }
            Event::CandidateDown => {
                let last = self.candidates(view).len().saturating_sub(1);
                self.candidate = (self.candidate + 1).min(last);
            }
            Event::Choose => {
                let Some((name, refusal)) = self.chosen(view, costs) else {
                    // Unrecognised text, or nothing enrolled at all. Deliberately nothing: the
                    // only way to create somebody is the key that says so.
                    return Step::Waiting;
                };
                // Unchanged by the override: *every* refusal refuses this key, the overridable
                // one included. Insisting is the other key's job.
                if refusal.is_some() {
                    return Step::Waiting;
                }
                self.decided.insert(view.number.to_string(), Mark::Answered);
                return Step::Answered(Answer::Named {
                    name,
                    anyway: false,
                });
            }
            Event::Anyway => {
                let Some((name, refusal)) = self.chosen(view, costs) else {
                    return Step::Waiting;
                };
                // Nothing on a row this key cannot help with, in either direction: a candidate
                // nothing refuses is answered by Enter and not by insisting, and the
                // heard-at-once veto is refused however insistent the answer is -- the library
                // enforces that second half too, and this mirror is what keeps the frame from
                // offering a key the library would then refuse.
                if !matches!(refusal, Some(Refusal::Taken { .. })) {
                    return Step::Waiting;
                }
                self.decided.insert(view.number.to_string(), Mark::Answered);
                return Step::Answered(Answer::Named { name, anyway: true });
            }
            Event::NewPerson => {
                let typed = self.filter.trim();
                if typed.is_empty() {
                    return Step::Waiting;
                }
                let name = typed.to_string();
                self.decided.insert(view.number.to_string(), Mark::Answered);
                return Step::Answered(Answer::Named {
                    name,
                    anyway: false,
                });
            }
            Event::SnippetUp => {
                self.snippet = self.snippet.saturating_sub(1);
            }
            Event::SnippetDown => {
                // Clamped so that the last snippet can be selected and no further. Clamping is
                // how the out-of-range case is defined out of existence rather than
                // bounds-checked at every use.
                let last = view.snippets.len().saturating_sub(1);
                self.snippet = (self.snippet + 1).min(last);
            }
            // The shell holds the samples, so it intercepts both of these before the state
            // machine sees them. Here so that the `match` is total and adding a key cannot
            // silently do nothing.
            Event::Play | Event::PlaySnippet => {}
            Event::Skip => {
                self.decided.insert(view.number.to_string(), Mark::Skipped);
                return Step::Answered(Answer::Skip);
            }
            // No mark and no touching `target` or `awaited`: this session is never drawn again,
            // and `arrive` resets the whole `Screen` when the session changes. A [`Mark`]
            // nothing can render would be state that exists to be believed and never read.
            //
            // `still_working` is already false here rather than being cleared: whenever a steer
            // is outstanding, `arrive` defers without drawing, so a key is only ever read with
            // no target set.
            Event::Leave => return Step::Answered(Answer::Leave),
            Event::Quit => return Step::Answered(Answer::Quit),
        }
        Step::Waiting
    }

    /// Which line the pane has selected: the state's index, clamped to the lines this voice has.
    ///
    /// One rule rather than two, because [`Screen::selected`] and [`View::snippet`] must not be
    /// able to disagree about which row is the selected one -- a mark on one row and audio from
    /// another is the failure this ticket exists to prevent.
    fn selected_index(&self, view: &VoiceView<'_>) -> usize {
        self.snippet.min(view.snippets.len().saturating_sub(1))
    }

    /// The line [`Event::PlaySnippet`] would play, and where it sits in the pane.
    ///
    /// `None` only for a voice with nothing transcribed. The index comes back with the snippet
    /// because playback has to be *marked* on the row it started from, and the user may page the
    /// pane while it sounds -- so the mark is the index at the spawn, not wherever the selection
    /// has since moved to.
    ///
    /// A [`Snippet`] by value: it is `Copy`, four words of borrowed text and borrowed samples, so
    /// this hands over no audio and copies none. That is the same trade
    /// [`VoiceView::clip_is_empty`] documents from the other side.
    pub fn selected<'a>(&self, view: &VoiceView<'a>) -> Option<(usize, Snippet<'a>)> {
        let index = self.selected_index(view);
        Some((index, *view.snippets.get(index)?))
    }

    /// Everything the panes draw, derived from the state, this voice and whatever the scan has
    /// found so far.
    ///
    /// `&mut self` because of the cost memo: the highlighted candidate's consequence is computed
    /// here, once per distinct name, and not once per keystroke.
    pub fn view<'a>(
        &'a mut self,
        view: &'a VoiceView<'a>,
        costs: &dyn Costs,
        context: Context<'_>,
    ) -> View<'a> {
        let rows = view
            .queue
            .iter()
            .map(|row| Row {
                number: row.number.to_string(),
                label: row.attribution.label().to_string(),
                speech_seconds: row.speech_seconds,
                similarity: match row.attribution {
                    Attribution::Identified { similarity, .. } => Some(*similarity),
                    Attribution::Unknown(_) | Attribution::Assigned { .. } => None,
                },
                below_floor: row.below_floor,
                mark: self.decided.get(row.number).copied(),
                current: row.number == view.number,
            })
            .collect();

        let names = self.candidates(view);
        let similar: BTreeMap<&str, (f32, usize)> = view
            .resembles
            .iter()
            .map(|r| (r.name.as_str(), (r.similarity, r.references)))
            .collect();
        let highlighted = names.get(self.candidate).cloned();
        // One `Costs::of`, and only for the row under the highlight -- which is what AC #9 is
        // about and why the refusals below are read out of the memo rather than asked for per row.
        let consequence = match &highlighted {
            Some(name) => self.cost(name, costs).summary.clone(),
            None => Vec::new(),
        };
        let candidates = names
            .iter()
            .map(|name| Candidate {
                name: name.clone(),
                similarity: similar.get(name.as_str()).map(|(s, _)| *s),
                references: similar.get(name.as_str()).map(|(_, r)| *r),
                refusal: self.costs.get(name).and_then(|cost| cost.refusal.clone()),
            })
            .collect::<Vec<_>>();

        View {
            session: view.session,
            position: view.position,
            number: view.number,
            label: view.attribution.label(),
            speech_seconds: view.speech_seconds,
            rows,
            cursor: self.cursor.min(view.queue.len().saturating_sub(1)),
            filter: &self.filter,
            resolution: resolve(&self.filter, view.enrolled),
            candidate: (!candidates.is_empty()).then_some(self.candidate),
            candidates,
            consequence,
            who: who(context, highlighted.as_deref()),
            snippets: view.snippets,
            snippet: self.selected_index(view),
            clip_is_empty: view.clip_is_empty,
            status: self.status.as_deref(),
        }
    }

    /// The candidate names, in order, for the filter as it stands.
    ///
    /// Blank filter is `resembles` in its own order -- descending similarity, ties by name. A
    /// filter is [`resolve`] against **every enrolled name** and not against `resembles`, for the
    /// reason `resolve` documents: ranking a voice against the database drops a person whose every
    /// stored recording is a stale embedding dimension, and a typo must not duplicate them.
    fn candidates(&self, view: &VoiceView<'_>) -> Vec<String> {
        if self.filter.trim().is_empty() {
            return view.resembles.iter().map(|r| r.name.clone()).collect();
        }
        match resolve(&self.filter, view.enrolled) {
            // Whitespace only, which the guard above has already handled; and nobody enrolled is
            // plausible, which is the new-person row's case and no candidate's.
            Resolution::Blank | Resolution::New(_) => Vec::new(),
            Resolution::Enrolled(name) => vec![name],
            Resolution::Candidates { matches, .. } => matches.into_iter().map(|m| m.name).collect(),
        }
    }

    /// The candidate under the highlight and why it cannot be chosen, or `None` where there is
    /// no candidate at all -- unrecognised text, or nobody enrolled.
    ///
    /// One implementation because two keys answer with this row and they must never disagree
    /// about which row it is: [`Event::Choose`] refuses every refusal and [`Event::Anyway`]
    /// proceeds on one of them, and a second lookup is how those two come to act on different
    /// candidates. The refusal is cloned rather than borrowed so the caller can go on to touch
    /// the rest of the state -- the memo behind [`Screen::cost`] holds it by `&mut self`.
    fn chosen(
        &mut self,
        view: &VoiceView<'_>,
        costs: &dyn Costs,
    ) -> Option<(String, Option<Refusal>)> {
        let name = self.candidates(view).get(self.candidate).cloned()?;
        let refusal = self.cost(&name, costs).refusal.clone();
        Some((name, refusal))
    }

    /// What one candidate costs, computed at most once per name per question.
    fn cost(&mut self, name: &str, costs: &dyn Costs) -> &Cost {
        if !self.costs.contains_key(name) {
            let cost = costs.of(name);
            self.costs.insert(name.to_string(), cost);
        }
        &self.costs[name]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::Cell;

    use meethook_enroll::{
        Answer, Enrolled, Position, Queued, Reference, Scan, Snippet, Unreadable, VoiceChange,
    };
    use meethook_session::SessionId;
    use meethook_transcribe::{Attribution, Resemblance};

    use super::{Context, Cost, Costs, Event, Mark, Named, Screen, Step, VoiceView, Who};

    /// Nothing costs anything, which is what every test that is not about the memo wants.
    struct Free;

    impl Costs for Free {
        fn of(&self, _name: &str) -> Cost {
            Cost {
                refusal: None,
                summary: Vec::new(),
            }
        }
    }

    /// One named candidate is refused, everything else is free.
    struct Vetoes(&'static str);

    impl Costs for Vetoes {
        fn of(&self, name: &str) -> Cost {
            Cost {
                refusal: (name == self.0).then(|| meethook_enroll::Refusal::Vetoed {
                    holder: Some("Unknown 2".to_string()),
                }),
                summary: vec![format!("would name this voice {name}")],
            }
        }
    }

    /// One named candidate is refused for taking a name off another voice, which is the only
    /// refusal an answer can override. [`Vetoes`]'s counterpart, so the two keys can be pressed
    /// against both refusals and told apart.
    struct Takes(&'static str);

    impl Costs for Takes {
        fn of(&self, name: &str) -> Cost {
            Cost {
                refusal: (name == self.0).then(|| meethook_enroll::Refusal::Taken {
                    voice: "Unknown 2".to_string(),
                    losing: "Bob".to_string(),
                }),
                summary: vec![format!("would name this voice {name}")],
            }
        }
    }

    /// Counts calls, which is the only way AC #9 is assertable: "once per highlighted candidate"
    /// is a claim about how many times the expensive thing ran.
    struct Counted(Cell<usize>);

    impl Costs for Counted {
        fn of(&self, name: &str) -> Cost {
            self.0.set(self.0.get() + 1);
            Cost {
                refusal: None,
                summary: vec![format!("would name this voice {name}")],
            }
        }
    }

    fn session() -> SessionId {
        SessionId::parse("20260819-100000").expect("a well-formed session id")
    }

    fn other_session() -> SessionId {
        SessionId::parse("20260819-110000").expect("a well-formed session id")
    }

    /// The queue rows, as `(number, speech seconds, below the floor)`.
    fn rows(spec: &[(&str, f64, bool)]) -> Vec<(String, Attribution, f64, bool)> {
        spec.iter()
            .map(|(number, seconds, below)| {
                (
                    (*number).to_string(),
                    Attribution::Unknown((*number).to_string()),
                    *seconds,
                    *below,
                )
            })
            .collect()
    }

    fn queue(owned: &[(String, Attribution, f64, bool)]) -> Vec<Queued<'_>> {
        owned
            .iter()
            .map(|(number, attribution, seconds, below)| Queued {
                number,
                attribution,
                speech_seconds: *seconds,
                below_floor: *below,
            })
            .collect()
    }

    /// One voice a reference is naming: `voice` in `session` reads `name` today and would go back
    /// to its number without this reference, which is the shape `scan` produces.
    pub fn names(session: &str, voice: &str, name: &str) -> VoiceChange {
        VoiceChange {
            session: SessionId::parse(session).expect("a well-formed session id"),
            voice: voice.to_string(),
            reads: name.to_string(),
            would_read: voice.to_string(),
        }
    }

    /// One enrolled person, described by what each of their references is naming. Handles are
    /// 1-based and in file order, as the scan numbers them.
    pub fn holding(name: &str, references: &[&[VoiceChange]]) -> Enrolled {
        Enrolled {
            name: name.to_string(),
            references: references
                .iter()
                .enumerate()
                .map(|(index, depends)| Reference {
                    handle: index + 1,
                    clip_seconds: None,
                    depends: depends.to_vec(),
                    elsewhere: Vec::new(),
                })
                .collect(),
        }
    }

    /// A scan as `meethook speakers` would have produced it, over three transcribed sessions of
    /// which `unreadable` had no opinion in them.
    pub fn scanned(people: Vec<Enrolled>, unreadable: &[&str]) -> Scan {
        Scan {
            people,
            sessions_found: 3,
            sessions_transcribed: 3,
            sessions_read: 3 - unreadable.len(),
            unreadable: unreadable
                .iter()
                .map(|session| Unreadable {
                    session: SessionId::parse(session).expect("a well-formed session id"),
                    why: "the clusters file is stale -- re-transcribe this session with --force"
                        .to_string(),
                })
                .collect(),
        }
    }

    fn resembles(spec: &[(&str, f32, usize)]) -> Vec<Resemblance> {
        spec.iter()
            .map(|(name, similarity, references)| Resemblance {
                name: (*name).to_string(),
                similarity: *similarity,
                references: *references,
            })
            .collect()
    }

    /// A snippet of a fixture line. The times and the audio are a real prompt's business, not
    /// this module's -- nothing in the state machine reads either -- so they are zeroed and
    /// only the text carries.
    pub(crate) fn snippet(text: &str) -> Snippet<'_> {
        Snippet {
            text,
            start: 0.0,
            duration: 0.0,
            audio: &[],
        }
    }

    /// A snippet as a real voice carries one: said at a moment, with samples of its own. What
    /// [`snippet`] leaves out, for the tests that are about which line would be played.
    pub(crate) fn heard<'a>(text: &'a str, start: f64, audio: &'a [f32]) -> Snippet<'a> {
        Snippet {
            text,
            start,
            duration: 1.0,
            audio,
        }
    }

    /// A question about `number`, with everything else defaulted by the caller's slices.
    #[allow(clippy::too_many_arguments)]
    fn view<'a>(
        session: &'a SessionId,
        number: &'a str,
        nth: usize,
        queue: &'a [Queued<'a>],
        snippets: &'a [Snippet<'a>],
        resembles: &'a [Resemblance],
        enrolled: &'a [&'a str],
        attribution: &'a Attribution,
    ) -> VoiceView<'a> {
        VoiceView {
            session,
            position: Position {
                nth,
                of: queue.len(),
            },
            number,
            speech_seconds: 42.0,
            attribution,
            queue,
            snippets,
            resembles,
            enrolled,
            clip_is_empty: false,
        }
    }

    /// AC #1 and AC #2: every voice the session has is a row, with its talk time, what it reads
    /// as, and whether the floor would have held it back -- so a below-floor voice and an
    /// already-named one are both visible rather than merely reachable.
    #[test]
    fn the_queue_lists_every_voice_with_its_talk_time_and_label() {
        let session = session();
        let owned = vec![
            (
                "Unknown 1".to_string(),
                Attribution::Identified {
                    name: "Milo".to_string(),
                    similarity: 0.81,
                },
                240.0,
                false,
            ),
            (
                "Unknown 2".to_string(),
                Attribution::Unknown("Unknown 2".to_string()),
                90.0,
                false,
            ),
            (
                "Unknown 3".to_string(),
                Attribution::Unknown("Unknown 3".to_string()),
                1.5,
                true,
            ),
        ];
        let queue = queue(&owned);
        let attribution = &owned[1].1;
        let voice = view(&session, "Unknown 2", 2, &queue, &[], &[], &[], attribution);
        let mut screen = Screen::default();
        assert_eq!(screen.arrive(&voice), None);
        screen.answer(&voice, Event::Skip, &Free);

        let derived = screen.view(&voice, &Free, Context::Reading);
        assert_eq!(derived.rows.len(), 3);
        assert_eq!(derived.rows[0].label, "Milo");
        assert_eq!(derived.rows[0].similarity, Some(0.81));
        assert_eq!(derived.rows[0].speech_seconds, 240.0);
        assert!(!derived.rows[0].below_floor);
        assert_eq!(derived.rows[1].label, "Unknown 2");
        assert_eq!(derived.rows[1].similarity, None);
        assert_eq!(derived.rows[1].mark, Some(Mark::Skipped));
        assert!(derived.rows[1].current);
        assert!(derived.rows[2].below_floor);
        assert_eq!(derived.cursor, 1);
    }

    /// AC #3, forwards: selecting a row ahead defers, the voices in between defer without the
    /// event loop running at all, and the target arrives with the cursor on it.
    #[test]
    fn selecting_a_voice_ahead_steers_the_queue_to_it() {
        let session = session();
        let owned = rows(&[
            ("Unknown 1", 60.0, false),
            ("Unknown 2", 60.0, false),
            ("Unknown 3", 60.0, false),
        ]);
        let queue = queue(&owned);
        let mut screen = Screen::default();

        let first = view(&session, "Unknown 1", 1, &queue, &[], &[], &[], &owned[0].1);
        assert_eq!(screen.arrive(&first), None);
        screen.answer(&first, Event::Down, &Free);
        screen.answer(&first, Event::Down, &Free);
        assert_eq!(
            screen.answer(&first, Event::Select, &Free),
            Step::Answered(Answer::Later)
        );
        assert!(screen.still_working());

        let second = view(&session, "Unknown 2", 2, &queue, &[], &[], &[], &owned[1].1);
        assert_eq!(screen.arrive(&second), Some(Answer::Later));

        let third = view(&session, "Unknown 3", 3, &queue, &[], &[], &[], &owned[2].1);
        assert_eq!(screen.arrive(&third), None);
        assert!(!screen.still_working());
        assert_eq!(screen.view(&third, &Free, Context::Reading).cursor, 2);
    }

    /// AC #3, backwards -- the case the sub-ticket exists for, asserted from this side. The pass
    /// that produced no answer must not end the session while the user is still steering.
    #[test]
    fn selecting_a_voice_behind_keeps_the_session_open_until_it_arrives() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false), ("Unknown 2", 60.0, false)]);
        let queue = queue(&owned);
        let mut screen = Screen::default();

        let second = view(&session, "Unknown 2", 2, &queue, &[], &[], &[], &owned[1].1);
        assert_eq!(screen.arrive(&second), None);
        screen.answer(&second, Event::Up, &Free);
        assert_eq!(
            screen.answer(&second, Event::Select, &Free),
            Step::Answered(Answer::Later)
        );
        assert!(screen.still_working());

        // The next pass re-offers both, in order. The first one is the target.
        let first = view(&session, "Unknown 1", 1, &queue, &[], &[], &[], &owned[0].1);
        assert_eq!(screen.arrive(&first), None);
        assert!(!screen.still_working());
    }

    /// The bound on the steering. A target that is not in the pass at all -- narrowed away by
    /// `--voice`, or already named by an earlier answer in this run -- is abandoned after exactly
    /// one lap rather than deferred forever, and `still_working` goes false so the loop can end.
    #[test]
    fn a_target_the_queue_never_offers_is_abandoned_after_one_lap() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false), ("Unknown 2", 60.0, false)]);
        let queue = queue(&owned);
        let mut screen = Screen::default();

        let first = view(&session, "Unknown 1", 1, &queue, &[], &[], &[], &owned[0].1);
        assert_eq!(screen.arrive(&first), None);
        screen.answer(&first, Event::Down, &Free);
        assert_eq!(
            screen.answer(&first, Event::Select, &Free),
            Step::Answered(Answer::Later)
        );

        // Voice 2 is never offered again -- an earlier answer named it. Voice 1 comes round.
        assert_eq!(screen.arrive(&first), None);
        assert!(!screen.still_working());
        assert!(
            screen
                .view(&first, &Free, Context::Reading)
                .status
                .is_some()
        );
    }

    /// The target is cleared both ways: by arriving at it, and by the user naming a different one
    /// once it has arrived.
    #[test]
    fn picking_a_different_target_replaces_the_first() {
        let session = session();
        let owned = rows(&[
            ("Unknown 1", 60.0, false),
            ("Unknown 2", 60.0, false),
            ("Unknown 3", 60.0, false),
        ]);
        let queue = queue(&owned);
        let mut screen = Screen::default();

        // Asked about voice 1, steer to voice 3. Voice 2 defers on the way.
        let first = view(&session, "Unknown 1", 1, &queue, &[], &[], &[], &owned[0].1);
        assert_eq!(screen.arrive(&first), None);
        screen.answer(&first, Event::Down, &Free);
        screen.answer(&first, Event::Down, &Free);
        screen.answer(&first, Event::Select, &Free);
        let second = view(&session, "Unknown 2", 2, &queue, &[], &[], &[], &owned[1].1);
        assert_eq!(screen.arrive(&second), Some(Answer::Later));

        // Voice 3 arrives, which clears the target; from there the user steers back to voice 2.
        let third = view(&session, "Unknown 3", 3, &queue, &[], &[], &[], &owned[2].1);
        assert_eq!(screen.arrive(&third), None);
        assert!(!screen.still_working());
        screen.answer(&third, Event::Up, &Free);
        assert_eq!(
            screen.answer(&third, Event::Select, &Free),
            Step::Answered(Answer::Later)
        );
        assert!(screen.still_working());

        // The next pass re-offers 1 and 2. Voice 1 is not the target, so it defers; voice 2 is.
        assert_eq!(screen.arrive(&first), Some(Answer::Later));
        assert_eq!(screen.arrive(&second), None);
        assert!(!screen.still_working());
    }

    /// AC #4: a blank filter offers `resembles` in its own order, each row carrying the
    /// similarity and the reference count the ranking gave it.
    #[test]
    fn a_blank_filter_offers_the_ranking_with_its_numbers() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 3), ("Ivan", 0.38, 1)]);
        let voice = view(
            &session,
            "Unknown 1",
            1,
            &queue,
            &[],
            &similar,
            &["Milo", "Ivan"],
            &owned[0].1,
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        let derived = screen.view(&voice, &Free, Context::Reading);
        let names: Vec<&str> = derived.candidates.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["Milo", "Ivan"]);
        assert_eq!(derived.candidates[0].similarity, Some(0.71));
        assert_eq!(derived.candidates[0].references, Some(3));
        assert_eq!(derived.candidate, Some(0));
    }

    /// AC #5: the filter resolves against every enrolled name, so a person the ranking dropped --
    /// every stored recording of them a stale embedding dimension -- is still reachable by typing.
    /// A name in neither list is not.
    #[test]
    fn typing_reaches_a_name_the_ranking_does_not_have() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 3)]);
        let enrolled = ["Milo", "Maya"];
        let voice = view(
            &session,
            "Unknown 1",
            1,
            &queue,
            &[],
            &similar,
            &enrolled,
            &owned[0].1,
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        for c in "Maya".chars() {
            screen.answer(&voice, Event::Filter(c), &Free);
        }
        let derived = screen.view(&voice, &Free, Context::Reading);
        let names: Vec<&str> = derived.candidates.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["Maya"]);
        // Reachable, and honest about having no similarity to report rather than inventing one.
        assert_eq!(derived.candidates[0].similarity, None);

        screen.answer(&voice, Event::ClearFilter, &Free);
        for c in "Quentin".chars() {
            screen.answer(&voice, Event::Filter(c), &Free);
        }
        assert!(
            screen
                .view(&voice, &Free, Context::Reading)
                .candidates
                .is_empty()
        );
    }

    /// AC #6: the answer carries the candidate's own spelling, which is what makes it write the
    /// same files the line prompt would have written for the same person.
    #[test]
    fn choosing_answers_with_the_enrolled_spelling_not_the_typed_text() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Marco", 0.6, 2)]);
        let enrolled = ["Marco"];
        let voice = view(
            &session,
            "Unknown 1",
            1,
            &queue,
            &[],
            &similar,
            &enrolled,
            &owned[0].1,
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        // A typo the near-miss tier reaches, in the user's own case: what lands on disk must be
        // the enrolled spelling and not this.
        for c in "marclo".chars() {
            screen.answer(&voice, Event::Filter(c), &Free);
        }
        assert_eq!(
            screen.answer(&voice, Event::Choose, &Free),
            Step::Answered(Answer::Named {
                name: "Marco".to_string(),
                anyway: false,
            })
        );
    }

    /// AC #7: a candidate the veto would refuse is unavailable, and choosing it does nothing.
    #[test]
    fn a_refused_candidate_cannot_be_chosen() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 3)]);
        let enrolled = ["Milo"];
        let voice = view(
            &session,
            "Unknown 1",
            1,
            &queue,
            &[],
            &similar,
            &enrolled,
            &owned[0].1,
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);
        let vetoes = Vetoes("Milo");

        assert_eq!(screen.answer(&voice, Event::Choose, &vetoes), Step::Waiting);
        let derived = screen.view(&voice, &vetoes, Context::Reading);
        assert_eq!(
            derived.candidates[0].refusal,
            Some(meethook_enroll::Refusal::Vetoed {
                holder: Some("Unknown 2".to_string())
            })
        );
    }

    /// A candidate refused for taking a name off another voice can still be answered with -- by
    /// its own key, never by the one that chooses an unrefused candidate. The frame has already
    /// printed which voice pays and what it loses in the pane beside this row, which is what the
    /// second key means and Enter does not.
    #[test]
    fn a_taken_candidate_can_be_chosen_anyway_by_its_own_key() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 3)]);
        let enrolled = ["Milo"];
        let voice = view(
            &session,
            "Unknown 1",
            1,
            &queue,
            &[],
            &similar,
            &enrolled,
            &owned[0].1,
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);
        let takes = Takes("Milo");

        assert_eq!(
            screen.answer(&voice, Event::Choose, &takes),
            Step::Waiting,
            "the ordinary key must go on refusing, or the two keys mean the same thing"
        );
        assert_eq!(
            screen.answer(&voice, Event::Anyway, &takes),
            Step::Answered(Answer::Named {
                name: "Milo".to_string(),
                anyway: true,
            })
        );
        assert_eq!(
            screen.view(&voice, &takes, Context::Reading).rows[0].mark,
            Some(Mark::Answered),
            "an answered voice is marked answered however it was answered"
        );
    }

    /// The heard-at-once veto is a different claim -- segmentation proved two voices are
    /// different people -- and this key is not the way to assert otherwise. Refused here as well
    /// as in the library, so the frame never offers a key the library would then refuse.
    #[test]
    fn the_heard_at_once_veto_is_refused_by_the_override_key_too() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 3)]);
        let enrolled = ["Milo"];
        let voice = view(
            &session,
            "Unknown 1",
            1,
            &queue,
            &[],
            &similar,
            &enrolled,
            &owned[0].1,
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);
        let vetoes = Vetoes("Milo");

        for key in [Event::Choose, Event::Anyway] {
            assert_eq!(
                screen.answer(&voice, key, &vetoes),
                Step::Waiting,
                "{key:?} must not defeat the heard-at-once veto"
            );
        }
    }

    /// And nothing on a row nothing refuses: the two keys are not interchangeable in either
    /// direction, which is what makes the override a decision rather than a second Enter.
    #[test]
    fn the_override_key_does_nothing_where_nothing_is_refused() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 3)]);
        let enrolled = ["Milo"];
        let voice = view(
            &session,
            "Unknown 1",
            1,
            &queue,
            &[],
            &similar,
            &enrolled,
            &owned[0].1,
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert_eq!(screen.answer(&voice, Event::Anyway, &Free), Step::Waiting);
        assert_eq!(
            screen.view(&voice, &Free, Context::Reading).rows[0].mark,
            None,
            "a key that did nothing may not mark the voice as answered"
        );
    }

    /// AC #8: unrecognised text with Choose pressed does nothing at all, and creating somebody is
    /// its own key. Trimmed on the way out, matching how a name given up front is normalised.
    #[test]
    fn creating_a_person_is_its_own_key_and_never_a_fallback() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let enrolled = ["Milo"];
        let voice = view(
            &session,
            "Unknown 1",
            1,
            &queue,
            &[],
            &[],
            &enrolled,
            &owned[0].1,
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        for c in " Maya ".chars() {
            screen.answer(&voice, Event::Filter(c), &Free);
        }
        assert_eq!(screen.answer(&voice, Event::Choose, &Free), Step::Waiting);
        assert_eq!(
            screen.answer(&voice, Event::NewPerson, &Free),
            Step::Answered(Answer::Named {
                name: "Maya".to_string(),
                anyway: false,
            })
        );

        // And a name of nothing but spaces is not a person either.
        let mut screen = Screen::default();
        screen.arrive(&voice);
        screen.answer(&voice, Event::Filter(' '), &Free);
        assert_eq!(
            screen.answer(&voice, Event::NewPerson, &Free),
            Step::Waiting
        );
    }

    /// AC #9: the consequence is computed once per *highlighted candidate*, not once per
    /// keystroke -- and the memo does not survive the question, because the database moves under
    /// it after every accepted answer.
    #[test]
    fn the_consequence_costs_one_call_per_highlighted_candidate() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Marco", 0.6, 2)]);
        let enrolled = ["Marco"];
        let voice = view(
            &session,
            "Unknown 1",
            1,
            &queue,
            &[],
            &similar,
            &enrolled,
            &owned[0].1,
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);
        let counted = Counted(Cell::new(0));

        // Six keystrokes, each followed by the redraw a real loop would do, all of them
        // highlighting the one candidate the filter can mean -- the first four by prefix and the
        // last two by near miss, so the highlight never moves off Marco.
        for c in "Marclo".chars() {
            screen.answer(&voice, Event::Filter(c), &counted);
            let derived = screen.view(&voice, &counted, Context::Reading);
            assert_eq!(derived.consequence, ["would name this voice Marco"]);
        }
        assert_eq!(counted.0.get(), 1, "one distinct highlight, one call");

        // The next question starts from nothing: a preview taken against the old database would
        // be stale as well as expensive.
        screen.arrive(&voice);
        screen.view(&voice, &counted, Context::Reading);
        assert_eq!(counted.0.get(), 2);
    }

    /// AC #10: both exits are available, and each is the answer `enroll_session` already honours.
    #[test]
    fn skipping_and_quitting_are_both_available() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let voice = view(&session, "Unknown 1", 1, &queue, &[], &[], &[], &owned[0].1);
        let mut screen = Screen::default();
        screen.arrive(&voice);
        assert_eq!(
            screen.answer(&voice, Event::Skip, &Free),
            Step::Answered(Answer::Skip)
        );
        assert_eq!(
            screen.answer(&voice, Event::Quit, &Free),
            Step::Answered(Answer::Quit)
        );
    }

    /// TASK-049 acceptance criterion #5, from the frame's side: leaving a session is an answer
    /// given while a voice is on the screen, so no steer is outstanding when the key is read and
    /// none is left behind by it -- the frame cannot make the exit look like a stalled pass.
    #[test]
    fn leaving_a_session_is_an_answer_and_not_a_steer() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false), ("Unknown 2", 30.0, false)]);
        let queue = queue(&owned);
        let voice = view(&session, "Unknown 1", 1, &queue, &[], &[], &[], &owned[0].1);
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert!(
            !screen.still_working(),
            "a key is only ever read with no target set"
        );
        assert_eq!(
            screen.answer(&voice, Event::Leave, &Free),
            Step::Answered(Answer::Leave)
        );
        assert!(
            !screen.still_working(),
            "leaving sets no target, so the loop is not told to keep the session open"
        );
    }

    /// AC #11: the snippets scroll past the three a line prompt prints, and clamp at both ends.
    #[test]
    fn snippets_scroll_past_three_and_clamp() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let snippets = ["one", "two", "three", "four", "five"].map(snippet);
        let voice = view(
            &session,
            "Unknown 1",
            1,
            &queue,
            &snippets,
            &[],
            &[],
            &owned[0].1,
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert_eq!(screen.view(&voice, &Free, Context::Reading).snippet, 0);
        screen.answer(&voice, Event::SnippetUp, &Free);
        assert_eq!(
            screen.view(&voice, &Free, Context::Reading).snippet,
            0,
            "clamped at the top"
        );
        for _ in 0..10 {
            screen.answer(&voice, Event::SnippetDown, &Free);
        }
        assert_eq!(
            screen.view(&voice, &Free, Context::Reading).snippet,
            4,
            "clamped with the last snippet at the top"
        );
        // The one-rule claim `selected_index` exists for: the marked row and the row that would
        // play are the same row at both clamps.
        let derived = screen.view(&voice, &Free, Context::Reading).snippet;
        assert_eq!(screen.selected(&voice).map(|(i, _)| i), Some(derived));
        for _ in 0..10 {
            screen.answer(&voice, Event::SnippetUp, &Free);
        }
        let derived = screen.view(&voice, &Free, Context::Reading).snippet;
        assert_eq!(derived, 0);
        assert_eq!(screen.selected(&voice).map(|(i, _)| i), Some(derived));
    }

    /// The selection is the unit of listening: whichever line is selected is the line whose
    /// samples the shell would be handed, so every turn the voice took is reachable rather than
    /// only the longest representative one.
    #[test]
    fn the_selected_line_is_the_one_that_would_play() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let audio: [[f32; 1]; 5] = [[0.1], [0.2], [0.3], [0.4], [0.5]];
        let snippets = [
            heard("one", 0.0, &audio[0]),
            heard("two", 12.0, &audio[1]),
            heard("three", 47.5, &audio[2]),
            heard("four", 61.0, &audio[3]),
            heard("five", 90.0, &audio[4]),
        ];
        let voice = view(
            &session,
            "Unknown 1",
            1,
            &queue,
            &snippets,
            &[],
            &[],
            &owned[0].1,
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert_eq!(
            screen.selected(&voice).map(|(i, s)| (i, s.text)),
            Some((0, "one"))
        );
        screen.answer(&voice, Event::SnippetDown, &Free);
        screen.answer(&voice, Event::SnippetDown, &Free);
        let (index, snippet) = screen.selected(&voice).expect("five lines to choose from");
        assert_eq!(index, 2);
        assert_eq!(snippet.text, "three");
        assert_eq!(snippet.start, 47.5);
        assert_eq!(
            snippet.audio, &audio[2],
            "the third line's samples, not the first's"
        );
    }

    /// A voice with nothing transcribed has no line to play, which is what lets the shell say so
    /// rather than spawn a player over no samples at all.
    #[test]
    fn a_voice_with_no_transcript_has_no_line_to_play() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let voice = view(&session, "Unknown 1", 1, &queue, &[], &[], &[], &owned[0].1);
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert!(screen.selected(&voice).is_none());
        // And moving the selection over nothing stays nothing rather than becoming an index.
        screen.answer(&voice, Event::SnippetDown, &Free);
        assert!(screen.selected(&voice).is_none());
    }

    /// A selected line whose samples are missing -- a truncated or absent `speaker.wav` -- is
    /// still a selected line. The emptiness is what the shell reports on rather than something
    /// that hides the row.
    #[test]
    fn a_selected_line_with_no_samples_is_still_the_selected_line() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let snippets = [heard("one", 0.0, &[]), heard("two", 3.0, &[])];
        let voice = view(
            &session,
            "Unknown 1",
            1,
            &queue,
            &snippets,
            &[],
            &[],
            &owned[0].1,
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        let (index, snippet) = screen.selected(&voice).expect("two lines to choose from");
        assert_eq!(index, 0);
        assert!(snippet.audio.is_empty());
    }

    /// A question about "Unknown 1", ranked against Milo alone, with Ivan enrolled and reachable
    /// only by typing. The fixture every "who is this" test below wants: one candidate the ranking
    /// has a count for and one it does not.
    fn asked<'a>(
        session: &'a SessionId,
        queue: &'a [Queued<'a>],
        similar: &'a [Resemblance],
        enrolled: &'a [&'a str],
        attribution: &'a Attribution,
    ) -> VoiceView<'a> {
        view(
            session,
            "Unknown 1",
            1,
            queue,
            &[],
            similar,
            enrolled,
            attribution,
        )
    }

    /// AC #1: the highlighted candidate says how many recordings of them the database holds --
    /// including for somebody the ranking has no count for at all, which is the case the candidate
    /// row can only draw as `--`.
    #[test]
    fn the_highlighted_candidate_says_how_many_references_they_hold() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 3)]);
        let enrolled = ["Milo", "Ivan"];
        let voice = asked(&session, &queue, &similar, &enrolled, &owned[0].1);
        let found = scanned(
            vec![
                holding(
                    "Milo",
                    &[&[names("20260819-100000", "Unknown 1", "Milo")], &[]],
                ),
                holding("Ivan", &[&[], &[], &[]]),
            ],
            &[],
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        let derived = screen.view(&voice, &Free, Context::Read(&found));
        assert_eq!(
            derived.who,
            Who::Known {
                references: 2,
                voices: 1,
                sessions: vec![Named {
                    session: "20260819-100000".to_string(),
                    voices: vec!["Unknown 1".to_string()],
                }],
                unreadable: 0,
            }
        );

        // Ivan is absent from the ranking, so the candidate row has no count to show. The scan has
        // one for everybody enrolled, which is what makes this pane worth its rows.
        for c in "Ivan".chars() {
            screen.answer(&voice, Event::Filter(c), &Free);
        }
        let derived = screen.view(&voice, &Free, Context::Read(&found));
        assert_eq!(derived.candidates[0].name, "Ivan");
        assert_eq!(
            derived.candidates[0].references, None,
            "the ranking has no count for somebody reachable only by typing"
        );
        assert!(
            matches!(derived.who, Who::Known { references: 3, .. }),
            "{:?}",
            derived.who
        );
    }

    /// AC #2: which sessions and which voices, grouped by meeting, most recent first, and one
    /// voice counted once however many of somebody's recordings are naming it.
    #[test]
    fn the_highlighted_candidate_says_which_sessions_and_voices_it_names() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 3)]);
        let enrolled = ["Milo"];
        let voice = asked(&session, &queue, &similar, &enrolled, &owned[0].1);
        let found = scanned(
            vec![holding(
                "Milo",
                &[
                    &[
                        names("20260810-101500", "Unknown 1", "Milo"),
                        names("20260809-052600", "Unknown 3", "Milo"),
                    ],
                    &[
                        // The same voice as the first reference names, and one more beside it.
                        names("20260810-101500", "Unknown 1", "Milo"),
                        names("20260810-101500", "Unknown 4", "Milo"),
                    ],
                ],
            )],
            &[],
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert_eq!(
            screen.view(&voice, &Free, Context::Read(&found)).who,
            Who::Known {
                references: 2,
                voices: 3,
                sessions: vec![
                    Named {
                        session: "20260810-101500".to_string(),
                        voices: vec!["Unknown 1".to_string(), "Unknown 4".to_string()],
                    },
                    Named {
                        session: "20260809-052600".to_string(),
                        voices: vec!["Unknown 3".to_string()],
                    },
                ],
                unreadable: 0,
            }
        );
    }

    /// The answer somebody at the reference cap is looking for, and the one that has to keep its
    /// scope clause: a reference naming nothing *in any session read* is not the same claim as one
    /// naming nothing.
    #[test]
    fn a_reference_naming_nothing_says_so_with_the_scope_it_was_read_against() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 2)]);
        let enrolled = ["Milo"];
        let voice = asked(&session, &queue, &similar, &enrolled, &owned[0].1);
        let found = scanned(vec![holding("Milo", &[&[], &[]])], &[]);
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert_eq!(
            screen.view(&voice, &Free, Context::Read(&found)).who,
            Who::Known {
                references: 2,
                voices: 0,
                sessions: Vec::new(),
                unreadable: 0,
            }
        );
    }

    /// AC #5: a session the scan could not read leaves the answer incomplete and says so, rather
    /// than failing the run or quietly reporting over less than it claims.
    #[test]
    fn a_session_that_could_not_be_read_leaves_the_context_incomplete() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 1)]);
        let enrolled = ["Milo"];
        let voice = asked(&session, &queue, &similar, &enrolled, &owned[0].1);
        let found = scanned(
            vec![holding(
                "Milo",
                &[&[names("20260810-101500", "Unknown 1", "Milo")]],
            )],
            &["20260809-052600"],
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        // What it could read is still reported, which is the whole point of not failing.
        assert_eq!(
            screen.view(&voice, &Free, Context::Read(&found)).who,
            Who::Known {
                references: 1,
                voices: 1,
                sessions: vec![Named {
                    session: "20260810-101500".to_string(),
                    voices: vec!["Unknown 1".to_string()],
                }],
                unreadable: 1,
            }
        );
    }

    /// The one wrong answer here that would look like a fact. Somebody enrolled during this very
    /// run is absent from a scan taken before that answer, and "0 reference(s), naming nothing"
    /// would be read as a person whose recordings do nothing -- which is the opposite of true.
    #[test]
    fn a_candidate_enrolled_during_this_run_is_not_reported_as_naming_nothing() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Maya", 0.64, 1)]);
        let enrolled = ["Maya"];
        let voice = asked(&session, &queue, &similar, &enrolled, &owned[0].1);
        // The scan ran before Maya existed, so it knows only Milo.
        let found = scanned(
            vec![holding(
                "Milo",
                &[&[names("20260819-100000", "Unknown 2", "Milo")]],
            )],
            &[],
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert_eq!(
            screen.view(&voice, &Free, Context::Read(&found)).who,
            Who::Unrecorded
        );
    }

    /// The pane is about the highlighted candidate, so both things that move the highlight move
    /// it: the candidate keys and the filter. And with nothing highlighted there is nobody to
    /// report on rather than an empty listing.
    #[test]
    fn moving_the_highlight_and_typing_both_move_the_context() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        // Two people who share a prefix, so the filter below leaves both candidates standing and
        // what moves the pane is the highlight itself rather than the list emptying out.
        let similar = resembles(&[("Marco", 0.71, 2), ("Marcel", 0.38, 1)]);
        let enrolled = ["Marco", "Marcel"];
        let voice = asked(&session, &queue, &similar, &enrolled, &owned[0].1);
        let found = scanned(
            vec![
                holding("Marco", &[&[], &[]]),
                holding(
                    "Marcel",
                    &[&[names("20260819-100000", "Unknown 2", "Marcel")]],
                ),
            ],
            &[],
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert!(
            matches!(
                screen.view(&voice, &Free, Context::Read(&found)).who,
                Who::Known { references: 2, .. }
            ),
            "the top of the ranking"
        );

        screen.answer(&voice, Event::CandidateDown, &Free);
        assert!(matches!(
            screen.view(&voice, &Free, Context::Read(&found)).who,
            Who::Known { references: 1, .. }
        ));

        screen.answer(&voice, Event::Filter('M'), &Free);
        assert!(
            matches!(
                screen.view(&voice, &Free, Context::Read(&found)).who,
                Who::Known { references: 2, .. }
            ),
            "the filter moved the highlight back to Marco"
        );

        for c in "Quentin".chars() {
            screen.answer(&voice, Event::Filter(c), &Free);
        }
        let derived = screen.view(&voice, &Free, Context::Read(&found));
        assert!(derived.candidates.is_empty());
        assert_eq!(derived.who, Who::Nobody);
    }

    /// What a reference *names* is the question, and `elsewhere` is not an answer to it: those are
    /// the voices a removal would *also* move, which is what `meethook forget` asks and nothing on
    /// this frame does. Counting them here would put voices in the pane that do not read this name
    /// at all.
    #[test]
    fn elsewhere_is_not_what_a_reference_names() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 1)]);
        let enrolled = ["Milo"];
        let voice = asked(&session, &queue, &similar, &enrolled, &owned[0].1);
        let found = scanned(
            vec![Enrolled {
                name: "Milo".to_string(),
                references: vec![Reference {
                    handle: 1,
                    clip_seconds: Some(42.5),
                    depends: Vec::new(),
                    // Removing this row would hand Ivan a name the veto is holding off. Real, and
                    // not what this row is naming.
                    elsewhere: vec![names("20260810-101500", "Unknown 7", "Ivan")],
                }],
            }],
            &[],
        );
        let mut screen = Screen::default();
        screen.arrive(&voice);

        assert_eq!(
            screen.view(&voice, &Free, Context::Read(&found)).who,
            Who::Known {
                references: 1,
                voices: 0,
                sessions: Vec::new(),
                unreadable: 0,
            }
        );
    }

    /// The session changing resets everything the state holds, so a cursor left on session A's
    /// row 7 cannot select session B's row 7.
    #[test]
    fn a_new_session_resets_the_cursor_the_filter_and_the_memo() {
        let a = session();
        let b = other_session();
        let owned = rows(&[("Unknown 1", 60.0, false), ("Unknown 2", 60.0, false)]);
        let queue = queue(&owned);
        let similar = resembles(&[("Milo", 0.71, 3)]);
        let enrolled = ["Milo"];
        let mut screen = Screen::default();
        let counted = Counted(Cell::new(0));

        let second = view(
            &a,
            "Unknown 2",
            2,
            &queue,
            &[],
            &similar,
            &enrolled,
            &owned[1].1,
        );
        screen.arrive(&second);
        // A filter that leaves a candidate highlighted, so the cost memo has something in it to
        // survive or not.
        screen.answer(&second, Event::Filter('M'), &counted);
        screen.answer(&second, Event::Up, &counted);
        screen.answer(&second, Event::Select, &counted);
        screen.view(&second, &counted, Context::Reading);
        assert!(screen.still_working());
        assert_eq!(counted.0.get(), 1);

        let elsewhere = view(
            &b,
            "Unknown 2",
            2,
            &queue,
            &[],
            &similar,
            &enrolled,
            &owned[1].1,
        );
        assert_eq!(screen.arrive(&elsewhere), None);
        assert!(!screen.still_working());
        let derived = screen.view(&elsewhere, &counted, Context::Reading);
        assert_eq!(derived.filter, "");
        assert_eq!(derived.cursor, 1, "snapped to this session's own row");
        assert!(derived.rows.iter().all(|row| row.mark.is_none()));
        assert_eq!(counted.0.get(), 2, "the memo did not survive the session");
    }
}
