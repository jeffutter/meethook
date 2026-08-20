//! The enrolment frame's state machine, with no terminal anywhere in it.
//!
//! Everything the full-screen interface *decides* is here: which voice the cursor is on, which
//! voice the user is steering toward, what has been typed into the filter, which candidate is
//! highlighted, how far the snippets are scrolled, and what this run has already done to each
//! voice. It takes typed [`Event`]s and returns either "still going" or an
//! [`Answer`].
//!
//! There is deliberately no `ratatui` path in this file, no [`std::io::Write`], no clock and no
//! [`Preview`](meethook_enroll::Preview). That is not tidiness: the sibling module `render` and
//! the shell in `super` both need a person in front of them, and this is the part that does not,
//! so this is where the tests are. The absence of a `ratatui` import is what keeps that honest.
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

use std::collections::{BTreeMap, BTreeSet};

use meethook_enroll::{Answer, Position, Queued, Refusal, Resolution, resolve};
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
    pub snippets: &'a [&'a str],
    pub resembles: &'a [Resemblance],
    pub enrolled: &'a [&'a str],
    /// Whether there is audio to play. The clip itself is the shell's business, and a state
    /// machine holding a quarter of a megabyte of samples per redraw would be paying for nothing.
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
    /// Answer with the typed text as somebody new. Its own key, never the fallback for
    /// unrecognised text.
    NewPerson,
    SnippetUp,
    SnippetDown,
    /// Play the clip. Handled by the shell, which is the only thing holding the samples; here for
    /// the sake of one total `match`.
    Play,
    Skip,
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
    /// Every snippet, with the pane scrolled to [`View::snippet`].
    pub snippets: &'a [&'a str],
    pub snippet: usize,
    pub clip_is_empty: bool,
    /// One line about what just happened -- a clip that would not play, a voice that turned out
    /// not to be reachable. Cleared by the next key.
    pub status: Option<&'a str>,
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

        // A filter, a candidate highlight and a scroll offset are all about the voice that was on
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
                let names = self.candidates(view);
                let Some(name) = names.get(self.candidate).cloned() else {
                    // Unrecognised text, or nothing enrolled at all. Deliberately nothing: the
                    // only way to create somebody is the key that says so.
                    return Step::Waiting;
                };
                if self.cost(&name, costs).refusal.is_some() {
                    return Step::Waiting;
                }
                self.decided.insert(view.number.to_string(), Mark::Answered);
                return Step::Answered(Answer::Named(name));
            }
            Event::NewPerson => {
                let typed = self.filter.trim();
                if typed.is_empty() {
                    return Step::Waiting;
                }
                let name = typed.to_string();
                self.decided.insert(view.number.to_string(), Mark::Answered);
                return Step::Answered(Answer::Named(name));
            }
            Event::SnippetUp => {
                self.snippet = self.snippet.saturating_sub(1);
            }
            Event::SnippetDown => {
                // Clamped so that the last snippet can reach the top of the pane and no further.
                // Clamping is how this scroll is defined out of existence rather than
                // bounds-checked at every use.
                let last = view.snippets.len().saturating_sub(1);
                self.snippet = (self.snippet + 1).min(last);
            }
            // The shell holds the samples, so it intercepts this before the state machine sees
            // it. Here so that the `match` is total and adding a key cannot silently do nothing.
            Event::Play => {}
            Event::Skip => {
                self.decided.insert(view.number.to_string(), Mark::Skipped);
                return Step::Answered(Answer::Skip);
            }
            Event::Quit => return Step::Answered(Answer::Quit),
        }
        Step::Waiting
    }

    /// Everything the panes draw, derived from the state and this voice.
    ///
    /// `&mut self` because of the cost memo: the highlighted candidate's consequence is computed
    /// here, once per distinct name, and not once per keystroke.
    pub fn view<'a>(&'a mut self, view: &'a VoiceView<'a>, costs: &dyn Costs) -> View<'a> {
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
            snippets: view.snippets,
            snippet: self.snippet.min(view.snippets.len().saturating_sub(1)),
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
mod tests {
    use std::cell::Cell;

    use meethook_enroll::{Answer, Position, Queued};
    use meethook_session::SessionId;
    use meethook_transcribe::{Attribution, Resemblance};

    use super::{Cost, Costs, Event, Mark, Screen, Step, VoiceView};

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

    fn resembles(spec: &[(&str, f32, usize)]) -> Vec<Resemblance> {
        spec.iter()
            .map(|(name, similarity, references)| Resemblance {
                name: (*name).to_string(),
                similarity: *similarity,
                references: *references,
            })
            .collect()
    }

    /// A question about `number`, with everything else defaulted by the caller's slices.
    #[allow(clippy::too_many_arguments)]
    fn view<'a>(
        session: &'a SessionId,
        number: &'a str,
        nth: usize,
        queue: &'a [Queued<'a>],
        snippets: &'a [&'a str],
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

        let derived = screen.view(&voice, &Free);
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
        assert_eq!(screen.view(&third, &Free).cursor, 2);
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
        assert!(screen.view(&first, &Free).status.is_some());
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

        let derived = screen.view(&voice, &Free);
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
        let derived = screen.view(&voice, &Free);
        let names: Vec<&str> = derived.candidates.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["Maya"]);
        // Reachable, and honest about having no similarity to report rather than inventing one.
        assert_eq!(derived.candidates[0].similarity, None);

        screen.answer(&voice, Event::ClearFilter, &Free);
        for c in "Quentin".chars() {
            screen.answer(&voice, Event::Filter(c), &Free);
        }
        assert!(screen.view(&voice, &Free).candidates.is_empty());
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
            Step::Answered(Answer::Named("Marco".to_string()))
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
        let derived = screen.view(&voice, &vetoes);
        assert_eq!(
            derived.candidates[0].refusal,
            Some(meethook_enroll::Refusal::Vetoed {
                holder: Some("Unknown 2".to_string())
            })
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
            Step::Answered(Answer::Named("Maya".to_string()))
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
            let derived = screen.view(&voice, &counted);
            assert_eq!(derived.consequence, ["would name this voice Marco"]);
        }
        assert_eq!(counted.0.get(), 1, "one distinct highlight, one call");

        // The next question starts from nothing: a preview taken against the old database would
        // be stale as well as expensive.
        screen.arrive(&voice);
        screen.view(&voice, &counted);
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

    /// AC #11: the snippets scroll past the three a line prompt prints, and clamp at both ends.
    #[test]
    fn snippets_scroll_past_three_and_clamp() {
        let session = session();
        let owned = rows(&[("Unknown 1", 60.0, false)]);
        let queue = queue(&owned);
        let snippets = ["one", "two", "three", "four", "five"];
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

        assert_eq!(screen.view(&voice, &Free).snippet, 0);
        screen.answer(&voice, Event::SnippetUp, &Free);
        assert_eq!(screen.view(&voice, &Free).snippet, 0, "clamped at the top");
        for _ in 0..10 {
            screen.answer(&voice, Event::SnippetDown, &Free);
        }
        assert_eq!(
            screen.view(&voice, &Free).snippet,
            4,
            "clamped with the last snippet at the top"
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
        screen.view(&second, &counted);
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
        let derived = screen.view(&elsewhere, &counted);
        assert_eq!(derived.filter, "");
        assert_eq!(derived.cursor, 1, "snapped to this session's own row");
        assert!(derived.rows.iter().all(|row| row.mark.is_none()));
        assert_eq!(counted.0.get(), 2, "the memo did not survive the session");
    }
}
