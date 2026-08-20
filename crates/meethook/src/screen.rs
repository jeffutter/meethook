//! The full-screen answerer for `enroll`: terminal setup, the key map, the event loop, teardown.
//!
//! Three files, and the split is testability rather than tidiness:
//!
//! - [`state`] is the state machine -- the cursors, the filter, the deferrals, the derived view.
//!   No terminal, no `ratatui`, no clock. That is where the tests are.
//! - [`render`] is the panes, given that view. Exercised through `ratatui`'s `TestBackend`.
//! - here: the parts that genuinely need a person in front of them, plus the two seams that let
//!   the other two files exist -- [`Costs`] over
//!   [`Preview`] and the key map.
//!
//! # What this module is careful about
//!
//! **The terminal comes back on every exit.** [`ratatui::try_init`] installs a panic hook that
//! restores before the original hook runs, which covers a panic during a draw; [`Drop`] covers the
//! ordinary and the `?` paths. Acquired lazily, on the first question that actually needs a frame,
//! so a run whose sessions are all passed over never flashes the alternate screen.
//!
//! **The narration is placed, not scrolled.** `Narrator` and `Interviewer` are two separate `&mut`
//! arguments to [`run_enroll`](meethook_enroll::run_enroll), so one object cannot be both. They
//! share a [`Shared`] buffer instead: `Lines` writes into it, this frame draws its tail, and
//! [`Interface::finish`] flushes the whole thing to stdout once the frame is down -- so a
//! full-screen run leaves the same scrollback a plain one does.

pub mod render;
pub mod state;

use std::cell::RefCell;
use std::io::{self, Write};
use std::rc::Rc;
use std::time::Duration;

use meethook_enroll::{Answer, Consequence, Interviewer, Preview, Voice, speech};
use meethook_session::{Displaced, Stored};
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{
    Event as Key, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, poll, read,
};

use crate::commands::{Clips, Progress};
use state::{Cost, Costs, Event, Screen, Step, VoiceView};

/// How often a frame with a clip playing redraws itself, and so how fast the position moves and
/// how promptly it disappears when the clip ends.
///
/// Four times a second is enough that a seconds-resolution position looks live, and cheap enough
/// that re-deriving the view that often is not worth measuring: the costs behind it are memoised,
/// so every redraw after the first of a question is a lookup.
const TICK: Duration = Duration::from_millis(250);

/// How long the frame should wait for a key before redrawing -- `None` to block until one arrives.
///
/// The whole of the rule, in a function next to [`event`] and for the same reason: it is then
/// decidable in `cargo test` with no terminal in front of it. Playback is the only thing that puts
/// a clock on this frame, and only while it lasts. An idle frame has nothing that would redraw
/// differently, so it blocks rather than waking four times a second to find that out.
fn wait(playing: Option<Progress>) -> Option<Duration> {
    playing.map(|_| TICK)
}

/// A narration buffer two owners can hold at once.
///
/// `Lines::new(&mut shared)` on one side and a clone of this on the other, which is the only shape
/// available: the run takes its narrator and its interviewer as two separate `&mut` arguments, so
/// they cannot be one object. `Rc<RefCell<_>>` and not a channel because both halves are on the
/// same thread by construction -- `Interviewer` is a `&mut dyn` with no `Send` bound anywhere near
/// it -- and a channel would add a failure mode that cannot occur.
#[derive(Clone, Default)]
pub struct Shared(Rc<RefCell<Vec<u8>>>);

impl Write for Shared {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(buf);
        Ok(buf.len())
    }

    /// Nothing to flush: the bytes are already in the buffer, and where they go from there is
    /// [`Interface::finish`]'s decision rather than a write's.
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Shared {
    /// Everything narrated so far, as lines, for the pane. Blank lines are kept: `Lines` uses one
    /// to separate a session's block from the previous one, and dropping it here would close a gap
    /// the wording relies on.
    fn lines(&self) -> Vec<String> {
        let bytes = self.0.borrow();
        String::from_utf8_lossy(&bytes)
            .lines()
            .map(str::to_string)
            .collect()
    }

    /// Writes the whole buffer out, byte for byte.
    fn drain_to(&self, out: &mut dyn Write) -> io::Result<()> {
        out.write_all(&self.0.borrow())
    }
}

/// The full-screen implementation of [`Interviewer`].
pub struct Interface {
    state: Screen,
    narration: Shared,
    clips: Clips,
    /// `Some` once the terminal has been acquired, and the flag that says a restore is owed.
    terminal: Option<DefaultTerminal>,
    /// Why the frame stopped, when it stopped for a reason the user has to be told about.
    ///
    /// Stashed rather than returned because [`Interviewer::identify`] is infallible by design: a
    /// terminal that cannot be opened is a run that stops, not a run that fails, and everything
    /// answered before it stopped is already on disk. Printed by [`Interface::finish`], after the
    /// screen is back.
    trouble: Option<String>,
}

impl Interface {
    /// A frame sharing `narration` with the run's [`Lines`](meethook_enroll::Lines).
    pub fn new(narration: Shared) -> Interface {
        Interface {
            state: Screen::default(),
            narration,
            clips: Clips::default(),
            terminal: None,
            trouble: None,
        }
    }

    /// Takes the frame down and writes the narration out.
    ///
    /// Called before `enroll` prints its run summary, which is the whole reason it exists as a
    /// step rather than as a `Drop`: the summary has to land on the restored screen, below the
    /// scrollback this flushes, and a frame that lived for the length of the function could not
    /// promise that.
    ///
    /// `&mut dyn Write` rather than `io::stdout()` so the flush is decidable against a `Vec<u8>`.
    pub fn finish(mut self, out: &mut dyn Write) -> io::Result<()> {
        self.restore();
        self.narration.drain_to(out)?;
        if let Some(trouble) = self.trouble.take() {
            writeln!(out, "{trouble}")?;
        }
        Ok(())
    }

    /// Puts the terminal back, if this frame ever took it. Idempotent, because both [`Drop`] and
    /// [`Interface::finish`] call it and either may come first.
    fn restore(&mut self) {
        if self.terminal.take().is_some() {
            ratatui::restore();
        }
    }
}

impl Drop for Interface {
    /// The `?` and panic-free-but-early paths. A panic *during a draw* is covered by the hook
    /// `ratatui::try_init` installs, which restores before the original hook runs.
    fn drop(&mut self) {
        self.restore();
    }
}

impl Interviewer for Interface {
    fn identify(&mut self, voice: &Voice<'_>) -> Answer {
        let view = VoiceView {
            session: voice.session,
            position: voice.position,
            number: voice.number,
            speech_seconds: voice.speech_seconds,
            attribution: voice.attribution,
            queue: voice.queue,
            snippets: &voice.snippets,
            resembles: &voice.resembles,
            enrolled: &voice.enrolled,
            clip_is_empty: voice.clip.is_empty(),
        };
        // The interface deferring on its own: the user is steering toward another voice, so this
        // question goes back in the queue without a frame being drawn or a key being read.
        if let Some(answer) = self.state.arrive(&view) {
            return answer;
        }

        // Lazily, and only here: a run that passes over every session should not flash the
        // alternate screen, and `run_enroll` emits run- and session-level notes before any
        // question.
        if self.terminal.is_none() {
            match ratatui::try_init() {
                Ok(terminal) => self.terminal = Some(terminal),
                Err(e) => {
                    self.trouble = Some(format!(
                        "the full-screen interface could not be opened ({e}) -- \
                         try meethook enroll --plain"
                    ));
                    return Answer::Quit;
                }
            }
        }

        // One call site, and one stop after it: answered, skipped, deferred, quit and a frame that
        // could not be drawn are five ways out of the loop, and every one of them has to leave the
        // audio behind. `Drop for Clips` covers the paths that never come back through here.
        let answer = self.ask(&view, voice);
        self.clips.stop();
        answer
    }

    /// The other half of the deferral contract. `true` exactly while the user is steering toward a
    /// voice, so a pass that produced no answer is not a finished session.
    fn still_working(&self) -> bool {
        self.state.still_working()
    }
}

impl Interface {
    /// Draws and reads keys until the question is answered, or until there is nothing to draw on.
    ///
    /// Split out of [`Interviewer::identify`] so playback has exactly one place to be stopped: see
    /// the comment at the call.
    fn ask(&mut self, view: &VoiceView<'_>, voice: &Voice<'_>) -> Answer {
        // Field by field, so the state machine and the terminal can be borrowed at once.
        let Interface {
            state,
            narration,
            clips,
            terminal,
            trouble,
        } = self;
        let Some(terminal) = terminal.as_mut() else {
            return Answer::Quit;
        };

        loop {
            // A clip that will not play is only knowable here, once the child has been reaped, so
            // the report lands an iteration after the key that started it rather than at the spawn.
            let playing = match clips.poll() {
                Ok(playing) => playing,
                Err(e) => {
                    state.say(format!("could not play the clip: {e}"));
                    None
                }
            };
            {
                let derived = state.view(view, &voice.preview);
                let lines = narration.lines();
                if let Err(e) =
                    terminal.draw(|frame| render::draw(frame, &derived, &lines, playing))
                {
                    *trouble = Some(format!("the frame could not be drawn ({e})"));
                    return Answer::Quit;
                }
            }

            // While something is playing, waking on a timeout is what moves the position and what
            // takes it away when the clip ends; the redraw at the top of the loop does both. With
            // nothing playing this falls straight through and blocks in `read`.
            if let Some(timeout) = wait(playing) {
                match poll(timeout) {
                    Ok(true) => {}
                    Ok(false) => continue,
                    Err(e) => {
                        *trouble = Some(format!("the terminal stopped being readable ({e})"));
                        return Answer::Quit;
                    }
                }
            }

            let key = match read() {
                Ok(Key::Key(key)) => key,
                // A resize, a mouse report, a paste. The loop redraws at the top of every
                // iteration, so a resize needs nothing else done about it.
                Ok(_) => continue,
                // End of input under raw mode, or a terminal that has gone away. Stopping is the
                // same ordinary outcome the line prompt gives it: everything answered is written.
                Err(e) => {
                    *trouble = Some(format!("the terminal stopped being readable ({e})"));
                    return Answer::Quit;
                }
            };
            let Some(event) = event(key) else {
                continue;
            };
            // Playback is intercepted here because the samples are the shell's: the state machine
            // deliberately holds no audio.
            if event == Event::Play {
                let problem = if voice.clip.is_empty() {
                    Some("there is no audio for this voice".to_string())
                } else {
                    clips
                        .start(voice.clip)
                        .err()
                        .map(|e| format!("could not play the clip: {e}"))
                };
                match problem {
                    Some(problem) => state.say(problem),
                    // A play that has now worked takes back what the last failed one said.
                    // Nothing else clears the footer for a key the state machine never sees.
                    None => state.hush(),
                }
                continue;
            }
            match state.answer(view, event, &voice.preview) {
                Step::Waiting => continue,
                Step::Answered(answer) => return answer,
            }
        }
    }
}

/// What a candidate costs, off the run's own dry run.
///
/// The only place in this crate that reads a [`Consequence`], and it has to be here rather than in
/// [`state`]: `Consequence`'s two state fields are crate-visible to `meethook-enroll`, so one
/// cannot be constructed from this crate at all and anything taking one would be untestable.
impl Costs for Preview<'_> {
    fn of(&self, name: &str) -> Cost {
        match Preview::of(self, name) {
            Some(consequence) => Cost {
                refusal: consequence.refused.clone(),
                summary: would(&consequence),
            },
            // A name of nothing but spaces, which is a skip rather than an answer. Nothing is
            // refused and nothing would be written.
            None => Cost {
                refusal: None,
                summary: vec!["a name of nothing but spaces writes nothing".to_string()],
            },
        }
    }
}

/// What answering with one name would do, as the lines the frame shows before it is chosen.
///
/// Read off [`Consequence`]'s public fields rather than restating its five outcomes: the mapping
/// from `stored` plus `session_only()` to a sentence is documented there, and a second copy of it
/// here is exactly what that module's doc forbids.
fn would(consequence: &Consequence) -> Vec<String> {
    let mut lines = Vec::new();
    match &consequence.stored {
        Some(Stored::Enrolled) => lines.push("enrols them, from this voice".to_string()),
        Some(Stored::Added { held }) => {
            lines.push(format!("stores another recording of them, {held} in all"));
        }
        Some(Stored::AlreadyHeld) => {
            lines.push("stores nothing new: they already hold this recording".to_string());
        }
        Some(Stored::Replaced {
            held,
            evicted_seconds,
        }) => lines.push(format!(
            "stores this recording in place of their shortest, {}, {held} in all",
            speech(*evicted_seconds)
        )),
        Some(Stored::AtCapacity { held, .. }) => lines.push(format!(
            "stores nothing: they hold {held} recordings and none is shorter than this voice"
        )),
        None => {}
    }
    if consequence.session_only() {
        lines.push("names this voice in this session only, storing no reference".to_string());
    }
    for Displaced { name, remaining } in &consequence.displaced {
        lines.push(format!(
            "takes a recording off {name}, leaving them {remaining}"
        ));
    }
    for name in &consequence.stale {
        lines.push(format!(
            "leaves a recording of this voice standing under {name}"
        ));
    }
    lines
}

/// Which key means what.
///
/// Two cursors and a filter that swallows typing, so the queue, the candidates and the snippets
/// each get their own pair and the two commitments -- go to that voice, answer with that name --
/// are separate keys rather than one Enter that means different things in different places.
///
/// Ctrl-C and Ctrl-D are both `Quit`: raw mode means no SIGINT arrives, and the line prompt
/// already treats end of input as stopping rather than as a failure.
///
/// A free function taking a `KeyEvent` because a `KeyEvent` is constructible without a terminal,
/// which is what makes this whole rule testable -- and it is where a stray Ctrl or a paste-shaped
/// burst of characters would otherwise go wrong.
fn event(key: KeyEvent) -> Option<Event> {
    // A release is not a press. Terminals that report both would otherwise act on every key
    // twice.
    if key.kind == KeyEventKind::Release {
        return None;
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match (key.code, ctrl) {
        (KeyCode::Char('c' | 'd'), true) => Some(Event::Quit),
        (KeyCode::Char('n'), true) => Some(Event::NewPerson),
        (KeyCode::Char('p'), true) => Some(Event::Play),
        (KeyCode::Char('s'), true) => Some(Event::Skip),
        (KeyCode::Up, false) => Some(Event::Up),
        (KeyCode::Down, false) => Some(Event::Down),
        (KeyCode::Right, false) => Some(Event::Select),
        (KeyCode::PageUp, false) => Some(Event::SnippetUp),
        (KeyCode::PageDown, false) => Some(Event::SnippetDown),
        (KeyCode::Tab, false) => Some(Event::CandidateDown),
        // Shift-Tab arrives as its own code, with the Shift modifier set, so the modifier is not
        // examined here.
        (KeyCode::BackTab, _) => Some(Event::CandidateUp),
        (KeyCode::Enter, false) => Some(Event::Choose),
        (KeyCode::Backspace, false) => Some(Event::Backspace),
        (KeyCode::Esc, false) => Some(Event::ClearFilter),
        // Everything else printable goes to the filter, which is why choosing and creating are
        // control keys. Control characters are excluded so a terminal reporting Ctrl-A as
        // `Char('\u{1}')` cannot type into the filter.
        (KeyCode::Char(c), false) if !c.is_control() => Some(Event::Filter(c)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::time::Duration;

    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    use super::state::Event;
    use super::{Interface, Progress, Shared, TICK, event, wait};

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    /// Every key this frame binds, in one table, so a rebinding that drops one fails here rather
    /// than in front of a user.
    #[test]
    fn every_bound_key_means_what_the_footer_says() {
        let bound = [
            (KeyCode::Up, KeyModifiers::NONE, Event::Up),
            (KeyCode::Down, KeyModifiers::NONE, Event::Down),
            (KeyCode::Right, KeyModifiers::NONE, Event::Select),
            (KeyCode::PageUp, KeyModifiers::NONE, Event::SnippetUp),
            (KeyCode::PageDown, KeyModifiers::NONE, Event::SnippetDown),
            (KeyCode::Tab, KeyModifiers::NONE, Event::CandidateDown),
            (KeyCode::BackTab, KeyModifiers::SHIFT, Event::CandidateUp),
            (KeyCode::Enter, KeyModifiers::NONE, Event::Choose),
            (KeyCode::Backspace, KeyModifiers::NONE, Event::Backspace),
            (KeyCode::Esc, KeyModifiers::NONE, Event::ClearFilter),
            (KeyCode::Char('n'), KeyModifiers::CONTROL, Event::NewPerson),
            (KeyCode::Char('p'), KeyModifiers::CONTROL, Event::Play),
            (KeyCode::Char('s'), KeyModifiers::CONTROL, Event::Skip),
            (KeyCode::Char('a'), KeyModifiers::NONE, Event::Filter('a')),
            (KeyCode::Char(' '), KeyModifiers::NONE, Event::Filter(' ')),
        ];
        for (code, modifiers, expected) in bound {
            assert_eq!(
                event(key(code, modifiers)),
                Some(expected),
                "{code:?} with {modifiers:?}"
            );
        }
    }

    /// Only playback puts a clock on this frame. An idle frame blocks for its next key rather than
    /// waking four times a second to redraw something that cannot have changed, and the bounds on
    /// `TICK` are what stop a later edit turning the rule into a spin at one end or a position that
    /// visibly lurches at the other.
    #[test]
    fn an_idle_frame_waits_for_its_next_key() {
        assert_eq!(wait(None), None, "nothing to redraw, so block");
        let playing = Progress {
            elapsed: Duration::from_secs(3),
            length: Duration::from_secs(30),
        };
        assert_eq!(wait(Some(playing)), Some(TICK));
        assert!(
            TICK >= Duration::from_millis(50) && TICK <= Duration::from_millis(500),
            "{TICK:?} is either a spin or a position that jumps"
        );
    }

    /// Both ways out. Raw mode means no SIGINT arrives, so Ctrl-C has to be bound or the frame
    /// cannot be left by the key everybody reaches for first.
    #[test]
    fn control_c_and_control_d_both_quit() {
        for c in ['c', 'd'] {
            assert_eq!(
                event(key(KeyCode::Char(c), KeyModifiers::CONTROL)),
                Some(Event::Quit)
            );
        }
    }

    /// An unbound key does nothing at all, and a release is not a press.
    #[test]
    fn an_unbound_key_and_a_release_are_both_ignored() {
        assert_eq!(event(key(KeyCode::F(5), KeyModifiers::NONE)), None);
        assert_eq!(event(key(KeyCode::Left, KeyModifiers::NONE)), None);
        assert_eq!(
            event(key(KeyCode::Char('a'), KeyModifiers::CONTROL)),
            None,
            "an unbound control key must not type into the filter"
        );
        let released = KeyEvent {
            kind: KeyEventKind::Release,
            ..key(KeyCode::Enter, KeyModifiers::NONE)
        };
        assert_eq!(event(released), None);
    }

    /// AC #12's other half: what the frame drew is not all the narration did. Everything written
    /// into the shared buffer reaches the stream once the frame is down, byte for byte -- which is
    /// what makes a full-screen run leave the same scrollback a plain one does.
    #[test]
    fn the_narration_reaches_the_stream_whole() {
        let mut shared = Shared::default();
        writeln!(shared, "\n20260819-100000  3 voice(s)").expect("a buffer cannot fail");
        writeln!(shared, "  named Milo").expect("a buffer cannot fail");

        assert_eq!(
            shared.lines(),
            ["", "20260819-100000  3 voice(s)", "  named Milo"]
        );

        let mut out: Vec<u8> = Vec::new();
        shared.drain_to(&mut out).expect("a Vec cannot fail");
        assert_eq!(
            String::from_utf8(out).expect("what went in was utf-8"),
            "\n20260819-100000  3 voice(s)\n  named Milo\n"
        );
    }

    /// The buffer is shared rather than copied, which is the whole reason it exists: `Lines` holds
    /// one handle and the frame holds another.
    #[test]
    fn both_halves_see_the_same_buffer() {
        let mut narrator = Shared::default();
        let frame = narrator.clone();
        writeln!(narrator, "one").expect("a buffer cannot fail");
        assert_eq!(frame.lines(), ["one"]);
    }

    /// AC #14, the half a test can reach without a terminal: everything the frame owes the screen
    /// is discharged before `finish` returns, so `enroll`'s run summary lands under the narration
    /// rather than on top of a live alternate screen. And a frame that never took the terminal
    /// must not put one back -- `ratatui::restore` writes `LeaveAlternateScreen` unconditionally,
    /// which on a plain stdout is an escape sequence in the scrollback.
    #[test]
    fn finishing_flushes_before_it_returns_and_restores_nothing_it_did_not_take() {
        let narration = Shared::default();
        let mut narrator = narration.clone();
        writeln!(narrator, "20260819-100000  named Milo").expect("a buffer cannot fail");

        let mut frame = Interface::new(narration);
        assert!(frame.terminal.is_none(), "nothing has asked for a frame");
        frame.trouble = Some("the terminal stopped being readable".to_string());

        let mut out: Vec<u8> = Vec::new();
        frame.finish(&mut out).expect("a Vec cannot fail");
        assert_eq!(
            String::from_utf8(out).expect("what went in was utf-8"),
            "20260819-100000  named Milo\nthe terminal stopped being readable\n"
        );
    }
}
