//! The full-screen presenter for `record`, behind the same [`Reporter`]
//! seam the plain run prints through.
//!
//! # Shape
//!
//! A frame thread owns the terminal; the run's loop keeps running on the main thread exactly
//! as it does in plain mode and hands its notes across a channel. That split is deliberate:
//! the loop's sequencing is the tested contract of `crate::record`, and this module must not
//! get its hands on the event channel -- two readers on one receiver would race, and the loop
//! is where an interrupt has to be *decided*, not just observed. What crosses is therefore
//! owned [`Note`]s one way (run to frame) and a stop signal the other (frame teardown), never
//! events.
//!
//! Raw mode means SIGINT no longer arrives, so Ctrl-C is delivered by the frame itself, through
//! the cloned sender the run's ctrlc handler uses in plain mode: the main loop finalizes the
//! session exactly as it would have. The frame is now a second producer of events in kind,
//! not just in that one case: a hand pick of a calendar offer rides the same cloned sender
//! as an [`Event::MeetingPicked`], and a committed roster correction as an
//! [`Event::RosterEdited`], each decided at the same single reader as a mic edge. What keeps
//! the contract intact is that the payloads are values the run resolves against the lists it
//! handed over -- never a meeting -- and that every match site in the loop owes a deliberate
//! answer per variant.
//!
//! # Teardown
//!
//! [`close`] settles the run once the loop has finished: the frame stops, the alternate
//! screen is left, the narration buffer flushes to stdout in the order the run said it
//! (scrollback parity with a piped run), and any stashed trouble is said on stderr. It is also
//! what a [`Drop`] falls back to when the run ends without reaching it -- an error between the
//! frame coming up and the loop finishing, or a panic -- so the terminal comes back on every
//! exit, the way `enroll`'s interface guarantees it. Wording lives in [`state`] and
//! [`render`]; this file only moves things between threads and restores a terminal.

mod render;
mod state;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use state::EditingField;

// The frame/run/settle machinery below drives a real terminal (spawns a thread, opens the
// alternate screen), so unlike `state` and `render` it has no platform-neutral unit-test path
// -- its only production caller is `record::Sink`, which is macOS-only. Gated to match that
// caller rather than left to the module's broader `any(macos, test)` gate, or a Linux `cargo
// clippy --all-targets` finds every item below dead.
#[cfg(target_os = "macos")]
use std::io::{self, Write};
#[cfg(target_os = "macos")]
use std::sync::mpsc;
#[cfg(target_os = "macos")]
use std::thread;
#[cfg(target_os = "macos")]
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
use anyhow::{Context, Result};
#[cfg(target_os = "macos")]
use meethook_session::Paths;
#[cfg(target_os = "macos")]
use ratatui::Terminal;
#[cfg(target_os = "macos")]
use ratatui::backend::CrosstermBackend;
#[cfg(target_os = "macos")]
use ratatui::crossterm::event::Event as CrosstermEvent;
#[cfg(target_os = "macos")]
use ratatui::crossterm::event::{poll, read};

#[cfg(target_os = "macos")]
use crate::record::{Event, Note, Reporter};
#[cfg(target_os = "macos")]
use state::{Phase, State};

/// How often the frame wakes when nothing else is waiting on it.
///
/// Bounded rather than blocking, on purpose: while idle the key poll is the only thing that
/// waits, and it must wait at most this long, or a Ctrl-C would sit unread until the next note
/// happened to arrive. Two hundred fifty milliseconds is invisible in a redraw and short enough
/// that the interface still feels like it is listening.
#[cfg(target_os = "macos")]
const TICK: Duration = Duration::from_millis(250);

/// The full-screen presenter: owns the alternate screen and the frame thread.
///
/// Constructed before the run starts and closed after it ends; in between it only receives
/// notes. See the module doc for why the run's event channel stays on the main thread.
///
/// The teardown parts are `Option`s so that exactly one of [`close`] and [`Drop`] takes them
/// out and settles them: `close` consumes the struct on the happy path, and a struct that is
/// dropped unsettled -- an early error or a panic in the run -- still gets its frame stopped
/// and its terminal restored by the fallback.
#[cfg(target_os = "macos")]
pub(crate) struct Screen {
    notes: mpsc::Sender<Note>,
    stop: Option<mpsc::Sender<()>>,
    done: Option<mpsc::Receiver<Result<State>>>,
    handle: Option<thread::JoinHandle<()>>,
}

#[cfg(target_os = "macos")]
impl Screen {
    /// Spawns the frame thread.
    ///
    /// `tx` is the clone the frame uses to deliver Ctrl-C as an [`Event::Interrupt`]; the
    /// original sender stays with the run's ctrlc handler, which remains the plain-mode path.
    pub(crate) fn new(tx: mpsc::Sender<Event>, paths: Paths) -> Self {
        let (notes_tx, notes_rx) = mpsc::channel();
        let (stop_tx, stop_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let handle = thread::Builder::new()
            .name("meethook-record-frame".into())
            .spawn(move || {
                let result = frame(tx, notes_rx, stop_rx, paths);
                // The verdict goes out even when the frame failed: `close` reports what the
                // frame knew rather than hanging on a receiver nothing feeds.
                let _ = done_tx.send(result);
            })
            .expect("spawning the frame thread cannot fail");
        Self {
            notes: notes_tx,
            stop: Some(stop_tx),
            done: Some(done_rx),
            handle: Some(handle),
        }
    }
}

#[cfg(target_os = "macos")]
impl Reporter for Screen {
    fn note(&mut self, note: Note) {
        // A frame that is already gone is a teardown race, not a fault: the run is ending
        // either way, and `close` drains whatever was queued before it stopped.
        let _ = self.notes.send(note);
    }
}

/// Settles whatever the frame owes the user and hands back the problems it met.
///
/// Total rather than fallible-on-first-error: the narration and the stashed trouble are owed
/// even when the frame itself failed, so a bad verdict is collected rather than returned at
/// once. The order is load-bearing: the frame stops first (nothing may draw over the
/// restored screen), then the narration flushes to stdout in the order the run said it, then
/// the stashed trouble is said on stderr.
#[cfg(target_os = "macos")]
fn settle(
    stop: mpsc::Sender<()>,
    done: mpsc::Receiver<Result<State>>,
    handle: thread::JoinHandle<()>,
) -> Vec<String> {
    // Stop the frame, then wait for its verdict. The join comes after the verdict so a frame
    // that dies without reporting is reported by the receiver instead of swallowed by the
    // join.
    let _ = stop.send(());
    let mut problems = Vec::new();
    match done.recv() {
        Ok(Ok(mut state)) => {
            // Scrollback parity: exactly what a plain run would have printed, in the order it
            // printed it. Flushed before the trouble so the transcript of the run reads top to
            // bottom.
            let narration = state.take_narration();
            if !narration.is_empty() {
                print!("{narration}");
                if io::stdout().flush().is_err() {
                    problems.push("could not flush the record narration".to_string());
                }
            }
            for line in &state.trouble {
                eprint!("{line}");
            }
        }
        Ok(Err(e)) => problems.push(format!("the full-screen interface failed: {e:#}")),
        Err(e) => problems.push(format!("the record frame ended without reporting: {e}")),
    }
    if handle.join().is_err() {
        problems.push("the record frame thread panicked".to_string());
    }
    problems
}

/// Tears the interface down and settles what it owes the user.
///
/// Called once per [`Screen`], after the run's loop has finished; the problems the settle met
/// become the run's error. A run that ends some other way never gets here and drops instead --
/// see [`Screen`]'s `Drop` -- which settles the same way and says its problems on stderr.
#[cfg(target_os = "macos")]
pub(crate) fn close(mut screen: Screen) -> Result<()> {
    let stop = screen.stop.take().expect("a screen settles once");
    let done = screen.done.take().expect("a screen settles once");
    let handle = screen.handle.take().expect("a screen settles once");
    let problems = settle(stop, done, handle);
    anyhow::ensure!(problems.is_empty(), "{}", problems.join("; "));
    Ok(())
}

#[cfg(target_os = "macos")]
impl Drop for Screen {
    /// The abnormal-exit guarantee: a run that errored out or panicked between the frame
    /// coming up and the loop finishing leaves this struct unsettled, and the terminal must
    /// come back regardless. Settle best-effort and say the problems on stderr -- the failure
    /// the user is reading is the run's own, and these add to it rather than replace it.
    fn drop(&mut self) {
        if let (Some(stop), Some(done), Some(handle)) =
            (self.stop.take(), self.done.take(), self.handle.take())
        {
            for problem in settle(stop, done, handle) {
                eprintln!("{problem}");
            }
        }
    }
}

/// The frame thread's whole job: acquire the terminal lazily, draw until told to stop, restore
/// the terminal unconditionally, and hand back whatever the notes accumulated.
#[cfg(target_os = "macos")]
fn frame(
    tx: mpsc::Sender<Event>,
    notes: mpsc::Receiver<Note>,
    stop: mpsc::Receiver<()>,
    paths: Paths,
) -> Result<State> {
    // Lazily, and only here: the presenter was chosen because both streams were attached when
    // the run started, but the terminal is actually opened now, on this thread. If it cannot
    // be opened, nothing has been written into it yet -- there is no partial escape sequence
    // to undo -- so the failure simply propagates and the run reports it at teardown.
    let mut term =
        ratatui::try_init().with_context(|| "could not open the full-screen interface")?;
    let mut state = State::new(paths);

    let mut dirty = true;
    // When the phase entered recording, so the clock can be drawn without the state machine
    // holding a clock: the frame derives elapsed time from this, and clears it the moment the
    // phase leaves recording.
    let mut recording_since: Option<Instant> = None;

    let result = run(
        &mut term,
        &tx,
        &notes,
        &stop,
        &mut state,
        &mut dirty,
        &mut recording_since,
    );

    // Unconditional, and ordered: the alternate screen goes away whether the loop ended
    // cleanly or hit an error, and only then is the outcome reported. Reaching here means
    // `try_init` succeeded -- a frame that never acquired the terminal returned before this,
    // and calling `restore` one would write `LeaveAlternateScreen` into a terminal it never
    // entered.
    ratatui::restore();
    result.map(|_| state)
}

/// Draws until the stop signal lands.
///
/// One iteration is: wake on a key (bounded by [`TICK`]), drain whatever notes queued since
/// the last draw, and redraw if anything changed or the clock is moving. Nothing here blocks
/// longer than [`TICK`], which is what keeps Ctrl-C responsive while the run is idle.
#[cfg(target_os = "macos")]
fn run(
    term: &mut Terminal<CrosstermBackend<io::Stdout>>,
    tx: &mpsc::Sender<Event>,
    notes: &mpsc::Receiver<Note>,
    stop: &mpsc::Receiver<()>,
    state: &mut State,
    dirty: &mut bool,
    recording_since: &mut Option<Instant>,
) -> Result<()> {
    loop {
        if stop.try_recv().is_ok() {
            break;
        }

        match poll(TICK) {
            // A single `read()`, not a loop: `poll` returning `Ok(true)` is the guarantee that
            // this one read will not block, and only that one -- `read()` itself always blocks
            // until an event exists, so a second call here would stall the frame (no redraws,
            // no note draining, no stop detection) until another key happened to arrive. Any
            // further buffered event is picked up on the loop's next iteration instead, which
            // still runs immediately, since `poll` finds it ready right away.
            Ok(true) => {
                if let Ok(CrosstermEvent::Key(key)) = read() {
                    // One context at a time, innermost first: editing beats the roster pane,
                    // which beats the selector, which beats the base. The panes are mutually
                    // exclusive in the state machine and editing only ever begins from the
                    // roster pane, so the derivation is total without focus bookkeeping.
                    let ctx = match state.editing {
                        Some(field) => KeyContext::RosterEditing(field),
                        None if state.roster_open => KeyContext::Roster,
                        None if state.selector_open => KeyContext::Selector,
                        None => KeyContext::Base,
                    };
                    match event(key, ctx) {
                        Some(Action::Interrupt) => {
                            // Raw mode swallows SIGINT, so the frame delivers the
                            // interrupt through the same channel the ctrlc handler uses
                            // in plain mode: the main loop finalizes the session exactly
                            // as it would have.
                            let _ = tx.send(Event::Interrupt);
                        }
                        Some(action) => {
                            // The selector's commands enter the state machine as typed
                            // commands, the way `enroll` feeds answers: the shell decides
                            // nothing about them, it only hands them over and marks the
                            // frame dirty so the cursor move is drawn.
                            match action {
                                Action::OpenSelector => state.open_selector(),
                                Action::Next => state.next(),
                                Action::Previous => state.previous(),
                                Action::CloseSelector => state.close_selector(),
                                Action::Confirm => {
                                    // The identifier, not the meeting: the run resolves
                                    // it against the list it handed over, and a pick it
                                    // cannot resolve settles nothing.
                                    if let Some(event_id) = state.confirm() {
                                        let _ = tx.send(Event::MeetingPicked(event_id));
                                    }
                                }
                                Action::OpenRoster => state.open_roster(),
                                Action::CloseRoster => state.close_roster(),
                                Action::RosterNext => state.roster_next(),
                                Action::RosterPrevious => state.roster_previous(),
                                Action::RemoveAttendee => {
                                    // The full edited roster crosses, addressed by the
                                    // meeting's own identifier; the run validates it
                                    // against the meetings it handed over and stashes the
                                    // last for the single finalize point.
                                    if let Some(edit) = state.remove_selected() {
                                        let _ = tx.send(Event::RosterEdited(edit));
                                    }
                                }
                                Action::EditName => state.begin_edit(EditingField::Name),
                                Action::EditEmail => state.begin_edit(EditingField::Email),
                                Action::Type(c) => state.feed_edit(c),
                                Action::DeleteChar => state.backspace_edit(),
                                Action::CommitField => {
                                    if let Some(edit) = state.commit_edit() {
                                        let _ = tx.send(Event::RosterEdited(edit));
                                    }
                                }
                                Action::CancelField => state.cancel_edit(),
                                // The stop never reaches this arm: the interrupt arm
                                // above decides Ctrl-C before the selector commands get
                                // a look at the key.
                                Action::Interrupt => {
                                    unreachable!("Ctrl-C is decided by the interrupt arm")
                                }
                            }
                            *dirty = true;
                        }
                        None => {}
                    }
                }
            }
            Ok(false) => {}
            Err(e) => {
                // The terminal went unreadable under us: deliver the interrupt so the run
                // finalizes and exits rather than recording forever into a dead tty, then
                // report the failure at teardown.
                let _ = tx.send(Event::Interrupt);
                return Err(anyhow::anyhow!("the terminal stopped being readable: {e}"));
            }
        }

        let mut got_note = false;
        while let Ok(note) = notes.try_recv() {
            state.apply(&note);
            got_note = true;
        }

        *recording_since = match state.phase {
            Phase::Recording => Some(*recording_since.get_or_insert(Instant::now())),
            _ => None,
        };
        let elapsed = recording_since.map(|since| since.elapsed());

        if *dirty || got_note || state.phase == Phase::Recording {
            term.draw(|f| render::draw(f, state, elapsed))?;
            *dirty = false;
        }
    }

    // One last drain after the stop: the run sends the stop only after it has finished saying
    // everything, so whatever is still queued is the tail of the run and belongs in the
    // narration the teardown will flush.
    while let Ok(note) = notes.try_recv() {
        state.apply(&note);
    }
    Ok(())
}

/// What a keypress means to the frame.
///
/// The stop is the base TUI's only binding; the rest is TASK-056.01's meeting selector, and
/// they grow this table rather than the loop that reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Stop the run and exit: the main loop finalizes the session exactly as a plain-mode
    /// Ctrl-C would have.
    Interrupt,
    /// Open the meeting selector: the numbered offers replace the notice region while it is
    /// open. Allowed even with an empty list -- that is the degraded view, and the frame says
    /// nothing is offered rather than hiding the key.
    OpenSelector,
    /// Move the cursor down one row, wrapping.
    Next,
    /// Move the cursor up one row, wrapping.
    Previous,
    /// Confirm the offer under the cursor: its identifier crosses into the run as an
    /// [`Event::MeetingPicked`]. The selector stays open until the run's settlement says the
    /// pick stuck -- a pick the run drops leaves the user looking at the list rather than a
    /// false confirmation.
    Confirm,
    /// Close the selector without picking.
    CloseSelector,
    /// Open the roster pane: the attached meeting's attendees replace the notice region while
    /// it is open. A no-op in the state machine when no meeting is attached, which is why the
    /// hint line advertises the key only while a roster is on screen.
    OpenRoster,
    /// Close the roster pane without committing anything; any in-flight correction dies with
    /// it.
    CloseRoster,
    /// Move the roster cursor down one row, wrapping.
    RosterNext,
    /// Move the roster cursor up one row, wrapping.
    RosterPrevious,
    /// Remove the attendee under the cursor: the full edited roster crosses into the run as
    /// an [`Event::RosterEdited`], addressed by the meeting's own identifier.
    RemoveAttendee,
    /// Begin correcting the selected row's name inline.
    EditName,
    /// Begin correcting the selected row's email inline.
    EditEmail,
    /// Feed one character into the field under correction. Bound only in the editing context:
    /// printables are consumed there and nowhere else, so roster input can never drive the
    /// selector or a background action.
    Type(char),
    /// Delete the last character of the field under correction.
    DeleteChar,
    /// Commit the field under correction: the full edited roster crosses as an
    /// [`Event::RosterEdited`].
    CommitField,
    /// Cancel the field under correction: the buffer is dropped locally and nothing crosses.
    CancelField,
}

/// The context a keypress lands in, derived from the state the shell holds: editing beats
/// the roster pane, which beats the selector, which beats the base. One context at a time is
/// what keeps the table below total without focus bookkeeping -- the panes are mutually
/// exclusive in the state machine, and editing only ever begins from the roster pane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyContext {
    Base,
    Selector,
    Roster,
    RosterEditing(EditingField),
}

/// The key map, as a free function of `(key, context)` because a `KeyEvent` is constructible
/// without a terminal -- which is what makes this rule testable, the way `enroll`'s `event`
/// is -- and because Enter genuinely means three things depending on context: opening the
/// question when there is no list on screen, choosing from it when there is, and committing
/// a field while one is being corrected. It is also where a stray Ctrl or a release-shaped
/// duplicate would otherwise go wrong.
fn event(key: KeyEvent, ctx: KeyContext) -> Option<Action> {
    // A release is not a press. Terminals that report both would otherwise act on every key
    // twice.
    if key.kind == KeyEventKind::Release {
        return None;
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    // The control-modifier check comes first, exactly as the base TUI had it: a Ctrl-prefixed
    // key is the stop regardless of what any pane is doing.
    if ctrl {
        // Ctrl-C and Ctrl-D are both stopping, the way `enroll`'s interface reads them: raw
        // mode means no SIGINT arrives, so the keys are the interrupt, and end-of-input is
        // stopping rather than failing. A real terminal delivers the control byte (lowercase),
        // but the pre-interface handler read either case, so both are kept.
        return match key.code {
            KeyCode::Char('c' | 'C' | 'd' | 'D') => Some(Action::Interrupt),
            _ => None,
        };
    }
    match (key.code, ctx) {
        // The cursor moves in every list context: the selector's when it is open, the
        // roster's when the roster pane is open. In the base it moves the row a later
        // selector open will land on; movement on an empty or closed list is the state
        // machine's no-op, not the table's refusal.
        (KeyCode::Up, KeyContext::Base | KeyContext::Selector) => Some(Action::Previous),
        (KeyCode::Down, KeyContext::Base | KeyContext::Selector) => Some(Action::Next),
        (KeyCode::Up, KeyContext::Roster) => Some(Action::RosterPrevious),
        (KeyCode::Down, KeyContext::Roster) => Some(Action::RosterNext),
        // Contextual: with no list on screen Enter opens the selector; with one it chooses
        // from it; while a field is being corrected it commits. In the roster pane itself it
        // does nothing -- the pane has no confirm action, only its row operations.
        (KeyCode::Enter, KeyContext::Base) => Some(Action::OpenSelector),
        (KeyCode::Enter, KeyContext::Selector) => Some(Action::Confirm),
        (KeyCode::Enter, KeyContext::RosterEditing(_)) => Some(Action::CommitField),
        // Esc unwinds one level: the field being corrected, else the pane it sits in. With
        // none open it has nothing to close, and a key that cannot work is not acted on.
        (KeyCode::Esc, KeyContext::RosterEditing(_)) => Some(Action::CancelField),
        (KeyCode::Esc, KeyContext::Selector) => Some(Action::CloseSelector),
        (KeyCode::Esc, KeyContext::Roster) => Some(Action::CloseRoster),
        // The roster pane's toggle: bound wherever the pane might be opened or closed, and a
        // no-op in the state machine when there is no attachment to open it on.
        (KeyCode::Char('r'), KeyContext::Base) => Some(Action::OpenRoster),
        (KeyCode::Char('r'), KeyContext::Roster) => Some(Action::CloseRoster),
        // The roster pane's row operations, bound only while the pane is open.
        (KeyCode::Char('x'), KeyContext::Roster) => Some(Action::RemoveAttendee),
        (KeyCode::Char('n'), KeyContext::Roster) => Some(Action::EditName),
        (KeyCode::Char('e'), KeyContext::Roster) => Some(Action::EditEmail),
        // The editing context consumes printables and Backspace, and nothing else: the guard
        // keeps control characters out of the buffer the same way enroll's search input keeps
        // them out of its filter, and any other key falls through to unbound -- so a key that
        // is not part of the correction can never drive the selector or a background action.
        (KeyCode::Char(c), KeyContext::RosterEditing(_)) if !c.is_control() => {
            Some(Action::Type(c))
        }
        (KeyCode::Backspace, KeyContext::RosterEditing(_)) => Some(Action::DeleteChar),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    /// A constructed keypress, the way a terminal would deliver it: no terminal involved.
    fn press(code: KeyCode, kind: KeyEventKind, ctrl: bool) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: if ctrl {
                KeyModifiers::CONTROL
            } else {
                KeyModifiers::NONE
            },
            kind,
            state: KeyEventState::NONE,
        }
    }

    /// The stop keeps its precedence in every context, and only a press of it counts:
    /// Ctrl-C and Ctrl-D interrupt even while a list or a field correction is in flight --
    /// a pick or an edit in progress must not keep a Ctrl-C from ending the run.
    #[test]
    fn the_stop_keeps_its_precedence_in_every_context() {
        let contexts = [
            KeyContext::Base,
            KeyContext::Selector,
            KeyContext::Roster,
            KeyContext::RosterEditing(EditingField::Name),
            KeyContext::RosterEditing(EditingField::Email),
        ];
        for ctx in contexts {
            for letter in ['c', 'C', 'd', 'D'] {
                assert_eq!(
                    event(press(KeyCode::Char(letter), KeyEventKind::Press, true), ctx),
                    Some(Action::Interrupt),
                    "Ctrl-{letter} on a press stops the run in {ctx:?}"
                );
                assert_eq!(
                    event(
                        press(KeyCode::Char(letter), KeyEventKind::Release, true),
                        ctx,
                    ),
                    None,
                    "a release of Ctrl-{letter} does nothing in {ctx:?}"
                );
            }
        }
    }

    /// Every other bound key means what the footer advertises in that context, and only a
    /// press of it: Enter opens the list when there is none on screen, chooses from it when
    /// there is, and commits a field while one is being corrected; up and down move the
    /// cursor of whichever pane is open; Esc unwinds one level.
    #[test]
    fn every_bound_key_means_what_the_footer_says_in_its_context() {
        let base = |code: KeyCode| event(press(code, KeyEventKind::Press, false), KeyContext::Base);
        let selector = |code: KeyCode| {
            event(
                press(code, KeyEventKind::Press, false),
                KeyContext::Selector,
            )
        };
        let roster =
            |code: KeyCode| event(press(code, KeyEventKind::Press, false), KeyContext::Roster);

        // The contextual keys: the same physical key, different meanings.
        assert_eq!(base(KeyCode::Enter), Some(Action::OpenSelector));
        assert_eq!(selector(KeyCode::Enter), Some(Action::Confirm));
        assert_eq!(
            roster(KeyCode::Enter),
            None,
            "the roster pane has no confirm action"
        );

        // The cursor moves in every list context, toward whatever list is on screen.
        assert_eq!(base(KeyCode::Up), Some(Action::Previous));
        assert_eq!(base(KeyCode::Down), Some(Action::Next));
        assert_eq!(selector(KeyCode::Up), Some(Action::Previous));
        assert_eq!(selector(KeyCode::Down), Some(Action::Next));
        assert_eq!(roster(KeyCode::Up), Some(Action::RosterPrevious));
        assert_eq!(roster(KeyCode::Down), Some(Action::RosterNext));

        // Esc has something to close only where a pane is on screen.
        assert_eq!(selector(KeyCode::Esc), Some(Action::CloseSelector));
        assert_eq!(roster(KeyCode::Esc), Some(Action::CloseRoster));
        assert_eq!(base(KeyCode::Esc), None, "nothing is open to close");

        // The roster toggle and row operations are bound wherever they can work.
        assert_eq!(base(KeyCode::Char('r')), Some(Action::OpenRoster));
        assert_eq!(roster(KeyCode::Char('r')), Some(Action::CloseRoster));
        assert_eq!(roster(KeyCode::Char('x')), Some(Action::RemoveAttendee));
        assert_eq!(roster(KeyCode::Char('n')), Some(Action::EditName));
        assert_eq!(roster(KeyCode::Char('e')), Some(Action::EditEmail));

        // Releases of the navigation and pane keys do nothing, in any context.
        for code in [KeyCode::Enter, KeyCode::Up, KeyCode::Down, KeyCode::Esc] {
            for ctx in [KeyContext::Base, KeyContext::Selector, KeyContext::Roster] {
                assert_eq!(
                    event(press(code, KeyEventKind::Release, false), ctx),
                    None,
                    "a release of {code:?} does nothing in {ctx:?}"
                );
            }
        }

        // Unbound keys are ignored outside their context, and a control-modified navigation
        // key is not a navigation key: the modifier check comes first.
        for code in [KeyCode::Tab, KeyCode::Left, KeyCode::Right] {
            for ctx in [KeyContext::Base, KeyContext::Selector, KeyContext::Roster] {
                assert_eq!(
                    event(press(code, KeyEventKind::Press, false), ctx),
                    None,
                    "unbound {code:?} does nothing in {ctx:?}"
                );
            }
        }
        assert_eq!(
            event(
                press(KeyCode::Up, KeyEventKind::Press, true),
                KeyContext::Roster
            ),
            None,
            "Ctrl-Up is not the cursor"
        );
    }

    /// Printables feed ONLY the editing context: the enroll search-input partitioning, scoped
    /// to a context instead of permanent. In every other context a printable stays unbound,
    /// so roster input can never drive the selector or a background action; and inside the
    /// editing context a control character is refused rather than buffered.
    #[test]
    fn printables_feed_only_the_editing_context() {
        for ctx in [KeyContext::Base, KeyContext::Selector, KeyContext::Roster] {
            assert_eq!(
                event(press(KeyCode::Char('g'), KeyEventKind::Press, false), ctx),
                None,
                "a printable is unbound outside editing ({ctx:?})"
            );
        }
        for field in [EditingField::Name, EditingField::Email] {
            let ctx = KeyContext::RosterEditing(field);
            for c in ['g', '.', '@', ' '] {
                assert_eq!(
                    event(press(KeyCode::Char(c), KeyEventKind::Press, false), ctx),
                    Some(Action::Type(c)),
                    "{c:?} feeds the {field:?} field"
                );
            }
            assert_eq!(
                event(press(KeyCode::Backspace, KeyEventKind::Press, false), ctx),
                Some(Action::DeleteChar)
            );
            // Control characters are refused in the buffer, exactly as enroll refuses them
            // in its filter.
            for c in ['\u{1}', '\t', '\r'] {
                assert_eq!(
                    event(press(KeyCode::Char(c), KeyEventKind::Press, false), ctx),
                    None,
                    "control {c:?} does not enter the field"
                );
            }
            // A release of a printable does nothing either.
            assert_eq!(
                event(press(KeyCode::Char('g'), KeyEventKind::Release, false), ctx),
                None
            );
        }
    }

    /// The roster pane's letters are unbound outside the pane: x, n and e do nothing in the
    /// base and selector contexts, so correcting a name cannot be triggered by typing while
    /// choosing a meeting.
    #[test]
    fn roster_row_keys_are_unbound_outside_the_roster_pane() {
        for code in [KeyCode::Char('x'), KeyCode::Char('n'), KeyCode::Char('e')] {
            assert_eq!(
                event(press(code, KeyEventKind::Press, false), KeyContext::Base),
                None,
                "{code:?} is unbound in the base"
            );
            assert_eq!(
                event(
                    press(code, KeyEventKind::Press, false),
                    KeyContext::Selector
                ),
                None,
                "{code:?} is unbound in the selector"
            );
        }
    }
}
