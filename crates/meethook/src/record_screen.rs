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
//! session exactly as it would have.
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

use std::io::{self, Write};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use meethook_session::Paths;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::Event as CrosstermEvent;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers, poll, read};

use crate::record::{Event, Note, Reporter};
use state::{Phase, State};

/// How often the frame wakes when nothing else is waiting on it.
///
/// Bounded rather than blocking, on purpose: while idle the key poll is the only thing that
/// waits, and it must wait at most this long, or a Ctrl-C would sit unread until the next note
/// happened to arrive. Two hundred fifty milliseconds is invisible in a redraw and short enough
/// that the interface still feels like it is listening.
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
pub(crate) struct Screen {
    notes: mpsc::Sender<Note>,
    stop: Option<mpsc::Sender<()>>,
    done: Option<mpsc::Receiver<Result<State>>>,
    handle: Option<thread::JoinHandle<()>>,
}

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
pub(crate) fn close(mut screen: Screen) -> Result<()> {
    let stop = screen.stop.take().expect("a screen settles once");
    let done = screen.done.take().expect("a screen settles once");
    let handle = screen.handle.take().expect("a screen settles once");
    let problems = settle(stop, done, handle);
    anyhow::ensure!(problems.is_empty(), "{}", problems.join("; "));
    Ok(())
}

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
            Ok(true) => {
                while let Ok(crossterm_event) = read() {
                    if let CrosstermEvent::Key(key) = crossterm_event
                        && event(key) == Some(Action::Interrupt)
                    {
                        // Raw mode swallows SIGINT, so the frame delivers the interrupt
                        // through the same channel the ctrlc handler uses in plain mode:
                        // the main loop finalizes the session exactly as it would have.
                        let _ = tx.send(Event::Interrupt);
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

/// What a keypress means to the frame. One action today; the base TUI has no navigation, and
/// TASK-056.01 grows this table rather than the loop that reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Action {
    /// Stop the run and exit: the main loop finalizes the session exactly as a plain-mode
    /// Ctrl-C would have.
    Interrupt,
}

/// The key map, as a free function taking a [`KeyEvent`] because a `KeyEvent` is constructible
/// without a terminal -- which is what makes this rule testable, the way `enroll`'s `event`
/// is. It is also where a stray Ctrl or a release-shaped duplicate would otherwise go wrong.
fn event(key: KeyEvent) -> Option<Action> {
    // A release is not a press. Terminals that report both would otherwise act on every key
    // twice.
    if key.kind == KeyEventKind::Release {
        return None;
    }
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    match (key.code, ctrl) {
        // Ctrl-C and Ctrl-D are both stopping, the way `enroll`'s interface reads them: raw
        // mode means no SIGINT arrives, so the keys are the interrupt, and end-of-input is
        // stopping rather than failing. A real terminal delivers the control byte (lowercase),
        // but the pre-interface handler read either case, so both are kept.
        (KeyCode::Char('c' | 'C' | 'd' | 'D'), true) => Some(Action::Interrupt),
        // Everything else is ignored: the base TUI binds nothing but the stop, and a key that
        // cannot work is not acted on.
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

    /// Every bound key means what the footer says, and only a press of it: Ctrl-C and Ctrl-D,
    /// either case, stop the run; their releases and every unbound key do nothing.
    #[test]
    fn every_bound_key_means_what_the_footer_says() {
        for letter in ['c', 'C', 'd', 'D'] {
            assert_eq!(
                event(press(KeyCode::Char(letter), KeyEventKind::Press, true)),
                Some(Action::Interrupt),
                "Ctrl-{letter} on a press stops the run"
            );
            assert_eq!(
                event(press(KeyCode::Char(letter), KeyEventKind::Release, true)),
                None,
                "a release of Ctrl-{letter} does nothing"
            );
        }

        // Unbound keys are ignored, including the ones a future navigation layer will want:
        // acting on them now would make the base TUI do something it does not advertise.
        for code in [
            KeyCode::Char('x'),
            KeyCode::Enter,
            KeyCode::Up,
            KeyCode::Down,
            KeyCode::Tab,
        ] {
            assert_eq!(
                event(press(code, KeyEventKind::Press, false)),
                None,
                "unbound {code:?} does nothing"
            );
        }
    }
}
