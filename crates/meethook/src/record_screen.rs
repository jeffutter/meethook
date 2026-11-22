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
//! as an [`Event::MeetingPicked`], a committed roster correction as an [`Event::RosterEdited`],
//! and its `s` key -- the one event that changes which session is live rather than describing it
//! -- as an [`Event::StopSession`], each decided at the same single reader as a mic edge. What
//! keeps the contract intact is that the payloads are values the run resolves against the lists it
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

mod keys;
mod render;
mod state;

// The frame/run/settle machinery below drives a real terminal (spawns a thread, opens the
// alternate screen), so unlike `state` and `render` it has no platform-neutral unit-test path
// -- its only production caller is `record::Sink`, which is macOS-only. Gated to match that
// caller rather than left to the module's broader `any(macos, test)` gate, or a Linux `cargo
// clippy --all-targets` finds every item below dead.
#[cfg(target_os = "macos")]
use std::io::{self, IsTerminal};
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
use crate::record::{Event, Narration, Note, Reporter};
#[cfg(target_os = "macos")]
use crate::tui_kit::{HelpOutcome, HelpOverlay, Theme};
#[cfg(target_os = "macos")]
use keys::{Action, Keys};
#[cfg(target_os = "macos")]
use state::{EditingField, Phase, State};

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
    /// `tx` is the clone the frame uses to deliver its own events -- Ctrl-C as
    /// [`Event::Interrupt`], its `s` key as [`Event::StopSession`], and the calendar pick or
    /// roster correction it commits; the original sender stays with the run's ctrlc handler,
    /// which remains the plain-mode path.
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
///
/// Both streams are [`Narration`]s, which is what lets this happen *after* the run has already
/// finalized the session: closing the terminal window hangs up stdout while the frame is still
/// settling, and bytes nobody can read are not a reason to report a saved meeting as a failure,
/// nor to panic on the way out. A stream that refused is therefore silent from then on and
/// contributes no problem -- the complaint could only be said on the same dead terminal anyway.
/// A frame that failed and a frame thread that panicked stay problems exactly as they were.
#[cfg(target_os = "macos")]
fn settle(
    stop: mpsc::Sender<()>,
    done: mpsc::Receiver<Result<State>>,
    handle: thread::JoinHandle<()>,
    out: &mut Narration,
    err: &mut Narration,
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
                out.say(&narration);
            }
            for line in &state.trouble {
                err.say(line);
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
    // Built here rather than held by the screen: the teardown paths never coexist, so each one
    // gets its own latch and a lost line on one stream says nothing about the other.
    let (mut out, mut err) = (Narration::stdout(), Narration::stderr());
    let problems = settle(stop, done, handle, &mut out, &mut err);
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
            let (mut out, mut err) = (Narration::stdout(), Narration::stderr());
            for problem in settle(stop, done, handle, &mut out, &mut err) {
                err.say(&format!("{problem}\n"));
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
    // The `?` overlay, open or not. Loop-local: while it is open it takes every key except the
    // interrupt, and it lives exactly as long as the frame does.
    let keys = Keys::new(Theme::detect(io::stdout().is_terminal(), |name| {
        std::env::var(name).ok()
    }));
    let mut help: Option<HelpOverlay> = None;
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
                    // Ctrl-C and Ctrl-D are decided before the overlay sees the key: an open
                    // help screen must never swallow the interrupt.
                    let action = keys.dispatch(state, &key);
                    if help.is_some() && action != Some(Action::Interrupt) {
                        let rows = keys.help_rows(state);
                        let page = term
                            .size()
                            .map(|size| HelpOverlay::page_height(size.into(), rows.len()))
                            .unwrap_or(1);
                        let overlay = help.as_mut().expect("checked open above");
                        if overlay.handle_key(key, rows.len(), page) == HelpOutcome::Close {
                            help = None;
                        }
                        *dirty = true;
                    } else {
                        match action {
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
                                    Action::Top => state.first(),
                                    Action::Bottom => state.last(),
                                    Action::Help => help = Some(HelpOverlay::default()),
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
                                    Action::RosterTop => state.roster_first(),
                                    Action::RosterBottom => state.roster_last(),
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
                                    Action::StopSession => {
                                        // Only a live session can be ended; the run decides what
                                        // that means and says so. A stray press with nothing
                                        // recording is dropped by the loop's idle arm as well, so
                                        // the frame gate saves the channel an event rather than
                                        // preventing a fault.
                                        if state.can_stop_session() {
                                            let _ = tx.send(Event::StopSession);
                                        }
                                    }
                                    Action::ClearMeeting => {
                                        // The removal rides to the run as an event; the frame
                                        // changes when the run's note comes back, so a removal
                                        // the run drops never reads as one. Gated on a live
                                        // session only -- not on offers or a guess, since it must
                                        // work with an empty selector and no calendar.
                                        if state.can_clear_meeting() {
                                            let _ = tx.send(Event::MeetingCleared);
                                        }
                                    }
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
        // Recomposed on every draw from the held inputs, so the figure ticks with the clock
        // while the plain run prints snapshots: same composer, different cadence. The frame
        // leaves the stop advice off because its footer already carries the keys.
        let overrun = match (state.phase, &state.overrun) {
            (Phase::Recording, Some(view)) => meethook_session::overrun_past(
                view.end,
                jiff::Timestamp::now(),
                crate::record::OVERRUN_GRACE,
            )
            .map(|over| crate::record::overrun_lines(&view.title, over, &view.holders, None)),
            _ => None,
        };

        if *dirty || got_note || state.phase == Phase::Recording {
            term.draw(|f| {
                render::draw(f, state, &keys, help.as_mut(), elapsed, overrun.as_deref())
            })?;
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
