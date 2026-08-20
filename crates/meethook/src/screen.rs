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
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::Duration;

use meethook_enroll::{Answer, Consequence, Interviewer, Preview, Scan, Snippet, Voice, speech};
use meethook_session::{Displaced, Paths, Stored};
use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{
    Event as Key, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, poll, read,
};

use crate::commands::{Clips, Progress};
use state::{Context, Cost, Costs, Event, Screen, Step, VoiceView};

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
/// decidable in `cargo test` with no terminal in front of it. An idle frame has nothing that would
/// redraw differently, so it blocks rather than waking four times a second to find that out.
///
/// Two things put a clock on this frame, and each only while it lasts. Playback is one: the
/// position moves and then disappears. An outstanding [`Background`] scan is the other, and
/// without it the "who is this" pane would sit on `reading the sessions...` until the user
/// happened to press a key. That scan is bounded at well under a second, so the
/// idle-frame-blocks property survives everywhere it mattered: a frame with nothing playing and
/// nothing outstanding still waits for its next key.
fn wait(playing: Option<Progress>, scanning: bool) -> Option<Duration> {
    (playing.is_some() || scanning).then_some(TICK)
}

/// What reading every session's speakers *is*, so a test can supply one that counts or one that
/// fails.
///
/// An [`Arc`] rather than a `Box` because each scan's thread needs its own handle on it: [`Paths`]
/// is cloned per scan and [`Scan`] is plain data, so both cross the boundary, and flattening the
/// error to a `String` there means [`Background`] carries no `meethook_enroll::Error` and the frame
/// has nothing to propagate.
type Scanner = Arc<dyn Fn(&Paths) -> Result<Scan, String> + Send + Sync>;

/// The cross-session scan, gathered off the thread that draws frames.
///
/// One scan of the whole root is 0.47 s over 53 sessions and 49 references on the machine this was
/// built against -- dominated by the JSON reads rather than by the dot products, and linear in
/// sessions times references. That single measurement decides the whole shape here, in both
/// directions: far too slow to pay per keystroke or to pay before the first frame appears, and far
/// too fast to be worth progress reporting, cancellation, incremental delivery or a scan that
/// reads only the sessions it needs. So: one scan on one thread, delivered whole, polled from the
/// event loop exactly the way [`Clips`] already is.
///
/// The `scanner` indirection is the seam that makes "not once per keystroke" decidable in
/// `cargo test` -- the same move [`Costs`] is, for the same reason -- so a test can hand over a
/// counting closure and assert what a hundred keys cost.
struct Background {
    scanner: Scanner,
    paths: Paths,
    /// The last scan delivered, `None` until the first one arrives.
    latest: Option<Result<Scan, String>>,
    /// The one in flight, if any. At most one ever is.
    pending: Option<Receiver<Result<Scan, String>>>,
    /// Whether an answer has moved the database since the in-flight scan started, and so whether
    /// another is owed once it lands.
    stale: bool,
}

impl Background {
    /// A scanner over the real root, with nothing read yet.
    fn new(paths: Paths) -> Background {
        Background {
            scanner: Arc::new(|paths| meethook_enroll::scan(paths).map_err(|e| e.to_string())),
            paths,
            latest: None,
            pending: None,
            stale: false,
        }
    }

    /// Starts a scan unless one is already in flight, in which case the one in flight will be
    /// followed by another when it lands.
    ///
    /// The thread is detached and never joined: if the frame goes away the [`Receiver`] drops, the
    /// send fails, and the thread ends on its own.
    fn start(&mut self) {
        if self.pending.is_some() {
            return;
        }
        let (deliver, delivered) = mpsc::channel();
        let scanner = Arc::clone(&self.scanner);
        let paths = self.paths.clone();
        thread::spawn(move || {
            // A receiver that has gone away is a frame that has finished, which is not a failure.
            let _ = deliver.send(scanner(&paths));
        });
        self.pending = Some(delivered);
        // Whatever this scan finds is being read now, so it already includes every answer written
        // before this moment.
        self.stale = false;
    }

    /// Takes delivery of a finished scan, and says whether one is still outstanding -- which is
    /// what [`wait`] needs in order to keep the frame awake until the pane can fill in.
    fn poll(&mut self) -> bool {
        if let Some(pending) = self.pending.as_ref() {
            match pending.try_recv() {
                Ok(found) => {
                    self.latest = Some(found);
                    self.pending = None;
                    if self.stale {
                        self.start();
                    }
                }
                Err(TryRecvError::Empty) => {}
                // The thread ended without sending, which needs a panic inside `scan` to happen
                // at all. Nothing is outstanding and whatever was last read still stands.
                Err(TryRecvError::Disconnected) => self.pending = None,
            }
        }
        self.pending.is_some()
    }

    /// Says the database has moved, so the scan on the screen is about to be one answer behind.
    ///
    /// Re-scans rather than labelling the pane "as this run began", because the candidate rows
    /// already show a live reference count off `Resemblance`: a stale count in the pane beside a
    /// live one in the list is a bug a reviewer would file. Between the answer and the next
    /// delivery the pane shows the previous scan, which is old rather than false -- nothing it
    /// says claims to be "now" -- and it converges within about half a second.
    fn invalidate(&mut self) {
        self.stale = true;
        self.start();
    }

    /// What the pane has to go on: nothing yet, the last scan, or why there is none.
    fn context(&self) -> Context<'_> {
        match &self.latest {
            None => Context::Reading,
            Some(Ok(found)) => Context::Read(found),
            Some(Err(why)) => Context::Failed(why),
        }
    }
}

/// What is sounding right now, as the frame needs to draw it.
///
/// One value rather than a progress and a line index side by side: the index is meaningless
/// unless something is playing, and `Option<Sounding>` is that sentence in the type. `progress`
/// is the player's report ([`Progress`]); `line` is the shell's own record of which transcript
/// line it handed over, `None` for the whole-voice clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sounding {
    pub progress: Progress,
    pub line: Option<usize>,
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
    /// What every enrolled name currently names, gathered off this thread.
    background: Background,
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
    /// A frame sharing `narration` with the run's [`Lines`](meethook_enroll::Lines), over the
    /// root the run itself is reading.
    ///
    /// `paths` is here rather than inside the state machine because the scan behind the "who is
    /// this" pane is I/O, and [`state`] is documented as having none.
    pub fn new(narration: Shared, paths: Paths) -> Interface {
        Interface {
            state: Screen::default(),
            narration,
            clips: Clips::default(),
            background: Background::new(paths),
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
            // Lazily and here for the same reason the terminal is: a run that passes over every
            // session should do no work at all, and this is the first point known not to be such
            // a run. The first frame draws before it lands, saying so in the pane.
            self.background.start();
        }

        // One call site, and one stop after it: answered, skipped, deferred, quit and a frame that
        // could not be drawn are five ways out of the loop, and every one of them has to leave the
        // audio behind. `Drop for Clips` covers the paths that never come back through here.
        let answer = self.ask(&view, voice);
        self.clips.stop();
        if rewrites(&answer) {
            self.background.invalidate();
        }
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
            background,
            terminal,
            trouble,
        } = self;
        let Some(terminal) = terminal.as_mut() else {
            return Answer::Quit;
        };

        // Which transcript line the last successful spawn came from, `None` for the whole voice.
        // A loop-local rather than a field because "what was handed to `afplay`" is a fact about
        // a spawn, and this loop is exactly as long as a spawn is allowed to live: `clips.stop()`
        // runs the moment `ask` returns.
        let mut line: Option<usize> = None;

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
            // Beside the clip poll, and for the same reason: what the frame draws is whatever has
            // arrived by the time it draws it.
            let scanning = background.poll();
            {
                let derived = state.view(view, &voice.preview, background.context());
                let lines = narration.lines();
                // `playing` is the poll's answer, so a clip that has finished takes the row's
                // mark with it without anything having to clear `line`.
                let sounding = playing.map(|progress| Sounding { progress, line });
                if let Err(e) =
                    terminal.draw(|frame| render::draw(frame, &derived, &lines, sounding))
                {
                    *trouble = Some(format!("the frame could not be drawn ({e})"));
                    return Answer::Quit;
                }
            }

            // While something is playing, waking on a timeout is what moves the position and what
            // takes it away when the clip ends; while a scan is outstanding it is what fills the
            // "who is this" pane in without a key being pressed. The redraw at the top of the loop
            // does all three. With neither, this falls straight through and blocks in `read`.
            if let Some(timeout) = wait(playing, scanning) {
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
            // deliberately holds no audio. Two keys, two sources of samples, and the same
            // say/hush tail -- deliberately not merged, because they differ in every other line:
            // where the samples come from, all three sentences, and whether a line was played.
            if event == Event::Play {
                let problem = if voice.clip.is_empty() {
                    Some("there is no audio for this voice".to_string())
                } else {
                    match clips.start(voice.clip) {
                        // The whole voice, so no row is the one that is sounding. A stale index
                        // here would mark a line that is not what is being heard.
                        Ok(()) => {
                            line = None;
                            None
                        }
                        Err(e) => Some(format!("could not play the clip: {e}")),
                    }
                };
                match problem {
                    Some(problem) => state.say(problem),
                    // A play that has now worked takes back what the last failed one said.
                    // Nothing else clears the footer for a key the state machine never sees.
                    None => state.hush(),
                }
                continue;
            }
            if event == Event::PlaySnippet {
                let problem = match line_to_play(state.selected(view)) {
                    Err(sentence) => Some(sentence),
                    Ok((index, audio)) => match clips.start(audio) {
                        Ok(()) => {
                            line = Some(index);
                            None
                        }
                        Err(e) => Some(format!("could not play that line: {e}")),
                    },
                };
                match problem {
                    Some(problem) => state.say(problem),
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

/// Whether an answer can have moved what the "who is this" pane reports, and so whether the scan
/// behind it is now one answer behind.
///
/// A function beside [`wait`] and [`event`], for the same reason: a rule about which of five
/// answers costs a re-scan is then decidable in `cargo test` with no terminal in front of it.
///
/// A name is the only answer that writes to `speakers.json`; the other four write nothing at all.
/// A `Named` can still be refused by the veto and write nothing either, so this over-triggers --
/// deliberately, because an extra background scan nobody waits for is cheaper than a wrong number
/// on the screen.
fn rewrites(answer: &Answer) -> bool {
    matches!(answer, Answer::Named(_))
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

/// What the play-the-line key should do about the selection it found: hand these samples over, or
/// say this instead.
///
/// A function beside [`event`] rather than three cases inside the loop, and for the same reason:
/// which sentence the key produces is then decidable in `cargo test`, with no terminal and no
/// audio device. The middle case is the one that needs it most, because it cannot be seen from
/// the outside at all -- [`Clips::start`] treats no samples as a successful no-op, so a line whose
/// `audio` is empty has to be refused *here* or the key silently does nothing while the frame says
/// nothing about it.
fn line_to_play(selected: Option<(usize, Snippet<'_>)>) -> Result<(usize, &[f32]), String> {
    match selected {
        None => Err("nothing was transcribed for this voice".to_string()),
        Some((_, snippet)) if snippet.audio.is_empty() => {
            Err("there is no audio for that line".to_string())
        }
        Some((index, snippet)) => Ok((index, snippet.audio)),
    }
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
/// Two play keys, because there are two things to hear: Ctrl-P is the whole voice and Ctrl-L is
/// the selected transcript line ("l for line"). Ctrl-L is conventionally "redraw", which this
/// frame does at the top of every iteration anyway, so nothing is being displaced.
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
        (KeyCode::Char('l'), true) => Some(Event::PlaySnippet),
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, mpsc};
    use std::thread;
    use std::time::Duration;

    use meethook_enroll::{Answer, Scan};
    use meethook_session::Paths;
    use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    use super::state::tests::heard;
    use super::state::{Context, Event};
    use super::{
        Background, Interface, Progress, Shared, TICK, event, line_to_play, rewrites, wait,
    };

    fn key(code: KeyCode, modifiers: KeyModifiers) -> KeyEvent {
        KeyEvent {
            code,
            modifiers,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    /// A scan that costs nothing and counts how often it was asked for. The root is never read,
    /// which is the point: what is being measured is how many times the frame asks.
    fn counting() -> (Background, Arc<AtomicUsize>) {
        let asked = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&asked);
        let mut background = Background::new(Paths::new("/nowhere"));
        background.scanner = Arc::new(move |_| {
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(Scan::default())
        });
        (background, asked)
    }

    /// A scan that does not finish until the test lets it, so "while one is in flight" is a state
    /// a test can actually be in rather than a race it hopes to win.
    fn gated() -> (mpsc::Sender<()>, Background, Arc<AtomicUsize>) {
        let (release, released) = mpsc::channel::<()>();
        let held = Mutex::new(released);
        let asked = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&asked);
        let mut background = Background::new(Paths::new("/nowhere"));
        background.scanner = Arc::new(move |_| {
            // A dropped sender is a test that has finished with this thread.
            let _ = held
                .lock()
                .expect("no scan panics while holding this")
                .recv();
            counted.fetch_add(1, Ordering::SeqCst);
            Ok(Scan::default())
        });
        (release, background, asked)
    }

    /// Polls until something becomes true, the way a real frame polls: at the top of an iteration
    /// it was woken for. A test cannot join a thread the design deliberately detaches, so the
    /// alternative to a bounded wait is no assertion at all.
    fn until(what: &str, mut settled: impl FnMut() -> bool) {
        for _ in 0..2_000 {
            if settled() {
                return;
            }
            thread::sleep(Duration::from_millis(1));
        }
        panic!("{what}");
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
            (
                KeyCode::Char('l'),
                KeyModifiers::CONTROL,
                Event::PlaySnippet,
            ),
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

    /// AC #4, the no-stall half: an outstanding scan wakes the frame, so the "who is this" pane
    /// fills in on its own rather than waiting for the user to press something. And an idle frame
    /// -- nothing playing, nothing outstanding -- still blocks for its next key rather than waking
    /// four times a second to redraw what cannot have changed. All four combinations, because the
    /// two clocks are independent and either one alone has to be enough.
    ///
    /// The bounds on `TICK` are what stop a later edit turning the rule into a spin at one end or a
    /// position that visibly lurches at the other.
    #[test]
    fn an_outstanding_scan_wakes_the_frame_and_an_idle_one_still_blocks() {
        let playing = Progress {
            elapsed: Duration::from_secs(3),
            length: Duration::from_secs(30),
        };
        assert_eq!(wait(None, false), None, "nothing to redraw, so block");
        assert_eq!(wait(Some(playing), false), Some(TICK));
        assert_eq!(
            wait(None, true),
            Some(TICK),
            "the pane has to fill in without a key being pressed"
        );
        assert_eq!(wait(Some(playing), true), Some(TICK));
        assert!(
            TICK >= Duration::from_millis(50) && TICK <= Duration::from_millis(500),
            "{TICK:?} is either a spin or a position that jumps"
        );
    }

    /// AC #4, the not-per-keystroke half. One scan of the real root is around half a second, so
    /// "the frame asks once and then polls" is the whole reason the pane is affordable at all --
    /// and the only way to assert it is to count how often the scan was asked for.
    #[test]
    fn the_scan_runs_once_and_not_once_per_key() {
        let (mut background, asked) = counting();
        background.start();
        // A hundred keys, each with the poll a real loop does at the top of its iteration.
        for _ in 0..100 {
            background.poll();
        }
        until("the scan never arrived", || !background.poll());
        assert_eq!(
            asked.load(Ordering::SeqCst),
            1,
            "one scan for a hundred keys"
        );

        // And a hundred more keys after it landed ask for nothing further.
        for _ in 0..100 {
            background.poll();
        }
        assert_eq!(asked.load(Ordering::SeqCst), 1);
    }

    /// Which answers cost a re-read. A name is the only answer that writes to `speakers.json`, so
    /// it is the only one that can move what the pane says -- and a deferral, a skip or a quit
    /// re-reading the whole root would be half a second of work per keypress for nothing.
    #[test]
    fn an_answer_re_reads_the_sessions_and_a_skip_does_not() {
        assert!(rewrites(&Answer::Named("Milo".to_string())));
        for quiet in [Answer::Skip, Answer::Later, Answer::Quit] {
            assert!(!rewrites(&quiet), "{quiet:?} writes nothing");
        }

        let (mut background, asked) = counting();
        background.start();
        until("the scan never arrived", || !background.poll());
        background.invalidate();
        until("the re-scan never arrived", || !background.poll());
        assert_eq!(asked.load(Ordering::SeqCst), 2, "the answer moved the file");
    }

    /// At most one scan is ever in flight, however many answers land while one is running. Two
    /// invalidations during a scan owe exactly one re-scan -- not two, and not none.
    #[test]
    fn only_one_scan_is_ever_outstanding() {
        let (release, mut background, asked) = gated();
        background.start();
        background.invalidate();
        background.invalidate();
        assert!(background.poll(), "the first scan is still running");
        assert_eq!(asked.load(Ordering::SeqCst), 0, "and has not been let go");

        release.send(()).expect("the scanner is waiting on this");
        until("the first scan never landed", || {
            background.poll();
            background.latest.is_some()
        });
        assert!(
            background.pending.is_some(),
            "two answers during one scan owe one more"
        );

        release.send(()).expect("the re-scan is waiting on this");
        until("the re-scan never landed", || {
            background.poll();
            background.pending.is_none()
        });
        assert_eq!(
            asked.load(Ordering::SeqCst),
            2,
            "two invalidations, one re-scan"
        );
    }

    /// AC #5 for the database rather than for one session: a scan that fails at all leaves the
    /// pane saying so and the frame still answering questions, because every answer already given
    /// is on disk and stopping the run would not put any of it back.
    #[test]
    fn a_scan_that_fails_leaves_the_frame_answering() {
        let mut background = Background::new(Paths::new("/nowhere"));
        background.scanner = Arc::new(|_| Err("speakers.json is not valid JSON".to_string()));
        background.start();
        until("the failure never arrived", || !background.poll());

        match background.context() {
            Context::Failed(why) => assert_eq!(why, "speakers.json is not valid JSON"),
            other => panic!("a failed scan is a failed context, not {other:?}"),
        }
    }

    /// AC #5: a line with nothing behind it says so rather than appearing to play. Both refusals
    /// have to happen before the spawn -- `Clips::start` would treat either as a success and leave
    /// the key looking like it did nothing at all.
    #[test]
    fn a_line_with_no_audio_says_so_rather_than_appearing_to_play() {
        static AUDIO: [f32; 3] = [0.1, 0.2, 0.3];
        let line = heard("right, next week", 107.0, &AUDIO);
        assert_eq!(line_to_play(Some((1, line))), Ok((1, &AUDIO[..])));

        let silent = heard("right, next week", 107.0, &[]);
        assert_eq!(
            line_to_play(Some((1, silent))),
            Err("there is no audio for that line".to_string())
        );
        assert_eq!(
            line_to_play(None),
            Err("nothing was transcribed for this voice".to_string())
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

        let mut frame = Interface::new(narration, Paths::new("/nowhere"));
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
