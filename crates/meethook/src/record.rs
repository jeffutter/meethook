//! Detect calls on the default microphone and sequence one session per call.
//!
//! The sequencing here is platform-neutral on purpose -- it is what makes the record loop
//! testable without a microphone -- but on non-macOS builds the only production caller of
//! any of it is macOS's `record`, so each item compiles for tests there rather than sitting
//! dead in the binary and tripping the no-dead-code gate.

// The channel types exist only where the record loop does: on Linux production builds the
// loop is excluded by its `any(macos, test)` gate, and an import nothing names is a warning.
// The bare `mpsc` path is narrower still -- its one production use is inside `record`, which
// is macOS-only, so even the test build leaves it unused off macOS.
#[cfg(target_os = "macos")]
use std::sync::mpsc;
#[cfg(any(target_os = "macos", test))]
use std::sync::mpsc::{Receiver, RecvTimeoutError};
#[cfg(any(target_os = "macos", test))]
use std::time::{Duration, Instant};

#[cfg(any(target_os = "macos", test))]
use std::fmt;
#[cfg(any(target_os = "macos", test))]
use std::path::PathBuf;

#[cfg(any(target_os = "macos", test))]
use crate::commands::Tty;
#[cfg(target_os = "macos")]
use anyhow::Context;
#[cfg(any(target_os = "macos", test))]
use anyhow::Result;
#[cfg(any(target_os = "macos", test))]
use meethook_enroll::{MeetingLabel, MeetingOffer};
#[cfg(target_os = "macos")]
use meethook_session::Paths;
#[cfg(any(target_os = "macos", test))]
use meethook_session::SessionId;
#[cfg(any(target_os = "macos", test))]
use meethook_session::{Attendee, Meeting, MeetingFit, RosterEdit};
// The capture backend exists only where its Apple frameworks compile; the platform-neutral
// sequencing in `record_loop` below stays ungated and keeps testing without it.
#[cfg(target_os = "macos")]
use meethook_record::{
    Activity, MicActivityWatcher, Recorder, RunningSession, meetings_for, preflight,
};

#[cfg(any(target_os = "macos", test))]
/// The four waits the record loop's behaviour depends on.
///
/// Gathered into one value so the sequencing can be exercised at millisecond scale. The live
/// figures would make the loop's tests take half a minute, and a slow test is one nobody
/// runs.
#[derive(Debug, Clone, Copy)]
struct Timing {
    /// How long microphone activity must stay stopped before a session is finalized.
    ///
    /// Asymmetric with the start side, which has no debounce at all, on purpose: three
    /// seconds of extra tail audio is harmless, while a premature finalize loses the end of
    /// a meeting and a late start loses its opening.
    grace: Duration,
    /// How long to wait between attempts at a session start that failed.
    retry: Duration,
    /// How many attempts one call gets before it is abandoned.
    ///
    /// Bounded because the failure may be permanent -- a revoked permission, a display that
    /// has gone away -- and an unbounded retry would spin for the length of the meeting.
    attempts: u32,
    /// How often the world is re-examined *while a session is live*. Two questions are asked
    /// on this interval, and it is the detection mechanism for only one of them.
    ///
    /// The activity level is a safety net: every edge is expected to arrive from a listener,
    /// and the re-check exists only because a release edge can be lost outright when the
    /// recomputation behind a notification reads the world a moment too early. See
    /// `MicActivityWatcher::recheck`.
    ///
    /// The microphone's liveness has no listener at all, so this interval *is* how a dead
    /// capture engine is found. `RunningSession::mic_stalled` declares a stall once the mic
    /// track's frame count has stood still for its own limit -- one second at 48 kHz, longer
    /// on a device that delivers more slowly -- and since that limit is shorter than this
    /// interval, the effective test at 48 kHz is "not one buffer arrived across a whole
    /// sampling interval", about 23 consecutive missed callbacks. Detection therefore lands
    /// 0-2 s after the engine dies. The rule is expressed as a duration rather than as "one
    /// non-advancing sample" so that it stays meaningful if this cadence changes, and so a
    /// slow device gets its extra margin automatically.
    ///
    /// Nothing is polled while idle.
    recheck: Duration,
}

#[cfg(any(target_os = "macos", test))]
impl Timing {
    // The live figures are what `record` runs on; the tests run on `LOOP_TIMING` instead, so
    // off macOS -- where `record` does not compile -- this constant has no user at all.
    #[cfg(target_os = "macos")]
    const LIVE: Timing = Timing {
        grace: Duration::from_secs(3),
        retry: Duration::from_secs(2),
        attempts: 5,
        // Two seconds of re-check plus the three-second grace is the worst-case stop
        // latency after a lost edge, against extra tail audio being harmless (see `grace`).
        // The cost while recording is one walk of the process objects every two seconds.
        recheck: Duration::from_secs(2),
    };
}

#[cfg(any(target_os = "macos", test))]
/// Anything the record loop waits on.
///
/// One enum, one channel: a Ctrl-C during a recording has to be seen at the same instant
/// as a microphone edge, and two separate waits cannot both be blocking. `pub(crate)` because
/// the full-screen frame is a second producer of these -- it delivers its Ctrl-C and its
/// calendar picks through a clone of this channel's sender; the event set itself stays private
/// to this module, and every match site below owes a deliberate answer per variant.
pub(crate) enum Event {
    Started,
    Stopped,
    /// The default input device moved. Not an edge of the activity predicate: it says the
    /// microphone engine is now bound to the wrong device, which every wait below has to have
    /// a deliberate answer for.
    InputDeviceChanged,
    Interrupt,
    /// The user confirmed one of the calendar offers the frame was showing, addressed by the
    /// event's own identifier rather than its title: titles repeat across calendars, and the
    /// identifier is what `finish` writes. The first payload-carrying variant -- kept a
    /// `String`, so the channel's contract (one reader, events decided at the loop) is
    /// unchanged in kind, and no `Meeting` crosses it.
    MeetingPicked(String),
    /// The user committed a correction to the attached meeting's roster in the frame's roster
    /// pane, riding the same addressing rule [`Event::MeetingPicked`] sets: the session
    /// crate's own [`RosterEdit`], identified by the event id the frame was shown, resolved
    /// against the held list at the loop. It carries the FULL edited roster rather than a
    /// delta -- last write wins, and merging partial changes across the channel would invent
    /// merge bugs the run does not need. No `Meeting` crosses it either: the invite content
    /// stays out of the frame's reach by construction.
    RosterEdited(RosterEdit),
}

#[cfg(any(target_os = "macos", test))]
/// The calendar's answer for the session being recorded.
///
/// Every meeting worth offering as a hand correction, and the one the automatic rule would
/// attach (with the fit that rule decided) -- or nothing, when there is no live session or no
/// calendar. Total rather than fallible, exactly as the lookup behind it is: a missing grant
/// or an empty calendar is an empty answer, and the selector degrades to "no meetings" rather
/// than erroring. The full values stay on the main thread where a pick resolves; what crosses
/// into the frame are projections of them.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Offered {
    /// Every meeting worth offering, in the listing's stable order.
    meetings: Vec<Meeting>,
    /// The one the automatic rule would attach, with the fit that rule decided.
    chosen: Option<Meeting>,
}

#[cfg(any(target_os = "macos", test))]
/// What the record loop needs from a capture backend.
///
/// The loop's whole responsibility is sequencing -- when to open a session, when to hold on
/// through a blip, when to finalize, when to give up -- and none of that needs a microphone
/// to decide. This three-method seam is what makes it decidable in `cargo test`: the live
/// implementation drives a [`Recorder`], and the test one records the order it was called
/// in. Everything either of them knows about audio stays on their side of it.
///
/// Each method takes the run's note sink rather than printing: the backend is where the
/// session's wording happens (its id, its rates, what a finish produced), and the sink is
/// what decides whether those words land on a terminal or in a frame. Passing it in rather
/// than storing it keeps the backend free of any reference into the run's presentation.
trait Capture {
    /// Begins a session and announces it.
    fn start(&mut self, sink: &mut dyn Reporter) -> Result<()>;
    /// The calendar's answer for the session being recorded, asked once per session start.
    fn candidates(&mut self) -> Offered;
    /// Finalizes the current session and reports what it produced.
    ///
    /// `hand` is the pick the user settled while recording, if any, and `roster_edit` the
    /// roster correction the frame committed, if any. Both are applied here, at the single
    /// finalize point, rather than written mid-flight: `session.json` does not exist until
    /// this write, and its presence is the completion marker every other process reads.
    fn finish(
        &mut self,
        sink: &mut dyn Reporter,
        hand: Option<Meeting>,
        roster_edit: Option<RosterEdit>,
    ) -> Result<()>;
    /// Whether the microphone track has stopped receiving audio.
    ///
    /// Asked only while a session is live, once per `Timing::recheck`. The live backend
    /// answers from the track's own delivered frame count, so it catches every way an input
    /// tap can go quiet rather than only the ones that post a notification; nothing here
    /// needs a microphone to decide what the answer means.
    fn mic_stalled(&mut self, sink: &mut dyn Reporter) -> bool;
}

#[cfg(any(target_os = "macos", test))]
/// Which presentation a run gets.
///
/// Two rather than three, because `record` has no up-front answerer the way `enroll` has
/// `--name`: every line it prints is decided by the loop itself, so the only question is
/// whether those lines land on a real terminal or in a buffer somebody is capturing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Presenter {
    /// The line-based output: what piped, redirected and CI runs have always gotten.
    Lines,
    /// The full-screen interface.
    Screen,
}

/// Which presenter a run gets, given `--plain` and what the streams are attached to.
///
/// The order is the rule, and it is deliberate:
///
/// 1. A pipe on *either* end is the line output. Both streams, not just one: the plain lines
///    go to stdout and the interface reads keys from stdin, so a run being driven -- by CI, by
///    a shell pipeline, by a subprocess -- must not write escape sequences into a captured
///    buffer and must not wait for a keypress a script cannot send.
/// 2. `--plain` is the explicit override, for somebody on a real terminal who wants the lines
///    back, or who needs the interface out of a reproduction.
///
/// A function rather than an inline `if`, for exactly the reason `answerer` and `meeting_line`
/// are functions: the rule is then decidable in `cargo test` with no terminal in front of it,
/// and "a driven run can never accidentally open a full-screen UI" stays independently
/// testable. Called once at the top of [`record`], before any prompt or print.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn presenter(plain: bool, tty: Tty) -> Presenter {
    if plain || !tty.is_attached() {
        Presenter::Lines
    } else {
        Presenter::Screen
    }
}

/// One thing the record run has to say, as the loop produces it.
///
/// Every user-visible message the run prints -- session start, device-change and mic-stall
/// notices, the finish summary, the give-up line -- flows through this type rather than a
/// `println!` at the point it happens. That is what lets the plain and the full-screen
/// presenters share one source of truth for wording: the composers below produce today's exact
/// literals, and both presenters render them.
///
/// The stream class is encoded in the variant itself rather than passed alongside, so that an
/// exhaustive match forces a deliberate stream choice per note -- the same discipline the
/// [`Recording`] enum applies to outcomes. Stdout-class notes describe the run to the user; in
/// screen mode they also feed the narration buffer that restores scrollback parity. Stderr-class
/// ones report faults and developer diagnostics; in screen mode they surface in-frame instead
/// of drawing over the alternate screen.
///
/// The string-carrying variants carry the composed sentence rather than the error or problem
/// value it came from: the calendar-problem text is static guidance that names no person, and
/// the errors are flattened at the point they happen because a `String` is what a notice pane
/// and a scrollback line alike can hold.
///
/// Three frame-only notes ride the same type with nothing to print: `MeetingOffered`,
/// `MeetingSettled` and `RosterAttached` carry data the interface renders, and composing
/// them to the empty string is what keeps a plain run silent about the calendar -- headless
/// output stays byte-identical to the pre-interface binary.
#[cfg(any(target_os = "macos", test))]
pub(crate) enum Note {
    // ---- stdout class ----
    /// The calendar grant is missing: guidance, not an error, printed once at startup.
    CalendarProblem(String),
    /// The loop is idle, waiting for a call.
    Watching,
    /// A call was already in progress when the process started, so recording began at once.
    AlreadyActive,
    /// A session opened: its id, directory, and the rates both engines actually came up at.
    SessionStarted {
        id: SessionId,
        dir: PathBuf,
        mic_rate: u32,
        mic_channels: u32,
        speaker_rate: u32,
    },
    /// The default input device moved mid-session; the old one is being finalized.
    DeviceChanged,
    /// The microphone track stopped delivering audio; the session is being finalized.
    MicStalled,
    /// The session is ending on purpose: the call ended or the user interrupted.
    Stopping,
    /// A finalize-and-restart found the call was already over, so nothing new was opened.
    NoNewSession,
    /// The calendar's view of the session just opened: the numbered offers and the guess among
    /// them, projected so far as a frame may show them.
    ///
    /// Composes to nothing -- see the type doc -- and is sent once per session start, even
    /// when both halves are empty, so a restarted session cannot inherit its predecessor's
    /// list.
    MeetingOffered {
        offered: Vec<MeetingOffer>,
        guess: Option<MeetingLabel>,
    },
    /// A hand pick settled: which offer the user confirmed, in the same projection the offers
    /// crossed in. Carries `MeetingFit::Confirmed` -- what `label_by_hand` will write at
    /// finish -- never the candidate's own fit, whose caveat would qualify a pick a human just
    /// made. Composes to nothing for the same reason [`Note::MeetingOffered`] does.
    MeetingSettled { label: MeetingLabel },
    /// The attached meeting's roster, for the frame's roster pane: name, email, status and
    /// `is_you` per attendee -- the [`Attendee`] fields are exactly the disclosure unit, and
    /// nothing more crosses. The invite content (`notes`, `location`, `url`) lives on the
    /// `Meeting`, which still never crosses this seam, so it stays out of the frame by
    /// construction (decision-008, as amended for the roster pane).
    ///
    /// Sent whenever the attachment changes: once at session start when the automatic rule
    /// chose a meeting, and again when a hand pick settles one -- the settled meeting
    /// supersedes whatever the frame was shown, and the frame replaces its copy wholesale.
    /// Composes to nothing for the same reason [`Note::MeetingOffered`] does: a plain run
    /// says nothing about the calendar's people.
    RosterAttached {
        event_id: String,
        attendees: Vec<Attendee>,
    },
    /// A session finished: what it produced, and the meeting it was matched to if any.
    ///
    /// The meeting crosses as a [`MeetingLabel`] -- title and fit only -- never as the full
    /// [`meethook_session::Meeting`]. Attendee names and addresses are written to
    /// `session.json` for speaker identification and are deliberately never printed, and the
    /// projection is what makes that a property of the type rather than a review of format
    /// strings.
    Recorded {
        id: SessionId,
        mic_secs: f64,
        speaker_secs: f64,
        dir: PathBuf,
        meeting: Option<MeetingLabel>,
    },

    // ---- stderr class ----
    /// The first failed attempt to open a session, in full.
    BeginFailed(String),
    /// A session that could not be finalized.
    FinishFailed(String),
    /// The start retry is exhausted; the loop goes back to watching.
    ///
    /// Stderr rather than stdout because the pre-interface run said it with `eprintln!`: the
    /// give-up is a fault report about the recorder's own machinery, not narration of the
    /// meeting, and AC #4 pins it to the stream it always had.
    GivingUp(u32),
    /// Developer diagnostics, gated at the call site on `MEETHOOK_ACTIVITY_DEBUG`.
    ActivityDebug(String),
}

#[cfg(any(target_os = "macos", test))]
impl Note {
    /// This note's exact text, trailing newline included, as the line-based run prints it.
    ///
    /// Multi-line notes compose their whole block here -- `SessionStarted`'s five lines,
    /// `Recorded`'s summary plus its conditional meeting clause -- so a presenter that prints
    /// one string reproduces the byte sequence of the separate `println!` calls it replaced,
    /// in the same order. `pub(crate)` because the full-screen state machine stores the
    /// composed text verbatim rather than re-deriving it.
    pub(crate) fn composed(&self) -> String {
        match self {
            Note::CalendarProblem(problem)
            | Note::BeginFailed(problem)
            | Note::FinishFailed(problem)
            | Note::ActivityDebug(problem) => format!("{problem}\n"),
            Note::Watching => format!("{WATCHING}\n"),
            Note::AlreadyActive => format!("{ALREADY_ACTIVE}\n"),
            Note::SessionStarted {
                id,
                dir,
                mic_rate,
                mic_channels,
                speaker_rate,
            } => session_started_lines(id, dir.display(), *mic_rate, *mic_channels, *speaker_rate),
            Note::DeviceChanged => format!("{DEVICE_CHANGED}\n"),
            Note::MicStalled => format!("{MIC_STALLED}\n"),
            Note::Stopping => format!("{STOPPING}\n"),
            Note::NoNewSession => format!("{NO_NEW_SESSION}\n"),
            // Frame-only: data for the interface, not words for a terminal. Composing them
            // away is what keeps the line-based run byte-identical.
            Note::MeetingOffered { .. }
            | Note::MeetingSettled { .. }
            | Note::RosterAttached { .. } => String::new(),
            Note::Recorded {
                id,
                mic_secs,
                speaker_secs,
                dir,
                meeting,
            } => recorded_lines(
                id,
                *mic_secs,
                *speaker_secs,
                dir.display(),
                meeting.as_ref(),
            ),
            Note::GivingUp(attempts) => giving_up_line(*attempts),
        }
    }

    /// Whether this note goes to stderr: the faults and the developer diagnostics, and only
    /// those. Everything the user is meant to read lands on stdout, which is the stream a
    /// full-screen frame would fight over and the one a pipe captures. `pub(crate)` because
    /// the full-screen state machine routes each note by its stream class rather than
    /// re-deciding it.
    pub(crate) fn to_stderr(&self) -> bool {
        matches!(
            self,
            Note::BeginFailed(_)
                | Note::FinishFailed(_)
                | Note::GivingUp(_)
                | Note::ActivityDebug(_)
        )
    }
}

/// The lines the run says while idle and between sessions.
///
/// Constants rather than inline formats because both presenters need them verbatim: the plain
/// one prints them and the full-screen state machine stores them as notices, and one spelling
/// is what keeps the frame and the scrollback from describing the same condition differently.
#[cfg(any(target_os = "macos", test))]
pub(crate) const WATCHING: &str = "Watching the default microphone. Press Ctrl-C to stop.";
#[cfg(any(target_os = "macos", test))]
pub(crate) const ALREADY_ACTIVE: &str = "A microphone is already in use; recording immediately.";
#[cfg(any(target_os = "macos", test))]
pub(crate) const RECORDING_PROMPT: &str = "Recording... press Ctrl-C to stop.";
#[cfg(any(target_os = "macos", test))]
pub(crate) const DEVICE_CHANGED: &str = "The default input device changed. The microphone engine is bound to the device that went away, so this session is being finalized and a new one opened on the new device.";
#[cfg(any(target_os = "macos", test))]
pub(crate) const MIC_STALLED: &str = "The microphone stopped delivering audio, so this session is being finalized and a new one opened. (The device did not change; something reconfigured or took the input, which stops the recording engine without any notice that it happened.)";
#[cfg(any(target_os = "macos", test))]
pub(crate) const STOPPING: &str = "Stopping...";
#[cfg(any(target_os = "macos", test))]
pub(crate) const NO_NEW_SESSION: &str =
    "That call has ended as well, so no new session was opened.";

/// The roster pane's degraded view: a meeting is attached but its invite lists nobody.
///
/// A separate literal from the empty-selector sentence -- which itself has a separate
/// plain-mode twin the `meeting` command owns: the two panes say different things (no
/// meeting found, versus a meeting found with nobody on it), and pinning each in its own
/// pane's tests keeps them from drifting into each other's words.
#[cfg(any(target_os = "macos", test))]
pub(crate) const NO_ROSTER: &str =
    "This meeting lists no attendees, so there is no roster to correct";

/// The line naming a session as it opens.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn session_id_line(id: impl fmt::Display) -> String {
    format!("Session {id}")
}

/// The line naming a session's directory.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn session_dir_line(dir: impl fmt::Display) -> String {
    format!("  {dir}")
}

/// The line reporting the microphone engine's rate and channel count.
///
/// The padding aligns `mic` and `speaker` into one column in the plain output; the full-screen
/// pane trims it away, since a bordered pane supplies its own margin.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn mic_line(rate: u32, channels: u32) -> String {
    format!("  mic       {rate} Hz, {channels} channel(s) reported by the input device")
}

/// The line reporting the speaker engine's rate.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn speaker_line(rate: u32) -> String {
    format!("  speaker   {rate} Hz")
}

/// The whole block a session opening prints: id, directory, both rates, and the prompt.
///
/// Printing both rates proves both engines actually came up; a user who sees only one line
/// knows something is wrong before the meeting rather than after it.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn session_started_lines(
    id: impl fmt::Display,
    dir: impl fmt::Display,
    mic_rate: u32,
    mic_channels: u32,
    speaker_rate: u32,
) -> String {
    format!(
        "{}\n{}\n{}\n{}\n{}\n",
        session_id_line(id),
        session_dir_line(dir),
        mic_line(mic_rate, mic_channels),
        speaker_line(speaker_rate),
        RECORDING_PROMPT,
    )
}

/// The whole block a finished session prints: the summary line, and the meeting clause if a
/// meeting was attached.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn recorded_lines(
    id: impl fmt::Display,
    mic_secs: f64,
    speaker_secs: f64,
    dir: impl fmt::Display,
    meeting: Option<&MeetingLabel>,
) -> String {
    let mut lines =
        format!("Recorded {id} ({mic_secs:.1}s mic, {speaker_secs:.1}s speaker) to {dir}\n");
    if let Some(meeting) = meeting {
        lines.push_str(&meeting_clause_line(meeting));
        lines.push('\n');
    }
    lines
}

/// The give-up line, with the attempt count the retry actually burned.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn giving_up_line(attempts: u32) -> String {
    format!("Giving up on this call after {attempts} attempts; still watching.\n")
}

/// Where a note goes.
///
/// The seam the loop writes into instead of a stream: the plain implementation reproduces the
/// pre-interface binary byte for byte, and the screen-mode one feeds the state machine and the
/// narration buffer. The loop's sequencing never learns which it holds.
#[cfg(any(target_os = "macos", test))]
pub(crate) trait Reporter {
    fn note(&mut self, note: Note);
}

/// The line-based reporter: each note composed and printed to its own stream.
///
/// Printing the composed block with `print!` rather than `println!` is byte-identical -- the
/// newline is in the block -- and it is what lets a multi-line note stay one write per note,
/// in the same order the old `println!`s ran.
#[cfg(target_os = "macos")]
pub(crate) struct Plain;

#[cfg(target_os = "macos")]
impl Reporter for Plain {
    fn note(&mut self, note: Note) {
        let text = note.composed();
        if note.to_stderr() {
            eprint!("{text}");
        } else {
            print!("{text}");
        }
    }
}

/// The two reporters [`record`] can hand the loop, behind one trait object.
#[cfg(target_os = "macos")]
enum Sink {
    Lines(Plain),
    Screen(crate::record_screen::Screen),
}

#[cfg(target_os = "macos")]
impl Reporter for Sink {
    fn note(&mut self, note: Note) {
        match self {
            Sink::Lines(plain) => plain.note(note),
            Sink::Screen(screen) => screen.note(note),
        }
    }
}

/// The live backend: one session at a time, plus the user-facing report of it.
#[cfg(target_os = "macos")]
struct SessionCapture<'a> {
    recorder: &'a Recorder,
    paths: &'a Paths,
    debug: bool,
    running: Option<RunningSession>,
}

#[cfg(target_os = "macos")]
impl Capture for SessionCapture<'_> {
    fn start(&mut self, sink: &mut dyn Reporter) -> Result<()> {
        let started_at = Instant::now();
        let session = self.recorder.start(self.paths, &jiff::Zoned::now())?;
        if self.debug {
            // This latency sits directly on the "no debounce, a late start loses the
            // opening" path, so it is worth being able to see rather than assume.
            sink.note(Note::ActivityDebug(format!(
                "[activity] Recorder::start took {:.1} ms",
                started_at.elapsed().as_secs_f64() * 1000.0
            )));
        }

        // Printing both rates proves both engines actually came up; a user who sees only
        // one line knows something is wrong before the meeting rather than after it.
        sink.note(Note::SessionStarted {
            id: session.id().clone(),
            dir: session.paths().dir().to_path_buf(),
            mic_rate: session.mic_sample_rate(),
            mic_channels: session.mic_channels(),
            speaker_rate: session.speaker_sample_rate(),
        });

        self.running = Some(session);
        Ok(())
    }

    fn candidates(&mut self) -> Offered {
        // Asked against the session's own start -- the moment a recording began, not now --
        // because that is the instant decision-009 says identifies the meeting. The query is
        // a pure read; the calendar *ask* happened once at process start, so asking mid-call
        // costs a brief main-thread pause rather than a prompt, and the capture engines run
        // on their own threads.
        let Some(session) = self.running.as_ref() else {
            return Offered::default();
        };
        let lookup = meetings_for(session.start_time());
        Offered {
            meetings: lookup.offered,
            chosen: lookup.chosen,
        }
    }

    fn finish(
        &mut self,
        sink: &mut dyn Reporter,
        hand: Option<Meeting>,
        roster_edit: Option<RosterEdit>,
    ) -> Result<()> {
        // Nothing running is not an error. The loop only finishes a start it saw succeed, so
        // defining the case away here is cheaper than a branch that can only ever be wrong.
        let Some(session) = self.running.take() else {
            return Ok(());
        };
        let recording = session.finish(hand, roster_edit)?;
        sink.note(Note::Recorded {
            id: recording.id.clone(),
            mic_secs: recording.mic.seconds(),
            speaker_secs: recording.speaker.seconds(),
            dir: recording.paths.dir().to_path_buf(),
            // Projected at the point the full meeting exists: from here on the run only ever
            // holds title and fit, which is what makes "nothing sensitive crosses" a property
            // of the type.
            meeting: recording.metadata.meeting.as_ref().map(MeetingLabel::from),
        });
        Ok(())
    }

    fn mic_stalled(&mut self, sink: &mut dyn Reporter) -> bool {
        // Nothing running cannot be stalled. The loop only asks while a session is live, so
        // defining the case away here is cheaper than a branch that can only ever be wrong --
        // the same reading `finish` above takes.
        let Some(session) = self.running.as_mut() else {
            return false;
        };
        let stalled = session.mic_stalled();
        if self.debug {
            // The only evidence a hardware run will have for what the counter was doing, so
            // it is printed on every re-check rather than only when it trips.
            sink.note(Note::ActivityDebug(format!(
                "[activity] mic delivered {} frames{}",
                session.mic_frames_delivered(),
                if stalled { "  <- STALLED" } else { "" }
            )));
        }
        stalled
    }
}

/// Records every call until the process is interrupted.
///
/// The two required permissions are checked first and separately, so a missing TCC grant costs
/// the user an error message rather than a silently unrecorded meeting. Calendar access is
/// asked for immediately afterwards and is *not* required: it only decides whether a session
/// can be named after the meeting it was recorded during, so a refusal prints guidance and the
/// recorder carries on.
///
/// The loop is deliberately forgiving of a failed session: a two-second false start that
/// produces a silent track prints an error and goes back to watching. Ending a day of
/// recording over one bad session would be a far worse failure than the session itself.
///
/// Everything past the setup lives in [`record_loop`], which is where the sequencing is and
/// where it can be tested without a microphone.
#[cfg(target_os = "macos")]
pub fn record(paths: &Paths, plain: bool) -> Result<()> {
    // Decided first, before `preflight` and before anything session-specific: a driven run
    // must never open the interface even when it aborts in preflight, so the decision cannot
    // wait for the first line the run would print.
    let presenter = presenter(plain, Tty::current());

    let authorized = preflight()?;
    let recorder = Recorder::new(authorized)?;

    let debug = std::env::var_os("MEETHOOK_ACTIVITY_DEBUG").is_some();

    let (tx, rx) = mpsc::channel::<Event>();

    // The screen sink takes a clone of the sender because raw mode means no SIGINT arrives and
    // the frame thread has to deliver Ctrl-C itself through the channel; the original stays
    // with the ctrlc handler below, which remains the plain-mode path and the out-of-band
    // fallback. A single Receiver either way: cloning Senders does not split the stream.
    let mut sink = match presenter {
        Presenter::Lines => Sink::Lines(Plain),
        Presenter::Screen => {
            Sink::Screen(crate::record_screen::Screen::new(tx.clone(), paths.clone()))
        }
    };

    // After `preflight` on purpose: a run about to abort for a missing microphone grant must
    // not first ask for an optional one, and the essential prompts should arrive before it.
    // Before the watcher on purpose too: nothing is being watched, and nothing can be
    // recorded, while this prompt is up.
    if let Some(problem) = meethook_record::request_calendar_access() {
        sink.note(Note::CalendarProblem(problem.to_string()));
    }

    // `ctrlc` runs its handler on a thread of its own rather than in signal context, so the
    // finalize path below is free to allocate and do I/O. The handler itself only signals.
    let interrupt_tx = tx.clone();
    ctrlc::set_handler(move || {
        let _ = interrupt_tx.send(Event::Interrupt);
    })
    .context("could not install the interrupt handler")?;

    let activity_tx = tx.clone();
    // Bound to the whole function: dropping the watcher removes its listeners, and a
    // recorder that has stopped watching is the failure this command exists to prevent.
    let (watcher, already_active) = MicActivityWatcher::start(move |activity| {
        let _ = activity_tx.send(match activity {
            Activity::Started => Event::Started,
            Activity::Stopped => Event::Stopped,
            Activity::InputDeviceChanged => Event::InputDeviceChanged,
        });
    })?;

    sink.note(Note::Watching);
    if already_active {
        sink.note(Note::AlreadyActive);
    }

    let mut capture = SessionCapture {
        recorder: &recorder,
        paths,
        debug,
        running: None,
    };
    record_loop(
        &rx,
        &mut capture,
        &|| watcher.recheck(),
        already_active,
        Timing::LIVE,
        &mut sink,
    );

    // The screen-mode tail, wrapped in a function boundary so the frame structurally cannot
    // outlive the call: frame down, then the narration flushed to stdout, then any stashed
    // trouble. Plain mode owes nothing and falls straight through.
    if let Sink::Screen(screen) = sink {
        crate::record_screen::close(screen)?;
    }

    Ok(())
}

#[cfg(any(target_os = "macos", test))]
/// Sequences one session per detected call until the process is interrupted.
///
/// `recheck` recomputes the activity level from the world, delivering any edge it finds
/// onto `rx` before returning that level. It is called from three places, for three reasons:
/// while a session is live it is the safety net for a release edge that was lost outright,
/// inside [`begin`] it is the level a start retry is driven by, and after a session finalized
/// by an input-device change or a dead microphone it decides whether there is still a call to
/// open a new one for.
///
/// The same timeout that drives the first of those also asks the capture whether the
/// microphone track is still receiving audio. That question has no listener behind it, so this
/// poll is its detection mechanism rather than a safety net; see `Timing::recheck`.
///
/// `already_active` skips the first idle wait, because a call that was already in progress
/// when this process started will not produce a start edge.
fn record_loop(
    rx: &Receiver<Event>,
    capture: &mut dyn Capture,
    recheck: &dyn Fn() -> bool,
    already_active: bool,
    timing: Timing,
    sink: &mut dyn Reporter,
) {
    let mut already_active = already_active;
    loop {
        if !already_active {
            match rx.recv() {
                Ok(Event::Started) => {}
                Ok(Event::Stopped) => continue,
                // A swap between calls is nothing to do: the next session opens the input
                // device afresh, so it already records from the new one. It emphatically must
                // not fall through to the arm below, which would end the recorder outright
                // because somebody unplugged their headphones.
                Ok(Event::InputDeviceChanged) => continue,
                // No session is live to attach a pick to -- it cannot normally arrive here --
                // and dropping it is the answer that keeps the idle wait idle.
                Ok(Event::MeetingPicked(_)) => continue,
                // No session is live to attach an edit to either; the same answer.
                Ok(Event::RosterEdited(_)) => continue,
                Ok(Event::Interrupt) | Err(_) => break,
            }
        }
        already_active = false;

        match begin(rx, capture, recheck, timing, sink) {
            Begin::Recording => {}
            Begin::Abandoned => {
                sink.note(Note::Watching);
                continue;
            }
            Begin::Interrupted => break,
        }

        // The calendar's answer for this session, asked once at start and held until finish:
        // the offers project into the frame, the full values stay here where a pick resolves.
        // Both locals die with the session, so a device-change or stall restart cannot
        // inherit a predecessor's list or pick.
        let offered = capture.candidates();
        let mut hand: Option<Meeting> = None;
        // The frame's committed roster correction, if any: stashed beside the pick so both
        // die with the session block, and applied at the single finalize point below.
        let mut roster_edit: Option<RosterEdit> = None;
        // Sent even when both halves are empty: a restarted session starts clean rather than
        // inheriting what the frame was last shown.
        sink.note(Note::MeetingOffered {
            guess: offered.chosen.as_ref().map(MeetingLabel::from),
            offered: offered.meetings.iter().map(MeetingOffer::from).collect(),
        });
        // The roster note goes out only when an attachment exists: the pane opens on the
        // attached meeting's people, and with no guess there is nothing to open on. A pick
        // later settles a different meeting and re-sends the note for it, superseding this
        // copy wholesale in the frame.
        if let Some(chosen) = &offered.chosen {
            sink.note(Note::RosterAttached {
                event_id: chosen.event_id.clone(),
                attendees: chosen.attendees().to_vec(),
            });
        }

        let outcome = loop {
            match rx.recv_timeout(timing.recheck) {
                Ok(Event::Stopped) => match await_end(rx, timing.grace) {
                    Outcome::CallEnded => break Recording::Ended,
                    Outcome::Interrupted => break Recording::Interrupted,
                    Outcome::Continue => {}
                },
                // A redundant start edge cannot happen while recording, but ignoring it is
                // the interpretation that keeps the session whole either way.
                Ok(Event::Started) => {}
                // The frame settled on one of the offers it was shown. Resolved against the
                // held list -- an unknown id changes nothing, since the frame can only offer
                // what it was given -- and stashed for the single finalize point; a later pick
                // replaces an earlier one, and the finish applies the last.
                Ok(Event::MeetingPicked(event_id)) => {
                    if let Some(meeting) = offered
                        .meetings
                        .iter()
                        .find(|m| m.event_id == event_id)
                        .cloned()
                    {
                        hand = Some(meeting.clone());
                        // Lifted out before the Confirmed stamp moves the meeting into the
                        // label: the supersedure below needs the same id.
                        let event_id = meeting.event_id.clone();
                        // A correction the frame already committed for this very meeting rides
                        // along instead of the pristine snapshot: the pane replaces its copy
                        // wholesale, so resending the calendar's people would visibly revert
                        // the edit -- and a further edit made against that reverted display
                        // would build on the wrong baseline and silently drop the first at
                        // finish.
                        let attendees = match &roster_edit {
                            Some(edit) if edit.event_id == event_id => edit.attendees.clone(),
                            _ => meeting.attendees().to_vec(),
                        };
                        // What crosses to the frame is what `label_by_hand` will write at
                        // finish -- the Confirmed stamp -- never the candidate's own fit,
                        // whose caveat would qualify a pick a human just made.
                        sink.note(Note::MeetingSettled {
                            label: MeetingLabel::from(&meeting.with_fit(MeetingFit::Confirmed)),
                        });
                        // Supersedure, not merge: the pane's copy is replaced wholesale,
                        // mirroring how the settlement supersedes the guess itself -- so what
                        // crosses here must already be the roster the frame should keep (the
                        // edit above, else the meeting's own people).
                        sink.note(Note::RosterAttached {
                            event_id,
                            attendees,
                        });
                    }
                }
                // The frame committed a roster correction, addressed by the event id it was
                // shown. Validated against the held list -- an unknown id changes nothing,
                // since the frame can only edit what it was given -- and stashed for the
                // single finalize point; a later edit replaces an earlier one. Unlike a pick,
                // nothing settles back: an edit changes neither the fit nor the label, so the
                // frame's local copy stays authoritative and no note goes home.
                Ok(Event::RosterEdited(edit)) => {
                    if offered.meetings.iter().any(|m| m.event_id == edit.event_id) {
                        roster_edit = Some(edit);
                    }
                }
                // The engine is bound to the device that was default when it started, so it
                // is now delivering nothing. Everything worth keeping is already on disk;
                // finalize it and open a new session on the new device.
                Ok(Event::InputDeviceChanged) => break Recording::DeviceChanged,
                Ok(Event::Interrupt) => break Recording::Interrupted,
                // The safety net, and the only reason this wait has a timeout at all. A
                // release edge can be lost outright -- the recomputation behind a
                // notification reads a world that can move under it -- and once the machine
                // settles no further notification is coming, so the session would run until
                // the user killed it. Recomputing here costs a few extra seconds of tail
                // instead.
                //
                // The result is dropped because it is the *edge*, not the level, that ends a
                // session: the watcher sends the `Stopped` itself, and the next pass through
                // this loop handles it exactly as it would a timely one.
                //
                // The mic's liveness is asked here too, and this is the only place it is
                // asked. Each place it is *not* asked has its own reason: `begin`'s retry has
                // no session and no engine, `await_end`'s session is already ending so a dead
                // engine changes nothing (the same argument its `InputDeviceChanged` arm
                // already makes), and the idle wait has no session at all. Naming the
                // consequence too: because the check rides on the timeout, an event arriving
                // every couple of seconds would starve it. Events while recording are rare by
                // construction, so that is a hazard to record rather than to engineer around.
                Err(RecvTimeoutError::Timeout) => {
                    // Cheapest question first, and one atomic load at that. A dead engine also
                    // makes the activity level moot until after the finalize, which recomputes
                    // it anyway.
                    if capture.mic_stalled(sink) {
                        break Recording::MicStalled;
                    }
                    let _ = recheck();
                }
                // Every sender is gone, so no edge can arrive again.
                Err(RecvTimeoutError::Disconnected) => break Recording::Interrupted,
            }
        };

        // Said before the finish line rather than after it, so the two session reports the
        // user is about to see read as a consequence of the swap rather than as a fault.
        match outcome {
            Recording::DeviceChanged => sink.note(Note::DeviceChanged),
            Recording::MicStalled => sink.note(Note::MicStalled),
            Recording::Ended | Recording::Interrupted => sink.note(Note::Stopping),
        }

        if let Err(e) = capture.finish(sink, hand.take(), roster_edit.take()) {
            sink.note(Note::FinishFailed(format!(
                "This session did not produce a usable recording: {e}"
            )));
        }

        match outcome {
            Recording::Interrupted => break,
            Recording::Ended => sink.note(Note::Watching),
            // The level, recomputed from the world, is what keeps a swap that coincides with
            // the call ending from opening a session for a call that is already over. When it
            // is still up, `already_active` is exactly the "record without waiting for a start
            // edge that has already happened" case the top of this loop already handles, so
            // the restart inherits `begin`'s bounded retry and its Ctrl-C responsiveness with
            // no second start path. A false answer has already sent `Stopped`, which the idle
            // wait above consumes harmlessly.
            //
            // A stall takes the same arm because the answer is the same: the audio so far is
            // on disk, and whether to open another session is a question about the call, not
            // about what killed the engine. Two variants sharing one arm rather than a single
            // `Restart(Reason)` carrying the message, so that exhaustiveness still forces a
            // deliberate answer for each and the shape TASK-011 landed stays put.
            Recording::DeviceChanged | Recording::MicStalled => {
                if recheck() {
                    already_active = true;
                    continue;
                }
                sink.note(Note::NoNewSession);
                sink.note(Note::Watching);
            }
        }
    }
}

/// How the inner recording loop ended.
///
#[cfg(any(target_os = "macos", test))]
/// An enum rather than the `interrupted` boolean it replaced, so that the one finalize point
/// after the loop stays the only one: three ways out of a recording, three answers to "what
/// happens after `finish`", and a single place where the audio is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Recording {
    /// The microphone went idle for the whole grace period: finalize and go back to watching.
    Ended,
    /// Ctrl-C, or every sender gone: finalize and exit.
    Interrupted,
    /// The default input device moved out from under the engine: finalize, and open a new
    /// session on the new device if the call is still up.
    DeviceChanged,
    /// The microphone track stopped receiving audio with no device change behind it -- the
    /// device reconfigured, something took it exclusively, or the machine slept. Same
    /// reaction as `DeviceChanged`, different sentence.
    MicStalled,
}

#[cfg(any(target_os = "macos", test))]
/// How the attempt to open a session for the call that just started resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Begin {
    /// A session is live and must be finalized.
    Recording,
    /// No session was opened: go back to watching.
    Abandoned,
    /// Ctrl-C arrived before a session existed: exit.
    Interrupted,
}

#[cfg(any(target_os = "macos", test))]
/// Opens a session, retrying for as long as the call is still up.
///
/// The retry is driven by the *level*, not by edges, and that is the whole point. Capture
/// can fail transiently -- a ScreenCaptureKit timeout, an input device caught mid-swap --
/// and returning to the idle wait after one failure would cost the entire meeting: the
/// microphone is still in use, so no further start edge can arrive until this call ends and
/// a different one begins. One timeout two seconds into a 45-minute meeting would leave
/// meethook watching an active microphone in silence for the rest of it.
///
/// Bounded, because a permanent failure should cost this call rather than the day, and the
/// loop stays responsive to Ctrl-C and to the call ending throughout.
fn begin(
    rx: &Receiver<Event>,
    capture: &mut dyn Capture,
    recheck: &dyn Fn() -> bool,
    timing: Timing,
    sink: &mut dyn Reporter,
) -> Begin {
    for attempt in 1..=timing.attempts {
        match capture.start(sink) {
            Ok(()) => return Begin::Recording,
            // Only the first failure is printed in full, and only the give-up line follows
            // it: five copies of one message is noise to read past rather than information.
            Err(e) if attempt == 1 => sink.note(Note::BeginFailed(format!(
                "Could not start recording: {e:#}"
            ))),
            Err(_) => {}
        }

        // Waiting on the channel rather than sleeping: the two things that should abandon
        // the retry both arrive here, and a sleep would delay both by up to the retry
        // interval.
        match rx.recv_timeout(timing.retry) {
            // The call ended while we were failing; there is nothing left to record.
            Ok(Event::Stopped) => return Begin::Abandoned,
            Ok(Event::Interrupt) => return Begin::Interrupted,
            // Cannot normally arrive, since the level was already true. Retrying at once is
            // the reading that loses the least if it does.
            Ok(Event::Started) => {}
            // A device caught mid-swap is one of the transient failures this retry exists for,
            // and the next attempt opens the input device afresh -- so the useful answer is to
            // retry immediately rather than wait out the interval against the old device.
            Ok(Event::InputDeviceChanged) => {}
            // No session exists yet to attach a pick to -- it cannot normally arrive here --
            // so the answer is the same as a redundant start edge: keep retrying.
            Ok(Event::MeetingPicked(_)) => {}
            // No session exists yet to attach an edit to either; the same answer.
            Ok(Event::RosterEdited(_)) => {}
            Err(RecvTimeoutError::Timeout) => {
                // The stop edge can be missed outright, so the level is recomputed from the
                // world here rather than inferred from the absence of a message. It has to
                // be a recomputation and not a cached value: a lost edge is precisely the
                // case where the cached level is the wrong one, and retrying against it
                // would burn every remaining attempt on a call that is already over.
                if !recheck() {
                    return Begin::Abandoned;
                }
            }
            // Every sender is gone, so no edge can arrive again. Nothing is recording, so
            // unlike the grace period there is nothing here worth finalizing.
            Err(RecvTimeoutError::Disconnected) => return Begin::Interrupted,
        }
    }

    sink.note(Note::GivingUp(timing.attempts));
    Begin::Abandoned
}

#[cfg(any(target_os = "macos", test))]
/// What the grace period after a stop edge resolved to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    /// The microphone stayed idle for the whole grace period: finalize.
    CallEnded,
    /// Activity resumed inside the grace period: this was a blip, keep the session.
    Continue,
    /// Ctrl-C arrived during the grace period: finalize now and exit.
    Interrupted,
}

#[cfg(any(target_os = "macos", test))]
/// Waits out the grace period following a stop edge.
///
/// A receive timeout rather than a cancellable timer: the thing that would cancel a timer
/// is a message on this very channel, so waiting on the channel *is* the timer, with no
/// second thread and no cancellation bookkeeping to get wrong.
///
/// The remaining time is recomputed each pass so that events which do not resolve the wait
/// cannot extend it indefinitely.
fn await_end(rx: &Receiver<Event>, grace: Duration) -> Outcome {
    let deadline = Instant::now() + grace;
    loop {
        match rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(Event::Started) => return Outcome::Continue,
            Ok(Event::Interrupt) => return Outcome::Interrupted,
            // Edges alternate, so a second stop should not be possible; wait out the
            // remainder rather than treating an unexpected message as the answer.
            Ok(Event::Stopped) => {}
            // Wait out the remainder too. This session is already ending and its engine is
            // already dead, so there is nothing left for a new device to capture into it --
            // returning `Continue` would rescue a session that has no audio coming.
            Ok(Event::InputDeviceChanged) => {}
            // The session is already ending; the grace period is seconds, and a pick lost here
            // is a recorded edge rather than a hole -- the post-hoc `meeting` command remains
            // the fallback. Waiting out the remainder keeps the grace honest.
            Ok(Event::MeetingPicked(_)) => {}
            // An edit lost here is dropped with the same reasoning: the session finalizes
            // within seconds and the pane closes with it, so no further edit could arrive
            // anyway -- waiting out the remainder keeps the grace honest.
            Ok(Event::RosterEdited(_)) => {}
            Err(RecvTimeoutError::Timeout) => return Outcome::CallEnded,
            // Every sender is gone, so nothing can resume this session. Finalizing is the
            // only outcome that does not lose the audio already captured.
            Err(RecvTimeoutError::Disconnected) => return Outcome::CallEnded,
        }
    }
}

/// The finish summary's meeting line, off the label: the prefix plus [`MeetingLabel::clause`]'s
/// wording, the same composer the enroll queue announcement derives its line from, so `record`
/// and `enroll` cannot print two shapes of the same meeting.
///
/// The title only, and only the title. It is the sole user-visible evidence that the calendar
/// lookup worked, and it is the one field of a meeting that is safe to put on a terminal:
/// attendee names and addresses are written to `session.json` for speaker identification and
/// are deliberately never printed, and neither is the invite body. A match the session's start
/// does not actually support is qualified rather than stated flat, so a session that merely
/// *sat inside* a booked hour does not read as that meeting; the wording is `MeetingFit`'s own
/// -- this crate owns the stream, the library owns the sentence -- and it names only the
/// timing, so the rule above survives it.
///
/// A function rather than the `println!` it replaced so that the whole rule is decidable in
/// `cargo test` with no terminal, which is the split this module's own documentation describes.
#[cfg(any(target_os = "macos", test))]
pub(crate) fn meeting_clause_line(label: &MeetingLabel) -> String {
    format!("  meeting   {}", label.clause())
}

/// The record loop's sequencing, exercised without a microphone.
///
/// This is where "a blip does not split a session", "a mute does not end one", "two calls
/// produce two sessions" and "Ctrl-C finalizes before it exits" are decidable in an
/// automated test. What is *not* decidable here is whether the trigger fires at all against
/// real hardware -- these tests feed the loop the edges a working watcher would produce, and
/// prove only what it does with them.
#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, mpsc};
    use std::thread;
    use std::time::{Duration, Instant};

    use crate::commands::Tty;
    use meethook_session::{Meeting, MeetingFit, RosterEdit, SessionId};

    use super::{
        Capture, DEVICE_CHANGED, Event, MIC_STALLED, Note, Offered, Outcome, Presenter, Timing,
        await_end, meeting_clause_line, mic_line, presenter, record_loop, recorded_lines,
        session_dir_line, session_id_line, speaker_line,
    };

    /// A reporter that records nothing.
    ///
    /// The sequencing tests below assert on the capture's call log, and a note they did not
    /// expect would surface in the composer tests instead: wording is pinned there, ordering
    /// here, and neither suite needs the other's eyes.
    #[derive(Default)]
    struct Silent;

    impl super::Reporter for Silent {
        fn note(&mut self, _note: Note) {}
    }

    /// Drives [`record_loop`] with a silent reporter at the test timing, so the call sites
    /// read like the loop itself rather than like a plumbing exercise.
    fn run(
        rx: &mpsc::Receiver<Event>,
        capture: &mut FakeCapture,
        recheck: &dyn Fn() -> bool,
        already_active: bool,
    ) {
        let mut silent = Silent;
        record_loop(
            rx,
            capture,
            recheck,
            already_active,
            LOOP_TIMING,
            &mut silent,
        );
    }

    /// A reporter that records the notes it is handed, in order.
    ///
    /// The pick tests below assert on what the loop *says* as well as what it does: the
    /// offer note and the settlement note are the frame's whole view of the calendar, so
    /// their order and content are part of the contract the call log alone cannot see.
    #[derive(Default)]
    struct Noted {
        notes: Vec<Note>,
    }

    impl super::Reporter for Noted {
        fn note(&mut self, note: Note) {
            self.notes.push(note);
        }
    }

    /// Drives [`record_loop`] with a note-recording reporter at the test timing, handing back
    /// what the loop said in the order it said it.
    fn run_noted(
        rx: &mpsc::Receiver<Event>,
        capture: &mut FakeCapture,
        recheck: &dyn Fn() -> bool,
        already_active: bool,
    ) -> Vec<Note> {
        let mut noted = Noted::default();
        record_loop(
            rx,
            capture,
            recheck,
            already_active,
            LOOP_TIMING,
            &mut noted,
        );
        noted.notes
    }

    /// A candidate meeting, the way the record crate's lookup would hand one over: the fit
    /// left at the candidate's own, since only a person choosing it makes it Confirmed.
    fn meeting_of(event_id: &str, title: &str) -> Meeting {
        Meeting::new(
            event_id.to_owned(),
            title.to_owned(),
            "Work".to_owned(),
            "2026-08-15T10:00:00Z".parse().unwrap(),
            "2026-08-15T11:00:00Z".parse().unwrap(),
        )
    }

    /// Long enough that scheduling noise cannot be mistaken for a timeout, short enough
    /// that the suite stays fast.
    const GRACE: Duration = Duration::from_millis(300);

    /// A channel plus a feeder thread that sends one event after `delay`.
    ///
    /// The returned `Sender` is the point of this helper rather than an accident: the
    /// feeder gets a *clone*, so the channel does not disconnect when it exits. In the
    /// real loop the original sender lives for the whole run, and a disconnect there means
    /// something quite different from "the feeder is done".
    fn feed(delay: Duration, event: Event) -> (mpsc::Sender<Event>, mpsc::Receiver<Event>) {
        let (tx, rx) = mpsc::channel::<Event>();
        let feeder = tx.clone();
        thread::spawn(move || {
            thread::sleep(delay);
            let _ = feeder.send(event);
        });
        (tx, rx)
    }

    /// The finish line says the title plainly for a match the start supports, and qualifies
    /// one it does not -- driven over every fit so a variant added later cannot slip through
    /// unqualified.
    ///
    /// It also re-asserts the standing rule on this line, against a meeting carrying every
    /// field that must never reach a terminal: no attendee, no organizer, no location and no
    /// invite body, however the fit came out.
    #[test]
    fn the_finish_line_qualifies_a_meeting_the_session_start_does_not_support() {
        use meethook_session::{Attendee, AttendeeStatus, Meeting, MeetingFit};

        for fit in MeetingFit::ALL {
            let meeting = Meeting::new(
                "EVENT-ABC".to_owned(),
                "Incident review".to_owned(),
                "Work".to_owned(),
                "2026-08-15T10:00:00Z".parse().unwrap(),
                "2026-08-15T11:00:00Z".parse().unwrap(),
            )
            .with_people(
                Some(Attendee {
                    name: Some("Alan Turing".to_owned()),
                    email: Some("alan@example.com".to_owned()),
                    status: AttendeeStatus::Accepted,
                    is_you: false,
                }),
                vec![Attendee {
                    name: Some("Grace Hopper".to_owned()),
                    email: Some("grace@example.com".to_owned()),
                    status: AttendeeStatus::Accepted,
                    is_you: true,
                }],
            )
            .with_invite(
                None,
                Some("Babbage Room".to_owned()),
                Some("Dial-in 555-0100, passcode 481516".to_owned()),
            )
            .with_fit(fit);

            let line = meeting_clause_line(&super::MeetingLabel::from(&meeting));
            assert!(line.contains("Incident review"), "{fit:?}: {line}");

            if fit.is_strong() {
                assert_eq!(line, "  meeting   Incident review", "{fit:?}");
            } else {
                let caveat = fit.caveat().expect("a weak fit has a caveat");
                assert!(line.contains(caveat), "{fit:?}: {line}");
                assert!(
                    line.contains("uncertain") || line.contains("unverified"),
                    "{line}"
                );
            }

            for secret in [
                "Grace",
                "Hopper",
                "grace@example.com",
                "Alan",
                "Turing",
                "@",
                "Babbage",
                "Dial-in",
                "481516",
            ] {
                assert!(
                    !line.contains(secret),
                    "the finish line leaks {secret:?}: {line}"
                );
            }
        }
    }

    #[test]
    fn silence_for_the_whole_grace_period_ends_the_call() {
        let (_tx, rx) = mpsc::channel::<Event>();
        let started = Instant::now();
        assert_eq!(await_end(&rx, GRACE), Outcome::CallEnded);
        assert!(
            started.elapsed() >= GRACE,
            "returned before the grace elapsed"
        );
    }

    /// A mute toggle, or any other blip: the microphone comes back before the grace
    /// expires, and the session must survive it whole.
    #[test]
    fn activity_inside_the_grace_period_keeps_the_session() {
        let (_tx, rx) = feed(GRACE / 6, Event::Started);

        let started = Instant::now();
        assert_eq!(await_end(&rx, GRACE), Outcome::Continue);
        // The early return is the point: waiting out the full grace and *then* continuing
        // would look identical from the outcome alone.
        assert!(
            started.elapsed() < GRACE,
            "waited out the whole grace period"
        );
    }

    #[test]
    fn an_interrupt_inside_the_grace_period_wins() {
        let (_tx, rx) = feed(GRACE / 6, Event::Interrupt);
        assert_eq!(await_end(&rx, GRACE), Outcome::Interrupted);
    }

    #[test]
    fn activity_after_the_grace_period_does_not_rescue_the_session() {
        let (_tx, rx) = feed(GRACE * 2, Event::Started);
        assert_eq!(await_end(&rx, GRACE), Outcome::CallEnded);
    }

    /// A device swap during the grace period must not rescue a session that is ending. The
    /// engine is dead either way, so "continue" would keep a session open with no audio
    /// arriving -- the same silent truncation the device change is reported to prevent.
    #[test]
    fn a_device_change_during_the_grace_period_does_not_rescue_the_session() {
        let (_tx, rx) = feed(GRACE / 6, Event::InputDeviceChanged);

        let started = Instant::now();
        assert_eq!(await_end(&rx, GRACE), Outcome::CallEnded);
        assert!(
            started.elapsed() >= GRACE,
            "the device change cut the wait short"
        );
    }

    #[test]
    fn a_stray_stop_does_not_shorten_or_resolve_the_wait() {
        let (_tx, rx) = feed(GRACE / 6, Event::Stopped);

        let started = Instant::now();
        assert_eq!(await_end(&rx, GRACE), Outcome::CallEnded);
        assert!(
            started.elapsed() >= GRACE,
            "the stray stop cut the wait short"
        );
    }

    /// Losing every sender cannot leave a live recording waiting forever.
    #[test]
    fn a_disconnected_channel_ends_the_call() {
        let (tx, rx) = mpsc::channel::<Event>();
        drop(tx);
        assert_eq!(await_end(&rx, GRACE), Outcome::CallEnded);
    }

    /// Timing for the whole-loop tests below.
    ///
    /// `SETTLE` is what separates one call from the next: comfortably longer than `grace`,
    /// so a gap the loop is meant to read as "the call ended" cannot be mistaken for
    /// scheduling noise. `BLIP` is the opposite -- short enough that it must land inside the
    /// grace window.
    ///
    /// `recheck` is well inside `SETTLE`, so every test below also exercises a safety-net
    /// re-check that has to stay inert.
    const LOOP_TIMING: Timing = Timing {
        grace: Duration::from_millis(120),
        retry: Duration::from_millis(30),
        attempts: 3,
        recheck: Duration::from_millis(40),
    };
    const SETTLE: Duration = Duration::from_millis(400);
    const BLIP: Duration = Duration::from_millis(20);

    /// A [`Capture`] that records the order it was called in.
    ///
    /// `start` is logged whether or not it succeeds, so the retry tests can count attempts
    /// as well as sessions.
    #[derive(Default)]
    struct FakeCapture {
        calls: Vec<&'static str>,
        /// How many of the next starts should fail.
        failing_starts: u32,
        /// How many of the next sessions should report a dead microphone on their first
        /// re-check. Decremented as it is consumed, mirroring `failing_starts`.
        stalling_sessions: u32,
        /// How many times `mic_stalled` was asked, and what that stood at when the first
        /// session opened.
        ///
        /// The second is snapshotted rather than read at the end because asking a capture with
        /// no session is the bug shape worth catching.
        mic_stalled_calls: usize,
        mic_stalled_before_the_first_session: Option<usize>,
        /// How many times the loop has re-checked, shared with the re-check closure.
        rechecks: Arc<AtomicUsize>,
        /// That counter as it stood when the first session opened.
        ///
        /// Snapshotted here rather than read at the end because it is the *idle* stretch
        /// before a session exists that is meant to recompute nothing.
        rechecks_before_the_first_session: Option<usize>,
        /// When the first session was finalized.
        ///
        /// `record_loop` returns only once it is interrupted, so its own elapsed time says
        /// nothing about *when* a session ended; this is what distinguishes a session the
        /// loop ended on its own from one an interrupt cleaned up afterwards.
        finished_at: Option<Instant>,
        /// The calendar's answer `candidates` hands back, scripted like the rest. Not logged
        /// into `calls`: the sequencing tests' exact call logs predate the seam, and asking
        /// for candidates is not a sequencing decision.
        candidates: Option<Offered>,
        /// What each `finish` was handed as the in-memory pick -- the event ids, in order --
        /// so a test can assert a pick reached the single finalize point without holding a
        /// full `Meeting` of its own.
        hands: Vec<Option<String>>,
        /// What each `finish` was handed as the roster edit -- the full values, in order --
        /// so a test can assert both the addressing id and the edited rows reached the
        /// single finalize point.
        roster_edits: Vec<Option<RosterEdit>>,
    }

    impl Capture for FakeCapture {
        fn start(&mut self, _sink: &mut dyn super::Reporter) -> super::Result<()> {
            self.calls.push("start");
            if self.rechecks_before_the_first_session.is_none() {
                self.rechecks_before_the_first_session = Some(self.rechecks.load(Ordering::SeqCst));
                self.mic_stalled_before_the_first_session = Some(self.mic_stalled_calls);
            }
            if self.failing_starts > 0 {
                self.failing_starts -= 1;
                anyhow::bail!("deliberate start failure");
            }
            Ok(())
        }

        fn finish(
            &mut self,
            _sink: &mut dyn super::Reporter,
            hand: Option<Meeting>,
            roster_edit: Option<RosterEdit>,
        ) -> super::Result<()> {
            self.calls.push("finish");
            self.finished_at.get_or_insert_with(Instant::now);
            self.hands.push(hand.map(|m| m.event_id));
            self.roster_edits.push(roster_edit);
            Ok(())
        }

        fn candidates(&mut self) -> Offered {
            self.candidates.clone().unwrap_or_default()
        }

        fn mic_stalled(&mut self, _sink: &mut dyn super::Reporter) -> bool {
            self.mic_stalled_calls += 1;
            if self.stalling_sessions > 0 {
                self.stalling_sessions -= 1;
                return true;
            }
            false
        }
    }

    /// Plays a script of events into a channel from its own thread.
    ///
    /// A script rather than pre-queued messages, because the loop's decisions are made of
    /// *gaps*: a stop followed immediately by a queued start is a blip, and the same pair
    /// separated by the grace period is two calls. Sending them all up front would collapse
    /// every one of those distinctions.
    ///
    /// The returned `Sender` is kept alive by the caller on purpose: in the real loop the
    /// original sender lives for the whole run, and a disconnect means something quite
    /// different from "the script is finished".
    fn script(events: Vec<(Duration, Event)>) -> (mpsc::Sender<Event>, mpsc::Receiver<Event>) {
        let (tx, rx) = mpsc::channel::<Event>();
        let feeder = tx.clone();
        thread::spawn(move || {
            for (delay, event) in events {
                thread::sleep(delay);
                if feeder.send(event).is_err() {
                    return;
                }
            }
        });
        (tx, rx)
    }

    /// Two consecutive calls in one process lifetime are two separate sessions.
    #[test]
    fn two_calls_produce_two_sessions() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (BLIP, Event::Stopped),
            (SETTLE, Event::Started),
            (BLIP, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture::default();
        run(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish", "start", "finish"]);
    }

    /// A device-state blip shorter than the grace period is one session, not two -- the same
    /// shape a mute toggle would take if it produced edges at all.
    #[test]
    fn a_blip_inside_the_grace_period_does_not_split_the_session() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (BLIP, Event::Stopped),
            (BLIP, Event::Started),
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture::default();
        run(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish"]);
    }

    /// Ctrl-C mid-recording finalizes before it exits. The alternative is a truncated WAV
    /// with no header and no `session.json`, which is unrecoverable audio.
    #[test]
    fn an_interrupt_while_recording_finalizes_first() {
        let (_tx, rx) = script(vec![(BLIP, Event::Started), (BLIP, Event::Interrupt)]);

        let mut capture = FakeCapture::default();
        run(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish"]);
    }

    /// A pick of an offered id reaches the single finalize point as the hand, and the frame
    /// is told about both halves in order: the offers when the session opens, the settlement
    /// when the pick lands -- carrying the Confirmed stamp `label_by_hand` will write, never
    /// the candidate's own fit.
    #[test]
    fn a_pick_reaches_finish_as_the_hand_and_settles_in_order() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (BLIP, Event::MeetingPicked("EVENT-A".to_owned())),
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture {
            candidates: Some(Offered {
                meetings: vec![
                    meeting_of("EVENT-A", "Standup"),
                    meeting_of("EVENT-B", "Planning"),
                ],
                chosen: Some(meeting_of("EVENT-A", "Standup")),
            }),
            ..FakeCapture::default()
        };
        let notes = run_noted(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish"]);
        assert_eq!(capture.hands, [Some("EVENT-A".to_owned())]);

        let mut seen_offer = false;
        let mut settled = None;
        for note in &notes {
            match note {
                Note::MeetingOffered { offered, guess } => {
                    assert!(!seen_offer, "two offer notes for one session");
                    seen_offer = true;
                    assert_eq!(offered.len(), 2);
                    assert_eq!(
                        guess.as_ref().map(|label| label.title.as_str()),
                        Some("Standup")
                    );
                }
                Note::MeetingSettled { label } => {
                    assert!(seen_offer, "the settlement crossed before the offers");
                    settled = Some(label.clone());
                }
                _ => {}
            }
        }
        assert!(
            seen_offer,
            "the session opened without offering its calendar"
        );
        let settled = settled.expect("no settlement reached the frame");
        assert_eq!(settled.title, "Standup");
        assert_eq!(settled.fit, MeetingFit::Confirmed);
    }

    /// A later pick replaces an earlier one: the finish applies the last, and the frame sees
    /// both settlements in the order they were made.
    #[test]
    fn a_later_pick_replaces_an_earlier_one() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (BLIP, Event::MeetingPicked("EVENT-A".to_owned())),
            (BLIP, Event::MeetingPicked("EVENT-B".to_owned())),
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture {
            candidates: Some(Offered {
                meetings: vec![
                    meeting_of("EVENT-A", "Standup"),
                    meeting_of("EVENT-B", "Planning"),
                ],
                chosen: Some(meeting_of("EVENT-A", "Standup")),
            }),
            ..FakeCapture::default()
        };
        let notes = run_noted(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish"]);
        assert_eq!(capture.hands, [Some("EVENT-B".to_owned())]);
        let settled: Vec<String> = notes
            .iter()
            .filter_map(|note| match note {
                Note::MeetingSettled { label } => Some(label.title.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            settled,
            vec!["Standup".to_owned(), "Planning".to_owned()],
            "both picks settled, in the order they were made"
        );
    }

    /// An id the frame was never offered changes nothing: no settlement crosses, and the
    /// finish settles by the automatic rule rather than a pick the run cannot resolve.
    #[test]
    fn an_unresolvable_pick_changes_nothing() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (BLIP, Event::MeetingPicked("NOT-OFFERED".to_owned())),
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture {
            candidates: Some(Offered {
                meetings: vec![meeting_of("EVENT-A", "Standup")],
                chosen: Some(meeting_of("EVENT-A", "Standup")),
            }),
            ..FakeCapture::default()
        };
        let notes = run_noted(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish"]);
        assert_eq!(
            capture.hands,
            [None],
            "the unresolvable pick reached finish as a pick"
        );
        assert!(
            !notes
                .iter()
                .any(|note| matches!(note, Note::MeetingSettled { .. })),
            "the frame was told a pick stuck that the run could not resolve"
        );
        assert!(
            notes
                .iter()
                .any(|note| matches!(note, Note::MeetingOffered { .. })),
            "the offers still went out"
        );
    }

    /// A device change right after a pick finalizes the first session with it and starts the
    /// next clean: the second finish sees no inherited pick, and the frame is offered again
    /// for the session it now describes.
    #[test]
    fn a_pick_does_not_leak_across_a_restart() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (BLIP, Event::MeetingPicked("EVENT-A".to_owned())),
            (BLIP, Event::InputDeviceChanged),
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture {
            candidates: Some(Offered {
                meetings: vec![meeting_of("EVENT-A", "Standup")],
                chosen: Some(meeting_of("EVENT-A", "Standup")),
            }),
            ..FakeCapture::default()
        };
        let notes = run_noted(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish", "start", "finish"]);
        assert_eq!(
            capture.hands,
            [Some("EVENT-A".to_owned()), None],
            "the second session inherited its predecessor's pick"
        );
        let offers = notes
            .iter()
            .filter(|note| matches!(note, Note::MeetingOffered { .. }))
            .count();
        assert_eq!(offers, 2, "each session got its own offer note");
    }

    /// A calendar that answers nothing still gets offered: the note goes out even when both
    /// halves are empty, so a restarted session cannot inherit its predecessor's list.
    #[test]
    fn an_empty_calendar_still_gets_offered() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture::default();
        let notes = run_noted(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish"]);
        let Note::MeetingOffered { offered, guess } = notes
            .iter()
            .find(|note| matches!(note, Note::MeetingOffered { .. }))
            .expect("no offer note for an empty calendar")
        else {
            unreachable!("the match above found a MeetingOffered")
        };
        assert!(offered.is_empty());
        assert!(guess.is_none());
        // No attachment means no roster note: the pane has nothing to open on, and the
        // frame's copy stays whatever it last had rather than being cleared by absence.
        assert!(
            !notes
                .iter()
                .any(|note| matches!(note, Note::RosterAttached { .. })),
            "no roster crossed for a calendar with nothing attached"
        );
    }

    /// An edit of the attached roster reaches the single finalize point addressed by the
    /// event id the frame was shown, and the frame was told the roster right after the
    /// offers -- composing to nothing, so a plain run says nothing about the people.
    #[test]
    fn an_edit_reaches_finish_addressed_by_event_id() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (
                BLIP,
                Event::RosterEdited(RosterEdit {
                    event_id: "EVENT-A".to_owned(),
                    attendees: vec![],
                }),
            ),
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture {
            candidates: Some(Offered {
                meetings: vec![meeting_of("EVENT-A", "Standup")],
                chosen: Some(meeting_of("EVENT-A", "Standup")),
            }),
            ..FakeCapture::default()
        };
        let notes = run_noted(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish"]);
        assert_eq!(capture.hands, [None]);
        assert_eq!(capture.roster_edits.len(), 1);
        let edit = capture.roster_edits[0]
            .as_ref()
            .expect("the edit did not reach finish");
        assert_eq!(edit.event_id, "EVENT-A");

        // The roster crossed into the frame after the offers, and composes to nothing.
        let mut seen_offer = false;
        for note in &notes {
            match note {
                Note::MeetingOffered { .. } => seen_offer = true,
                Note::RosterAttached { event_id, .. } => {
                    assert!(seen_offer, "the roster crossed before the offers");
                    assert_eq!(event_id, "EVENT-A");
                    assert_eq!(note.composed(), "", "a roster note must compose to nothing");
                }
                _ => {}
            }
        }
        assert!(
            seen_offer,
            "the session opened without offering its calendar"
        );
    }

    /// A pick and an edit of the same meeting both reach finish: the mismatch question is
    /// deliberately NOT pre-filtered in the loop -- the documented drop lives in
    /// `apply_roster_edit` alone, so the semantics have one home. The settlement also
    /// re-sends the roster for the picked meeting, superseding the guess's copy.
    #[test]
    fn an_edit_rides_alside_a_hand_pick() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (BLIP, Event::MeetingPicked("EVENT-B".to_owned())),
            (
                BLIP,
                Event::RosterEdited(RosterEdit {
                    event_id: "EVENT-B".to_owned(),
                    attendees: vec![],
                }),
            ),
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture {
            candidates: Some(Offered {
                meetings: vec![
                    meeting_of("EVENT-A", "Standup"),
                    meeting_of("EVENT-B", "Planning"),
                ],
                chosen: Some(meeting_of("EVENT-A", "Standup")),
            }),
            ..FakeCapture::default()
        };
        let notes = run_noted(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish"]);
        assert_eq!(capture.hands, [Some("EVENT-B".to_owned())]);
        assert_eq!(capture.roster_edits.len(), 1);
        assert_eq!(
            capture.roster_edits[0]
                .as_ref()
                .map(|e| e.event_id.as_str()),
            Some("EVENT-B"),
            "the edit rode alongside the pick, addressed to the picked meeting"
        );
        // Two attachments in order: the guess at start, the settled pick after.
        let attached: Vec<&str> = notes
            .iter()
            .filter_map(|note| match note {
                Note::RosterAttached { event_id, .. } => Some(event_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(attached, ["EVENT-A", "EVENT-B"]);
    }

    /// Re-picking a meeting whose roster the frame already edited must not revert the frame to
    /// the pristine calendar snapshot: the resent attachment carries the committed edit, and
    /// the loop's stashed edit is untouched by the re-pick alone.
    #[test]
    fn re_picking_an_already_edited_meeting_keeps_the_frame_on_the_edit() {
        use meethook_session::{Attendee, AttendeeStatus};

        let alan = Attendee {
            name: Some("Alan Turing".to_owned()),
            email: Some("alan@example.com".to_owned()),
            status: AttendeeStatus::Accepted,
            is_you: false,
        };
        let seeded = meeting_of("EVENT-A", "Standup").with_people(None, vec![alan.clone()]);

        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (
                BLIP,
                Event::RosterEdited(RosterEdit {
                    event_id: "EVENT-A".to_owned(),
                    attendees: vec![],
                }),
            ),
            (BLIP, Event::MeetingPicked("EVENT-A".to_owned())),
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture {
            candidates: Some(Offered {
                meetings: vec![seeded.clone()],
                chosen: Some(seeded),
            }),
            ..FakeCapture::default()
        };
        let notes = run_noted(&rx, &mut capture, &|| true, false);

        // Two attachments in order: the guess at start (pristine, one attendee), then the
        // re-pick -- which must carry the committed edit (empty), not the pristine snapshot.
        let rosters: Vec<Vec<Attendee>> = notes
            .iter()
            .filter_map(|note| match note {
                Note::RosterAttached {
                    event_id,
                    attendees,
                } if event_id == "EVENT-A" => Some(attendees.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(
            rosters,
            [vec![alan], vec![]],
            "the re-pick reverted the frame to the pristine roster"
        );

        // The pick still settles the hand, and the stashed edit rides to finish untouched.
        assert_eq!(capture.calls, ["start", "finish"]);
        assert_eq!(capture.hands, [Some("EVENT-A".to_owned())]);
        assert_eq!(capture.roster_edits.len(), 1);
        let edit = capture.roster_edits[0]
            .as_ref()
            .expect("the edit reached finish");
        assert_eq!(edit.event_id, "EVENT-A");
        assert!(edit.attendees.is_empty());
    }

    /// An edit addressed to a meeting the frame was never shown changes nothing: finish
    /// receives no edit, exactly as an unresolvable pick settles nothing.
    #[test]
    fn an_unknown_roster_id_changes_nothing() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (
                BLIP,
                Event::RosterEdited(RosterEdit {
                    event_id: "NOT-OFFERED".to_owned(),
                    attendees: vec![],
                }),
            ),
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture {
            candidates: Some(Offered {
                meetings: vec![meeting_of("EVENT-A", "Standup")],
                chosen: Some(meeting_of("EVENT-A", "Standup")),
            }),
            ..FakeCapture::default()
        };
        run_noted(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish"]);
        assert_eq!(capture.hands, [None]);
        assert_eq!(
            capture.roster_edits,
            [None],
            "the unresolvable edit reached finish as an edit"
        );
    }

    /// A device change right after an edit finalizes the first session with it and starts
    /// the next clean: the second finish sees no inherited edit, mirroring the pick's
    /// restart-clean guarantee.
    #[test]
    fn an_edit_does_not_leak_across_a_restart() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (
                BLIP,
                Event::RosterEdited(RosterEdit {
                    event_id: "EVENT-A".to_owned(),
                    attendees: vec![],
                }),
            ),
            (BLIP, Event::InputDeviceChanged),
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture {
            candidates: Some(Offered {
                meetings: vec![meeting_of("EVENT-A", "Standup")],
                chosen: Some(meeting_of("EVENT-A", "Standup")),
            }),
            ..FakeCapture::default()
        };
        run_noted(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish", "start", "finish"]);
        assert_eq!(
            capture.roster_edits,
            [
                Some(RosterEdit {
                    event_id: "EVENT-A".to_owned(),
                    attendees: vec![]
                }),
                None,
            ],
            "the second session inherited its predecessor's edit"
        );
    }

    /// A call already in progress at startup is recorded without waiting for an edge that
    /// has already happened.
    #[test]
    fn an_already_active_microphone_records_without_a_start_edge() {
        let (_tx, rx) = script(vec![(BLIP, Event::Stopped), (SETTLE, Event::Interrupt)]);

        let mut capture = FakeCapture::default();
        run(&rx, &mut capture, &|| true, true);

        assert_eq!(capture.calls, ["start", "finish"]);
    }

    /// The defect this retry exists for: a transient start failure must not cost the whole
    /// meeting. No second start edge can arrive while the call is still up, so a loop that
    /// went back to waiting for one would watch an active microphone in silence.
    #[test]
    fn a_transient_start_failure_still_records_the_call() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture {
            failing_starts: 1,
            ..FakeCapture::default()
        };
        run(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "start", "finish"]);
    }

    /// A start that keeps failing costs this call and nothing more: the retry is bounded,
    /// and the loop is still watching when the next call arrives.
    #[test]
    fn a_permanent_start_failure_gives_up_without_ending_the_loop() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (SETTLE, Event::Stopped),
            (BLIP, Event::Started),
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture {
            failing_starts: LOOP_TIMING.attempts,
            ..FakeCapture::default()
        };
        run(&rx, &mut capture, &|| true, false);

        let attempts = LOOP_TIMING.attempts as usize;
        assert_eq!(capture.calls.len(), attempts + 2, "{:?}", capture.calls);
        assert!(capture.calls[..attempts].iter().all(|c| *c == "start"));
        assert_eq!(capture.calls[attempts..], ["start", "finish"]);
    }

    /// A call that ends while the start is still failing stops the retry, without waiting
    /// for the remaining attempts and without opening a session for a call that is over.
    #[test]
    fn a_start_failure_stops_retrying_once_the_microphone_goes_idle() {
        let (_tx, rx) = script(vec![(BLIP, Event::Started), (SETTLE, Event::Interrupt)]);

        let mut capture = FakeCapture {
            failing_starts: LOOP_TIMING.attempts,
            ..FakeCapture::default()
        };
        // The level is false by the time the first retry is due: the stop edge was missed
        // outright, which is exactly the case the level check is there to cover.
        run(&rx, &mut capture, &|| false, false);

        assert_eq!(capture.calls, ["start"]);
    }

    /// The TASK-005.03 defect: the release notification arrived, the recomputation behind
    /// it read the world microseconds too early and answered `true`, and no further
    /// notification was ever coming -- so the session ran for 1862.8 s until Ctrl-C.
    ///
    /// Nothing in this script delivers a stop edge, which is the point: only the safety-net
    /// re-check can end this session. The interrupt is there solely to let `record_loop`
    /// return, and *when* the session was finalized is therefore the real assertion -- an
    /// unrecovered session would also finish, just not until the interrupt cleaned it up.
    #[test]
    fn a_missed_stop_edge_is_recovered_by_the_recheck() {
        let (tx, rx) = script(vec![(BLIP, Event::Started), (SETTLE * 3, Event::Interrupt)]);

        let stopped = tx.clone();
        let calls = AtomicUsize::new(0);
        // Two re-checks still lose the race, and the third catches up. What it does then is
        // what the watcher does on hardware: it emits the edge itself rather than returning
        // it, so the loop handles an ordinary `Stopped` and needs no second path.
        let recheck = move || {
            if calls.fetch_add(1, Ordering::SeqCst) < 2 {
                return true;
            }
            let _ = stopped.send(Event::Stopped);
            false
        };

        let mut capture = FakeCapture::default();
        let started = Instant::now();
        run(&rx, &mut capture, &recheck, false);

        assert_eq!(capture.calls, ["start", "finish"]);
        let finished = capture
            .finished_at
            .expect("the session was never finalized")
            .duration_since(started);
        assert!(
            finished < SETTLE * 3,
            "the interrupt ended this session after {finished:?}, not the recheck"
        );
    }

    /// The defect this ticket exists for: `AVAudioEngine` binds its input node to whatever
    /// device was default at start, so a swap mid-session leaves the microphone track silently
    /// receiving nothing for the rest of the meeting. Two sessions is the answer -- each with
    /// its own sample rate in its own WAV header, and its own mic/speaker lag for `transcribe`
    /// to measure.
    #[test]
    fn a_device_change_while_recording_splits_the_session() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (BLIP, Event::InputDeviceChanged),
            (BLIP, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture::default();
        run(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish", "start", "finish"]);
    }

    /// The guard on that restart: a swap in the same breath as the call ending must not open a
    /// session for a call that is over. The level, recomputed from the world, is what decides.
    #[test]
    fn a_device_change_does_not_reopen_a_session_for_a_call_that_ended() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            (BLIP, Event::InputDeviceChanged),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture::default();
        run(&rx, &mut capture, &|| false, false);

        assert_eq!(capture.calls, ["start", "finish"]);
    }

    /// A swap between calls is nothing to do -- the next session opens the input device afresh
    /// -- and above all it must not end the recorder, which is what unplugging a pair of
    /// headphones would cost if this event fell into the idle wait's interrupt arm.
    #[test]
    fn a_device_change_while_idle_does_not_start_a_session() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::InputDeviceChanged),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture::default();
        let started = Instant::now();
        run(&rx, &mut capture, &|| true, false);

        assert!(capture.calls.is_empty(), "{:?}", capture.calls);
        // "Returned early" and "returned on the interrupt" are indistinguishable from an empty
        // call log alone, and the first of those is the recorder exiting on a device swap.
        assert!(
            started.elapsed() >= SETTLE,
            "the loop returned on the device change, not on the interrupt"
        );
    }

    /// A device caught mid-swap is one of the transient failures the start retry exists for, so
    /// the change arriving during it is a reason to try again at once rather than to give up.
    #[test]
    fn a_device_change_during_a_start_retry_keeps_retrying() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            // Queued immediately, so it is waiting when the first failed start reaches the
            // retry wait; a delay near `retry` would race the timeout instead.
            (Duration::ZERO, Event::InputDeviceChanged),
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture {
            failing_starts: 1,
            ..FakeCapture::default()
        };
        run(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "start", "finish"]);
    }

    /// The defect this ticket exists for: an input tap can stop delivering buffers with the
    /// default device exactly where it was -- the device reconfigured its rate, something took
    /// it exclusively, the machine slept -- and no notification says so. The frame count
    /// standing still is what says so, and the answer is the same split a device change gets.
    ///
    /// Nothing in this script delivers a stop edge before the interrupt, which is the point:
    /// only the stall check can end the first session, so *when* it was finalized is the real
    /// assertion. An undetected stall would also finish, just not until the interrupt.
    #[test]
    fn a_stalled_microphone_splits_the_session() {
        let (_tx, rx) = script(vec![(BLIP, Event::Started), (SETTLE * 3, Event::Interrupt)]);

        let mut capture = FakeCapture {
            stalling_sessions: 1,
            ..FakeCapture::default()
        };
        let started = Instant::now();
        run(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish", "start", "finish"]);
        let finished = capture
            .finished_at
            .expect("the first session was never finalized")
            .duration_since(started);
        assert!(
            finished < SETTLE * 3,
            "the interrupt ended this session after {finished:?}, not the stall check"
        );
    }

    /// The guard on that restart, same as the device-change one: a mic that dies in the same
    /// breath as the call ending must not open a session for a call that is over.
    #[test]
    fn a_stalled_microphone_does_not_reopen_a_session_for_a_call_that_ended() {
        let (_tx, rx) = script(vec![(BLIP, Event::Started), (SETTLE, Event::Interrupt)]);

        let mut capture = FakeCapture {
            stalling_sessions: 1,
            ..FakeCapture::default()
        };
        run(&rx, &mut capture, &|| false, false);

        assert_eq!(capture.calls, ["start", "finish"]);
    }

    /// The false-positive guard at loop level: a session spanning many re-check intervals with
    /// a live microphone is one session. A quiet stretch of a meeting looks exactly like this,
    /// because a live tap delivers silence as samples.
    #[test]
    fn a_live_microphone_is_never_mistaken_for_a_stalled_one() {
        let (_tx, rx) = script(vec![
            (BLIP, Event::Started),
            // Ten re-check intervals of recording before the call ends.
            (SETTLE, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture {
            stalling_sessions: 0,
            ..FakeCapture::default()
        };
        run(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish"]);
        assert!(
            capture.mic_stalled_calls > 0,
            "the stall check never ran, so this proves nothing"
        );
    }

    /// Asking a capture with no session whether its microphone is dead is a bug shape worth a
    /// test rather than a comment.
    #[test]
    fn a_stall_is_never_asked_about_while_idle() {
        // `SETTLE` is ten re-check intervals, so an idle path that asked at all is caught here
        // several times over.
        let (_tx, rx) = script(vec![
            (SETTLE, Event::Started),
            (BLIP, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture::default();
        run(&rx, &mut capture, &|| true, false);

        assert_eq!(capture.calls, ["start", "finish"]);
        assert_eq!(
            capture.mic_stalled_before_the_first_session,
            Some(0),
            "the idle wait asked a capture with no session whether its microphone was dead"
        );
    }

    /// Nothing is recomputed until there is a session to protect.
    ///
    /// The re-check is a safety net for a lost *release* edge, not the detection mechanism:
    /// start edges arrive from listeners, and polling an idle machine would cost a walk of
    /// every audio process object every couple of seconds for nothing.
    #[test]
    fn the_idle_wait_never_rechecks() {
        // `SETTLE` is ten re-check intervals, so an idle path that polled at all would be
        // caught here several times over.
        let (_tx, rx) = script(vec![
            (SETTLE, Event::Started),
            (BLIP, Event::Stopped),
            (SETTLE, Event::Interrupt),
        ]);

        let mut capture = FakeCapture::default();
        let rechecks = Arc::clone(&capture.rechecks);
        run(
            &rx,
            &mut capture,
            &|| {
                rechecks.fetch_add(1, Ordering::SeqCst);
                true
            },
            false,
        );

        assert_eq!(capture.calls, ["start", "finish"]);
        assert_eq!(
            capture.rechecks_before_the_first_session,
            Some(0),
            "the idle wait recomputed the activity level"
        );
    }

    /// The plain mode's guarantee, pinned: every composer produces the pre-interface literal
    /// verbatim. This suite is what makes AC #4 a test rather than a review -- a rewording
    /// that slipped into a composer fails here instead of showing up in a user's scrollback.
    #[test]
    fn the_composers_reproduce_the_pre_interface_literals_verbatim() {
        let id = SessionId::parse("20260809-052600").unwrap();
        let dir = std::path::Path::new("/tmp/meethook/sessions/20260809-052600");

        // The stream class first: the faults and the diagnostics go to stderr, everything
        // the user reads goes to stdout.
        assert!(!Note::CalendarProblem(String::new()).to_stderr());
        assert!(!Note::Watching.to_stderr());
        assert!(!Note::AlreadyActive.to_stderr());
        assert!(
            !Note::SessionStarted {
                id: id.clone(),
                dir: dir.to_path_buf(),
                mic_rate: 48_000,
                mic_channels: 1,
                speaker_rate: 44_100,
            }
            .to_stderr()
        );
        assert!(!Note::DeviceChanged.to_stderr());
        assert!(!Note::MicStalled.to_stderr());
        assert!(!Note::Stopping.to_stderr());
        assert!(!Note::NoNewSession.to_stderr());
        assert!(
            !Note::Recorded {
                id: id.clone(),
                mic_secs: 1.0,
                speaker_secs: 2.0,
                dir: dir.to_path_buf(),
                meeting: None,
            }
            .to_stderr()
        );
        assert!(Note::GivingUp(3).to_stderr());
        assert!(Note::BeginFailed(String::new()).to_stderr());
        assert!(Note::FinishFailed(String::new()).to_stderr());
        assert!(Note::ActivityDebug(String::new()).to_stderr());

        // Then the wording itself, byte for byte.
        assert_eq!(
            Note::Watching.composed(),
            "Watching the default microphone. Press Ctrl-C to stop.\n"
        );
        assert_eq!(
            Note::AlreadyActive.composed(),
            "A microphone is already in use; recording immediately.\n"
        );
        assert_eq!(Note::Stopping.composed(), "Stopping...\n");
        assert_eq!(
            Note::NoNewSession.composed(),
            "That call has ended as well, so no new session was opened.\n"
        );
        assert_eq!(
            Note::DeviceChanged.composed(),
            format!("{DEVICE_CHANGED}\n")
        );
        assert_eq!(Note::MicStalled.composed(), format!("{MIC_STALLED}\n"));
        assert_eq!(
            Note::GivingUp(5).composed(),
            "Giving up on this call after 5 attempts; still watching.\n"
        );

        assert_eq!(session_id_line(&id), "Session 20260809-052600");
        assert_eq!(
            session_dir_line(dir.display()),
            "  /tmp/meethook/sessions/20260809-052600"
        );
        assert_eq!(
            mic_line(48_000, 1),
            "  mic       48000 Hz, 1 channel(s) reported by the input device"
        );
        assert_eq!(speaker_line(44_100), "  speaker   44100 Hz");

        assert_eq!(
            Note::SessionStarted {
                id: id.clone(),
                dir: dir.to_path_buf(),
                mic_rate: 48_000,
                mic_channels: 1,
                speaker_rate: 44_100,
            }
            .composed(),
            "Session 20260809-052600\n  /tmp/meethook/sessions/20260809-052600\n  mic       48000 Hz, 1 channel(s) reported by the input device\n  speaker   44100 Hz\nRecording... press Ctrl-C to stop.\n"
        );

        assert_eq!(
            recorded_lines(&id, 7.5, 7.5, dir.display(), None),
            "Recorded 20260809-052600 (7.5s mic, 7.5s speaker) to /tmp/meethook/sessions/20260809-052600\n"
        );
        assert_eq!(
            Note::Recorded {
                id: id.clone(),
                mic_secs: 7.5,
                speaker_secs: 7.5,
                dir: dir.to_path_buf(),
                meeting: None,
            }
            .composed(),
            recorded_lines(&id, 7.5, 7.5, dir.display(), None)
        );
    }

    /// `Recorded` carries only the meeting clause, never the meeting's other fields.
    ///
    /// Attendee names and addresses are written to `session.json` for speaker identification
    /// and are deliberately never printed: the note holds a [`super::MeetingLabel`] projection,
    /// so the assertion below can hold at all, and it pins the clause shape both presenters
    /// render.
    #[test]
    fn the_recorded_block_carries_only_the_meeting_clause() {
        use meethook_session::{Attendee, AttendeeStatus, Meeting};

        let meeting = Meeting::new(
            "EVENT-ABC".to_owned(),
            "Incident review".to_owned(),
            "Work".to_owned(),
            "2026-08-15T10:00:00Z".parse().unwrap(),
            "2026-08-15T11:00:00Z".parse().unwrap(),
        )
        .with_people(
            Some(Attendee {
                name: Some("Alan Turing".to_owned()),
                email: Some("alan@example.com".to_owned()),
                status: AttendeeStatus::Accepted,
                is_you: false,
            }),
            vec![Attendee {
                name: Some("Grace Hopper".to_owned()),
                email: Some("grace@example.com".to_owned()),
                status: AttendeeStatus::Accepted,
                is_you: true,
            }],
        )
        .with_invite(
            None,
            Some("Babbage Room".to_owned()),
            Some("Dial-in 555-0100, passcode 481516".to_owned()),
        );

        let id = SessionId::parse("20260809-052600").unwrap();
        let dir = std::path::Path::new("/tmp/meethook/sessions/20260809-052600");
        let label = super::MeetingLabel::from(&meeting);
        let block = recorded_lines(&id, 7.5, 7.5, dir.display(), Some(&label));

        // The clause, and only the clause: title plus caveat if the fit is weak.
        assert!(block.contains("  meeting   Incident review"), "{block}");
        for secret in [
            "Grace",
            "Hopper",
            "grace@example.com",
            "Alan",
            "Turing",
            "@",
            "Babbage",
            "Dial-in",
            "481516",
        ] {
            assert!(
                !block.contains(secret),
                "the finish block leaks {secret:?}: {block}"
            );
        }
    }

    /// Which presenter a run gets, given `--plain` and what the streams are attached to.
    ///
    /// AC #1, decidable with no terminal: a pipe on either end keeps driven runs on the line
    /// output, `--plain` forces it on a real terminal, and only an attached run without the
    /// flag opens the interface.
    const STREAMS: [Tty; 4] = [
        Tty {
            stdin: false,
            stdout: false,
        },
        Tty {
            stdin: true,
            stdout: false,
        },
        Tty {
            stdin: false,
            stdout: true,
        },
        Tty {
            stdin: true,
            stdout: true,
        },
    ];
    const ATTACHED: Tty = Tty {
        stdin: true,
        stdout: true,
    };

    #[test]
    fn a_pipe_on_either_end_is_the_lines() {
        for tty in STREAMS.iter().filter(|t| !t.is_attached()) {
            assert_eq!(presenter(false, *tty), Presenter::Lines);
            assert_eq!(presenter(true, *tty), Presenter::Lines);
        }
    }

    #[test]
    fn plain_forces_the_lines_even_on_a_terminal() {
        assert_eq!(presenter(true, ATTACHED), Presenter::Lines);
    }

    #[test]
    fn an_attached_run_without_plain_is_the_screen() {
        assert_eq!(presenter(false, ATTACHED), Presenter::Screen);
    }
}
