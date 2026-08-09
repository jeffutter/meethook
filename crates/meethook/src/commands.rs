//! Subcommand bodies.
//!
//! `enroll` is a stub in this slice; `record` and `transcribe` do the real work. Both are
//! thin: the rules they enforce live in `meethook-record` and `meethook-transcribe`, where
//! they can be tested without a terminal.

use std::io::{self, Write};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use meethook_models::ensure_model;
use meethook_record::{Activity, MicActivityWatcher, Recorder, preflight};
use meethook_session::{Paths, SessionId};
use meethook_transcribe::{SpeechToText, WHISPER_MODEL, WhisperEngine, run_batch};

/// How long microphone activity must stay stopped before a session is finalized.
///
/// Asymmetric with the start side, which has no debounce at all, on purpose: three seconds
/// of extra tail audio is harmless, while a premature finalize loses the end of a meeting
/// and a late start loses its opening.
const STOP_GRACE: Duration = Duration::from_secs(3);

/// Anything the record loop waits on.
///
/// One enum, one channel: a Ctrl-C during a recording has to be seen at the same instant
/// as a microphone edge, and two separate waits cannot both be blocking.
enum Event {
    Started,
    Stopped,
    Interrupt,
}

/// Records every call until the process is interrupted.
///
/// Permissions are checked first and separately, so a missing TCC grant costs the user an
/// error message rather than a silently unrecorded meeting.
///
/// The loop is deliberately forgiving of a failed session: a two-second false start that
/// produces a silent track prints an error and goes back to watching. Ending a day of
/// recording over one bad session would be a far worse failure than the session itself.
pub fn record(paths: &Paths) -> Result<()> {
    let authorized = preflight()?;
    let recorder = Recorder::new(authorized)?;
    let debug = std::env::var_os("MEETHOOK_ACTIVITY_DEBUG").is_some();

    let (tx, rx) = mpsc::channel::<Event>();

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
    let (_watcher, mut already_active) = MicActivityWatcher::start(move |activity| {
        let _ = activity_tx.send(match activity {
            Activity::Started => Event::Started,
            Activity::Stopped => Event::Stopped,
        });
    })?;

    println!("Watching the default microphone. Press Ctrl-C to stop.");
    if already_active {
        println!("A microphone is already in use; recording immediately.");
    }

    loop {
        // Idle. The already-active case skips the wait rather than holding out for an edge
        // that happened before this process started.
        if !already_active {
            match rx.recv() {
                Ok(Event::Started) => {}
                Ok(Event::Stopped) => continue,
                Ok(Event::Interrupt) | Err(_) => break,
            }
        }
        already_active = false;

        let started_at = Instant::now();
        let session = match recorder.start(paths, &jiff::Zoned::now()) {
            Ok(session) => session,
            Err(e) => {
                eprintln!("Could not start recording: {e}");
                println!("Watching the default microphone. Press Ctrl-C to stop.");
                continue;
            }
        };
        if debug {
            // This latency sits directly on the "no debounce, a late start loses the
            // opening" path, so it is worth being able to see rather than assume.
            eprintln!(
                "[activity] Recorder::start took {:.1} ms",
                started_at.elapsed().as_secs_f64() * 1000.0
            );
        }

        // Printing both rates proves both engines actually came up; a user who sees only
        // one line knows something is wrong before the meeting rather than after it.
        println!("Session {}", session.id());
        println!("  {}", session.paths().dir().display());
        println!(
            "  mic       {} Hz, {} channel(s) reported by the input device",
            session.mic_sample_rate(),
            session.mic_channels()
        );
        println!("  speaker   {} Hz", session.speaker_sample_rate());
        println!("Recording... press Ctrl-C to stop.");

        let interrupted = loop {
            match rx.recv() {
                Ok(Event::Stopped) => match await_end(&rx, STOP_GRACE) {
                    Outcome::CallEnded => break false,
                    Outcome::Interrupted => break true,
                    Outcome::Continue => {}
                },
                // A redundant start edge cannot happen while recording, but ignoring it is
                // the interpretation that keeps the session whole either way.
                Ok(Event::Started) => {}
                Ok(Event::Interrupt) | Err(_) => break true,
            }
        };
        println!("Stopping...");

        match session.finish() {
            Ok(recording) => println!(
                "Recorded {} ({:.1}s mic, {:.1}s speaker) to {}",
                recording.id,
                recording.mic.seconds(),
                recording.speaker.seconds(),
                recording.paths.dir().display()
            ),
            Err(e) => eprintln!("This session did not produce a usable recording: {e}"),
        }

        if interrupted {
            break;
        }
        println!("Watching the default microphone. Press Ctrl-C to stop.");
    }

    Ok(())
}

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
            Err(RecvTimeoutError::Timeout) => return Outcome::CallEnded,
            // Every sender is gone, so nothing can resume this session. Finalizing is the
            // only outcome that does not lose the audio already captured.
            Err(RecvTimeoutError::Disconnected) => return Outcome::CallEnded,
        }
    }
}

/// Transcribes recorded sessions.
///
/// Fully non-interactive, deliberately: this is meant to be aimed at a directory of
/// meetings and left alone. The model is acquired lazily through the factory below, so a
/// run that turns out to have nothing to do never pays for a download.
pub fn transcribe(paths: &Paths, session_ids: &[String], force: bool) -> Result<()> {
    let requested = parse_session_ids(session_ids)?;

    let models_dir = paths.models_dir();
    let mut open_engine = || -> std::result::Result<
        Box<dyn SpeechToText>,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let mut report = DownloadProgress::default();
        let model = ensure_model(&models_dir, &WHISPER_MODEL, &mut |done, total| {
            report.update(done, total)
        })?;
        Ok(Box::new(WhisperEngine::load(&model)?))
    };

    let stdout = io::stdout();
    let report = run_batch(
        paths,
        &requested,
        force,
        &mut open_engine,
        &mut stdout.lock(),
    )?;

    // Skips -- including orphans -- are normal and stay at exit 0; a session that genuinely
    // failed is what makes the run unsuccessful.
    if report.failed > 0 {
        bail!(
            "{} of {} session(s) failed to transcribe",
            report.failed,
            report.failed + report.transcribed
        );
    }
    Ok(())
}

/// Prints download progress on stderr, one line, rewritten in place.
///
/// stderr rather than stdout so the batch's own output stays a clean, greppable record of
/// what happened to each session. Throttled to whole percent because a 1.6 GB download at a
/// megabyte a chunk would otherwise emit sixteen hundred lines.
#[derive(Default)]
struct DownloadProgress {
    last_percent: u64,
    started: bool,
}

impl DownloadProgress {
    fn update(&mut self, done: u64, total: u64) {
        if !self.started {
            self.started = true;
            eprintln!(
                "Fetching {} ({:.1} GB, one time)",
                WHISPER_MODEL.file_name,
                total as f64 / 1e9
            );
        }
        let percent = (done * 100).checked_div(total).unwrap_or(0);
        if percent == self.last_percent && done < total {
            return;
        }
        self.last_percent = percent;
        let mut stderr = io::stderr();
        let _ = write!(stderr, "\r  {percent}%");
        if done >= total {
            let _ = writeln!(stderr);
        }
        let _ = stderr.flush();
    }
}

pub fn enroll(_paths: &Paths, session_ids: &[String]) -> Result<()> {
    parse_session_ids(session_ids)?;
    println!("meethook enroll: not implemented in this slice.");
    Ok(())
}

/// Validates positional ids before any filesystem work, so a typo fails immediately with a
/// message about the typo rather than as a confusing "not found" much later.
fn parse_session_ids(raw: &[String]) -> Result<Vec<SessionId>> {
    let mut ids = Vec::with_capacity(raw.len());
    for value in raw {
        match SessionId::parse(value) {
            Ok(id) => ids.push(id),
            Err(e) => bail!(e),
        }
    }
    Ok(ids)
}

/// The grace-period state machine, exercised without a microphone.
///
/// This is where "a blip does not split a session" and "a mute does not end one" are
/// decidable in an automated test: both reduce to activity resuming inside the grace
/// window, and both must leave the session running.
#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::{Duration, Instant};

    use super::{Event, Outcome, await_end};

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
}
