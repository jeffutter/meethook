//! Subcommand bodies.
//!
//! All four are thin: the rules they enforce live in `meethook-record`,
//! `meethook-transcribe` and `meethook-enroll`, where they can be tested without a terminal.
//! What is left here is the terminal itself -- printing, prompting, and playing audio --
//! which is exactly the part no test can decide.

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use meethook_enroll::{
    Answer, Enrolment, Interviewer, Offer, Voice, VoiceSelector, run_enroll, run_speakers,
    write_clip,
};
use meethook_models::{ModelSpec, ensure_model};
use meethook_record::{Activity, MicActivityWatcher, Recorder, RunningSession, preflight};
use meethook_session::{Paths, SessionId};
use meethook_transcribe::{
    Attribution, EMBEDDING_MODEL, Engines, OnnxDiarizer, SEGMENTATION_MODEL, SILERO_VAD_MODEL,
    WHISPER_MODEL, WhisperEngine, run_batch,
};

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
    /// How often the activity level is recomputed *while a session is live*.
    ///
    /// A safety net rather than the detection mechanism: every edge is expected to arrive
    /// from a listener, and this exists only because a release edge can be lost outright
    /// when the recomputation behind a notification reads the world a moment too early. See
    /// `MicActivityWatcher::recheck`. Nothing is polled while idle.
    recheck: Duration,
}

impl Timing {
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

/// Anything the record loop waits on.
///
/// One enum, one channel: a Ctrl-C during a recording has to be seen at the same instant
/// as a microphone edge, and two separate waits cannot both be blocking.
enum Event {
    Started,
    Stopped,
    /// The default input device moved. Not an edge of the activity predicate: it says the
    /// microphone engine is now bound to the wrong device, which every wait below has to have
    /// a deliberate answer for.
    InputDeviceChanged,
    Interrupt,
}

/// What the record loop needs from a capture backend.
///
/// The loop's whole responsibility is sequencing -- when to open a session, when to hold on
/// through a blip, when to finalize, when to give up -- and none of that needs a microphone
/// to decide. This two-method seam is what makes it decidable in `cargo test`: the live
/// implementation drives a [`Recorder`], and the test one records the order it was called
/// in. Everything either of them knows about audio stays on their side of it.
trait Capture {
    /// Begins a session and announces it.
    fn start(&mut self) -> Result<()>;
    /// Finalizes the current session and reports what it produced.
    fn finish(&mut self) -> Result<()>;
}

/// The live backend: one session at a time, plus the user-facing report of it.
struct SessionCapture<'a> {
    recorder: &'a Recorder,
    paths: &'a Paths,
    debug: bool,
    running: Option<RunningSession>,
}

impl Capture for SessionCapture<'_> {
    fn start(&mut self) -> Result<()> {
        let started_at = Instant::now();
        let session = self.recorder.start(self.paths, &jiff::Zoned::now())?;
        if self.debug {
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

        self.running = Some(session);
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        // Nothing running is not an error. The loop only finishes a start it saw succeed, so
        // defining the case away here is cheaper than a branch that can only ever be wrong.
        let Some(session) = self.running.take() else {
            return Ok(());
        };
        let recording = session.finish()?;
        println!(
            "Recorded {} ({:.1}s mic, {:.1}s speaker) to {}",
            recording.id,
            recording.mic.seconds(),
            recording.speaker.seconds(),
            recording.paths.dir().display()
        );
        Ok(())
    }
}

/// Records every call until the process is interrupted.
///
/// Permissions are checked first and separately, so a missing TCC grant costs the user an
/// error message rather than a silently unrecorded meeting.
///
/// The loop is deliberately forgiving of a failed session: a two-second false start that
/// produces a silent track prints an error and goes back to watching. Ending a day of
/// recording over one bad session would be a far worse failure than the session itself.
///
/// Everything past the setup lives in [`record_loop`], which is where the sequencing is and
/// where it can be tested without a microphone.
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
    let (watcher, already_active) = MicActivityWatcher::start(move |activity| {
        let _ = activity_tx.send(match activity {
            Activity::Started => Event::Started,
            Activity::Stopped => Event::Stopped,
            Activity::InputDeviceChanged => Event::InputDeviceChanged,
        });
    })?;

    announce_watching();
    if already_active {
        println!("A microphone is already in use; recording immediately.");
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
    );

    Ok(())
}

fn announce_watching() {
    println!("Watching the default microphone. Press Ctrl-C to stop.");
}

/// Sequences one session per detected call until the process is interrupted.
///
/// `recheck` recomputes the activity level from the world, delivering any edge it finds
/// onto `rx` before returning that level. It is called from three places, for three reasons:
/// while a session is live it is the safety net for a release edge that was lost outright,
/// inside [`begin`] it is the level a start retry is driven by, and after a session finalized
/// by an input-device change it decides whether there is still a call to open a new one for.
///
/// `already_active` skips the first idle wait, because a call that was already in progress
/// when this process started will not produce a start edge.
fn record_loop(
    rx: &Receiver<Event>,
    capture: &mut dyn Capture,
    recheck: &dyn Fn() -> bool,
    already_active: bool,
    timing: Timing,
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
                Ok(Event::Interrupt) | Err(_) => break,
            }
        }
        already_active = false;

        match begin(rx, capture, recheck, timing) {
            Begin::Recording => {}
            Begin::Abandoned => {
                announce_watching();
                continue;
            }
            Begin::Interrupted => break,
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
                Err(RecvTimeoutError::Timeout) => {
                    let _ = recheck();
                }
                // Every sender is gone, so no edge can arrive again.
                Err(RecvTimeoutError::Disconnected) => break Recording::Interrupted,
            }
        };

        // Said before the finish line rather than after it, so the two session reports the
        // user is about to see read as a consequence of the swap rather than as a fault.
        match outcome {
            Recording::DeviceChanged => println!(
                "The default input device changed. The microphone engine is bound to the \
                 device that went away, so this session is being finalized and a new one \
                 opened on the new device."
            ),
            Recording::Ended | Recording::Interrupted => println!("Stopping..."),
        }

        if let Err(e) = capture.finish() {
            eprintln!("This session did not produce a usable recording: {e}");
        }

        match outcome {
            Recording::Interrupted => break,
            Recording::Ended => announce_watching(),
            // The level, recomputed from the world, is what keeps a swap that coincides with
            // the call ending from opening a session for a call that is already over. When it
            // is still up, `already_active` is exactly the "record without waiting for a start
            // edge that has already happened" case the top of this loop already handles, so
            // the restart inherits `begin`'s bounded retry and its Ctrl-C responsiveness with
            // no second start path. A false answer has already sent `Stopped`, which the idle
            // wait above consumes harmlessly.
            Recording::DeviceChanged => {
                if recheck() {
                    already_active = true;
                    continue;
                }
                println!("That call has ended as well, so no new session was opened.");
                announce_watching();
            }
        }
    }
}

/// How the inner recording loop ended.
///
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
}

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
) -> Begin {
    for attempt in 1..=timing.attempts {
        match capture.start() {
            Ok(()) => return Begin::Recording,
            // Only the first failure is printed in full, and only the give-up line follows
            // it: five copies of one message is noise to read past rather than information.
            Err(e) if attempt == 1 => eprintln!("Could not start recording: {e:#}"),
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

    eprintln!(
        "Giving up on this call after {} attempts; still watching.",
        timing.attempts
    );
    Begin::Abandoned
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
            // Wait out the remainder too. This session is already ending and its engine is
            // already dead, so there is nothing left for a new device to capture into it --
            // returning `Continue` would rescue a session that has no audio coming.
            Ok(Event::InputDeviceChanged) => {}
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
/// meetings and left alone. All four models are acquired lazily through the one factory
/// below, so a run that turns out to have nothing to do never pays for a download.
pub fn transcribe(paths: &Paths, session_ids: &[String], force: bool) -> Result<()> {
    let requested = parse_session_ids(session_ids)?;

    let models_dir = paths.models_dir();
    let mut open_engine =
        || -> std::result::Result<Engines, Box<dyn std::error::Error + Send + Sync>> {
            // Smallest first, so that a first run reaches its slowest download last -- by which
            // point everything else is known to be in place. The VAD weights are 885 KB,
            // diarization's two models are 32 MB, and Whisper is 1.6 GB.
            let silero = fetch(&models_dir, &SILERO_VAD_MODEL)?;
            let segmentation = fetch(&models_dir, &SEGMENTATION_MODEL)?;
            let embedding = fetch(&models_dir, &EMBEDDING_MODEL)?;
            let whisper = fetch(&models_dir, &WHISPER_MODEL)?;

            let diarizer = OnnxDiarizer::load(&segmentation, &embedding)?;
            if !diarizer.accelerated() {
                // Correct but several times slower. Worth a line: the alternative is a user
                // wondering why transcribing a meeting suddenly takes minutes.
                eprintln!("Note: CoreML declined these graphs; diarization is running on CPU.");
            }

            let asr = WhisperEngine::load(&whisper, &silero)?;
            if !asr.accelerated() {
                // Only reachable via MEETHOOK_CPU, so this confirms an explicit choice rather
                // than reporting a surprise -- and says out loud what that choice costs.
                eprintln!(
                    "Note: MEETHOOK_CPU is set; speech recognition is running on the CPU and \
                     will be much slower."
                );
            }

            Ok(Engines {
                asr: Box::new(asr),
                diarizer: Box::new(diarizer),
            })
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

/// Acquires one model, reporting its download identically to every other model's.
fn fetch(
    models_dir: &std::path::Path,
    spec: &'static ModelSpec,
) -> meethook_models::Result<PathBuf> {
    let mut report = DownloadProgress::new(spec);
    ensure_model(models_dir, spec, &mut |done, total| {
        report.update(done, total)
    })
}

/// Prints download progress on stderr, one line, rewritten in place.
///
/// stderr rather than stdout so the batch's own output stays a clean, greppable record of
/// what happened to each session. Throttled to whole percent because a 1.6 GB download at a
/// megabyte a chunk would otherwise emit sixteen hundred lines.
struct DownloadProgress {
    spec: &'static ModelSpec,
    last_percent: u64,
    started: bool,
}

impl DownloadProgress {
    fn new(spec: &'static ModelSpec) -> DownloadProgress {
        DownloadProgress {
            spec,
            last_percent: 0,
            started: false,
        }
    }

    fn update(&mut self, done: u64, total: u64) {
        if !self.started {
            self.started = true;
            // Megabytes for the two diarization graphs, gigabytes for Whisper: "0.0 GB"
            // against a 6 MB file reads as a bug in the downloader.
            let size = if total >= 1_000_000_000 {
                format!("{:.1} GB", total as f64 / 1e9)
            } else {
                format!("{:.0} MB", total as f64 / 1e6)
            };
            eprintln!("Fetching {} ({size}, one time)", self.spec.file_name);
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

pub fn enroll(
    paths: &Paths,
    session_ids: &[String],
    voice: Option<&str>,
    all: bool,
    correct: bool,
    force_reference: bool,
) -> Result<()> {
    let requested = parse_session_ids(session_ids)?;
    let mut terminal = Terminal::default();
    // Unlike a session id there is nothing to validate here: a selector that matches nothing is
    // answered against the session's actual voices, which is a better message than anything
    // this edge could produce without having read them.
    let selector = voice.map(VoiceSelector::from);
    // Named at the one production call site, so which flag answers which question is readable
    // here rather than positional.
    let offer = Offer {
        quiet: all,
        named: correct,
    };
    // A separate axis from `offer`: that one decides which voices are asked about, this one
    // what an answer to a quiet voice writes.
    let enrolment = if force_reference {
        Enrolment::Always
    } else {
        Enrolment::AboveTheFloor
    };
    let report = run_enroll(
        paths,
        &requested,
        selector.as_ref(),
        offer,
        enrolment,
        &mut terminal,
        &mut io::stdout(),
    )?;

    println!(
        "\n{} named, {} skipped, {} session(s) passed over",
        report.named, report.skipped, report.passed_over
    );
    // A sub-count of `named` rather than a separate outcome, so this reads as a qualification
    // of the line above rather than as more voices. Says where the name went, because "named"
    // on its own would leave a user expecting those people to be recognised in the next
    // meeting -- which is exactly what a session-scoped name does not do.
    if report.session_only > 0 {
        println!(
            "{} of those named in their own session only, with no reference stored -- \
             each said why on its own line above",
            report.session_only
        );
    }
    // A voice left as it was found is a kept identification, not an unanswered question, and
    // only ever arises under `--correct`.
    if report.kept > 0 {
        println!("{} identification(s) kept as they were", report.kept);
    }
    // Only when there were any: a run that asked about everything should not end on a line
    // about the nothing it held back.
    if report.held_back > 0 {
        println!(
            "{} quieter voice(s) not offered -- meethook enroll --all asks about those too",
            report.held_back
        );
    }
    // An answer declined because honouring it would have taken a name off another voice. Only
    // when there were any, like the two above, and worth its own line rather than being folded
    // into the skips: those voices are still unnamed *and* the user has already answered them,
    // so the next step is to read the refusal lines rather than to answer again.
    if report.refused > 0 {
        println!(
            "{} answer(s) refused: honouring them would have un-named another voice",
            report.refused
        );
    }
    // Skips and pass-overs are ordinary; a request that could not be served -- a session that
    // could not be read, an id that is not on disk, a `--voice` matching no voice or several --
    // is what makes the run unsuccessful, exactly as in `transcribe`. Each has already printed
    // the line saying which it was, so this only has to make the exit status say so too.
    if report.failed > 0 {
        bail!("{} enroll request(s) could not be served", report.failed);
    }
    Ok(())
}

/// Reports who is enrolled and what each of their stored recordings is currently naming.
///
/// The thinnest of the four, and read-only: everything printed comes back from one call, and
/// the only decision left here is the exit status.
pub fn speakers(paths: &Paths) -> Result<()> {
    let scan = run_speakers(paths, &mut io::stdout())?;

    // A report whose entire claim is its completeness must not exit 0 while admitting it could
    // not read three sessions. The same rule `enroll` and `transcribe` apply to a request they
    // could not serve, and as there, each one has already printed the line saying which it was
    // -- above the whole listing, which is still printed.
    if !scan.unreadable.is_empty() {
        bail!(
            "{} session(s) could not be read, so this listing is incomplete",
            scan.unreadable.len()
        );
    }
    Ok(())
}

/// The interactive half of `enroll`: what a prompt looks like, and how a clip gets played.
///
/// Everything about *which* voice is asked about, and what an answer writes, is on the other
/// side of the `Interviewer` seam in `meethook-enroll`. What is left here needs a person in
/// front of it, so none of it is tested; keeping it this small is what makes that acceptable.
#[derive(Default)]
struct Terminal {
    /// Where clips are written for the player, created on first use and removed when the run
    /// ends. `afplay` has no start offset, so playing part of a recording means handing it a
    /// file that contains only that part.
    clips: Option<tempfile::TempDir>,
}

impl Terminal {
    /// Plays a clip and waits for it to finish, reporting anything that stopped it.
    ///
    /// Never fatal. A missing `afplay`, a full temp directory, a truncated `speaker.wav` --
    /// none of them are a reason to stop asking, because the snippets above the prompt are
    /// often enough to recognise somebody on their own.
    fn play(&mut self, clip: &[f32]) {
        if clip.is_empty() {
            println!("    (no audio for this voice)");
            return;
        }

        let played = || -> Result<()> {
            let dir = match &self.clips {
                Some(dir) => dir,
                None => self.clips.insert(tempfile::tempdir()?),
            };
            let path = dir.path().join("clip.wav");
            write_clip(&path, clip)?;
            let status = Command::new("afplay").arg(&path).status()?;
            if !status.success() {
                bail!("afplay exited with {status}");
            }
            Ok(())
        }();
        if let Err(e) = played {
            println!("    (could not play the clip: {e})");
        }
    }
}

impl Interviewer for Terminal {
    fn identify(&mut self, voice: &Voice<'_>) -> Answer {
        // An already-named voice is a different question -- "is this right", not "who is
        // this" -- and asking the second one with a name already on the screen invites the
        // user to type that name straight back in. The basis clause says which question it is.
        let (label, basis) = match voice.attribution {
            Attribution::Identified { name, similarity } => (
                name.as_str(),
                format!(", identified at {similarity:.2} confidence"),
            ),
            // No confidence to print, and saying so matters: this name is here because
            // somebody typed it, and it is recorded against this session rather than as a
            // reference, so it will not follow the person into the next meeting.
            Attribution::Assigned { name } => {
                (name.as_str(), ", named for this session".to_string())
            }
            Attribution::Unknown(label) => (label.as_str(), String::new()),
        };
        // The position sits right after the session id, at a fixed column: what it is for is
        // being findable by eye on every header, which a suffix after a variable-length speech
        // time would not be.
        println!(
            "\n{}  {}  {label} -- {} of speech{basis}",
            voice.session,
            voice.position,
            speech(voice.speech_seconds)
        );
        if voice.snippets.is_empty() {
            println!("    (nothing was transcribed for this voice)");
        }
        for snippet in &voice.snippets {
            println!("    \"{snippet}\"");
        }
        self.play(voice.clip);

        // Enter is `Answer::Skip` either way, and `Skip` writes nothing -- which is exactly
        // what keeping an identification means, so no new answer variant is needed.
        if voice.attribution.is_named() {
            print!(
                "Who is this? (name to correct, Enter to keep {}, Ctrl-D to stop) ",
                voice.attribution.label()
            );
        } else {
            print!("Who is this? (name, Enter to skip, Ctrl-D to stop) ");
        }
        // Without this the question sits in the buffer behind the answer.
        if io::stdout().flush().is_err() {
            return Answer::Quit;
        }

        // End of input is somebody stopping, not a failure: every name accepted so far is
        // already on disk, and so is the transcript it changed. Ctrl-C, which kills the
        // process outright rather than arriving here, is safe for the same reason.
        let mut line = String::new();
        match io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => Answer::Quit,
            Ok(_) if line.trim().is_empty() => Answer::Skip,
            Ok(_) => Answer::Named(line.trim().to_string()),
        }
    }
}

/// How much this voice said, in the units a person would say it in.
fn speech(seconds: f64) -> String {
    let seconds = seconds.round() as u64;
    match seconds / 60 {
        0 => format!("{seconds}s"),
        minutes => format!("{minutes}m {:02}s", seconds % 60),
    }
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

    use super::{Capture, Event, Outcome, Timing, await_end, record_loop};

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
    }

    impl Capture for FakeCapture {
        fn start(&mut self) -> super::Result<()> {
            self.calls.push("start");
            if self.rechecks_before_the_first_session.is_none() {
                self.rechecks_before_the_first_session = Some(self.rechecks.load(Ordering::SeqCst));
            }
            if self.failing_starts > 0 {
                self.failing_starts -= 1;
                anyhow::bail!("deliberate start failure");
            }
            Ok(())
        }

        fn finish(&mut self) -> super::Result<()> {
            self.calls.push("finish");
            self.finished_at.get_or_insert_with(Instant::now);
            Ok(())
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
        record_loop(&rx, &mut capture, &|| true, false, LOOP_TIMING);

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
        record_loop(&rx, &mut capture, &|| true, false, LOOP_TIMING);

        assert_eq!(capture.calls, ["start", "finish"]);
    }

    /// Ctrl-C mid-recording finalizes before it exits. The alternative is a truncated WAV
    /// with no header and no `session.json`, which is unrecoverable audio.
    #[test]
    fn an_interrupt_while_recording_finalizes_first() {
        let (_tx, rx) = script(vec![(BLIP, Event::Started), (BLIP, Event::Interrupt)]);

        let mut capture = FakeCapture::default();
        record_loop(&rx, &mut capture, &|| true, false, LOOP_TIMING);

        assert_eq!(capture.calls, ["start", "finish"]);
    }

    /// A call already in progress at startup is recorded without waiting for an edge that
    /// has already happened.
    #[test]
    fn an_already_active_microphone_records_without_a_start_edge() {
        let (_tx, rx) = script(vec![(BLIP, Event::Stopped), (SETTLE, Event::Interrupt)]);

        let mut capture = FakeCapture::default();
        record_loop(&rx, &mut capture, &|| true, true, LOOP_TIMING);

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
        record_loop(&rx, &mut capture, &|| true, false, LOOP_TIMING);

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
        record_loop(&rx, &mut capture, &|| true, false, LOOP_TIMING);

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
        record_loop(&rx, &mut capture, &|| false, false, LOOP_TIMING);

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
        record_loop(&rx, &mut capture, &recheck, false, LOOP_TIMING);

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
        record_loop(&rx, &mut capture, &|| true, false, LOOP_TIMING);

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
        record_loop(&rx, &mut capture, &|| false, false, LOOP_TIMING);

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
        record_loop(&rx, &mut capture, &|| true, false, LOOP_TIMING);

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
        record_loop(&rx, &mut capture, &|| true, false, LOOP_TIMING);

        assert_eq!(capture.calls, ["start", "start", "finish"]);
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
        record_loop(
            &rx,
            &mut capture,
            &|| {
                rechecks.fetch_add(1, Ordering::SeqCst);
                true
            },
            false,
            LOOP_TIMING,
        );

        assert_eq!(capture.calls, ["start", "finish"]);
        assert_eq!(
            capture.rechecks_before_the_first_session,
            Some(0),
            "the idle wait recomputed the activity level"
        );
    }
}
