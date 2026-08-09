//! Subcommand bodies.
//!
//! `enroll` is a stub in this slice; `record` and `transcribe` do the real work. Both are
//! thin: the rules they enforce live in `meethook-record` and `meethook-transcribe`, where
//! they can be tested without a terminal.

use std::io::{self, Write};
use std::sync::mpsc;

use anyhow::{Context, Result, bail};
use meethook_models::ensure_model;
use meethook_record::{Recorder, preflight};
use meethook_session::{Paths, SessionId};
use meethook_transcribe::{SpeechToText, WHISPER_MODEL, WhisperEngine, run_batch};

/// Records one session, from launch until the process is interrupted.
///
/// Permissions are checked first and separately, so a missing TCC grant costs the user an
/// error message rather than a silently unrecorded meeting.
pub fn record(paths: &Paths) -> Result<()> {
    let authorized = preflight()?;
    let recorder = Recorder::new(authorized)?;

    // `ctrlc` runs its handler on a thread of its own rather than in signal context, so the
    // finalize path below is free to allocate and do I/O. The handler itself only signals.
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    ctrlc::set_handler(move || {
        let _ = stop_tx.send(());
    })
    .context("could not install the interrupt handler")?;

    let session = recorder.start(paths, &jiff::Zoned::now())?;

    // Printing both rates proves both engines actually came up; a user who sees only one
    // line knows something is wrong before the meeting rather than after it.
    println!("Session {}", session.id());
    println!("  {}", session.paths().dir().display());
    println!(
        "  mic       {} Hz, {} channel(s) reported by the input device",
        session.mic_sample_rate(),
        session.mic_channels()
    );
    println!("  speaker   {} Hz", session.speaker_sample_rate());
    println!("Recording... press Ctrl-C to stop.");

    // A dropped sender would mean the handler is gone, which should not happen; treating it
    // as a stop is the safe interpretation either way.
    let _ = stop_rx.recv();
    println!("Stopping...");

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
