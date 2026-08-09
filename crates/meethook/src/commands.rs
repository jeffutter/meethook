//! Subcommand bodies.
//!
//! `enroll` is a stub in this slice; `record` captures for real, and `transcribe` already
//! does the real discovery work, which is what makes the session contract observable from
//! the CLI rather than only from tests.

use std::sync::mpsc;

use anyhow::{Context, Result, bail};
use meethook_record::{Recorder, preflight};
use meethook_session::{Paths, SessionId, discover_sessions};

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

/// Lists discovered sessions and their classification.
///
/// Transcription itself lands in a later slice; the discovery result this prints is the
/// same value that slice will consume, so the listing is a temporary presentation of
/// permanent code, not throwaway scaffolding.
pub fn transcribe(paths: &Paths, session_ids: &[String], force: bool) -> Result<()> {
    let requested = parse_session_ids(session_ids)?;
    let discovered = discover_sessions(paths)?;

    if force {
        println!("--force accepted; re-transcription lands with transcription itself.");
    }

    let selected: Vec<_> = if requested.is_empty() {
        discovered.iter().collect()
    } else {
        discovered
            .iter()
            .filter(|session| requested.contains(&session.id))
            .collect()
    };

    // An id the user asked for that is not on disk is worth naming individually; silently
    // transcribing three of four requested sessions would look like success.
    for id in &requested {
        if !discovered.iter().any(|session| &session.id == id) {
            println!("{id}  not found");
        }
    }

    if selected.is_empty() && requested.is_empty() {
        println!("No sessions found in {}", paths.sessions_dir().display());
        return Ok(());
    }

    for session in selected {
        println!("{}  {}", session.id, session.classification);
    }

    println!("meethook transcribe: transcription is not implemented in this slice.");
    Ok(())
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
