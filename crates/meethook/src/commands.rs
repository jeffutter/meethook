//! Subcommand bodies.
//!
//! `record` and `enroll` are stubs in this slice; `transcribe` already does the real
//! discovery work, which is what makes the session contract observable from the CLI rather
//! than only from tests.

use anyhow::{Result, bail};
use meethook_session::{Paths, SessionId, discover_sessions};

pub fn record(_paths: &Paths) -> Result<()> {
    println!("meethook record: not implemented in this slice.");
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
