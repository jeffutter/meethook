//! The `meethook` CLI.
//!
//! One binary, three subcommands. The spec describes `record` and `transcribe` as
//! "two binaries" meaning they share no process, no IPC, and no state -- only the on-disk
//! session contract. Subcommands preserve that: everything below talks to
//! [`meethook_session`] and to nothing else.

mod commands;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use meethook_session::Paths;

#[derive(Debug, Parser)]
#[command(
    name = "meethook",
    version,
    about = "Local meeting recorder and transcriber",
    long_about = None,
)]
struct Cli {
    /// meethook data directory holding sessions/, models/, and speakers.json
    /// (default: ~/meethook)
    #[arg(long, global = true, value_name = "PATH", env = "MEETHOOK_ROOT")]
    root: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Record meetings until interrupted
    ///
    /// Watches the default microphone and records each call as a session. Takes no
    /// options: there is nothing to configure that the tool cannot detect itself.
    Record,

    /// Transcribe recorded sessions
    ///
    /// With no session ids, every discovered session is considered.
    Transcribe {
        /// Session ids to transcribe; omit to consider all discovered sessions
        #[arg(value_name = "SESSION_ID")]
        session_ids: Vec<String>,

        /// Re-transcribe sessions that already have a transcript
        #[arg(long)]
        force: bool,
    },

    /// Name speakers that transcription could not identify
    ///
    /// With no session ids, every session with unresolved speakers is considered.
    Enroll {
        /// Session ids to enroll speakers for; omit to consider all sessions
        #[arg(value_name = "SESSION_ID")]
        session_ids: Vec<String>,

        /// Ask about every unresolved voice, including ones too quiet to be offered by default
        #[arg(long)]
        all: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let paths = Paths::new(resolve_root(cli.root)?);

    match cli.command {
        Command::Record => commands::record(&paths),
        Command::Transcribe { session_ids, force } => {
            commands::transcribe(&paths, &session_ids, force)
        }
        Command::Enroll { session_ids, all } => commands::enroll(&paths, &session_ids, all),
    }
}

/// Resolves the data directory: `--root`, else `$MEETHOOK_ROOT` (applied by clap), else
/// `~/meethook`.
///
/// The override exists so tests and manual checks can point at a scratch directory instead
/// of the user's real recordings.
fn resolve_root(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(root) = explicit {
        return Ok(root);
    }
    let home =
        std::env::home_dir().context("could not determine the home directory; pass --root")?;
    Ok(home.join("meethook"))
}
