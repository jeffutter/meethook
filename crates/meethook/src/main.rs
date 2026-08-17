//! The `meethook` CLI.
//!
//! One binary, five subcommands. The spec describes `record` and `transcribe` as
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

    /// Template every transcript.md is rendered through
    /// (default: transcript.md.jinja in the data directory, else meethook's built-in one)
    ///
    /// Global rather than an option of `transcribe`, deliberately: `enroll` and `forget`
    /// rewrite transcripts they did not write, and a per-command flag would let one of them
    /// silently revert a transcript to the built-in shape. A template named here that is
    /// missing or malformed is an error; it never falls back.
    #[arg(long, global = true, value_name = "PATH", env = "MEETHOOK_TEMPLATE")]
    template: Option<PathBuf>,

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

        /// Ask about one voice and nothing else
        ///
        /// Either the number in "Unknown 3" -- not the cluster id, which nothing else shows
        /// you -- or the name that voice currently reads as. Needs exactly one session id,
        /// since a voice belongs to one session. Reaches a voice that is too quiet to be
        /// offered or that already has a name, so --all and --correct add nothing to it.
        #[arg(long, value_name = "VOICE")]
        voice: Option<String>,

        /// Ask about every unresolved voice, including ones too quiet to be offered by default
        #[arg(long)]
        all: bool,

        /// Also ask about voices already named, so a wrong identification can be corrected
        #[arg(long)]
        correct: bool,

        /// Store a reference for every name given, even from a voice too short to make a
        /// reliable one; without this a quiet voice is named in its own session only
        #[arg(long)]
        force_reference: bool,
    },

    /// Report who is enrolled and what each stored recording of them is naming
    ///
    /// A person can hold several recordings, and in speakers.json they are indistinguishable
    /// from each other -- which is a problem at the point one of them has to go. For each
    /// recording this names the voices, by session and by the label they read as, that would
    /// stop reading that person if it were removed; one naming nothing in any session on disk
    /// is the one to drop. Reads every transcribed session under --root and writes nothing.
    ///
    /// Takes no options on purpose: the report's whole claim is the scope it scanned, so it
    /// also prints how many sessions it read and names any it could not.
    Speakers,

    /// Remove a stored recording of somebody, or remove them entirely
    ///
    /// With --reference, drops the one recording that number addresses in meethook speakers;
    /// without it, drops every recording of that person, which is that person removed. Prints
    /// what the removal costs first -- the voices that stop reading them, the ones that start
    /// reading somebody else, and the ones that gain a name -- and writes nothing until --yes.
    ///
    /// A removed reference cannot be rebuilt: the audio it was made from is not consulted and
    /// may be long deleted. The transcripts of every session whose labelling changes are brought
    /// in line in the same run.
    Forget {
        /// Who to remove, exactly as meethook speakers prints the name
        #[arg(value_name = "NAME")]
        name: String,

        /// Remove only this one of their recordings, by the number meethook speakers gives it
        #[arg(long, value_name = "N")]
        reference: Option<usize>,

        /// Perform the removal; without this the consequences are printed and nothing is written
        #[arg(long)]
        yes: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let paths = Paths::new(resolve_root(cli.root)?);
    let template = cli.template.as_deref();

    match cli.command {
        Command::Record => commands::record(&paths),
        Command::Transcribe { session_ids, force } => {
            commands::transcribe(&paths, &session_ids, force, template)
        }
        Command::Enroll {
            session_ids,
            voice,
            all,
            correct,
            force_reference,
        } => commands::enroll(
            &paths,
            &session_ids,
            voice.as_deref(),
            all,
            correct,
            force_reference,
            template,
        ),
        Command::Speakers => commands::speakers(&paths),
        Command::Forget {
            name,
            reference,
            yes,
        } => commands::forget(&paths, &name, reference, yes, template),
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
