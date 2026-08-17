//! The `meethook` CLI.
//!
//! One binary, five subcommands. The spec describes `record` and `transcribe` as
//! "two binaries" meaning they share no process, no IPC, and no state -- only the on-disk
//! session contract. Subcommands preserve that: everything below talks to
//! [`meethook_session`] and to nothing else.

mod commands;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use meethook_session::{Paths, TranscriptTime};
use meethook_transcribe::mixdown;

/// Parses a pan width, refusing one outside the range rather than clamping it.
///
/// The mixdown's constant-power panning clamps internally, which is right for a value that
/// crate computed and wrong for one a user typed: clamping turns `--pan 30` into a hard pan
/// and says nothing about having done so. NaN falls out as a refusal too, because a range
/// never contains it.
fn parse_pan(value: &str) -> Result<f32, String> {
    let pan: f32 = value
        .parse()
        .map_err(|_| format!("`{value}` is not a number"))?;
    if !(mixdown::PAN_MIN..=mixdown::PAN_MAX).contains(&pan) {
        return Err(format!(
            "`{value}` is outside {}..={}",
            mixdown::PAN_MIN,
            mixdown::PAN_MAX
        ));
    }
    Ok(pan)
}

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
    // Negative numbers reach the value parsers rather than clap's argument scanner, so that
    // `--pan -0.1` is refused by the range check and names the range, instead of being
    // reported as an unexpected argument `-0`. Safe here because no option takes a value that
    // could be confused with a flag, and session ids never begin with a hyphen.
    #[command(allow_negative_numbers = true)]
    Transcribe {
        /// Session ids to transcribe; omit to consider all discovered sessions
        #[arg(value_name = "SESSION_ID")]
        session_ids: Vec<String>,

        /// Re-transcribe sessions that already have a transcript
        #[arg(long)]
        force: bool,

        /// Bitrate of the meeting.opus mixdown, in bits per second
        ///
        /// The default was settled by listening to a real meeting rather than taken from a
        /// table: the step below it was already clean, and this sits one above for margin.
        ///
        /// An option of `transcribe` rather than a global one, unlike --template: nothing
        /// re-writes meeting.opus after the fact, so there is no second command that could
        /// silently disagree with the one that produced it.
        #[arg(
            long,
            value_name = "BPS",
            default_value_t = mixdown::BITRATE_BPS,
            value_parser = clap::value_parser!(u32)
                .range(i64::from(mixdown::BITRATE_MIN_BPS)..=i64::from(mixdown::BITRATE_MAX_BPS)),
        )]
        bitrate: u32,

        /// How far from centre each track is panned: 0.0 is mono, 1.0 is hard left and right
        ///
        /// Also settled by listening. Wide enough to tell the two sides apart, narrow enough
        /// that neither ear is ever doing the listening alone over an hour.
        #[arg(
            long,
            value_name = "WIDTH",
            default_value_t = mixdown::PAN_POSITION,
            value_parser = parse_pan,
        )]
        pan: f32,
    },

    /// Name speakers that transcription could not identify
    ///
    /// With no session ids, every session with unresolved speakers is considered.
    Enroll(EnrollArgs),

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

/// `enroll`'s options.
///
/// A struct rather than fields on the variant because there are seven of them and three are
/// bools: named at the one call site below, they cannot be transposed, which two adjacent
/// `Option<String>`s passed positionally certainly could be.
#[derive(Debug, Args)]
pub struct EnrollArgs {
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

    /// Ask about whoever was speaking at this moment of the session, and nothing else
    ///
    /// A timestamp exactly as transcript.md prints it -- MM:SS, minutes not wrapped at 60, so
    /// 90:05 is an hour and a half in -- for naming somebody you can see in the transcript
    /// without working out which "Unknown N" they are. Names the whole voice and not just that
    /// turn, and says how much it renamed. Needs exactly one session id, like --voice, which it
    /// is the alternative to rather than a companion of.
    #[arg(long, value_name = "MM:SS", conflicts_with = "voice")]
    at: Option<TranscriptTime>,

    /// Answer with this name instead of prompting
    ///
    /// Needs --at or --voice: a name given up front is never shown the voice it lands on, so
    /// there has to be exactly one voice, chosen by you. Everything it writes -- the reference,
    /// the session-scoped name, the transcript -- is what typing the same name at the prompt
    /// would have written.
    #[arg(long, value_name = "NAME")]
    name: Option<String>,

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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let paths = Paths::new(resolve_root(cli.root)?);
    let template = cli.template.as_deref();

    match cli.command {
        Command::Record => commands::record(&paths),
        Command::Transcribe {
            session_ids,
            force,
            bitrate,
            pan,
        } => commands::transcribe(
            &paths,
            &session_ids,
            force,
            template,
            mixdown::Settings {
                bitrate_bps: bitrate,
                pan,
            },
        ),
        Command::Enroll(args) => commands::enroll(&paths, &args, template),
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The bitrate and pan `meethook transcribe` would run with, given these arguments.
    fn transcribe_mixdown(args: &[&str]) -> (u32, f32) {
        match Cli::try_parse_from(args).expect("should parse").command {
            Command::Transcribe { bitrate, pan, .. } => (bitrate, pan),
            other => panic!("expected a transcribe command, got {other:?}"),
        }
    }

    /// Asserts the refusal happened at the edge, and mentions the offending value.
    fn refused(args: &[&str]) -> String {
        let error = Cli::try_parse_from(args).expect_err("should have been refused");
        error.to_string()
    }

    #[test]
    fn the_mixdown_defaults_are_the_constants_the_listening_run_settled() {
        // Compared against the constants rather than against `32000` and `0.3`, so that
        // changing a value settled by listening stays a one-line edit in `mixdown.rs` and
        // does not need this test edited to match. This is also what pins "omitting both
        // flags encodes exactly what the previous build encoded".
        assert_eq!(
            transcribe_mixdown(&["meethook", "transcribe"]),
            (mixdown::BITRATE_BPS, mixdown::PAN_POSITION)
        );
    }

    #[test]
    fn both_settings_are_overridable() {
        let (bitrate, pan) = transcribe_mixdown(&[
            "meethook",
            "transcribe",
            "--bitrate",
            "48000",
            "--pan",
            "0.5",
        ]);
        assert_eq!(bitrate, 48_000);
        assert!((pan - 0.5).abs() < f32::EPSILON, "{pan}");
    }

    #[test]
    fn a_bitrate_outside_what_opus_accepts_is_refused_at_the_edge() {
        // Both ends, because a range check that only guards one is the usual way this is got
        // wrong -- and being refused here is the whole point: the alternative is the encoder
        // failing partway through a batch that has already spent an hour on recognition.
        let under = mixdown::BITRATE_MIN_BPS - 1;
        let over = mixdown::BITRATE_MAX_BPS + 1;
        assert!(
            refused(&["meethook", "transcribe", "--bitrate", &under.to_string()]).contains("6000")
        );
        assert!(
            refused(&["meethook", "transcribe", "--bitrate", &over.to_string()]).contains("510000")
        );
    }

    #[test]
    fn a_pan_outside_the_range_is_refused_rather_than_clamped() {
        // `constant_power` would happily clamp all three of these into a legal pan, which is
        // exactly the silence this refusal exists to break.
        for value in ["1.5", "-0.1", "NaN"] {
            let message = refused(&["meethook", "transcribe", "--pan", value]);
            assert!(message.contains(value), "{value} not named in: {message}");
        }
    }
}
