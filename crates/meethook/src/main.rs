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

/// Parses a loudness target, refusing one the measurement could never reach.
///
/// Same posture as [`parse_pan`]: the bounds are what the arithmetic admits, not what sounds
/// sensible, so an unwise-but-meaningful target is the user's to choose and an impossible one is
/// refused where it was typed. NaN falls out as a refusal because a range never contains it.
fn parse_target_lufs(value: &str) -> Result<f64, String> {
    parse_bounded(
        value,
        mixdown::TARGET_MIN_LUFS,
        mixdown::TARGET_MAX_LUFS,
        "LUFS",
    )
}

/// Parses a boost cap, refusing one outside the span any target could ask for.
fn parse_max_boost_db(value: &str) -> Result<f64, String> {
    parse_bounded(
        value,
        mixdown::MAX_BOOST_MIN_DB,
        mixdown::MAX_BOOST_MAX_DB,
        "dB",
    )
}

/// The shared body of the two levelling parsers: parse, range-check, and name the range and the
/// unit in the refusal.
fn parse_bounded(value: &str, min: f64, max: f64, unit: &str) -> Result<f64, String> {
    let parsed: f64 = value
        .parse()
        .map_err(|_| format!("`{value}` is not a number"))?;
    if !(min..=max).contains(&parsed) {
        return Err(format!("`{value}` is outside {min}..={max} {unit}"));
    }
    Ok(parsed)
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
    // Negative numbers reach the value parsers rather than clap's argument scanner. `--pan -0.1`
    // needs this to be refused by the range check and name the range, instead of being reported
    // as an unexpected argument `-0` -- and `--target-lufs -16` needs it simply to work, since a
    // loudness target is negative every time anyone types one. Safe because session ids never
    // begin with a hyphen, so nothing positional can be swallowed by it.
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

        /// Loudness each track is brought to before mixing, in LUFS
        ///
        /// Negative, always: 0 is digital full scale. The default is the podcast convention
        /// rather than EBU R 128's -23, which arrives too quiet on headphones and on a laptop
        /// speaker.
        #[arg(
            long,
            value_name = "LUFS",
            default_value_t = mixdown::TARGET_LUFS,
            value_parser = parse_target_lufs,
            allow_negative_numbers = true,
        )]
        target_lufs: f64,

        /// Most a quiet track may be turned up on its way to the target, in dB
        ///
        /// The ceiling exists because gain applied to a quiet track brings its hiss and room
        /// rumble up with the voice. Raise it to match a very quiet microphone at the cost of
        /// its noise floor; a track needing more than this is left short of the target rather
        /// than amplified past it.
        #[arg(
            long,
            value_name = "DB",
            default_value_t = mixdown::MAX_BOOST_DB,
            value_parser = parse_max_boost_db,
        )]
        max_boost_db: f64,
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

    /// Correct, or clear, the meeting a session was labelled with
    ///
    /// The calendar match is a guess over start and end times, and a guess is sometimes wrong:
    /// a session that began twenty minutes into a booked hour is either a late join or an
    /// unrelated call, and a double-booked hour resolves to whichever invite the recorder
    /// preferred. This is how a person who was there settles it.
    ///
    /// With neither flag, prints the label the session carries and the meetings around it,
    /// numbered, and writes nothing. --event attaches one of them; --clear records that the
    /// session was not recorded during a meeting at all and needs no calendar access. Either
    /// way the label is marked as one a human chose, so nothing guesses over it afterwards,
    /// and the session's transcript.md is brought in line in the same run.
    Meeting {
        /// The session to relabel, exactly as its directory is named
        #[arg(value_name = "SESSION_ID")]
        session_id: String,

        /// Attach the Nth meeting of the list this command prints when given neither flag
        ///
        /// The number beside the meeting in that list, not a calendar identifier: nothing
        /// shows you one of those, and they are not typeable.
        #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
        event: Option<u32>,

        /// Record that this session was not recorded during any meeting
        ///
        /// The other half of a correction, and the half that needs no calendar: a session
        /// matched to a meeting that was never held is fixed on a machine with the grant
        /// refused.
        #[arg(long, conflicts_with = "event")]
        clear: bool,
    },
}

/// `enroll`'s options.
///
/// A struct rather than fields on the variant because there are eight of them and four are
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

    /// Ask line by line, never opening the full-screen interface
    ///
    /// What you get anyway when either end of the command is a pipe rather than a terminal,
    /// so a script, a shell pipeline or CI needs nothing here. Give it on a real terminal to
    /// keep the plain question-and-answer prompt, or to leave the interface out of a bug
    /// report. --name outranks it: a name given up front is not asked about at all.
    #[arg(long)]
    plain: bool,
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
            target_lufs,
            max_boost_db,
        } => commands::transcribe(
            &paths,
            &session_ids,
            force,
            template,
            mixdown::Settings {
                bitrate_bps: bitrate,
                pan,
                normalization: mixdown::Normalization {
                    target_lufs,
                    max_boost_db,
                },
            },
        ),
        Command::Enroll(args) => commands::enroll(&paths, &args, template),
        Command::Speakers => commands::speakers(&paths),
        Command::Forget {
            name,
            reference,
            yes,
        } => commands::forget(&paths, &name, reference, yes, template),
        Command::Meeting {
            session_id,
            event,
            clear,
        } => commands::meeting(&paths, &session_id, event, clear, template),
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

    /// The mixdown settings `meethook transcribe` would run with, given these arguments.
    fn transcribe_mixdown(args: &[&str]) -> mixdown::Settings {
        match Cli::try_parse_from(args).expect("should parse").command {
            Command::Transcribe {
                bitrate,
                pan,
                target_lufs,
                max_boost_db,
                ..
            } => mixdown::Settings {
                bitrate_bps: bitrate,
                pan,
                normalization: mixdown::Normalization {
                    target_lufs,
                    max_boost_db,
                },
            },
            other => panic!("expected a transcribe command, got {other:?}"),
        }
    }

    /// Asserts the refusal happened at the edge, and mentions the offending value.
    fn refused(args: &[&str]) -> String {
        let error = Cli::try_parse_from(args).expect_err("should have been refused");
        error.to_string()
    }

    /// The options `meethook enroll` would run with, given these arguments.
    fn enroll_args(args: &[&str]) -> EnrollArgs {
        match Cli::try_parse_from(args).expect("should parse").command {
            Command::Enroll(args) => args,
            other => panic!("expected an enroll command, got {other:?}"),
        }
    }

    #[test]
    fn the_mixdown_defaults_are_the_constants_the_listening_run_settled() {
        // Compared against `Settings::default()` rather than against `32000`, `0.3`, `-16` and
        // `18`, so that changing a value settled by listening stays a one-line edit in
        // `mixdown.rs` and does not need this test edited to match. This is also what pins
        // "omitting every flag encodes exactly what the previous build encoded".
        assert_eq!(
            transcribe_mixdown(&["meethook", "transcribe"]),
            mixdown::Settings::default()
        );
    }

    #[test]
    fn every_mixdown_setting_is_overridable() {
        let settings = transcribe_mixdown(&[
            "meethook",
            "transcribe",
            "--bitrate",
            "48000",
            "--pan",
            "0.5",
            "--target-lufs",
            "-20",
            "--max-boost-db",
            "24",
        ]);
        assert_eq!(settings.bitrate_bps, 48_000);
        assert!(
            (settings.pan - 0.5).abs() < f32::EPSILON,
            "{}",
            settings.pan
        );
        assert_eq!(settings.normalization.target_lufs, -20.0);
        assert_eq!(settings.normalization.max_boost_db, 24.0);
    }

    #[test]
    fn a_negative_loudness_target_is_a_value_rather_than_a_flag() {
        // The ordinary way this option gets typed: a loudness target is negative every time.
        // Without `allow_negative_numbers` on the subcommand, clap reads `-20` as an unknown
        // short flag and the option is unusable, so this pins the setting rather than leaving
        // it resting on the comment beside it.
        for value in ["-16", "-23.5", "-0.5"] {
            let settings = transcribe_mixdown(&["meethook", "transcribe", "--target-lufs", value]);
            assert_eq!(
                settings.normalization.target_lufs,
                value.parse::<f64>().unwrap(),
                "--target-lufs {value} did not survive parsing"
            );
        }
    }

    #[test]
    fn a_levelling_value_outside_what_the_measurement_admits_is_refused_at_the_edge() {
        // Both ends of both options. The floor on a target is the loudness gate: below it no
        // block survives measurement, so the target names a loudness nothing can be brought to.
        // The ceiling on a cap is the widest correction any legal target can ask for, above
        // which the cap can never bind.
        for (option, value) in [
            ("--target-lufs", mixdown::TARGET_MIN_LUFS - 1.0),
            ("--target-lufs", mixdown::TARGET_MAX_LUFS + 1.0),
            ("--max-boost-db", mixdown::MAX_BOOST_MIN_DB - 1.0),
            ("--max-boost-db", mixdown::MAX_BOOST_MAX_DB + 1.0),
        ] {
            let typed = value.to_string();
            let message = refused(&["meethook", "transcribe", option, &typed]);
            assert!(
                message.contains(&typed),
                "{option} {typed} not named in: {message}"
            );
        }
    }

    #[test]
    fn a_levelling_value_that_is_not_a_number_is_refused() {
        for option in ["--target-lufs", "--max-boost-db"] {
            let message = refused(&["meethook", "transcribe", option, "NaN"]);
            assert!(message.contains("NaN"), "{option}: {message}");
        }
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

    /// What `meethook meeting` would do, given these arguments: the session, and the two flags
    /// that decide between showing, attaching and clearing.
    fn meeting_args(args: &[&str]) -> (String, Option<u32>, bool) {
        match Cli::try_parse_from(args).expect("should parse").command {
            Command::Meeting {
                session_id,
                event,
                clear,
            } => (session_id, event, clear),
            other => panic!("expected a meeting command, got {other:?}"),
        }
    }

    /// The preview is what you get for typing the least, which is the shape this command is
    /// built around: nothing is written until a flag says which correction to make.
    #[test]
    fn a_meeting_command_with_no_flag_is_the_listing() {
        assert_eq!(
            meeting_args(&["meethook", "meeting", "20260809-052600"]),
            ("20260809-052600".to_owned(), None, false)
        );
    }

    #[test]
    fn a_meeting_can_be_attached_by_number_or_cleared() {
        assert_eq!(
            meeting_args(&["meethook", "meeting", "20260809-052600", "--event", "2"]),
            ("20260809-052600".to_owned(), Some(2), false)
        );
        assert_eq!(
            meeting_args(&["meethook", "meeting", "20260809-052600", "--clear"]),
            ("20260809-052600".to_owned(), None, true)
        );
    }

    /// Attaching a meeting and recording that there was none are contradictory, and a run that
    /// silently picked one of them would write the wrong answer to `session.json` -- so clap
    /// refuses the pair rather than this being an ordering decision inside the command.
    #[test]
    fn attaching_and_clearing_at_once_is_refused() {
        let message = refused(&[
            "meethook",
            "meeting",
            "20260809-052600",
            "--event",
            "1",
            "--clear",
        ]);
        assert!(message.contains("--clear"), "{message}");
        assert!(message.contains("--event"), "{message}");
    }

    /// The listing is 1-based, as it prints. `--event 0` is refused where the range can be
    /// named rather than reaching the library as an index nobody offered.
    #[test]
    fn an_event_number_below_the_first_one_is_refused_at_the_edge() {
        let message = refused(&["meethook", "meeting", "20260809-052600", "--event", "0"]);
        assert!(message.contains('0'), "{message}");
        assert!(message.contains('1'), "the range is not named: {message}");
    }

    /// A session id is required: there is no "all sessions" reading of a correction, and
    /// guessing one would be a write to a session nobody named.
    #[test]
    fn a_meeting_command_needs_a_session() {
        let message = refused(&["meethook", "meeting"]);
        assert!(message.contains("SESSION_ID"), "{message}");
    }

    /// Typing the least still means exactly what it meant before `--plain` existed. Every
    /// field is asserted rather than the new one alone, so a stray `default_value` or a
    /// reordering that changed what a bare `enroll` does would be caught here.
    #[test]
    fn a_bare_enroll_is_unchanged_by_the_plain_flag_existing() {
        let args = enroll_args(&["meethook", "enroll", "20260809-052600"]);
        assert_eq!(args.session_ids, ["20260809-052600"]);
        assert_eq!(args.voice, None);
        assert_eq!(args.at, None);
        assert_eq!(args.name, None);
        assert!(!args.all);
        assert!(!args.correct);
        assert!(!args.force_reference);
        assert!(!args.plain, "--plain is on without being asked for");
    }

    /// `--plain` is compatible with everything, deliberately: `--plain --name Alice` is
    /// contradictory but harmless -- the name wins and nothing prompts -- and refusing it
    /// would break a driver that passes `--plain` unconditionally for safety and adds
    /// `--name` when it happens to know the answer, which is the caller this flag is for.
    #[test]
    fn plain_sits_alongside_every_other_enroll_flag() {
        let args = enroll_args(&[
            "meethook",
            "enroll",
            "20260809-052600",
            "--voice",
            "Unknown 2",
            "--name",
            "Alice",
            "--all",
            "--correct",
            "--force-reference",
            "--plain",
        ]);
        assert_eq!(args.voice.as_deref(), Some("Unknown 2"));
        assert_eq!(args.name.as_deref(), Some("Alice"));
        assert!(args.all);
        assert!(args.correct);
        assert!(args.force_reference);
        assert!(args.plain);

        // `--at` is the other selector, so it gets its own run rather than sharing that one.
        let args = enroll_args(&[
            "meethook",
            "enroll",
            "20260809-052600",
            "--at",
            "12:34",
            "--plain",
        ]);
        assert!(args.at.is_some(), "--at did not survive --plain");
        assert!(args.plain);
    }

    /// The one conflict `enroll` does have, pinned so that adding a flag beside it did not
    /// quietly drop it: two selectors would name two different voices.
    #[test]
    fn a_voice_and_a_timestamp_at_once_is_still_refused() {
        let message = refused(&[
            "meethook",
            "enroll",
            "20260809-052600",
            "--voice",
            "2",
            "--at",
            "12:34",
        ]);
        assert!(message.contains("--voice"), "{message}");
        assert!(message.contains("--at"), "{message}");
    }
}
