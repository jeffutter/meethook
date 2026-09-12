//! The two `ENVIRONMENT:` blocks on the capture side, seen from outside the process.
//!
//! `MEETHOOK_ACTIVITY_DEBUG` and `MEETHOOK_CALENDAR_DEBUG` have no flags, and clap renders
//! `[env: ...]` only beside a flag that already exists -- upstream closed the request to render an
//! env-only variable (`clap-rs/clap#6039`) -- so they reach help as text alone. Text is easy to
//! lose silently: by rewording an epilogue, by moving it somewhere `-h` does not reach, or by
//! renaming a variable on one side of the boundary between `meethook-record` and the CLI. So this
//! asserts against the constants those crates actually read, never against a second copy of a name.
//!
//! What the wording has to hold up is subtler than `MEETHOOK_CPU`'s. Both of these enable on
//! *presence*, empty string included, which is the opposite of the CPU knob; copying its sentence
//! ("any non-empty value counts") onto them would ship a wrong claim. Hence the negative assertion
//! below, which refuses the copy rather than trusting the prose.
//!
//! The whole file is macOS-only, and not merely because `record` is: both variables are read only
//! by code that does not compile off macOS, and the constants naming them live in
//! `meethook-record`, which a Linux build of this package does not have in its graph at all, so an
//! ungated reference here would not even resolve there. `LINUX.md` already records that the
//! calendar variable "has nothing to print" off macOS, which is why the blocks disappear with the
//! behaviour rather than arriving with a disclaimer.

#![cfg(target_os = "macos")]

use std::process::{Command, Stdio};

/// Runs the built binary with `args` and returns its stdout, which is where help goes.
fn stdout_of(args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_meethook"))
        .args(args)
        // Nothing here answers a prompt, and a child that inherits the harness's terminal is a
        // child that could ask a question nobody is watching for.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("running the built meethook");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("help printed as utf-8")
}

/// Every help form of one subcommand, which are three different renderings: clap splits a doc
/// comment into `about` and `long_about`, so prose alone reaches only `--help`, while an epilogue
/// reaches `-h`, `--help` and the `help` subcommand alike.
fn help_forms(command: &str) -> [Vec<&str>; 3] {
    [
        vec!["help", command],
        vec![command, "-h"],
        vec![command, "--help"],
    ]
}

/// `record` honours both variables itself: the microphone watcher and the calendar lookup both run
/// inside a recording. Naming either without the rule attached is worth little, because a reader
/// who has just read `transcribe`'s block assumes its non-empty rule, and that rule is false here.
#[test]
fn record_names_both_debug_variables_and_says_presence_is_what_counts() {
    for args in help_forms("record") {
        let typed = args.join(" ");
        let help = stdout_of(&args);
        for name in [
            meethook_record::ACTIVITY_DEBUG_ENV_VAR,
            meethook_record::CALENDAR_DEBUG_ENV_VAR,
        ] {
            assert!(
                help.contains(name),
                "`{typed}` did not name {name}:\n{help}"
            );
        }
        assert!(
            help.contains("Presence turns it on"),
            "`{typed}` named the variables but left out what counts as setting them:\n{help}"
        );
        // The task that added `MEETHOOK_CPU`'s block worded its rule "any non-empty value
        // counts". True there, false here, and copied here it would tell a person running
        // `MEETHOOK_ACTIVITY_DEBUG=` that they had turned nothing on while lines printed anyway.
        assert!(
            !help.contains("non-empty"),
            "`{typed}` borrowed the CPU variable's non-empty rule for variables that enable on \
             presence:\n{help}"
        );
    }
}

/// `meeting` consults the same calendar lookup, so the same entry belongs to it too -- and only
/// that entry. The activity watcher has no part in correcting a label, and help that promised the
/// microphone log to someone running `meeting` would send them looking for lines that never print.
#[test]
fn meeting_names_the_calendar_variable_and_stays_silent_about_the_microphone_one() {
    for args in help_forms("meeting") {
        let typed = args.join(" ");
        let help = stdout_of(&args);
        assert!(
            help.contains(meethook_record::CALENDAR_DEBUG_ENV_VAR),
            "`{typed}` did not name the variable its own calendar lookup reads:\n{help}"
        );
        assert!(
            !help.contains(meethook_record::ACTIVITY_DEBUG_ENV_VAR),
            "`{typed}` named a variable nothing in `{typed}` reads:\n{help}"
        );
    }
}

/// Neither variable does anything outside the capture path, and clap propagates an epilogue
/// nowhere, so root help naming them would be a false promise about `enroll`, `speakers` and the
/// rest. This is what stops a future move of the blocks onto `Cli` -- which reads as the simpler
/// edit -- from quietly making root help overstate the whole CLI.
#[test]
fn the_root_help_stays_silent_about_variables_only_the_capture_path_reads() {
    for args in [&["--help"][..], &["-h"][..]] {
        let typed = args.join(" ");
        let help = stdout_of(args);
        for name in [
            meethook_record::ACTIVITY_DEBUG_ENV_VAR,
            meethook_record::CALENDAR_DEBUG_ENV_VAR,
        ] {
            assert!(
                !help.contains(name),
                "`meethook {typed}` named a variable its other subcommands ignore:\n{help}"
            );
        }
    }
}
