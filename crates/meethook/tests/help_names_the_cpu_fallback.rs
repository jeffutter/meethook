//! The one promise `meethook transcribe` makes about a variable it has no flag for, seen from
//! outside the process.
//!
//! `MEETHOOK_CPU` is the way forward out of the "no usable Metal device" refusal, and clap renders
//! `[env: ...]` only beside a flag that already exists -- upstream closed the request to render an
//! env-only variable (`clap-rs/clap#6039`) -- so here it survives in help as text alone. Text is
//! easy to lose silently: by rewording the epilogue, by moving it somewhere `-h` does not reach, or
//! by renaming the variable on one side of the boundary between `meethook-transcribe` and the CLI.
//! So this asserts against `CPU_ENV_VAR`, the constant `gpu.rs` actually reads, rather than against
//! a second copy of the name: a rename that leaves the help advertising a variable nothing reads
//! fails here instead of in someone's sandbox.

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

/// All three help forms are pinned because they are three different renderings: clap splits a doc
/// comment into `about` and `long_about`, so prose alone reaches only `--help`, while an epilogue
/// reaches `-h`, `--help` and the `help` subcommand alike. And naming the variable is worth little
/// without the rule attached to it -- any non-empty value opts in, `0` included, which is not what
/// a reader assumes of a boolean-looking name.
#[test]
fn all_help_forms_name_the_cpu_variable_and_what_counts_as_setting_it() {
    for args in [
        &["transcribe", "-h"][..],
        &["transcribe", "--help"][..],
        &["help", "transcribe"][..],
    ] {
        let typed = args.join(" ");
        let help = stdout_of(args);
        assert!(
            help.contains(meethook_transcribe::CPU_ENV_VAR),
            "`{typed}` did not name the variable:\n{help}"
        );
        assert!(
            help.contains("non-empty"),
            "`{typed}` named the variable but left out what counts as setting it:\n{help}"
        );
    }
}

/// The block belongs to `transcribe` because `transcribe` is the only command that honours the
/// variable, and clap does not propagate an epilogue downward. So the root help naming it would be
/// a false promise about `enroll`, `sessions` and the rest; this is what keeps a future decision to
/// move the epilogue onto `Cli` -- which reads as the simpler edit -- from quietly making root help
/// lie.
#[test]
fn the_root_help_stays_silent_about_a_variable_only_transcribe_honours() {
    for args in [&["--help"][..], &["-h"][..]] {
        let typed = args.join(" ");
        let help = stdout_of(args);
        assert!(
            !help.contains(meethook_transcribe::CPU_ENV_VAR),
            "`meethook {typed}` named a variable its other subcommands ignore:\n{help}"
        );
    }
}
