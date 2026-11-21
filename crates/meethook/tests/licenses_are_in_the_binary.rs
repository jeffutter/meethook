//! `meethook licenses`, seen from outside the process.
//!
//! The release is one file, and the licenses of what it contains (ONNX Runtime, MLX, whisper.cpp,
//! the Rust crates) require their notices to accompany it. So the notices are built into the
//! binary, and this pins the three things that make that true: the command exists, it needs no
//! data directory (it must work with `$HOME` unset, on a machine meethook never ran on), and it
//! prints the real text rather than a stub.

use std::process::{Command, Stdio};

fn run(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_meethook"))
        .args(args)
        // Nothing the command should need: not the data directory, not a terminal.
        .env_remove("HOME")
        .env_remove("MEETHOOK_ROOT")
        .stdin(Stdio::null())
        .output()
        .expect("running the built meethook")
}

#[test]
fn prints_the_notices_without_a_data_directory() {
    let output = run(&["licenses"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).expect("notices are utf-8");
    assert!(text.starts_with("# Third-party licenses"));
    assert!(text.contains("ONNX Runtime"));
    assert!(text.contains("Permission is hereby granted"));
    // The generated file is about half a megabyte; a stub would not be.
    assert!(text.len() > 200_000, "only {} bytes", text.len());
}

#[test]
fn is_listed_in_help() {
    let help = String::from_utf8(run(&["--help"]).stdout).unwrap();
    assert!(help.contains("licenses"), "{help}");
}
