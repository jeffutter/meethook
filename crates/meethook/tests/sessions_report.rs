//! `meethook sessions` seen from outside the process: the exit status, the whole listing, and the
//! claim that a report which promises to write nothing leaves no byte behind.
//!
//! The wording itself is pinned byte-for-byte in `commands.rs`'s own tests, where the report is
//! called through its writer seam. What these tests add is everything a seam cannot see: that the
//! subcommand exists with no options on every platform, that it exits 0 on roots whose contents it
//! cannot fully describe, that an unreadable `sessions/` is the one case that does not, and that
//! the liveness probe -- which reaches the kernel, not a value passed in -- agrees with a recorder
//! running in another process.

mod common;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use common::mixed_root;
use meethook_session::{Acquisition, Paths, RecordLock, recording_in_progress};

/// Every file under `root`, by path relative to it and by bytes: the whole state a run could have
/// touched.
///
/// Whole-root means whole, which is worth knowing about `record`: `<root>/record.lock` would
/// appear here for a run that took it, and would make the "this command wrote nothing" assertion
/// below fail confusingly. This command never takes that lock -- it only asks the kernel who holds
/// it, which opens the file read-only and writes nothing -- but a test that acquired it in-process
/// would still trip on the holder metadata, so the snapshot runs against a root nobody holds.
fn snapshot(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn walk(dir: &Path, base: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, base, out);
            } else {
                out.insert(
                    path.strip_prefix(base).unwrap().to_path_buf(),
                    std::fs::read(&path).unwrap(),
                );
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// The built binary, pointed at this root and given the one subcommand this file exercises.
fn meethook(root: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_meethook"))
        .arg("--root")
        .arg(root)
        .arg("sessions")
        // Nothing here answers a prompt, and a child that inherits the harness's terminal is a
        // child that could ask a question nobody is watching for.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("running the built meethook")
}

/// Exit 0 and the whole listing, including the loss figure read out of the truncated track's own
/// header rather than quoted from a constant.
#[test]
fn a_mixed_root_is_listed_in_full_and_the_run_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let paths = mixed_root(dir.path());

    let output = meethook(dir.path());
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!(
            concat!(
                "4 session(s) in {dir}: 1 transcribed, 1 valid, 2 orphaned\n",
                "\n",
                "20260809-052500  orphaned\n",
                "    no session.json: no transcript is possible.\n",
                "    That file held the single clock both tracks share, so neither can be placed ",
                "on a common timeline however much of either one plays.\n",
                "    The mic track declares 0.2 s more audio than the file holds, and that part is ",
                "not on disk.\n",
                "    The speaker track holds 0.1 s past the end its header declares; a player that ",
                "trusts that number stops there, so that part does not play.\n",
                "    Nothing about this needs fixing: the audio that reached disk is kept as ",
                "recorded.\n",
                "20260809-052600  orphaned\n",
                "    no session.json: no transcript is possible.\n",
                "    That file held the single clock both tracks share, so neither can be placed ",
                "on a common timeline however much of either one plays.\n",
                "    The mic track never reached disk, so there is nothing to place on the ",
                "timeline.\n",
                "    The speaker track never reached disk, so there is nothing to place on the ",
                "timeline.\n",
                "    Nothing about this needs fixing: the audio that reached disk is kept as ",
                "recorded.\n",
                "20260809-052700  transcribed\n",
                "20260809-052800  valid\n",
            ),
            dir = paths.sessions_dir().display()
        )
    );
}

/// The command's own promise, checked on disk rather than believed: every file under the root,
/// before and after, identical.
#[test]
fn reporting_what_became_of_every_session_changes_nothing_anywhere() {
    let dir = tempfile::tempdir().unwrap();
    mixed_root(dir.path());
    let before = snapshot(dir.path());
    assert!(
        !before.is_empty(),
        "a fixture that writes nothing proves nothing"
    );

    let output = meethook(dir.path());
    assert!(output.status.success());
    assert_eq!(snapshot(dir.path()), before);
}

/// An absent `sessions/` is the first-run case, not an error, and it prints one line naming the
/// directory it looked in.
#[test]
fn a_root_that_has_never_recorded_reports_that_and_exits_zero() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::new(dir.path());

    let output = meethook(dir.path());
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!(
            "no sessions under {} -- meethook record writes one\n",
            paths.sessions_dir().display()
        )
    );
}

/// A recorder in another process is the case the whole hedge exists for: the child has to reach
/// the same kernel answer this test holds, and then say nothing about any directory stopping.
#[test]
fn a_recorder_holding_the_root_elsewhere_is_said_once_and_nothing_is_called_interrupted() {
    let dir = tempfile::tempdir().unwrap();
    let paths = mixed_root(dir.path());
    let guard = match RecordLock::acquire(&paths) {
        Ok(Acquisition::Held(guard)) => guard,
        Ok(Acquisition::Taken(holder)) => panic!("expected to hold the lock: {holder:?}"),
        Err(e) => panic!("acquiring the lock: {e}"),
    };

    let output = meethook(dir.path());
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains(&recording_in_progress()), "{text}");
    assert!(text.contains("20260809-052500  orphaned"), "{text}");
    assert!(!text.contains("no transcript is possible"), "{text}");
    assert!(!text.contains("The mic track"), "{text}");
    // Held until the child has finished answering, which is the whole point of the test.
    drop(guard);
}

/// The one input that does leave a nonzero status. The report's whole claim is the scope it
/// scanned; a `sessions/` it cannot open has to fail loudly rather than print a census of nothing,
/// which is the same rule `speakers` applies to a scan it could not finish.
#[test]
fn a_sessions_path_that_is_not_a_directory_fails_naming_the_path() {
    let dir = tempfile::tempdir().unwrap();
    let paths = Paths::new(dir.path());
    std::fs::write(paths.sessions_dir(), b"not a directory").unwrap();

    let output = meethook(dir.path());
    assert!(!output.status.success());
    let text = String::from_utf8_lossy(&output.stderr);
    assert!(
        text.contains(&paths.sessions_dir().display().to_string()),
        "{text}"
    );
}
