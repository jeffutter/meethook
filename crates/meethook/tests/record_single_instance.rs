//! The guard a live `record` puts on its root, seen from outside the process that takes it.
//!
//! Two things are proven here, from opposite directions:
//!
//! - `record` itself refuses (macOS only, because the subcommand does not exist elsewhere): the
//!   built binary is spawned against a root whose lock this test holds, and it has to come back
//!   non-zero naming the holder. Driven with `spawn` and a bounded `try_wait` rather than
//!   `output()` on purpose -- if the guard ever stopped being taken, the child would walk into a
//!   TCC permission prompt and this test would hang for its timeout instead of failing.
//! - Everything else is unaffected by a live recording, on every platform. Differential, not
//!   end-to-end: a real `transcribe` needs 1.6 GB of model weights, which is not something to
//!   download in a test, so each sibling command runs twice over otherwise-identical roots and
//!   the locked run has to behave exactly like the unlocked one.

use std::path::Path;
use std::process::{Child, Command, Output, Stdio};
// Only the refused-`record` test compares roots byte for byte, and that test cannot exist off
// macOS because the subcommand does not. Off macOS these would be dead code, which `-D warnings`
// refuses to let sit.
#[cfg(target_os = "macos")]
use std::collections::BTreeMap;
#[cfg(target_os = "macos")]
use std::path::PathBuf;
use std::time::{Duration, Instant};

use meethook_session::{Acquisition, Paths, RecordLock};

/// How long a refused `record` is given to refuse.
///
/// Generous, because the failure mode being guarded against is a child that got *further* than
/// the lock and is now waiting on a human at a permission dialog: the wait has to end, and the
/// child has to die, so that the test reports the regression rather than hanging on it.
const REFUSAL_TIMEOUT: Duration = Duration::from_secs(90);

/// Every file under `root`, by path relative to it and by bytes.
#[cfg(target_os = "macos")]
fn files_under(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    fn walk(dir: &Path, root: &Path, out: &mut BTreeMap<PathBuf, Vec<u8>>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, root, out);
            } else {
                out.insert(
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    std::fs::read(&path).unwrap(),
                );
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// Holds `root`'s lock for as long as the returned guard lives.
fn hold_lock(root: &Path) -> RecordLock {
    match RecordLock::acquire(&Paths::new(root)) {
        Ok(Acquisition::Held(lock)) => lock,
        Ok(Acquisition::Taken(holder)) => {
            panic!("expected to be the only holder of {root:?}, found {holder:?}")
        }
        Err(e) => panic!("acquiring the lock for {root:?}: {e}"),
    }
}

/// Runs the built binary with `args`, without blocking on it.
fn spawn_meethook(root: &Path, args: &[&str]) -> Child {
    Command::new(env!("CARGO_BIN_EXE_meethook"))
        .arg("--root")
        .arg(root)
        .args(args)
        // Nothing here answers a prompt, and a child that inherits the test harness's terminal
        // is a child that could ask a question nobody is watching for.
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawning the built meethook")
}

/// Waits for a spawned run to finish, killing it if it never does.
///
/// Returns the collected output, or panics with whatever the child managed to print -- which is
/// the evidence a hung child is most likely to be leaving behind.
fn finish(mut child: Child, context: &str) -> Output {
    let deadline = Instant::now() + REFUSAL_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {}
            Err(e) => panic!("waiting on {context}: {e}"),
        }
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            let partial = match child.wait_with_output() {
                Ok(output) => String::from_utf8_lossy(&output.stderr).into_owned(),
                Err(_) => String::new(),
            };
            panic!(
                "{context} never finished within {REFUSAL_TIMEOUT:?}; stderr so far:\n{partial}"
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    child
        .wait_with_output()
        .expect("collecting the finished child's output")
}

/// Blocks until the run is done. Only for commands with no plausible reason to block: the ones
/// that reach a prompt would take [`REFUSAL_TIMEOUT`] and get reported through it.
fn run_to_completion(root: &Path, args: &[&str]) -> Output {
    finish(
        spawn_meethook(root, args),
        &format!("meethook {}", args.join(" ")),
    )
}

/// The outcome of a run with the scratch root's own path edited out, so two roots holding the
/// same fixture can be compared byte for byte.
fn outcome(output: &Output, root: &Path) -> (Option<i32>, String, String) {
    let scrub = |bytes: &[u8]| {
        String::from_utf8_lossy(bytes).replace(&root.display().to_string(), "<root>")
    };
    (
        output.status.code(),
        scrub(&output.stdout),
        scrub(&output.stderr),
    )
}

/// AC#1 and AC#2: a second `record` against a root that is already recording refuses, fails,
/// says who it lost to, and leaves the root otherwise exactly as it found it.
#[cfg(target_os = "macos")]
#[test]
fn a_second_record_names_the_instance_that_already_holds_the_root() {
    let dir = tempfile::tempdir().unwrap();
    let _lock = hold_lock(dir.path());
    let mut before = files_under(dir.path());

    // `--plain` so that even a run that got past the lock would not try to take the terminal
    // this test is running in.
    let child = spawn_meethook(dir.path(), &["record", "--plain"]);
    let output = finish(child, "a second `record`");

    assert!(
        !output.status.success(),
        "a refused `record` must not look like a successful recording to a script"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("meethook record cannot start"),
        "the refusal has to read like every other startup refusal; got: {stderr}"
    );
    assert!(
        stderr.contains(&format!("pid {}", std::process::id())),
        "the refusal has to name the live holder's pid; got: {stderr}"
    );
    let exe = std::env::current_exe().unwrap();
    assert!(
        stderr.contains(exe.file_name().unwrap().to_str().unwrap()),
        "the refusal has to say how the holder was started; got: {stderr}"
    );
    assert!(
        stderr.contains("record.lock"),
        "and where the lock lives, so the message is debuggable; got: {stderr}"
    );

    // AC#2: refusing is not a reason to touch anything. The lock file is the one thing this run
    // had an opinion about, and even there its opinion was to say nothing -- so it is compared
    // apart, byte for byte, rather than excused from the comparison as a whole.
    let mut after = files_under(dir.path());
    let lock = PathBuf::from("record.lock");
    let lock_before = before.remove(&lock);
    let lock_after = after.remove(&lock);
    assert_eq!(
        lock_before, lock_after,
        "a refused `record` rewrote what the live holder wrote about itself"
    );
    assert_eq!(
        before, after,
        "a refused `record` left something behind in the root besides the lock file"
    );
}

/// AC#3: the guard is `record`'s alone. Every sibling command sees the same thing whether or not
/// a recording holds the root, which is what lets `enroll` bring a stale transcript up to date
/// while the next call is being captured.
///
/// Runs on every platform: the commands themselves are not macOS-only, and neither is the lock.
#[test]
fn a_live_recording_does_not_disturb_any_other_command() {
    // Each of these reaches real work and stops at the first missing thing, so none of them can
    // reach the network or a human: a session id that matches nothing sends `transcribe` to
    // "not found" rather than to the gigabytes of model weights a real session would need, and
    // `meeting --clear` is the half of that command that asks macOS for nothing at all.
    const COMMANDS: &[&[&str]] = &[
        &["speakers"],
        &["enroll", "--list"],
        &["forget", "Nobody", "--yes"],
        &["meeting", "20260101-000000", "--clear"],
        &["transcribe", "20260101-000000"],
    ];

    for args in COMMANDS {
        let quiet = tempfile::tempdir().unwrap();
        let busy = tempfile::tempdir().unwrap();
        let _lock = hold_lock(busy.path());

        let without = run_to_completion(quiet.path(), args);
        let with = run_to_completion(busy.path(), args);

        assert_eq!(
            outcome(&without, quiet.path()),
            outcome(&with, busy.path()),
            "`meethook {}` behaves differently against a root a recording is holding",
            args.join(" ")
        );
        let stderr = String::from_utf8_lossy(&with.stderr);
        assert!(
            !stderr.contains("is held by"),
            "`meethook {}` noticed the lock at all: {stderr}",
            args.join(" ")
        );
    }
}
