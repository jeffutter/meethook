//! The single-instance guard for `record`.
//!
//! Two live `meethook record` processes classify each other as somebody else's meeting: a
//! plain Rust binary reports no bundle id to CoreAudio, so neither is excluded by the
//! mic-activity predicate and neither ever sees a stop edge. Observed on hardware
//! (TASK-005.03.01), where a leftover instance held another recording open for 41 minutes.
//! The predicate excludes a second instance by *executable path*, which does not hold for an
//! installed user: a Nix install moves `/nix/store/<hash>-meethook-<version>/bin/meethook` at
//! every upgrade, so a recorder that survived one looks like a foreign process to the next one
//! forever. That class is defined away here instead: `record` acquires this lock as its first
//! action and refuses to start while a live `record` holds it.
//!
//! # Why a kernel lock rather than a pidfile
//!
//! Whether the lock is held is answered by the kernel -- `EAGAIN` from a non-blocking
//! acquisition means somebody alive holds it -- and never by reading a pid out of the file.
//! Every pidfile liveness check is racy in some way that pid reuse, PID namespaces and
//! reparenting make unavoidable, whereas the kernel drops this lock when the holder dies,
//! however it dies. That is why there is no stale-file cleanup anywhere in this module: the
//! file persists and the lock does not outlive its owner, so "stale" is not a state the file
//! can be in.
//!
//! An OFD (`F_OFD_SETLK`) lock rather than a POSIX `F_SETLK` one, because POSIX record locks
//! are owned per-process-per-inode instead of per open file description. Two consequences of
//! that ownership model matter here, both reproduced locally while writing this: opening and
//! closing the lock path itself anywhere in the holding process silently drops a POSIX lock --
//! which is exactly what rewriting the holder metadata below would do -- and an atomic rename
//! over the lock path installs a fresh inode under the name while the holder keeps locking the
//! unlinked old one, which makes the guard decorative. OFD locks survive both. The rename
//! hazard is also why `record.lock` never goes near `write_atomic`.
//!
//! Nothing ever unlinks the file either, which is not squeamishness about deletion but the
//! standard answer to the two-holders-locking-two-inodes race: once a file can be deleted while
//! held, a second candidate can create and lock a *different* inode under the same name and
//! both believe they won.
//!
//! # What the file's contents are for
//!
//! Messaging only. The holder writes one line of JSON about itself -- pid, argv, start time --
//! *after* acquiring, through the descriptor it holds, and a loser reads it to say who it lost
//! to. Contents never decide anything: a file full of nonsense, or none at all, leaves the
//! verdict exactly where it was. The pid cannot come from the kernel anyway; measured on macOS
//! 26.6.2, `F_OFD_GETLK` reports that a write lock is held but returns `l_pid = -1`.
//!
//! Because the winner writes microseconds after acquiring, a loser that finds the file empty
//! waits briefly and then says plainly that the pid is unknown rather than guessing.
//!
//! # Advisory, and whose guard it is
//!
//! This is coordination, not enforcement: it stops a second `meethook record`, and does nothing
//! about some other tool that has grabbed the input device, which stays the activity
//! predicate's problem.
//!
//! It also belongs to `record` alone. `enroll`, `transcribe`, `speakers`, `forget` and
//! `meeting` are safe to run concurrently with a recording and must never be blocked by one --
//! `enroll` has to be able to bring a stale transcript up to date while the next call is being
//! captured. There is no flag to opt out of that asymmetry; the policy is enforced by who calls
//! [`RecordLock::acquire`], which is why it is written here rather than made configurable.
//!
//! # Network-mounted roots
//!
//! Locking over NFS and macOS SMB mounts is unreliable, and detecting those mounts reliably is
//! about as hard as locking them. The position is therefore to state it and do neither: with
//! `<root>` on a network share, this guard holds against processes on the local machine and may
//! not hold across machines.

use std::ffi::CString;
use std::io::Read;
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::{Duration, Instant};

use jiff::Zoned;
use serde_json::Value;

use crate::{Error, Paths, Result};

/// How long a loser waits for the winner's self-description to land in the file.
///
/// Short, because it is purely cosmetic latency: the refusal is already decided by the kernel
/// before this starts, and waiting longer than a few keystrokes' worth of milliseconds buys
/// nothing a clearer sentence could not say.
const METADATA_WINDOW: Duration = Duration::from_millis(100);

/// Poll interval inside [`METADATA_WINDOW`].
const METADATA_POLL: Duration = Duration::from_millis(20);

/// The live guard, held for as long as the process that took it means to own the name.
///
/// Releasing the lock is dropping this value, which closes the descriptor and nothing else:
/// there is deliberately no `unlock`, because an explicit `F_UNLCK` would leave the descriptor
/// open and would hide an inherited-fd leak from the test that looks for one, and because an
/// unlock immediately followed by close adds a window in which another process can take the
/// lock while this one still has a hand on the file.
#[derive(Debug)]
pub struct RecordLock {
    file: std::fs::File,
}

/// What a loser learned about the instance that beat it.
///
/// Every field is optional because the file is a message the holder chose to write about
/// itself, not a record anybody verifies: a holder from an older build, a truncated write, or
/// a hand-edited file all leave gaps, and a gap means *unknown*, never a guess.
#[derive(Debug)]
pub struct Holder {
    pub pid: Option<u32>,
    /// The command line that started it, joined with spaces for display.
    pub argv: Option<String>,
    pub started: Option<Zoned>,
}

/// The outcome of [`RecordLock::acquire`].
#[derive(Debug)]
pub enum Acquisition {
    /// We are the only `record` in this root. Keep the guard alive for the whole run.
    Held(RecordLock),
    /// Another live `record` holds it. Refuse, and name it.
    Taken(Holder),
}

impl RecordLock {
    /// Takes `<root>/record.lock` on behalf of this process, creating the root and the file if
    /// neither exists yet.
    ///
    /// Non-blocking, always, with no wait flag and no take-over flag: both of those invite the
    /// exact shape this guard exists to end -- two instances each waiting for, or stealing from,
    /// the other. A caller that gets [`Acquisition::Taken`] is expected to refuse and exit.
    ///
    /// Creating the root is deliberate: nothing else creates it before `record` needs it (the
    /// session directory is only made once devices are open), so refusing when it is absent
    /// would make a fresh install unable to record at all.
    ///
    /// Anything other than success or a genuine "already held" answer is an error rather than a
    /// silent pass. Losing the guard without noticing is the failure mode this module exists to
    /// fix, so an unexpected errno names the path and the OS reason and lets the run fail
    /// loudly -- including on an OS too old to have OFD locks at all.
    pub fn acquire(paths: &Paths) -> Result<Acquisition> {
        let path = paths.record_lock();
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;

        let file = open_lock_file(&path)?;

        if let Err(e) = try_ofd_write_lock(file.as_raw_fd()) {
            return match e.raw_os_error() {
                // The only two answers that mean "held by somebody else": EAGAIN for a lock
                // that would block, EACCES as the alternate spelling POSIX allows.
                Some(libc::EAGAIN) | Some(libc::EACCES) => {
                    Ok(Acquisition::Taken(read_holder(&path)))
                }
                _ => Err(Error::io(&path, e)),
            };
        }

        // After acquiring, never before: a loser that wrote here would clobber the winner's
        // message with its own, which is the one thing that would make the file worth trusting.
        write_holder_metadata(&file, &path)?;

        Ok(Acquisition::Held(RecordLock { file }))
    }
}

impl AsRawFd for RecordLock {
    /// Exposed so the descriptor's flags can be asserted rather than assumed -- see the
    /// `O_CLOEXEC` note on `open_lock_file`.
    fn as_raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

/// Opens the lock path read-write, creating it, with `FD_CLOEXEC` set at the open.
///
/// `O_CLOEXEC` is load-bearing, not boilerplate. Rust's `std` sets it on descriptors it opens
/// itself, so `OpenOptions` would have got it here too, but this call goes through the OS
/// directly and therefore has to say it: an inherited lock descriptor turns any long-lived
/// spawned child into a co-owner of the lock that outlives us -- `clips.rs` spawns audio
/// players, and the interrupt test spawns this very binary under a pty.
fn open_lock_file(path: &Path) -> Result<std::fs::File> {
    // A path containing an interior NUL is not a path the OS can open either, so refusing here
    // costs nothing real; the error is spelled as the io failure it would become.
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| Error::io(path, std::io::Error::from_raw_os_error(libc::EINVAL)))?;

    // SAFETY: `c_path` is a valid NUL-terminated C string kept alive by this binding for the
    // duration of the call. `libc::open` returns a fresh descriptor or -1 with errno set, and
    // the -1 branch below never reaches `from_raw_fd`, so exactly one `File` owns the fd.
    let fd = unsafe {
        libc::open(
            c_path.as_ptr(),
            libc::O_RDWR | libc::O_CREAT | libc::O_CLOEXEC,
            0o644,
        )
    };
    if fd < 0 {
        return Err(Error::io(path, std::io::Error::last_os_error()));
    }
    // SAFETY: `fd` was returned by `libc::open` above and is not otherwise owned, so taking
    // ownership of it here is exclusive.
    Ok(unsafe { std::fs::File::from_raw_fd(fd) })
}

/// Requests a whole-file OFD write lock without waiting.
///
/// Whole-file because `l_len == 0` means "to EOF, whatever that becomes", which is what a
/// mutual-exclusion lock wants: byte ranges would let two holders coexist on either side of an
/// empty file.
fn try_ofd_write_lock(fd: RawFd) -> std::io::Result<()> {
    // Field types differ per platform (`l_type` is `c_uchar` on Apple, `i16` on Linux gnu), so
    // the conversions are left to inference rather than spelled out for one target.
    let mut request = libc::flock {
        l_type: libc::F_WRLCK as _,
        l_whence: libc::SEEK_SET as _,
        l_start: 0,
        l_len: 0,
        l_pid: 0,
    };

    // SAFETY: `request` is a live local that the kernel copies in and may write back; the
    // variadic third argument is the `struct flock *` that `F_OFD_SETLK` is specified to take.
    let rc = unsafe { libc::fcntl(fd, libc::F_OFD_SETLK, &mut request) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

/// Records who the holder is, for the benefit of whoever loses next.
///
/// Written through the held descriptor: truncating and rewriting the file is safe precisely
/// because the lock is already ours, and doing it without reopening the path keeps a second
/// descriptor for the same inode out of the process.
fn write_holder_metadata(file: &std::fs::File, path: &Path) -> Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let line = serde_json::json!({
        "pid": std::process::id(),
        "argv": argv,
        "started": Zoned::now().to_string(),
    })
    .to_string();

    // Truncate first, because a previous holder's message may be longer than this one's and
    // leaving its tail behind would put two holders' claims in one file. Written at offset 0
    // rather than by seeking, so nothing here depends on where the descriptor happens to sit.
    file.set_len(0)
        .and_then(|()| file.write_all_at(line.as_bytes(), 0))
        .map_err(|e| Error::io(path, e))
}

/// Reads the holder's self-description, giving it [`METADATA_WINDOW`] to show up.
///
/// Read-only, and never truncating: a loser must not disturb the winner's message.
fn read_holder(path: &Path) -> Holder {
    let deadline = Instant::now() + METADATA_WINDOW;
    loop {
        let mut file = match std::fs::File::open(path) {
            Ok(file) => file,
            // The holder created the file before locking it, so its absence means the holder
            // died in between. There is nothing to report, and nothing to wait for.
            Err(_) => break,
        };
        let mut text = String::new();
        // Whatever came through is worth parsing, so a short read is not treated as no read.
        let _ = file.read_to_string(&mut text);
        if let Some(holder) = parse_holder(&text) {
            return holder;
        }
        if Instant::now() >= deadline {
            break;
        }
        std::thread::sleep(METADATA_POLL);
    }
    Holder {
        pid: None,
        argv: None,
        started: None,
    }
}

/// Parses the holder line without requiring any of it.
///
/// Returns `None` only when nothing recognizable is in there, which is what tells
/// [`read_holder`] to keep polling for a few milliseconds instead of settling for a blank
/// refusal.
fn parse_holder(text: &str) -> Option<Holder> {
    let value: Value = serde_json::from_str(text.trim()).ok()?;

    let pid = value
        .get("pid")
        .and_then(Value::as_u64)
        .map(|pid| pid as u32);
    let argv = value.get("argv").and_then(|argv| match argv {
        // The spelling this build writes. Joined for display rather than kept as a list: the
        // refusal quotes one line, and re-quoting each argument would be inventing a shell.
        Value::Array(args) => {
            let joined = args
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(" ");
            (!joined.is_empty()).then_some(joined)
        }
        Value::String(argv) => (!argv.is_empty()).then_some(argv.clone()),
        _ => None,
    });
    let started = value
        .get("started")
        .and_then(Value::as_str)
        .and_then(parse_started);

    if pid.is_none() && argv.is_none() && started.is_none() {
        None
    } else {
        Some(Holder { pid, argv, started })
    }
}

/// Accepts either spelling of a timestamp: with a zone annotation, or a bare offset instant.
fn parse_started(text: &str) -> Option<Zoned> {
    if let Ok(started) = text.parse::<Zoned>() {
        return Some(started);
    }
    // `to_zoned` is infallible: any timestamp that parsed at all is inside jiff's range.
    text.parse::<jiff::Timestamp>()
        .ok()
        .map(|stamp| stamp.to_zoned(jiff::tz::TimeZone::system()))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::os::unix::process::ExitStatusExt;

    use crate::EnrolledSpeakers;

    /// Acquires and forgets, so the guard is dropped at the end of the scope.
    fn acquire(root: &Path) -> Acquisition {
        RecordLock::acquire(&Paths::new(root)).unwrap()
    }

    fn held(root: &Path) -> RecordLock {
        match acquire(root) {
            Acquisition::Held(lock) => lock,
            Acquisition::Taken(holder) => panic!("expected to take the lock, held by {holder:?}"),
        }
    }

    fn taken(root: &Path) -> Holder {
        match acquire(root) {
            Acquisition::Held(_) => panic!("expected the lock to be held by someone else"),
            Acquisition::Taken(holder) => holder,
        }
    }

    /// OFD locks are owned by the open file description, so a second acquire in the *same*
    /// process loses just as a second process would -- which is what makes the mechanism
    /// testable without a second process, and is right for `record`, which never records twice
    /// in one process.
    #[test]
    fn a_second_acquire_against_a_held_root_refuses_and_names_the_holder() {
        let dir = tempfile::tempdir().unwrap();
        let lock = held(dir.path());

        let holder = taken(dir.path());
        assert_eq!(
            holder.pid,
            Some(std::process::id()),
            "the refusal has to name the process that holds the lock"
        );
        assert!(
            holder.argv.unwrap().contains("record_lock"),
            "the refusal has to say how the holder was started"
        );
        assert!(holder.started.is_some(), "and when it was started");

        // Releasing is dropping the guard: no unlock call, no unlink, nothing to clean up.
        drop(lock);
        let _again = held(dir.path());
    }

    /// The verdict comes from the kernel, not from the file: garbage in the file changes the
    /// wording of the refusal, never who wins.
    #[test]
    fn the_file_contents_never_decide_anything() {
        let dir = tempfile::tempdir().unwrap();
        let path = Paths::new(dir.path()).record_lock();
        let lock = held(dir.path());

        std::fs::write(&path, b"not json, and not anybody's pidfile either").unwrap();
        let holder = taken(dir.path());
        assert!(
            holder.pid.is_none() && holder.argv.is_none() && holder.started.is_none(),
            "an unreadable file has to say so plainly rather than invent a holder: {holder:?}"
        );

        // And the reverse: a file pointing at nobody alive is not what holds the lock either,
        // so once the real holder is gone the next acquire succeeds untouched by the decoy.
        drop(lock);
        std::fs::write(
            &path,
            br#"{"pid":999999,"argv":"whoever","started":"nope"}"#,
        )
        .unwrap();
        let _again = held(dir.path());
    }

    /// AC#2 of the guard: the kernel releases on `SIGKILL`, with no stale-file handling to
    /// thank for it. The holder is a re-executed copy of this test binary, because the point of
    /// the test is a process we cannot ask nicely.
    #[test]
    fn a_sigkilled_holder_releases_the_lock() {
        const HELPER_ENV: &str = "MEETHOOK_RECORD_LOCK_HOLDER_HELPER";
        if std::env::var_os(HELPER_ENV).is_some() {
            // The helper branch: take the lock named by the parent and sit on it until killed.
            let _lock = held(Path::new(&std::env::var(HELPER_ENV).unwrap()));
            std::thread::sleep(Duration::from_secs(60));
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let lock_path = paths.record_lock();

        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("record_lock::tests::a_sigkilled_holder_releases_the_lock")
            // Quiet: the helper's harness chatter would interleave with this test's output.
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .env(HELPER_ENV, dir.path())
            .spawn()
            .expect("spawning the holder helper");

        // The metadata is written only after the lock is taken, so its arrival is proof the
        // helper holds it -- no sleep-and-hope.
        let deadline = Instant::now() + Duration::from_secs(30);
        let marker = format!("\"pid\":{}", child.id());
        loop {
            let bytes = std::fs::read(&lock_path).unwrap_or_default();
            if String::from_utf8_lossy(&bytes).contains(&marker) {
                break;
            }
            if Instant::now() > deadline || child.try_wait().unwrap().is_some() {
                panic!("the helper never acquired {lock_path:?}");
            }
            std::thread::sleep(Duration::from_millis(20));
        }

        // SAFETY: `pid` names this test's own child, still owned by `child` and not yet waited
        // on, so it is a live process this test is entitled to signal.
        let killed = unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGKILL) } == 0;
        assert!(killed, "signalling our own child failed");
        let status = child.wait().unwrap();
        assert_eq!(
            status.signal(),
            Some(libc::SIGKILL),
            "the helper has to die by the signal, not by exiting"
        );

        // No cleanup, no waiting, no removing the file: the lock died with the process.
        let _again = held(dir.path());
    }

    /// The descriptor must not survive an `exec`, or a spawned player or a spawned `meethook`
    /// would keep the lock held after `record` itself quit.
    #[test]
    fn the_descriptor_is_close_on_exec() {
        let dir = tempfile::tempdir().unwrap();
        let lock = held(dir.path());

        // SAFETY: `F_GETFD` reads flags for a descriptor this process owns; a -1 return is
        // checked before the value is used.
        let flags = unsafe { libc::fcntl(lock.as_raw_fd(), libc::F_GETFD) };
        assert!(flags >= 0, "F_GETFD: {}", std::io::Error::last_os_error());
        assert_ne!(
            flags & libc::FD_CLOEXEC,
            0,
            "the lock descriptor is missing FD_CLOEXEC"
        );
    }

    /// The behavioural version of the flag above, and the one that actually matters: once the
    /// parent is done with the lock, a child that had inherited the descriptor would still be
    /// holding it, and the next `record` would be refused by a process that does not know it is
    /// a lock holder.
    #[test]
    fn a_spawned_child_does_not_keep_the_lock() {
        let dir = tempfile::tempdir().unwrap();
        let lock = held(dir.path());

        let mut child = std::process::Command::new("/bin/sleep")
            .arg("30")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .expect("spawning a long-lived child");

        drop(lock);

        // If the descriptor had leaked into the child, the child would still co-own the lock
        // and this would come back Taken while it lives.
        let _again = held(dir.path());

        // SAFETY: `pid` names this test's own child, still owned by `child` and not yet waited
        // on. SIGTERM ends the `sleep` so the test does not leave it behind.
        let signalled = unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) } == 0;
        assert!(signalled, "cleaning up the child failed");
        child.wait().unwrap();
    }

    /// The file is created by `record` and by nothing else, and removed by nothing at all --
    /// the counterpart to the root-exactness assertions elsewhere in this crate, which prove
    /// the same thing from the other side by listing what a non-`record` write leaves behind.
    #[test]
    fn only_acquiring_touches_the_lock_file_and_nothing_deletes_it() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());

        EnrolledSpeakers::new(Vec::new()).write(&paths).unwrap();
        assert!(
            !paths.record_lock().exists(),
            "writing the speaker database must not create the record lock"
        );

        let lock = held(dir.path());
        assert!(paths.record_lock().exists());
        drop(lock);

        // Dropping the guard closes a descriptor; it does not remove a file.
        assert!(paths.record_lock().exists());
        let _again = held(dir.path());
    }

    /// A brand-new root is normal: `--root` pointing at a directory that does not exist yet has
    /// to end in a recording, not in a complaint about the directory.
    #[test]
    fn acquiring_creates_a_root_that_is_not_there_yet() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("never-created/meethook");
        let paths = Paths::new(&root);

        let _lock = held(&root);

        assert!(paths.record_lock().is_file());
        assert_eq!(
            std::fs::read_dir(&root).unwrap().count(),
            1,
            "creating the root must not create anything beside the lock"
        );
    }
}
