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
//! # Asking whether a recorder is live without becoming one
//!
//! [`RecordLock::probe`] answers the same kernel question with an `F_OFD_GETLK` query over a
//! descriptor opened *without* `O_CREAT`: no acquisition, no holder-message rewrite, no file
//! created where none was. Liveness was always queryable this way -- the acquisition path already
//! reads `EAGAIN` as "held" -- but nothing outside `record` had cause to ask until an unfinished
//! session directory became something worth reporting, because such a directory also describes a
//! call being captured at this very second.
//!
//! The query cannot name the holder: `fcntl(2)` on macOS reports `l_pid = -1` for an OFD lock
//! (measured here on macOS 26.6.2, where `F_OFD_GETLK` says a write lock is held and returns no
//! pid), so identity still comes from the holder's own line via [`read_holder`]. The xnu-private
//! `F_OFD_GETLKPID` and `F_SETCONFINED` would answer it directly and are off-limits: portable
//! code may not rely on them, and this module has to work on Linux too.
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
//!
//! What that costs differs by path, deliberately. [`RecordLock::probe`] maps `EINVAL` -- Apple's
//! documented wording for "a file that does not support locking" -- and `ENOLCK`, Linux's spelling
//! of the same condition on NFS/SMB, to [`LockState::Unknown`], which behaves like `Held`: a report
//! over such a root keeps printing its once-per-run hedge and never calls a session interrupted,
//! without ever saying why. [`RecordLock::acquire`] treats the same errnos as hard errors, so
//! `record` refuses to start with an errno rather than a sentence. Falling toward silence on the
//! read side is the shipped decision rather than an oversight -- asserting that a call in progress
//! was interrupted is the one claim a wrong answer cannot be allowed to produce -- and the write
//! side keeps failing loudly because a guard lost without anyone noticing is what this module
//! exists to end. Neither errno is ever a version signal -- Apple documents `EINVAL` only as "a
//! file that does not support locking" and never acknowledges the "kernel doesn't support OFD
//! locks" case glibc names, so nothing here needs to know which macOS it is on (doc-008 section 4).
//! User-facing wording lives in README and `LINUX.md` (TASK-067.05.07.02), so no second explanation
//! belongs here.

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Holder {
    pub pid: Option<u32>,
    /// The command line that started it, joined with spaces for display.
    pub argv: Option<String>,
    pub started: Option<Zoned>,
}

/// What a read-only question about the lock can come back with.
///
/// Three answers rather than two because "the kernel would not say" is not the same fact as
/// "nobody holds it": an unreadable answer must never let a report claim that a call happening
/// now was interrupted. [`LockState::recorder_may_be_live`] encodes that asymmetry in the type
/// rather than leaving it to each caller to remember.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockState {
    /// A live process holds the lock. The [`Holder`] is its self-description, which may be
    /// partly empty: contents describe the holder, they never decide whether it is holding.
    Held(Holder),
    /// Nothing holds it. The file may well exist -- presence is not liveness, which is the whole
    /// reason this probe exists alongside the file.
    Free,
    /// The kernel gave an answer that means nothing either way: the filesystem does not support
    /// locking (`EINVAL`, which Apple also answers for an unlockable file), the remote-locking
    /// protocol failed (`ENOLCK`, Linux's NFS/SMB spelling), or the open went wrong for a reason
    /// other than absence. See the module's note on network-mounted roots.
    Unknown,
}

impl LockState {
    /// Whether a `record` may be capturing right now.
    ///
    /// True for [`LockState::Unknown`] on purpose. A caller about to tell a user their session
    /// was interrupted and left audio on disk has to withhold that sentence while one might be
    /// false, and "could not ask" is exactly such a case: the default falls toward saying
    /// nothing rather than toward saying something wrong.
    pub fn recorder_may_be_live(&self) -> bool {
        !matches!(self, LockState::Free)
    }
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
    /// loudly. That is not expected to mean "this OS is too old": the kernel has answered cmds
    /// 90/91/92 since OS X 10.11 and Apple made them public API in macOS 14 (doc-008 section 4),
    /// while this binary declares `minos 14.0` in its own load command and the recorder's trigger
    /// needs macOS 14.4+ (`meethook-record/src/activity.rs`). The unexpected errno is therefore a
    /// filesystem talking -- see the network-mount note above -- which is also why it stays loud
    /// rather than becoming another flavour of `Unknown`.
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

    /// Whether a `record` holds `<root>/record.lock` *right now*, asked of the kernel.
    ///
    /// Read-only in the strictest sense: the path is opened without `O_CREAT`, so probing a root
    /// that has never recorded cannot leave a byte behind there, and nothing is ever written,
    /// truncated or unlinked -- rewriting the holder message belongs to the holder alone
    /// ([`crate::Paths::record_lock`]), and a probe that created the file would break the
    /// assertion that non-`record` commands never touch it.
    ///
    /// This is what lets a report say "this directory holds WAVs and no `session.json`, so
    /// something interrupted it" without lying about a recording that is happening this second:
    /// `enroll`, `transcribe` and `speakers` are built to run alongside a live `record`, so an
    /// unfinished directory legitimately means *being recorded* as often as it means *was
    /// abandoned*. Liveness is also never inferred from the file's existence or contents, for
    /// the reasons in the module doc -- a killed holder leaves the file standing.
    ///
    /// Not a member of the guard's lifecycle, just filed beside it: it neither takes nor releases
    /// anything, and holding the returned answer proves nothing about who owns the lock.
    pub fn probe(paths: &Paths) -> LockState {
        probe_path(&paths.record_lock())
    }
}

/// The kernel's answer about `path`, without acquiring, creating, or reading a verdict out of
/// the file's contents.
fn probe_path(path: &Path) -> LockState {
    let c_path = match CString::new(path.as_os_str().as_bytes()) {
        Ok(c_path) => c_path,
        // Not a path the OS could open either; "could not ask" is the honest answer.
        Err(_) => return LockState::Unknown,
    };

    // `O_RDONLY | O_CLOEXEC` and deliberately no `O_CREAT`: see [`RecordLock::probe`]. The mode
    // argument is ignored by the kernel without `O_CREAT`, and is passed as zero for that reason
    // rather than as a permission this call has no intention of applying.
    // SAFETY: `c_path` is a valid NUL-terminated C string kept alive by this binding for the
    // duration of the call. `libc::open` returns a fresh descriptor or -1 with errno set, and
    // the -1 branch below never reaches `from_raw_fd`, so exactly one `File` owns the fd.
    let fd = unsafe { libc::open(c_path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC, 0) };
    if fd < 0 {
        return match std::io::Error::last_os_error().raw_os_error() {
            // No file, so nothing can be locked in it. This is the common answer for a root that
            // has never recorded, not a failure to ask.
            Some(libc::ENOENT) => LockState::Free,
            _ => LockState::Unknown,
        };
    }
    // SAFETY: `fd` was returned by `libc::open` above and is not otherwise owned.
    let file = unsafe { std::fs::File::from_raw_fd(fd) };

    let mut request = libc::flock {
        l_type: libc::F_WRLCK as _,
        l_whence: libc::SEEK_SET as _,
        l_start: 0,
        l_len: 0,
        // Zeroed, because Linux answers `EINVAL` to a query whose `l_pid` is not. The lock we ask
        // about is the same whole-file range [`try_ofd_write_lock`] takes.
        l_pid: 0,
    };

    // `EINTR` is the one error worth retrying: it says the query was never attempted. The others
    // -- `EINVAL` (no locking on this filesystem), `ENOLCK` (remote-locking failure), `EBADF`,
    // `EFAULT` -- say nothing about who holds the name, so they become `Unknown` rather than
    // collapsing into `Free`. `EAGAIN`/`EACCES` are deliberately absent: those answer `SETLK`,
    // never `GETLK`, which reports a conflict by overwriting the struct instead.
    for attempt in 0..2 {
        // SAFETY: `request` is a live local that the kernel copies in and writes back; the
        // variadic third argument is the `struct flock *` that `F_OFD_GETLK` is specified to take.
        let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_OFD_GETLK, &mut request) };
        if rc == 0 {
            // The kernel answered by rewriting the struct: `F_UNLCK` means no conflicting lock
            // exists, anything else describes one. Compared through `as _` because `l_type` is a
            // `c_short` on Apple and the constants are `c_int` on Linux gnu -- the same narrowing
            // `try_ofd_write_lock` relies on, so no `cfg` is needed for either target.
            //
            // `l_pid` is not read: macOS answers -1 for an OFD lock (measured on macOS 26.6.2, and
            // documented in `fcntl(2)`), so identity comes from the holder's own line instead.
            return if request.l_type as libc::c_int == libc::F_UNLCK as libc::c_int {
                LockState::Free
            } else {
                // Reusing the loser's reader means a holder that acquired a microsecond before
                // this query can cost up to `METADATA_WINDOW` here. That is chosen, not inherited:
                // a report that names the recording is worth a tenth of a second, and a second
                // implementation of "what did the holder say about itself" is a second source of
                // truth about it.
                LockState::Held(read_holder(path))
            };
        }
        if attempt == 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
            continue;
        }
        return LockState::Unknown;
    }
    LockState::Unknown
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
    // `F_WRLCK` and `SEEK_SET` are `c_int` constants, but `l_type` and `l_whence` are `c_short`
    // on both Apple and Linux gnu (libc 0.2.189, checked for both) -- the same narrowing `as`
    // either platform needs, so the conversions are left to inference rather than spelled out.
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

    /// The environment a re-executed copy of this binary reads to become a lock holder that sits
    /// on the lock until killed, holding the root named by its value.
    const HOLDER_HELPER_ENV: &str = "MEETHOOK_RECORD_LOCK_HOLDER_HELPER";

    /// Spawns a holder in its own process and waits for proof it acquired.
    ///
    /// Proof, not sleep: the helper writes its metadata only *after* taking the lock, so the
    /// arrival of its pid in the file means the kernel has granted it.
    fn spawn_holder(root: &Path) -> std::process::Child {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("record_lock::tests::a_sigkilled_holder_releases_the_lock")
            // Quiet: the helper's harness chatter would interleave with this test's output.
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .env(HOLDER_HELPER_ENV, root)
            .spawn()
            .expect("spawning the holder helper");

        let path = Paths::new(root).record_lock();
        let deadline = Instant::now() + Duration::from_secs(30);
        let marker = format!("\"pid\":{}", child.id());
        loop {
            let bytes = std::fs::read(&path).unwrap_or_default();
            if String::from_utf8_lossy(&bytes).contains(&marker) {
                return child;
            }
            if Instant::now() > deadline || child.try_wait().unwrap().is_some() {
                panic!("the helper never acquired {path:?}");
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// Ends a helper spawned by [`spawn_holder`] by signal, never by asking it politely.
    fn kill_holder(child: &mut std::process::Child) {
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
    }

    /// AC#2 of the guard: the kernel releases on `SIGKILL`, with no stale-file handling to
    /// thank for it. The holder is a re-executed copy of this test binary, because the point of
    /// the test is a process we cannot ask nicely.
    #[test]
    fn a_sigkilled_holder_releases_the_lock() {
        if std::env::var_os(HOLDER_HELPER_ENV).is_some() {
            // The helper branch: take the lock named by the parent and sit on it until killed.
            let _lock = held(Path::new(&std::env::var(HOLDER_HELPER_ENV).unwrap()));
            std::thread::sleep(Duration::from_secs(60));
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let mut child = spawn_holder(dir.path());
        kill_holder(&mut child);

        // No cleanup, no waiting, no removing the file: the lock died with the process.
        let _again = held(dir.path());
    }

    /// The read-only question, asked while another process holds the answer: the probe reports
    /// it held, and touches nothing. It sees the *holder's own line* here rather than a pid,
    /// because macOS answers `l_pid = -1` for an OFD lock -- which is why identity comes from the
    /// file even though the verdict never does.
    #[test]
    fn a_probe_reports_another_process_holding_it_and_leaves_the_file_alone() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let path = paths.record_lock();
        let mut child = spawn_holder(dir.path());

        let written = std::fs::read(&path).unwrap();
        match RecordLock::probe(&paths) {
            LockState::Held(holder) => {
                assert_eq!(
                    holder.pid,
                    Some(child.id()),
                    "the probe has to name the process that holds the lock"
                );
                assert!(holder.argv.is_some(), "and say how it was started");
            }
            other => panic!("expected the probe to report a live recorder, got {other:?}"),
        }
        assert_eq!(
            std::fs::read(&path).unwrap(),
            written,
            "asking must not disturb the holder's message"
        );

        kill_holder(&mut child);
    }

    /// A second descriptor in this process is a different holder as far as the kernel is
    /// concerned -- the same fact `a_second_acquire_against_a_held_root_refuses_and_names_the_holder`
    /// exploits -- so liveness is testable without leaving the process at all.
    #[test]
    fn a_probe_reports_a_holder_in_this_very_process() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let _lock = held(dir.path());

        let state = RecordLock::probe(&paths);
        let LockState::Held(holder) = state else {
            panic!("a held root must not read as free: {state:?}");
        };
        assert_eq!(holder.pid, Some(std::process::id()));
    }

    /// The whole reason a non-`record` command may ask: the answer costs nothing, including on a
    /// root that has never recorded, where asking must not leave a trace. `speakers` asserts the
    /// same absence from the outside; this pins it at the probe itself.
    #[test]
    fn probing_a_root_that_has_never_recorded_leaves_nothing_behind() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("meethook");
        std::fs::create_dir(&root).unwrap();
        let paths = Paths::new(&root);

        assert_eq!(RecordLock::probe(&paths), LockState::Free);
        assert!(!paths.record_lock().exists(), "no file was created");
        assert_eq!(
            std::fs::read_dir(&root).unwrap().count(),
            0,
            "and nothing else appeared in the root either"
        );
    }

    /// Presence is not liveness: the file outlives every holder by design, and the probe says
    /// `Free` anyway. Without this, `.02`/`.03` could not tell an abandoned directory from a
    /// recording in progress.
    #[test]
    fn a_dropped_guard_stops_being_live_though_the_file_still_stands() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let lock = held(dir.path());
        assert!(
            RecordLock::probe(&paths).recorder_may_be_live(),
            "while the guard is up the probe must not read the root as free"
        );

        drop(lock);

        assert!(
            paths.record_lock().exists(),
            "dropping closes, never unlinks"
        );
        assert_eq!(RecordLock::probe(&paths), LockState::Free);
    }

    /// The strongest available statement that neither existence nor contents decide anything: a
    /// holder killed by signal leaves its file behind, intact, and the probe reads it as free.
    #[test]
    fn a_sigkilled_holder_is_reported_free_though_its_file_remains() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let mut child = spawn_holder(dir.path());

        kill_holder(&mut child);

        assert!(
            paths.record_lock().is_file(),
            "nothing cleans up after a holder"
        );
        assert_eq!(
            RecordLock::probe(&paths),
            LockState::Free,
            "and the file's continued existence says nothing about a recorder"
        );
    }

    /// The safety default, asserted directly because every caller leans on it: only a definite
    /// `Free` permits saying a session was interrupted, and an answer the kernel refused to give
    /// is not a definite anything.
    #[test]
    fn an_unknown_answer_leans_toward_a_recorder_being_live() {
        assert!(
            LockState::Held(Holder {
                pid: None,
                argv: None,
                started: None
            })
            .recorder_may_be_live()
        );
        assert!(LockState::Unknown.recorder_may_be_live());
        assert!(!LockState::Free.recorder_may_be_live());
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
