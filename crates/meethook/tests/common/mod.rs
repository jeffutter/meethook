#![allow(dead_code)]
// every test target compiles its own copy of this module and uses a
// subset of it; the parts one binary ignores are not dead code.

//! Driving the built binary through a real pty.
//!
//! The interactive frames only draw when their stdout is a terminal, so a test that wants to
//! watch one -- or type into it, or kill it mid-keystroke -- cannot settle for the piped stdio
//! `std::process::Command::output` gives it. This module spawns the compiled `meethook` binary on
//! a pty, hands the test the master to read the screen from and write keystrokes to, and blocks
//! until the frame is up. It lives in a `common` subdirectory so neither libtest nor nextest
//! treats it as a test target of its own; each test that wants it declares `mod common;`.
//!
//! A child spawned here is owned by the `Driver`, which kills and reaps it on every path out of
//! the test -- including the panicking ones. That reap is load-bearing rather than tidy: see the
//! comment on the `Drop` impl, which records the sleeping orphan it exists to prevent.

use std::io::{Read, Write};
use std::os::fd::{AsFd, AsRawFd, FromRawFd};
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

/// The pty geometry every frame is drawn into: wide enough that neither key row wraps, tall
/// enough for header, status, key hints and footer with room to spare.
///
/// A test that reconstructs the screen has to agree with the child about this rectangle, so it is
/// stated once here instead of as a private constant in each test file. Even so, a grid model has
/// to grow to fit what the frame actually addresses -- requested geometry and delivered geometry
/// have been observed to disagree on real hardware, which `open_pty`'s second sizing exists to
/// make rare rather than impossible.
pub(crate) const PTY_ROWS: u16 = 30;
pub(crate) const PTY_COLS: u16 = 100;

/// One open pty: the master the test reads and writes, the slave the child takes as its
/// controlling terminal. The size is set before the spawn because the frame draws into
/// whatever rectangle it is handed.
pub(crate) fn open_pty() -> (std::fs::File, std::fs::File) {
    // SAFETY: `master`/`slave`/`win` are valid stack locals passed by pointer to a well-formed
    // libc call; `openpty` fills `master`/`slave` with fresh, valid fds on success (checked via
    // `rc`), which `from_raw_fd` then takes ownership of.
    //
    // `&mut win` is required because macOS's `libc::openpty` declares `winp` as `*mut winsize`
    // (unlike Linux's POSIX-matching `*const`), which clippy running on Linux CI can't see.
    #[allow(clippy::unnecessary_mut_passed)]
    unsafe {
        let mut master: libc::c_int = -1;
        let mut slave: libc::c_int = -1;
        let mut name = [0u8; 256];
        let mut win = libc::winsize {
            ws_row: PTY_ROWS,
            ws_col: PTY_COLS,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        let rc = libc::openpty(
            &mut master,
            &mut slave,
            name.as_mut_ptr().cast(),
            std::ptr::null_mut(),
            &mut win,
        );
        assert_eq!(rc, 0, "openpty failed: {}", std::io::Error::last_os_error());
        let master = std::fs::File::from_raw_fd(master);
        let slave = std::fs::File::from_raw_fd(slave);
        // Ask twice, because the first ask has been observed not to hold. On a granted Mac the
        // frame addressed its footer at row 71 while `win` said thirty, which is what a
        // reconstructed grid looks like when the geometry underneath it changed after creation --
        // a cloned pty inheriting the window it was cloned from is the usual way that happens, and
        // nothing here can rule it out from the parent side. Setting the size on the descriptor
        // afterwards is the form that measurably sticks; a test still must not assume it did, so
        // `Screen` grows on demand regardless.
        //
        // `slave` is a valid open pty descriptor owned by the binding above, and `&mut win` points
        // at a live `winsize`; TIOCSWINSZ takes exactly these two arguments. Same outer `unsafe`
        // block as the `openpty` call, so no nested block of its own.
        let rc = libc::ioctl(slave.as_fd().as_raw_fd(), libc::TIOCSWINSZ, &mut win);
        assert_eq!(
            rc,
            0,
            "could not size the pty {}x{}: {}",
            PTY_ROWS,
            PTY_COLS,
            std::io::Error::last_os_error()
        );
        (master, slave)
    }
}

/// The built binary pointed at `root`, driven interactively through a pty.
pub(crate) struct Driver {
    pub(crate) child: Child,
    pub(crate) master: std::fs::File,
    /// Everything the child has written, kept so a failure can show what the frame saw.
    pub(crate) out: Vec<u8>,
    pub(crate) buf: [u8; 16384],
}

impl Driver {
    /// The built binary, pointed at `root` first (the global `--root` precedes the subcommand),
    /// then whatever subcommand and arguments the caller needs, on a pty.
    pub(crate) fn spawn(root: &Path, args: &[&str]) -> Self {
        let (master, slave) = open_pty();
        let master = {
            let fd = master.as_fd().as_raw_fd();
            // SAFETY: `fd` is `master`'s own descriptor, kept alive by the borrow above, and
            // `F_GETFL` takes no pointer argument -- a plain read of the fd's current flags.
            let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
            assert!(
                flags >= 0,
                "fcntl F_GETFL failed: {}",
                std::io::Error::last_os_error()
            );
            // SAFETY: same fd as above; `flags` was just read from it, so OR-ing in
            // `O_NONBLOCK` and writing it back changes no bit this call did not just observe.
            let rc = unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) };
            assert_eq!(
                rc,
                0,
                "fcntl F_SETFL failed: {}",
                std::io::Error::last_os_error()
            );
            master
        };

        let mut cmd = Command::new(env!("CARGO_BIN_EXE_meethook"));
        cmd.arg("--root")
            .arg(root)
            .args(args)
            .env("TERM", "xterm-256color")
            // The three diagnostic switches print through the record crate's `Output`, which
            // writes stderr -- and stderr is wired to the same pty slave as the frame's stdout.
            // A test that reconstructs the screen from those bytes would therefore parse
            // `[activity] ...` lines as if the frame had painted them. Scrubbing them costs a
            // debugging avenue inside the test only; the manual recipe in the live-proof ticket
            // runs the same binary with the variable set, which is where reading the walker's
            // decisions actually belongs.
            //
            // Spelled as literals rather than reused from `meethook_record::{ACTIVITY_DEBUG_ENV_VAR,
            // CALENDAR_DEBUG_ENV_VAR}`: this module compiles on every platform, and the record
            // crate is a macOS-only dependency of the CLI crate.
            .env_remove("MEETHOOK_ACTIVITY_DEBUG")
            .env_remove("MEETHOOK_CALENDAR_DEBUG")
            .env_remove("MEETHOOK_TIMING_DEBUG")
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave.try_clone().unwrap()))
            .stderr(Stdio::from(slave));
        // `std::process::Child` does not kill when it is dropped -- there is no `kill_on_drop`
        // outside `tokio::process` -- so this child outlives a panicking test unless `Driver`
        // itself reaps it. See the `Drop` below.
        let child = cmd.spawn().expect("spawning the built binary");
        Driver {
            child,
            master,
            out: Vec::new(),
            buf: [0; 16384],
        }
    }

    /// Pulls whatever is readable into `out` without blocking.
    pub(crate) fn pump(&mut self) {
        loop {
            let n = match self.master.read(&mut self.buf) {
                Ok(n) => n,
                Err(_) => return, // nothing yet, or the pty closed with the child gone
            };
            if n == 0 {
                return;
            }
            self.out.extend_from_slice(&self.buf[..n]);
        }
    }

    /// Blocks until either the frame has taken the terminal or the child has gone, whichever
    /// comes first. Returns whether the frame won: a re-run pointed at a root the interrupt
    /// left fully converged finds no question worth a frame and exits clean instead.
    pub(crate) fn wait_for_frame_or_exit(&mut self, timeout: Duration) -> bool {
        const ALT_SCREEN: &[u8] = b"\x1b[?1049h";
        let seen = |out: &[u8]| out.windows(ALT_SCREEN.len()).any(|w| w == ALT_SCREEN);
        let deadline = Instant::now() + timeout;
        loop {
            self.pump();
            if seen(&self.out) {
                return true;
            }
            match self.child.try_wait() {
                Ok(Some(_)) => return false,
                Ok(None) => {}
                Err(e) => panic!("waiting on the child: {e}"),
            }
            if Instant::now() > deadline {
                panic!(
                    "neither frame nor exit within {timeout:?}; got {} bytes:\n{}",
                    self.out.len(),
                    String::from_utf8_lossy(&self.out)
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Feeds keystrokes with a settle between them: the pty queues raw-mode input, but the
    /// frame redraws on a timer and each mark must land on its own row.
    pub(crate) fn feed(&mut self, keys: &[&[u8]]) {
        for key in keys {
            self.master.write_all(key).unwrap();
            self.pump();
            std::thread::sleep(Duration::from_millis(150));
        }
    }

    /// Writes bytes to the pty and nothing else -- for the keystroke that starts the commits,
    /// where the very next instruction is to watch the disk, and a settle would let the whole
    /// burst finish before the watcher starts.
    pub(crate) fn write_now(&mut self, bytes: &[u8]) {
        self.master.write_all(bytes).unwrap();
    }

    /// Blocks until the child is gone, keeping the pty drained so it never blocks on a full
    /// buffer. Returns its exit status.
    pub(crate) fn wait_exit(&mut self, timeout: Duration) -> ExitStatus {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    self.pump();
                    return status;
                }
                Ok(None) => {
                    self.pump();
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(e) => panic!("waiting on the child: {e}"),
            }
        }
        let _ = self.child.kill();
        let status = self.child.wait().unwrap();
        panic!(
            "the run did not finish in {timeout:?}; status {status}; tail:\n{}",
            String::from_utf8_lossy(&self.out[self.out.len().saturating_sub(2000)..])
        );
    }

    /// The last stretch of what the frame wrote, for a failure message.
    pub(crate) fn tail(&self) -> String {
        String::from_utf8_lossy(&self.out[self.out.len().saturating_sub(2000)..]).into_owned()
    }
}

/// Kills and reaps the child on every path out of a `Driver`, including the ones that never
/// reach `wait_exit`: the deadline panics inside the waits, the two panic sites in
/// `interrupt_once`, and `complete_and_verify`'s no-frame branch, which falls straight through to
/// the disk assertions and drops the driver with the child already gone but still unreaped.
///
/// This is load-bearing in a way most test cleanups are not. The child is `enroll` under a pty:
/// once the frame is up it blocks reading stdin that the test may have stopped writing, and
/// nextest runs each test in its own process, so a test failure leaves the child alive to be
/// reparented by launchd -- where a `meethook enroll` asleep for days with PPID 1 was once found
/// on the dev machine. `std::process::Child`'s own Drop does nothing about that, and the standard
/// library has no `kill_on_drop` to ask for (that one is `tokio::process`'s), so owning the reap
/// here is the whole guarantee. It covers unwinding, which is how libtest reports a failure; a
/// test process killed outright still orphans its child, because macOS offers no parent-death
/// signal to arm at spawn time.
impl Drop for Driver {
    fn drop(&mut self) {
        // Once the child has been waited on, `try_wait` answers from the cached status without a
        // syscall, so an already-reaped child is never signalled again -- its pid may by then
        // belong to some other process. A failed `try_wait` is treated the same way: nothing
        // this test can do about it is better than leaving the pid alone.
        if !matches!(self.child.try_wait(), Ok(None)) {
            return;
        }
        // The child is alive; SIGKILL does not wait on it draining the pty, and both errors are
        // uninteresting -- a child that died between `try_wait` and `kill` is the outcome wanted.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
