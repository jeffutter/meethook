#![allow(dead_code)]
// every test target compiles its own copy of this module and uses a
// subset of it -- the pty driver in some, the session fixture in others,
// usually not all of either; the parts one binary ignores are not dead code.

//! The two ways a test drives the built binary: through a real pty, and over a synthetic root.
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
//!
//! The other half builds what that binary is pointed at: a root holding a known mix of session
//! shapes, assembled from honest WAV forensics so a truncated header is a real truncated header.
//! Anything that runs a whole-root command -- `sessions`, `transcribe <id>`, `enroll --list` --
//! against more than one shape at a time should start here rather than lay the same four
//! directories down again.
//!
//! There are two mothers because there are two audiences. [`mixed_root`] reproduces the four shapes
//! a user is told about, and README quotes its output byte for byte; [`forensic_root`] holds the
//! foreign-file shapes nobody advertises. They stay separate on purpose -- see the doc comment on
//! [`forensic_root`] for the four goldens that would churn if the rows were merged back together.

use std::io::{Read, Write};
use std::os::fd::{AsFd, AsRawFd, FromRawFd};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use meethook_session::wav::TrackEvidence;
use meethook_session::{Paths, SessionId};

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

/// A session directory holding exactly `files`, each written as a placeholder.
///
/// Honest here because the report tests those two names for presence and never opens them, which
/// is how `classify` decides what a directory is.
pub(crate) fn placeholder_session(
    paths: &Paths,
    id: &str,
    files: &[&str],
) -> meethook_session::SessionPaths {
    let session = paths.session(&SessionId::parse(id).unwrap());
    std::fs::create_dir_all(session.dir()).unwrap();
    for file in files {
        std::fs::write(session.dir().join(file), b"placeholder").unwrap();
    }
    session
}

/// A finalized WAV of `seconds` of silence: mono 16 kHz float32, so 64 000 bytes of `data` a
/// second. Truncating one afterwards is what makes a header declare more than the file holds.
pub(crate) fn clip(path: &Path, seconds: f64) {
    let samples = vec![0.0f32; (seconds * 16_000.0) as usize];
    meethook_enroll::write_clip(path, &samples).unwrap();
}

/// The same four directories as the unit golden, built the same way: an orphan whose mic stops
/// 0.2 s short of what its header declares and whose speaker track runs 0.1 s past it, an orphan
/// with nothing in it, one transcribed and one valid.
pub(crate) fn mixed_root(root: &Path) -> Paths {
    let paths = Paths::new(root);
    std::fs::create_dir_all(paths.sessions_dir()).unwrap();
    placeholder_session(
        &paths,
        "20260809-052700",
        &["session.json", "transcript.json"],
    );
    placeholder_session(&paths, "20260809-052800", &["session.json"]);
    placeholder_session(&paths, "20260809-052600", &[]);
    let orphaned = paths.session(&SessionId::parse("20260809-052500").unwrap());
    std::fs::create_dir_all(orphaned.dir()).unwrap();
    clip(&orphaned.mic_wav(), 1.0);
    let mic = orphaned.mic_wav();
    let len = std::fs::metadata(&mic).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&mic)
        .unwrap()
        .set_len(len - 12_800)
        .unwrap();
    use std::io::Write;
    clip(&orphaned.speaker_wav(), 1.0);
    std::fs::OpenOptions::new()
        .append(true)
        .open(orphaned.speaker_wav())
        .unwrap()
        .write_all(&[0u8; 6_400])
        .unwrap();
    paths
}

/// Whether this process reads any file whatever its mode bits say.
///
/// The single copy of this guard for the CLI crate's integration targets. A `chmod 0000` file is
/// not a test fixture under root -- the open succeeds and the state under test simply never
/// happens -- so a test that builds one has to say which expectation it is holding, rather than
/// quietly asserting nothing. `meethook-session`'s own unit test for the same state
/// (`wav.rs`'s `a_file_that_cannot_be_read_is_reported_rather_than_skipped`) guards the same way,
/// but inside that crate's `#[cfg(test)] mod tests`, which compiles into its own unit-test binary
/// and cannot be imported from here; this is the one copy this crate shares rather than one per
/// target.
pub(crate) fn root_reads_anything() -> bool {
    // SAFETY: `geteuid` takes no arguments and cannot fail.
    (unsafe { libc::geteuid() }) == 0
}

/// One directory of [`forensic_root`]: the shape it was laid down to make, and the reading a
/// correct reader owes it.
///
/// The expected evidence is written next to the construction rather than derived from it, so the
/// assertion that a surface really printed this state is not a tautology about the builder.
pub(crate) struct ForensicRow {
    pub(crate) id: &'static str,
    /// What the directory is a picture of, named in the failure message when it stops being that.
    pub(crate) meaning: &'static str,
    pub(crate) mic: TrackEvidence,
    /// Every row's speaker track is a plain whole clip: one fault per directory keeps the brief to
    /// a single clause, which is what makes "the brief named the track the detail reported" a sharp
    /// assertion instead of one an unrelated clause could satisfy.
    pub(crate) speaker: TrackEvidence,
}

/// A root of seven orphaned sessions, each holding exactly one of the track states that
/// [`mixed_root`] does not reach: `HeaderOnly`, `NotAWav`, `Unreadable` twice, `Unknown` twice and
/// `NoDeclaredLength`.
///
/// These are the shapes a foreign file produces -- a copied-then-truncated track, an image renamed
/// `.wav`, a file nobody may read, a header whose chunk walk ran off the read window -- and so the
/// states a support request actually arrives with. Each has had a unit test inside `wav.rs` and a
/// rendering assertion inside `interrupted.rs` since it was added; what none of them had was ever
/// reaching a surface a person reads, which is what the callers of this mother exist to prove.
///
/// A second mother rather than more rows on [`mixed_root`], for the same reason
/// `readme_quotes_the_renderers.rs` keeps its fifth shape local: `mixed_root`'s output is pinned
/// byte for byte in four places -- the listing in `sessions_report.rs`, the census and the
/// classification table in `three_surfaces_one_directory.rs`, and README's sample block -- and an
/// exotic row would churn all four while advertising in the documentation a state nobody should
/// have to meet. Nothing here touches any of them: the ids are fresh, and the four pinned
/// classifications are unchanged.
pub(crate) fn forensic_root(root: &Path) -> (Paths, Vec<ForensicRow>) {
    let paths = Paths::new(root);
    std::fs::create_dir_all(paths.sessions_dir()).unwrap();
    let whole = TrackEvidence::CompleteAsDeclared;
    let mut rows = Vec::new();

    // A header declaring zero `data` bytes and nothing written onto it. Not reachable by dropping a
    // writer on the floor: through the public create path that leaves a *zero-byte* file, which is
    // `NotAWav`. Writing a real clip with no samples is the route that produces the declared-zero
    // header, which is also what finalizing a writer that never received a sample would leave.
    let session = orphan_session(&paths, "20260101-000001");
    clip(&session.mic_wav(), 0.0);
    clip(&session.speaker_wav(), 1.0);
    rows.push(ForensicRow {
        id: "20260101-000001",
        meaning: "a header that was born declaring nothing and never got a sample",
        mic: TrackEvidence::HeaderOnly,
        speaker: whole,
    });

    // An honest PNG renamed `.wav` -- the shape of the support request, not a random blob.
    let session = orphan_session(&paths, "20260101-100002");
    std::fs::write(
        session.mic_wav(),
        [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a],
    )
    .unwrap();
    clip(&session.speaker_wav(), 1.0);
    rows.push(ForensicRow {
        id: "20260101-100002",
        meaning: "an image that was renamed .wav",
        mic: TrackEvidence::NotAWav,
        speaker: whole,
    });

    // A real clip nobody may read. Under root the mode bits mean nothing, so the row's declared
    // expectation follows the same guard that gates the chmod -- a row that silently stopped
    // asserting its state is the unfalsifiable branch this whole ticket is about. The mode is left
    // set: deleting a file is a privilege of its directory, which `TempDir` owns outright, so
    // cleanup stays boring and every surface in the calling test still meets the state.
    let session = orphan_session(&paths, "20260101-200003");
    clip(&session.mic_wav(), 1.0);
    let unreadable_by_mode = !root_reads_anything();
    if unreadable_by_mode {
        std::fs::set_permissions(session.mic_wav(), std::fs::Permissions::from_mode(0o000))
            .unwrap();
    }
    clip(&session.speaker_wav(), 1.0);
    rows.push(ForensicRow {
        id: "20260101-200003",
        meaning: if unreadable_by_mode {
            "a track the user cannot read"
        } else {
            "a track the user cannot read -- skipped: running as root, so mode 0000 still opens"
        },
        mic: if unreadable_by_mode {
            TrackEvidence::Unreadable
        } else {
            TrackEvidence::CompleteAsDeclared
        },
        speaker: whole,
    });

    // The same state with no permission bits anywhere in it: `File::open` on a directory succeeds
    // on macOS and the read afterwards answers EISDIR. This is why the row above being skipped
    // under root does not cost the suite its proof that a surface prints the sentence.
    let session = orphan_session(&paths, "20260101-300004");
    std::fs::create_dir(session.mic_wav()).unwrap();
    clip(&session.speaker_wav(), 1.0);
    rows.push(ForensicRow {
        id: "20260101-300004",
        meaning: "a directory wearing the name of a track",
        mic: TrackEvidence::Unreadable,
        speaker: whole,
    });

    // A prologue, a parsed `fmt `, and then the file ends before any `data` chunk is even named:
    // the walk found no audio to weigh. Truncating a real clip just past its `fmt ` chunk is the
    // cheap route, and the offset comes out of the file rather than out of a constant.
    let session = orphan_session(&paths, "20260101-400005");
    let mic = session.mic_wav();
    clip(&mic, 1.0);
    let bytes = std::fs::read(&mic).unwrap();
    let fmt_len = usize::try_from(u32_at(&bytes, 16)).expect("absurd fmt chunk size");
    assert_eq!(
        &bytes[20 + fmt_len..24 + fmt_len],
        b"data",
        "the clip's layout is not the one this row truncates by"
    );
    std::fs::OpenOptions::new()
        .write(true)
        .open(&mic)
        .unwrap()
        .set_len(u64::try_from(20 + fmt_len).unwrap())
        .unwrap();
    clip(&session.speaker_wav(), 1.0);
    rows.push(ForensicRow {
        id: "20260101-400005",
        meaning: "a header that ends before it names any audio",
        mic: TrackEvidence::Unknown,
        speaker: whole,
    });

    // The variant's own second cause: the chunk walk ran off the read window. A `JUNK` prelude
    // longer than `wav`'s 64 KiB window sits ahead of `fmt `, so the walk stops inside metadata
    // that a foreign recorder genuinely writes. Pushing `data` past the window instead does not
    // work -- `Chunks::next` hands back an over-window chunk with the bytes in hand, and the answer
    // is `ShortBy`. The `RIFF` size is written honestly even though nothing consults it, so this
    // directory is a picture of one state rather than of two lies.
    let session = orphan_session(&paths, "20260101-500006");
    let mic = session.mic_wav();
    clip(&mic, 1.0);
    let audio = std::fs::read(&mic).unwrap()[12..].to_vec();
    const JUNK_PAYLOAD: usize = 70_000; // > the 64 KiB header window
    let body = [
        b"JUNK".as_slice(),
        &(JUNK_PAYLOAD as u32).to_le_bytes(),
        &vec![0u8; JUNK_PAYLOAD],
        &audio,
    ]
    .concat();
    let declared = u32::try_from(body.len() + 4).expect("prelude overflowed the riff size");
    std::fs::write(
        &mic,
        [b"RIFF".as_slice(), &declared.to_le_bytes(), b"WAVE", &body].concat(),
    )
    .unwrap();
    clip(&session.speaker_wav(), 1.0);
    rows.push(ForensicRow {
        id: "20260101-500006",
        meaning: "metadata deeper than the header window, hiding the chunks behind it",
        mic: TrackEvidence::Unknown,
        speaker: whole,
    });

    // TASK-067.05.08's sentinel: `data` declaring `0xFFFFFFFF` over audio that is really there.
    // FFmpeg writes headers like this, which is how a file with no length at all turns up here.
    let session = orphan_session(&paths, "20260101-600007");
    let mic = session.mic_wav();
    clip(&mic, 1.0);
    let bytes = std::fs::read(&mic).unwrap();
    let fmt_len = usize::try_from(u32_at(&bytes, 16)).expect("absurd fmt chunk size");
    let mut file = std::fs::OpenOptions::new().write(true).open(&mic).unwrap();
    {
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::Start(u64::try_from(24 + fmt_len).unwrap()))
            .unwrap();
    }
    file.write_all(&u32::MAX.to_le_bytes()).unwrap();
    clip(&session.speaker_wav(), 1.0);
    rows.push(ForensicRow {
        id: "20260101-600007",
        meaning: "audio sitting on a header that declares no length",
        mic: TrackEvidence::NoDeclaredLength(meethook_session::wav::TrackSpan {
            bytes: 64_000,
            millis: 1_000,
        }),
        speaker: whole,
    });

    // Row 3 has to hand back a file the caller can still classify, so the chmod lives inside the
    // construction rather than in a guard the caller has to remember to arm.

    (paths, rows)
}

/// A directory named like a session and holding nothing yet.
fn orphan_session(paths: &Paths, id: &str) -> meethook_session::SessionPaths {
    let session = paths.session(&SessionId::parse(id).unwrap());
    std::fs::create_dir_all(session.dir()).unwrap();
    session
}

/// A little-endian u32 read out of a file's own bytes.
fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
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
