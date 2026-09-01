//! The binary-level half of the group commit's interrupt rule (TASK-046.09, AC #4).
//!
//! The library proves the write sequence in isolation (the unwritable-root tricks in
//! `meethook-enroll`'s own suite exercise the identical fixed-order walk). What only a live
//! process can prove is the thing this test arranges: a person watching three split voices,
//! marking them as one person through the real frame, and the machine dying between the
//! members' commits. The built binary is spawned against a three-voice root whose first two
//! clusters segmentation heard at once, driven through a real pty, and SIGKILL'd the moment
//! the first member's database row appears on disk.
//!
//! Every session file is written atomically (temp file, sync, rename), so a kill anywhere in
//! the walk leaves each member either fully committed or absent -- which is what makes the
//! assertions below independent of *where* in the window the kill landed. The landing itself
//! is timed rather than deterministic: the window is tens of milliseconds wide in a debug
//! build, and an attempt that misses it (the process exits clean, or the kill lands after the
//! last member) rebuilds the fixture and tries again.
//!
//! What the test owes the ticket:
//! - (a) every committed member is fully written in the fixed order -- the database before the
//!   names file, never the reverse, and no member past the prefix touched at all;
//! - (b) a re-run through the same frame completes what the interrupt left behind;
//! - (c) a third pass changes nothing on disk.
//!
//! Along the way the re-run drives the full frame with real `Preview::group` data at up to
//! three marks including the heard-at-once pair -- the composition .02's fake-costs tests and
//! .01's scripted-library test each cover only in part.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::os::fd::{AsFd, AsRawFd, FromRawFd};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

use meethook_session::{
    Paths, RepresentativeSegment, SessionId, SessionMetadata, SourceTrack, SpeakerCluster,
    SpeakerClusters, TrackSync, Transcript, TranscriptContext, TranscriptTemplate, Turn,
};

const SESSION: &str = "20260809-052600";
const NAME: &str = "Grace";
/// The kill is timed, not deterministic: attempts that land outside the window start over.
const MAX_ATTEMPTS: u32 = 8;

/// The fixture's embeddings, one axis each, so no reference can ever identify a voice against
/// another's. Cluster order is queue order (first appearance in the transcript).
fn embedding(cluster: u32) -> Vec<f32> {
    let mut v = vec![0.0f32; 4];
    v[cluster as usize] = 1.0;
    v
}

/// Three voices, the one person this meeting held split across clusters 0 and 1, which
/// segmentation heard at once -- the veto a lone naming cannot pass and the group answer
/// overrides. Nobody enrolled: nothing resolves by identification before the run.
fn fixture(root: &Path) {
    let paths = Paths::new(root.to_path_buf());
    let id = SessionId::parse(SESSION).unwrap();
    let session = paths.session(&id);
    std::fs::create_dir_all(session.dir()).unwrap();

    let sync = TrackSync {
        host_ticks: 1,
        timebase_numer: 125,
        timebase_denom: 3,
    };
    let metadata = SessionMetadata::new(
        id.clone(),
        "2026-08-09T05:26:00Z".parse().unwrap(),
        sync,
        sync,
    );
    metadata.write(&session.session_json()).unwrap();

    let mut clusters = (0..3u32)
        .map(|id| SpeakerCluster {
            id,
            embedding: embedding(id),
            speech_seconds: f64::from(40 - 10 * id),
            first_spoke_seconds: f64::from(id),
            heard_at_once_with: Vec::new(),
            representatives: vec![RepresentativeSegment {
                start: 0.0,
                end: 1.0,
            }],
        })
        .collect::<Vec<_>>();
    // Written on both sides, as `speaker_clusters.json` documents it.
    clusters[0].heard_at_once_with = vec![1];
    clusters[1].heard_at_once_with = vec![0];
    SpeakerClusters::new(id.clone(), clusters)
        .write(&session)
        .unwrap();

    Transcript::new(
        id,
        vec![
            turn(0.0, 0, "Unknown 1", "hi there"),
            turn(1.0, 1, "Unknown 2", "and from me"),
            turn(2.0, 2, "Unknown 3", "counting in"),
            turn(3.0, 0, "Unknown 1", "let us start"),
        ],
    )
    .write(
        &session,
        &TranscriptTemplate::resolve(&paths, None).unwrap(),
        &TranscriptContext::now(&metadata),
    )
    .unwrap();
}

fn turn(start: f64, cluster: u32, speaker: &str, text: &str) -> Turn {
    Turn {
        speaker: speaker.to_string(),
        start,
        end: start + 1.0,
        text: text.to_string(),
        source_track: SourceTrack::Speaker,
        cluster: Some(cluster),
        speaker_id_confidence: None,
    }
}

// --- the pty seam --------------------------------------------------------------------------

/// One open pty: the master the test reads and writes, the slave the child takes as its
/// controlling terminal. The size is set before the spawn because the frame draws into
/// whatever rectangle it is handed.
fn open_pty() -> (std::fs::File, std::fs::File) {
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
            ws_row: 30,
            ws_col: 100,
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
        (
            std::fs::File::from_raw_fd(master),
            std::fs::File::from_raw_fd(slave),
        )
    }
}

/// The built binary pointed at `root`, enrolled interactively through a pty.
struct Driver {
    child: Child,
    master: std::fs::File,
    /// Everything the child has written, kept so a failure can show what the frame saw.
    out: Vec<u8>,
    buf: [u8; 16384],
}

impl Driver {
    fn spawn(root: &Path) -> Self {
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
        cmd.args(["--root"])
            .arg(root)
            .args(["enroll", SESSION])
            .env("TERM", "xterm-256color")
            .stdin(Stdio::from(slave.try_clone().unwrap()))
            .stdout(Stdio::from(slave.try_clone().unwrap()))
            .stderr(Stdio::from(slave));
        let child = cmd.spawn().expect("spawning the built binary");
        Driver {
            child,
            master,
            out: Vec::new(),
            buf: [0; 16384],
        }
    }

    /// Pulls whatever is readable into `out` without blocking.
    fn pump(&mut self) {
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
    fn wait_for_frame_or_exit(&mut self, timeout: Duration) -> bool {
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
    fn feed(&mut self, keys: &[&[u8]]) {
        for key in keys {
            self.master.write_all(key).unwrap();
            self.pump();
            std::thread::sleep(Duration::from_millis(150));
        }
    }

    /// Writes bytes to the pty and nothing else -- for the keystroke that starts the commits,
    /// where the very next instruction is to watch the disk, and a settle would let the whole
    /// burst finish before the watcher starts.
    fn write_now(&mut self, bytes: &[u8]) {
        self.master.write_all(bytes).unwrap();
    }

    /// Blocks until the child is gone, keeping the pty drained so it never blocks on a full
    /// buffer. Returns its exit status.
    fn wait_exit(&mut self, timeout: Duration) -> ExitStatus {
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
    fn tail(&self) -> String {
        String::from_utf8_lossy(&self.out[self.out.len().saturating_sub(2000)..]).into_owned()
    }
}

// --- what the disk says --------------------------------------------------------------------

/// Which fixture clusters hold a `NAME` reference in the enrolled database, matched by
/// embedding because the rows carry no cluster id.
fn db_clusters(root: &Path) -> BTreeSet<u32> {
    let path = root.join("speakers.json");
    let Ok(bytes) = std::fs::read(&path) else {
        return BTreeSet::new();
    };
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["speakers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["name"] == NAME)
        .flat_map(|row| {
            let emb: Vec<f32> = row["embedding"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            (0..3u32).find(|c| embedding(*c) == emb)
        })
        .collect()
}

/// Which clusters hold a `NAME` row in this session's own names file.
fn names_clusters(root: &Path) -> BTreeSet<u32> {
    let path = root
        .join("sessions")
        .join(SESSION)
        .join("speaker_names.json");
    let Ok(bytes) = std::fs::read(&path) else {
        return BTreeSet::new();
    };
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    value["names"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["name"] == NAME)
        .map(|row| row["cluster"].as_u64().unwrap() as u32)
        .collect()
}

/// Every file under `root`, keyed by relative path: the whole state a pass could have touched.
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

// --- the attempt ---------------------------------------------------------------------------

/// An attempt that caught nothing: the process finished before the kill, or the kill landed
/// after the last member. The fixture is rebuilt and the geometry retried.
struct Missed;

/// Spawns the run, marks all three voices as one person, and kills the process the moment the
/// first member's database row is on disk. On success returns the members the walk reached in
/// the database before it died.
fn interrupt_once(root: &Path) -> Result<BTreeSet<u32>, Missed> {
    let mut driver = Driver::spawn(root);
    if !driver.wait_for_frame_or_exit(Duration::from_secs(15)) {
        // A fresh fixture always asks, so an early exit is a failure, not a miss.
        let status = driver.child.wait().unwrap();
        panic!(
            "the run died before the frame ({status}); output:\n{}",
            driver.tail()
        );
    }

    // Mark every row -- the cursor starts on the first (only) question -- then give the group
    // its name through the typed-name door, which commits the whole staged group at once. The
    // name and the confirming key go out in one breath: from that byte on disk, the walk's
    // first write is racing this test's first poll, and there is nothing to settle.
    driver.feed(&[b"\x0b", b"\x1b[B", b"\x0b", b"\x1b[B", b"\x0b"]);
    driver.write_now(NAME.as_bytes());
    driver.write_now(b"\x0e");
    let t_apply = Instant::now();

    // The first member's database row is the walk's first write: from its appearance on disk
    // until the process exits is the window the kill has to land in.
    let db_file = root.join("speakers.json");
    let pid = driver.child.id() as libc::pid_t;
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut killed = false;
    let mut detected_at: Option<Duration> = None;
    while Instant::now() < deadline {
        if std::fs::read(&db_file)
            .map(|bytes| bytes.windows(NAME.len()).any(|w| w == NAME.as_bytes()))
            .unwrap_or(false)
        {
            detected_at = Some(t_apply.elapsed());
            // SAFETY: `pid` is this test's own child, still owned by `driver` and not yet
            // waited on, so it names a live process this test is entitled to signal.
            killed = unsafe { libc::kill(pid, libc::SIGKILL) == 0 };
            break;
        }
        std::thread::sleep(Duration::from_micros(200));
    }
    if !killed && Instant::now() >= deadline {
        panic!(
            "the run never reached the first commit; output:\n{}",
            driver.tail()
        );
    }

    // The kill won the race only if the process is what it stopped.
    let status = driver.child.wait().unwrap();
    eprintln!(
        "DEBUG: killed={killed} since_apply={}ms detected={:?} status={{code:{:?},signal:{:?}}} db={:?} names={:?}",
        t_apply.elapsed().as_millis(),
        detected_at.map(|d| d.as_millis()),
        status.code(),
        status.signal(),
        db_clusters(root),
        names_clusters(root)
    );
    if !matches!(status.signal(), Some(s) if s == libc::SIGKILL) {
        if killed || status.success() {
            // The process exited before the signal landed: the window was missed, not broken.
            return Err(Missed);
        }
        panic!(
            "the run died on its own ({status}); output:\n{}",
            driver.tail()
        );
    }

    let db = db_clusters(root);
    let names = names_clusters(root);

    // (a) The fixed order, read off the corpse: the database precedes the names file, so a
    // names row without a database row is the torn state the write order exists to prevent.
    assert!(
        names.is_subset(&db),
        "a names row outlived its database row: db {db:?}, names {names:?}"
    );
    // The walk commits in queue order, so the database holds a prefix starting at cluster 0.
    assert!(
        db.iter().copied().eq(0..db.len() as u32),
        "the database holds {db:?}: not a queue-order prefix"
    );
    assert!(!db.is_empty(), "killed, yet no member reached the database");
    // A landing after the last member proved nothing new: rebuild and try again.
    if db.len() == 3 && names.len() == 3 {
        return Err(Missed);
    }
    Ok(db)
}

// --- the test ------------------------------------------------------------------------------

#[test]
fn an_interrupt_mid_group_commit_leaves_a_prefix_and_a_rerun_converges() {
    for attempt in 1..=MAX_ATTEMPTS {
        let dir = tempfile::tempdir().unwrap();
        fixture(dir.path());

        let db = match interrupt_once(dir.path()) {
            Ok(db) => db,
            Err(Missed) => {
                eprintln!("attempt {attempt}: the kill missed the window; rebuilding");
                continue;
            }
        };
        complete_and_verify(dir.path(), &db);
        return;
    }
    panic!("the SIGKILL never landed mid-group in {MAX_ATTEMPTS} attempts");
}

/// (b) and (c): the same frame, pointed at the interrupted root, finishes the group; a third
/// pass then finds nothing left to do and writes nothing.
fn complete_and_verify(root: &Path, db: &BTreeSet<u32>) {
    // The corpse and the disk agree on what the walk reached.
    assert_eq!(&db_clusters(root), db);
    let mut driver = Driver::spawn(root);
    if driver.wait_for_frame_or_exit(Duration::from_secs(15)) {
        // The gesture is the group door again, over every row the queue offers -- including the
        // ones the interrupt already committed. The group's forced tier stands each member's
        // declaration up in both stores, which is what makes the re-run converge rather than
        // merely finish. (A plain confirmation of an identified voice now does the same for a
        // stranded member: since TASK-055 it keeps -- rather than forgets -- the names-file row
        // when the database already holds the reference, so the natural recovery gesture no
        // longer leaves the voice demoted. The group door remains the canonical recovery.)
        driver.feed(&[b"\x0b", b"\x1b[B", b"\x0b", b"\x1b[B", b"\x0b"]);
        driver.write_now(NAME.as_bytes());
        driver.write_now(b"\x0e");
        let status = driver.wait_exit(Duration::from_secs(30));
        assert!(
            status.success(),
            "the re-run ended badly ({status}); output:\n{}",
            driver.tail()
        );
    }
    // A re-run that found nothing unresolved took no frame at all and exited clean: the
    // interrupt left a converged root, and the pass brought any stale transcript up to date
    // on the way past. The assertions below hold either way.

    // Every speaker-track turn reads the group's name: the interrupt lost no answer that was
    // given, and the re-run supplied the rest.
    let paths = Paths::new(root.to_path_buf());
    let session = paths.session(&SessionId::parse(SESSION).unwrap());
    let transcript = Transcript::read(&session.transcript_json()).unwrap();
    let speakers: Vec<&str> = transcript
        .turns
        .iter()
        .filter(|t| t.source_track == SourceTrack::Speaker)
        .map(|t| t.speaker.as_str())
        .collect();
    assert!(
        speakers.iter().all(|who| *who == NAME),
        "after the re-run the transcript still holds {speakers:?}"
    );
    assert_eq!(
        db_clusters(root),
        (0..3).collect(),
        "every reference stands"
    );

    // (c) A third pass: nothing unresolved, nothing written. Piped stdin takes the line-prompt
    // answerer, which is the point -- the pass must find no question worth a frame.
    let before = snapshot(root);
    let output = Command::new(env!("CARGO_BIN_EXE_meethook"))
        .args(["--root"])
        .arg(root)
        .args(["enroll", SESSION])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(snapshot(root), before, "the third pass wrote something");
}
