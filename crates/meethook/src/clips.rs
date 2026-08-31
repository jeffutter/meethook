//! Plays a voice's audio through whatever player the box has.
//!
//! Both answerers need playback and neither needs it differently, so the whole engine --
//! finding a player, the scratch files it reads, the child's life, and where a clip stands --
//! sits here rather than leaking out of the command bodies as pub(crate) items.

use std::fs;
#[cfg(not(target_os = "macos"))]
use std::path::Path;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use meethook_enroll::write_clip;
use meethook_transcribe::TARGET_RATE;

/// How far through a clip playback has got.
///
/// Wall time, not a position read off the player: `afplay` reports nothing about where it is --
/// `-v -t -r -q -d` is the whole of its option set -- so this is how long ago the child was
/// spawned, measured against the clip's own length. It is therefore approximate, and drifts by
/// whatever the audio device spent starting up, so the wording that shows it must not pretend to
/// be a cursor into the audio.
///
/// `elapsed` is clamped to `length`: a clip that has run over its own length is one whose child
/// has not been reaped yet, and "playing 1m 52s of 1m 47s" reads as a bug rather than as latency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Progress {
    /// How long ago playback started, never more than `length`.
    pub(crate) elapsed: Duration,
    /// How long the clip runs for, from its sample count.
    pub(crate) length: Duration,
}

/// One clip being played, and everything owed when it stops.
struct Playing {
    child: Child,
    /// The clip's own file, unlinked when the child is reaped: fifty replays of a three-minute
    /// clip would otherwise sit in the scratch directory until the run ended.
    path: PathBuf,
    /// Which player owns the child, so a failure names the program that actually ran.
    player: &'static str,
    started: Instant,
    length: Duration,
}

/// How long a clip runs for, from its sample count.
///
/// Its own function because it is the one piece of arithmetic here that `cargo test` can reach,
/// and a wrong denominator is invisible except as a position that drifts against the audio.
fn length(clip: &[f32]) -> Duration {
    Duration::from_secs_f64(clip.len() as f64 / f64::from(TARGET_RATE))
}

/// The player this platform plays clips with, and the arguments it takes before the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Player {
    program: &'static str,
    /// Verbosity/display flags that keep the player from painting over the terminal or
    /// opening a video window for what is a bare WAV.
    args: &'static [&'static str],
}

/// What the refusal names when no player is found, so the message and the search agree.
#[cfg(target_os = "macos")]
const PLAYER_SEARCH_LIST: &str = "afplay";
#[cfg(not(target_os = "macos"))]
const PLAYER_SEARCH_LIST: &str = "paplay, aplay, ffplay, mpv";

/// Which of the known players the given PATH actually contains, first match winning.
///
/// Pure over the PATH string rather than reading the environment itself, so the choice --
/// and the order in which players are preferred -- is decidable in `cargo test`.
#[cfg(target_os = "macos")]
fn choose_player(_path_env: &str) -> Option<Player> {
    // `afplay` ships with every Mac and takes nothing but the file.
    Some(Player {
        program: "afplay",
        args: &[],
    })
}

#[cfg(not(target_os = "macos"))]
fn choose_player(path_env: &str) -> Option<Player> {
    // Native WAV players first -- the clip is a WAV and neither of them has to decode
    // anything -- then the general players, quietest flags first, so a headless box with
    // only ffmpeg still gets playback without a video window or a wall of log lines.
    let candidates = [
        Player {
            program: "paplay",
            args: &[],
        },
        Player {
            program: "aplay",
            args: &[],
        },
        Player {
            program: "ffplay",
            args: &["-nodisp", "-autoexit", "-loglevel", "quiet"],
        },
        Player {
            program: "mpv",
            args: &["--really-quiet", "--no-video"],
        },
    ];
    // Empty entries are skipped: a bare ":" in PATH would otherwise be read as the current
    // directory, where a file named `mpv` is a plausible coincidence.
    let dirs: Vec<&str> = path_env.split(':').filter(|dir| !dir.is_empty()).collect();
    candidates.into_iter().find(|candidate| {
        dirs.iter()
            .any(|dir| Path::new(dir).join(candidate.program).is_file())
    })
}

/// [`choose_player`] pointed at this process's real PATH. A missing PATH means no player can
/// be found at all, which is the same answer as an empty one.
fn resolve_player() -> Option<Player> {
    let path = std::env::var_os("PATH")?;
    // Bound rather than chained: `to_string_lossy` may borrow its input, and a temporary
    // `Cow` would be dropped while `choose_player` still held it.
    let path_env = path.to_string_lossy();
    choose_player(&path_env)
}

/// Playing a voice's audio, for whichever answerer is asking.
///
/// Both of them need this and neither of them needs it differently: the samples are the same
/// samples and `afplay` is the same `afplay`. What they do *not* share is where the words go --
/// the line prompt prints a parenthetical under the snippets, the frame puts a status line in a
/// pane -- so this returns what went wrong rather than saying it, and the empty-clip case stays
/// with the callers for the same reason.
///
/// What they also do not share is *waiting*. [`Clips::play`] starts and waits, which is right
/// behind a line prompt; the frame calls [`Clips::start`], [`Clips::poll`] and [`Clips::stop`] so
/// a three-minute clip does not hold the screen. Both go through the same spawn, so there is one
/// wording for a failure and not two.
#[derive(Default)]
pub(crate) struct Clips {
    /// Where clips are written for the player, created on first use and removed when the run
    /// ends. `afplay` has no start offset, so playing part of a recording means handing it a
    /// file that contains only that part.
    dir: Option<tempfile::TempDir>,
    /// The child, while there is one.
    playing: Option<Playing>,
    /// How many clips this run has written, and so what the next one is called. Unique names
    /// rather than one `clip.wav`: overwriting the file a live `afplay` is reading is the bug a
    /// fixed name invites the moment playback stops being synchronous.
    written: usize,
}

impl Clips {
    /// Spawns a player for `clip` and returns immediately.
    ///
    /// Anything already playing is stopped first, so pressing play again restarts the clip from
    /// the beginning whether or not the previous one had finished.
    ///
    /// Never fatal to a run, for the reason [`Clips::play`] gives. An empty `clip` is a
    /// successful no-op and the caller's sentence to write.
    pub(crate) fn start(&mut self, clip: &[f32]) -> Result<()> {
        if clip.is_empty() {
            return Ok(());
        }
        self.stop();
        // Owned, and taken before `written` is bumped: `&self.dir` and the write to `self.written`
        // cannot both be live.
        let dir = match &self.dir {
            Some(dir) => dir.path().to_path_buf(),
            None => self.dir.insert(tempfile::tempdir()?).path().to_path_buf(),
        };
        self.written += 1;
        let path = dir.join(format!("clip-{}.wav", self.written));
        write_clip(&path, clip)?;
        // A missing player is a reportable condition, not a crash: enrollment degrades to
        // reading the snippets, which is often enough to recognise somebody.
        let player = resolve_player().ok_or_else(|| {
            anyhow::anyhow!("no audio player found on PATH (looked for {PLAYER_SEARCH_LIST})")
        })?;
        // All three streams closed. Under raw mode and an alternate screen one line of
        // `AudioQueueStart failed` from the child paints over the frame, and ratatui's diffed
        // redraw will not clear a cell it never wrote.
        let mut command = Command::new(player.program);
        command.args(player.args);
        command.arg(&path);
        let child = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        self.playing = Some(Playing {
            child,
            path,
            player: player.program,
            started: Instant::now(),
            length: length(clip),
        });
        Ok(())
    }

    /// Reaps a finished player, and says how far through the clip a running one has got.
    ///
    /// `Ok(None)` covers both "nothing is playing" and "it has just finished cleanly", which are
    /// the same thing to a caller: there is no position to show. `Err` is a player that exited
    /// non-zero -- a clip that will not play is only knowable here, once the child is reaped --
    /// and in that case playback has already been torn down.
    pub(crate) fn poll(&mut self) -> Result<Option<Progress>> {
        let Some(playing) = self.playing.as_mut() else {
            return Ok(None);
        };
        let reaped = playing.child.try_wait();
        let progress = Progress {
            elapsed: playing.started.elapsed().min(playing.length),
            length: playing.length,
        };
        let player = playing.player;
        match reaped {
            Ok(None) => Ok(Some(progress)),
            Ok(Some(status)) => {
                self.stop();
                if status.success() {
                    Ok(None)
                } else {
                    bail!("{player} exited with {status}");
                }
            }
            Err(e) => {
                self.stop();
                Err(e).context(format!("waiting on {player}"))
            }
        }
    }

    /// Kills whatever is playing, reaps it, and takes its file with it.
    ///
    /// Infallible on purpose: there is nothing a caller could do about a kill that failed, and
    /// this runs on the way out of every question. The reap is the part that matters -- an
    /// unreaped child is a zombie outliving the question it belongs to.
    pub(crate) fn stop(&mut self) {
        if let Some(mut playing) = self.playing.take() {
            let _ = playing.child.kill();
            let _ = playing.child.wait();
            let _ = fs::remove_file(&playing.path);
        }
    }

    /// Plays a clip and waits for it to finish.
    ///
    /// Never fatal to a run. A missing `afplay`, a full temp directory, a truncated
    /// `speaker.wav` -- none of them are a reason to stop asking, because the snippets are often
    /// enough to recognise somebody on their own. Hence `Result` and not `?` at the call sites.
    ///
    /// An empty `clip` is a successful no-op here and the caller's sentence to write: it is not a
    /// failure of anything, and the two answerers word it differently.
    ///
    /// [`Clips::start`] plus the wait, so the spawn and the wording of a failure are written once.
    pub(crate) fn play(&mut self, clip: &[f32]) -> Result<()> {
        self.start(clip)?;
        let Some(mut playing) = self.playing.take() else {
            return Ok(());
        };
        let status = playing.child.wait()?;
        let player = playing.player;
        let _ = fs::remove_file(&playing.path);
        if !status.success() {
            bail!("{player} exited with {status}");
        }
        Ok(())
    }
}

impl Drop for Clips {
    /// No `afplay` outlives the run, on any path out of it -- and the ordering is the point:
    /// `drop` runs before any field is dropped, so the child is dead and its file unlinked before
    /// the `TempDir` removes the directory it was reading from.
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{choose_player, length};

    /// The clip's own length, which is the whole denominator of the position the frame shows.
    /// `afplay` reports nothing, so a wrong rate here is invisible except as an indicator that
    /// drifts against what is coming out of the speakers.
    #[test]
    fn a_clip_is_as_long_as_its_samples_at_sixteen_kilohertz() {
        assert_eq!(length(&vec![0.0; 16_000]), Duration::from_secs(1));
        assert_eq!(length(&vec![0.0; 8_000]), Duration::from_millis(500));
        assert_eq!(length(&[]), Duration::ZERO);
    }

    /// A directory that presents exactly the players named, so the choice is decided by
    /// presence rather than by whatever happens to be on the machine running the test.
    #[cfg(not(target_os = "macos"))]
    fn fake_path(dir: &std::path::Path, present: &[&str]) {
        for program in present {
            std::fs::write(dir.join(*program), b"fake").unwrap();
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_always_uses_afplay() {
        let player = choose_player("").expect("afplay is part of every macOS");
        assert_eq!(player.program, "afplay");
        assert!(player.args.is_empty(), "afplay takes nothing but the file");
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn the_player_is_chosen_from_what_the_path_actually_contains() {
        let dir = tempfile::tempdir().unwrap();
        fake_path(dir.path(), &["ffplay", "mpv"]);
        let path = dir.path().to_str().unwrap();

        // Neither native WAV player present, so the first general player found wins -- with
        // its quieting flags pinned, since a visible video window or log wall would defeat
        // the purpose of playing a clip behind a prompt.
        let player = choose_player(path).expect("ffplay should be found");
        assert_eq!(player.program, "ffplay");
        assert_eq!(player.args, ["-nodisp", "-autoexit", "-loglevel", "quiet"]);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn native_wav_players_are_preferred_over_general_ones() {
        let dir = tempfile::tempdir().unwrap();
        fake_path(dir.path(), &["mpv", "paplay"]);
        let path = dir.path().to_str().unwrap();

        let player = choose_player(path).expect("paplay should be found");
        assert_eq!(player.program, "paplay");
        assert!(player.args.is_empty());
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn an_empty_path_finds_no_player_and_empty_entries_are_not_the_cwd() {
        let dir = tempfile::tempdir().unwrap();
        // A file named like a player in the *current* directory must not count: empty PATH
        // entries are skipped rather than read as ".".
        std::fs::write("mpv", b"fake").unwrap();
        std::fs::remove_file("mpv").unwrap();
        assert!(choose_player(dir.path().to_str().unwrap()).is_none());
        assert!(choose_player(":").is_none());
    }
}
