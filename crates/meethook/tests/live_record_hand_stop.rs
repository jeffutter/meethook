#![cfg(target_os = "macos")]

//! The hand stop (`s`) proven against the real frame over a real pty, on real hardware.
//!
//! TASK-066.07.01 proves the route with a scripted capture: that `record_loop` breaks to its one
//! finalize point on `Event::StopSession` and comes back to watching rather than quitting. What
//! only a live process can prove is the claim TASK-066.07 AC #4 actually makes -- that a person
//! watching the *real* frame presses `s` mid-call, that call's `session.json` lands, and the run
//! goes on watching for the next one. This file arranges exactly that, and nothing else.
//!
//! ## Why a microphone is involved at all
//!
//! There is no fake capture reachable from a test. `crates/meethook` is bin-only, `trait Capture`
//! is private to the crate, and `FakeCapture` lives inside its `#[cfg(test)] mod tests`; the only
//! handle from here is `env!("CARGO_BIN_EXE_meethook")`. So the session this test ends has to be
//! opened by something that really is capturing audio, which is what
//! `crates/meethook-record`'s `mic-hold` example stands in for.
//!
//! The trigger counts *which processes are capturing input*, never how loud the room is:
//! `MicActivityWatcher` derives activity from CoreAudio's per-process `IsRunningInput` objects,
//! minus this process itself. There is no level meter and no threshold anywhere in this workspace,
//! so no tone, no clap, and no volume change can start a session -- and muting the machine cannot
//! split one either. Any text in this repo that says the recorder triggers on loudness is a bug.
//!
//! `mic-hold` therefore proves only that the machine has a usable default input device: without a
//! microphone grant `AVAudioEngine` hands back silence rather than an error, so a holder that
//! prints its `pid N holding...` line says nothing about grants. Grant diagnosis comes from the
//! child's own refusal text, which is precise, and is quoted by the failure messages below.
//!
//! ## Machine conditions, and what each one looks like
//!
//! Nothing here skips. The single `#[ignore]` is the only thing that declines to run, and it names
//! what it needs. Inside the test every unmet condition panics carrying the reconstructed grid,
//! the child's status, and the byte counts. Five things can fail this while the implementation is
//! correct, and all five are about the machine:
//!
//! - **A TCC grant absent.** `record` refuses outright in well under a second (measured 0.74 s),
//!   because only a `NotDetermined` status ever waits for a human; `Denied` and `Restricted` answer
//!   immediately. macOS charges these grants to the *terminal application*, so the fix is granted
//!   to iTerm/Terminal/Ghostty, never to a "meethook" entry. This shows up as
//!   `wait_for_frame_or_exit` returning `false`, and the panic quotes the refusal verbatim.
//! - **A TCC prompt unanswered.** Capture *and* calendar prompts each wait 120 s
//!   (`PROMPT_TIMEOUT`), and the calendar one runs before the frame exists at all. Settle all
//!   three once, by hand, in the same terminal app, and startup drops to seconds;
//!   `FRAME_UP_BUDGET` is deliberately generous so a first-run prompt storm reads as what it is
//!   rather than as a hang. If the captured output the driver prints contains
//!   `Asking macOS for calendar access`, the two capture grants were answered and the calendar one
//!   is the dialog sitting unanswered.
//! - **No usable default input device.** Caught cheaply by the holder's own non-zero exit, whose
//!   message the panic quotes. That exit code means "no input device", never "ungranted".
//! - **Something else on the machine holding the microphone.** Under the predicate a third holder
//!   is indistinguishable from ours. Mostly harmless -- it can only delay the falling edge -- but
//!   a phantom holder that keeps the boolean true after our holder dies starves the re-arm phase
//!   of its rising edge. See below for reading who holds what.
//! - **A leftover `meethook record` from an earlier run.** The child refuses at the lock, which is
//!   a clear failure rather than a mystery. Different roots do not contend, so this is only ever
//!   about a second run of *this* proof.
//!
//! One more worth knowing before debugging a missing `session.json`: a track that never received a
//! *single* buffer makes `finish` refuse to write metadata at all (`Error::SilentTrack`), and the
//! frame shows that as a trouble pane. That is a property of the machine's capture stack, not of
//! the keypress.
//!
//! ## Reading who holds the microphone
//!
//! The driver scrubs `MEETHOOK_ACTIVITY_DEBUG` (and the calendar and timing switches) from the
//! child, because those diagnostics print to stderr and stderr shares the pty slave with the
//! frame's stdout: letting them through would feed the grid reconstruction lines the frame never
//! painted. So you cannot debug a failing proof by setting that variable here. Run the manual
//! recipe -- `MEETHOOK_ACTIVITY_DEBUG=1 meethook record --root /tmp/somewhere` alongside
//! `cargo run --manifest-path crates/meethook-record/Cargo.toml --example mic-hold -- 30` -- and
//! the walker names every process it counts.
//!
//! ## Running it
//!
//! ```text
//! cargo build --manifest-path crates/meethook-record/Cargo.toml --example mic-hold
//! cargo nextest run -p meethook --run-ignored only -E 'binary(live_record_hand_stop)' --nocapture
//! ```
//!
//! `--nocapture` is what lets the run's own output be pasted into the ticket: nextest swallows the
//! stdout of a *passing* test otherwise.
//!
//! Typical wall time, once the three prompts are settled: a couple of minutes. Worst case, with a
//! prompt storm: nearer twenty minutes. The budgets below are what give, never `Timing`'s
//! intervals -- those are tuned against CoreAudio's notification burst rate (measured elsewhere at
//! 9+ callbacks a second during a Meet join, most with no state change), and shortening them to
//! make a test faster would trade the trigger's correctness for someone else's convenience.
//!
//! ## What it does to the machine while it runs
//!
//! The alternate screen it takes over is the pty's, not yours. What is genuinely system-global is
//! the microphone: while this runs, the machine's default input device is being watched, so a real
//! call that happens on it during the run will start a real session in whichever `record` sees the
//! edge first. Run it when you are not about to take a call. Sessions land in a fresh temporary
//! root that is deleted on the way out, and no model weights are touched -- `record` never calls
//! `ensure_model`.

mod common;

use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use common::{Driver, PTY_COLS, PTY_ROWS};
use meethook_session::wav::{self, TrackEvidence};
use meethook_session::{Classification, DiscoveredSession, LockState, Paths, discover_sessions};

// ---------------------------------------------------------------------------------------------
// Budgets. Each names the machine fact it is derived from; none of them is a tuning knob.
// ---------------------------------------------------------------------------------------------

/// Long enough to survive the worst startup this command can have: three first-run TCC prompts
/// (Screen, Microphone, Calendar), each waiting out the shared 120 s `PROMPT_TIMEOUT` -- the
/// calendar one from `record.rs` before the frame even exists. Settled prompts bring this to
/// seconds; see the module doc.
const FRAME_UP_BUDGET: Duration = Duration::from_secs(400);

/// How long a holder is given to open the device and print its line. Generous because the wait
/// ends in a definite answer either way: the printed line, or EOF plus its stderr.
const HOLDER_START_BUDGET: Duration = Duration::from_secs(60);

/// Idle has no timer at all (`Timing::IDLE` is grace 0 / retry 1 / recheck 0), so the only thing
/// that can start a session is the watcher's listener firing on the rising edge. Thirty seconds
/// says that edge was lost, not that it was slow.
const SESSION_START_BUDGET: Duration = Duration::from_secs(30);

/// Strictly longer than `CHECKPOINT_INTERVAL` (5 s, `track.rs`), so growth across this wait can
/// only mean the capture callback kept writing and the header kept being rewritten and synced.
const GROWTH_WAIT: Duration = Duration::from_secs(7);

/// The hand stop's own path: grace 3 s + retry 2 s is the *worst-case natural* stop latency, and
/// the hand stop takes neither -- it breaks straight to the one finalize point. Ten seconds is
/// therefore already ~2x the longest the keypress route can legitimately take.
const FINALIZE_BUDGET: Duration = Duration::from_secs(10);

/// Strictly longer than grace + recheck, so "no second session appeared" outruns any
/// re-check-and-reopen the implementation might be doing behind our back.
const NO_RESTART_WINDOW: Duration = Duration::from_secs(10);

/// Strictly longer than the 2 s re-check cadence, so a release edge that merely went unnoticed is
/// ruled out rather than blamed for the next phase's failure.
const REARM_WAIT: Duration = Duration::from_secs(6);

/// A second holder plus a start edge plus `begin`'s bounded retry (5 attempts at 2 s).
const SECOND_SESSION_BUDGET: Duration = Duration::from_secs(40);

/// Ctrl-C is delivered as a keystroke into a raw-mode tty, so this covers the frame noticing it,
/// the loop breaking, a whole second finalize, and the frame's teardown.
const EXIT_BUDGET: Duration = Duration::from_secs(20);

/// Longer than every phase sequence put together, so the holder is never the thing that ends a
/// session -- the falling edge that would do that is only ever produced deliberately here.
const HOLDER_SECONDS: u64 = 180;

/// How often the finalize watch pokes the frame. See `wake_frame` for why poking is necessary at
/// all, and for why this has to be far below any plausible time `finish` takes.
const WAKE_EVERY: Duration = Duration::from_millis(2);

/// A settle long enough for the frame thread to wake, drain its notes and draw, used where a
/// single keystroke's effect is being measured.
const SETTLE: Duration = Duration::from_millis(300);

// ---------------------------------------------------------------------------------------------
// The screen, reconstructed
// ---------------------------------------------------------------------------------------------

/// The grid the frame's bytes paint, rebuilt rather than grepped.
///
/// Grepping the raw pty stream for painted words does not work, and absence-by-bytes is unsound,
/// because ratatui draws by diff: a cell whose glyph *and* style are unchanged emits nothing.
/// Measured against this frame's own strings with `Buffer::diff`, idle -> recording loses the final
/// `g` of `recording` and the last three characters of the session id to cells the diff skipped,
/// and the `stop and exit` row keeps letters across the return to idle wherever the two frames
/// happen to want the same glyph in the same cell -- so "the hint disappeared" cannot be read off
/// the stream at all, and "the word appeared" depends on which digits of the elapsed clock happened
/// to change.
///
/// Presence of a *repaint*, as opposed to presence of a word, is readable off the stream, and is
/// used that way where an idle frame has nothing else left to say: every draw hides the cursor
/// whether or not any cell changed (`Terminal::hide_cursor` goes to the backend without consulting
/// the diff, and this frame never calls `Frame::set_cursor_position`, so it always takes that
/// arm). Fresh bytes therefore mean a draw happened; they never say what it painted.
///
/// So this is a small terminal emulator: enough of one to follow what this frame emits and nothing
/// more. Cells are written *at* the cursor and overwrite, which is what makes a retained-but-not-
/// repainted glyph correct and makes the absence assertions sound. Wide-glyph widths are not
/// modelled (`●` advances one cell here): any drift inside a run is re-synced by the next cursor
/// move, and every needle is matched as a substring rather than at a column, so a one-cell error
/// cannot flip an assertion. Writes outside the grid are ignored rather than panicked on, because
/// a test that dies on a resize tells you nothing about the keypress it was testing.
///
/// Deliberately private to this file. It earns a place in `tests/common/` when a second test needs
/// pixel assertions; the pty driver moved there because two tests needed *that*.
mod screen {
    /// One absorbing call and two queries; the parser stays inside.
    pub(crate) struct Screen {
        cells: Vec<Vec<char>>,
        row: usize,
        col: usize,
        saved: Option<(usize, usize)>,
        /// A UTF-8 sequence split across reads. Escape sequences are ASCII, so a lead byte is the
        /// only place a partial character can begin.
        pending: Vec<u8>,
    }

    impl Screen {
        pub(crate) fn new(rows: usize, cols: usize) -> Self {
            Screen {
                cells: vec![vec![' '; cols]; rows],
                row: 0,
                col: 0,
                saved: None,
                pending: Vec::new(),
            }
        }

        /// Feeds every byte appended to the stream since the last call.
        pub(crate) fn absorb(&mut self, bytes: &[u8]) {
            let mut i = 0;
            while i < bytes.len() {
                let b = bytes[i];
                if b == 0x1b {
                    // An escape interrupts an unfinished character: it was never going to
                    // complete, so drop it and parse the escape.
                    self.pending.clear();
                    i = self.escape(bytes, i + 1);
                    continue;
                }
                if b >= 0x80 {
                    self.pending.push(b);
                    i += 1;
                    self.flush_pending();
                    continue;
                }
                if !self.pending.is_empty() {
                    self.pending.clear();
                }
                match b {
                    b'\r' => self.col = 0,
                    b'\n' => {
                        self.row += 1;
                        self.col = 0;
                    }
                    0x08 => self.col = self.col.saturating_sub(1),
                    0x09 => self.col = (self.col / 8 + 1) * 8,
                    _ if b.is_ascii_control() => {}
                    _ => {
                        self.put(b as char);
                    }
                }
                i += 1;
            }
        }

        /// Whether the grid currently shows `needle` somewhere on one line.
        pub(crate) fn contains(&self, needle: &str) -> bool {
            self.text().lines().any(|line| line.contains(needle))
        }

        /// The grid as text, for a failure message.
        pub(crate) fn text(&self) -> String {
            self.cells
                .iter()
                .map(|row| {
                    let line: String = row.iter().collect();
                    line.trim_end().to_string()
                })
                .collect::<Vec<_>>()
                .join("\n")
        }

        /// Writes one character, growing the grid when the frame addresses a cell outside the
        /// rectangle the driver asked the pty for.
        ///
        /// Growing is not politeness -- it is the difference between a diagnosis and a false
        /// accusation. The first run on real hardware addressed its footer at row 71 although the
        /// pty was requested thirty rows tall, and a grid fixed at the requested size dropped that
        /// row silently. The proof then reported that `record` had never painted the path to its
        /// own sessions directory, which was untrue: the bytes were there, the model threw them
        /// away. A terminal model may be wrong about geometry; it may not lose bytes quietly.
        fn put(&mut self, ch: char) {
            while self.cells.len() <= self.row {
                self.cells.push(Vec::new());
            }
            let row = &mut self.cells[self.row];
            while row.len() <= self.col {
                row.push(' ');
            }
            row[self.col] = ch;
            self.col += 1;
        }

        /// Whether a private-mode sequence is the alternate-screen switch.
        fn switches_alt_screen(body: &[u8]) -> bool {
            // `body` starts with the `?` that made this a private sequence.
            matches!(std::str::from_utf8(&body[1..]), Ok(modes) if modes.contains("1049"))
        }

        /// Emits the pending bytes if they now form a character; keeps them if they are merely
        /// incomplete, and drops them if they can no longer become one.
        fn flush_pending(&mut self) {
            let pending = std::mem::take(&mut self.pending);
            match std::str::from_utf8(&pending) {
                Ok(text) => {
                    for ch in text.chars() {
                        self.put(ch);
                    }
                }
                // Still incomplete: hand the bytes back and wait for the rest of the character.
                Err(e) if e.error_len().is_none() => self.pending = pending,
                // No prefix of this can become a character: drop it and resync on the next byte.
                Err(_) => {}
            }
        }

        fn clear_all(&mut self) {
            for row in &mut self.cells {
                row.fill(' ');
            }
            self.row = 0;
            self.col = 0;
        }

        fn clear_line(&mut self, from: usize) {
            if self.row >= self.cells.len() {
                return;
            }
            let width = self.cells[0].len();
            let (start, end) = match from {
                0 => (self.col.min(width), width),
                1 => (0, self.col.min(width)),
                _ => (0, width),
            };
            for cell in &mut self.cells[self.row][start..end] {
                *cell = ' ';
            }
        }

        /// Parses one escape sequence starting just after the `ESC`, and returns the index one
        /// past it. Unrecognised sequences are consumed and ignored: an unknown control is not
        /// licence to misparse the rest of the frame.
        fn escape(&mut self, bytes: &[u8], i: usize) -> usize {
            if i >= bytes.len() {
                return i;
            }
            match bytes[i] {
                b'[' => {
                    let start = i + 1;
                    let mut end = start;
                    // Parameter and intermediate bytes, then one final byte in @ ..= ~.
                    while end < bytes.len() && bytes[end] < 0x40 {
                        end += 1;
                    }
                    if end == bytes.len() {
                        return end;
                    }
                    let body = &bytes[start..end];
                    let private = body.first() == Some(&b'?');
                    // Parameters are separated by `;`, so the separator must survive until the
                    // split: filtering to digits first folds `3;7` into the single number 37 and
                    // lands every cursor address in the wrong cell. Anything else that is not a
                    // digit -- intermediates, the private `?` -- is dropped within its own field.
                    let fields = || {
                        let body = if private { &body[1..] } else { body };
                        body.split(|byte| *byte == b';').map(|field| {
                            field
                                .iter()
                                .copied()
                                .filter(|byte| byte.is_ascii_digit())
                                .collect::<Vec<u8>>()
                        })
                    };
                    // First parameter only, which is all this frame's clears and moves carry.
                    let numeric = |default: usize| -> usize {
                        match fields()
                            .next()
                            .and_then(|digits| String::from_utf8(digits).ok())
                            .and_then(|text| text.parse::<usize>().ok())
                        {
                            Some(v) if v > 0 => v,
                            _ => default,
                        }
                    };
                    let params = || -> Vec<usize> {
                        fields()
                            .map(|digits| {
                                String::from_utf8(digits)
                                    .ok()
                                    .and_then(|text| text.parse::<usize>().ok())
                                    .unwrap_or(1)
                                    .max(1)
                            })
                            .collect()
                    };
                    match bytes[end] {
                        b'H' | b'f' => {
                            let p = params();
                            self.row = p.first().copied().unwrap_or(1) - 1;
                            self.col = p.get(1).copied().unwrap_or(1) - 1;
                        }
                        b'G' => self.col = numeric(1) - 1,
                        b'd' => self.row = numeric(1) - 1,
                        b'A' => self.row = self.row.saturating_sub(numeric(1)),
                        b'B' => self.row += numeric(1),
                        b'C' => self.col += numeric(1),
                        b'D' => self.col = self.col.saturating_sub(numeric(1)),
                        b'J' => self.clear_all(),
                        b'K' => self.clear_line(numeric(0)),
                        b'm' => {} // SGR: styling changes glyphs' appearance, never their cells.
                        b's' => self.saved = Some((self.row, self.col)),
                        b'u' => {
                            if let Some((row, col)) = self.saved {
                                self.row = row;
                                self.col = col;
                            }
                        }
                        // Entering or leaving the alternate screen is the reset point, the same
                        // oracle `wait_for_frame_or_exit` uses: the frame owns a clean slate either
                        // way, and anything the child printed before taking the terminal belongs to
                        // plain mode, not to this grid.
                        b'h' | b'l' if private && Self::switches_alt_screen(body) => {
                            self.clear_all()
                        }
                        _ => {}
                    }
                    end + 1
                }
                b']' => {
                    // OSC: terminated by BEL or ST (`ESC \`).
                    let mut j = i + 1;
                    while j < bytes.len() {
                        if bytes[j] == 0x07 {
                            return j + 1;
                        }
                        if bytes[j] == 0x1b && bytes.get(j + 1) == Some(&b'\\') {
                            return j + 2;
                        }
                        j += 1;
                    }
                    j
                }
                b'(' | b')' => (i + 2).min(bytes.len()),
                _ => (i + 1).min(bytes.len()),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        //! The frame's own strings, fed through the emulator.
        //!
        ////! These are the reason the emulator is trusted to say "absent": the live proof cannot tell a
        ////! needle the parser misplaced from a needle the frame never painted, and on a machine with the
        ////! grants that confusion costs an hour of wall clock rather than a failed assertion. They run
        ////! in the ordinary suite -- no hardware, no grants, no device.
        use super::Screen;
        use std::sync::mpsc;
        use std::time::Duration;

        /// Paint a byte stream onto a fresh grid, as several writes: ratatui flushes per draw, and the
        /// pty delivers per read, so a single call would test a stream this parser never sees.
        fn painted(chunks: &[&[u8]]) -> Screen {
            let mut screen = Screen::new(30, 100);
            for chunk in chunks {
                screen.absorb(chunk);
            }
            screen
        }

        fn row(screen: &Screen, n: usize) -> String {
            screen.text().lines().nth(n).unwrap_or("").to_string()
        }

        /// The first idle frame exactly as it came off a real Mac's pty, parked there by a failing
        /// run's evidence dump rather than typed by hand. Nothing above reproduces the one thing
        /// that mattered: this stream addresses its footer at row 71 while the pty had been asked
        /// for thirty rows, so a grid fixed at the requested height loses the line naming this
        /// run's root, and the proof then accuses `record` of never painting it. Real bytes, so the
        /// argument is settled without a microphone, without grants, and without anyone's weekend.
        static CAPTURED_IDLE_FRAME: &[u8] =
            b"\x1b[?1049h\x1b[1;1H\x1b[1mmeethook record\x1b[22m\x1b[2m  \xc2\xb7 watching\x1b[2;1Hwatching for the next call\x1b[4;1H\x1b[22mWatching\x1b[4;10Hthe\x1b[4;14Hdefault\x1b[4;22Hmicrophone.\x1b[4;34HPress\x1b[4;40HCtrl-C\x1b[4;47Hto\x1b[4;50Hstop.\x1b[6;1H\x1b[1mCtrl-C / Ctrl-D\x1b[22m\x1b[2m  stop and exit\x1b[71;1Hsessions  /var/folders/1r/hgl3f3q90yq88xfs02hjdyh00000gn/T/.tmppRB2bi/sessions\x1b[39m\x1b[49m\x1b[59m\x1b[0m\x1b[?25l";

        #[test]
        fn the_frame_captured_on_hardware_lands_every_row_it_addresses() {
            let screen = painted(&[CAPTURED_IDLE_FRAME]);
            let text = screen.text();
            assert!(
                text.contains("watching for the next call"),
                "the status line is the needle phase 1 waits on:\n{text}"
            );
            assert!(
                text.contains("Ctrl-C / Ctrl-D  stop and exit"),
                "the key row arrives as a bold run followed by a dimmed tail, which is the shape \
                 most likely to be misplaced by a parser that confuses styling with text:\n{text}"
            );
            assert!(
                text.contains("/T/.tmppRB2bi/sessions"),
                "the footer is addressed at row 71, forty-one rows below the grid the driver asked \
                 for. Silently dropping it is what turned a working recorder into a failing proof \
                 on first contact with hardware -- and it fails as a false claim about `record`, \
                 not as a broken model:\n{text}"
            );
        }

        #[test]
        fn plain_text_lands_in_the_grid_and_walks_the_cursor() {
            let screen = painted(&[b"watching for the next call"]);
            assert_eq!(row(&screen, 0), "watching for the next call");
        }

        /// Every byte must move the reader forward, whatever it is.
        ///
        /// A parser that consumes nothing for some byte spins inside `absorb` forever, and no phase
        /// budget can save it: the deadline is checked in the caller, and the caller is never reached.
        /// Measured once already -- a build of this file burned an hour at full tilt on a granted Mac
        /// without ever reaching an assertion. So this feeds a stream with a piece of everything the
        /// frame and its neighbours emit, and demands it come back.
        #[test]
        fn every_byte_of_a_mixed_stream_is_consumed() {
            let stream: Vec<u8> = [
                b"\x1b[?1049h\x1b[?25l".to_vec(),
                b"\x1b[1;1H\x1b[38;5;39mmeethook \xe2\x97\x8f recording 42s".to_vec(),
                b"\r\n\x1b[2;1Hexpires in 12m\t\x08\x1b[K".to_vec(),
                b"\x1b]0;meethook\x07\x1b]0;other\x1b\\".to_vec(),
                b"\x1b[3;5Hsession abcd1234\x1b[s\x1b[5;9Hnote\x1b[u".to_vec(),
                b"\x1b[?25h\x1b[?1049l".to_vec(),
                b"bye\n".to_vec(),
            ]
            .concat();
            let (tx, rx) = mpsc::channel();
            std::thread::spawn(move || {
                let mut screen = Screen::new(30, 100);
                screen.absorb(&stream);
                let _ = tx.send(screen.text());
            });
            let text = rx.recv_timeout(Duration::from_secs(10)).expect(
                "absorb returned; a parser that stops advancing shows up here instead of hanging",
            );
            assert!(text.contains("bye"), "the stream ended unparsed:\n{text}");
        }

        #[test]
        fn cursor_addressing_writes_at_the_named_cell() {
            let screen = painted(&[b"abcdef", b"\x1b[3;7Hxy"]);
            assert_eq!(row(&screen, 0), "abcdef");
            assert_eq!(row(&screen, 2), "      xy");
        }

        #[test]
        fn relative_moves_walk_the_cursor_within_the_grid() {
            let screen = painted(&[
                b"\x1b[5;10H",
                b"A",
                b"\x1b[1G",
                b"B",
                b"\x1b[3d",
                b"C",
                b"\x1b[2A",
                b"D",
                b"\x1b[C",
                b"E",
            ]);
            assert_eq!(row(&screen, 4), "B        A");
            assert_eq!(row(&screen, 2), " C");
            assert_eq!(row(&screen, 0), "  D E");
        }

        #[test]
        fn erase_line_clears_after_before_or_around_the_cursor() {
            let after = painted(&[b"abcdef", b"\x1b[1;4H", b"\x1b[K"]);
            assert_eq!(row(&after, 0), "abc");
            let before = painted(&[b"abcdef", b"\x1b[1;4H", b"\x1b[1K"]);
            assert_eq!(row(&before, 0), "   def");
            let whole = painted(&[b"abcdef", b"\x1b[1;4H", b"\x1b[2K"]);
            assert_eq!(row(&whole, 0), "");
        }

        #[test]
        fn erase_screen_blanks_every_row_and_homes_the_cursor() {
            let screen = painted(&[b"one\r\ntwo\r\nthree", b"\x1b[2J", b"x"]);
            assert_eq!(row(&screen, 0), "x");
            assert_eq!(row(&screen, 1), "");
            assert_eq!(row(&screen, 2), "");
        }

        #[test]
        fn styling_never_changes_a_cell() {
            let styled = painted(&[
                b"\x1b[1;1H\x1b[38;5;39m\x1b[1mrecording",
                b"\x1b[1;1H\x1b[38;5;214m\x1b[7mwatching!\x1b[27m\x1b[39m",
            ]);
            assert_eq!(
                row(&styled, 0),
                "watching!",
                "a redraw in different colours occupies exactly the same cells -- if SGR bytes moved \
                 the cursor, every needle here would be reading shifted text"
            );
        }

        #[test]
        fn the_alternate_screen_switch_is_the_reset_point_and_other_modes_are_not() {
            let alt = painted(&[b"first", b"\x1b[?1049h", b"second"]);
            assert!(alt.contains("second"));
            assert!(
                !alt.contains("first"),
                "taking the alternate screen promises a clean slate; a retained `first` would make \
                 an absence assertion about the pre-frame output sound like the frame itself"
            );
            let cursor_mode = painted(&[b"first", b"\x1b[?25l", b" second"]);
            assert_eq!(
                row(&cursor_mode, 0),
                "first second",
                "hiding the cursor draws over the same grid -- clearing here would erase the very \
                 glyphs the proof asserts on"
            );
        }

        #[test]
        fn osc_sequences_are_skipped_whether_terminated_by_bel_or_string_terminator() {
            let bel = painted(&[b"\x1b]0;meethook\x07", b"title"]);
            assert_eq!(row(&bel, 0), "title");
            let st = painted(&[b"\x1b]2;wrapped\x1b\\", b"grid"]);
            assert_eq!(row(&st, 0), "grid");
        }

        #[test]
        fn an_unknown_escape_is_consumed_rather_than_misparsed() {
            let screen = painted(&[b"\x1b[?42Z", b"\x1b=", b"after"]);
            assert_eq!(
                row(&screen, 0),
                "after",
                "an unrecognised control must not be licence to misread the bytes after it"
            );
        }

        #[test]
        fn a_character_split_across_reads_reassembles() {
            let screen = painted(&[b"\x1b[1;1H", b"\xe2", b"\x97", b"\x8f", b" recording"]);
            assert_eq!(
                row(&screen, 0),
                "● recording",
                "the header's dot arrives as three separate reads; a parser that dropped the \
                 fragments would lose the one needle that says the frame is up"
            );
        }

        #[test]
        fn an_escape_interrupts_an_unfinished_character() {
            let screen = painted(&[b"\xe2\x97", b"\x1b[1;1H", b"x"]);
            assert_eq!(row(&screen, 0), "x");
            assert!(
                !screen.contains("●"),
                "the half character was never going to complete, so it must not be guessed"
            );
        }

        #[test]
        fn a_write_outside_the_requested_grid_is_kept_not_dropped() {
            // Measured against hardware: a frame that addresses row 71 while the pty was asked for
            // thirty rows is not broken, and neither is a test that watches it. What must not
            // happen is the grid deciding, silently, that a line nobody asked it to model was
            // therefore never painted.
            let mut screen = Screen::new(2, 5);
            screen.absorb(b"\x1b[9;9Hgone");
            assert!(
                screen.contains("gone"),
                "a needle the frame addressed outside the requested rectangle must still be \\n                 findable, not discarded for being unexpected"
            );
            assert_eq!(
                row(&screen, 8).trim_start(),
                "gone",
                "and it belongs at the row and column the frame named, not wherever the model \\n                 felt like putting it"
            );
            screen.absorb(b"\x1b[1;1Hok");
            assert_eq!(row(&screen, 0), "ok", "and the grid still works afterwards");
        }

        #[test]
        fn carriage_return_newline_backspace_and_tab_move_the_cursor() {
            let screen = painted(&[
                b"abcdef",
                b"\r",
                b"XY",
                b"\x1b[2;1HZ",
                b"\x08",
                b"Y",
                b"\t",
                b"T",
            ]);
            assert_eq!(row(&screen, 0), "XYcdef");
            assert_eq!(row(&screen, 1), "Y       T");
        }

        #[test]
        fn saved_cursor_returns_where_the_frame_put_it_down() {
            let screen = painted(&[b"\x1b[4;3H", b"\x1b[s", b"here", b"\x1b[u", b"back"]);
            assert_eq!(row(&screen, 3), "  back");
        }

        /// The property every absence assertion in the proof rests on: cells are written *at* the
        /// cursor and overwrite, so a glyph nobody redraws survives, and a glyph someone does gets
        /// replaced. Both halves matter -- the first is why `stop and exit` can still be on the grid
        /// after a return to idle, the second is why a stale word cannot hide the keypress.
        #[test]
        fn a_retained_glyph_survives_a_repaint_and_a_redrawn_one_is_replaced() {
            let mut screen = Screen::new(3, 20);
            screen.absorb(b"\x1b[1;1H\x1b[38;5;39mrecording");
            assert!(screen.contains("recording"));
            screen.absorb(b"\x1b[?25l\x1b[39m\x1b[?25h");
            assert!(
                screen.contains("recording"),
                "a draw that changes no cell emits styling and cursor hiding only; the word is \
                 still on screen and the grid must say so"
            );
            screen.absorb(b"\x1b[1;1Hwatching ");
            assert!(screen.contains("watching"));
            assert!(
                !screen.contains("recording"),
                "redrawn cells replace what was there, so a hint that really disappeared cannot \
                 still be read off the grid"
            );
        }
    }
}

use screen::Screen;

// ---------------------------------------------------------------------------------------------
// Watching the frame
// ---------------------------------------------------------------------------------------------

/// The child, its byte stream, and the grid that stream has painted -- with a mark, so a phase can
/// ask what the frame said *after* it started watching rather than what it has ever said.
///
/// The mark is the whole reason ordering assertions are possible: `finalizing` and then `watching`
/// is a claim about two successive grids, and a grid rebuilt from byte 0 can only answer "does it
/// say watching now", which a stale `watching` from before the keypress would satisfy.
struct Frame<'a> {
    driver: &'a mut Driver,
    screen: Screen,
    mark: usize,
}

impl Frame<'_> {
    /// Drains the pty and feeds whatever is new through the emulator.
    fn paint(&mut self) {
        self.driver.pump();
        if self.driver.out.len() > self.mark {
            self.screen.absorb(&self.driver.out[self.mark..]);
            self.mark = self.driver.out.len();
        }
    }

    /// Whether the run is still alive.
    fn alive(&mut self) -> bool {
        match self.driver.child.try_wait() {
            Ok(None) => true,
            Ok(Some(_)) => false,
            Err(e) => panic!("waiting on the child: {e}"),
        }
    }

    /// The pid the lock should be naming. Reached through the frame because the frame holds the
    /// only borrow of the driver for as long as it exists.
    fn pid(&self) -> u32 {
        self.driver.child.id()
    }

    /// Hands the child back to the pty driver's own bounded reap.
    fn wait_exit(&mut self, timeout: Duration) -> ExitStatus {
        self.driver.wait_exit(timeout)
    }

    /// Types bytes without waiting: the keystrokes under test must land the instant the test
    /// decides on them, and every wait after them is a separate, named observation.
    fn type_now(&mut self, bytes: &[u8]) {
        self.driver.write_now(bytes);
    }

    /// Wakes the frame thread so that a transient phase is actually painted.
    ///
    /// The frame blocks in `poll(TICK)` with TICK = 250 ms and drains its note channel *after* the
    /// poll returns -- so a note the run sends while that poll is asleep is drawn only when
    /// something wakes it. Left alone, a finalize that completes inside one tick gets both the
    /// `Stopping` and the `Recorded` note applied in a single drain, and `finalizing the session`
    /// is never painted even though the code said it correctly. An arrow key costs the run nothing:
    /// it reaches `Action::Next`/`Previous`, which move a cursor in the frame's own state machine
    /// and send no event at all. `s` would end a session and Ctrl-C would end the run, which is
    /// precisely what must not happen while measuring the frame.
    ///
    /// Poking only helps if a poke lands *between* the two notes, so the rate has to sit well under
    /// however long `finish` takes -- and that duration is two WAV finalizations, their fsyncs and
    /// an atomic metadata write, which no test may assume anything about. At tens of milliseconds
    /// it is short enough that a twenty-millisecond poke misses it outright, so this pokes every
    /// few milliseconds instead and leaves the assumption on the table.
    fn wake_frame(&mut self) {
        self.driver.write_now(b"\x1b[B");
    }

    /// Everything known about the world, for a panic message.
    fn evidence(&mut self) -> String {
        self.paint();
        format!(
            "grid:\n{}\n\nlast {} bytes from the pty:\n{}",
            self.screen.text(),
            2000,
            self.driver.tail()
        )
    }
}

/// A failing run happens on somebody else's machine, and the root it runs in is a `TempDir` whose
/// own `Drop` deletes it. Everything that could explain the failure has to reach somewhere durable
/// before the unwind finishes, or it goes away with the run -- which is how the first hardware
/// attempt ended: a hang of seventy-seven minutes whose entire frame stream was gone by the time
/// anyone looked for it.
///
/// The raw stream is the important half of the bargain. Fed back through [`Screen`] it reproduces
/// an emulator bug exactly, on a machine with no microphone and no grants, and that is the only way
/// a defect reachable only by real frame bytes can be fixed by someone who was not there when it
/// happened. The grid is written beside it because the assertion quotes the grid, so the pair
/// answers both halves of the question: what the test believed it saw, and what got painted.
///
/// Written only while panicking, since a passing run has nothing to say. Fixed names under the
/// system temp directory, overwritten each time, so whatever is there is the last failure.
impl Drop for Frame<'_> {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            return;
        }
        self.paint();
        let dir = std::env::temp_dir();
        let raw = dir.join("meethook-live-proof.bin");
        let grid = dir.join("meethook-live-proof.txt");
        // Both writes are best effort. This runs during an unwind that is already carrying the real
        // diagnosis, so a failure to park a copy of it must neither replace that diagnosis nor
        // raise a second panic from inside the first.
        let raw_written = std::fs::write(&raw, &self.driver.out).is_ok();
        let report = format!(
            "final grid after {} bytes from the pty\n\ngrid:\n{}\n\ntwo dumps of the same thing, for a failure on a \nmachine nobody else can reach: the grid the test believed, and every byte the pty \nsent. Replay the second through `Screen::absorb` to reproduce this offline. Raw copy: {}\n",
            self.driver.out.len(),
            self.screen.text(),
            if raw_written {
                raw.display().to_string()
            } else {
                "(could not be written)".to_string()
            },
        );
        let grid_written = std::fs::write(&grid, &report).is_ok();
        eprintln!(
            "\nevidence: {}",
            match (grid_written, raw_written) {
                (true, true) => format!("{} and {}", grid.display(), raw.display()),
                (true, false) => grid.display().to_string(),
                _ => "could not be written; the panic message above is all there is".to_string(),
            }
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The stand-in holder, and reading the disk
// ---------------------------------------------------------------------------------------------

/// Where the built `mic-hold` example lives.
///
/// Located, never built: no cargo is spawned from a test, the holder is not duplicated into this
/// crate, and the two workspaces stay uncoupled -- `crates/meethook-record` roots its own workspace
/// precisely so nothing in this build graph asks a non-Apple toolchain to compile it. Release
/// profiles are not searched: this proof is run by a person at a keyboard, who builds the way every
/// other gate here does.
fn holder() -> PathBuf {
    let candidates = [
        std::env::var_os("CARGO_TARGET_DIR").map(|dir| {
            PathBuf::from(dir)
                .join("debug")
                .join("examples")
                .join("mic-hold")
        }),
        Some(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(Path::parent)
                .map(|repo| {
                    repo.join("crates")
                        .join("meethook-record")
                        .join("target")
                        .join("debug")
                        .join("examples")
                        .join("mic-hold")
                }),
        )
        .flatten(),
    ];
    candidates
        .into_iter()
        .flatten()
        .find(|path| path.is_file())
        .unwrap_or_else(|| {
            panic!(
                "the holder example is not built. Run: cargo build --manifest-path \
                 crates/meethook-record/Cargo.toml --example mic-hold"
            )
        })
}

/// Spawns a holder and waits for the line it prints once the engine is up -- the sync point, so
/// the test never sleeps hoping a device opened.
///
/// Its early exit is read as what it means: `mic-hold` fails non-zero only when the default input
/// device reports an unusable format or the engine would not start, i.e. "this machine has no
/// usable input device". It is *not* a grant probe -- without a grant `AVAudioEngine` delivers
/// silence and the example succeeds -- so nothing here claims otherwise.
fn spawn_holder(seconds: u64) -> Holder {
    let exe = holder();
    let mut child = Command::new(&exe)
        .arg(seconds.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("could not run {}: {e}", exe.display()));

    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");
    let (tx, rx) = mpsc::channel::<Result<String, String>>();
    thread::spawn(move || {
        let mut line = String::new();
        match BufReader::new(stdout).read_line(&mut line) {
            Ok(1..) => {
                let _ = tx.send(Ok(line));
            }
            _ => {
                // EOF before the line: the process died, and its stderr says why.
                let mut why = String::new();
                let mut err = BufReader::new(stderr);
                let _ = err.read_to_string(&mut why);
                let _ = tx.send(Err(if why.is_empty() {
                    "(no stderr)".to_string()
                } else {
                    why
                }));
            }
        }
    });

    let opened = match rx.recv_timeout(HOLDER_START_BUDGET) {
        Ok(Ok(line)) => line,
        Ok(Err(why)) => {
            let status = child
                .try_wait()
                .ok()
                .flatten()
                .map(|s| s.to_string())
                .unwrap_or_else(|| "still running".to_string());
            child.kill().ok();
            child.wait().ok();
            panic!(
                "{exe:?} exited before it opened the device (status {status}, stderr {why}). \
                 That exit is the machine saying it has no usable default input device -- check \
                 System Settings > Sound > Input. It does not indicate a missing permission: \
                 without the microphone grant this example succeeds and captures silence."
            );
        }
        Err(_) => {
            child.kill().ok();
            child.wait().ok();
            panic!(
                "{exe:?} printed nothing within {HOLDER_START_BUDGET:?}; killed. Either the \
                 audio stack is wedged on this machine or the process blocked opening the device."
            );
        }
    };

    assert!(
        opened.starts_with("pid "),
        "{exe:?} printed something unexpected as its first line: {opened:?}"
    );
    println!("  holder: {}", opened.trim_end());
    Holder {
        child,
        line: opened.trim_end().to_string(),
    }
}

/// A holder, and the promise that it stops holding when this test is done with it -- including on
/// the panic paths, where no explicit `kill` line is ever reached.
///
/// The promise is not tidiness. A `mic-hold` left alive keeps the microphone claimed for the rest
/// of its 180 s, and the activity predicate cannot tell it apart from the holder the next run is
/// waiting for: the phantom-holder failure the re-arm phase exists to catch would then be caused
/// by the previous run rather than found by it. Owning the reap -- exactly as `common::Driver` does
/// for the child -- is what keeps one failed proof from poisoning the next.
struct Holder {
    child: Child,
    /// The line printed when the engine came up: pid, sample rate, channel count. Kept because a
    /// failure that names the device actually held is diagnosable and one that names only a pid is
    /// not.
    line: String,
}

impl Holder {
    fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Live or dead, phrased for a failure message. Best-effort deliberately: a diagnostic that
    /// itself panics replaces the finding it was sent to deliver with something unrelated.
    fn describe(&mut self) -> String {
        match self.child.try_wait() {
            Ok(None) => format!("pid {} capturing", self.pid()),
            Ok(Some(status)) => format!("{} exited ({status})", self.line),
            Err(e) => format!("pid {} unqueryable ({e})", self.pid()),
        }
    }
}

impl Drop for Holder {
    fn drop(&mut self) {
        // An already-reaped child answers `try_wait` from its cached status, so a pid that may by
        // now belong to somebody else is never signalled again.
        if !matches!(self.child.try_wait(), Ok(None)) {
            return;
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Every session directory under `root`, read through the crate that owns the layout -- the same
/// eyes `meethook sessions` uses, so this test cannot develop a private opinion about what a
/// half-finished session means.
fn sessions(paths: &Paths) -> Vec<DiscoveredSession> {
    discover_sessions(paths).expect("reading the sessions directory")
}

/// Both tracks of one session, as the files themselves describe them. The session carries its own
/// paths, so the root is not needed here -- and asking for it would invite reading the tracks of
/// some other session than the one under discussion.
fn tracks(found: &DiscoveredSession) -> (TrackEvidence, TrackEvidence) {
    (
        wav::track(&found.paths.mic_wav()),
        wav::track(&found.paths.speaker_wav()),
    )
}

/// Whether a track file proves capture is happening *right now*.
///
/// Mid-session a track legitimately declares fewer bytes than it holds (`BeyondDeclaration`, the
/// shape of a writer that has not finalized yet) and may disagree with itself in the other
/// direction; what it must not be is absent, not-a-WAV, unreadable, or a header that has never had
/// a sample written under it.
///
/// `NoDeclaredLength` is deliberately absent from that list for the same reason `BeyondDeclaration`
/// is: a file in that state holds audio, which is precisely what this predicate asks. Nothing the
/// recorder writes can produce it either -- `track.rs` checkpoints a real length -- so listing it
/// here would assert something untrue about a file to guard against a shape that cannot occur.
fn capturing(evidence: TrackEvidence) -> bool {
    !matches!(
        evidence,
        TrackEvidence::Absent
            | TrackEvidence::NotAWav
            | TrackEvidence::Unreadable
            | TrackEvidence::HeaderOnly
    )
}

fn mic_bytes(found: &DiscoveredSession) -> u64 {
    std::fs::metadata(found.paths.mic_wav())
        .map(|m| m.len())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------------------------
// The proof
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "drives a real microphone, a real pty and the real frame: needs macOS 14.2+ (below \
            that floor CoreAudio's per-process input objects do not exist, so the trigger can \
            only ever report nothing), a terminal application granted both Microphone and Screen \
            & System Audio Recording, a working default input device, nothing else capturing on \
            this Mac, and the holder example built. Run it with: cargo nextest run -p meethook \
            --run-ignored only -E 'binary(live_record_hand_stop)' --nocapture"]
fn the_real_frame_ends_a_live_session_on_s_and_keeps_watching() {
    // Declared first so it outlives the driver: the root is deleted when this guard drops, and a
    // child still writing into it would be a different bug report.
    let dir = tempfile::tempdir().expect("a temporary root");
    let root = dir.path();
    let paths = Paths::new(root.to_path_buf());

    let run_started = Instant::now();
    let mut driver = Driver::spawn(root, &["record"]);

    // --- Phase 1: up, and idle ---------------------------------------------------------------
    //
    // `false` means the child exited before taking the terminal, which for this command means it
    // refused: preflight names the missing grant(s) and explains that macOS charges the grant to
    // the terminal app. A denial answers in well under a second (measured 0.74 s), because only a
    // never-asked `NotDetermined` status waits out the 120 s prompt timeout.
    if !driver.wait_for_frame_or_exit(FRAME_UP_BUDGET) {
        panic!(
            "`record` exited before painting a frame. Its own words, verbatim:\n{}\n\
             If that names a permission, grant it to the terminal application this test was \
             launched from -- macOS attributes the grant to the responsible app, which children \
             inherit, and there will never be a \"meethook\" entry to grant.",
            driver.tail()
        );
    }

    let mut frame = Frame {
        driver: &mut driver,
        screen: Screen::new(PTY_ROWS as usize, PTY_COLS as usize),
        mark: 0,
    };
    // The first draw races the alt-screen escape the wait returns on, so watch for it rather than
    // assuming it already arrived.
    let up = Instant::now();
    while !frame.screen.contains("watching for the next call") {
        frame.paint();
        assert!(
            frame.alive(),
            "the run died after taking the terminal.\n{}",
            frame.evidence()
        );
        assert!(
            up.elapsed() < SETTLE * 10,
            "the frame took the terminal but never painted its idle status.\n{}",
            frame.evidence()
        );
        std::thread::sleep(Duration::from_millis(20));
    }

    // The footer paints this root's sessions directory, which is what ties the pixels to *this*
    // run rather than to some other `record` on the machine.
    assert!(
        frame.screen.contains(&root.display().to_string()),
        "the idle frame does not advertise this run's root.\n{}",
        frame.evidence()
    );
    assert!(
        !frame.screen.contains("end this session"),
        "an idle frame advertises a key that ends a session there is no session to end.\n{}",
        frame.evidence()
    );

    // --- Phase 2: a foreign process opens the device, and a session opens --------------------
    let mut holder_one = spawn_holder(HOLDER_SECONDS);
    let started = Instant::now();
    let found = loop {
        let all = sessions(&paths);
        if all.len() > 1 {
            panic!(
                "one holder opened {} session directories; the trigger opens at most one session \
                 per rising edge.\n{}",
                all.len(),
                frame.evidence()
            );
        }
        if all.len() == 1 {
            break all.into_iter().next().unwrap();
        }
        assert!(
            frame.alive(),
            "the run died while a process held the microphone.\n{}",
            frame.evidence()
        );
        assert!(
            started.elapsed() < SESSION_START_BUDGET,
            "no session directory appeared within {SESSION_START_BUDGET:?} of a process opening \
             the default input device. The trigger counts *capturing processes*, not loudness, so \
             this is not a quiet-room problem: either the watcher missed the rising edge or it is \
             counting a different process. Read the walker with the manual recipe in the module \
             doc.\nholder: {}\n{}",
            holder_one.describe(),
            frame.evidence()
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(
        found.classification,
        Classification::Orphaned,
        "a session that is being recorded right now has no session.json yet, which the crate \
         that owns the layout classifies as Orphaned -- a normal state, never an error.\n{}",
        frame.evidence()
    );
    let first_id = found.id.to_string();

    // The frame follows the disk within one tick while it is recording, since it redraws every
    // iteration in that phase.
    let shown = Instant::now();
    while !(frame.screen.contains("● recording")
        && frame.screen.contains(&first_id)
        && frame.screen.contains(" end this session"))
    {
        frame.paint();
        assert!(
            frame.alive(),
            "the run died just after opening a session.\n{}",
            frame.evidence()
        );
        assert!(
            shown.elapsed() < SETTLE * 20,
            "the frame never showed the live session it had already opened ({first_id}). If it \
             got as far as `opening a session` and fell back to `watching`, the trigger fired but \
             `begin` abandoned after its five retries -- a capture failure, not a detection one.\n{}",
            frame.evidence()
        );
        std::thread::sleep(Duration::from_millis(20));
    }

    phase(
        run_started,
        "up: idle frame painted, session opened by a foreign holder",
    );

    // --- Phase 3: capture was genuinely running before any keypress --------------------------
    //
    // Files existing proves a directory was created; files growing proves samples were arriving.
    // A silent-but-open device still delivers callbacks, so growth is expected even though the
    // holder records nothing audible -- silence is not a failure mode here, which is the whole
    // point of counting processes rather than levels.
    let before_growth = mic_bytes(&found);
    std::thread::sleep(GROWTH_WAIT);
    let after_growth = mic_bytes(&found);
    assert!(
        after_growth > before_growth,
        "mic.wav did not grow across {GROWTH_WAIT:?}, which is longer than the 5 s checkpoint \
         interval: nothing was being captured, so ending this session would prove nothing about \
         ending a call.\n{}",
        frame.evidence()
    );
    let (mic_before_stop, speaker_before_stop) = tracks(&found);
    assert!(
        capturing(mic_before_stop) && capturing(speaker_before_stop),
        "a live session's tracks must prove they are being written: mic {mic_before_stop:?}, \
         speaker {speaker_before_stop:?}.\n{}",
        frame.evidence()
    );

    // --- Phase 4: `s`, with the holder still holding -------------------------------------------
    //
    // The ordering is what makes this discriminating. Pressing `s` and *then* releasing the holder
    // would prove nothing, because the release is itself the falling edge that ends a session.
    // While the holder holds, the predicate is unchanged and the natural route to a finalize is
    // unavailable: the only thing that can produce `session.json` in the next seconds is this
    // keypress reaching the loop's stop arm and running the one finalize point.
    frame.type_now(b"s");
    let pressing = Instant::now();
    // Each sighting is recorded with the moment it first happened, so "finalizing and then
    // watching" is settled by comparing two timestamps rather than by the order a pair of flags
    // happened to be set -- and so a run that went straight from recording to watching reaches the
    // assertion below, which names exactly that, instead of spinning to the budget expiry, which
    // would blame the keypress for a narration fault.
    let mut saw_finalizing: Option<Duration> = None;
    let mut saw_watching: Option<Duration> = None;
    let mut finalized: Option<Duration> = None;
    while !(saw_watching.is_some() && finalized.is_some()) {
        frame.wake_frame();
        std::thread::sleep(WAKE_EVERY);
        frame.paint();
        if saw_finalizing.is_none() && frame.screen.contains("finalizing the session") {
            saw_finalizing = Some(pressing.elapsed());
        }
        if saw_watching.is_none() && frame.screen.contains("watching for the next call") {
            saw_watching = Some(pressing.elapsed());
        }
        if finalized.is_none() && found.paths.session_json().exists() {
            finalized = Some(pressing.elapsed());
        }
        assert!(
            frame.alive(),
            "the run exited on `s` instead of ending the session and staying.\n{}",
            frame.evidence()
        );
        assert!(
            pressing.elapsed() < FINALIZE_BUDGET,
            "within {FINALIZE_BUDGET:?} of `s` the run had said: finalizing {saw_finalizing:?}, \
             back to watching {saw_watching:?}; session.json written {finalized:?}. The hand stop \
             takes the immediate break to the single finalize point, so the budget is already \
             twice the worst-case *natural* stop latency.\n{}",
            frame.evidence()
        );
    }
    assert!(
        saw_finalizing.is_some(),
        "the frame went from recording to watching without ever painting `finalizing the \
         session`, even though it was woken every {WAKE_EVERY:?} between the keypress and the \
         finalize. Said before the finish line is part of the contract.\n{}",
        frame.evidence()
    );
    assert!(
        saw_finalizing <= saw_watching,
        "the frame painted `watching for the next call` ({saw_watching:?}) before it painted \
         `finalizing the session` ({saw_finalizing:?}): the one finalize point narrated out of \
         order."
    );

    // Same facts the crate's own report would print, read through the same functions it uses.
    let stopped = sessions(&paths)
        .into_iter()
        .find(|s| s.id.to_string() == first_id)
        .expect("the session this test watched appear");
    assert_eq!(
        stopped.classification,
        Classification::Valid,
        "`s` left a session directory with no usable session.json behind.\n{}",
        frame.evidence()
    );
    stopped
        .load_metadata()
        .expect("a Valid session's metadata reads");
    let (mic, speaker) = tracks(&stopped);
    assert!(
        mic == TrackEvidence::CompleteAsDeclared && speaker == TrackEvidence::CompleteAsDeclared,
        "the finalize point did not close both tracks: mic {mic:?}, speaker {speaker:?}.\n{}",
        frame.evidence()
    );

    phase(
        run_started,
        "hand stop: session.json written, both tracks closed, frame back to watching",
    );

    // --- Phase 5: still watching, asserted four ways ------------------------------------------
    //
    // The frame alone is not a liveness oracle: an idle frame is exactly what a run about to exit
    // looks like for its last few hundred milliseconds.
    let pid = frame.pid();
    let window = Instant::now();
    while window.elapsed() < NO_RESTART_WINDOW {
        assert!(
            frame.alive(),
            "the run died after the hand stop.\n{}",
            frame.evidence()
        );
        match record_lock_state(&paths) {
            LockState::Held(holder) => assert_eq!(
                holder.pid,
                Some(pid),
                "the record lock names someone other than this run while it is supposed to be \
                 watching: {holder:?}"
            ),
            other => panic!(
                "the run gave up its record lock while idling after the hand stop: {other:?}.\n{}",
                frame.evidence()
            ),
        }
        assert!(
            sessions(&paths).len() == 1,
            "a second session opened while the counted process still held the microphone. The \
             hand stop is not the device-change restart: it goes back to watching and waits for a \
             rising edge, so this is the run re-opening on a level that never changed.\n{}",
            frame.evidence()
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    // Painted first: the window above asked the disk and the process table rather than the
    // stream, so the grid still holds whatever the last pump left on it.
    frame.paint();
    assert!(
        frame.screen.contains("watching for the next call")
            && !frame.screen.contains(" end this session"),
        "after the hand stop the frame is not back to an idle offer.\n{}",
        frame.evidence()
    );

    // A stray `s` while watching: dropped by the idle arm, and refused by the frame's own gate.
    // Byte growth is the right instrument for "something was repainted" -- the grid is the right
    // instrument for *what* was painted -- and an idle frame has nothing else left to say, so fresh
    // bytes can only be this keystroke's redraw.
    //
    // Measured from a drained baseline. The liveness window above asked the disk and the process
    // table rather than the stream, so whatever those seconds left in the pty's buffer would
    // otherwise count as growth and the assertion would pass on bytes the keystroke never caused.
    std::thread::sleep(SETTLE);
    frame.paint();
    std::thread::sleep(SETTLE);
    frame.paint();
    let quiet_at = frame.driver.out.len();
    frame.type_now(b"s");
    std::thread::sleep(SETTLE);
    frame.paint();
    assert!(
        frame.driver.out.len() > quiet_at,
        "a stray `s` while watching produced no repaint at all ({} bytes before and after): the \
         frame acknowledged the key not at all, which is not the same thing as dropping an event \
         it cannot act on.\n{}",
        quiet_at,
        frame.evidence()
    );
    assert!(
        sessions(&paths).len() == 1,
        "a stray `s` while watching opened a session.\n{}",
        frame.evidence()
    );

    phase(
        run_started,
        "still watching: alive, lock held, no second session, stray `s` dropped",
    );

    // --- Phase 6: re-armed, and a second session with no restart ------------------------------
    //
    // Process death tears down the holder's CoreAudio IO, which is the release the predicate sees.
    // Waiting past the re-check cadence before holding again distinguishes "the release was seen"
    // from "the watcher never noticed anything and this second session is a coincidence".
    let first_holder = holder_one.line.clone();
    drop(holder_one);
    std::thread::sleep(REARM_WAIT);
    assert!(
        sessions(&paths).len() == 1,
        "releasing the holder made the run reopen a session with nothing left to record.\n{}",
        frame.evidence()
    );

    let mut holder_two = spawn_holder(HOLDER_SECONDS);
    let rearming = Instant::now();
    let second = loop {
        let all = sessions(&paths);
        if all.len() == 2 {
            break all.into_iter().last().expect("two sessions");
        }
        assert!(
            frame.alive(),
            "the run died between the two sessions.\n{}",
            frame.evidence()
        );
        assert!(
            rearming.elapsed() < SECOND_SESSION_BUDGET,
            "holding the microphone again opened no second session within \
             {SECOND_SESSION_BUDGET:?}. This is the phase a phantom holder breaks: some other \
             process keeping the predicate true after the first holder died leaves no rising edge \
             to open one for. The manual recipe in the module doc names every process the walker \
             counts.\nfirst holder: {first_holder}\nsecond holder: {}\n{}",
            holder_two.describe(),
            frame.evidence()
        );
        std::thread::sleep(Duration::from_millis(100));
    };
    assert_eq!(
        second.classification,
        Classification::Orphaned,
        "the second session is live, so it has no session.json yet.\n{}",
        frame.evidence()
    );
    let shown = Instant::now();
    while !(frame.screen.contains("● recording") && frame.screen.contains(&second.id.to_string()))
    {
        frame.paint();
        assert!(
            shown.elapsed() < SETTLE * 20,
            "the run opened a second session ({}) without showing it.\n{}",
            second.id,
            frame.evidence()
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(
        frame.pid(),
        pid,
        "the process this test is driving is not the one holding the lock"
    );

    phase(
        run_started,
        "re-armed: a second session opened with no restart of the process",
    );

    // --- Phase 7: Ctrl-C still means leave ---------------------------------------------------
    //
    // In raw mode the byte is a keystroke the frame turns into the same event a SIGINT would
    // raise, and it takes the same single finalize point as a call that ended on its own. The
    // driver never gives the child a controlling terminal, so do not expect the line discipline
    // to raise a signal on its own.
    frame.type_now(b"\x03");
    let status = frame.wait_exit(EXIT_BUDGET);
    drop(holder_two);
    assert!(
        status.success(),
        "Ctrl-C left the run with {status}.\n{}",
        frame.evidence()
    );

    let all = sessions(&paths);
    assert_eq!(
        all.len(),
        2,
        "the run opened {} sessions; one holder-release pair should make exactly two",
        all.len()
    );
    let (mic, speaker) = tracks(&second);
    assert_eq!(
        all.last().expect("the newest session").classification,
        Classification::Valid,
        "the interrupt did not take the session through the same finalize point the hand stop \
         did.\n{}",
        frame.evidence()
    );
    assert!(
        mic == TrackEvidence::CompleteAsDeclared && speaker == TrackEvidence::CompleteAsDeclared,
        "the interrupted session's tracks were not closed: mic {mic:?}, speaker {speaker:?}.\n{}",
        frame.evidence()
    );
    assert_eq!(
        record_lock_state(&paths),
        LockState::Free,
        "the run exited still holding its record lock"
    );
    phase(
        run_started,
        "Ctrl-C exited clean: second session finalized, record lock free",
    );
}

/// What the record lock says about `root` right now.
///
/// Asked through the crate that owns the lock file rather than by reading `<root>/record.lock`:
/// presence is not liveness there, and this test's whole claim about "still watching" would be
/// resting on a different definition of the word than the recorder's own.
fn record_lock_state(paths: &Paths) -> LockState {
    meethook_session::RecordLock::probe(paths)
}

/// Prints a phase boundary with the elapsed wall time.
///
/// The proof is run by hand, in a terminal, and its output is pasted into a ticket as the evidence
/// that the hardware case ran. nextest swallows a passing test's stdout without `--nocapture`, so
/// these lines are also the reason that flag is part of the documented command.
fn phase(since: Instant, what: &str) {
    println!("  {:>6.1}s  {what}", since.elapsed().as_secs_f64());
}

// ---------------------------------------------------------------------------------------------
// Is the reconstructor faithful?
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod render_fidelity {
    //! The emulator checked against the thing it claims to reconstruct.
    //!
    //! `screen::tests` feed the parser hand-written bytes, which proves only that it does what the
    //! mind that wrote it intended: two defects lived in it anyway, because the hand-written stream
    //! is chosen by the same assumptions that shaped the parser. So this feeds it real output --
    //! ratatui drawing through the same `CrosstermBackend` the recorder uses -- and demands that the
    //! grid rebuilt from those bytes match, row for row, what the authoritative `TestBackend` says
    //! it drew.
    //!
    //! Twice, deliberately. The first draw writes every cell; the second changes some and leaves the
    //! rest, so its bytes carry no glyphs at all for most of the screen. That second pass is the
    //! case every absence assertion in the proof rests on, and the one an emulator gets wrong by
    //! either clearing too much or retaining too much.
    use super::Screen;
    use ratatui::backend::{CrosstermBackend, TestBackend};
    use ratatui::layout::Rect;
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};
    use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
    use ratatui::{Terminal, TerminalOptions, Viewport};
    use std::cell::RefCell;
    use std::io::Write;
    use std::rc::Rc;

    const ROWS: usize = 30;
    const COLS: usize = 100;

    /// A sink that keeps what the backend writes without owning it, so the bytes can be read while
    /// the terminal drawing them is still alive.
    #[derive(Default, Clone)]
    struct Sink(Rc<RefCell<Vec<u8>>>);

    impl Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.borrow_mut().extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Header, bordered body, footer hint: the shapes this frame really draws, so the stream under
    /// test contains border glyphs, styled spans, wrapping, cursor addressing and clears.
    fn frame_lines(recording: bool, elapsed: &str, hint: &str) -> Vec<Line<'static>> {
        let header = if recording {
            Line::from(vec![
                Span::styled("meethook ", Style::default().add_modifier(Modifier::BOLD)),
                Span::styled(
                    format!("● recording {elapsed}"),
                    Style::default().fg(Color::LightRed),
                ),
            ])
        } else {
            Line::from(Span::styled(
                "meethook watching for the next call",
                Style::default().fg(Color::DarkGray),
            ))
        };
        vec![
            header,
            Line::from(format!("session 20260815-101500 · expires in {elapsed}")),
            Line::from(""),
            Line::from(hint.to_string()),
        ]
    }

    fn widget(lines: Vec<Line<'static>>) -> Paragraph<'static> {
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title(" meeting "))
    }

    fn rows(buffer: &ratatui::buffer::Buffer) -> Vec<String> {
        (0..buffer.area().height)
            .map(|y| {
                (0..buffer.area().width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn the_grid_rebuilt_from_real_ratatui_bytes_matches_the_buffer_it_came_from() {
        let area = Rect::new(0, 0, COLS as u16, ROWS as u16);
        let mut reference = Terminal::new(TestBackend::new(COLS as u16, ROWS as u16))
            .expect("a test backend never fails to start");
        let sink = Sink::default();
        let mut drawn = Terminal::with_options(
            CrosstermBackend::new(sink.clone()),
            TerminalOptions {
                viewport: Viewport::Fixed(area),
            },
        )
        .expect("writing to a vector cannot fail to start");

        // Recording, then idle with a different hint: the transition the proof asserts on, whose
        // second frame reaches the stream almost empty.
        let variants = [
            frame_lines(true, "42s", "s end this session"),
            frame_lines(false, "9m", "Ctrl-C stop and exit"),
        ];

        let mut screen = Screen::new(ROWS, COLS);
        let mut seen = 0;
        let mut painted_grid = Vec::new();
        for (n, variant) in variants.iter().enumerate() {
            let lines = variant.clone();
            reference
                .draw(|f| f.render_widget(widget(lines.clone()), f.area()))
                .expect("drawing into a buffer cannot fail");
            drawn
                .draw(|f| f.render_widget(widget(lines), f.area()))
                .expect("drawing into a vector cannot fail");
            let bytes = sink.0.borrow();
            assert!(
                bytes.len() > seen,
                "draw {n} emitted no bytes at all, so this comparison would prove nothing"
            );
            screen.absorb(&bytes[seen..]);
            seen = bytes.len();

            let want = rows(reference.backend().buffer());
            let got = screen
                .text()
                .lines()
                .map(str::to_string)
                .collect::<Vec<_>>();
            for (row, (want, got)) in want.iter().zip(got.iter()).enumerate() {
                assert_eq!(
                    want, got,
                    "frame {n}, row {row}: the buffer ratatui drew and the grid rebuilt from its \
                     own bytes disagree, so a needle this test cannot see is a needle the proof \
                     cannot trust"
                );
            }
            assert_eq!(want.len(), got.len());
            painted_grid.push(screen.text());
        }

        // The needles the proof hunts, each read at the moment it should be readable. They are here
        // to say the exercise drew a frame worth comparing rather than an empty one -- and, below,
        // to say the second frame replaced the first where a cell was redrawn, which is the half of
        // the property that makes an absence assertion mean anything.
        let recording = &painted_grid[0];
        assert!(recording.contains("● recording"), "{recording}");
        assert!(recording.contains("end this session"));
        let idle = &painted_grid[1];
        assert!(idle.contains("watching for the next call"), "{idle}");
        assert!(idle.contains("stop and exit"));
        assert!(
            !idle.contains("end this session"),
            "the hint row was redrawn and still reads as the old hint, so the grid is retaining \\n             what the frame replaced"
        );
    }
}
