//! Writing to a terminal that may already be gone.
//!
//! Rust sets `SIGPIPE` to `SIG_IGN` before `main` runs, so writing to a descriptor whose
//! reader has closed comes back as an error rather than killing the process -- and `print!`
//! answers such an error by *panicking*. Closing the terminal window hangs up the pty, and a
//! hung-up pty answers every further write with `EIO` forever, so the failure is permanent
//! and not something worth retrying or branching on.
//!
//! That combination costs a recording. [`crate::RunningSession::finish`] prints its
//! diagnostics through [`crate::calendar`] and this module *before* it writes
//! `session.json`, so a print that panics lands between stopping the engines and finalizing
//! the session: both WAVs are on disk, nothing says what they are, and no later command will
//! claim them. A diagnostic nobody read is the cheaper loss, so the write failure is absorbed
//! here and the first refusal latches the stream shut -- everything after it is discarded
//! without another attempt at the descriptor.
//!
//! The bin crate makes the same decision in `crates/meethook/src/record.rs`'s `Narration`.
//! Two copies exist because these are two workspaces: this crate cannot depend on a binary,
//! and the shared crate underneath (`meethook-session`) is the on-disk contract -- every
//! path, file name, id format and metadata field -- so a printing policy is off-mission
//! there, and would drag terminal behaviour into the schema crate.
//!
//! Only the failure handling lives here. Destinations and bytes are unchanged from the
//! `eprintln!`/`println!` calls this replaced, and a guard test in this module's tests keeps
//! bare prints from coming back.

use std::fmt;
use std::io::{self, Write};

/// One stream of this crate's output, written to as though the terminal might already be gone.
///
/// Constructed per call site rather than stored: the activity log's writer takes `&self` and
/// runs on a private serial queue, so holding a `Box<dyn Write>` in it would mean interior
/// mutability and `Send` gymnastics for nothing -- building `io::stderr()` is free. The cost
/// of that choice is that after stderr dies, each later line still attempts exactly one write
/// before latching again, bounded by the few lines per second these paths can produce.
///
/// One instance per stream, never one process-wide flag: a diagnostic that could not reach
/// stderr is no reason to withhold the heads-up this crate prints on stdout.
pub(crate) struct Output {
    dest: Box<dyn Write>,
    /// Set by the first refusal; from then on [`Output::line`] discards silently.
    dead: bool,
}

impl Output {
    /// A stream over any writer.
    ///
    /// Production names its two streams below; this is the injection seam the tests use to
    /// hand an [`Output`] a writer that refuses. Un-gated rather than `#[cfg(test)]`-only
    /// because the two named constructors are built on it.
    pub(crate) fn to(dest: impl Write + 'static) -> Self {
        Output {
            dest: Box::new(dest),
            dead: false,
        }
    }

    /// The stream the user reads.
    pub(crate) fn stdout() -> Self {
        Output::to(io::stdout())
    }

    /// The stream the diagnostics and faults go to.
    pub(crate) fn stderr() -> Self {
        Output::to(io::stderr())
    }

    /// Print one line, as `eprintln!` would have, and never fail.
    ///
    /// Takes [`fmt::Arguments`] rather than a `&str` because these call sites carry inline
    /// format args (`{label:<8}`, `{:.3}`) whose alignment a human diffs against the previous
    /// run -- going through `str` would mean a `format!` allocation per line on paths that
    /// allocate nothing today.
    ///
    /// Every error kind is swallowed: a closed pipe, an `EIO`, a full disk all mean the same
    /// thing here, and std's Unix errno table has no arm for `EIO` anyway, so a hung-up pty
    /// arrives as `Uncategorized`, which std itself says not to match. Flushing is part of the
    /// same promise -- the calendar-access heads-up must leave the process *before* the system
    /// dialog covers it, and a run whose stdout is a pipe only drains at an exit that may
    /// never come -- and a failed flush latches too, since a stream that cannot drain its
    /// buffer will not drain the next line's either.
    pub(crate) fn line(&mut self, args: fmt::Arguments<'_>) {
        if self.dead {
            return;
        }
        if writeln!(self.dest, "{args}").is_err() || self.dest.flush().is_err() {
            self.dead = true;
        }
    }
}

/// The scripted streams this crate's tests drive an [`Output`] with.
///
/// Shared with `lib.rs`, whose test of `finalize` needs a terminal that refuses every byte.
/// Lives here rather than in a `test_support` module because these exist only to exercise this
/// one type; a second consumer would read them at their source anyway.
#[cfg(test)]
pub(crate) mod scripted {
    use std::io::{self, Write};
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// What a [`ScriptedStream`] did about the bytes it was handed.
    ///
    /// Shared so a test can read it back after the stream itself has moved into an
    /// [`super::super::Output`].
    #[derive(Clone, Default)]
    pub(crate) struct StreamLog {
        /// The bytes the stream accepted, in order.
        heard: Arc<Mutex<String>>,
        /// How many writes or flushes it refused.
        refusals: Arc<AtomicUsize>,
    }

    impl StreamLog {
        pub(crate) fn heard(&self) -> String {
            self.heard.lock().unwrap().clone()
        }

        pub(crate) fn refusals(&self) -> usize {
            self.refusals.load(Ordering::SeqCst)
        }
    }

    /// A stream whose willingness to accept bytes is scripted.
    ///
    /// This is the hung-up terminal: the one that refuses forever, the one that refuses once
    /// and then recovers (so the latch is visible from outside rather than assumed), and the
    /// healthy one that must hear everything, since a latch that fired early would look
    /// exactly like a passing suite.
    pub(crate) struct ScriptedStream {
        log: StreamLog,
        /// Built per refusal, because [`io::Error`] is not `Clone` and a stream refuses more
        /// than once.
        error: fn() -> io::Error,
        /// How many more writes to refuse before accepting again.
        refusals_left: usize,
        /// Whether the flush refuses even when the write is accepted.
        flush_refuses: bool,
    }

    impl ScriptedStream {
        /// Refuses the next `refusals` writes and then accepts and records, with the flush
        /// behaving as `flush_refuses` dictates.
        pub(crate) fn flaky(
            error: fn() -> io::Error,
            refusals: usize,
            flush_refuses: bool,
        ) -> (Self, StreamLog) {
            let log = StreamLog::default();
            (
                ScriptedStream {
                    log: log.clone(),
                    error,
                    refusals_left: refusals,
                    flush_refuses,
                },
                log,
            )
        }

        /// Refuses every write forever, the way a terminal whose master has closed does.
        pub(crate) fn hung_up(error: fn() -> io::Error) -> (Self, StreamLog) {
            Self::flaky(error, usize::MAX, false)
        }

        /// Accepts everything, for a test that must show the latch stayed asleep.
        pub(crate) fn healthy() -> (Self, StreamLog) {
            Self::flaky(
                || io::Error::other("unused: this stream never refuses"),
                0,
                false,
            )
        }

        /// Takes the bytes and then cannot drain them.
        pub(crate) fn unflushable() -> (Self, StreamLog) {
            Self::flaky(|| io::Error::from_raw_os_error(5), 0, true)
        }
    }

    impl Write for ScriptedStream {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            if self.refusals_left > 0 {
                self.refusals_left -= 1;
                self.log.refusals.fetch_add(1, Ordering::SeqCst);
                return Err((self.error)());
            }
            self.log
                .heard
                .lock()
                .unwrap()
                .push_str(&String::from_utf8_lossy(buf));
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            if self.flush_refuses {
                self.log.refusals.fetch_add(1, Ordering::SeqCst);
                return Err((self.error)());
            }
            Ok(())
        }
    }

    /// What a closed pipe reports -- the shape a broken-pipe helper would recognise.
    pub(crate) fn broken_pipe() -> io::Error {
        io::Error::from(io::ErrorKind::BrokenPipe)
    }

    /// The error a hung-up pty actually answers with, which is *not* what a closed pipe gives.
    ///
    /// 5 is `EIO` on both macOS and Linux, which is why no `libc` dev-dependency is needed.
    pub(crate) fn input_output_error() -> io::Error {
        io::Error::from_raw_os_error(5)
    }
}

#[cfg(test)]
mod tests {
    use super::Output;
    use super::scripted::{ScriptedStream, broken_pipe, input_output_error};

    /// A broken pipe is swallowed rather than panicked on, and the stream goes quiet after it.
    ///
    /// The panic is the bug: it is what `eprintln!` raises, and inside `finish` it lands
    /// before `session.json` is written. Silence costs the user a line they could no longer
    /// have read.
    #[test]
    fn a_broken_pipe_is_swallowed_and_latches_the_stream() {
        let (stream, log) = ScriptedStream::hung_up(broken_pipe);
        let mut out = Output::to(stream);

        out.line(format_args!("[timing] diagnostics"));
        out.line(format_args!("  mic   no buffers received"));

        assert_eq!(log.refusals(), 1, "one attempt, then the latch holds");
        assert_eq!(log.heard(), "");
    }

    /// The same for the error a closed terminal really returns, which no broken-pipe helper
    /// would recognise: the kind is deliberately not asserted, because nothing here branches
    /// on it -- every refusal means the same thing.
    #[test]
    fn an_input_output_error_is_swallowed_and_latches_the_stream() {
        let (stream, log) = ScriptedStream::hung_up(input_output_error);
        let mut out = Output::to(stream);

        out.line(format_args!("[timing] diagnostics"));
        out.line(format_args!("  mic   no buffers received"));

        assert_eq!(log.refusals(), 1, "one attempt, then the latch holds");
        assert_eq!(log.heard(), "");
    }

    /// Latching is a decision to stop trying, not a reaction to one transient failure: a
    /// stream that recovers is still written to no further.
    #[test]
    fn a_stream_that_recovers_after_one_refusal_is_still_not_written_to_again() {
        let (stream, log) = ScriptedStream::flaky(broken_pipe, 1, false);
        let mut out = Output::to(stream);

        out.line(format_args!("first"));
        out.line(format_args!("second"));

        assert_eq!(log.refusals(), 1);
        assert_eq!(log.heard(), "", "the recovered stream hears nothing either");
    }

    /// Bytes that went in but never drained count as a failure: the promise is that the line
    /// reaches the terminal or the latch closes, not that it sits in memory until an exit that
    /// may never come.
    #[test]
    fn a_stream_that_cannot_flush_latches_on_its_own() {
        let (stream, log) = ScriptedStream::unflushable();
        let mut out = Output::to(stream);

        out.line(format_args!("first"));
        out.line(format_args!("second"));

        assert_eq!(log.refusals(), 1, "the failed flush alone is enough");
    }

    /// A stream that is merely fine hears every line in full, newline included.
    ///
    /// The other direction from the tests above: without it, a latch that fired on the first
    /// write of every run would keep the teardown safe and the user's terminal blank.
    #[test]
    fn a_stream_that_accepts_writes_hears_every_line() {
        let (stream, log) = ScriptedStream::healthy();
        let mut out = Output::to(stream);

        out.line(format_args!(
            "  {label:<8} {:.2} buffers",
            1.5,
            label = "mic"
        ));
        out.line(format_args!("[calendar] picked a meeting"));

        assert_eq!(
            log.heard(),
            "  mic      1.50 buffers\n[calendar] picked a meeting\n",
            "byte for byte, in order, alignment preserved"
        );
        assert_eq!(log.refusals(), 0, "nothing latched");
    }

    /// The two streams this crate speaks through fail apart from each other.
    ///
    /// One global flag would be simpler and wrong: a timing report that could not reach stderr
    /// is no reason to withhold the calendar-access heads-up printed on stdout.
    #[test]
    fn a_stream_that_went_quiet_does_not_silence_the_other() {
        let (stdout_stream, out_log) = ScriptedStream::healthy();
        let (stderr_stream, err_log) = ScriptedStream::hung_up(input_output_error);
        let mut out = Output::to(stdout_stream);
        let mut err = Output::to(stderr_stream);

        err.line(format_args!("[timing] refused"));
        out.line(format_args!("Asking macOS for calendar access."));
        err.line(format_args!("[timing] refused again"));
        out.line(format_args!("and the next line"));

        assert_eq!(err_log.refusals(), 1, "two lines refused, one attempted");
        assert_eq!(
            out_log.heard(),
            "Asking macOS for calendar access.\nand the next line\n"
        );
    }

    /// No print site in `src/` bypasses this module.
    ///
    /// Walks the crate's sources rather than trusting a review to notice the next bare print:
    /// the invariant is "no print that can panic anywhere", and the only thing that keeps it
    /// true is something that fails when it isn't. Known false positive: a forbidden token
    /// inside a *trailing* comment on a code line (a full-line comment is skipped) -- cargo's
    /// own equivalent parses the files with `syn`, which is more machinery than a crate with
    /// fifteen print sites deserves. `examples/*.rs` sit outside `src/` and stay free to
    /// print: they are developer tools run in a live terminal.
    #[test]
    fn nothing_prints_bypassing_the_latch() {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let found: Vec<String> = FORBIDDEN
            .iter()
            .flat_map(|(tokens, replacement)| {
                offenders(&src, tokens)
                    .into_iter()
                    .map(move |(path, line_no, line)| {
                        format!("{path}:{line_no}: {line:?} -> use {replacement}")
                    })
            })
            .collect();

        assert!(
            found.is_empty(),
            "prints that bypass the latch:\n{}",
            found.join("\n")
        );
    }

    /// The banned spellings and what each one becomes.
    ///
    /// Built by concatenation so this table cannot match itself.
    const FORBIDDEN: [(&[&str], &str); 2] = [
        (
            &[
                concat!("print", "!"),
                concat!("println", "!"),
                concat!("eprint", "!"),
                concat!("eprintln", "!"),
            ],
            "crate::output::Output::stderr",
        ),
        (
            &[concat!("io::", "stdout("), concat!("io::", "stderr(")],
            "crate::output::Output::stdout / ::stderr",
        ),
    ];

    /// Every non-comment line under `root` that carries one of `tokens`.
    ///
    /// `output.rs` is exempt: it holds the forbidden spellings in prose and in this scan's own
    /// token table.
    fn offenders(root: &std::path::Path, tokens: &[&str]) -> Vec<(String, usize, String)> {
        let mut found = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("this crate's src directory is readable") {
                let path = entry.expect("a readable directory entry").path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path.file_name().is_some_and(|n| n == "output.rs")
                    || path.extension().is_none_or(|e| e != "rs")
                {
                    continue;
                }
                let text = std::fs::read_to_string(&path).expect("a source file is readable");
                for (line_no, line) in text.lines().enumerate() {
                    if tokens.iter().any(|t| line.contains(t))
                        && !line.trim_start().starts_with("//")
                    {
                        found.push((
                            path.display().to_string(),
                            line_no + 1,
                            line.trim().to_owned(),
                        ));
                    }
                }
            }
        }
        found
    }

    /// The scan itself is checked, so a walk that silently visited nothing could not pass the
    /// test above.
    ///
    /// The planted file is written at run time rather than kept in the repo: a committed
    /// offender would be caught by the real scan, which is the point.
    #[test]
    fn the_guard_finds_a_bare_print_planted_in_a_source_file() {
        let dir = tempfile::tempdir().expect("a scratch directory");
        std::fs::write(
            dir.path().join("planted.rs"),
            format!("fn n() {{ {}!(\"x\"); }}\n", concat!("eprint", "ln")),
        )
        .expect("the scratch file is writable");

        assert_eq!(
            offenders(dir.path(), &[concat!("println", "!")]).len(),
            1,
            "one planted print, one finding"
        );
    }
}
