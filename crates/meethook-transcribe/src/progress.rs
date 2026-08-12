//! Saying what a long phase is doing, while it is doing it.
//!
//! Transcribing a 49-minute session spends minutes at a time inside single loops -- reading
//! and resampling half a gigabyte of audio, cancelling echo over 294,000 frames, running a
//! segmentation graph once per ten seconds of meeting -- and until this module existed every
//! one of those stretches was indistinguishable from a wedged process.
//!
//! # The three decisions worth knowing before reading the code
//!
//! *Everything goes to stderr.* The `progress`/`out` writer threaded through
//! [`crate::transcribe_session`] and [`crate::run_batch`] is the batch's per-session record,
//! meant to stay greppable; operational notes belong beside the model downloads and the
//! speech gate instead. That is the same argument already written down on [`crate::asr`]'s
//! `report`, and it is also what makes this module usable from places the writer does not
//! reach at all -- inside AEC3's frame loop, or inside a `'static` whisper.cpp callback.
//!
//! *Lines are ordinary lines.* No `\r`, no ANSI: unlike the model download, which rewrites
//! one line in place and therefore turns into a single enormous line when redirected, a
//! phase's output has to read correctly in `2> progress.log` as well as in a terminal. It
//! also means no phase has to arrange to terminate its own last line, which matters for the
//! whisper callback, since nothing can reach into it once decoding starts.
//!
//! *A phase that is quick says nothing at all.* Throttling on elapsed time rather than on
//! units done is what keeps unit tests, short tracks and the `import`/`levels`/`trials` paths
//! silent with no size threshold to pick: a phase that finishes inside [`INTERVAL`] never
//! reaches its first line, and never prints a closing one either.

use std::time::{Duration, Instant};

/// How long a phase stays quiet between lines.
///
/// The whole point is a heartbeat, not a progress bar: at five seconds a 49-minute session's
/// pre-pass emits a few dozen lines in total, which is enough to tell working from wedged and
/// few enough to page through afterwards.
const INTERVAL: Duration = Duration::from_secs(5);

/// Roughly how many times a phase is willing to read the clock over its whole run.
///
/// [`Phase::at`] is called once per *unit*, and the units differ by five orders of magnitude
/// between callers -- 141 million samples in the read loop against 294 windows in
/// segmentation. A fixed stride cannot serve both: one that keeps the read loop cheap would
/// leave segmentation permanently silent, and one that lets segmentation speak would put an
/// `Instant::now()` on a per-sample path. So the stride is derived from the phase's total,
/// giving every phase about this many clock reads however long it is.
const CLOCK_READS: u64 = 1024;

/// The most calls a phase will ever skip between clock reads.
///
/// Bounds the coarseness for the very longest loops: at 141 million samples this is still
/// ~2,000 reads, and a read costs tens of nanoseconds, so the whole cost of instrumenting the
/// read loop stays well under a millisecond.
const MAX_STRIDE: u64 = 65_536;

/// One long-running stretch of work, reporting on stderr while it runs.
///
/// Constructed where the work starts, ticked inside its loop, and finished when the loop
/// ends:
///
/// ```ignore
/// let mut phase = Phase::start("cancelling echo");
/// for (index, frame) in frames.enumerate() {
///     phase.at(index, frame_count);
///     // ...
/// }
/// phase.done();
/// ```
///
/// Neither [`Phase::start`] nor an early [`Phase::at`] prints anything, so a caller pays no
/// output at all for work that turns out to be fast.
pub(crate) struct Phase {
    label: String,
    /// The `total` the current stride was computed from, so a caller whose total changes
    /// between calls re-derives it rather than keeping a stride sized for the old one.
    total: usize,
    /// `stride - 1`, so the skip test is a mask rather than a division.
    mask: u64,
    ticks: u64,
    started: Instant,
    /// When the last line was printed, or when the phase started if there has been none. The
    /// first line is therefore [`INTERVAL`] after the start, not immediately.
    last_spoke: Instant,
    spoke: bool,
}

impl Phase {
    /// Begins a phase called `label`, printing nothing.
    ///
    /// `label` is what a user reads, so it is a lowercase description of the work in progress
    /// -- `reading mic.wav`, `cancelling echo`, `diarize: segmenting` -- matching the
    /// `speech gate:` lines it appears next to.
    pub(crate) fn start(label: impl Into<String>) -> Phase {
        Phase::started_at(label, Instant::now())
    }

    /// [`Phase::start`] with the starting instant supplied, so a test can hold the whole
    /// phase on one clock it controls rather than racing the real one by microseconds.
    fn started_at(label: impl Into<String>, now: Instant) -> Phase {
        Phase {
            label: label.into(),
            // Zero rather than the first caller's total, so the first `at` always derives a
            // stride; a `total` of zero is also the one value that would divide by zero.
            total: 0,
            mask: 0,
            ticks: 0,
            started: now,
            last_spoke: now,
            spoke: false,
        }
    }

    /// Reports that `done` of `total` units are complete, printing at most one line.
    ///
    /// Cheap enough to call per sample: most calls do nothing but increment a counter and
    /// test a mask. `total` is passed per call rather than at construction because the
    /// whisper callback is handed a percentage while every other caller counts its own units,
    /// and because a phase whose total is only known as an upper bound can revise it.
    ///
    /// `done` and `total` must be in the unit the loop *ticks* in, which is not always the
    /// unit it iterates over: the stride is derived from `total` (see [`CLOCK_READS`]), so a
    /// frame loop reporting sample counts would skip several hundred ticks between clock
    /// reads and go quiet for the whole phase. Overstating `total` only costs extra clock
    /// reads, so an upper bound is safe where an exact count is not available.
    pub(crate) fn at(&mut self, done: usize, total: usize) {
        if let Some(line) = self.tick(done, total, Instant::now) {
            eprintln!("{line}");
        }
    }

    /// Ends the phase, printing a closing line only if it ever spoke.
    ///
    /// The condition is what keeps a fast phase from leaving an orphan `done` line with no
    /// progress above it.
    pub(crate) fn done(self) {
        if let Some(line) = self.closing(Instant::now()) {
            eprintln!("{line}");
        }
    }

    /// The line [`Phase::at`] would print, if this call is one that should speak.
    ///
    /// Split out with the clock as a parameter so the throttle and the stride can be tested
    /// directly: a test can count how often its clock is consulted, which is the property
    /// that keeps `Instant::now()` off the per-sample path, and no test has to capture
    /// stderr to assert on the text.
    fn tick<F: FnOnce() -> Instant>(
        &mut self,
        done: usize,
        total: usize,
        clock: F,
    ) -> Option<String> {
        if total != self.total {
            self.total = total;
            self.mask = stride_for(total) - 1;
        }

        self.ticks = self.ticks.wrapping_add(1);
        if self.ticks & self.mask != 0 {
            return None;
        }

        let now = clock();
        if !should_speak(self.last_spoke, now) {
            return None;
        }
        self.last_spoke = now;
        self.spoke = true;
        Some(line(
            &self.label,
            done,
            total,
            now.saturating_duration_since(self.started),
        ))
    }

    /// The closing line, or `None` for a phase that was quiet throughout.
    fn closing(&self, now: Instant) -> Option<String> {
        self.spoke.then(|| {
            format!(
                "{}: done ({})",
                self.label,
                elapsed(now.saturating_duration_since(self.started))
            )
        })
    }
}

/// How many [`Phase::at`] calls to skip between clock reads, for a phase of `total` units.
///
/// Always a power of two, so callers can mask instead of dividing.
fn stride_for(total: usize) -> u64 {
    ((total as u64 / CLOCK_READS).max(1))
        .next_power_of_two()
        .min(MAX_STRIDE)
}

/// Whether enough time has passed since the last line for another one.
fn should_speak(last_spoke: Instant, now: Instant) -> bool {
    now.saturating_duration_since(last_spoke) >= INTERVAL
}

/// One progress line: `cancelling echo: 42% (1m12s)`.
fn line(label: &str, done: usize, total: usize, since_start: Duration) -> String {
    format!(
        "{label}: {}% ({})",
        percent(done, total),
        elapsed(since_start)
    )
}

/// `done` as a percentage of `total`, clamped to 0..=100.
///
/// A `total` of zero reports 0 rather than dividing: a phase can legitimately be handed an
/// empty track, and "0%" beside a phase that immediately ends is not misleading.
fn percent(done: usize, total: usize) -> u64 {
    if total == 0 {
        return 0;
    }
    ((done as u128 * 100 / total as u128) as u64).min(100)
}

/// A duration as `47s` or `4m07s`.
///
/// Whole seconds, because this is a heartbeat and a millisecond field would only be noise on
/// a line that appears every five seconds.
fn elapsed(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds >= 60 {
        format!("{}m{:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    /// A clock the test moves by hand, counting how often it is read.
    struct FakeClock {
        base: Instant,
        offset: Cell<Duration>,
        reads: Cell<usize>,
    }

    impl FakeClock {
        fn new() -> FakeClock {
            FakeClock {
                base: Instant::now(),
                offset: Cell::new(Duration::ZERO),
                reads: Cell::new(0),
            }
        }

        fn advance(&self, by: Duration) {
            self.offset.set(self.offset.get() + by);
        }

        fn now(&self) -> Instant {
            self.reads.set(self.reads.get() + 1);
            self.base + self.offset.get()
        }
    }

    /// Ticks `calls` times against `clock`, returning every line that came out.
    fn run(phase: &mut Phase, clock: &FakeClock, calls: usize, total: usize) -> Vec<String> {
        (0..calls)
            .filter_map(|done| phase.tick(done, total, || clock.now()))
            .collect()
    }

    #[test]
    fn a_phase_that_finishes_inside_the_interval_says_nothing_at_all() {
        let clock = FakeClock::new();
        let mut phase = Phase::started_at("reading mic.wav", clock.base);

        let lines = run(&mut phase, &clock, 4096, 4096);
        clock.advance(INTERVAL - Duration::from_millis(1));

        assert!(lines.is_empty(), "{lines:?}");
        assert_eq!(phase.closing(clock.now()), None);
    }

    #[test]
    fn a_line_is_emitted_once_per_interval_rather_than_once_per_call() {
        let clock = FakeClock::new();
        let mut phase = Phase::started_at("cancelling echo", clock.base);

        // Three intervals' worth of work, with plenty of calls inside each one.
        let mut lines = Vec::new();
        for _ in 0..3 {
            clock.advance(INTERVAL);
            lines.extend(run(&mut phase, &clock, 4096, 4096));
        }

        assert_eq!(lines.len(), 3, "{lines:?}");
    }

    #[test]
    fn the_line_names_the_phase_its_progress_and_how_long_it_has_been_running() {
        let clock = FakeClock::new();
        let mut phase = Phase::started_at("cancelling echo", clock.base);
        clock.advance(Duration::from_secs(72));

        let line = phase.tick(42, 100, || clock.now()).unwrap();

        assert_eq!(line, "cancelling echo: 42% (1m12s)");
    }

    #[test]
    fn a_phase_that_spoke_closes_with_a_done_line() {
        let clock = FakeClock::new();
        let mut phase = Phase::started_at("diarize: segmenting", clock.base);
        clock.advance(Duration::from_secs(90));
        phase.tick(1, 2, || clock.now()).unwrap();

        assert_eq!(
            phase.closing(clock.now()).as_deref(),
            Some("diarize: segmenting: done (1m30s)")
        );
    }

    /// The read loop calls `at` once per sample -- 141 million times on a 49-minute session --
    /// so a clock read on every call would be a real cost on the very path this exists to
    /// narrate. Asserting the stride here is what stops a later edit putting one back.
    #[test]
    fn the_clock_is_read_about_a_thousand_times_however_many_units_a_phase_has() {
        for total in [1_000usize, 100_000, 10_000_000] {
            let clock = FakeClock::new();
            let mut phase = Phase::started_at("reading speaker.wav", clock.base);

            run(&mut phase, &clock, total, total);

            let reads = clock.reads.get() as u64;
            assert!(
                reads <= total as u64 / stride_for(total) + 1,
                "{total} units read the clock {reads} times"
            );
            // And enough of them that a phase of this length can still speak every interval.
            assert!(
                reads >= 256,
                "{total} units read the clock only {reads} times"
            );
        }
    }

    /// Segmentation has a few hundred windows and diarization's clustering a few thousand
    /// turns, so short phases have to tick on every call or they would never report at all.
    #[test]
    fn a_phase_with_fewer_units_than_the_stride_budget_ticks_every_call() {
        assert_eq!(stride_for(0), 1);
        assert_eq!(stride_for(294), 1);
        assert_eq!(stride_for(CLOCK_READS as usize), 1);
    }

    #[test]
    fn the_stride_is_capped_so_it_stays_a_heartbeat_on_the_longest_loops() {
        assert_eq!(stride_for(usize::MAX), MAX_STRIDE);
        assert!(stride_for(141_000_000) <= MAX_STRIDE);
    }

    #[test]
    fn an_empty_phase_reports_zero_rather_than_dividing_by_it() {
        assert_eq!(percent(0, 0), 0);
        assert_eq!(
            line("reading mic.wav", 0, 0, Duration::from_secs(7)),
            "reading mic.wav: 0% (7s)"
        );
    }

    #[test]
    fn percent_is_clamped_to_a_hundred_when_a_total_turns_out_to_be_an_underestimate() {
        assert_eq!(percent(200, 100), 100);
        assert_eq!(percent(0, 100), 0);
        assert_eq!(percent(1, 3), 33);
    }

    #[test]
    fn elapsed_reads_as_minutes_and_seconds_past_a_minute() {
        assert_eq!(elapsed(Duration::ZERO), "0s");
        assert_eq!(elapsed(Duration::from_secs(47)), "47s");
        assert_eq!(elapsed(Duration::from_secs(60)), "1m00s");
        assert_eq!(elapsed(Duration::from_secs(247)), "4m07s");
    }

    /// Redirection is a supported way to run this -- `2> progress.log` and read it back --
    /// so no line may carry a carriage return or an escape sequence the way the model
    /// download's in-place rewrite does.
    #[test]
    fn lines_are_plain_text_so_they_survive_being_redirected_to_a_file() {
        let clock = FakeClock::new();
        let mut phase = Phase::started_at("writing mic.cleaned.wav", clock.base);
        clock.advance(INTERVAL);
        let line = phase.tick(1, 2, || clock.now()).unwrap();

        for text in [line, phase.closing(clock.now()).unwrap()] {
            assert!(
                !text.contains(['\r', '\n', '\u{1b}']),
                "{text:?} would not read back cleanly from a file"
            );
        }
    }
}
