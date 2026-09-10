//! Stopping both capture engines exactly once, whichever path gets there first.
//!
//! A live session owns two engines that record into files on disk. Until now the only way to
//! stop them was [`crate::RunningSession::finish`], which consumes the session -- so every
//! path that *didn't* finish one (a panic anywhere in the record loop, an early return added
//! by a later edit, any unwind at all) left `AVAudioEngine` recording into `mic.wav` and the
//! `SCStream` still capturing system audio, with writer threads appending to WAV files nobody
//! would ever finalize. The process could be gone and the microphone still claimed.
//!
//! This module holds the rule that fixes it: the two engines live in one slot, and only two
//! things can empty it -- [`Engines::settle`] and [`Drop`] -- never both, because emptying it
//! consumes it. Stopping an engine twice is therefore not something this code can express. [`Engines::settle`] is the reported teardown `finish` uses; [`Drop`] runs
//! [`Engine::abandon`] instead, which stops the same engines and says nothing. Either may come
//! first; the second finds an empty slot and does nothing. Precedent for exactly this shape,
//! including the "either may come first" wording: `Interface::restore` in
//! `crates/meethook/src/screen.rs`.
//!
//! Two constraints shape what `abandon` is allowed to do, both because it runs while the
//! process may be unwinding:
//!
//! - **It must not block.** A hang during unwinding is worse than the leak it prevents, so the
//!   drop path passes ScreenCaptureKit a null completion handler -- the selector's handler is
//!   nullable, and OBS does the same in its teardown -- rather than waiting out the ten-second
//!   bounded wait that `settle` pays because someone is there to receive the answer.
//! - **It must not panic.** A panic inside `Drop` while another panic is unwinding aborts the
//!   process. So the drop path discards raises with `let _ =`, never reads the stream
//!   delegate's error slot (whose reader `.expect()`s on a poisoned lock), and turns no
//!   `NSError` into anything anyone might print.
//!
//! What neither path does is *salvage* a dropped session. Alignment data (first-buffer host
//! ticks and the mach timebase) exists only in memory until `session.json`, so a session that
//! never wrote one cannot be aligned honestly, and no `session.json` is written for it -- the
//! directory stays an orphan, which is the classification the rest of the tool already
//! understands. Both paths do finalize both WAV headers, though: it costs one join that is
//! already bounded, and it turns an orphan from playable-minus-the-checkpoint-slack into
//! complete to the last sample.

use crate::Result;
use crate::track::TrackSummary;

/// One live capture, seen only as the thing that must be stopped exactly once.
///
/// Generic over the two capture types so the rule can be decided with no framework and no
/// audio device present -- see the tests. Concrete capabilities the session still asks about
/// mid-session (sample rate, stall detection) stay inherent methods on the real captures,
/// reached through [`Engines::mic`], rather than growing this trait into a shadow copy of them.
pub(crate) trait Engine {
    /// Stops the engine, waits wherever waiting buys an answer, finalizes the WAV, and reports
    /// what it learned.
    fn settle(self) -> Result<TrackSummary>;

    /// Stops the engine and walks away: no completion-handler wait, nothing reported. For the
    /// paths that cannot afford to block and have nobody left to tell.
    ///
    /// Still finalizes the WAV. Total silence is the point; a truncated file is not.
    fn abandon(self);
}

/// A session's two live captures, held as one slot so each is stopped exactly once.
///
/// One `Option` around the **pair**, not one per engine: both are always stopped together, so
/// a half-stopped state would be representable and meaningless -- nothing could repair it. The
/// tuple order is the mic-then-speaker order that `finish`'s error precedence depends on.
///
/// The bounds sit on the struct rather than on its impls because `Drop` demands that: which is
/// another way of saying a value of this type always has an owner that will stop it, in either
/// of the two ways, for as long as it exists.
pub(crate) struct Engines<M: Engine, S: Engine>(Option<(M, S)>);

impl<M: Engine, S: Engine> Engines<M, S> {
    pub(crate) fn new(mic: M, speaker: S) -> Self {
        Engines(Some((mic, speaker)))
    }

    /// Borrows the mic while it is still held. Answers `None` once the engines have been
    /// stopped, which no caller can observe: both ways of stopping them end the session.
    pub(crate) fn mic(&self) -> Option<&M> {
        self.0.as_ref().map(|(mic, _)| mic)
    }

    pub(crate) fn mic_mut(&mut self) -> Option<&mut M> {
        self.0.as_mut().map(|(mic, _)| mic)
    }

    /// Takes both engines and stops them, issuing **both** stops before inspecting either
    /// result, and hands both results back untouched.
    ///
    /// Deciding which failure wins is `finish`'s job, not this one's: returning early on a mic
    /// failure would leave the speaker WAV unfinalized, which is a worse outcome than a late
    /// error. That ordering is why this returns a tuple rather than a `Result`.
    ///
    /// Consumes the guard, so there is no "already settled" state left behind for a second
    /// caller to find -- and the `Drop` that runs at the end of this method sees the empty
    /// slot and does nothing.
    pub(crate) fn settle(mut self) -> (Result<TrackSummary>, Result<TrackSummary>) {
        // `take` rather than a destructure: this method owns `self`, whose type implements
        // `Drop`, so its fields cannot be moved out directly.
        let Some((mic, speaker)) = self.0.take() else {
            // Unreachable through any caller: this method consumes the guard, and the only
            // other thing that empties the slot is `Drop`, which abandons instead of settling.
            // So there is no double-settle to answer for, and no made-up summary to invent.
            unreachable!("settle consumes both engines, so it cannot run twice")
        };
        (mic.settle(), speaker.settle())
    }
}

impl<M: Engine, S: Engine> Drop for Engines<M, S> {
    /// The paths that never reached `finish`: a panic in the record loop, a `?` on a new
    /// early return, anything that drops a live session.
    ///
    /// Silent by design -- every report belongs on `finish`, the fallible path with a caller
    /// left to hear it. Std draws the same line: `BufWriter` documents that errors on drop are
    /// ignored, and `tempfile` answers it with an explicit `close()` rather than a loud drop.
    fn drop(&mut self) {
        if let Some((mic, speaker)) = self.0.take() {
            mic.abandon();
            speaker.abandon();
        }
    }
}

/// The rule, decided with no framework and no audio device anywhere near it.
///
/// Everything these fakes can decide is *which* method ran and *how many times*, which is the
/// whole of this module's logic. What they cannot decide is that
/// `stopCaptureWithCompletionHandler(None)` really releases the capture and lets the next
/// process open the device -- that needs a machine, and TASK-067.03.01 carries that row.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// Which method ran on which engine, in the order it ran.
    type Log = Rc<RefCell<Vec<String>>>;

    struct Fake {
        name: &'static str,
        log: Log,
        fails: bool,
    }

    impl Fake {
        fn new(name: &'static str, log: &Log, fails: bool) -> Self {
            Fake {
                name,
                log: Rc::clone(log),
                fails,
            }
        }
    }

    impl Engine for Fake {
        fn settle(self) -> Result<TrackSummary> {
            self.log.borrow_mut().push(format!("settle:{}", self.name));
            if self.fails {
                Err(Error::ScreenCaptureKit("fake".to_owned()))
            } else {
                Ok(summary())
            }
        }

        fn abandon(self) {
            self.log.borrow_mut().push(format!("abandon:{}", self.name));
        }
    }

    fn summary() -> TrackSummary {
        TrackSummary {
            sample_rate: 48_000,
            frames: 0,
            first_buffer: None,
            timing: None,
        }
    }

    fn engines(mic_fails: bool, speaker_fails: bool) -> (Engines<Fake, Fake>, Log) {
        let log: Log = Rc::new(RefCell::new(Vec::new()));
        let engines = Engines::new(
            Fake::new("mic", &log, mic_fails),
            Fake::new("speaker", &log, speaker_fails),
        );
        (engines, log)
    }

    fn taken(log: &Log) -> Vec<String> {
        std::mem::take(&mut log.borrow_mut())
    }

    /// The bug this module exists for: dropping a session that was never finished used to stop
    /// nothing, and leave the microphone claimed by a process that had already left.
    #[test]
    fn a_session_dropped_without_finishing_abandons_both_engines() {
        let (engines, log) = engines(false, false);

        drop(engines);

        assert_eq!(
            taken(&log),
            ["abandon:mic", "abandon:speaker"],
            "a dropped session stops both engines, mic first"
        );
    }

    /// The other half of exactly-once: `finish` settles, and the `Drop` that follows it must
    /// then have nothing to do.
    #[test]
    fn finishing_first_leaves_the_drop_nothing_to_do() {
        let (engines, log) = engines(false, false);

        let (mic_stop, speaker_stop) = engines.settle();

        assert!(mic_stop.is_ok() && speaker_stop.is_ok());
        assert_eq!(taken(&log), ["settle:mic", "settle:speaker"]);
    }

    /// What protects `finish`'s error precedence, which no existing test covers: the mic
    /// failing must not cost the speaker its stop, nor its result.
    #[test]
    fn both_stops_are_issued_before_either_result_is_inspected() {
        let (engines, log) = engines(true, false);

        let (mic_stop, speaker_stop) = engines.settle();

        assert!(
            mic_stop.is_err(),
            "the mic's failure is handed back, not taken"
        );
        assert!(
            speaker_stop.is_ok(),
            "the speaker's own result survives the mic's failure"
        );
        assert_eq!(taken(&log), ["settle:mic", "settle:speaker"]);
    }

    /// A failing engine must not make the drop path panic -- a second panic while unwinding
    /// aborts -- and must not be retried into a second stop.
    #[test]
    fn an_engine_that_fails_to_stop_is_still_only_stopped_once() {
        let (engines, log) = engines(true, true);

        drop(engines);

        assert_eq!(taken(&log), ["abandon:mic", "abandon:speaker"]);
    }

    /// Reading the mic mid-session -- rate, stall, frames -- must not disturb the rule.
    #[test]
    fn borrowing_the_mic_leaves_it_for_the_teardown() {
        let (mut engines, log) = engines(false, false);

        assert!(engines.mic().is_some());
        assert!(engines.mic_mut().is_some());
        assert!(taken(&log).is_empty(), "asking a question stops nothing");

        drop(engines);
        assert_eq!(taken(&log), ["abandon:mic", "abandon:speaker"]);
    }
}
