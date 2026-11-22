//! The pacing queue between the run's notes and the frame's state machine.
//!
//! # Why it exists
//!
//! The run sends `Stopping`, finalizes, then sends `Recorded` (or `FinishFailed`) and
//! `Watching` with no work between the last two. A frame that paints each note the moment it
//! arrives shows `finalizing the session` for exactly as long as the finalize takes, and a
//! failure for microseconds: the notice that exists to explain why recording stopped is
//! structurally unreadable. The floor belongs here, in the frame's intake, and not in the
//! run's loop: sleeping in the loop would delay a start edge and would change what the plain
//! presenter does, and neither may happen.
//!
//! # Rules
//!
//! - A *held* note ([`Note::Stopping`], [`Note::DeviceChanged`], [`Note::MicStalled`],
//!   [`Note::FinishFailed`]) is an end-of-session notice. When one is released at time `t`,
//!   nothing after it is released before `t + `[`END_NOTICE_FLOOR`].
//! - Notes that arrive during the hold queue up in arrival order, so `Watching` still follows.
//! - A *bypass* note (one that opens a session) is released at once, together with everything
//!   queued ahead of it, and clears the hold: a new call opening during the notice wins
//!   immediately.
//! - Everything else is released as soon as no hold is running, so an idle frame adds no
//!   latency.
//!
//! The pacer takes `now` as a monotonic [`Duration`] from the caller, so its tests need no
//! clock and no sleep. The frame wakes every tick, so a hold that has expired releases within
//! one tick of the floor, not at the exact instant.

use std::collections::VecDeque;
use std::time::Duration;

use crate::record::Note;

/// The least time an end-of-session notice stays on screen once painted.
///
/// 1.5 seconds: a fixation on the header row (about 0.3 s to find it and settle) plus a short
/// sentence at reading speed (the longest held text, `finalizing the session`, is three words,
/// about 0.5 s at 200 words per minute), with the remainder as margin for a person glancing
/// over from another window. Long enough to read, short enough that the next call opening
/// is never made to wait for it -- and a start edge never does, see the bypass rule.
pub(super) const END_NOTICE_FLOOR: Duration = Duration::from_millis(1500);

/// How a note takes part in pacing.
enum Class {
    /// An end-of-session notice that must stay visible for the floor.
    Held,
    /// A note that opens a session: released at once, clearing any hold.
    Bypass,
    /// Anything else: released when no hold is running.
    Plain,
}

/// One decision per [`Note`] variant, with no wildcard: a new variant must choose.
fn classify(note: &Note) -> Class {
    match note {
        Note::Stopping | Note::DeviceChanged | Note::MicStalled | Note::FinishFailed(_) => {
            Class::Held
        }
        Note::AlreadyActive | Note::SessionStarted { .. } => Class::Bypass,
        Note::CalendarProblem(_)
        | Note::Watching
        | Note::NoNewSession
        | Note::MeetingOffered { .. }
        | Note::MeetingSettled { .. }
        | Note::MeetingCleared
        | Note::RosterAttached { .. }
        | Note::Recorded { .. }
        | Note::Overrun { .. }
        | Note::BeginFailed(_)
        | Note::GivingUp(_)
        | Note::ActivityDebug(_) => Class::Plain,
    }
}

/// Holds notes back so each end-of-session notice is on screen for [`END_NOTICE_FLOOR`].
#[derive(Default)]
pub(super) struct NotePacer {
    queue: VecDeque<Note>,
    /// No non-bypass note is released before this time.
    hold_until: Duration,
}

impl NotePacer {
    /// Accepts a note the run sent. Whether and when it is released is [`NotePacer::release`]'s
    /// decision.
    pub(super) fn push(&mut self, note: Note) {
        self.queue.push_back(note);
    }

    /// The notes due at `now`, in arrival order.
    pub(super) fn release(&mut self, now: Duration) -> Vec<Note> {
        let mut out = Vec::new();
        loop {
            // A session start anywhere in the queue ends the hold: everything ahead of it is
            // released now, in order, so the new session is never kept behind a notice.
            if let Some(at) = self
                .queue
                .iter()
                .position(|n| matches!(classify(n), Class::Bypass))
            {
                self.hold_until = Duration::ZERO;
                out.extend(self.queue.drain(..=at));
                continue;
            }
            match self.queue.front().map(classify) {
                Some(Class::Held) if now >= self.hold_until => {
                    self.hold_until = now + END_NOTICE_FLOOR;
                }
                Some(Class::Plain) if now >= self.hold_until => {}
                _ => break,
            }
            out.extend(self.queue.pop_front());
        }
        out
    }

    /// Everything still queued, in order, regardless of time: for teardown, which owes the
    /// narration and the stashed trouble every note the run said.
    pub(super) fn drain(&mut self) -> Vec<Note> {
        self.hold_until = Duration::ZERO;
        self.queue.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record_screen::state::{Phase, State};
    use meethook_session::SessionId;

    const T0: Duration = Duration::ZERO;
    const EPS: Duration = Duration::from_nanos(1);

    fn started() -> Note {
        Note::SessionStarted {
            id: SessionId::parse("20260809-050000").unwrap(),
            dir: std::path::PathBuf::from("/tmp/meethook/sessions/20260809-050000"),
            mic_rate: 48_000,
            mic_channels: 1,
            speaker_rate: 44_100,
        }
    }

    fn failed() -> Note {
        Note::FinishFailed("This session did not produce a usable recording".into())
    }

    fn names(notes: &[Note]) -> Vec<&'static str> {
        notes
            .iter()
            .map(|n| match n {
                Note::Stopping => "stopping",
                Note::Watching => "watching",
                Note::FinishFailed(_) => "failed",
                Note::SessionStarted { .. } => "started",
                Note::MicStalled => "stalled",
                _ => "other",
            })
            .collect()
    }

    #[test]
    fn stopping_is_held_for_the_floor_before_watching_follows() {
        let mut p = NotePacer::default();
        p.push(Note::Stopping);
        p.push(Note::Watching);
        assert_eq!(names(&p.release(T0)), ["stopping"]);
        assert!(p.release(END_NOTICE_FLOOR - EPS).is_empty());
        assert_eq!(names(&p.release(END_NOTICE_FLOOR)), ["watching"]);
    }

    #[test]
    fn a_failure_is_held_under_the_same_floor_as_stopping() {
        let mut p = NotePacer::default();
        p.push(Note::Stopping);
        p.push(failed());
        p.push(Note::Watching);
        assert_eq!(names(&p.release(T0)), ["stopping"]);
        assert_eq!(names(&p.release(END_NOTICE_FLOOR)), ["failed"]);
        assert!(p.release(END_NOTICE_FLOOR * 2 - EPS).is_empty());
        assert_eq!(names(&p.release(END_NOTICE_FLOOR * 2)), ["watching"]);
    }

    #[test]
    fn a_session_start_during_the_hold_wins_at_once_and_clears_it() {
        let mut p = NotePacer::default();
        p.push(Note::Stopping);
        p.push(Note::Watching);
        let now = END_NOTICE_FLOOR / 3;
        assert_eq!(names(&p.release(T0)), ["stopping"]);
        p.push(started());
        assert_eq!(names(&p.release(now)), ["watching", "started"]);
        // The hold is gone: the next note is not made to wait out the old one.
        p.push(Note::Watching);
        assert_eq!(names(&p.release(now)), ["watching"]);
    }

    #[test]
    fn a_note_after_the_hold_expired_is_released_at_once() {
        let mut p = NotePacer::default();
        p.push(Note::MicStalled);
        assert_eq!(names(&p.release(T0)), ["stalled"]);
        let later = END_NOTICE_FLOOR * 10;
        p.push(Note::Watching);
        assert_eq!(names(&p.release(later)), ["watching"]);
    }

    #[test]
    fn drain_returns_everything_in_order_regardless_of_time() {
        let mut p = NotePacer::default();
        p.push(Note::Stopping);
        p.push(failed());
        p.push(Note::Watching);
        assert_eq!(names(&p.release(T0)), ["stopping"]);
        assert_eq!(names(&p.drain()), ["failed", "watching"]);
        assert!(p.drain().is_empty());
    }

    #[test]
    fn the_phase_stays_finalizing_until_the_floor_has_passed() {
        let mut p = NotePacer::default();
        let mut s = State::default();
        p.push(Note::Stopping);
        p.push(Note::Watching);
        for note in p.release(T0) {
            s.apply(&note);
        }
        assert_eq!(s.phase, Phase::Finalizing);
        for note in p.release(END_NOTICE_FLOOR - EPS) {
            s.apply(&note);
        }
        assert_eq!(s.phase, Phase::Finalizing);
        for note in p.release(END_NOTICE_FLOOR) {
            s.apply(&note);
        }
        assert_eq!(s.phase, Phase::Idle);
    }
}
