//! The record frame as a pure state machine.
//!
//! What the frame shows is derived from the same typed notes the plain run prints: this type
//! consumes a sequence of [`crate::record::Note`]s and holds whatever the frame needs to draw
//! -- which phase the run is in, which session is live, what the calendar offered it, what the
//! last finish produced, what it should say -- and nothing else. No clock in it (elapsed time
//! is the shell's, because it is derived from one), no terminal in it, no audio in it. That is
//! what lets the whole frame be exercised in `cargo test` by feeding it note sequences, the
//! way `enroll`'s screen state is exercised by feeding it answers.
//!
//! Keys enter as typed commands rather than events: the shell translates a keypress into one
//! of the selector methods below, exactly the way `enroll` feeds answers, and the translation
//! table is where a stray modifier or a release-shaped duplicate goes wrong. That split is the
//! decision-012 bar for this file -- no crossterm, no ratatui, no clock import here -- and it
//! is what makes the selector's transitions unit-testable by feeding command sequences.
//!
//! # Wording
//!
//! Every sentence the frame displays comes off the composers in `crate::record`, verbatim:
//! the notice pane shows the same constants the plain run prints, the session pane shows the
//! same lines, and the narration buffer stores the composed stdout-class text so that leaving
//! the interface flushes exactly what a piped run would have written. This state machine never
//! invents a sentence; `render` may only lay out what lands here.
//!
//! # Narration and trouble
//!
//! Two buffers, two stream classes. Stdout-class notes are appended to the narration buffer
//! as their composed text: when the interface closes, that buffer is flushed to stdout, which
//! is what makes scrollback after a full-screen run byte-identical to a plain run's. Stderr-class
//! notes are stashed in the trouble list instead of being printed on arrival: printing them
//! mid-run would write over the alternate screen, and holding them means they can be said in
//! full, in order, once the screen is back.
//!
//! One deliberate exception to "everything the plain run says, said again": [`Note::ActivityDebug`]
//! notes are dropped in screen mode rather than narrated. They fire at the re-check cadence
//! while recording, and narrating them would turn the notice pane into a ticker and bloat the
//! scrollback with diagnostics that exist for somebody watching a hardware run line by line.
//! The drop is documented here because it is the one place the two presenters diverge in what
//! they keep.

use meethook_enroll::{MeetingLabel, MeetingOffer};
use meethook_session::{Paths, SessionId};

use crate::record::{
    ALREADY_ACTIVE, DEVICE_CHANGED, MIC_STALLED, NO_NEW_SESSION, Note, STOPPING, WATCHING,
    giving_up_line,
};

/// Which phase of the run the frame is showing.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Idle: nothing is being recorded, the run is waiting for a call.
    #[default]
    Idle,
    /// A start is in flight or being retried.
    Beginning,
    /// A session is open and delivering audio.
    Recording,
    /// A session is ending: the call ended, the device moved, the mic stalled, or the user
    /// interrupted.
    Finalizing,
}

/// The word the header shows for each phase. UI chrome, not domain prose: it names what the
/// frame is doing, and only `render` ever sees it.
impl Phase {
    pub fn word(self) -> &'static str {
        match self {
            Phase::Idle => "watching",
            Phase::Beginning => "opening",
            Phase::Recording => "recording",
            Phase::Finalizing => "finalizing",
        }
    }
}

/// A session that is open right now: its identity and the rates both engines came up at.
///
/// The rates cross from the [`Note::SessionStarted`] note, where the live backend measured
/// them, rather than being re-read from the engine: the frame must show what the run actually
/// started, and the note is the run saying so.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveSession {
    pub id: SessionId,
    pub dir: String,
    pub mic_rate: u32,
    pub mic_channels: u32,
    pub speaker_rate: u32,
}

/// The most recent finished session: enough to say what it was and what it matched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishedSession {
    pub id: SessionId,
    /// The meeting it was attached to, projected to title and fit like everything else the
    /// run says about meetings. `None` when the session matched nothing.
    pub meeting: Option<MeetingLabel>,
}

/// Everything the frame draws, derived from the notes the run has said so far.
#[derive(Debug)]
pub struct State {
    /// Where sessions live, for the footer: the frame names the directory a new session will
    /// land in before there is a session to name.
    pub paths: Paths,
    pub phase: Phase,
    /// The live session, if one is open.
    pub session: Option<ActiveSession>,
    /// The most recent finish, kept so the frame can still say what it was after the run goes
    /// back to watching.
    pub last: Option<FinishedSession>,
    /// What the run currently wants said about itself: the idle prompt, a device-change or
    /// mic-stall notice, the stopping line. Cleared when a session opens.
    pub notice: Option<String>,
    /// The calendar's offers for the live session, projected so far as the frame may show
    /// them: what the selector lists. Empty when there is no grant or nothing nearby -- the
    /// degraded view, not an error.
    pub offered: Vec<MeetingOffer>,
    /// The one the automatic rule would attach, with the fit that rule decided -- or none.
    /// Shown through its clause, so a weak fit carries its caveat verbatim.
    pub guess: Option<MeetingLabel>,
    /// The offer the user confirmed by hand, if any: it supersedes the guess wherever the
    /// frame states the meeting, and carries `MeetingFit::Confirmed` because that is what
    /// `finish` will write.
    pub settled: Option<MeetingLabel>,
    /// Whether the selector is open: the numbered offers replace the notice region while it
    /// is, and the shell advertises only the keys that work in that context.
    pub selector_open: bool,
    /// The row under the cursor within `offered`. Meaningful while the selector is open;
    /// reset with the pane rather than trusted across sessions.
    pub cursor: usize,
    /// Stderr-class notes, held back from the alternate screen until it is restored.
    pub trouble: Vec<String>,
    /// Composed stdout-class text, in the order the notes arrived. Flushed to stdout when the
    /// interface closes; the frame itself never renders it.
    narration: String,
}

impl State {
    pub fn new(paths: Paths) -> Self {
        Self {
            paths,
            phase: Phase::Idle,
            session: None,
            last: None,
            notice: None,
            trouble: Vec::new(),
            narration: String::new(),
            offered: Vec::new(),
            guess: None,
            settled: None,
            selector_open: false,
            cursor: 0,
        }
    }

    /// The meeting pane dies with the session it described: a new session starting and a
    /// session finishing both call it, rather than spelling the six assignments twice.
    fn clear_meeting_pane(&mut self) {
        self.offered.clear();
        self.guess = None;
        self.settled = None;
        self.selector_open = false;
        self.cursor = 0;
    }

    /// Folds one note into the frame.
    ///
    /// Always advances the buffers; whether anything *visible* changed is the shell's to gate
    /// on, because the shell also redraws on the clock while recording.
    pub fn apply(&mut self, note: &Note) {
        // The stream-class split first: narration takes the stdout class, trouble the stderr
        // class, and the debug notes take neither (see the module doc for why they are dropped).
        match note {
            Note::ActivityDebug(_) => {}
            _ if !note.to_stderr() => self.narration.push_str(&note.composed()),
            _ => self.trouble.push(note.composed()),
        }

        match note {
            Note::CalendarProblem(problem) => {
                self.notice = Some(problem.clone());
            }
            Note::Watching => {
                self.phase = Phase::Idle;
                self.notice = Some(WATCHING.to_string());
            }
            Note::AlreadyActive => {
                self.phase = Phase::Beginning;
                self.notice = Some(ALREADY_ACTIVE.to_string());
            }
            Note::SessionStarted {
                id,
                dir,
                mic_rate,
                mic_channels,
                speaker_rate,
            } => {
                self.session = Some(ActiveSession {
                    id: id.clone(),
                    dir: dir.display().to_string(),
                    mic_rate: *mic_rate,
                    mic_channels: *mic_channels,
                    speaker_rate: *speaker_rate,
                });
                self.phase = Phase::Recording;
                self.notice = None;
                // The predecessor's meeting pane dies with the predecessor: a new session
                // starts before its own offers have arrived, and showing the old ones in the
                // meantime would mislabel it.
                self.clear_meeting_pane();
            }
            Note::MeetingOffered { offered, guess } => {
                // The pane is replaced, not merged: whatever the predecessor showed is gone,
                // and the selector is closed -- the run asking the calendar again is what
                // reopens the question, not a key the user did not press.
                self.offered = offered.clone();
                self.guess = guess.clone();
                self.settled = None;
                self.selector_open = false;
                self.cursor = 0;
            }
            Note::MeetingSettled { label } => {
                // Replaces any earlier pick, and closes the selector: the run saying the pick
                // stuck is what confirms it, not the keypress -- so a pick the run drops never
                // reads as one.
                self.settled = Some(label.clone());
                self.selector_open = false;
            }
            Note::DeviceChanged | Note::MicStalled | Note::Stopping => {
                self.phase = Phase::Finalizing;
                // The session is ending, so picks stop being accepted; the pane itself stays
                // up -- the user should still see what the session was named while it
                // finalizes.
                self.selector_open = false;
                self.notice = Some(
                    match note {
                        Note::DeviceChanged => DEVICE_CHANGED,
                        Note::MicStalled => MIC_STALLED,
                        _ => STOPPING,
                    }
                    .to_string(),
                );
            }
            Note::NoNewSession => {
                self.phase = Phase::Idle;
                self.notice = Some(NO_NEW_SESSION.to_string());
            }
            Note::Recorded { id, meeting, .. } => {
                self.last = Some(FinishedSession {
                    id: id.clone(),
                    meeting: meeting.clone(),
                });
                self.session = None;
                self.phase = Phase::Idle;
                // The pane's outcome lives in `last.meeting` now; the offers die with the
                // session rather than lingering over an idle frame.
                self.clear_meeting_pane();
            }
            Note::GivingUp(attempts) => {
                self.phase = Phase::Idle;
                // The trailing newline is a print concern, not a notice concern.
                self.notice = Some(giving_up_line(*attempts).trim_end().to_string());
            }
            // The stderr class and the dropped class need no visible change of their own:
            // trouble is already buffered above, and `BeginFailed` arriving while a previous
            // notice is up keeps that notice rather than blanking the pane.
            Note::BeginFailed(_) | Note::FinishFailed(_) | Note::ActivityDebug(_) => {}
        }
    }

    /// Takes out the narration accumulated since the last call.
    ///
    /// Called once, at teardown: the frame never renders the buffer, so "take" is the only
    /// access it gets.
    pub fn take_narration(&mut self) -> String {
        std::mem::take(&mut self.narration)
    }

    /// Opens the selector while a session is recording.
    ///
    /// Allowed even with an empty offer list: that is the degraded view (no grant, empty
    /// calendar), and the frame says nothing is offered rather than hiding the key. Outside
    /// recording there is no session to correct, so the key does nothing -- which is also
    /// why the footer does not advertise it there.
    ///
    /// Does not reset the cursor: the pane's replacement resets it, and a user who scrolls,
    /// closes with Esc and reopens is mid-list, not at its top.
    pub fn open_selector(&mut self) {
        if self.phase == Phase::Recording {
            self.selector_open = true;
        }
    }

    /// Moves the cursor down one row, wrapping. No-op on an empty list: there is nothing to
    /// move through, and the frame keeps saying so.
    pub fn next(&mut self) {
        if !self.offered.is_empty() {
            self.cursor = (self.cursor + 1) % self.offered.len();
        }
    }

    /// Moves the cursor up one row, wrapping. No-op on an empty list.
    ///
    /// `(cursor + len - 1) % len` rather than `wrapping_sub(1) % len`: the underflow route
    /// computes `usize::MAX % len`, which lands back on zero for lengths that divide it --
    /// silently a no-op where a wrap was owed.
    pub fn previous(&mut self) {
        if !self.offered.is_empty() {
            self.cursor = (self.cursor + self.offered.len() - 1) % self.offered.len();
        }
    }

    /// The `event_id` under the cursor, addressed back to the run -- or none when the
    /// selector has no row to confirm.
    ///
    /// Does **not** close the selector: closing waits for the run's [`Note::MeetingSettled`],
    /// so a pick that is silently dropped leaves the user looking at the list rather than a
    /// false confirmation.
    pub fn confirm(&mut self) -> Option<String> {
        self.offered
            .get(self.cursor)
            .map(|offer| offer.event_id.clone())
    }

    /// Closes the selector without picking: the escape, or the session ending underneath it.
    pub fn close_selector(&mut self) {
        self.selector_open = false;
    }
}

impl Default for State {
    fn default() -> Self {
        // Tests build states without a real data root; a throwaway path keeps `State`
        // constructible anywhere a `MEETHOOK_ROOT` cannot be.
        Self::new(Paths::new("/tmp/meethook-test"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{
        meeting_clause_line, recorded_lines, session_id_line, session_started_lines,
    };
    use meethook_enroll::MeetingOffer;
    use meethook_session::{Meeting, MeetingFit};

    fn id(n: u8) -> SessionId {
        SessionId::parse(format!("20260809-05{n:02}00").as_str()).unwrap()
    }

    fn started(n: u8) -> Note {
        Note::SessionStarted {
            id: id(n),
            dir: std::path::PathBuf::from(format!("/tmp/meethook/sessions/20260809-05{n:02}00")),
            mic_rate: 48_000,
            mic_channels: 1,
            speaker_rate: 44_100,
        }
    }

    fn recorded(n: u8, meeting: Option<&Meeting>) -> Note {
        Note::Recorded {
            id: id(n),
            mic_secs: 7.5,
            speaker_secs: 7.5,
            dir: std::path::PathBuf::from(format!("/tmp/meethook/sessions/20260809-05{n:02}00")),
            meeting: meeting.map(MeetingLabel::from),
        }
    }

    /// A candidate meeting, the way the record crate's lookup would hand one over.
    fn meeting_of(event_id: &str, title: &str) -> Meeting {
        Meeting::new(
            event_id.to_owned(),
            title.to_owned(),
            "Work".to_owned(),
            "2026-08-15T10:00:00Z".parse().unwrap(),
            "2026-08-15T11:00:00Z".parse().unwrap(),
        )
    }

    /// Its projection, the way the loop projects every offer before it crosses.
    fn offer_of(event_id: &str, title: &str) -> MeetingOffer {
        MeetingOffer::from(&meeting_of(event_id, title))
    }

    /// An offer replaces the pane whole and arrives with the selector closed: whatever the
    /// predecessor showed is gone, and a restarted session cannot inherit its list or pick.
    #[test]
    fn an_offer_resets_the_pane_and_a_restart_cannot_inherit_it() {
        let mut s = State::default();
        s.apply(&started(1));
        s.apply(&Note::MeetingOffered {
            offered: vec![offer_of("EVENT-A", "Standup")],
            guess: Some(MeetingLabel::from(&meeting_of("EVENT-A", "Standup"))),
        });
        s.open_selector();
        assert!(s.selector_open);

        // The session ends: the pane dies with it, pick included.
        s.apply(&Note::Stopping);
        s.apply(&recorded(1, None));
        assert!(!s.selector_open);
        assert!(s.offered.is_empty(), "the offers die with the session");
        assert!(s.settled.is_none(), "a pick does not survive its session");

        // The next session starts clean, before its own offers have even arrived.
        s.apply(&started(2));
        assert!(
            s.offered.is_empty(),
            "no predecessor's list on a fresh session"
        );
        s.apply(&Note::MeetingOffered {
            offered: vec![offer_of("EVENT-B", "Planning")],
            guess: None,
        });
        assert_eq!(s.offered.len(), 1);
        assert_eq!(s.offered[0].event_id, "EVENT-B");
        assert!(
            !s.selector_open,
            "the offer arrives with the selector closed"
        );
    }

    /// The cursor wraps at both ends and does nothing on an empty list -- where `confirm`
    /// also answers none, so there is never a row addressed that is not one of the offers.
    #[test]
    fn the_cursor_wraps_and_an_empty_list_opens_but_confirms_nothing() {
        let mut s = State::default();
        s.apply(&started(3));
        s.apply(&Note::MeetingOffered {
            offered: vec![
                offer_of("EVENT-A", "First"),
                offer_of("EVENT-B", "Second"),
                offer_of("EVENT-C", "Third"),
            ],
            guess: None,
        });
        s.open_selector();

        assert_eq!(s.confirm(), Some("EVENT-A".to_owned()));
        s.next();
        assert_eq!(s.confirm(), Some("EVENT-B".to_owned()));
        s.next();
        s.next();
        assert_eq!(s.confirm(), Some("EVENT-A".to_owned()), "down wraps");
        s.previous();
        assert_eq!(s.confirm(), Some("EVENT-C".to_owned()), "up wraps");

        // The degraded view: no grant, empty calendar. The list still opens -- the frame
        // says nothing is offered rather than hiding the key -- and confirms nothing.
        let mut empty = State::default();
        empty.apply(&started(4));
        empty.apply(&Note::MeetingOffered {
            offered: vec![],
            guess: None,
        });
        empty.open_selector();
        assert!(empty.selector_open, "an empty list still opens");
        empty.next();
        empty.previous();
        assert_eq!(empty.cursor, 0, "movement is a no-op on an empty list");
        assert_eq!(empty.confirm(), None, "nothing offered, nothing to confirm");
    }

    /// A pick survives the round trip: `confirm` addresses the cursor, the run's settlement
    /// supersedes the guess with the Confirmed label and closes the selector, and a later
    /// pick replaces the first.
    #[test]
    fn a_pick_round_trips_through_settlement() {
        let standup = meeting_of("EVENT-A", "Standup");
        let mut s = State::default();
        s.apply(&started(5));
        s.apply(&Note::MeetingOffered {
            offered: vec![
                offer_of("EVENT-A", "Standup"),
                offer_of("EVENT-B", "Incident review"),
            ],
            guess: Some(MeetingLabel::from(&standup)),
        });
        s.open_selector();
        s.next();
        assert_eq!(s.confirm(), Some("EVENT-B".to_owned()));

        // What crosses back carries Confirmed -- what `label_by_hand` will write -- never the
        // candidate's own fit, whose caveat would qualify a pick a human just made.
        let picked = meeting_of("EVENT-B", "Incident review").with_fit(MeetingFit::Confirmed);
        s.apply(&Note::MeetingSettled {
            label: MeetingLabel::from(&picked),
        });
        assert!(!s.selector_open, "the run's settlement closes the selector");
        let settled = s.settled.as_ref().unwrap();
        assert_eq!(settled.title, "Incident review");
        assert_eq!(settled.fit, MeetingFit::Confirmed);
        assert_eq!(
            s.guess.as_ref().unwrap().title,
            "Standup",
            "the guess stays held; the pick supersedes it in display"
        );

        // A later pick replaces the earlier one.
        s.open_selector();
        s.previous();
        assert_eq!(s.confirm(), Some("EVENT-A".to_owned()));
        s.apply(&Note::MeetingSettled {
            label: MeetingLabel::from(&standup.with_fit(MeetingFit::Confirmed)),
        });
        assert_eq!(s.settled.as_ref().unwrap().title, "Standup");
    }

    /// Stopping closes the selector while keeping the pane: picks stop being accepted, but
    /// the frame still says what the session was named while it finalizes.
    #[test]
    fn stopping_closes_the_selector_but_keeps_the_pane() {
        let mut s = State::default();
        s.apply(&started(6));
        s.apply(&Note::MeetingOffered {
            offered: vec![offer_of("EVENT-A", "Standup")],
            guess: None,
        });
        s.open_selector();
        s.apply(&Note::Stopping);
        assert!(!s.selector_open, "picks stop being accepted while ending");
        assert_eq!(
            s.offered.len(),
            1,
            "the pane outlives the selector through finalization"
        );
    }

    /// Feeds a note sequence the way the loop produces one, and asserts the phase walk.
    #[test]
    fn the_frame_walks_the_phases_the_loop_walks_them() {
        let mut s = State::default();

        s.apply(&Note::Watching);
        assert_eq!(s.phase, Phase::Idle);
        assert_eq!(s.notice.as_deref(), Some(WATCHING));

        s.apply(&Note::AlreadyActive);
        assert_eq!(s.phase, Phase::Beginning);
        assert_eq!(s.notice.as_deref(), Some(ALREADY_ACTIVE));

        s.apply(&started(1));
        assert_eq!(s.phase, Phase::Recording);
        assert!(s.notice.is_none(), "a live session replaces the notice");
        assert_eq!(s.session.as_ref().unwrap().id, id(1));

        s.apply(&Note::Stopping);
        assert_eq!(s.phase, Phase::Finalizing);
        assert_eq!(s.notice.as_deref(), Some(STOPPING));

        s.apply(&recorded(1, None));
        assert_eq!(s.phase, Phase::Idle);
        assert!(s.session.is_none());
        assert_eq!(s.last.as_ref().unwrap().id, id(1));

        s.apply(&Note::Watching);
        assert_eq!(s.phase, Phase::Idle);
    }

    /// A device change or a stall finalizes the session and says why, in the composer's words.
    #[test]
    fn a_swap_says_why_in_the_composer_words() {
        for (note, words) in [
            (&Note::DeviceChanged, DEVICE_CHANGED),
            (&Note::MicStalled, MIC_STALLED),
        ] {
            let mut s = State::default();
            s.apply(&started(2));
            s.apply(note);
            assert_eq!(s.phase, Phase::Finalizing);
            assert_eq!(s.notice.as_deref(), Some(words));
        }
    }

    /// `NoNewSession` and give-up return the frame to idle with the run's own wording.
    #[test]
    fn an_empty_restart_and_a_give_up_land_back_in_idle() {
        let mut s = State::default();
        s.apply(&started(3));
        s.apply(&Note::DeviceChanged);
        s.apply(&Note::NoNewSession);
        assert_eq!(s.phase, Phase::Idle);
        assert_eq!(s.notice.as_deref(), Some(NO_NEW_SESSION));

        let mut s = State::default();
        s.apply(&Note::GivingUp(5));
        assert_eq!(s.phase, Phase::Idle);
        assert_eq!(
            s.notice.as_deref(),
            Some("Giving up on this call after 5 attempts; still watching.")
        );
    }

    /// The finished-session line carries the meeting clause through the same composer the
    /// plain run prints it with, and nothing more.
    #[test]
    fn the_last_session_carries_only_the_meeting_clause() {
        let meeting = Meeting::new(
            "EVENT-ABC".to_owned(),
            "Incident review".to_owned(),
            "Work".to_owned(),
            "2026-08-15T10:00:00Z".parse().unwrap(),
            "2026-08-15T11:00:00Z".parse().unwrap(),
        )
        .with_people(
            Some(meethook_session::Attendee {
                name: Some("Alan Turing".to_owned()),
                email: Some("alan@example.com".to_owned()),
                status: meethook_session::AttendeeStatus::Accepted,
                is_you: false,
            }),
            vec![],
        );

        let mut s = State::default();
        s.apply(&recorded(4, Some(&meeting)));
        let last = s.last.as_ref().unwrap();
        let label = last.meeting.as_ref().unwrap();
        assert_eq!(label.title, "Incident review");
        // The clause is what render prints; the rest of the meeting is not in the state at all.
        assert!(meeting_clause_line(label).contains("Incident review"));
        assert!(!format!("{label:?}").contains("Turing"));
    }

    /// The narration buffer is the composed stdout-class text, in order, and only that:
    /// stderr-class notes and debug notes never enter it.
    #[test]
    fn the_narration_is_the_plain_run_byte_for_byte() {
        let mut s = State::default();
        s.apply(&Note::Watching);
        s.apply(&started(5));
        s.apply(&Note::BeginFailed("could not open the input".to_string()));
        s.apply(&Note::ActivityDebug("[activity] frames 0".to_string()));
        s.apply(&Note::Stopping);
        s.apply(&recorded(5, None));

        let dir = "/tmp/meethook/sessions/20260809-050500";
        assert!(!s.narration.contains("[activity]"));
        assert!(!s.narration.contains("could not open the input"));
        assert_eq!(
            s.take_narration(),
            format!(
                "{WATCHING}\n{}{STOPPING}\n{}",
                session_started_lines(id(5), dir, 48_000, 1, 44_100),
                recorded_lines(id(5), 7.5, 7.5, dir, None)
            )
        );
        assert!(
            s.trouble
                .iter()
                .any(|t| t.contains("could not open the input"))
        );
    }

    /// Trouble is held, in order, and says nothing the plain run would not have said.
    #[test]
    fn trouble_is_held_in_order_until_teardown() {
        let mut s = State::default();
        s.apply(&Note::BeginFailed("first failure".to_string()));
        s.apply(&Note::FinishFailed("second failure".to_string()));
        assert_eq!(s.trouble, ["first failure\n", "second failure\n"]);
    }

    /// The active-session pane shows the lines the plain run printed, so a rate the run
    /// reported is the rate the frame shows.
    #[test]
    fn the_active_session_keeps_the_measured_rates() {
        let mut s = State::default();
        s.apply(&started(6));
        let session = s.session.as_ref().unwrap();
        assert_eq!(session.mic_rate, 48_000);
        assert_eq!(session.mic_channels, 1);
        assert_eq!(session.speaker_rate, 44_100);
        assert!(session_id_line(&session.id).starts_with("Session 20260809-050600"));
    }
}
