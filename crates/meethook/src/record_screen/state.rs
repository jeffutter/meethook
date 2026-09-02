//! The record frame as a pure state machine.
//!
//! What the frame shows is derived from the same typed notes the plain run prints: this type
//! consumes a sequence of [`crate::record::Note`]s and holds whatever the frame needs to draw
//! -- which phase the run is in, which session is live, what the last finish produced, what it
//! should say -- and nothing else. No clock in it (elapsed time is the shell's, because it is
//! derived from one), no terminal in it, no audio in it. That is what lets the whole frame be
//! exercised in `cargo test` by feeding it note sequences, the way `enroll`'s screen state is
//! exercised by feeding it answers.
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

use meethook_enroll::MeetingLabel;
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
        }
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
            }
            Note::DeviceChanged | Note::MicStalled | Note::Stopping => {
                self.phase = Phase::Finalizing;
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
    use meethook_session::Meeting;

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
