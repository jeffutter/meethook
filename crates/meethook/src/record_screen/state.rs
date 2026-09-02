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
use meethook_session::{Attendee, Paths, RosterEdit, SessionId};

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

/// Which field of the selected roster row is being corrected inline.
///
/// A typed command rather than a key event: the shell's key map decides which letter means
/// which field, and this type is what the state machine holds while the correction is open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditingField {
    Name,
    Email,
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
    /// The attached meeting's roster, as the run showed it: the roster pane lists these rows,
    /// and every committed correction rides back to the run addressed by `roster_event_id`.
    /// `None` when no meeting is attached -- there is then no roster to show, and the key
    /// that opens the pane does nothing rather than opening an empty one. Replaced wholesale
    /// by each [`crate::record::Note::RosterAttached`], so a restarted session cannot inherit
    /// its predecessor's people.
    pub roster: Option<Vec<Attendee>>,
    /// The calendar event the roster belongs to: the address every correction rides back on.
    /// Set together with `roster`, never apart.
    pub roster_event_id: Option<String>,
    /// Whether the roster pane is open: the attached meeting's attendees replace the notice
    /// region while it is, exactly as the selector does. Mutually exclusive with the
    /// selector -- one context at a time keeps the key map total without focus bookkeeping,
    /// and makes the privacy scoping crisp (the selector's rows and the roster's rows are
    /// never painted together).
    pub roster_open: bool,
    /// The row under the cursor within `roster`. Reset with the pane rather than trusted
    /// across sessions.
    pub roster_cursor: usize,
    /// The field being corrected inline, if any: entered from the roster pane, where it is
    /// the innermost context -- printables feed the buffer, Enter commits, Esc cancels.
    pub editing: Option<EditingField>,
    /// The text typed into the field under correction. Crosses nowhere until committed.
    pub edit_buffer: String,
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
            roster: None,
            roster_event_id: None,
            roster_open: false,
            roster_cursor: 0,
            editing: None,
            edit_buffer: String::new(),
        }
    }

    /// The meeting pane dies with the session it described: a new session starting and a
    /// session finishing both call it, rather than spelling the assignments twice.
    fn clear_meeting_pane(&mut self) {
        self.offered.clear();
        self.guess = None;
        self.settled = None;
        self.selector_open = false;
        self.cursor = 0;
        self.roster = None;
        self.roster_event_id = None;
        self.roster_open = false;
        self.roster_cursor = 0;
        self.editing = None;
        self.edit_buffer.clear();
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
            Note::RosterAttached {
                event_id,
                attendees,
            } => {
                // The pane's copy is replaced, not merged: a pick settling a different
                // meeting supersedes the roster the guess first rode in with, and a
                // restarted session starts clean rather than inheriting its predecessor's
                // people. The pane itself stays closed -- the run showing a roster is not
                // the user opening it.
                self.roster = Some(attendees.clone());
                self.roster_event_id = Some(event_id.clone());
                self.roster_open = false;
                self.roster_cursor = 0;
                self.editing = None;
                self.edit_buffer.clear();
            }
            Note::DeviceChanged | Note::MicStalled | Note::Stopping => {
                self.phase = Phase::Finalizing;
                // The session is ending, so picks and edits stop being accepted; the meeting
                // pane itself stays up -- the user should still see what the session was
                // named while it finalizes. The roster PANE, unlike the selector's list, IS
                // the thing being shown, so it closes to make room for the notice, and an
                // in-flight correction is cancelled along with it.
                self.selector_open = false;
                self.roster_open = false;
                self.editing = None;
                self.edit_buffer.clear();
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
    ///
    /// Closes the roster pane: the two panes share the notice region and are mutually
    /// exclusive -- unreachable through the key map today (Enter is unbound while the roster
    /// is open), but the state machine keeps the invariant total rather than relying on the
    /// table above it, and an in-flight correction dies with the pane.
    pub fn open_selector(&mut self) {
        if self.phase == Phase::Recording {
            self.selector_open = true;
            self.roster_open = false;
            self.editing = None;
            self.edit_buffer.clear();
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

    /// Opens the roster pane while a session is recording and a meeting is attached.
    ///
    /// Requires an attachment: unlike the selector, whose degraded view is "no meetings",
    /// there is no roster to degrade to when no meeting is attached, so the key does nothing
    /// rather than opening an empty pane -- which is also why the footer advertises it only
    /// while a roster is on screen. Closes the selector: the two panes share the notice
    /// region, and one context at a time is what keeps the key map total.
    ///
    /// Does not reset the cursor: the note that replaces the roster resets it, and a user
    /// who scrolls, closes with Esc and reopens is mid-list, not at its top.
    pub fn open_roster(&mut self) {
        if self.phase == Phase::Recording && self.roster.is_some() {
            self.selector_open = false;
            self.roster_open = true;
        }
    }

    /// Moves the roster cursor down one row, wrapping. No-op on an empty roster: there is
    /// nothing to move through, and the frame keeps saying so.
    pub fn roster_next(&mut self) {
        if let Some(roster) = &self.roster
            && !roster.is_empty()
        {
            self.roster_cursor = (self.roster_cursor + 1) % roster.len();
        }
    }

    /// Moves the roster cursor up one row, wrapping. No-op on an empty roster.
    ///
    /// `(cursor + len - 1) % len` rather than `wrapping_sub(1) % len`: the underflow route
    /// computes `usize::MAX % len`, which lands back on zero for lengths that divide it --
    /// silently a no-op where a wrap was owed. Same argument as [`State::previous`].
    pub fn roster_previous(&mut self) {
        if let Some(roster) = &self.roster
            && !roster.is_empty()
        {
            self.roster_cursor = (self.roster_cursor + roster.len() - 1) % roster.len();
        }
    }

    /// Removes the attendee under the cursor and hands back the full edited roster,
    /// addressed by the meeting it came from -- or none when there is no roster to remove
    /// from.
    ///
    /// The whole roster crosses, never the removed row: last write wins, and the run applies
    /// whatever it was last given at the single finalize point. Removal commits immediately --
    /// there is no cancel for it -- so the cursor clamps to the shortened list.
    pub fn remove_selected(&mut self) -> Option<RosterEdit> {
        if self.roster.as_ref()?.is_empty() {
            return None;
        }
        // Cloned before the roster is touched: `roster` and `roster_event_id` are always set
        // together (§ their doc comments), so this cannot fail in practice, but failing here
        // keeps a would-be violation from mutating local state while still reporting nothing
        // to cross.
        let event_id = self.roster_event_id.clone()?;
        let roster = self.roster.as_mut()?;
        roster.remove(self.roster_cursor.min(roster.len() - 1));
        self.roster_cursor = self.roster_cursor.min(roster.len().saturating_sub(1));
        Some(RosterEdit {
            event_id,
            attendees: roster.clone(),
        })
    }

    /// Enters inline correction of the selected row's `field`: the buffer starts empty, and
    /// committing replaces the field wholesale -- so a cleared buffer clears the field.
    /// Does nothing when there is no row to correct.
    pub fn begin_edit(&mut self, field: EditingField) {
        if self
            .roster
            .as_ref()
            .and_then(|r| r.get(self.roster_cursor))
            .is_some()
        {
            self.editing = Some(field);
            self.edit_buffer.clear();
        }
    }

    /// Feeds one character into the field under correction.
    ///
    /// Only the shell's key map calls this, and it feeds non-control characters only -- the
    /// same guard enroll's search input applies to its filter -- so a stray control byte can
    /// never land in the text a correction will write.
    pub fn feed_edit(&mut self, c: char) {
        if self.editing.is_some() {
            self.edit_buffer.push(c);
        }
    }

    /// Deletes the last character of the field under correction.
    pub fn backspace_edit(&mut self) {
        if self.editing.is_some() {
            self.edit_buffer.pop();
        }
    }

    /// Commits the field under correction: the buffer replaces the selected row's field
    /// (empty buffer clears it), editing exits, and the full edited roster is handed back
    /// addressed by the meeting it came from.
    ///
    /// Like [`State::remove_selected`], the whole roster crosses -- the run applies the last
    /// committed state at the single finalize point, so the frame's copy and the run's stash
    /// stay the same value by construction.
    pub fn commit_edit(&mut self) -> Option<RosterEdit> {
        let field = self.editing?;
        // Cloned before the roster is touched: see the matching comment in `remove_selected`.
        let event_id = self.roster_event_id.clone()?;
        let buffer = std::mem::take(&mut self.edit_buffer);
        let text = (!buffer.is_empty()).then_some(buffer);
        let roster = self.roster.as_mut()?;
        let row = roster.get_mut(self.roster_cursor)?;
        match field {
            EditingField::Name => row.name = text,
            EditingField::Email => row.email = text,
        }
        self.editing = None;
        Some(RosterEdit {
            event_id,
            attendees: roster.clone(),
        })
    }

    /// Cancels the field under correction: the buffer is dropped and the row is untouched.
    ///
    /// Crosses nothing -- the run's stash keeps the pre-edit roster, so a cancel cannot
    /// desync the frame's copy from what will be written.
    pub fn cancel_edit(&mut self) {
        self.editing = None;
        self.edit_buffer.clear();
    }

    /// Closes the roster pane without committing anything: the toggle, the escape, or the
    /// session ending underneath it. Any in-flight correction is cancelled along with it.
    pub fn close_roster(&mut self) {
        self.roster_open = false;
        self.editing = None;
        self.edit_buffer.clear();
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
    use meethook_session::{Attendee, AttendeeStatus, Meeting, MeetingFit};

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

    /// The attached meeting's roster, the way the loop sends it: the disclosure unit per
    /// attendee -- name, email, status, `is_you` -- addressed by the event id.
    fn roster_of(event_id: &str) -> Note {
        Note::RosterAttached {
            event_id: event_id.to_owned(),
            attendees: vec![
                Attendee {
                    name: Some("Alan Turing".to_owned()),
                    email: Some("alan@example.com".to_owned()),
                    status: AttendeeStatus::Accepted,
                    is_you: false,
                },
                Attendee {
                    name: Some("Grace Hopper".to_owned()),
                    email: Some("grace@example.com".to_owned()),
                    status: AttendeeStatus::Declined,
                    is_you: true,
                },
            ],
        }
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

    /// The roster pane opens only while recording with an attached meeting, and it and the
    /// selector never share the notice region: opening one closes the other, both ways.
    #[test]
    fn the_roster_pane_requires_an_attachment_and_excludes_the_selector() {
        // No attachment: the key does nothing rather than opening an empty pane.
        let mut s = State::default();
        s.apply(&started(20));
        s.open_roster();
        assert!(!s.roster_open, "no attachment, no pane");

        // No session at all: there is nothing to correct.
        let mut idle = State::default();
        idle.apply(&roster_of("EVENT-A"));
        idle.open_roster();
        assert!(!idle.roster_open, "no session, no pane");

        // With a roster: it opens, and closes the selector if one was open.
        let mut s = State::default();
        s.apply(&started(21));
        s.apply(&Note::MeetingOffered {
            offered: vec![offer_of("EVENT-A", "Standup")],
            guess: None,
        });
        s.apply(&roster_of("EVENT-A"));
        s.open_selector();
        s.open_roster();
        assert!(s.roster_open);
        assert!(!s.selector_open, "opening the roster closes the selector");

        // And back the other way.
        s.open_selector();
        assert!(s.selector_open);
        assert!(!s.roster_open, "opening the selector closes the roster");
    }

    /// An attached meeting whose invite lists nobody still opens the pane -- the frame says
    /// so rather than hiding the key -- and movement and removal are no-ops on the empty
    /// list, mirroring the empty-selector rule.
    #[test]
    fn an_empty_roster_still_opens_but_moves_and_removes_nothing() {
        let mut s = State::default();
        s.apply(&started(22));
        s.apply(&Note::RosterAttached {
            event_id: "EVENT-A".to_owned(),
            attendees: vec![],
        });
        s.open_roster();
        assert!(s.roster_open, "an empty roster still opens");
        s.roster_next();
        s.roster_previous();
        assert_eq!(s.roster_cursor, 0, "movement is a no-op on an empty roster");
        assert_eq!(s.remove_selected(), None, "nothing to remove");
        s.begin_edit(EditingField::Name);
        assert!(s.editing.is_none(), "no row to correct");
        assert_eq!(s.commit_edit(), None, "nothing to commit");
    }

    /// The roster cursor wraps at both ends, like the selector's.
    #[test]
    fn the_roster_cursor_wraps_at_both_ends() {
        let mut s = State::default();
        s.apply(&started(23));
        s.apply(&roster_of("EVENT-A"));
        s.open_roster();
        s.roster_next();
        assert_eq!(s.roster_cursor, 1);
        s.roster_next();
        assert_eq!(s.roster_cursor, 0, "down wraps");
        s.roster_previous();
        assert_eq!(s.roster_cursor, 1, "up wraps");
    }

    /// Every committed local change crosses exactly once, as the full edited roster: a
    /// removal drops a row and clamps the cursor, a committed correction rewrites the field
    /// wholesale (an empty buffer clears it), and a cancelled correction crosses nothing.
    #[test]
    fn committed_changes_cross_the_full_roster_once_each() {
        let mut s = State::default();
        s.apply(&started(24));
        s.apply(&roster_of("EVENT-A"));
        s.open_roster();

        // Removal: the full roster comes back minus the row, and the cursor clamps.
        let removed = s.remove_selected().expect("a removal crosses");
        assert_eq!(removed.event_id, "EVENT-A");
        assert_eq!(removed.attendees.len(), 1);
        assert_eq!(removed.attendees[0].name.as_deref(), Some("Grace Hopper"));
        assert_eq!(
            s.roster_cursor, 0,
            "the cursor clamps to the shortened list"
        );

        // Name correction: typed text replaces the field wholesale.
        s.begin_edit(EditingField::Name);
        for c in "Grace M. Hopper".chars() {
            s.feed_edit(c);
        }
        let committed = s.commit_edit().expect("a commit crosses");
        assert_eq!(committed.event_id, "EVENT-A");
        assert_eq!(
            committed.attendees[0].name.as_deref(),
            Some("Grace M. Hopper"),
            "the commit rewrites the field"
        );
        assert!(s.editing.is_none(), "editing exits on commit");
        assert_eq!(s.edit_buffer, "");

        // A cleared buffer clears the field: committing empty removes it.
        s.begin_edit(EditingField::Email);
        let cleared = s.commit_edit().expect("an empty commit crosses");
        assert_eq!(
            cleared.attendees[0].email.as_ref(),
            None,
            "an empty buffer clears the field"
        );

        // Cancel: the row is untouched and nothing crossed.
        s.begin_edit(EditingField::Name);
        s.feed_edit('x');
        s.cancel_edit();
        assert!(s.editing.is_none());
        assert_eq!(s.edit_buffer, "");
        assert_eq!(
            s.roster.as_ref().unwrap()[0].name.as_deref(),
            Some("Grace M. Hopper"),
            "a cancel reverts locally"
        );
    }

    /// `roster` and `roster_event_id` are documented to move together; this pins the fallback
    /// if that invariant were ever violated elsewhere: neither commit path mutates the roster
    /// before it has confirmed it has an id to address the edit with, so a violation reports
    /// nothing to cross rather than silently dropping a local mutation on the floor.
    #[test]
    fn a_missing_roster_event_id_leaves_the_roster_untouched() {
        let mut s = State::default();
        s.apply(&started(30));
        s.apply(&roster_of("EVENT-A"));
        s.open_roster();
        s.roster_event_id = None;

        assert_eq!(s.remove_selected(), None, "no id to address a removal with");
        assert_eq!(
            s.roster.as_ref().unwrap().len(),
            2,
            "the roster was not touched"
        );

        s.begin_edit(EditingField::Name);
        s.feed_edit('x');
        assert_eq!(s.commit_edit(), None, "no id to address a commit with");
        assert_eq!(
            s.roster.as_ref().unwrap()[0].name.as_deref(),
            Some("Alan Turing"),
            "the row was not touched"
        );
    }

    /// The roster copy is replaced wholesale by each note, and dies with the session it
    /// described: a settled pick supersedes it, and a finished session wipes it -- so a
    /// restarted session cannot inherit its predecessor's people.
    #[test]
    fn a_roster_note_replaces_the_pane_whole_and_a_restart_cannot_inherit_it() {
        let mut s = State::default();
        s.apply(&started(25));
        s.apply(&roster_of("EVENT-A"));
        s.open_roster();
        s.begin_edit(EditingField::Name);
        s.feed_edit('x');

        // A note for a different meeting supersedes the copy wholesale...
        s.apply(&Note::RosterAttached {
            event_id: "EVENT-B".to_owned(),
            attendees: vec![],
        });
        assert_eq!(s.roster_event_id.as_deref(), Some("EVENT-B"));
        assert_eq!(s.roster.as_ref().unwrap().len(), 0);
        assert!(
            !s.roster_open,
            "the replacement arrives with the pane closed"
        );
        assert!(
            s.editing.is_none(),
            "an in-flight correction dies with the copy"
        );

        // ...and the session ending wipes it entirely.
        s.apply(&Note::Stopping);
        s.apply(&recorded(25, None));
        assert!(s.roster.is_none(), "the roster dies with the session");
        assert!(s.roster_event_id.is_none());
    }

    /// The relaxation's boundary, asserted at the state level: a stuffed roster note drives
    /// the buffers, and neither narration nor trouble carries any of its secrets. The note
    /// composes to nothing, and the pane's rows live only in fields the frame renders while
    /// the alternate screen is active -- the dial-in-PIN incident is why this must hold.
    #[test]
    fn a_roster_note_never_reaches_the_narration_or_trouble() {
        let mut s = State::default();
        s.apply(&started(26));
        s.apply(&roster_of("EVENT-ABC"));
        s.open_roster();

        assert_eq!(
            roster_of("EVENT-ABC").composed(),
            "",
            "the roster note composes to nothing"
        );
        let narration = s.take_narration();
        for secret in ["Turing", "Hopper", "@example.com"] {
            assert!(
                !narration.contains(secret),
                "the narration leaks {secret:?}"
            );
            assert!(
                !s.trouble.iter().any(|t| t.contains(secret)),
                "trouble leaks {secret:?}"
            );
        }
    }

    /// Esc closes the selector without picking: unlike the run's own settlement, which only
    /// closes it on a pick that stuck, a direct close leaves nothing to confirm -- though the
    /// list itself survives the close, so reopening still offers the same row.
    #[test]
    fn closing_the_selector_leaves_nothing_to_confirm() {
        let mut s = State::default();
        s.apply(&started(31));
        s.apply(&Note::MeetingOffered {
            offered: vec![offer_of("EVENT-A", "Standup")],
            guess: None,
        });
        s.open_selector();
        assert!(s.selector_open);

        s.close_selector();
        assert!(!s.selector_open);
        s.open_selector();
        assert_eq!(
            s.confirm(),
            Some("EVENT-A".to_owned()),
            "the list survives a close"
        );
    }

    /// Closing the roster pane cancels whatever correction was in flight, the same as a
    /// session ending underneath it: the buffer and the editing field both clear.
    #[test]
    fn closing_the_roster_pane_cancels_an_in_flight_correction() {
        let mut s = State::default();
        s.apply(&started(32));
        s.apply(&roster_of("EVENT-A"));
        s.open_roster();
        s.begin_edit(EditingField::Name);
        s.feed_edit('x');

        s.close_roster();
        assert!(!s.roster_open);
        assert!(
            s.editing.is_none(),
            "an in-flight correction dies with the pane"
        );
        assert_eq!(s.edit_buffer, "");
    }

    /// Backspace deletes the last character of the field under correction, and is a no-op
    /// outside an edit or on an already-empty buffer -- never a panic.
    #[test]
    fn backspace_deletes_the_last_character_of_the_field_under_correction() {
        let mut s = State::default();
        s.apply(&started(33));
        s.apply(&roster_of("EVENT-A"));
        s.open_roster();

        // No-op outside an edit.
        s.backspace_edit();
        assert_eq!(s.edit_buffer, "");

        s.begin_edit(EditingField::Name);
        for c in "Ada".chars() {
            s.feed_edit(c);
        }
        s.backspace_edit();
        assert_eq!(s.edit_buffer, "Ad");

        // A no-op on an already-empty buffer.
        s.edit_buffer.clear();
        s.backspace_edit();
        assert_eq!(s.edit_buffer, "");
    }
}
