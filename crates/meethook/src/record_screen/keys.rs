//! The record frame's key table: one [`Keymap`] that drives dispatch, the hint footer and the `?`
//! help overlay, so a key cannot be bound but missing from the help, or advertised but dead.
//!
//! Four contexts, one at a time, innermost first: editing a roster field beats the roster pane,
//! which beats the meeting selector, which beats the base. The panes are mutually exclusive in
//! the state machine and editing only ever begins from the roster pane, so [`Keys::context`] is
//! total without focus bookkeeping.
//!
//! Two rules are load-bearing and are what the tests pin:
//!
//! - Ctrl-C and Ctrl-D are bound first in *every* context, editing and help included, and are
//!   never disabled: the run must always be leavable.
//! - `s` (end this session) is not bound in [`EDITING`], where printables belong to the field
//!   being corrected, and is enabled only while a session is recording. Typing a name must never
//!   end a call.

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::state::{EditingField, State};
use crate::tui_kit::{Binding, Context, HelpRow, Hint, KeySpec, Keymap, Theme};

pub const BASE: Context = Context("base");
pub const SELECTOR: Context = Context("selector");
pub const ROSTER: Context = Context("roster");
pub const EDITING: Context = Context("editing");

/// What a keypress means to the frame.
///
/// Two of these leave the frame: leaving the run (`Interrupt`) and ending the session that is
/// recording while the run keeps watching (`StopSession`). The rest is the meeting selector and
/// the roster correction, and they grow this table rather than the loop that reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Stop the run and exit: the main loop finalizes the session exactly as a plain-mode
    /// Ctrl-C would have.
    Interrupt,
    /// End the session that is recording right now and keep the run watching for the next call:
    /// crosses as an `Event::StopSession`. Deliberately not `Interrupt` - "end this recording"
    /// and "quit" are two different questions.
    StopSession,
    /// Open the meeting selector. Allowed even with an empty list: that is the degraded view,
    /// and the frame says nothing is offered rather than hiding the key.
    OpenSelector,
    /// Move the selector cursor down one row, wrapping.
    Next,
    /// Move the selector cursor up one row, wrapping.
    Previous,
    /// Jump the selector cursor to the first row.
    Top,
    /// Jump the selector cursor to the last row.
    Bottom,
    /// Confirm the offer under the cursor: its identifier crosses into the run as an
    /// `Event::MeetingPicked`. The selector stays open until the run's settlement says the
    /// pick stuck.
    Confirm,
    /// Close the selector without picking.
    CloseSelector,
    /// Remove the meeting the session is attached to: crosses as an `Event::MeetingCleared`.
    ClearMeeting,
    /// Open the roster pane: the attached meeting's attendees replace the notice region.
    OpenRoster,
    /// Close the roster pane without committing anything; any in-flight correction dies with
    /// it.
    CloseRoster,
    /// Move the roster cursor down one row, wrapping.
    RosterNext,
    /// Move the roster cursor up one row, wrapping.
    RosterPrevious,
    /// Jump the roster cursor to the first row.
    RosterTop,
    /// Jump the roster cursor to the last row.
    RosterBottom,
    /// Remove the attendee under the cursor: the full edited roster crosses as an
    /// `Event::RosterEdited`.
    RemoveAttendee,
    /// Begin correcting the selected row's name inline.
    EditName,
    /// Begin correcting the selected row's email inline.
    EditEmail,
    /// Feed one character into the field under correction. Produced only in [`EDITING`], by the
    /// printable fallback in [`Keys::dispatch`]; it is not a table binding.
    Type(char),
    /// Delete the last character of the field under correction.
    DeleteChar,
    /// Commit the field under correction: the full edited roster crosses as an
    /// `Event::RosterEdited`.
    CommitField,
    /// Cancel the field under correction: the buffer is dropped locally and nothing crosses.
    CancelField,
    /// Open the `?` overlay.
    Help,
}

fn b(
    context: Context,
    key: KeySpec,
    action: Action,
    hint: &'static str,
    help: &'static str,
) -> Binding<Action> {
    Binding {
        context,
        key,
        action,
        hint,
        help,
    }
}

fn table() -> Vec<Binding<Action>> {
    use Action as A;
    let ch = KeySpec::ch;
    let key = KeySpec::key;
    let ctrl = KeySpec::ctrl;

    let mut t = Vec::new();
    for context in [BASE, SELECTOR, ROSTER, EDITING] {
        // First in every context so nothing can shadow it. A real terminal delivers the control
        // byte lowercase, but the pre-kit handler read either case, so both are kept.
        t.extend([
            b(
                context,
                ctrl('c'),
                A::Interrupt,
                "stop and exit",
                "Stop and exit",
            ),
            b(
                context,
                ctrl('d'),
                A::Interrupt,
                "",
                "Stop and exit (alias)",
            ),
            b(context, ctrl('C'), A::Interrupt, "", ""),
            b(context, ctrl('D'), A::Interrupt, "", ""),
        ]);
    }

    let (n, s, r, e) = (BASE, SELECTOR, ROSTER, EDITING);
    t.extend([
        // Base: the cursor a later selector open will land on moves here too.
        b(
            n,
            key(KeyCode::Enter),
            A::OpenSelector,
            "choose a meeting",
            "Open the meeting list",
        ),
        b(n, ch('j'), A::Next, "", "Move the cursor down"),
        b(
            n,
            key(KeyCode::Down),
            A::Next,
            "",
            "Move the cursor down (alias)",
        ),
        b(n, ch('k'), A::Previous, "", "Move the cursor up"),
        b(
            n,
            key(KeyCode::Up),
            A::Previous,
            "",
            "Move the cursor up (alias)",
        ),
        b(n, ch('g'), A::Top, "", "First row"),
        b(n, ch('G'), A::Bottom, "", "Last row"),
        b(
            n,
            ch('r'),
            A::OpenRoster,
            "roster",
            "Open the attendee roster",
        ),
        b(
            n,
            ch('c'),
            A::ClearMeeting,
            "clear meeting",
            "Remove the meeting from this session",
        ),
        b(
            n,
            ch('s'),
            A::StopSession,
            "end this session",
            "End this session and keep watching",
        ),
        b(n, ch('?'), A::Help, "help", "Show this help"),
        // Selector.
        b(s, ch('j'), A::Next, "down", "Move down"),
        b(s, key(KeyCode::Down), A::Next, "", "Move down (alias)"),
        b(s, ch('k'), A::Previous, "up", "Move up"),
        b(s, key(KeyCode::Up), A::Previous, "", "Move up (alias)"),
        b(s, ch('g'), A::Top, "", "First row"),
        b(s, ch('G'), A::Bottom, "", "Last row"),
        b(
            s,
            key(KeyCode::Enter),
            A::Confirm,
            "choose",
            "Choose the meeting under the cursor",
        ),
        b(
            s,
            key(KeyCode::Esc),
            A::CloseSelector,
            "back",
            "Close the list",
        ),
        b(
            s,
            ch('c'),
            A::ClearMeeting,
            "clear meeting",
            "Remove the meeting from this session",
        ),
        b(
            s,
            ch('s'),
            A::StopSession,
            "end this session",
            "End this session and keep watching",
        ),
        b(s, ch('?'), A::Help, "help", "Show this help"),
        // Roster pane.
        b(r, ch('j'), A::RosterNext, "down", "Move down"),
        b(
            r,
            key(KeyCode::Down),
            A::RosterNext,
            "",
            "Move down (alias)",
        ),
        b(r, ch('k'), A::RosterPrevious, "up", "Move up"),
        b(
            r,
            key(KeyCode::Up),
            A::RosterPrevious,
            "",
            "Move up (alias)",
        ),
        b(r, ch('g'), A::RosterTop, "", "First row"),
        b(r, ch('G'), A::RosterBottom, "", "Last row"),
        b(
            r,
            ch('x'),
            A::RemoveAttendee,
            "remove",
            "Remove the attendee under the cursor",
        ),
        b(
            r,
            ch('n'),
            A::EditName,
            "name",
            "Correct the attendee's name",
        ),
        b(
            r,
            ch('e'),
            A::EditEmail,
            "email",
            "Correct the attendee's email",
        ),
        b(
            r,
            key(KeyCode::Esc),
            A::CloseRoster,
            "back",
            "Close the roster",
        ),
        b(r, ch('r'), A::CloseRoster, "", "Close the roster (alias)"),
        b(
            r,
            ch('s'),
            A::StopSession,
            "end this session",
            "End this session and keep watching",
        ),
        b(r, ch('?'), A::Help, "help", "Show this help"),
        // Editing: no letter is bound, so every printable is typed text (see `Keys::dispatch`).
        b(
            e,
            key(KeyCode::Enter),
            A::CommitField,
            "save",
            "Save the field",
        ),
        b(
            e,
            key(KeyCode::Esc),
            A::CancelField,
            "cancel",
            "Discard the change",
        ),
        b(
            e,
            key(KeyCode::Backspace),
            A::DeleteChar,
            "",
            "Delete a character",
        ),
    ]);
    t
}

/// Whether `binding` is true of this state right now. A disabled binding is neither dispatched
/// nor advertised, so the footer never offers a key that would do nothing.
fn live(state: &State, binding: &Binding<Action>) -> bool {
    use super::state::Phase;
    let roster_rows = state.roster.as_ref().is_some_and(|r| !r.is_empty());
    match binding.action {
        Action::StopSession => state.can_stop_session() && state.editing.is_none(),
        Action::ClearMeeting => state.can_clear_meeting(),
        Action::OpenSelector => state.phase == Phase::Recording,
        Action::OpenRoster => state.phase == Phase::Recording && state.roster.is_some(),
        Action::RemoveAttendee | Action::EditName | Action::EditEmail => roster_rows,
        _ => true,
    }
}

pub struct Keys {
    keymap: Keymap<Action>,
    pub theme: Theme,
}

impl Keys {
    pub fn new(theme: Theme) -> Keys {
        Keys {
            keymap: Keymap::new(table()),
            theme,
        }
    }

    /// One context at a time, innermost first.
    pub fn context(state: &State) -> Context {
        match state.editing {
            Some(_) => EDITING,
            None if state.roster_open => ROSTER,
            None if state.selector_open => SELECTOR,
            None => BASE,
        }
    }

    /// The context's name for the header.
    pub fn context_name(state: &State) -> &'static str {
        match (state.editing, Self::context(state)) {
            (Some(EditingField::Name), _) => "editing name",
            (Some(EditingField::Email), _) => "editing email",
            (None, c) if c == ROSTER => "roster",
            (None, c) if c == SELECTOR => "meeting list",
            (None, _) => "overview",
        }
    }

    /// The action this key means for this state, if any. While a field is being corrected an
    /// unbound printable is the typed text.
    pub fn dispatch(&self, state: &State, key: &KeyEvent) -> Option<Action> {
        if key.kind == KeyEventKind::Release {
            return None;
        }
        let ctx = Self::context(state);
        if let Some(action) = self.keymap.dispatch(ctx, &|b| live(state, b), key) {
            return Some(action);
        }
        match key.code {
            KeyCode::Char(c)
                if ctx == EDITING
                    && !c.is_control()
                    && !key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                Some(Action::Type(c))
            }
            _ => None,
        }
    }

    pub fn hints(&self, state: &State) -> Vec<Hint> {
        self.keymap.hints(Self::context(state), &|b| live(state, b))
    }

    pub fn help_rows(&self, state: &State) -> Vec<HelpRow> {
        self.keymap
            .help_rows(Self::context(state), &|b| live(state, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Note;
    use meethook_enroll::{MeetingLabel, MeetingOffer};
    use meethook_session::{Attendee, AttendeeStatus, Meeting, MeetingFit, SessionId};
    use ratatui::crossterm::event::KeyEventState;

    fn keys() -> Keys {
        Keys::new(Theme::default())
    }

    fn press(code: KeyCode, kind: KeyEventKind, ctrl: bool) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: if ctrl {
                KeyModifiers::CONTROL
            } else {
                KeyModifiers::NONE
            },
            kind,
            state: KeyEventState::NONE,
        }
    }

    fn tap(state: &State, code: KeyCode) -> Option<Action> {
        keys().dispatch(state, &press(code, KeyEventKind::Press, false))
    }

    fn chr(state: &State, c: char) -> Option<Action> {
        tap(state, KeyCode::Char(c))
    }

    fn started() -> Note {
        Note::SessionStarted {
            id: SessionId::parse("20260809-050700").unwrap(),
            dir: std::path::PathBuf::from("/tmp/meethook/sessions/20260809-050700"),
            mic_rate: 48_000,
            mic_channels: 1,
            speaker_rate: 44_100,
        }
    }

    fn meeting(id: &str, title: &str) -> Meeting {
        Meeting::new(
            id.to_owned(),
            title.to_owned(),
            "Work".to_owned(),
            "2026-08-15T10:00:00Z".parse().unwrap(),
            "2026-08-15T11:00:00Z".parse().unwrap(),
        )
        .with_fit(MeetingFit::Started)
    }

    fn attendee(name: &str) -> Attendee {
        Attendee {
            name: Some(name.to_owned()),
            email: Some(format!("{name}@example.com")),
            status: AttendeeStatus::Accepted,
            is_you: false,
        }
    }

    /// A recording session with two offers and a three-row roster, in the base context.
    fn recording() -> State {
        let a = meeting("EVENT-A", "Standup");
        let b = meeting("EVENT-B", "Planning");
        let mut s = State::default();
        s.apply(&started());
        s.apply(&Note::MeetingOffered {
            offered: vec![MeetingOffer::from(&a), MeetingOffer::from(&b)],
            guess: Some(MeetingLabel::from(&a)),
        });
        s.apply(&Note::RosterAttached {
            event_id: "EVENT-A".to_owned(),
            attendees: vec![attendee("ann"), attendee("bob"), attendee("cy")],
        });
        s
    }

    fn selector() -> State {
        let mut s = recording();
        s.open_selector();
        s
    }

    fn roster() -> State {
        let mut s = recording();
        s.open_roster();
        s
    }

    fn editing(field: EditingField) -> State {
        let mut s = roster();
        s.begin_edit(field);
        s
    }

    fn idle() -> State {
        let mut s = State::default();
        s.apply(&Note::Watching);
        s
    }

    /// Ctrl-C and Ctrl-D interrupt in every context, editing and the open help overlay's
    /// precondition included, and only a press of them counts.
    #[test]
    fn ctrl_c_and_ctrl_d_interrupt_in_every_context() {
        let states = [
            ("idle", idle()),
            ("base", recording()),
            ("selector", selector()),
            ("roster", roster()),
            ("editing name", editing(EditingField::Name)),
            ("editing email", editing(EditingField::Email)),
        ];
        for (name, state) in &states {
            for letter in ['c', 'C', 'd', 'D'] {
                assert_eq!(
                    keys().dispatch(
                        state,
                        &press(KeyCode::Char(letter), KeyEventKind::Press, true)
                    ),
                    Some(Action::Interrupt),
                    "Ctrl-{letter} in {name}"
                );
                assert_eq!(
                    keys().dispatch(
                        state,
                        &press(KeyCode::Char(letter), KeyEventKind::Release, true)
                    ),
                    None,
                    "a release of Ctrl-{letter} in {name}"
                );
            }
            assert!(
                keys()
                    .hints(state)
                    .iter()
                    .any(|h| h.key == "^C" && h.label == "stop and exit"),
                "the footer names the interrupt in {name}"
            );
        }
    }

    /// j/k, the arrows and g/G move the selector's cursor; g/G jump without wrapping.
    #[test]
    fn selector_cursor_keys() {
        let state = selector();
        assert_eq!(chr(&state, 'j'), Some(Action::Next));
        assert_eq!(tap(&state, KeyCode::Down), Some(Action::Next));
        assert_eq!(chr(&state, 'k'), Some(Action::Previous));
        assert_eq!(tap(&state, KeyCode::Up), Some(Action::Previous));
        assert_eq!(chr(&state, 'g'), Some(Action::Top));
        assert_eq!(chr(&state, 'G'), Some(Action::Bottom));
        assert_eq!(tap(&state, KeyCode::Enter), Some(Action::Confirm));

        let mut s = selector();
        s.last();
        assert_eq!(s.cursor, 1);
        s.last();
        assert_eq!(s.cursor, 1);
        s.first();
        assert_eq!(s.cursor, 0);
    }

    /// The same keys move the roster cursor, and the roster has no confirm action.
    #[test]
    fn roster_cursor_keys() {
        let state = roster();
        assert_eq!(chr(&state, 'j'), Some(Action::RosterNext));
        assert_eq!(tap(&state, KeyCode::Down), Some(Action::RosterNext));
        assert_eq!(chr(&state, 'k'), Some(Action::RosterPrevious));
        assert_eq!(tap(&state, KeyCode::Up), Some(Action::RosterPrevious));
        assert_eq!(chr(&state, 'g'), Some(Action::RosterTop));
        assert_eq!(chr(&state, 'G'), Some(Action::RosterBottom));
        assert_eq!(tap(&state, KeyCode::Enter), None);
        assert_eq!(chr(&state, 'x'), Some(Action::RemoveAttendee));
        assert_eq!(chr(&state, 'n'), Some(Action::EditName));
        assert_eq!(chr(&state, 'e'), Some(Action::EditEmail));
        // Control-modified navigation is not navigation.
        assert_eq!(
            keys().dispatch(&state, &press(KeyCode::Up, KeyEventKind::Press, true)),
            None
        );
    }

    /// The base moves the cursor a later selector open will land on.
    #[test]
    fn base_cursor_keys_and_opening_keys() {
        let state = recording();
        assert_eq!(chr(&state, 'j'), Some(Action::Next));
        assert_eq!(tap(&state, KeyCode::Up), Some(Action::Previous));
        assert_eq!(chr(&state, 'g'), Some(Action::Top));
        assert_eq!(chr(&state, 'G'), Some(Action::Bottom));
        assert_eq!(tap(&state, KeyCode::Enter), Some(Action::OpenSelector));
        assert_eq!(chr(&state, 'r'), Some(Action::OpenRoster));
        assert_eq!(chr(&state, 'c'), Some(Action::ClearMeeting));
        assert_eq!(chr(&state, 'S'), None, "`S` is not the hand stop");
        assert_eq!(
            keys().dispatch(
                &state,
                &press(KeyCode::Char('s'), KeyEventKind::Press, true)
            ),
            None,
            "Ctrl-S is not the hand stop"
        );
    }

    /// Esc closes exactly one level in every context.
    #[test]
    fn esc_unwinds_one_level_per_context() {
        assert_eq!(tap(&recording(), KeyCode::Esc), None, "nothing to close");
        assert_eq!(tap(&selector(), KeyCode::Esc), Some(Action::CloseSelector));
        assert_eq!(tap(&roster(), KeyCode::Esc), Some(Action::CloseRoster));
        assert_eq!(
            tap(&editing(EditingField::Name), KeyCode::Esc),
            Some(Action::CancelField)
        );

        // Driving the actions the way the frame does: editing -> roster -> base.
        let mut s = editing(EditingField::Email);
        assert_eq!(Keys::context(&s), EDITING);
        s.cancel_edit();
        assert_eq!(
            Keys::context(&s),
            ROSTER,
            "Esc left the field, not the pane"
        );
        s.close_roster();
        assert_eq!(Keys::context(&s), BASE);
        s.open_selector();
        s.close_selector();
        assert_eq!(Keys::context(&s), BASE);
    }

    /// The hand stop is bound and advertised only where it works: a recording session, outside
    /// a field. Everywhere else the key is dead, so the live proof's absence checks hold.
    #[test]
    fn stop_is_enabled_only_outside_editing_and_while_recording() {
        for state in [recording(), selector(), roster()] {
            assert_eq!(chr(&state, 's'), Some(Action::StopSession));
            assert!(
                keys()
                    .hints(&state)
                    .iter()
                    .any(|h| h.label == "end this session")
            );
        }
        let mut finalizing = recording();
        finalizing.apply(&Note::Stopping);
        for (name, state) in [("idle", idle()), ("finalizing", finalizing)] {
            assert_eq!(chr(&state, 's'), None, "{name}");
            assert!(
                !keys()
                    .hints(&state)
                    .iter()
                    .any(|h| h.label == "end this session"),
                "{name}"
            );
        }
        let state = editing(EditingField::Name);
        assert_eq!(chr(&state, 's'), Some(Action::Type('s')));
        assert!(
            !keys()
                .hints(&state)
                .iter()
                .any(|h| h.label == "end this session")
        );
    }

    /// While a field is edited every printable (including the ones other contexts bind) is
    /// typed text, `?` included - help is deliberately unreachable there - and control
    /// characters never enter the buffer.
    #[test]
    fn printables_are_text_only_while_editing() {
        for field in [EditingField::Name, EditingField::Email] {
            let state = editing(field);
            for c in ['s', 'j', 'k', 'g', 'G', '?', 'c', 'x', '.', '@', ' '] {
                assert_eq!(chr(&state, c), Some(Action::Type(c)), "{c:?} in {field:?}");
            }
            assert_eq!(tap(&state, KeyCode::Backspace), Some(Action::DeleteChar));
            assert_eq!(tap(&state, KeyCode::Enter), Some(Action::CommitField));
            for c in ['\u{1}', '\t', '\r'] {
                assert_eq!(chr(&state, c), None, "control {c:?}");
            }
            assert_eq!(
                keys().dispatch(
                    &state,
                    &press(KeyCode::Char('g'), KeyEventKind::Release, false)
                ),
                None
            );
        }
        // And printables outside editing are not text.
        for state in [recording(), selector(), roster()] {
            assert_eq!(chr(&state, 'z'), None);
        }
    }

    /// Letters that act on a roster row, or on a meeting, are dead where they cannot work.
    #[test]
    fn keys_are_unbound_where_they_cannot_work() {
        for code in ['x', 'n', 'e'] {
            assert_eq!(chr(&recording(), code), None, "{code} in base");
            assert_eq!(chr(&selector(), code), None, "{code} in selector");
        }
        // `c` is not a roster-pane key, and not a base key once the meeting is cleared.
        assert_eq!(chr(&roster(), 'c'), None);
        let mut cleared = recording();
        cleared.apply(&Note::MeetingCleared);
        assert_eq!(chr(&cleared, 'c'), None);
        // No roster attached: `r` is dead; idle: Enter is dead.
        let mut no_roster = State::default();
        no_roster.apply(&started());
        assert_eq!(chr(&no_roster, 'r'), None);
        assert_eq!(tap(&idle(), KeyCode::Enter), None);
        // An attached but empty roster has no row operations.
        let mut empty = State::default();
        empty.apply(&started());
        empty.apply(&Note::RosterAttached {
            event_id: "EVENT-A".to_owned(),
            attendees: vec![],
        });
        empty.open_roster();
        assert_eq!(chr(&empty, 'x'), None);
        assert_eq!(chr(&empty, 'n'), None);
        // Unbound navigation keys stay unbound.
        for code in [KeyCode::Tab, KeyCode::Left, KeyCode::Right] {
            assert_eq!(tap(&recording(), code), None);
        }
    }

    /// `?` opens the help everywhere but while editing, and the overlay rows are drawn from the
    /// same table as the footer: every footer hint has a help row.
    #[test]
    fn help_is_bound_outside_editing_and_shares_the_footer_table() {
        for state in [recording(), selector(), roster(), idle()] {
            assert_eq!(chr(&state, '?'), Some(Action::Help));
        }
        for state in [
            recording(),
            selector(),
            roster(),
            editing(EditingField::Name),
            idle(),
        ] {
            let rows = keys().help_rows(&state);
            for hint in keys().hints(&state) {
                assert!(
                    rows.iter().any(|r| r.key == hint.key),
                    "{} is advertised but missing from the help",
                    hint.key
                );
            }
        }
    }

    /// The header names every context.
    #[test]
    fn the_context_has_a_name() {
        assert_eq!(Keys::context_name(&recording()), "overview");
        assert_eq!(Keys::context_name(&selector()), "meeting list");
        assert_eq!(Keys::context_name(&roster()), "roster");
        assert_eq!(
            Keys::context_name(&editing(EditingField::Name)),
            "editing name"
        );
        assert_eq!(
            Keys::context_name(&editing(EditingField::Email)),
            "editing email"
        );
    }
}
