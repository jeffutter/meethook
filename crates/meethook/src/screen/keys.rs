//! The enroll frame's key table: one [`Keymap`] that drives dispatch, the hint footer and the `?`
//! help overlay, so a key cannot be bound but missing from the help, or advertised but dead.
//!
//! Two contexts. In [`NORMAL`] every printable key is a command and no action needs a Ctrl chord
//! (Ctrl-C quits; Ctrl-U/Ctrl-D page). In [`FILTER`], entered with `/`, printable keys type into
//! the candidate filter and Esc leaves. This breaks the old Ctrl-chord muscle memory on purpose:
//! the chords existed only because the filter swallowed every plain key.
//!
//! Several actions have two bindings on one key, a *labelled* one and a hidden fallback after it.
//! The predicate in [`live`] enables the labelled one only while its wording is true (`play` vs
//! `play (no audio)`, `choose` hidden on a row that can only be overridden), so the footer says
//! what the key would do *now* while the key itself still reaches the state machine, which owns
//! the sentence for the cases that cannot work.
//!
//! Nothing is bound on Ctrl-I or Ctrl-M: a terminal sends the same byte as Tab and Enter.

use meethook_enroll::Refusal;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use super::Sounding;
use super::state::{Event, Focus, Mode, View};
use crate::tui_kit::{Binding, Context, HelpRow, Hint, KeySpec, Keymap, Theme, key_label};

pub const NORMAL: Context = Context("normal");
pub const FILTER: Context = Context("filter");

/// The play key's hint when the voice has no samples.
const PLAY_NO_AUDIO: &str = "play (no audio)";
/// The play-the-line key's hint when the selected line has no samples.
/// Enter's hint on the voice list while the cursor is off the voice being asked about.
const WORK_ON: &str = "work on";
const LINE_NO_AUDIO: &str = "line (no audio)";

pub struct Keys {
    keymap: Keymap<Event>,
    pub theme: Theme,
}

fn b(
    context: Context,
    key: KeySpec,
    action: Event,
    hint: &'static str,
    help: &'static str,
) -> Binding<Event> {
    Binding {
        context,
        key,
        action,
        hint,
        help,
    }
}

fn table() -> Vec<Binding<Event>> {
    use Event as E;
    let ch = KeySpec::ch;
    let key = KeySpec::key;
    let ctrl = KeySpec::ctrl;
    let n = NORMAL;
    let f = FILTER;
    vec![
        // Movement. Acts on whichever pane has the focus.
        b(
            n,
            ch('j'),
            E::MoveDown,
            "down",
            "Move down in the focused pane",
        ),
        b(n, key(KeyCode::Down), E::MoveDown, "", "Move down (alias)"),
        b(n, ch('k'), E::MoveUp, "up", "Move up in the focused pane"),
        b(n, key(KeyCode::Up), E::MoveUp, "", "Move up (alias)"),
        b(n, ch('g'), E::Top, "", "First row"),
        b(n, key(KeyCode::Home), E::Top, "", "First row (alias)"),
        b(n, ch('G'), E::Bottom, "", "Last row"),
        b(n, key(KeyCode::End), E::Bottom, "", "Last row (alias)"),
        b(n, key(KeyCode::PageUp), E::PageUp, "", "Page up"),
        b(n, key(KeyCode::PageDown), E::PageDown, "", "Page down"),
        b(n, ctrl('u'), E::HalfUp, "", "Half page up"),
        b(n, ctrl('d'), E::HalfDown, "", "Half page down"),
        // Focus.
        b(
            n,
            key(KeyCode::Tab),
            E::FocusNext,
            "pane",
            "Next pane (voices, candidates)",
        ),
        b(n, key(KeyCode::BackTab), E::FocusPrev, "", "Previous pane"),
        b(n, ch('h'), E::FocusVoices, "", "Focus the voice list"),
        b(
            n,
            key(KeyCode::Left),
            E::FocusVoices,
            "",
            "Focus the voice list (alias)",
        ),
        b(n, ch('l'), E::FocusCandidates, "", "Focus the candidates"),
        b(
            n,
            key(KeyCode::Right),
            E::FocusCandidates,
            "",
            "Focus the candidates (alias)",
        ),
        // Answering.
        b(
            n,
            key(KeyCode::Enter),
            E::Choose,
            WORK_ON,
            "Work on the voice under the cursor (voice list, on another row)",
        ),
        b(
            n,
            key(KeyCode::Enter),
            E::Choose,
            "choose",
            "Answer with the highlighted candidate",
        ),
        b(n, key(KeyCode::Enter), E::Choose, "", ""),
        b(
            n,
            ch('o'),
            E::Anyway,
            "anyway",
            "Answer anyway, taking the name off the voice that holds it",
        ),
        b(n, ch('y'), E::Confirm, "confirm", "Confirm the guess"),
        b(n, ch('r'), E::Deny, "reject", "Reject the guess"),
        b(
            n,
            ch('n'),
            E::NewPerson,
            "new",
            "Answer with the filter text as somebody new",
        ),
        b(
            n,
            ch('a'),
            E::Assert,
            "assert",
            "Assert the whole session is the highlighted person",
        ),
        b(
            n,
            ch('m'),
            E::Mark,
            "mark",
            "Mark or unmark the voice under the cursor as one person with the others",
        ),
        b(
            n,
            ch('w'),
            E::Select,
            "work on",
            "Work on the voice under the cursor next",
        ),
        b(n, ch('s'), E::Skip, "skip", "Skip this voice"),
        b(
            n,
            ch('x'),
            E::Leave,
            "leave",
            "Leave this session and open the next meeting",
        ),
        // Playback; the shell holds the samples.
        b(n, ch('p'), E::Play, "play", "Play the voice"),
        b(n, ch('p'), E::Play, PLAY_NO_AUDIO, ""),
        b(n, ch('P'), E::PlaySnippet, "line", "Play the selected line"),
        b(n, ch('P'), E::PlaySnippet, LINE_NO_AUDIO, ""),
        b(n, ch('P'), E::PlaySnippet, "", ""),
        b(n, ch('['), E::SnippetUp, "", "Select the previous line"),
        b(n, ch(']'), E::SnippetDown, "", "Select the next line"),
        // Filter.
        b(
            n,
            ch('/'),
            E::EnterFilter,
            "filter",
            "Type into the candidate filter",
        ),
        b(n, key(KeyCode::Esc), E::ClearFilter, "", "Clear the filter"),
        b(n, ch('?'), E::Help, "help", "Show this help"),
        b(n, ch('q'), E::Quit, "quit", "Quit"),
        b(n, ctrl('c'), E::Quit, "", "Quit (alias)"),
        // Filter mode: everything printable types, so only these are commands.
        b(
            f,
            key(KeyCode::Esc),
            E::LeaveFilter,
            "done",
            "Stop typing, keep the filter",
        ),
        b(
            f,
            key(KeyCode::Enter),
            E::Choose,
            "choose",
            "Answer with the highlighted candidate",
        ),
        b(f, key(KeyCode::Enter), E::Choose, "", ""),
        b(
            f,
            key(KeyCode::Down),
            E::CandidateDown,
            "down",
            "Highlight the next candidate",
        ),
        b(
            f,
            ctrl('n'),
            E::CandidateDown,
            "",
            "Highlight the next candidate (alias)",
        ),
        b(
            f,
            key(KeyCode::Up),
            E::CandidateUp,
            "up",
            "Highlight the previous candidate",
        ),
        b(
            f,
            ctrl('p'),
            E::CandidateUp,
            "",
            "Highlight the previous candidate (alias)",
        ),
        b(
            f,
            key(KeyCode::Backspace),
            E::Backspace,
            "",
            "Delete a character",
        ),
        b(f, ctrl('c'), E::Quit, "quit", "Quit"),
    ]
}

/// Whether `binding` is true of this view right now.
fn live(view: &View<'_>, binding: &Binding<Event>) -> bool {
    match (binding.action, binding.hint) {
        (Event::Play, PLAY_NO_AUDIO) => view.clip_is_empty,
        (Event::Play, _) => !view.clip_is_empty,
        // The labelled variants come first; the unlabelled fallback is always live so the key
        // reaches the shell, which says why there is nothing to play.
        (Event::PlaySnippet, "line") => view
            .snippets
            .get(view.snippet)
            .is_some_and(|s| !s.audio.is_empty()),
        (Event::PlaySnippet, LINE_NO_AUDIO) => view
            .snippets
            .get(view.snippet)
            .is_some_and(|s| s.audio.is_empty()),
        // Enter does nothing on a row refused for taking a name off another voice, and `o`
        // does nothing on any other, so the labels are swapped rather than both offered.
        (Event::Choose, WORK_ON) => view.mode == Mode::Normal && cursor_elsewhere(view),
        (Event::Choose, hint) => hint.is_empty() || (!cursor_elsewhere(view) && !taken(view)),
        (Event::Anyway, _) => taken(view),
        (Event::Confirm, _) => view
            .guess
            .is_some_and(|g| view.candidates.iter().any(|c| c.name == g)),
        (Event::Deny, _) => view.guess.is_some(),
        (Event::NewPerson, _) => !view.filter.trim().is_empty(),
        (Event::ClearFilter, _) => !view.filter.is_empty(),
        _ => true,
    }
}

/// Whether Enter would go to another voice: the voice list has the focus and its cursor is off
/// the row being asked about.
fn cursor_elsewhere(view: &View<'_>) -> bool {
    view.focus == Focus::Voices && view.rows.get(view.cursor).is_some_and(|row| !row.current)
}

fn taken(view: &View<'_>) -> bool {
    matches!(
        view.highlighted().and_then(|c| c.refusal.as_ref()),
        Some(Refusal::Taken { .. })
    )
}

impl Keys {
    pub fn new(theme: Theme) -> Keys {
        Keys {
            keymap: Keymap::new(table()),
            theme,
        }
    }

    pub fn context(view: &View<'_>) -> Context {
        match view.mode {
            Mode::Normal => NORMAL,
            Mode::Filter => FILTER,
        }
    }

    /// The event this key means for this view, if any. In filter mode an unbound printable
    /// character is the typed text.
    pub fn dispatch(&self, view: &View<'_>, key: &KeyEvent) -> Option<Event> {
        if key.kind == KeyEventKind::Release {
            return None;
        }
        let ctx = Self::context(view);
        if let Some(event) = self.keymap.dispatch(ctx, &|b| live(view, b), key) {
            return Some(event);
        }
        match key.code {
            KeyCode::Char(c)
                if ctx == FILTER
                    && !c.is_control()
                    && !key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                Some(Event::Filter(c))
            }
            _ => None,
        }
    }

    /// The footer's items: while a clip plays the four keys that mean something mid-clip, else
    /// every enabled hint for the current mode. In filter mode the plain letters type into the
    /// filter even mid-clip, so the filter's own hints stand.
    pub fn hints(&self, view: &View<'_>, sounding: Option<Sounding>) -> Vec<Hint> {
        if let Some(sounding) = sounding.filter(|_| Self::context(view) == NORMAL) {
            let restart = match sounding.line {
                Some(_) => Event::PlaySnippet,
                None => Event::Play,
            };
            return [
                (restart, "restart"),
                (Event::Skip, "skip"),
                (Event::Leave, "leave"),
                (Event::Quit, "quit"),
            ]
            .into_iter()
            .filter_map(|(action, label)| {
                let binding = self
                    .keymap
                    .bindings()
                    .iter()
                    .find(|b| b.context == NORMAL && b.action == action)?;
                Some(Hint {
                    key: key_label(&binding.key),
                    label: label.to_string(),
                })
            })
            .collect();
        }
        self.keymap.hints(Self::context(view), &|b| live(view, b))
    }

    pub fn help_rows(&self, view: &View<'_>) -> Vec<HelpRow> {
        self.keymap
            .help_rows(Self::context(view), &|b| live(view, b))
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    use meethook_enroll::{Position, Queued, Snippet};
    use meethook_session::SessionId;
    use meethook_transcribe::{Attribution, Resemblance};

    use super::super::state::tests::{Free, heard};
    use super::super::state::{Context as Scan, Cost, Costs, Screen, VoiceView};

    /// Every candidate is refused for taking a name off another voice.
    struct Takes;

    impl Costs for Takes {
        fn of(&self, _name: &str) -> Cost {
            Cost {
                refusal: Some(Refusal::Taken {
                    voice: "Unknown 2".to_string(),
                    losing: "Bob".to_string(),
                }),
                summary: Vec::new(),
                assertion: None,
            }
        }
        fn group_of(&self, _: &str, _: &[&str]) -> Option<meethook_enroll::GroupConsequence> {
            None
        }
        fn fragment_group_of(
            &self,
            _: &str,
            _: &[&str],
        ) -> Option<meethook_enroll::GroupConsequence> {
            None
        }
    }

    /// One question put to a fresh screen, `setup` keys applied, then `check` run on the view.
    /// `guess` makes the voice carry a tentative guess of Ivan; `audio` gives the voice and its
    /// one line samples.
    fn viewed<R>(
        costs: &dyn Costs,
        guess: bool,
        audio: bool,
        setup: &[Event],
        check: impl FnOnce(&View<'_>) -> R,
    ) -> R {
        static AUDIO: [f32; 2] = [0.1, 0.2];
        let samples: &[f32] = if audio { &AUDIO } else { &[] };
        let session = SessionId::parse("20260819-100000").expect("a session id");
        let attribution = match guess {
            true => Attribution::Tentative {
                name: "Ivan".to_string(),
                similarity: 0.38,
            },
            false => Attribution::Unknown("Unknown 2".to_string()),
        };
        let queue = [Queued {
            number: "Unknown 2",
            attribution: &attribution,
            speech_seconds: 95.0,
            below_floor: false,
        }];
        let resembles = [
            Resemblance {
                name: "Milo".to_string(),
                similarity: 0.71,
                references: 3,
            },
            Resemblance {
                name: "Ivan".to_string(),
                similarity: 0.38,
                references: 1,
            },
        ];
        let snippets: [Snippet<'_>; 1] = [heard("right, next week", 107.0, samples)];
        let enrolled = ["Milo", "Ivan"];
        let voice = VoiceView {
            session: &session,
            meeting: None,
            position: Position { nth: 1, of: 1 },
            number: "Unknown 2",
            speech_seconds: 95.0,
            attribution: &attribution,
            queue: &queue,
            snippets: &snippets,
            resembles: &resembles,
            enrolled: &enrolled,
            fragment_groups: &[],
            bundle_members: None,
            clip_is_empty: !audio,
        };
        let mut screen = Screen::default();
        screen.arrive(&voice);
        for event in setup {
            screen.answer(&voice, *event, costs);
        }
        check(&screen.view(&voice, costs, Scan::Reading))
    }

    fn keys() -> Keys {
        Keys::new(Theme::default())
    }

    fn event_of(setup: &[Event], key: KeyEvent) -> Option<Event> {
        viewed(&Free, false, true, setup, |view| {
            keys().dispatch(view, &key)
        })
    }

    pub fn press(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }
    pub fn plain(c: char) -> KeyEvent {
        press(KeyCode::Char(c))
    }
    pub fn ctrl(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL)
    }

    /// Every binding with a hint also has help, so the footer never advertises a key the help
    /// does not list. The two variant bindings carry no help of their own, so the check is by
    /// key label.
    #[test]
    fn every_hinted_key_is_in_the_help() {
        let keys = Keys::new(Theme::default());
        for ctx in [NORMAL, FILTER] {
            let rows: Vec<String> = keys
                .keymap
                .help_rows(ctx, &|_| true)
                .into_iter()
                .map(|r| r.key)
                .collect();
            for binding in keys.keymap.bindings().iter().filter(|b| b.context == ctx) {
                if !binding.hint.is_empty() {
                    let label = key_label(&binding.key);
                    assert!(
                        rows.contains(&label),
                        "{label} ({}) is in the footer but not the help",
                        binding.hint
                    );
                }
            }
        }
    }

    /// No action in normal mode needs a Ctrl chord except quit; Ctrl is for paging only.
    #[test]
    fn ctrl_chords_are_only_quit_and_paging() {
        for binding in table().iter().filter(|b| b.context == NORMAL && b.key.ctrl) {
            assert!(
                matches!(
                    binding.action,
                    Event::Quit | Event::HalfUp | Event::HalfDown
                ),
                "{:?} is on a Ctrl chord",
                binding.action
            );
        }
    }

    /// Nothing is bound on a key whose byte is Tab or Enter.
    #[test]
    fn nothing_is_bound_on_ctrl_i_or_ctrl_m() {
        for binding in table() {
            assert!(
                !(binding.key.ctrl && matches!(binding.key.code, KeyCode::Char('i' | 'm'))),
                "{:?}",
                binding.action
            );
        }
    }

    /// Every plain key in normal mode, in one table, so a rebinding that drops one fails here
    /// rather than in front of a user. No action but quit needs a Ctrl chord.
    #[test]
    fn normal_mode_keys_mean_what_the_footer_says() {
        let table = [
            (plain('j'), Event::MoveDown),
            (press(KeyCode::Down), Event::MoveDown),
            (plain('k'), Event::MoveUp),
            (press(KeyCode::Up), Event::MoveUp),
            (plain('g'), Event::Top),
            (plain('G'), Event::Bottom),
            (press(KeyCode::Home), Event::Top),
            (press(KeyCode::End), Event::Bottom),
            (press(KeyCode::PageUp), Event::PageUp),
            (press(KeyCode::PageDown), Event::PageDown),
            (ctrl('u'), Event::HalfUp),
            (ctrl('d'), Event::HalfDown),
            (press(KeyCode::Tab), Event::FocusNext),
            (press(KeyCode::BackTab), Event::FocusPrev),
            (plain('h'), Event::FocusVoices),
            (press(KeyCode::Left), Event::FocusVoices),
            (plain('l'), Event::FocusCandidates),
            (press(KeyCode::Right), Event::FocusCandidates),
            (press(KeyCode::Enter), Event::Choose),
            (plain('w'), Event::Select),
            (plain('p'), Event::Play),
            (plain('P'), Event::PlaySnippet),
            (plain('['), Event::SnippetUp),
            (plain(']'), Event::SnippetDown),
            (plain('s'), Event::Skip),
            (plain('x'), Event::Leave),
            (plain('a'), Event::Assert),
            (plain('m'), Event::Mark),
            (plain('/'), Event::EnterFilter),
            (plain('?'), Event::Help),
            (plain('q'), Event::Quit),
            (ctrl('c'), Event::Quit),
        ];
        for (key, expected) in table {
            assert_eq!(event_of(&[], key), Some(expected), "{key:?}");
        }
        // Shift-Tab arrives with the modifier set.
        let shifted = KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT);
        assert_eq!(event_of(&[], shifted), Some(Event::FocusPrev));
    }

    /// The old Ctrl chords are gone: they must not do what they did, and an unbound key, a
    /// control key nothing owns and a release do nothing at all.
    #[test]
    fn the_old_ctrl_chords_are_gone_and_unbound_keys_do_nothing() {
        for c in ['n', 'p', 'l', 's', 'g', 'o', 'a', 'k', 'y', 'r', 'b'] {
            assert_eq!(event_of(&[], ctrl(c)), None, "^{c}");
        }
        assert_eq!(event_of(&[], plain('z')), None);
        assert_eq!(event_of(&[], press(KeyCode::F(5))), None);
        let released = KeyEvent {
            kind: KeyEventKind::Release,
            ..press(KeyCode::Enter)
        };
        assert_eq!(event_of(&[], released), None);
    }

    /// `/` is the way in; from there every printable key types -- the action letters included --
    /// and Esc is the way out, keeping the text.
    #[test]
    fn filter_mode_types_every_printable_key_and_escape_leaves() {
        let typing = [Event::EnterFilter];
        for c in ['a', 'q', 'j', 'w', '/', '?', ' ', 'P', 'é'] {
            assert_eq!(event_of(&typing, plain(c)), Some(Event::Filter(c)), "{c}");
        }
        assert_eq!(
            event_of(&typing, press(KeyCode::Esc)),
            Some(Event::LeaveFilter)
        );
        assert_eq!(
            event_of(&typing, press(KeyCode::Enter)),
            Some(Event::Choose)
        );
        assert_eq!(
            event_of(&typing, press(KeyCode::Backspace)),
            Some(Event::Backspace)
        );
        assert_eq!(
            event_of(&typing, press(KeyCode::Down)),
            Some(Event::CandidateDown)
        );
        assert_eq!(event_of(&typing, ctrl('n')), Some(Event::CandidateDown));
        assert_eq!(
            event_of(&typing, press(KeyCode::Up)),
            Some(Event::CandidateUp)
        );
        assert_eq!(event_of(&typing, ctrl('p')), Some(Event::CandidateUp));
        assert_eq!(event_of(&typing, ctrl('c')), Some(Event::Quit));
        assert_eq!(
            event_of(&typing, ctrl('b')),
            None,
            "an unbound control key must not type into the filter"
        );
        // And once out of the mode the same letters are commands again.
        let left = [Event::EnterFilter, Event::Filter('a'), Event::LeaveFilter];
        assert_eq!(event_of(&left, plain('a')), Some(Event::Assert));
        assert_eq!(event_of(&left, plain('q')), Some(Event::Quit));
    }

    /// Keys that cannot work are neither advertised nor live.
    #[test]
    fn a_key_that_cannot_work_is_neither_in_the_footer_nor_live() {
        let labels = |view: &View<'_>| {
            keys()
                .hints(view, None)
                .into_iter()
                .map(|h| format!("{} {}", h.key, h.label))
                .collect::<Vec<_>>()
        };
        // No guess, no filter text, nothing refused.
        viewed(&Free, false, true, &[], |view| {
            let hints = labels(view);
            for gone in ["[y]", "y confirm", "r reject", "o anyway", "n new"] {
                assert!(!hints.iter().any(|h| h == gone), "{gone} in {hints:?}");
            }
            assert!(hints.contains(&"Enter choose".to_string()));
            assert!(hints.contains(&"p play".to_string()));
            assert!(hints.contains(&"P line".to_string()));
        });
        assert_eq!(event_of(&[], plain('y')), None);
        assert_eq!(event_of(&[], plain('r')), None);
        assert_eq!(event_of(&[], plain('o')), None);
        assert_eq!(event_of(&[], plain('n')), None, "no name typed yet");
        assert_eq!(event_of(&[], press(KeyCode::Esc)), None, "nothing to clear");

        // A guess brings both guess keys, and the typed name brings `n` and Esc.
        viewed(&Free, true, true, &[], |view| {
            let hints = labels(view);
            assert!(hints.contains(&"y confirm".to_string()), "{hints:?}");
            assert!(hints.contains(&"r reject".to_string()));
        });
        let typed = [Event::EnterFilter, Event::Filter('Z'), Event::LeaveFilter];
        assert_eq!(event_of(&typed, plain('n')), Some(Event::NewPerson));
        assert_eq!(
            event_of(&typed, press(KeyCode::Esc)),
            Some(Event::ClearFilter)
        );

        // A refused-for-taking row swaps Enter for `o`; Enter still reaches the state machine.
        viewed(&Takes, false, true, &[], |view| {
            let hints = labels(view);
            assert!(hints.contains(&"o anyway".to_string()), "{hints:?}");
            assert!(!hints.contains(&"Enter choose".to_string()));
            assert_eq!(
                keys().dispatch(view, &press(KeyCode::Enter)),
                Some(Event::Choose)
            );
            assert_eq!(keys().dispatch(view, &plain('o')), Some(Event::Anyway));
        });

        // No audio is said, and the key still reaches the shell, which says why.
        viewed(&Free, false, false, &[], |view| {
            let hints = labels(view);
            assert!(
                hints.contains(&"p play (no audio)".to_string()),
                "{hints:?}"
            );
            assert!(hints.contains(&"P line (no audio)".to_string()));
            assert_eq!(keys().dispatch(view, &plain('p')), Some(Event::Play));
            assert_eq!(keys().dispatch(view, &plain('P')), Some(Event::PlaySnippet));
        });
    }

    /// While a clip sounds the footer offers only restart, skip, leave and quit, with the restart
    /// key following what is sounding.
    #[test]
    fn the_playing_footer_names_the_restart_key_by_what_sounds() {
        use crate::clips::Progress;
        use std::time::Duration;
        let sounding = |line| Sounding {
            progress: Progress {
                elapsed: Duration::from_secs(1),
                length: Duration::from_secs(9),
            },
            line,
        };
        viewed(&Free, false, true, &[], |view| {
            let whole: Vec<_> = keys()
                .hints(view, Some(sounding(None)))
                .into_iter()
                .map(|h| format!("{} {}", h.key, h.label))
                .collect();
            assert_eq!(whole, ["p restart", "s skip", "x leave", "q quit"]);
            let line = keys().hints(view, Some(sounding(Some(0))));
            assert_eq!(line[0].key, "P");
        });
    }

    /// In filter mode the letters type, so a sounding clip must not bring the normal-mode keys
    /// back into the footer.
    #[test]
    fn the_playing_footer_in_filter_mode_lists_no_plain_letters() {
        use crate::clips::Progress;
        use std::time::Duration;
        let sounding = Sounding {
            progress: Progress {
                elapsed: Duration::from_secs(1),
                length: Duration::from_secs(9),
            },
            line: None,
        };
        viewed(&Free, false, true, &[Event::EnterFilter], |view| {
            let hints = keys().hints(view, Some(sounding));
            let keys_shown: Vec<_> = hints.iter().map(|h| h.key.as_str()).collect();
            for letter in ["s", "x", "q", "p"] {
                assert!(!keys_shown.contains(&letter), "{letter} in {keys_shown:?}");
            }
            assert!(keys_shown.contains(&"Esc"), "{keys_shown:?}");
            assert!(keys_shown.contains(&"^C"), "{keys_shown:?}");
        });
    }

    /// The help lists every key, aliases included, and dims the ones that cannot work now.
    #[test]
    fn help_lists_aliases_and_dims_what_cannot_work() {
        viewed(&Free, false, true, &[], |view| {
            let rows = keys().help_rows(view);
            for key in [
                "j", "Down", "g", "Home", "PgUp", "^U", "^D", "h", "Left", "^C",
            ] {
                assert!(rows.iter().any(|r| r.key == key), "{key} missing");
            }
            let guess = rows.iter().find(|r| r.key == "y").expect("a confirm row");
            assert!(!guess.enabled, "no guess to confirm");
            let play = rows.iter().find(|r| r.key == "p").expect("a play row");
            assert!(play.enabled);
        });
    }
}
