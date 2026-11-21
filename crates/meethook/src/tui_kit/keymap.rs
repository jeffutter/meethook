//! The keymap table: the single source for dispatch, footer hints and help rows.

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

/// A caller-defined name for a screen state ("list", "search", ...). The kit only compares
/// them; it never interprets the name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Context(pub &'static str);

/// A key plus the modifiers it requires. Ctrl and Alt must match exactly. Shift is ignored,
/// because for characters it is already in the char (`'G'`) and terminals report `BackTab`
/// with or without it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeySpec {
    pub code: KeyCode,
    pub ctrl: bool,
    pub alt: bool,
}

impl KeySpec {
    pub const fn key(code: KeyCode) -> Self {
        KeySpec {
            code,
            ctrl: false,
            alt: false,
        }
    }
    pub const fn ch(c: char) -> Self {
        Self::key(KeyCode::Char(c))
    }
    pub const fn ctrl(c: char) -> Self {
        KeySpec {
            code: KeyCode::Char(c),
            ctrl: true,
            alt: false,
        }
    }

    fn matches(&self, ev: &KeyEvent) -> bool {
        ev.kind != KeyEventKind::Release
            && ev.code == self.code
            && ev.modifiers.contains(KeyModifiers::CONTROL) == self.ctrl
            && ev.modifiers.contains(KeyModifiers::ALT) == self.alt
    }
}

/// The one place a key becomes text, so footer and help cannot spell it differently.
pub fn key_label(k: &KeySpec) -> String {
    let base = match k.code {
        KeyCode::Char(' ') => "Space".to_string(),
        KeyCode::Char(c) if k.ctrl => return format!("^{}", c.to_ascii_uppercase()),
        KeyCode::Char(c) => c.to_string(),
        KeyCode::Enter => "Enter".into(),
        KeyCode::Tab => "Tab".into(),
        KeyCode::BackTab => "S-Tab".into(),
        KeyCode::Esc => "Esc".into(),
        KeyCode::Backspace => "Bksp".into(),
        KeyCode::Delete => "Del".into(),
        KeyCode::Up => "Up".into(),
        KeyCode::Down => "Down".into(),
        KeyCode::Left => "Left".into(),
        KeyCode::Right => "Right".into(),
        KeyCode::PageUp => "PgUp".into(),
        KeyCode::PageDown => "PgDn".into(),
        KeyCode::Home => "Home".into(),
        KeyCode::End => "End".into(),
        KeyCode::F(n) => format!("F{n}"),
        other => format!("{other:?}"),
    };
    if k.alt { format!("M-{base}") } else { base }
}

#[derive(Debug, Clone, Copy)]
pub struct Binding<A> {
    pub context: Context,
    pub key: KeySpec,
    pub action: A,
    /// Footer text; empty keeps the binding out of the footer (aliases, rarely useful keys).
    pub hint: &'static str,
    /// Help-overlay text; empty keeps the binding out of the help.
    pub help: &'static str,
}

/// One footer item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Hint {
    pub key: String,
    pub label: String,
}

/// One help-overlay row. Disabled rows stay listed (dimmed) so the user can see the key exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HelpRow {
    pub key: String,
    pub help: String,
    pub enabled: bool,
}

/// Bindings in precedence order: the first enabled match wins.
///
/// Enabled-ness is a caller-supplied predicate on each call (not stored), so it can close over
/// the current view state. A catch-all such as "printable char goes to the filter" is not a
/// binding: `dispatch` returns `None` and the caller handles the key itself.
#[derive(Debug, Clone)]
pub struct Keymap<A> {
    bindings: Vec<Binding<A>>,
}

impl<A: Copy> Keymap<A> {
    pub fn new(bindings: Vec<Binding<A>>) -> Self {
        Keymap { bindings }
    }

    fn in_context(&self, ctx: Context) -> impl Iterator<Item = &Binding<A>> {
        self.bindings.iter().filter(move |b| b.context == ctx)
    }

    pub fn dispatch(
        &self,
        ctx: Context,
        enabled: &dyn Fn(&Binding<A>) -> bool,
        ev: &KeyEvent,
    ) -> Option<A> {
        self.in_context(ctx)
            .find(|b| b.key.matches(ev) && enabled(b))
            .map(|b| b.action)
    }

    /// Enabled bindings with a footer hint, in table order.
    pub fn hints(&self, ctx: Context, enabled: &dyn Fn(&Binding<A>) -> bool) -> Vec<Hint> {
        self.in_context(ctx)
            .filter(|b| !b.hint.is_empty() && enabled(b))
            .map(|b| Hint {
                key: key_label(&b.key),
                label: b.hint.to_string(),
            })
            .collect()
    }

    /// Every binding with help text, enabled or not.
    pub fn help_rows(&self, ctx: Context, enabled: &dyn Fn(&Binding<A>) -> bool) -> Vec<HelpRow> {
        self.in_context(ctx)
            .filter(|b| !b.help.is_empty())
            .map(|b| HelpRow {
                key: key_label(&b.key),
                help: b.help.to_string(),
                enabled: enabled(b),
            })
            .collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Act {
        Quit,
        Next,
        Prev,
        Save,
        Edit,
        Other,
    }

    pub const LIST: Context = Context("list");
    const OTHER: Context = Context("other");

    pub fn sample() -> Keymap<Act> {
        let b = |context, key, action, hint, help| Binding {
            context,
            key,
            action,
            hint,
            help,
        };
        Keymap::new(vec![
            b(LIST, KeySpec::ch('q'), Act::Quit, "quit", "Quit"),
            b(
                LIST,
                KeySpec::key(KeyCode::Down),
                Act::Next,
                "next",
                "Next row",
            ),
            b(LIST, KeySpec::ch('j'), Act::Next, "", "Next row (alias)"),
            b(
                LIST,
                KeySpec::key(KeyCode::Up),
                Act::Prev,
                "prev",
                "Previous row",
            ),
            b(LIST, KeySpec::ctrl('s'), Act::Save, "save", "Save"),
            b(LIST, KeySpec::ch('s'), Act::Edit, "edit", "Edit"),
            b(LIST, KeySpec::ch('q'), Act::Other, "never", "shadowed"),
            b(OTHER, KeySpec::ch('q'), Act::Other, "other", "Other"),
        ])
    }

    pub fn all(_: &Binding<Act>) -> bool {
        true
    }

    fn press(code: KeyCode, mods: KeyModifiers) -> KeyEvent {
        KeyEvent::new(code, mods)
    }

    #[test]
    fn first_match_wins_and_context_scopes() {
        let km = sample();
        let q = press(KeyCode::Char('q'), KeyModifiers::NONE);
        assert_eq!(km.dispatch(LIST, &all, &q), Some(Act::Quit));
        assert_eq!(km.dispatch(OTHER, &all, &q), Some(Act::Other));
        assert_eq!(km.dispatch(Context("nope"), &all, &q), None);
    }

    #[test]
    fn release_ignored_and_modifiers_distinguished() {
        let km = sample();
        let mut rel = press(KeyCode::Char('q'), KeyModifiers::NONE);
        rel.kind = KeyEventKind::Release;
        assert_eq!(km.dispatch(LIST, &all, &rel), None);
        let ctrl_s = press(KeyCode::Char('s'), KeyModifiers::CONTROL);
        let plain_s = press(KeyCode::Char('s'), KeyModifiers::NONE);
        assert_eq!(km.dispatch(LIST, &all, &ctrl_s), Some(Act::Save));
        assert_eq!(km.dispatch(LIST, &all, &plain_s), Some(Act::Edit));
        let shift_q = press(KeyCode::Char('q'), KeyModifiers::SHIFT);
        assert_eq!(
            km.dispatch(LIST, &all, &shift_q),
            Some(Act::Quit),
            "shift is ignored"
        );
    }

    #[test]
    fn disabled_binding_is_skipped_hidden_and_dimmed() {
        let km = sample();
        let no_save = |b: &Binding<Act>| b.action != Act::Save;
        let ctrl_s = press(KeyCode::Char('s'), KeyModifiers::CONTROL);
        assert_eq!(km.dispatch(LIST, &no_save, &ctrl_s), None);
        assert!(km.hints(LIST, &no_save).iter().all(|h| h.label != "save"));
        let row = km
            .help_rows(LIST, &no_save)
            .into_iter()
            .find(|r| r.help == "Save")
            .unwrap();
        assert!(!row.enabled);
        // A disabled earlier binding lets a later one on the same key through.
        let no_quit = |b: &Binding<Act>| b.action != Act::Quit;
        let q = press(KeyCode::Char('q'), KeyModifiers::NONE);
        assert_eq!(km.dispatch(LIST, &no_quit, &q), Some(Act::Other));
    }

    #[test]
    fn empty_hint_stays_out_of_footer_but_in_help() {
        let km = sample();
        assert!(km.hints(LIST, &all).iter().all(|h| h.key != "j"));
        assert!(km.help_rows(LIST, &all).iter().any(|r| r.key == "j"));
    }

    #[test]
    fn labels() {
        assert_eq!(key_label(&KeySpec::ctrl('p')), "^P");
        assert_eq!(key_label(&KeySpec::key(KeyCode::Enter)), "Enter");
        assert_eq!(key_label(&KeySpec::ch(' ')), "Space");
        assert_eq!(key_label(&KeySpec::key(KeyCode::PageDown)), "PgDn");
        assert_eq!(
            key_label(&KeySpec {
                code: KeyCode::Char('x'),
                ctrl: false,
                alt: true
            }),
            "M-x"
        );
    }

    #[test]
    fn hint_and_help_share_key_labels() {
        let km = sample();
        let hints = km.hints(LIST, &all);
        let help = km.help_rows(LIST, &all);
        for h in &hints {
            assert!(
                help.iter().any(|r| r.key == h.key),
                "{} missing from help",
                h.key
            );
        }
        for b in km.in_context(LIST) {
            let label = key_label(&b.key);
            if !b.hint.is_empty() {
                assert!(hints.iter().any(|h| h.key == label && h.label == b.hint));
            }
            if !b.help.is_empty() {
                assert!(help.iter().any(|r| r.key == label && r.help == b.help));
            }
        }
    }
}
