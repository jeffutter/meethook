//! Colour and glyph policy. One place decides NO_COLOR and the non-Unicode fallback so no
//! widget checks the environment itself.

use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols::border;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme {
    pub color: bool,
    pub unicode: bool,
}

impl Default for Theme {
    /// Plain colourless Unicode: what tests render with, so expected buffers carry no styles.
    fn default() -> Self {
        Theme {
            color: false,
            unicode: true,
        }
    }
}

const ASCII_BORDER: border::Set<'static> = border::Set {
    top_left: "+",
    top_right: "+",
    bottom_left: "+",
    bottom_right: "+",
    vertical_left: "|",
    vertical_right: "|",
    horizontal_top: "-",
    horizontal_bottom: "-",
};

impl Theme {
    /// `is_tty` is the caller's existing "stdout is a terminal" fact; `env` looks up an
    /// environment variable (injected so tests do not touch the process environment).
    /// Colour is off for a non-empty `NO_COLOR` or a non-tty. Unicode is off when the first
    /// non-empty of `LC_ALL`/`LC_CTYPE`/`LANG` is not UTF-8, or `TERM` is `linux` or `dumb`.
    pub fn detect(is_tty: bool, env: impl Fn(&str) -> Option<String>) -> Self {
        let set = |name: &str| env(name).filter(|v| !v.is_empty());
        let color = is_tty && set("NO_COLOR").is_none();
        let locale = ["LC_ALL", "LC_CTYPE", "LANG"].iter().find_map(|n| set(n));
        let utf8_locale = locale.is_some_and(|l| {
            let l = l.to_ascii_lowercase();
            l.contains("utf-8") || l.contains("utf8")
        });
        let dumb_term = set("TERM").is_some_and(|t| t == "linux" || t == "dumb");
        Theme {
            color,
            unicode: utf8_locale && !dumb_term,
        }
    }

    pub fn key_style(&self) -> Style {
        if self.color {
            Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::new().add_modifier(Modifier::BOLD)
        }
    }

    pub fn border_style(&self, active: bool) -> Style {
        match (self.color, active) {
            (true, true) => Style::new().fg(Color::Yellow),
            (true, false) => Style::new().fg(Color::DarkGray),
            (false, true) => Style::new().add_modifier(Modifier::BOLD),
            (false, false) => Style::new(),
        }
    }

    pub fn dim(&self) -> Style {
        if self.color {
            Style::new().fg(Color::DarkGray)
        } else {
            Style::new().add_modifier(Modifier::DIM)
        }
    }

    pub fn border_set(&self) -> border::Set<'static> {
        if self.unicode {
            border::PLAIN
        } else {
            ASCII_BORDER
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detect(tty: bool, vars: &[(&str, &str)]) -> Theme {
        Theme::detect(tty, |k| {
            vars.iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.to_string())
        })
    }

    #[test]
    fn detect_matrix() {
        let utf = [("LANG", "en_US.UTF-8")];
        assert_eq!(
            detect(true, &utf),
            Theme {
                color: true,
                unicode: true
            }
        );
        assert!(!detect(false, &utf).color, "non-tty drops colour");
        assert!(!detect(true, &[("NO_COLOR", "1"), utf[0]]).color);
        assert!(
            detect(true, &[("NO_COLOR", ""), utf[0]]).color,
            "empty NO_COLOR is unset"
        );
        assert!(!detect(true, &[("LANG", "C")]).unicode);
        assert!(!detect(true, &[]).unicode, "no locale at all is not UTF-8");
        assert!(!detect(true, &[utf[0], ("TERM", "dumb")]).unicode);
        assert!(!detect(true, &[utf[0], ("TERM", "linux")]).unicode);
        assert!(detect(true, &[utf[0], ("TERM", "xterm-256color")]).unicode);
        // LC_ALL wins over LANG.
        assert!(!detect(true, &[("LC_ALL", "C"), utf[0]]).unicode);
        // An empty LC_ALL is skipped.
        assert!(detect(true, &[("LC_ALL", ""), utf[0]]).unicode);
    }
}
