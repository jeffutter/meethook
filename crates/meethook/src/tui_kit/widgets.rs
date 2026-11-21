//! Footer hint bar and the bordered pane.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Widget};

use super::keymap::Hint;
use super::theme::Theme;

const SEP: &str = " | ";

/// Pack `[key] label` items greedily into rows no wider than `width`, separated by " | ".
/// There is no row cap: a narrow terminal gets more rows instead of clipped hints. An item
/// wider than `width` gets a row to itself (the Paragraph truncates it). Always returns at
/// least one row, so a footer area never collapses to nothing.
pub fn hint_lines(hints: &[Hint], width: u16, theme: &Theme) -> Vec<Line<'static>> {
    let width = width as usize;
    let mut rows: Vec<Vec<Span<'static>>> = vec![vec![]];
    let mut used = 0usize;
    for h in hints {
        let key = format!("[{}]", h.key);
        let item_w = key.chars().count() + 1 + h.label.chars().count();
        if used > 0 && used + SEP.len() + item_w > width {
            rows.push(vec![]);
            used = 0;
        }
        let row = rows.last_mut().expect("at least one row");
        if used > 0 {
            row.push(Span::raw(SEP));
            used += SEP.len();
        }
        row.push(Span::styled(key, theme.key_style()));
        row.push(Span::raw(format!(" {}", h.label)));
        used += item_w;
    }
    rows.into_iter().map(Line::from).collect()
}

/// Rows the footer needs at `width`; always equals `hint_lines(..).len()`.
pub fn footer_height(hints: &[Hint], width: u16, theme: &Theme) -> u16 {
    hint_lines(hints, width, theme).len() as u16
}

pub fn render_footer(buf: &mut Buffer, area: Rect, hints: &[Hint], theme: &Theme) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    Paragraph::new(hint_lines(hints, area.width, theme)).render(area, buf);
}

/// A bordered block; the active pane is highlighted, using the theme's glyphs.
pub fn bordered_pane(title: &str, active: bool, theme: &Theme) -> Block<'static> {
    Block::bordered()
        .border_set(theme.border_set())
        .border_style(theme.border_style(active))
        .title(title.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::{Color, Modifier};

    fn hints() -> Vec<Hint> {
        ["quit", "next", "prev", "save", "edit", "delete"]
            .iter()
            .enumerate()
            .map(|(i, l)| Hint {
                key: format!("k{i}"),
                label: l.to_string(),
            })
            .collect()
    }

    fn text(l: &Line) -> String {
        l.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn wraps_without_clipping_or_losing_hints() {
        let t = Theme::default();
        let h = hints();
        let mut prev_rows = 0;
        for width in [80u16, 40, 10] {
            let lines = hint_lines(&h, width, &t);
            assert!(lines.len() >= prev_rows, "narrower never needs fewer rows");
            prev_rows = lines.len();
            assert_eq!(footer_height(&h, width, &t) as usize, lines.len());
            let joined = lines.iter().map(text).collect::<Vec<_>>().join("\n");
            for hint in &h {
                let item = format!("[{}] {}", hint.key, hint.label);
                assert_eq!(joined.matches(&item).count(), 1, "{item} at {width}");
            }
            for l in &lines {
                // Only a single oversized item may exceed the width (Paragraph truncates it).
                let single = l.spans.len() == 2;
                assert!(
                    single || text(l).chars().count() <= width as usize,
                    "row too wide at {width}"
                );
            }
        }
        assert_eq!(hint_lines(&h, 80, &t).len(), 1);
        assert!(hint_lines(&h, 40, &t).len() > 1);
    }

    #[test]
    fn oversized_item_and_zero_width_do_not_panic() {
        let t = Theme::default();
        let h = hints();
        assert_eq!(hint_lines(&h, 0, &t).len(), h.len());
        assert_eq!(hint_lines(&[], 10, &t).len(), 1);
        let mut buf = Buffer::empty(Rect::new(0, 0, 4, 2));
        render_footer(&mut buf, Rect::new(0, 0, 4, 2), &h, &t);
        render_footer(&mut buf, Rect::new(0, 0, 0, 0), &h, &t);
    }

    #[test]
    fn footer_renders_expected_rows() {
        let t = Theme::default();
        let area = Rect::new(0, 0, 24, 2);
        let mut buf = Buffer::empty(area);
        let h = vec![
            Hint {
                key: "q".into(),
                label: "quit".into(),
            },
            Hint {
                key: "Enter".into(),
                label: "open".into(),
            },
            Hint {
                key: "?".into(),
                label: "help".into(),
            },
        ];
        render_footer(&mut buf, area, &h, &t);
        let rows: Vec<String> = (0..2)
            .map(|y| (0..24).map(|x| buf[(x, y)].symbol().to_string()).collect())
            .collect();
        assert_eq!(rows[0], "[q] quit | [Enter] open ");
        assert_eq!(rows[1], "[?] help                ");
    }

    #[test]
    fn pane_active_vs_inactive_and_ascii_fallback() {
        let area = Rect::new(0, 0, 8, 3);
        let color = Theme {
            color: true,
            unicode: true,
        };
        let draw = |theme: &Theme, active| {
            let mut buf = Buffer::empty(area);
            bordered_pane("T", active, theme).render(area, &mut buf);
            buf
        };
        assert_eq!(draw(&color, true)[(0, 0)].fg, Color::Yellow);
        assert_eq!(draw(&color, false)[(0, 0)].fg, Color::DarkGray);
        assert_eq!(draw(&color, true)[(0, 0)].symbol(), "┌");
        let mono = Theme {
            color: false,
            unicode: true,
        };
        assert!(draw(&mono, true)[(0, 0)].modifier.contains(Modifier::BOLD));
        assert!(!draw(&mono, false)[(0, 0)].modifier.contains(Modifier::BOLD));
        let ascii = Theme {
            color: false,
            unicode: false,
        };
        let buf = draw(&ascii, true);
        assert_eq!(buf[(0, 0)].symbol(), "+");
        assert_eq!(buf[(7, 2)].symbol(), "+");
        assert_eq!(buf[(0, 1)].symbol(), "|");
        assert_eq!(buf[(3, 2)].symbol(), "-");
    }
}
