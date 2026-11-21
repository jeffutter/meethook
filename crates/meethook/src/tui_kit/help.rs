//! The `?` overlay: the full keymap table for the active context, scrollable.

use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind};
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Clear, Paragraph, Widget};

use super::keymap::HelpRow;
use super::theme::Theme;
use super::widgets::bordered_pane;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HelpOutcome {
    Open,
    Close,
}

#[derive(Debug, Clone, Default)]
pub struct HelpOverlay {
    pub scroll: usize,
}

const MAX_WIDTH: u16 = 64;

/// Centred popup: as tall as the rows need, capped by the terminal.
fn popup(area: Rect, rows: usize) -> Rect {
    let w = area.width.min(MAX_WIDTH);
    let h = (rows.min(u16::MAX as usize - 2) as u16 + 2).min(area.height);
    Rect::new(
        area.x + (area.width - w) / 2,
        area.y + (area.height - h) / 2,
        w,
        h,
    )
}

impl HelpOverlay {
    /// Rows visible at once in the popup for `rows` rows on a screen of `area`; the page size
    /// for [`HelpOverlay::handle_key`].
    pub fn page_height(area: Rect, rows: usize) -> usize {
        popup(area, rows).height.saturating_sub(2) as usize
    }

    /// Esc, `?` and `q` close; arrows, PgUp/PgDn, Home/End scroll. The scroll is clamped to
    /// `rows_len - page`, so it never leaves blank space below the last row.
    pub fn handle_key(&mut self, ev: KeyEvent, rows_len: usize, page: usize) -> HelpOutcome {
        if ev.kind == KeyEventKind::Release {
            return HelpOutcome::Open;
        }
        let max = rows_len.saturating_sub(page);
        let page = page.max(1);
        match ev.code {
            KeyCode::Esc | KeyCode::Char('?') | KeyCode::Char('q') => return HelpOutcome::Close,
            KeyCode::Up | KeyCode::Char('k') => self.scroll = self.scroll.saturating_sub(1),
            KeyCode::Down | KeyCode::Char('j') => self.scroll += 1,
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(page),
            KeyCode::PageDown => self.scroll += page,
            KeyCode::Home => self.scroll = 0,
            KeyCode::End => self.scroll = max,
            _ => {}
        }
        self.scroll = self.scroll.min(max);
        HelpOutcome::Open
    }

    /// Draw over whatever is in `area`. Clamps the stored scroll first, so a resize that made
    /// the popup taller cannot leave it out of range.
    pub fn render(
        &mut self,
        buf: &mut Buffer,
        area: Rect,
        title: &str,
        rows: &[HelpRow],
        theme: &Theme,
    ) {
        let rect = popup(area, rows.len());
        if rect.width < 3 || rect.height < 3 {
            return;
        }
        let inner_h = rect.height as usize - 2;
        self.scroll = self.scroll.min(rows.len().saturating_sub(inner_h));
        let key_w = rows
            .iter()
            .map(|r| r.key.chars().count())
            .max()
            .unwrap_or(0);

        let mut block_title = title.to_string();
        if rows.len() > inner_h {
            block_title = format!(
                "{title} ({}-{}/{})",
                self.scroll + 1,
                self.scroll + inner_h,
                rows.len()
            );
        }
        let lines: Vec<Line> = rows
            .iter()
            .skip(self.scroll)
            .take(inner_h)
            .map(|r| {
                let style = |s| if r.enabled { s } else { theme.dim() };
                Line::from(vec![
                    Span::styled(format!("{:<key_w$}", r.key), style(theme.key_style())),
                    Span::styled(format!("  {}", r.help), style(Default::default())),
                ])
            })
            .collect();

        Clear.render(rect, buf);
        Paragraph::new(lines)
            .block(bordered_pane(&block_title, true, theme))
            .render(rect, buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::crossterm::event::KeyModifiers;
    use ratatui::style::Color;

    fn rows(n: usize) -> Vec<HelpRow> {
        (0..n)
            .map(|i| HelpRow {
                key: format!("k{i}"),
                help: format!("does {i}"),
                enabled: i != 1,
            })
            .collect()
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn dump(buf: &Buffer, area: Rect) -> Vec<String> {
        (0..area.height)
            .map(|y| {
                (0..area.width)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn renders_full_table_when_it_fits() {
        let area = Rect::new(0, 0, 20, 6);
        let mut buf = Buffer::empty(area);
        HelpOverlay::default().render(&mut buf, area, "Help", &rows(3), &Theme::default());
        assert_eq!(
            dump(&buf, area),
            vec![
                "┌Help──────────────┐",
                "│k0  does 0        │",
                "│k1  does 1        │",
                "│k2  does 2        │",
                "└──────────────────┘",
                "                    ",
            ]
        );
    }

    #[test]
    fn scrolls_when_taller_than_terminal_and_clamps() {
        let area = Rect::new(0, 0, 30, 6);
        let theme = Theme::default();
        let all = rows(10);
        let page = HelpOverlay::page_height(area, all.len());
        assert_eq!(page, 4);
        let mut h = HelpOverlay::default();
        let mut buf = Buffer::empty(area);
        h.render(&mut buf, area, "Help", &all, &theme);
        assert!(dump(&buf, area)[0].contains("(1-4/10)"));
        assert!(dump(&buf, area)[1].contains("does 0"));

        assert_eq!(
            h.handle_key(key(KeyCode::End), all.len(), page),
            HelpOutcome::Open
        );
        assert_eq!(h.scroll, 6);
        h.handle_key(key(KeyCode::Down), all.len(), page);
        assert_eq!(h.scroll, 6, "clamped at the end");
        let mut buf = Buffer::empty(area);
        h.render(&mut buf, area, "Help", &all, &theme);
        let d = dump(&buf, area);
        assert!(d[0].contains("(7-10/10)"));
        assert!(d[4].contains("does 9"));

        h.handle_key(key(KeyCode::PageUp), all.len(), page);
        assert_eq!(h.scroll, 2);
        h.handle_key(key(KeyCode::Home), all.len(), page);
        assert_eq!(h.scroll, 0);
        h.handle_key(key(KeyCode::Up), all.len(), page);
        assert_eq!(h.scroll, 0);
    }

    #[test]
    fn render_clamps_scroll_after_resize() {
        let mut h = HelpOverlay { scroll: 6 };
        let area = Rect::new(0, 0, 30, 20);
        let mut buf = Buffer::empty(area);
        h.render(&mut buf, area, "Help", &rows(10), &Theme::default());
        assert_eq!(h.scroll, 0);
    }

    #[test]
    fn esc_question_and_q_close() {
        let mut h = HelpOverlay::default();
        for c in [KeyCode::Esc, KeyCode::Char('?'), KeyCode::Char('q')] {
            assert_eq!(h.handle_key(key(c), 5, 3), HelpOutcome::Close);
        }
        assert_eq!(
            h.handle_key(key(KeyCode::Char('x')), 5, 3),
            HelpOutcome::Open
        );
        let mut rel = key(KeyCode::Esc);
        rel.kind = KeyEventKind::Release;
        assert_eq!(h.handle_key(rel, 5, 3), HelpOutcome::Open);
    }

    #[test]
    fn disabled_rows_are_dimmed_and_tiny_areas_are_safe() {
        let area = Rect::new(0, 0, 20, 6);
        let mut buf = Buffer::empty(area);
        let theme = Theme {
            color: true,
            unicode: true,
        };
        HelpOverlay::default().render(&mut buf, area, "Help", &rows(3), &theme);
        assert_eq!(buf[(1, 2)].fg, Color::DarkGray, "disabled row k1");
        assert_ne!(buf[(1, 3)].fg, Color::DarkGray);
        for (w, h) in [(0, 0), (2, 2), (3, 3)] {
            let a = Rect::new(0, 0, w, h);
            HelpOverlay::default().render(&mut Buffer::empty(a), a, "T", &rows(4), &theme);
        }
    }
}
