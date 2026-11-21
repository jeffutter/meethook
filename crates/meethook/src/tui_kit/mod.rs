//! Shared terminal-UI kit: one keymap table that drives key dispatch, the hint footer and the
//! `?` help overlay, plus the theme and pane widgets they draw with.
//!
//! A screen declares its bindings once as a [`Keymap`] (context, key, action, footer hint, help
//! text). The kit does the rest: [`Keymap::dispatch`] turns a key event into an action,
//! [`Keymap::hints`] feeds [`render_footer`], and [`Keymap::help_rows`] feeds [`HelpOverlay`].
//! All three read the same table and format keys with the same function, so a key cannot be
//! bound but missing from the help, or advertised but dead.
//!
//! The enroll screen adopted the kit in MH-42. `record_screen` has not yet (that migration is
//! separate follow-up work, so record behaves exactly as before); the allow below covers the
//! parts only it will use.
#![allow(dead_code, unused_imports)]

mod help;
mod keymap;
mod theme;
mod widgets;

pub use help::{HelpOutcome, HelpOverlay};
pub use keymap::{Binding, Context, HelpRow, Hint, KeySpec, Keymap, key_label};
pub use theme::Theme;
pub use widgets::{bordered_pane, footer_height, hint_lines, render_footer};
