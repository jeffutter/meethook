//! The six panes, given a [`View`]. Every widget in this binary is built here and nowhere else.
//!
//! Nothing in this module decides anything: it takes the derived view the state machine produced
//! and places it. That is what lets it be exercised through [`ratatui::backend::TestBackend`]
//! without a terminal, and what keeps the part that *is* decidable in `state`.
//!
//! # The one place this crate invents wording
//!
//! `narration.rs` in `meethook-enroll` tells an interface not to invent sentences, and it is
//! right: moving a sentence into the thing that displays it moves it out of `cargo test`. The
//! consequence and refusal lines are the exception, and it is worth naming why. `Lines`
//! renders what a run *did* -- past tense, prefixed with a session id -- and there is no `Note`
//! for a dry run. Inventing one would put "what would happen" into a type documented as "one
//! thing a run has to say". So the consequence lines come off
//! [`Consequence::would_do`](meethook_enroll::Consequence) in `super` (the only place able to
//! read a `Consequence`) and the refusal sentence comes off the fully public
//! [`meethook_enroll::Refusal::sentence`] here.
//!
//! The other exception is the "and N more session(s)" line in [`who`], which is layout rather than
//! domain prose -- how much of a list fits in a pane is not a fact about enrolment -- but it is
//! still a sentence invented in this file, so it is named here too. Everything else that pane says
//! about what a name currently names is either
//! [`incomplete`] or `run_speakers`' own wording, so the frame and
//! `meethook speakers` cannot come to describe one scan differently.
//!
//! Two further exceptions are the key-tied UI sentences: the assertion line ([`assert_line`]) and
//! the group header line ([`group_line`]). Their wording is invented here because the keybinding
//! is a fact about the binary -- the run never learns which key pressed its answer -- but their
//! numbers come off the library's own data: each reads its counts off the same
//! [`meethook_enroll::Assertion`] or [`meethook_enroll::GroupConsequence`] the commit reports from,
//! so a preview and its write cannot disagree about what they counted.

use meethook_enroll::{Assertion, GroupConsequence, Resolution, incomplete, speech};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};

use super::Sounding;
use super::keys::Keys;
use super::state::{Candidate, Focus, Mark, Mode, Row, View, Who};
use crate::clips::Progress;
use crate::tui_kit::{HelpOverlay, Hint, Theme, bordered_pane, footer_height, render_footer};

/// Width at and above which the panes sit side by side. Below it they stack: at 80 columns two
/// half-width panes clip the voice rows and candidate names that the whole frame exists to show.
const WIDE: u16 = 100;

/// Where every pane goes for one frame. `None` is a pane that yields when the terminal is too
/// short to give it a useful height, never one that is squeezed to nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Panes {
    pub banner: Option<Rect>,
    pub voices: Rect,
    pub question: Rect,
    pub candidates: Rect,
    pub consequence: Rect,
    pub who: Option<Rect>,
    pub said: Rect,
    pub log: Option<Rect>,
}

/// The side-by-side layout: voices left; question, candidates, consequence and who right; the
/// snippets and the run's log underneath.
///
/// Four bands in the right column. At the 100x24 floor of this layout the top band is 12 to 13
/// rows with a two-row footer, so `1 + 3 + 4 + 5` just fits with the candidate list down to a row
/// or two; at 120x40 the candidates get 19. The "who" pane is adjacent to the candidate it
/// describes, which is the whole point of it -- a horizontal split of the `run` band was
/// considered and rejected, because `log` clips rather than wraps and halving its width would cut
/// narration sentences mid-word.
///
/// The banner row exists only when this session carries a meeting: reserving it either way would
/// cost a session without one a row it has nothing to say.
fn wide_layout(body: Rect, meeting: bool) -> Panes {
    let (banner, top, said, run) = if meeting {
        let whole = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(6),
            Constraint::Length(4),
            Constraint::Length(4),
        ])
        .split(body);
        (Some(whole[0]), whole[1], whole[2], whole[3])
    } else {
        let whole = Layout::vertical([
            Constraint::Min(6),
            Constraint::Length(5),
            Constraint::Length(5),
        ])
        .split(body);
        (None, whole[0], whole[1], whole[2])
    };
    let columns =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(top);
    let right = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(3),
        Constraint::Length(4),
        Constraint::Length(5),
    ])
    .split(columns[1]);
    Panes {
        banner,
        voices: columns[0],
        question: right[0],
        candidates: right[1],
        consequence: right[2],
        who: Some(right[3]),
        said,
        log: Some(run),
    }
}

/// The stacked layout: one column, top to bottom in the order the eye works through a question.
///
/// What has to be there -- banner, the voices, the question, the candidates, the consequence, the
/// snippets -- is sized first; "who" and then the run log are added only while rows are left over
/// (drop order: log first, then who, so the log never appears without "who"), and whatever remains after that goes to the voices and the
/// candidates, a third and two thirds. A pane that cannot have a useful height yields entirely
/// rather than being squeezed to nothing.
fn narrow_layout(body: Rect, meeting: bool) -> Panes {
    const VOICES_MIN: u16 = 4;
    const CANDIDATES_MIN: u16 = 4;
    const CONSEQUENCE: u16 = 4;
    const SAID: u16 = 4;
    const WHO: u16 = 5;
    const LOG: u16 = 4;

    let banner = u16::from(meeting);
    let required = banner + 1 + CONSEQUENCE + SAID + VOICES_MIN + CANDIDATES_MIN;
    let mut spare = body.height.saturating_sub(required);
    let who = (spare >= WHO).then_some(WHO);
    spare -= who.unwrap_or(0);
    let log = (who.is_some() && spare >= LOG).then_some(LOG);
    spare -= log.unwrap_or(0);
    let voices = VOICES_MIN + spare / 3;
    let candidates = CANDIDATES_MIN + spare - spare / 3;

    let mut heights = Vec::new();
    if meeting {
        heights.push(1);
    }
    heights.extend([voices, 1, candidates, CONSEQUENCE]);
    heights.extend(who);
    heights.push(SAID);
    heights.extend(log);
    let rects = Layout::vertical(heights.iter().map(|h| Constraint::Length(*h))).split(body);
    let mut next = rects.iter().copied();
    Panes {
        banner: meeting.then(|| next.next().expect("a banner row")),
        voices: next.next().expect("a voices row"),
        question: next.next().expect("a question row"),
        candidates: next.next().expect("a candidates row"),
        consequence: next.next().expect("a consequence row"),
        who: who.map(|_| next.next().expect("a who row")),
        said: next.next().expect("a snippets row"),
        log: log.map(|_| next.next().expect("a log row")),
    }
}

/// Chosen from the body's width on every call, so a resize re-chooses with nothing stored.
pub fn layout(body: Rect, meeting: bool) -> Panes {
    if body.width >= WIDE {
        wide_layout(body, meeting)
    } else {
        narrow_layout(body, meeting)
    }
}

/// Places every pane for one frame.
///
/// `sounding` is a parameter rather than a [`View`] field on purpose: it is derived from a clock
/// and from what the shell handed to the player, and [`state`](super::state) is documented as
/// having neither in it. A fourth argument says in the signature that the shell computed this and
/// the state machine did not. `keys` is the table the footer and the help overlay read, and
/// `help` the overlay itself when it is open.
pub fn draw(
    frame: &mut Frame,
    view: &View<'_>,
    narration: &[String],
    sounding: Option<Sounding>,
    keys: &Keys,
    help: Option<&mut HelpOverlay>,
) {
    let theme = &keys.theme;
    let area = frame.area();
    // The footer's height is decided first, from the hints at this width, so a narrow terminal
    // gets a taller footer and the panes give the rows back instead of the hints being clipped.
    let hints = keys.hints(view, sounding);
    let status_only = sounding.is_none() && view.status.is_some();
    let lead = u16::from(sounding.is_some());
    let wanted = match status_only {
        true => 1,
        false => lead + footer_height(&hints, area.width, theme),
    };
    let footer_rows = wanted.min(area.height / 2);
    let split = Layout::vertical([Constraint::Min(0), Constraint::Length(footer_rows)]).split(area);
    let panes = layout(split[0], view.meeting.is_some());

    if let Some(banner) = panes.banner {
        meeting_banner(frame, banner, view);
    }
    voices(frame, panes.voices, view, theme);
    question(frame, panes.question, view);
    candidates(frame, panes.candidates, view, theme);
    consequence(frame, panes.consequence, view);
    if let Some(area) = panes.who {
        who(frame, area, view);
    }
    snippets(frame, panes.said, view, sounding.and_then(|s| s.line));
    if let Some(area) = panes.log {
        log(frame, area, narration);
    }
    footer(frame, split[1], view, sounding, &hints, theme);

    if let Some(help) = help {
        let rows = keys.help_rows(view);
        help.render(frame.buffer_mut(), area, " keys ", &rows, theme);
    }
}

/// The phrase the banner row begins with: the line prompt's `    meeting   ` minus its
/// indent, so the frame and the plain prompt say the same thing about the same meeting.
const MEETING_PREFIX: &str = "meeting   ";

/// Which meeting this session was recorded during, in its own row above the panes.
///
/// The voices pane's title cannot hold it: at the 80x24 floor that pane is 40 columns wide and
/// its border leaves 38 for a title that already runs to 31, and ratatui clips a block title
/// silently. A row of its own, present only when there is a meeting, is what keeps the frame
/// from asserting a clipped title as the whole one.
///
/// A bare [`Paragraph`], styled like the question band: a one-row bordered block has no
/// interior. The phrase is the one the line prompt prints under the count line, minus the
/// indent, so the two surfaces say the same thing about the same meeting -- and when the row
/// runs short, [`clause_within`] decides which half of it yields.
fn meeting_banner(frame: &mut Frame, area: Rect, view: &View<'_>) {
    let Some(meeting) = view.meeting else { return };
    let width = area.width.saturating_sub(MEETING_PREFIX.len() as u16) as usize;
    let clause = clause_within(&meeting.title, meeting.fit.caveat(), width);
    frame.render_widget(
        Paragraph::new(Line::from(format!("{MEETING_PREFIX}{clause}")).bold()),
        area,
    );
}

/// [`MeetingLabel::clause`](meethook_enroll::MeetingLabel::clause) fitted to a row: the whole
/// clause when it fits, otherwise the title cut with an ellipsis and the caveat kept whole.
///
/// The caveat is the safety device -- a bare title would assert a match the tool does not have
/// -- so it is never the half that yields while there is a title to yield instead. Only when
/// the caveat alone outgrows the row -- the 80-column floor with the longest of them -- does it
/// take the cut itself, from the end, keeping the word that says the match is not strong.
fn clause_within(title: &str, caveat: Option<&str>, width: usize) -> String {
    let Some(caveat) = caveat else {
        return within(title, width);
    };
    let clause = format!("{title}  ({caveat})");
    if clause.chars().count() <= width {
        return clause;
    }
    let decorated = format!("  ({caveat})");
    let room = width.saturating_sub(decorated.chars().count());
    if room > 0 {
        return format!("{}{decorated}", within(title, room));
    }
    within(&format!("({caveat})"), width)
}

/// The longest prefix of `text` that fits `width` characters, with an ellipsis standing in for
/// what was cut.
///
/// Marking the cut is the point: a shortened title that looks complete reads as the whole one.
/// Character counts rather than cell widths, the way the footer assumes of its key list -- a
/// title wider than its character count clips at the buffer's edge, which is a degradation
/// rather than a failure.
fn within(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    if width == 0 {
        return String::new();
    }
    let mut out: String = text.chars().take(width - 1).collect();
    out.push('…');
    out
}

/// The voice queue: every voice the session has, with the quiet ones under a separator.
///
/// A separator rather than a count at the end, because a user hunting for their own two-second
/// fragment has to be able to see that it exists at all.
fn voices(frame: &mut Frame, area: Rect, view: &View<'_>, theme: &Theme) {
    let mut items: Vec<ListItem> = Vec::with_capacity(view.rows.len() + 1);
    let mut selected = view.cursor;
    let mut separated = false;
    for (index, row) in view.rows.iter().enumerate() {
        if row.below_floor && !separated {
            separated = true;
            items.push(ListItem::new(Line::from(
                "-- quieter than the prompt floor --".dim(),
            )));
            if index <= view.cursor {
                selected += 1;
            }
        }
        items.push(ListItem::new(voice_line(row)));
    }

    let title = format!(" voices  {}  {} ", view.session, view.position);
    let list = List::new(items)
        .block(bordered_pane(&title, view.focus == Focus::Voices, theme))
        .highlight_symbol("> ")
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
    let mut state = ListState::default().with_selected(Some(selected));
    frame.render_stateful_widget(list, area, &mut state);
}

fn voice_line(row: &Row) -> Line<'static> {
    let mut spans = match row.fragments {
        // A composite stands in for its members: the count and the total talk time say what
        // answering would commit, not what any one member said, and "most like Ivan" rides the
        // label/similarity pair the ordinary rows use, so the composite borrows their styling.
        // An empty label is a bundle with nothing enrolled to resemble, and carries no clause.
        Some(count) => {
            let mut spans = vec![Span::raw(format!("{:<12} ", row.number))];
            spans.push(Span::raw(format!(
                "{count} fragments · {}",
                speech(row.speech_seconds)
            )));
            if !row.label.is_empty() {
                spans.push(Span::raw(" · ").dim());
                spans.push(Span::raw(row.label.clone()).bold());
                if let Some(similarity) = row.similarity {
                    spans.push(Span::raw(format!(" ({similarity:.2})")).dim());
                }
            }
            spans.push(Span::raw("  "));
            spans
        }
        None => {
            let mut spans = vec![Span::raw(format!(
                "{:<12} {:>8}  ",
                row.number,
                speech(row.speech_seconds)
            ))];
            if row.label == row.number {
                spans.push(Span::raw("--".to_string()).dim());
            } else {
                spans.push(Span::raw(row.label.clone()).bold());
                if let Some(similarity) = row.similarity {
                    spans.push(Span::raw(format!(" {similarity:.2}")).dim());
                }
            }
            spans
        }
    };
    // A guess is the database's claim, not a naming the user made: the tag says so beside the
    // similarity, the way [group] sits beside the decision marks rather than instead of them.
    if row.guess {
        spans.push(Span::raw("  [guess]"));
    }
    if let Some(mark) = row.mark {
        spans.push(Span::raw(match mark {
            Mark::Answered => "  [named]",
            Mark::Skipped => "  [skipped]",
            Mark::Deferred => "  [later]",
        }));
    }
    // Beside the decision mark, not instead of it: a row skipped while staged renders both,
    // which is honest -- it was skipped and it is still staged.
    if row.in_group {
        spans.push(Span::raw("  [group]"));
    }
    // Which row is the question, as distinct from which row the cursor is on. They part company
    // the moment the user moves the cursor to line up the next voice, and without this the queue
    // would stop saying which one the candidates pane is about.
    if row.current {
        spans.push(Span::raw("  <- asking").dim());
    }
    let line = Line::from(spans);
    if row.below_floor { line.dim() } else { line }
}

/// The question itself, over the candidates that answer it.
///
/// Two questions rather than one, for the reason the line prompt gives: an already-named voice is
/// asking "is this right", not "who is this", and putting a name on the screen under the first
/// wording invites the user to type it straight back in.
fn question(frame: &mut Frame, area: Rect, view: &View<'_>) {
    let heard = speech(view.speech_seconds);
    let text = match view.label == view.number {
        true => format!(" who is {}?  {heard}", view.number),
        false => {
            // A guess owns its own question mark in the label; appending the frame's would
            // double it. Driven off the attribution kind, not a suffix sniff: a person named
            // "Ivan?" would otherwise lose their mark.
            let mark = match view.guess {
                Some(_) => "",
                None => "?",
            };
            format!(" is {} {}{mark}  {heard}", view.number, view.label)
        }
    };
    frame.render_widget(Paragraph::new(Line::from(text).bold()), area);
}

/// The candidates, narrowed as the user types, plus the always-present create-somebody entry.
fn candidates(frame: &mut Frame, area: Rect, view: &View<'_>, theme: &Theme) {
    let mut items: Vec<ListItem> = view
        .candidates
        .iter()
        .map(|candidate| ListItem::new(candidate_line(candidate)))
        .collect();
    // Its own entry, always there, and actuated by its own key -- never the thing that happens
    // when text matches nothing.
    items.push(ListItem::new(Line::from(new_person(view))));

    // A trailing underscore is the cursor: it is what says printable keys are being typed rather
    // than obeyed, which is the one thing about the mode that has to be visible.
    let title = match (view.mode, view.filter.is_empty()) {
        (Mode::Filter, _) => format!(" filter: {}_ ", view.filter),
        (Mode::Normal, true) => " resembles ".to_string(),
        (Mode::Normal, false) => format!(" filter: {} ", view.filter),
    };
    let list = List::new(items)
        .block(bordered_pane(
            &title,
            view.focus == Focus::Candidates,
            theme,
        ))
        .highlight_symbol("> ")
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
    let mut state = ListState::default().with_selected(view.candidate);
    frame.render_stateful_widget(list, area, &mut state);
}

fn candidate_line(candidate: &Candidate) -> Line<'static> {
    let numbers = match (candidate.similarity, candidate.references) {
        (Some(similarity), Some(references)) => format!("{similarity:>6.2}  {references:>2} ref"),
        // Reachable by typing but absent from the ranking: every stored recording of them is a
        // stale embedding dimension. Blank rather than a zero, which would read as "sounds
        // nothing like this voice" instead of "not comparable".
        _ => "     --      ".to_string(),
    };
    let mut spans = vec![
        Span::raw(format!("{:<20} ", candidate.name)),
        Span::raw(numbers).dim(),
    ];
    // The tag here and the sentence in the "would" pane below, rather than both on this row: a
    // refusal takes a clause to explain -- which other voice, and what it would lose -- and a
    // clause does not fit beside a name and two numbers in half a terminal. Truncated to
    // "unavailable: Unknow" it would say less than the tag does.
    if candidate.refusal.is_some() {
        spans.push(Span::raw("  [unavailable]"));
    }
    let line = Line::from(spans);
    if candidate.refusal.is_some() {
        line.dim()
    } else {
        line
    }
}

/// The create-somebody entry, which reads differently when nobody enrolled is plausible but is
/// the same entry either way.
fn new_person(view: &View<'_>) -> Vec<Span<'static>> {
    match view.filter.trim().is_empty() {
        true => vec![Span::raw("+ somebody new (type a name first)").dim()],
        false => {
            let typed = view.filter.trim().to_string();
            let note = match view.resolution {
                Resolution::New(_) => "  (nobody enrolled matches)",
                _ => "",
            };
            vec![Span::raw(format!(
                "+ enrol \"{typed}\" as somebody new{note}"
            ))]
        }
    }
}

/// What choosing the highlighted candidate would do, before it is chosen -- or why it cannot be.
///
/// One pane for both, because they are the same question asked of the same row and only one of
/// them ever has an answer: a refused candidate would do nothing at all.
fn consequence(frame: &mut Frame, area: Rect, view: &View<'_>) {
    // The staged group previews the operation the user is about to perform -- what committing
    // it with the highlighted name would do -- so while it is on screen the pane shows only
    // that: two previews in a two-inner-row pane is noise, and the group's own lines already
    // carry the refusal of every member it refused. The bundle owns the pane the same way: it
    // is the operation the answering keys will perform, and staging is refused while it is open,
    // so the two doors never contend here.
    let (title, lines) = if let Some(bundle) = view.bundle {
        (
            " would ",
            std::iter::once(Line::from(bundle_line(bundle)))
                .chain(bundle.would_do().into_iter().map(Line::from))
                .collect::<Vec<_>>(),
        )
    } else if let Some(group) = view.group {
        (
            " would ",
            std::iter::once(Line::from(group_line(group)))
                .chain(group.would_do().into_iter().map(Line::from))
                .collect::<Vec<_>>(),
        )
    } else {
        let highlighted = view
            .highlighted()
            .and_then(|candidate| candidate.refusal.as_ref());
        let (title, mut lines): (&str, Vec<Line>) = match (highlighted, view.consequence.is_empty())
        {
            (Some(refusal), _) => (" cannot ", vec![Line::from(refusal.sentence())]),
            (None, true) => (
                " would ",
                vec![Line::from(Span::raw("(nothing highlighted)").dim())],
            ),
            (None, false) => (
                " would ",
                view.consequence
                    .iter()
                    .map(|line| Line::from(line.as_str()))
                    .collect(),
            ),
        };
        // What asserting one remote speaker would do to the *session*, previewed beside what
        // choosing would do to the *voice*: the two numbers the commit reports, off the same
        // [`Assertion`]. Present on a refused row too -- Enter cannot work there, but the
        // assertion can, and both facts sit in the pane.
        if let (Some(assertion), Some(candidate)) = (view.assertion, view.highlighted()) {
            lines.push(Line::from(assert_line(&candidate.name, &assertion)));
        }
        (title, lines)
    };
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title(title)),
        area,
    );
}

/// The pane's assertion line, in the run's own labelling: the same "voice(s)" and "veto(s)
/// overridden" the summary note prints once the assertion has run, so a preview and its write
/// cannot disagree about what they counted.
fn assert_line(name: &str, assertion: &Assertion) -> String {
    format!(
        "[a] asserts one remote speaker: {voices} voice(s) will read as {name}, {vetoes} veto(s) overridden",
        voices = assertion.voices,
        vetoes = assertion.vetoes_overridden,
    )
}

/// The pane's group header line, in the run's own labelling: the same "voice(s)" and "veto(s)
/// overridden" the group commit reports, so a preview and its write cannot disagree about what
/// they counted. The key prefix is the binary's fact; the numbers come off the same
/// [`GroupConsequence`] the commit reports from.
fn group_line(group: &GroupConsequence) -> String {
    format!(
        "[m] marks {} voice(s) as {}: {} reference(s), {} veto(s) overridden",
        group.applied.len(),
        group.name,
        group.references_after,
        group.vetoes_overridden
    )
}

/// The pane's bundle header line, in the run's own labelling: the same "reference(s)" and
/// "veto(s) overridden" the fan-out reports once the answer has walked, so a preview and its
/// write cannot disagree about what they counted. The numbers come off the same
/// [`GroupConsequence`] the fan-out reports from.
fn bundle_line(bundle: &GroupConsequence) -> String {
    format!(
        "answering names {} fragment(s) as {}: {} reference(s), {} veto(s) overridden",
        bundle.applied.len(),
        bundle.name,
        bundle.references_after,
        bundle.vetoes_overridden
    )
}

/// Who the highlighted candidate already is: how many recordings of them the database holds, and
/// which voices in which meetings read that name because of them.
///
/// The pane that makes "who is Ivan again?" answerable without leaving the prompt, which is why it
/// sits directly under the candidate list rather than anywhere roomier. Three rows of content at
/// every terminal size, so what fits is decided by [`listed`] rather than by hoping.
///
/// Every sentence here is either `run_speakers`' own or [`incomplete`]'s, so the frame and
/// `meethook speakers` cannot come to describe one scan differently -- including the scope clause
/// in "naming nothing in any session read", which is what keeps that claim honest about having
/// read only the sessions under this root.
///
/// Not wrapped, for the reason `log` is not: a row per fact, clipped at the pane's edge, so a long
/// session list cannot push the incompleteness line off the bottom.
fn who(frame: &mut Frame, area: Rect, view: &View<'_>) {
    let highlighted = view.highlighted().map(|candidate| candidate.name.as_str());
    // Dynamic like the candidates pane's title already is: the pane is about one person, and
    // naming them in the border is what stops the rows below reading as being about the voice.
    let title = match highlighted {
        Some(name) => format!(" who {name} is "),
        None => " who ".to_string(),
    };

    let mut lines: Vec<Line> = Vec::new();
    match &view.who {
        // The exact phrase the " would " pane uses for the same state, because it is the same
        // state: there is no candidate under the highlight to say anything about.
        Who::Nobody => lines.push(Line::from(Span::raw("(nothing highlighted)").dim())),
        Who::Reading => lines.push(Line::from(Span::raw("reading the sessions...").dim())),
        Who::Failed(why) => lines.push(Line::from(format!(
            "could not read the enrolled speakers: {why}"
        ))),
        Who::Unrecorded => lines.push(Line::from(
            "enrolled during this run, so nothing has been read for them yet",
        )),
        Who::Known {
            references,
            voices,
            sessions,
            unreadable,
        } => {
            lines.push(Line::from(match voices {
                0 => format!("{references} reference(s), naming nothing in any session read"),
                voices => format!(
                    "{references} reference(s), naming {voices} voice(s) in {} session(s)",
                    sessions.len()
                ),
            }));
            // The incompleteness line is claimed out of the budget before the sessions are, so a
            // person naming forty voices cannot cost the sentence that says the answer is partial.
            let room = area.height.saturating_sub(2) as usize;
            let (shown, more) = listed(sessions.len(), room.saturating_sub(1 + *unreadable));
            for named in &sessions[..shown] {
                lines.push(Line::from(format!(
                    "  {}  {}",
                    named.session,
                    named.voices.join(", ")
                )));
            }
            if more {
                lines.push(Line::from(
                    Span::raw(format!(
                        "  ... and {} more session(s)",
                        sessions.len() - shown
                    ))
                    .dim(),
                ));
            }
            if *unreadable > 0 {
                lines.push(Line::from(Span::raw(incomplete(*unreadable)).dim()));
            }
        }
    }

    frame.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title)),
        area,
    );
}

/// How many sessions the pane lists, and whether it owes a line saying how many it left out.
///
/// Its own function because it is the one piece of arithmetic in this file that `cargo test` can
/// reach, and getting it wrong is a pane that silently drops the last session or spends a row
/// saying nothing.
///
/// With a budget of one row, that row goes to a session rather than to a count: the summary line
/// above already says how many sessions there are, so "... and 3 more session(s)" over an empty
/// list would be the same number twice and no session at all.
fn listed(sessions: usize, budget: usize) -> (usize, bool) {
    match sessions {
        sessions if sessions <= budget => (sessions, false),
        _ if budget <= 1 => (budget, false),
        _ => (budget - 1, true),
    }
}

/// What this voice said, with the selected line marked and timed. Every snippet is here; which
/// one is selected is the state machine's, and which one is sounding is the shell's.
///
/// The selected line is the row at the top of the pane, so the marker is always on the first
/// visible row -- one index rather than a cursor and an offset, which is what keeps the clamping
/// to a single rule in `state`.
///
/// The time is spelled with [`speech`], which is what the question, the queue and the footer
/// already use for a duration, so no second time formatter appears in this binary. It reads as
/// "said this far in" rather than as a clock, and it rounds -- two lines less than a second apart
/// can show the same time. That is fine: the marker says which row is selected, and the time is
/// orientation rather than an index.
fn snippets(frame: &mut Frame, area: Rect, view: &View<'_>, sounding: Option<usize>) {
    let title = match view.snippets.len() {
        0 => " said  (nothing was transcribed for this voice) ".to_string(),
        total => format!(" said  line {} of {total}  [ ] moves ", view.snippet + 1),
    };
    let lines: Vec<Line> = view
        .snippets
        .iter()
        .enumerate()
        .skip(view.snippet)
        .map(|(index, snippet)| {
            let selected = index == view.snippet;
            let mut spans = vec![
                // The same marker the voices and the candidates panes use for their selections,
                // so a third selectable pane looks like the other two.
                Span::raw(if selected { "> " } else { "  " }),
                Span::raw(format!("{:>7}  ", speech(snippet.start))).dim(),
            ];
            let text = Span::raw(format!("\"{}\"", snippet.text));
            spans.push(if selected { text.bold() } else { text });
            // Two different facts about a row -- which one is selected, which one is being heard
            // -- each a suffixed clause, in the shape `voice_line` uses for "<- asking".
            if sounding == Some(index) {
                spans.push(Span::raw("  <- playing").dim());
            }
            Line::from(spans)
        })
        .collect();
    frame.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title)),
        area,
    );
}

/// The run's own narration, placed in the frame rather than scrolled past it.
fn log(frame: &mut Frame, area: Rect, narration: &[String]) {
    let height = area.height.saturating_sub(2) as usize;
    let tail = narration.len().saturating_sub(height);
    let lines: Vec<Line> = narration[tail..]
        .iter()
        .map(|line| Line::from(line.as_str()))
        .collect();
    frame.render_widget(
        Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" run ")),
        area,
    );
}

/// The footer: how far through a clip playback has got, else whatever just happened, else the keys.
///
/// Playback outranks the status because it is the only one of the three that changes on its own,
/// and because the two cannot both be true: a play that failed clears the progress and sets the
/// status in the same iteration. While a clip sounds, its position is the first row and the
/// hints under it are only the four keys that mean something mid-clip. The position is spelled
/// with [`speech`], which is already what the question and the queue use for a duration.
///
/// The hints come from the key table, already filtered to the keys that work for this selection
/// and wrapped to the width by the kit; the caller has sized `area` to fit them, so nothing is
/// clipped.
fn footer(
    frame: &mut Frame,
    area: Rect,
    view: &View<'_>,
    sounding: Option<Sounding>,
    hints: &[Hint],
    theme: &Theme,
) {
    match (sounding, view.status) {
        (
            Some(Sounding {
                progress: Progress { elapsed, length },
                ..
            }),
            _,
        ) => {
            let rows = Layout::vertical([Constraint::Length(1), Constraint::Min(0)]).split(area);
            let text = format!(
                "playing {} of {}",
                speech(elapsed.as_secs_f64()),
                speech(length.as_secs_f64())
            );
            frame.render_widget(Paragraph::new(Line::from(text).dim()), rows[0]);
            render_footer(frame.buffer_mut(), rows[1], hints, theme);
        }
        (None, Some(status)) => {
            frame.render_widget(Paragraph::new(Line::from(status.to_string()).dim()), area);
        }
        (None, None) => render_footer(frame.buffer_mut(), area, hints, theme),
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use meethook_enroll::{
        Assertion, FragmentGroup, GroupConsequence, MeetingLabel, Position, Queued, Refusal,
    };
    use meethook_session::{MeetingFit, SessionId};
    use meethook_transcribe::{Attribution, Resemblance};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    use meethook_enroll::Snippet;

    use super::super::keys::Keys;
    use super::super::state::tests::heard;
    use super::super::state::{Context, Cost, Costs, Event, Mark, Row, Screen, VoiceView};
    use super::super::who::tests::{holding, names, scanned};
    use super::{Progress, Sounding, clause_within, draw, incomplete, layout, listed, voice_line};
    use crate::tui_kit::{HelpOverlay, Theme};

    use meethook_session::Displaced;

    struct Free;

    impl Costs for Free {
        fn of(&self, name: &str) -> Cost {
            Cost {
                refusal: None,
                summary: vec![format!("would enrol {name} from this voice")],
                assertion: None,
            }
        }

        fn group_of(&self, _name: &str, _members: &[&str]) -> Option<GroupConsequence> {
            None
        }

        fn fragment_group_of(&self, _name: &str, _members: &[&str]) -> Option<GroupConsequence> {
            None
        }
    }

    /// [`Free`]'s counterpart where the highlighted candidate also previews an assertion: the
    /// pane must show what asserting would do beside what choosing would do.
    struct Asserts;

    impl Costs for Asserts {
        fn of(&self, name: &str) -> Cost {
            Cost {
                refusal: None,
                summary: vec![format!("would enrol {name} from this voice")],
                assertion: Some(Assertion {
                    voices: 4,
                    vetoes_overridden: 1,
                }),
            }
        }

        fn group_of(&self, _name: &str, _members: &[&str]) -> Option<GroupConsequence> {
            None
        }

        fn fragment_group_of(&self, _name: &str, _members: &[&str]) -> Option<GroupConsequence> {
            None
        }
    }

    struct Vetoes;

    impl Costs for Vetoes {
        fn of(&self, _name: &str) -> Cost {
            Cost {
                refusal: Some(Refusal::Vetoed {
                    holder: Some("Unknown 2".to_string()),
                }),
                summary: Vec::new(),
                assertion: None,
            }
        }

        fn group_of(&self, _name: &str, _members: &[&str]) -> Option<GroupConsequence> {
            None
        }

        fn fragment_group_of(&self, _name: &str, _members: &[&str]) -> Option<GroupConsequence> {
            None
        }
    }

    /// Everything is refused for taking a name off another voice: the one refusal an answer can
    /// override, and so the one the footer offers a key for.
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

        fn group_of(&self, _name: &str, _members: &[&str]) -> Option<GroupConsequence> {
            None
        }

        fn fragment_group_of(&self, _name: &str, _members: &[&str]) -> Option<GroupConsequence> {
            None
        }
    }

    /// The lines the fixture voice said, as a real voice arrives with them: each said at a moment
    /// in the recording, each with samples of its own.
    fn said() -> [Snippet<'static>; 2] {
        static AUDIO: [f32; 2] = [0.1, 0.2];
        [
            heard("so where did we land on the migration", 12.0, &AUDIO),
            heard("right, next week", 107.0, &AUDIO),
        ]
    }

    /// The whole frame as text, one string per terminal row, so an assertion can name what it
    /// expects to see rather than pinning every cell.
    fn painted(width: u16, height: u16, costs: &dyn Costs, keys: &[Event]) -> Vec<String> {
        painted_with(
            width,
            height,
            costs,
            keys,
            &said(),
            None,
            None,
            Context::Reading,
            None,
        )
    }

    /// [`painted`], with a cross-session scan already gathered. The "who" pane is the only thing
    /// the context reaches, so its tests vary that and nothing else.
    fn knowing(width: u16, height: u16, keys: &[Event], context: Context<'_>) -> Vec<String> {
        painted_with(
            width,
            height,
            &Free,
            keys,
            &said(),
            None,
            None,
            context,
            None,
        )
    }

    /// [`painted`], plus the five things only the banner, snippet, footer and "who" tests vary:
    /// what this voice said, what is sounding, what the frame last had to say, what has been
    /// read about the enrolled speakers, and the meeting the seam named for the session. The
    /// middle two reach the footer from different directions -- one is a parameter to `draw`,
    /// the other a field of the view -- which is exactly what the precedence between them has
    /// to be pinned against.
    #[allow(clippy::too_many_arguments)]
    fn painted_with(
        width: u16,
        height: u16,
        costs: &dyn Costs,
        keys: &[Event],
        snippets: &[Snippet<'_>],
        sounding: Option<Sounding>,
        status: Option<&str>,
        context: Context<'_>,
        meeting: Option<&MeetingLabel>,
    ) -> Vec<String> {
        frame(
            width,
            height,
            costs,
            keys,
            snippets,
            sounding,
            status,
            context,
            meeting,
            Attribution::Unknown("Unknown 2".to_string()),
            vec!["20260819-100000  3 voice(s) to ask about".to_string()],
        )
    }

    /// [`painted_with`], with the middle voice carrying a marked guess rather than an
    /// unconfident number: the row the guess tests ask about. Ivan is already in the
    /// fixture's ranking, so the guess is reachable by Enter+Tab and both guess keys are live.
    fn guessed(width: u16, height: u16, costs: &dyn Costs, keys: &[Event]) -> Vec<String> {
        frame(
            width,
            height,
            costs,
            keys,
            &said(),
            None,
            None,
            Context::Reading,
            None,
            Attribution::Tentative {
                name: "Ivan".to_string(),
                similarity: 0.38,
            },
            vec!["20260819-100000  3 voice(s) to ask about".to_string()],
        )
    }

    /// The whole frame as text, one string per terminal row, given the fixture's middle voice
    /// and the run's narration: everything above delegates here with the defaults.
    #[allow(clippy::too_many_arguments)]
    fn frame(
        width: u16,
        height: u16,
        costs: &dyn Costs,
        keys: &[Event],
        snippets: &[Snippet<'_>],
        sounding: Option<Sounding>,
        status: Option<&str>,
        context: Context<'_>,
        meeting: Option<&MeetingLabel>,
        middle: Attribution,
        narration: Vec<String>,
    ) -> Vec<String> {
        let buffer = frame_buffer(
            width, height, costs, keys, snippets, sounding, status, context, meeting, middle,
            narration, false,
        );
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect()
    }

    /// [`frame`] as the buffer, so a test can look at styles as well as text, and with the help
    /// overlay open when asked.
    #[allow(clippy::too_many_arguments)]
    fn frame_buffer(
        width: u16,
        height: u16,
        costs: &dyn Costs,
        events: &[Event],
        snippets: &[Snippet<'_>],
        sounding: Option<Sounding>,
        status: Option<&str>,
        context: Context<'_>,
        meeting: Option<&MeetingLabel>,
        middle: Attribution,
        narration: Vec<String>,
        help_open: bool,
    ) -> ratatui::buffer::Buffer {
        let session = SessionId::parse("20260819-100000").expect("a well-formed session id");
        let labels = [
            (
                "Unknown 1".to_string(),
                Attribution::Identified {
                    name: "Milo".to_string(),
                    similarity: 0.81,
                },
                240.0,
                false,
            ),
            ("Unknown 2".to_string(), middle, 95.0, false),
            (
                "Unknown 3".to_string(),
                Attribution::Unknown("Unknown 3".to_string()),
                1.5,
                true,
            ),
        ];
        let queue: Vec<Queued<'_>> = labels
            .iter()
            .map(|(number, attribution, seconds, below)| Queued {
                number,
                attribution,
                speech_seconds: *seconds,
                below_floor: *below,
            })
            .collect();
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
        let enrolled = ["Milo", "Ivan"];
        let voice = VoiceView {
            session: &session,
            meeting,
            position: Position { nth: 2, of: 3 },
            number: "Unknown 2",
            speech_seconds: 95.0,
            attribution: &labels[1].1,
            queue: &queue,
            snippets,
            resembles: &resembles,
            enrolled: &enrolled,
            fragment_groups: &[],
            bundle_members: None,
            clip_is_empty: false,
        };

        let mut screen = Screen::default();
        screen.arrive(&voice);
        for event in events {
            screen.answer(&voice, *event, costs);
        }
        // After the keys: `answer` clears the status, which is the behaviour a caller asking for
        // one is working around.
        if let Some(status) = status {
            screen.say(status.to_string());
        }
        let view = screen.view(&voice, costs, context);

        let keys = Keys::new(Theme::default());
        let mut help = help_open.then(HelpOverlay::default);
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("a test backend");
        terminal
            .draw(|frame| draw(frame, &view, &narration, sounding, &keys, help.as_mut()))
            .expect("drawing into a buffer cannot fail");
        terminal.backend().buffer().clone()
    }

    /// The six panes, at a comfortable size: the queue with its talk times and its separator for
    /// the quiet voices, the ranked candidates with their numbers, the consequence, who the
    /// highlighted candidate is, the snippets and the run's own narration.
    #[test]
    fn the_frame_places_all_six_panes() {
        let painted = painted(140, 30, &Free, &[]);
        let whole = painted.join("\n");
        assert!(whole.contains("voices  20260819-100000  2/3"), "{whole}");
        assert!(whole.contains("who Milo is"), "{whole}");
        assert!(whole.contains("Unknown 1"), "{whole}");
        assert!(whole.contains("Milo"), "{whole}");
        assert!(whole.contains("quieter than the prompt floor"), "{whole}");
        assert!(whole.contains("0.71"), "{whole}");
        assert!(whole.contains("3 ref"), "{whole}");
        assert!(
            whole.contains("would enrol Milo from this voice"),
            "{whole}"
        );
        assert!(whole.contains("so where did we land"), "{whole}");
        assert!(whole.contains("3 voice(s) to ask about"), "{whole}");
        assert!(whole.contains("+ somebody new"), "{whole}");
        assert!(whole.contains("[s] skip"), "{whole}");
    }

    /// [`frame`], with the quiet tail folded into one open bundle: two above-floor voices, then
    /// three below-floor fragments the run bundled and offers as the question about its first
    /// member. The composite-row tests vary nothing else.
    fn bundled(width: u16, height: u16, costs: &dyn Costs, keys: &[Event]) -> Vec<String> {
        let session = SessionId::parse("20260819-100000").expect("a well-formed session id");
        let labels = [
            (
                "Unknown 1".to_string(),
                Attribution::Identified {
                    name: "Milo".to_string(),
                    similarity: 0.81,
                },
                240.0,
                false,
            ),
            (
                "Unknown 2".to_string(),
                Attribution::Unknown("Unknown 2".to_string()),
                95.0,
                false,
            ),
            (
                "Unknown 3".to_string(),
                Attribution::Unknown("Unknown 3".to_string()),
                1.5,
                true,
            ),
            (
                "Unknown 4".to_string(),
                Attribution::Unknown("Unknown 4".to_string()),
                0.9,
                true,
            ),
            (
                "Unknown 5".to_string(),
                Attribution::Unknown("Unknown 5".to_string()),
                2.0,
                true,
            ),
        ];
        let queue: Vec<Queued<'_>> = labels
            .iter()
            .map(|(number, attribution, seconds, below)| Queued {
                number,
                attribution,
                speech_seconds: *seconds,
                below_floor: *below,
            })
            .collect();
        let members = vec![
            "Unknown 3".to_string(),
            "Unknown 4".to_string(),
            "Unknown 5".to_string(),
        ];
        let groups = [FragmentGroup {
            members: members.clone(),
            speech_seconds: 4.4,
            best: Some(Resemblance {
                name: "Ivan".to_string(),
                similarity: 0.71,
                references: 1,
            }),
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
        let enrolled = ["Milo", "Ivan"];
        let snippets = said();
        let narration: Vec<String> = Vec::new();
        let voice = VoiceView {
            session: &session,
            meeting: None,
            position: Position { nth: 3, of: 5 },
            number: "Unknown 3",
            speech_seconds: 1.5,
            attribution: &labels[2].1,
            queue: &queue,
            snippets: &snippets,
            resembles: &resembles,
            enrolled: &enrolled,
            fragment_groups: &groups,
            bundle_members: Some(&members),
            clip_is_empty: false,
        };

        let mut screen = Screen::default();
        screen.arrive(&voice);
        for key in keys {
            screen.answer(&voice, *key, costs);
        }
        let view = screen.view(&voice, costs, Context::Reading);

        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("a test backend");
        terminal
            .draw(|frame| {
                draw(
                    frame,
                    &view,
                    &narration,
                    None,
                    &Keys::new(Theme::default()),
                    None,
                )
            })
            .expect("drawing into a buffer cannot fail");
        let buffer = terminal.backend().buffer().clone();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect()
    }

    /// AC from the drawing side: an open bundle renders as one composite row -- the count and
    /// total talk time say what answering would commit, and the resemblance rides the
    /// label/similarity pair the ordinary rows use -- while the members behind it are hidden
    /// rather than rendered as rows of their own.
    #[test]
    fn an_open_bundle_renders_as_one_composite_row_on_the_frame() {
        let painted = bundled(140, 30, &Free, &[]);
        let whole = painted.join("\n");
        // The composite stands in for its three members, reporting the fan-out's numbers and
        // resemblance rather than any single fragment's.
        assert!(whole.contains("3 fragments"), "{whole}");
        assert!(whole.contains("most like Ivan"), "{whole}");
        // The fold hides the members behind their composite; only the anchor's handle survives.
        assert!(
            !whole.contains("Unknown 4"),
            "hidden members must not render: {whole}"
        );
        assert!(
            !whole.contains("Unknown 5"),
            "hidden members must not render: {whole}"
        );
        assert!(
            whole.contains("Unknown 3"),
            "the composite keeps its anchor's handle: {whole}"
        );
        // The floor separator still marks where the quiet tail begins.
        assert!(whole.contains("quieter than the prompt floor"), "{whole}");
    }

    /// AC #7 from the drawing side: the reason is on the screen, beside the candidate it refuses.
    #[test]
    fn a_refused_candidate_says_why_on_the_frame() {
        let painted = painted(120, 30, &Vetoes, &[]);
        let whole = painted.join("\n");
        assert!(whole.contains("[unavailable]"), "{whole}");
        assert!(whole.contains("cannot"), "{whole}");
        assert!(
            whole.contains("Unknown 2 was heard at the same time"),
            "{whole}"
        );
    }

    /// A clip that is playing says how far through it is, in the frame's own wording for a
    /// duration, and gives the line over to the keys that still mean something mid-clip.
    #[test]
    fn a_playing_clip_says_how_far_through_it_is() {
        let sounding = Some(Sounding {
            progress: Progress {
                elapsed: Duration::from_secs(12),
                length: Duration::from_secs(107),
            },
            line: None,
        });
        let whole = painted_with(
            110,
            30,
            &Free,
            &[],
            &said(),
            sounding,
            None,
            Context::Reading,
            None,
        )
        .join("\n");
        assert!(whole.contains("playing 12s of 1m 47s"), "{whole}");
        assert!(whole.contains("[p] restart"), "{whole}");
        assert!(
            !whole.contains("[w] work on"),
            "the key list gives way to the position\n{whole}"
        );
    }

    /// TASK-049 acceptance criterion #6: the key that leaves the session is named, both idle and
    /// mid-clip, and it reads as its own scope beside the two it sits between -- skip one voice,
    /// leave this session, quit the run.
    ///
    /// Painted at 140 columns, as the candidate-and-cost tests are: the key list is past 130 now
    /// that it advertises the mark key too, and a narrower frame is measuring truncation
    /// rather than the footer's wording.
    #[test]
    fn the_footer_names_the_key_that_leaves_the_session() {
        let idle = painted(140, 30, &Free, &[]).join("\n");
        assert!(idle.contains("[s] skip | [x] leave"), "{idle}");

        let sounding = Some(Sounding {
            progress: Progress {
                elapsed: Duration::from_secs(12),
                length: Duration::from_secs(107),
            },
            line: None,
        });
        let playing = painted_with(
            140,
            30,
            &Free,
            &[],
            &said(),
            sounding,
            None,
            Context::Reading,
            None,
        )
        .join("\n");
        assert!(
            playing.contains("[s] skip | [x] leave | [q] quit"),
            "leaving is as meaningful with a clip sounding as skipping is\n{playing}"
        );
    }

    /// TASK-050.01 acceptance criterion #2, the footer half: the assertion key is advertised,
    /// beside the two answer keys it sits between -- choose one voice, or name the whole track.
    /// The mark key sits beside it, advertising the staging the group preview previews.
    #[test]
    fn the_footer_advertises_the_assertion_key() {
        let idle = painted(130, 30, &Free, &[]).join("\n");
        assert!(
            idle.contains("[Enter] choose | [a] assert | [m] mark"),
            "{idle}"
        );
    }

    /// TASK-050.01 acceptance criterion #2, the pane half: what asserting would do is previewed
    /// in the consequence pane, reading its numbers off the same [`Assertion`] the commit
    /// reports from -- how many voices it names and how many vetoes it overrides, in the run's
    /// own labelling for both.
    ///
    /// Painted at 200 columns so the line does not wrap: the claim is about the wording, and a
    /// wrap would split the sentence the assertion is pinned by.
    #[test]
    fn the_consequence_pane_previews_what_an_assertion_would_do() {
        let painted = painted(200, 30, &Asserts, &[]);
        let whole = painted.join("\n");
        assert!(
            whole.contains("would enrol Milo from this voice"),
            "{whole}"
        );
        assert!(
            whole.contains(
                "[a] asserts one remote speaker: 4 voice(s) will read as Milo, 1 veto(s) overridden"
            ),
            "{whole}"
        );
    }

    /// The byte-identical half of the preview: with nothing to assert, the pane shows exactly
    /// what it showed before the key existed -- no assertion line, no trace of one.
    #[test]
    fn an_absent_assertion_changes_nothing_in_the_pane() {
        let painted = painted(200, 30, &Free, &[]);
        let whole = painted.join("\n");
        assert!(
            whole.contains("would enrol Milo from this voice"),
            "{whole}"
        );
        assert!(!whole.contains("asserts one remote speaker"), "{whole}");
    }

    /// The group door's counterpart to [`Asserts`]: the single-voice cost previews nothing
    /// special, but the aggregate dry run reports a hand-built displacement, which is what the
    /// pane must show while marks are active.
    struct StagesGroup;

    impl Costs for StagesGroup {
        fn of(&self, name: &str) -> Cost {
            Cost {
                refusal: None,
                summary: vec![format!("would enrol {name} from this voice")],
                assertion: None,
            }
        }

        fn group_of(&self, name: &str, _members: &[&str]) -> Option<GroupConsequence> {
            Some(GroupConsequence {
                name: name.to_string(),
                applied: vec!["Unknown 2".to_string(), "Unknown 3".to_string()],
                refused: Vec::new(),
                vetoes_overridden: 1,
                references_after: 2,
                displaced: vec![Displaced {
                    name: "Ivan".to_string(),
                    remaining: 0,
                }],
                stale: Vec::new(),
            })
        }

        fn fragment_group_of(&self, _name: &str, _members: &[&str]) -> Option<GroupConsequence> {
            None
        }
    }

    /// A row staged into the group renders the suffix beside its decision mark, not instead of
    /// it: skipped-and-staged is honest, because both facts are true.
    #[test]
    fn a_marked_row_renders_the_group_suffix() {
        let row = Row {
            number: "Unknown 2".to_string(),
            label: "Unknown 2".to_string(),
            speech_seconds: 95.0,
            similarity: None,
            below_floor: false,
            mark: Some(Mark::Skipped),
            guess: false,
            in_group: true,
            current: false,
            fragments: None,
        };
        let rendered: String = voice_line(&row)
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(rendered.contains("[skipped]"), "{rendered}");
        assert!(rendered.contains("[group]"), "{rendered}");
        assert!(
            rendered.find("[skipped]").unwrap() < rendered.find("[group]").unwrap(),
            "the group suffix follows the mark suffix: {rendered}"
        );

        // And a row nobody marked renders no trace of one.
        let quiet = Row {
            number: "Unknown 3".to_string(),
            label: "Unknown 3".to_string(),
            speech_seconds: 1.5,
            similarity: None,
            below_floor: true,
            mark: None,
            guess: false,
            in_group: false,
            current: false,
            fragments: None,
        };
        let rendered: String = voice_line(&quiet)
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(!rendered.contains("[group]"), "{rendered}");
    }

    /// AC #1, the drawing side: a guessed row carries its similarity like an identification and
    /// a `[guess]` tag beside it that neither an identified row nor a plain unknown row has --
    /// the three kinds of label read differently, and the tag sits before the question mark on
    /// the line the way `[group]` sits before the decision marks.
    #[test]
    fn a_tentative_row_carries_the_guess_tag() {
        let row = Row {
            number: "Unknown 2".to_string(),
            label: "Ivan?".to_string(),
            speech_seconds: 95.0,
            similarity: Some(0.38),
            below_floor: false,
            mark: None,
            guess: true,
            in_group: false,
            current: true,
            fragments: None,
        };
        let rendered: String = voice_line(&row)
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(rendered.contains("Ivan?"), "{rendered}");
        assert!(
            rendered.contains("0.38"),
            "the machine similarity is shown: {rendered}"
        );
        assert!(rendered.contains("[guess]"), "{rendered}");
        assert!(
            rendered.find("0.38").unwrap() < rendered.find("[guess]").unwrap(),
            "the tag follows the similarity: {rendered}"
        );
        assert!(
            rendered.find("[guess]").unwrap() < rendered.find("<- asking").unwrap(),
            "the tag precedes the question mark on the line: {rendered}"
        );

        // An identified row keeps its similarity without the tag...
        let named = Row {
            number: "Unknown 1".to_string(),
            label: "Milo".to_string(),
            speech_seconds: 240.0,
            similarity: Some(0.81),
            below_floor: false,
            mark: None,
            guess: false,
            in_group: false,
            current: false,
            fragments: None,
        };
        let rendered: String = voice_line(&named)
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(rendered.contains("0.81"), "{rendered}");
        assert!(
            !rendered.contains("[guess]"),
            "an identification is not a guess: {rendered}"
        );

        // ...and a plain unknown row carries neither.
        let unknown = Row {
            number: "Unknown 3".to_string(),
            label: "Unknown 3".to_string(),
            speech_seconds: 60.0,
            similarity: None,
            below_floor: false,
            mark: None,
            guess: false,
            in_group: false,
            current: false,
            fragments: None,
        };
        let rendered: String = voice_line(&unknown)
            .iter()
            .map(|span| span.content.as_ref())
            .collect();
        assert!(!rendered.contains("[guess]"), "{rendered}");
    }

    /// The question about a guessed fragment reads the guess once: the mark rides the label,
    /// and appending the frame's own would double it into a typo.
    #[test]
    fn the_question_does_not_double_the_guess_mark() {
        let whole = guessed(140, 30, &Free, &[]).join("\n");
        assert!(
            whole.contains("is Unknown 2 Ivan?  1m 35s"),
            "one question mark, on the label\n{whole}"
        );
        assert!(
            !whole.contains("Ivan??"),
            "the mark is not doubled\n{whole}"
        );

        // A voice nothing guessed about keeps the frame's own mark.
        let plain = painted(140, 30, &Free, &[]).join("\n");
        assert!(plain.contains("who is Unknown 2?"), "{plain}");
    }

    /// AC #2, the drawing side: while marks are active the consequence pane previews the
    /// operation the user is about to perform -- the header line off the commit's own numbers,
    /// then the would-do lines -- and nothing else: the single-voice preview gives way, because
    /// two previews in a two-inner-row pane is noise.
    ///
    /// Painted at 200 columns so the lines do not wrap: the claim is about the wording, and a
    /// wrap would split the sentence the header is pinned by.
    #[test]
    fn the_consequence_pane_previews_the_staged_group() {
        let painted = painted(200, 30, &StagesGroup, &[Event::Mark]);
        let whole = painted.join("\n");
        assert!(
            whole.contains("[m] marks 2 voice(s) as Milo: 2 reference(s), 1 veto(s) overridden"),
            "{whole}"
        );
        assert!(
            whole.contains("takes a recording off Ivan, leaving them 0"),
            "{whole}"
        );
        assert!(
            !whole.contains("would enrol Milo from this voice"),
            "the single-voice preview gives way to the group's own lines\n{whole}"
        );
        assert!(!whole.contains("asserts one remote speaker"), "{whole}");
    }

    /// The footer's precedence, both ways round: a live clip is the only one of the three that
    /// moves on its own, so it outranks a status -- and the moment it stops, the same view says
    /// what it had to say.
    #[test]
    fn a_playing_clip_outranks_the_status_line() {
        let sounding = Some(Sounding {
            progress: Progress {
                elapsed: Duration::from_secs(3),
                length: Duration::from_secs(30),
            },
            line: None,
        });
        let status = Some("could not play the clip: afplay exited with exit status: 1");

        let over = painted_with(
            110,
            30,
            &Free,
            &[],
            &said(),
            sounding,
            status,
            Context::Reading,
            None,
        )
        .join("\n");
        assert!(over.contains("playing 3s of 30s"), "{over}");
        assert!(!over.contains("could not play the clip"), "{over}");

        let stopped = painted_with(
            110,
            30,
            &Free,
            &[],
            &said(),
            None,
            status,
            Context::Reading,
            None,
        )
        .join("\n");
        assert!(stopped.contains("could not play the clip"), "{stopped}");
        assert!(!stopped.contains("playing 3s"), "{stopped}");
    }

    /// AC #1: the pane says which line is selected and when in the recording it was said, and the
    /// keys that move the selection are named where the selection is.
    #[test]
    fn the_said_pane_marks_the_selected_line_and_when_it_was_said() {
        let first = painted(110, 30, &Free, &[]).join("\n");
        assert!(first.contains("said  line 1 of 2  [ ] moves"), "{first}");
        assert!(
            first.contains(">     12s  \"so where did we land"),
            "the selected row, marked and timed\n{first}"
        );
        assert!(
            first.contains("  1m 47s  \"right, next week\""),
            "the unselected row keeps its time and loses the marker\n{first}"
        );

        let second = painted(110, 30, &Free, &[Event::SnippetDown]).join("\n");
        assert!(second.contains("said  line 2 of 2  [ ] moves"), "{second}");
        assert!(
            second.contains(">  1m 47s  \"right, next week\""),
            "the marker moved with the selection\n{second}"
        );
        assert!(
            !second.contains("so where did we land"),
            "the selection is the top of the pane\n{second}"
        );
    }

    /// AC #4: the row that is sounding says so while it sounds, and no row says so when nothing
    /// is.
    #[test]
    fn the_line_that_is_sounding_says_so() {
        let progress = Progress {
            elapsed: Duration::from_secs(1),
            length: Duration::from_secs(4),
        };
        let playing = painted_with(
            110,
            30,
            &Free,
            &[],
            &said(),
            Some(Sounding {
                progress,
                line: Some(0),
            }),
            None,
            Context::Reading,
            None,
        )
        .join("\n");
        assert!(
            playing.contains("\"so where did we land on the migration\"  <- playing"),
            "{playing}"
        );

        // The whole-voice clip: something is playing, but no transcript line is.
        let voice = painted_with(
            110,
            30,
            &Free,
            &[],
            &said(),
            Some(Sounding {
                progress,
                line: None,
            }),
            None,
            Context::Reading,
            None,
        )
        .join("\n");
        assert!(!voice.contains("<- playing"), "{voice}");

        // And nothing playing at all marks nothing.
        let quiet = painted(110, 30, &Free, &[]).join("\n");
        assert!(!quiet.contains("<- playing"), "{quiet}");
    }

    /// AC #7 and AC #5 from the footer's side: the key that plays the selected line is named, and
    /// it says what it would actually do -- nothing at all when there is no line, and "no audio"
    /// when the line has none behind it.
    #[test]
    fn the_footer_names_the_key_that_plays_the_selected_line() {
        let with_audio = painted(120, 30, &Free, &[]).join("\n");
        assert!(with_audio.contains("[P] line"), "{with_audio}");

        static NONE: [f32; 0] = [];
        let silent = [
            heard("so where did we land on the migration", 12.0, &NONE),
            heard("right, next week", 107.0, &NONE),
        ];
        let no_audio = painted_with(
            120,
            30,
            &Free,
            &[],
            &silent,
            None,
            None,
            Context::Reading,
            None,
        )
        .join("\n");
        assert!(no_audio.contains("[P] line (no audio)"), "{no_audio}");

        let nothing_said =
            painted_with(120, 30, &Free, &[], &[], None, None, Context::Reading, None).join("\n");
        assert!(
            !nothing_said.contains("[P]"),
            "a key that cannot work is not offered\n{nothing_said}"
        );
        assert!(
            nothing_said.contains("nothing was transcribed for this voice"),
            "{nothing_said}"
        );
    }

    /// AC #6, the half a test can reach: the restart key names whichever key started what is
    /// sounding, so a line mid-play is restarted by the key that played it.
    #[test]
    fn a_playing_line_restarts_with_the_key_that_started_it() {
        let sounding = Some(Sounding {
            progress: Progress {
                elapsed: Duration::from_secs(1),
                length: Duration::from_secs(4),
            },
            line: Some(1),
        });
        let whole = painted_with(
            110,
            30,
            &Free,
            &[],
            &said(),
            sounding,
            None,
            Context::Reading,
            None,
        )
        .join("\n");
        assert!(whole.contains("[P] restart"), "{whole}");
        assert!(
            !whole.contains("[p] restart"),
            "naming p would name a key that starts something else\n{whole}"
        );
    }

    /// The override key is advertised exactly where it would work, and the key it displaces is
    /// not advertised where it would not. Enter does nothing on a refused row and `o` does
    /// nothing on any other, so they are swapped rather than both shown -- the frame's rule being
    /// that a key which cannot work is not offered.
    #[test]
    fn the_override_key_is_offered_only_where_it_would_work() {
        let taken = painted(120, 30, &Takes, &[]).join("\n");
        assert!(taken.contains("[o] anyway"), "{taken}");
        assert!(
            !taken.contains("[Enter] choose"),
            "enter does nothing on a refused row\n{taken}"
        );

        for (what, costs) in [
            ("nothing refused", &Free as &dyn Costs),
            ("the heard-at-once veto", &Vetoes),
        ] {
            let whole = painted(120, 30, costs, &[]).join("\n");
            assert!(whole.contains("[Enter] choose"), "{what}\n{whole}");
            assert!(
                !whole.contains("[o] anyway"),
                "the override is not offered where it would be refused: {what}\n{whole}"
            );
        }
    }

    /// AC #4, the drawing side: the guess keys are advertised exactly where the state machine
    /// lets them work -- both while the question is about a guessed fragment whose guess is
    /// still resolvable, only `r` once a divergent filter takes the guess out of the candidates,
    /// and neither on any other voice. Asserted adjacent to their neighbours rather than as a
    /// whole line, for the width reason the footer's stance documents.
    #[test]
    fn the_footer_offers_the_guess_keys_only_where_they_work() {
        let live = guessed(140, 30, &Free, &[]).join("\n");
        assert!(
            live.contains("[Enter] choose | [y] confirm | [r] reject | [a] assert"),
            "beside the choose slot they interact with\n{live}"
        );

        // A divergent filter leaves the guess out of the resolved candidates: confirming could
        // not be reached by Enter+Tab, so the key that cannot work is not offered -- refusing
        // the guess on screen stays unambiguous, so `r` remains.
        let typing = guessed(140, 30, &Free, &[Event::Filter('z')]).join("\n");
        assert!(
            typing.contains("[Enter] choose | [r] reject | [n] new | [a] assert"),
            "{typing}"
        );
        assert!(
            !typing.contains("[y] confirm"),
            "unreachable confirms are not advertised\n{typing}"
        );

        // Neither key on a voice nothing guessed about.
        let plain = painted(140, 30, &Free, &[]).join("\n");
        assert!(!plain.contains("[y] confirm"), "{plain}");
        assert!(!plain.contains("[r] reject"), "{plain}");
    }

    /// AC #3, the log-pane half: the denial note the run narrates lands in the pane through the
    /// existing string plumbing, asserted once rather than re-invented here.
    #[test]
    fn the_log_pane_shows_the_denial_note() {
        let narration = vec![
            "20260819-100000  not Ivan: Ivan? reads Unknown 2 again -- meethook will not guess \
             Ivan for this voice again"
                .to_string(),
        ];
        let rows = frame(
            140,
            30,
            &Free,
            &[],
            &said(),
            None,
            None,
            Context::Reading,
            None,
            Attribution::Tentative {
                name: "Ivan".to_string(),
                similarity: 0.38,
            },
            narration,
        );
        let whole = rows.join("\n");
        assert!(
            whole.contains("not Ivan: Ivan? reads Unknown 2 again"),
            "{whole}"
        );
    }

    /// The minimum this frame claims to work at. At 80x24 the panes stack, and what has to be on
    /// screen is the voices, the candidates, the consequence and the snippets: "who" and the
    /// run's log are the two that yield (log first), because there are not rows for them to
    /// have a useful height. Every pane that is drawn still has a border and a title, which is
    /// what says nothing was laid out at a negative height. Painted with each cost the footer
    /// varies on, since the override swaps a hint.
    #[test]
    fn every_pane_survives_eighty_by_twenty_four() {
        // The consequence pane is the one title that follows the cost: what an answer would do,
        // or why it cannot.
        for (costs, consequence) in [(&Free as &dyn Costs, " would "), (&Takes, " cannot ")] {
            let painted = painted(80, 24, costs, &[]);
            assert_eq!(painted.len(), 24);
            let whole = painted.join("\n");
            for title in [" voices ", " resembles ", consequence, " said "] {
                assert!(
                    whole.contains(title.trim()),
                    "{title} missing from\n{whole}"
                );
            }
            for yielded in [" who Milo is ", " run "] {
                assert!(
                    !whole.contains(yielded.trim()),
                    "{yielded} should have yielded at 80x24\n{whole}"
                );
            }
        }

        // And the guess context, which has two more hints and so a taller footer.
        let guessed = guessed(80, 24, &Free, &[]);
        assert_eq!(guessed.len(), 24);
        let whole = guessed.join("\n");
        for title in [" voices ", " resembles ", " would ", " said "] {
            assert!(
                whole.contains(title.trim()),
                "{title} missing from\n{whole}"
            );
        }
    }

    /// TASK-051.02 acceptance criteria #1 and #2: while the session is being asked about, the
    /// frame names the meeting it was recorded during -- plainly when the fit states it
    /// plainly, qualified with the same caveat `meethook record` prints otherwise.
    #[test]
    fn the_frame_names_the_meeting_above_the_panes() {
        let plain = MeetingLabel {
            title: "Incident review".to_owned(),
            fit: MeetingFit::Started,
            event_id: "EVENT-1".to_owned(),
        };
        let rows = painted_with(
            110,
            30,
            &Free,
            &[],
            &said(),
            None,
            None,
            Context::Reading,
            Some(&plain),
        );
        assert_eq!(
            rows.first().map(String::as_str),
            Some("meeting   Incident review"),
            "a strong fit is stated plainly\n{rows:?}"
        );

        let late = MeetingLabel {
            title: "Incident review".to_owned(),
            fit: MeetingFit::JoinedLate,
            event_id: "EVENT-1".to_owned(),
        };
        let rows = painted_with(
            110,
            30,
            &Free,
            &[],
            &said(),
            None,
            None,
            Context::Reading,
            Some(&late),
        );
        assert_eq!(
            rows.first().map(String::as_str),
            Some(
                "meeting   Incident review  (uncertain: the recording began after this meeting had started)"
            ),
            "the caveat `meethook record` prints, not a second wording\n{rows:?}"
        );

        // Absent: the top row belongs to the panes, and nothing reserves space for a title
        // that is not there.
        let absent = painted(110, 30, &Free, &[]);
        assert!(!absent[0].contains("meeting"), "{absent:?}");
    }

    /// Acceptance criterion #5: at the 80x24 floor with a long invite title, the caveat
    /// survives and the title yields -- and every pane keeps its border, its title and an
    /// inner row, the candidate list among them gaining a row rather than losing one.
    #[test]
    fn the_banner_yields_to_the_caveat_at_the_floor() {
        let late = MeetingLabel {
            title: "Quarterly infrastructure planning and migration review with the platform group"
                .to_owned(),
            fit: MeetingFit::JoinedLate,
            event_id: "EVENT-1".to_owned(),
        };
        let rows = painted_with(
            80,
            24,
            &Free,
            &[],
            &said(),
            None,
            None,
            Context::Reading,
            Some(&late),
        );
        assert_eq!(rows.len(), 24);
        let whole = rows.join("\n");
        let banner = &rows[0];
        assert!(
            banner.contains("Quar…"),
            "the title yields with a marked cut\n{banner}"
        );
        assert!(
            banner.contains("uncertain: the recording began after this meeting had started"),
            "the caveat is the safety device and never the half clipped off\n{banner}"
        );
        for title in [" voices ", " resembles ", " would ", " said "] {
            assert!(
                whole.contains(title.trim()),
                "{title} missing from\n{whole}"
            );
        }

        // The longest caveat alone outgrows the floor's row: it keeps its beginning -- the
        // word that says the match is not strong -- and takes the cut itself.
        let unknown = MeetingLabel {
            title: "Standup".to_owned(),
            fit: MeetingFit::Unknown,
            event_id: "EVENT-1".to_owned(),
        };
        let rows = painted_with(
            80,
            24,
            &Free,
            &[],
            &said(),
            None,
            None,
            Context::Reading,
            Some(&unknown),
        );
        let banner = &rows[0];
        assert!(
            banner.contains("unverified:"),
            "the qualifier survives\n{banner}"
        );
        assert!(banner.ends_with('…'), "the cut is marked\n{banner}");
    }

    /// The banner's budget without a terminal: the whole clause when it fits, the title cut
    /// with an ellipsis and the caveat kept whole when it does not, and the caveat's own
    /// beginning when even it alone outgrows the row.
    #[test]
    fn the_clause_budget_protects_the_caveat() {
        assert_eq!(clause_within("Standup", None, 20), "Standup");
        assert_eq!(clause_within("Standup", None, 4), "Sta…");
        assert_eq!(clause_within("Standup", None, 0), "");
        assert_eq!(
            clause_within("Standup", Some("late"), 40),
            "Standup  (late)"
        );
        // The title yields ground until the whole clause fits; the caveat stays whole.
        assert_eq!(
            clause_within("A longer title", Some("late"), 14),
            "A lon…  (late)"
        );
        // Even the caveat alone outgrows the row: it keeps its beginning and takes the cut.
        assert_eq!(
            clause_within("Standup", Some("a caveat that will not fit"), 10),
            "(a caveat…"
        );
    }

    /// AC #1 and AC #2 from the drawing side: how many recordings the highlighted candidate has,
    /// and which voices in which meetings read that name today -- beside the candidate row, so
    /// "who is Ivan again?" is answerable without leaving the prompt.
    #[test]
    fn the_frame_says_who_the_highlighted_candidate_is() {
        let scan = scanned(
            vec![holding(
                "Milo",
                &[
                    &[names("20260810-101500", "Unknown 1", "Milo")],
                    &[names("20260809-052600", "Unknown 3", "Milo")],
                    &[],
                ],
            )],
            &[],
        );
        let whole = knowing(110, 30, &[], Context::Read(&scan)).join("\n");
        assert!(whole.contains("who Milo is"), "{whole}");
        assert!(
            whole.contains("3 reference(s), naming 2 voice(s) in 2 session(s)"),
            "{whole}"
        );
        assert!(whole.contains("20260810-101500  Unknown 1"), "{whole}");
        assert!(whole.contains("20260809-052600  Unknown 3"), "{whole}");

        // Somebody the ranking has no count for at all is still described here, which is the whole
        // reason this pane is not just the "N ref" column read out.
        let al = knowing(
            110,
            30,
            &[Event::CandidateDown],
            Context::Read(&scanned(
                vec![
                    holding("Milo", &[&[]]),
                    holding("Ivan", &[&[names("20260810-101500", "Unknown 7", "Ivan")]]),
                ],
                &[],
            )),
        )
        .join("\n");
        assert!(al.contains("who Ivan is"), "{al}");
        assert!(
            al.contains("1 reference(s), naming 1 voice(s) in 1 session(s)"),
            "{al}"
        );

        // More sessions than the pane has rows: it says how many it left out rather than looking
        // like the whole answer.
        let crowded = scanned(
            vec![holding(
                "Milo",
                &[
                    &[names("20260810-101500", "Unknown 1", "Milo")],
                    &[names("20260809-052600", "Unknown 3", "Milo")],
                    &[names("20260808-140000", "Unknown 2", "Milo")],
                    &[names("20260807-090000", "Unknown 5", "Milo")],
                ],
            )],
            &[],
        );
        let clipped = knowing(110, 30, &[], Context::Read(&crowded)).join("\n");
        assert!(
            clipped.contains("... and 3 more session(s)"),
            "three rows hold the summary, one session and the count\n{clipped}"
        );
    }

    /// AC #5 from the drawing side: a scan that could not read every session says so on the frame,
    /// in the same sentence `meethook speakers` fails with, and still reports what it did read.
    #[test]
    fn an_incomplete_scan_says_so_on_the_frame() {
        let scan = scanned(
            vec![holding(
                "Milo",
                &[&[names("20260810-101500", "Unknown 1", "Milo")]],
            )],
            &["20260809-052600"],
        );
        // Wide enough for the shared sentence to land whole: the pane clips rather than wraps, and
        // this test is about the wording rather than about the width.
        let whole = knowing(160, 30, &[], Context::Read(&scan)).join("\n");
        assert!(
            whole.contains(&incomplete(1)),
            "the sentence `meethook speakers` uses, not a second one\n{whole}"
        );
        assert!(
            whole.contains("20260810-101500  Unknown 1"),
            "what it could read is still reported\n{whole}"
        );
    }

    /// The two states that are not an answer yet. A scan still running says so, and one that failed
    /// says why -- either way the pane is never an empty box that reads as "this name does nothing".
    #[test]
    fn a_scan_still_running_says_so_rather_than_looking_empty() {
        let reading = knowing(110, 30, &[], Context::Reading).join("\n");
        assert!(reading.contains("reading the sessions..."), "{reading}");

        let failed = knowing(
            160,
            30,
            &[],
            Context::Failed("no such file or directory (os error 2)"),
        )
        .join("\n");
        assert!(
            failed.contains("could not read the enrolled speakers: no such file"),
            "{failed}"
        );
    }

    /// The pane's budget, at every size it can be asked for: the last session is never silently
    /// dropped, and a lone row goes to a session rather than to a count of the ones missing.
    #[test]
    fn the_session_list_never_drops_a_session_silently() {
        assert_eq!(listed(0, 3), (0, false));
        assert_eq!(listed(3, 3), (3, false), "an exact fit needs no count line");
        assert_eq!(listed(4, 3), (2, true), "the count line costs a session");
        assert_eq!(listed(4, 1), (1, false), "one row says more than a count");
        assert_eq!(listed(4, 0), (0, false), "no room at all is not a panic");
    }

    /// The first row of the buffer that holds `needle`, so a test can find a pane by its title.
    fn row_of(buffer: &ratatui::buffer::Buffer, needle: &str) -> u16 {
        (0..buffer.area.height)
            .find(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, *y)].symbol())
                    .collect::<String>()
                    .contains(needle)
            })
            .unwrap_or_else(|| panic!("{needle} is not on the frame"))
    }

    fn buffer_at(
        width: u16,
        height: u16,
        events: &[Event],
        help_open: bool,
    ) -> ratatui::buffer::Buffer {
        frame_buffer(
            width,
            height,
            &Free,
            events,
            &said(),
            None,
            None,
            Context::Reading,
            None,
            Attribution::Unknown("Unknown 2".to_string()),
            vec!["20260819-100000  3 voice(s) to ask about".to_string()],
            help_open,
        )
    }

    /// Below the width threshold the panes stack in one column; at and above it they sit side by
    /// side. Each draw chooses from the area it was given, so one view drawn at two sizes gets
    /// both layouts with nothing carried between them.
    #[test]
    fn the_layout_stacks_below_the_threshold_and_re_chooses_on_every_draw() {
        let wide = layout(Rect::new(0, 0, 120, 37), false);
        assert!(wide.candidates.x > wide.voices.x, "side by side: {wide:?}");
        assert_eq!(wide.voices.y, wide.question.y);

        for (width, height) in [(80, 21), (60, 21), (99, 30)] {
            let body = Rect::new(0, 0, width, height);
            let stacked = layout(body, false);
            assert_eq!(
                stacked.voices.x, stacked.candidates.x,
                "{width}: {stacked:?}"
            );
            assert_eq!(stacked.voices.width, width);
            assert!(stacked.candidates.y > stacked.voices.y);
            assert!(stacked.said.y > stacked.consequence.y);
            for rect in [
                stacked.voices,
                stacked.question,
                stacked.candidates,
                stacked.consequence,
                stacked.said,
            ] {
                assert!(
                    rect.height > 0 && rect.bottom() <= body.bottom(),
                    "{rect:?}"
                );
            }
        }
        assert!(layout(Rect::new(0, 0, 99, 40), false).voices.width == 99);
        assert!(layout(Rect::new(0, 0, 100, 40), false).voices.width < 100);

        // The same view at two sizes: stacked, then side by side, then stacked again.
        let at = |width| {
            let buffer = buffer_at(width, 30, &[], false);
            let voices = row_of(&buffer, "voices");
            let resembles = row_of(&buffer, "resembles");
            (voices, resembles)
        };
        let (voices, resembles) = at(80);
        assert!(resembles > voices, "stacked: candidates under voices");
        let (voices, resembles) = at(140);
        assert_eq!(
            resembles,
            voices + 1,
            "side by side: the question sits above the candidates"
        );
        let (voices, resembles) = at(60);
        assert!(resembles > voices);
    }

    /// When rows run out the log goes first and then "who", never the panes a question is
    /// answered with; and the banner row, when there is one, takes its row from the panes.
    #[test]
    fn a_short_stack_drops_the_log_then_who() {
        let at = |height, meeting| layout(Rect::new(0, 0, 80, height), meeting);
        let tall = at(40, false);
        assert!(tall.who.is_some() && tall.log.is_some());
        let medium = at(24, false);
        assert!(medium.who.is_some() && medium.log.is_none(), "{medium:?}");
        let short = at(21, false);
        assert!(short.who.is_none() && short.log.is_none(), "{short:?}");
        assert!(at(21, true).banner.is_some());
        assert!(at(21, false).banner.is_none());
        // However short, the answering panes keep a useful height.
        let tiny = at(17, false);
        assert!(tiny.candidates.height >= 4 && tiny.voices.height >= 4);
    }

    /// 80x24 and a narrower 60x24 are stacked and complete; 120x40 is still side by side with all
    /// six panes.
    #[test]
    fn the_frame_stacks_at_eighty_and_sixty_columns_and_not_at_one_twenty() {
        for width in [80, 60] {
            let rows = painted(width, 24, &Free, &[]);
            assert_eq!(rows.len(), 24);
            let whole = rows.join("\n");
            for title in ["voices", "resembles", "would", "said"] {
                assert!(whole.contains(title), "{title} at {width}\n{whole}");
            }
            let voices = rows.iter().position(|r| r.contains("voices")).unwrap();
            let resembles = rows.iter().position(|r| r.contains("resembles")).unwrap();
            assert!(resembles > voices, "stacked at {width}\n{whole}");
            assert!(
                rows.iter().all(|r| r.chars().count() <= width as usize),
                "nothing outruns the width"
            );
        }
        let rows = painted(120, 40, &Free, &[]);
        let whole = rows.join("\n");
        for title in ["voices", "resembles", "would", "who Milo is", "said", "run"] {
            assert!(whole.contains(title), "{title}\n{whole}");
        }
        let voices = rows.iter().find(|r| r.contains("voices")).unwrap();
        assert!(
            voices.contains("resembles") || voices.contains("who is"),
            "{voices}"
        );
    }

    /// The active pane is the one with the marked border: bold where there is no colour, so the
    /// mark does not depend on the terminal's palette. Focus moves it.
    #[test]
    fn the_focused_pane_has_the_marked_border() {
        let bold = |buffer: &ratatui::buffer::Buffer, title: &str| {
            let y = row_of(buffer, title);
            buffer[(0, y)]
                .modifier
                .contains(ratatui::style::Modifier::BOLD)
        };
        let candidates = buffer_at(80, 30, &[], false);
        assert!(
            bold(&candidates, "resembles"),
            "the candidates start focused"
        );
        assert!(!bold(&candidates, "voices"));
        let voices = buffer_at(80, 30, &[Event::FocusVoices], false);
        assert!(bold(&voices, "voices"));
        assert!(!bold(&voices, "resembles"));
    }

    /// Typing is visible: the candidates' title carries the text and a cursor while the mode is
    /// on, and loses the cursor, keeping the text, when it is left.
    #[test]
    fn filter_mode_shows_a_cursor_in_the_candidates_title() {
        let typing = painted(100, 30, &Free, &[Event::EnterFilter, Event::Filter('M')]).join("\n");
        assert!(typing.contains("filter: M_"), "{typing}");
        let left = painted(
            100,
            30,
            &Free,
            &[Event::EnterFilter, Event::Filter('M'), Event::LeaveFilter],
        )
        .join("\n");
        assert!(left.contains("filter: M "), "{left}");
        assert!(!left.contains("filter: M_"), "{left}");
    }

    /// The footer wraps rather than clips: at 60 columns every hint that is enabled is on the
    /// screen whole, and the panes gave up the rows it took.
    #[test]
    fn a_narrow_footer_wraps_and_loses_no_hint() {
        let rows = guessed(60, 24, &Free, &[]);
        assert_eq!(rows.len(), 24);
        let whole = rows.join("\n");
        for hint in [
            "[j] down",
            "[k] up",
            "[Tab] pane",
            "[Enter] choose",
            "[y] confirm",
            "[r] reject",
            "[a] assert",
            "[m] mark",
            "[w] work on",
            "[s] skip",
            "[x] leave",
            "[p] play",
            "[P] line",
            "[/] filter",
            "[?] help",
            "[q] quit",
        ] {
            assert!(whole.contains(hint), "{hint} clipped from\n{whole}");
        }
        let footer_rows = rows.iter().rev().take_while(|r| r.contains('[')).count();
        assert!(
            footer_rows >= 3,
            "60 columns need several footer rows: {footer_rows}"
        );
        let wide = painted(200, 24, &Free, &[]);
        assert_eq!(
            wide.iter().rev().take_while(|r| r.contains('[')).count(),
            1,
            "one row where it fits"
        );
    }

    /// `?` draws the key table over the frame, dimming nothing it can do and listing aliases.
    #[test]
    fn the_help_overlay_lists_the_keys_over_the_frame() {
        let buffer = buffer_at(100, 40, &[], true);
        let rows: Vec<String> = (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol())
                    .collect()
            })
            .collect();
        let whole = rows.join("\n");
        assert!(whole.contains(" keys "), "{whole}");
        assert!(whole.contains("Quit"), "{whole}");
        assert!(whole.contains("Type into the candidate filter"), "{whole}");
        assert!(whole.contains("Move down in the focused pane"), "{whole}");
        assert!(
            !buffer_at(100, 40, &[], false)
                .content()
                .iter()
                .map(|c| c.symbol())
                .collect::<String>()
                .contains(" keys "),
            "closed it draws nothing"
        );
    }
}
