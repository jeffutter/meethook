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
//! [`Refusal::sentence`] here.
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

use meethook_enroll::{Assertion, GroupConsequence, Refusal, Resolution, incomplete, speech};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};

use super::Sounding;
use super::state::{Candidate, Mark, Row, View, Who};
use crate::clips::Progress;

/// Places every pane for one frame.
///
/// `sounding` is a parameter rather than a [`View`] field on purpose: it is derived from a clock
/// and from what the shell handed to the player, and [`state`](super::state) is documented as
/// having neither in it. A fourth argument says in the signature that the shell computed this and
/// the state machine did not.
pub fn draw(frame: &mut Frame, view: &View<'_>, narration: &[String], sounding: Option<Sounding>) {
    // The banner row exists only when this session carries a meeting: reserving it either way
    // would cost a session without one a row it has nothing to say, and an absent title must
    // reserve no space at all. Present, the snippets and log bands each give up one row and
    // the flexible top band absorbs the remainder -- at the 80x24 floor the candidate list
    // gains an inner row (1 -> 2) instead of losing its last one.
    let (banner, top, said, run, keys) = match view.meeting {
        Some(_) => {
            let whole = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1),
                    Constraint::Min(6),
                    Constraint::Length(4),
                    Constraint::Length(4),
                    Constraint::Length(1),
                ])
                .split(frame.area());
            (Some(whole[0]), whole[1], whole[2], whole[3], whole[4])
        }
        None => {
            let whole = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(6),
                    Constraint::Length(5),
                    Constraint::Length(5),
                    Constraint::Length(1),
                ])
                .split(frame.area());
            (None, whole[0], whole[1], whole[2], whole[3])
        }
    };
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(top);
    // Four bands in the right column, and the sizes are pinned by the 80x24 floor the tests hold
    // this frame to: there the top band is 13 rows and `1 + 3 + 4 + 5` fits it exactly, with the
    // candidate list down to one visible row; at 120x40 it is 29 and the candidates get 19. With
    // the meeting banner present the top band is one row taller -- 14 at the floor -- so the
    // candidates gain their second inner row rather than the frame squeezing them. The "who"
    // pane is adjacent to the candidate it describes, which is the whole point of it -- a
    // horizontal split of the `run` band was considered and rejected, because `log` clips rather
    // than wraps and halving its width would cut narration sentences mid-word.
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(4),
            Constraint::Length(5),
        ])
        .split(top[1]);

    if let Some(banner) = banner {
        meeting_banner(frame, banner, view);
    }
    voices(frame, top[0], view);
    question(frame, right[0], view);
    candidates(frame, right[1], view);
    consequence(frame, right[2], view);
    who(frame, right[3], view);
    snippets(frame, said, view, sounding.and_then(|s| s.line));
    log(frame, run, narration);
    footer(frame, keys, view, sounding);
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
fn voices(frame: &mut Frame, area: Rect, view: &View<'_>) {
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
        .block(Block::default().borders(Borders::ALL).title(title))
        .highlight_symbol("> ")
        .highlight_style(Style::default().add_modifier(Modifier::REVERSED));
    let mut state = ListState::default().with_selected(Some(selected));
    frame.render_stateful_widget(list, area, &mut state);
}

fn voice_line(row: &Row) -> Line<'static> {
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
fn candidates(frame: &mut Frame, area: Rect, view: &View<'_>) {
    let mut items: Vec<ListItem> = view
        .candidates
        .iter()
        .map(|candidate| ListItem::new(candidate_line(candidate)))
        .collect();
    // Its own entry, always there, and actuated by its own key -- never the thing that happens
    // when text matches nothing.
    items.push(ListItem::new(Line::from(new_person(view))));

    let title = match view.filter.is_empty() {
        true => " resembles ".to_string(),
        false => format!(" filter: {} ", view.filter),
    };
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(title))
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
    // carry the refusal of every member it refused.
    let (title, lines) = if let Some(group) = view.group {
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
        "^A asserts one remote speaker: {voices} voice(s) will read as {name}, {vetoes} veto(s) overridden",
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
        "^K marks {} voice(s) as {}: {} reference(s), {} veto(s) overridden",
        group.applied.len(),
        group.name,
        group.references_after,
        group.vetoes_overridden
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
        total => format!(
            " said  line {} of {total}  pgup/pgdn moves ",
            view.snippet + 1
        ),
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

/// One line: how far through a clip playback has got, else whatever just happened, else the keys.
///
/// Playback outranks the status because it is the only one of the three that changes on its own,
/// and because the two cannot both be true: a play that failed clears the progress and sets the
/// status in the same iteration.
///
/// The position is spelled with [`speech`], which is already what the question and the queue use
/// for a duration, so "playing 12s of 1m 47s" reads the way the rest of the frame does and no
/// second time formatter appears in this binary. Only the four keys that mean something mid-clip
/// are kept: the full list -- now with both play keys on it -- passes 100 columns, and it was
/// already wider than 80 before the second one was added. Leaving the session is on that short
/// list because it is as meaningful with a clip sounding as skipping one voice is.
///
/// The restart key follows what is sounding rather than naming one of the two: saying "^P restart"
/// while a line is playing would name a key that starts something else.
///
/// The choose key follows the highlighted candidate for the same reason: where that candidate is
/// refused for taking a name off another voice, the key that works is Ctrl-O and Enter is the one
/// that would do nothing.
fn footer(frame: &mut Frame, area: Rect, view: &View<'_>, sounding: Option<Sounding>) {
    let text = match (sounding, view.status) {
        (
            Some(Sounding {
                progress: Progress { elapsed, length },
                line,
            }),
            _,
        ) => format!(
            "playing {} of {}  {} restart  ^S skip  ^G leave  ^C quit",
            speech(elapsed.as_secs_f64()),
            speech(length.as_secs_f64()),
            match line {
                Some(_) => "^L",
                None => "^P",
            }
        ),
        (None, Some(status)) => status.to_string(),
        (None, None) => {
            let clip = match view.clip_is_empty {
                true => "^P no audio",
                false => "^P play",
            };
            // `view.snippet` is already clamped by `Screen::view`, so this is a lookup and not a
            // bounds check. Both play keys say what they would do *now*, which is what stops ^L
            // from looking like a key that did nothing.
            let line = match view.snippets.get(view.snippet) {
                Some(snippet) if !snippet.audio.is_empty() => "  ^L line",
                Some(_) => "  ^L no audio",
                // Nothing transcribed. The pane says so, and a key for it would be a key that
                // cannot work.
                None => "",
            };
            // Enter genuinely does nothing on a refused row, and Ctrl-O does nothing on any
            // other, so the two are swapped rather than both offered: the frame's rule is that a
            // key which cannot work is not advertised, and swapping also leaves the line the
            // width it already was.
            let choose = match view.highlighted().and_then(|c| c.refusal.as_ref()) {
                Some(Refusal::Taken { .. }) => "^O anyway",
                _ => "enter choose",
            };
            // The guess keys actuate the choosing act, so they sit beside the slot they
            // interact with -- early enough to survive the clip at the pane edge, where a
            // tail append would hide them entirely. Both advertised only while the question
            // is about a guessed fragment, mirroring the gate the events carry in `state`;
            // ^Y additionally only while the guess is still among the resolved candidates,
            // because a confirm that could not be reached by Enter+Tab would be a key that
            // cannot work.
            let guess_keys = match view.guess {
                Some(guess) => {
                    let confirm = view
                        .candidates
                        .iter()
                        .any(|candidate| candidate.name == guess)
                        .then_some("^Y confirm");
                    [confirm, Some("^R reject")]
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>()
                        .join("  ")
                }
                None => String::new(),
            };
            let guess_segment = if guess_keys.is_empty() {
                String::new()
            } else {
                format!("  {guess_keys}")
            };
            format!(
                "up/down voice  right work on it  tab candidate  {choose}{guess_segment}  \
                 ^A assert  ^K mark  ^N new  {clip}{line}  ^S skip  ^G leave  ^C quit"
            )
        }
    };
    frame.render_widget(Paragraph::new(Line::from(text).dim()), area);
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use meethook_enroll::{Assertion, GroupConsequence, MeetingLabel, Position, Queued, Refusal};
    use meethook_session::{MeetingFit, SessionId};
    use meethook_transcribe::{Attribution, Resemblance};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use meethook_enroll::Snippet;

    use super::super::state::tests::heard;
    use super::super::state::{Context, Cost, Costs, Event, Mark, Row, Screen, VoiceView};
    use super::super::who::tests::{holding, names, scanned};
    use super::{Progress, Sounding, clause_within, draw, incomplete, listed, voice_line};

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
            clip_is_empty: false,
        };

        let mut screen = Screen::default();
        screen.arrive(&voice);
        for key in keys {
            screen.answer(&voice, *key, costs);
        }
        // After the keys: `answer` clears the status, which is the behaviour a caller asking for
        // one is working around.
        if let Some(status) = status {
            screen.say(status.to_string());
        }
        let view = screen.view(&voice, costs, context);

        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("a test backend");
        terminal
            .draw(|frame| draw(frame, &view, &narration, sounding))
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
        assert!(whole.contains("^S skip"), "{whole}");
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
        assert!(whole.contains("^P restart"), "{whole}");
        assert!(
            !whole.contains("right work on it"),
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
        assert!(idle.contains("^S skip  ^G leave  ^C quit"), "{idle}");

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
            playing.contains("^S skip  ^G leave  ^C quit"),
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
            idle.contains("enter choose  ^A assert  ^K mark  ^N new"),
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
                "^A asserts one remote speaker: 4 voice(s) will read as Milo, 1 veto(s) overridden"
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
            whole.contains("^K marks 2 voice(s) as Milo: 2 reference(s), 1 veto(s) overridden"),
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
        assert!(
            first.contains("said  line 1 of 2  pgup/pgdn moves"),
            "{first}"
        );
        assert!(
            first.contains(">     12s  \"so where did we land"),
            "the selected row, marked and timed\n{first}"
        );
        assert!(
            first.contains("  1m 47s  \"right, next week\""),
            "the unselected row keeps its time and loses the marker\n{first}"
        );

        let second = painted(110, 30, &Free, &[Event::SnippetDown]).join("\n");
        assert!(
            second.contains("said  line 2 of 2  pgup/pgdn moves"),
            "{second}"
        );
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
        assert!(with_audio.contains("^L line"), "{with_audio}");

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
        assert!(no_audio.contains("^L no audio"), "{no_audio}");

        let nothing_said =
            painted_with(120, 30, &Free, &[], &[], None, None, Context::Reading, None).join("\n");
        assert!(
            !nothing_said.contains("^L"),
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
        assert!(whole.contains("^L restart"), "{whole}");
        assert!(
            !whole.contains("^P restart"),
            "naming ^P would name a key that starts something else\n{whole}"
        );
    }

    /// The override key is advertised exactly where it would work, and the key it displaces is
    /// not advertised where it would not. Enter does nothing on a refused row and Ctrl-O does
    /// nothing on any other, so they are swapped rather than both shown -- the frame's rule being
    /// that a key which cannot work is not offered.
    #[test]
    fn the_override_key_is_offered_only_where_it_would_work() {
        let taken = painted(120, 30, &Takes, &[]).join("\n");
        assert!(taken.contains("^O anyway"), "{taken}");
        assert!(
            !taken.contains("enter choose"),
            "enter does nothing on a refused row\n{taken}"
        );

        for (what, costs) in [
            ("nothing refused", &Free as &dyn Costs),
            ("the heard-at-once veto", &Vetoes),
        ] {
            let whole = painted(120, 30, costs, &[]).join("\n");
            assert!(whole.contains("enter choose"), "{what}\n{whole}");
            assert!(
                !whole.contains("^O anyway"),
                "the override is not offered where it would be refused: {what}\n{whole}"
            );
        }
    }

    /// AC #4, the drawing side: the guess keys are advertised exactly where the state machine
    /// lets them work -- both while the question is about a guessed fragment whose guess is
    /// still resolvable, only ^R once a divergent filter takes the guess out of the candidates,
    /// and neither on any other voice. Asserted adjacent to their neighbours rather than as a
    /// whole line, for the width reason the footer's stance documents.
    #[test]
    fn the_footer_offers_the_guess_keys_only_where_they_work() {
        let live = guessed(140, 30, &Free, &[]).join("\n");
        assert!(
            live.contains("enter choose  ^Y confirm  ^R reject  ^A assert"),
            "beside the choose slot they interact with\n{live}"
        );

        // A divergent filter leaves the guess out of the resolved candidates: confirming could
        // not be reached by Enter+Tab, so the key that cannot work is not offered -- refusing
        // the guess on screen stays unambiguous, so ^R remains.
        let typing = guessed(140, 30, &Free, &[Event::Filter('z')]).join("\n");
        assert!(
            typing.contains("enter choose  ^R reject  ^A assert"),
            "{typing}"
        );
        assert!(
            !typing.contains("^Y confirm"),
            "unreachable confirms are not advertised\n{typing}"
        );

        // Neither key on a voice nothing guessed about.
        let plain = painted(140, 30, &Free, &[]).join("\n");
        assert!(!plain.contains("^Y confirm"), "{plain}");
        assert!(!plain.contains("^R reject"), "{plain}");
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

    /// The minimum this frame claims to work at. Every pane still has a border and a title, which
    /// is what says nothing was laid out at a negative height. Painted with each cost the footer
    /// now varies on, since the override swaps the widest key on the line.
    #[test]
    fn every_pane_survives_eighty_by_twenty_four() {
        // The consequence pane is the one title that follows the cost: what an answer would do,
        // or why it cannot.
        for (costs, consequence) in [(&Free as &dyn Costs, " would "), (&Takes, " cannot ")] {
            let painted = painted(80, 24, costs, &[]);
            assert_eq!(painted.len(), 24);
            let whole = painted.join("\n");
            for title in [
                " voices ",
                " resembles ",
                consequence,
                // The one title that names its subject, so "who" alone would also match the
                // question line above it.
                " who Milo is ",
                " said ",
                " run ",
            ] {
                assert!(
                    whole.contains(title.trim()),
                    "{title} missing from\n{whole}"
                );
            }
        }

        // And the guess context: the row carries its tag and the footer its two keys, which is
        // the widest the idle line gets.
        let guessed = guessed(80, 24, &Free, &[]);
        assert_eq!(guessed.len(), 24);
        let whole = guessed.join("\n");
        for title in [
            " voices ",
            " resembles ",
            " would ",
            " who ",
            " said ",
            " run ",
        ] {
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
        for title in [
            " voices ",
            " resembles ",
            " would ",
            " who ",
            " said ",
            " run ",
        ] {
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
}
