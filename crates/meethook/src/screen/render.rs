//! The five panes, given a [`View`]. Every widget in this binary is built here and nowhere else.
//!
//! Nothing in this module decides anything: it takes the derived view the state machine produced
//! and places it. That is what lets it be exercised through [`ratatui::backend::TestBackend`]
//! without a terminal, and what keeps the part that *is* decidable in `state`.
//!
//! # The one place this crate invents wording
//!
//! `narration.rs` in `meethook-enroll` tells an interface not to invent sentences, and it is
//! right: moving a sentence into the thing that displays it moves it out of `cargo test`. The
//! refusal line below is the exception, and it is worth naming why. `Lines` renders what a run
//! *did* -- past tense, prefixed with a session id -- and there is no `Note` for a dry run.
//! Inventing one would put "what would happen" into a type documented as "one thing a run has to
//! say". So the consequence lines come off [`Consequence`](meethook_enroll::Consequence) in
//! `super` (the only place able to read it) and the refusal sentence comes off the fully public
//! [`Refusal`] here.

use meethook_enroll::{Refusal, Resolution, speech};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};

use super::Sounding;
use super::state::{Candidate, Mark, Row, View};
use crate::commands::Progress;

/// Places every pane for one frame.
///
/// `sounding` is a parameter rather than a [`View`] field on purpose: it is derived from a clock
/// and from what the shell handed to the player, and [`state`](super::state) is documented as
/// having neither in it. A fourth argument says in the signature that the shell computed this and
/// the state machine did not.
pub fn draw(frame: &mut Frame, view: &View<'_>, narration: &[String], sounding: Option<Sounding>) {
    let whole = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),
            Constraint::Length(5),
            Constraint::Length(5),
            Constraint::Length(1),
        ])
        .split(frame.area());
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(whole[0]);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(4),
        ])
        .split(top[1]);

    voices(frame, top[0], view);
    question(frame, right[0], view);
    candidates(frame, right[1], view);
    consequence(frame, right[2], view);
    snippets(frame, whole[1], view, sounding.and_then(|s| s.line));
    log(frame, whole[2], narration);
    footer(frame, whole[3], view, sounding);
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
    if let Some(mark) = row.mark {
        spans.push(Span::raw(match mark {
            Mark::Answered => "  [named]",
            Mark::Skipped => "  [skipped]",
            Mark::Deferred => "  [later]",
        }));
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
        false => format!(" is {} {}?  {heard}", view.number, view.label),
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

/// Why a candidate cannot be chosen. Off the public [`Refusal`], for the reason the module doc
/// gives.
fn refused(refusal: &Refusal) -> String {
    match refusal {
        Refusal::Vetoed {
            holder: Some(voice),
        } => format!(
            "unavailable: {voice} was heard at the same time as this voice and would keep the name"
        ),
        Refusal::Vetoed { holder: None } => {
            "unavailable: the name would not end up on this voice".to_string()
        }
        Refusal::Taken { voice, losing } => {
            format!("unavailable: {voice} would stop reading {losing}")
        }
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
    let highlighted = view
        .candidate
        .and_then(|index| view.candidates.get(index))
        .and_then(|candidate| candidate.refusal.as_ref());
    let (title, lines): (&str, Vec<Line>) = match (highlighted, view.consequence.is_empty()) {
        (Some(refusal), _) => (" cannot ", vec![Line::from(refused(refusal))]),
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
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(Block::default().borders(Borders::ALL).title(title)),
        area,
    );
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
/// second time formatter appears in this binary. Only the three keys that mean something mid-clip
/// are kept: the full list -- now with both play keys on it -- passes 100 columns, and it was
/// already wider than 80 before the second one was added.
///
/// The restart key follows what is sounding rather than naming one of the two: saying "^P restart"
/// while a line is playing would name a key that starts something else.
fn footer(frame: &mut Frame, area: Rect, view: &View<'_>, sounding: Option<Sounding>) {
    let text = match (sounding, view.status) {
        (
            Some(Sounding {
                progress: Progress { elapsed, length },
                line,
            }),
            _,
        ) => format!(
            "playing {} of {}  {} restart  ^S skip  ^C quit",
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
            format!(
                "up/down voice  right work on it  tab candidate  enter choose  \
                 ^N new  {clip}{line}  ^S skip  ^C quit"
            )
        }
    };
    frame.render_widget(Paragraph::new(Line::from(text).dim()), area);
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use meethook_enroll::{Position, Queued, Refusal};
    use meethook_session::SessionId;
    use meethook_transcribe::{Attribution, Resemblance};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    use meethook_enroll::Snippet;

    use super::super::state::tests::heard;
    use super::super::state::{Cost, Costs, Event, Screen, VoiceView};
    use super::{Progress, Sounding, draw};

    struct Free;

    impl Costs for Free {
        fn of(&self, name: &str) -> Cost {
            Cost {
                refusal: None,
                summary: vec![format!("would enrol {name} from this voice")],
            }
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
            }
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
        painted_with(width, height, costs, keys, &said(), None, None)
    }

    /// [`painted`], plus the three things only the snippet and footer tests vary: what this voice
    /// said, what is sounding, and what the frame last had to say. The last two reach the footer
    /// from different directions -- one is a parameter to `draw`, the other a field of the view --
    /// which is exactly what the precedence between them has to be pinned against.
    fn painted_with(
        width: u16,
        height: u16,
        costs: &dyn Costs,
        keys: &[Event],
        snippets: &[Snippet<'_>],
        sounding: Option<Sounding>,
        status: Option<&str>,
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
        let view = screen.view(&voice, costs);
        let narration = vec!["20260819-100000  3 voice(s) to ask about".to_string()];

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

    /// The five panes, at a comfortable size: the queue with its talk times and its separator for
    /// the quiet voices, the ranked candidates with their numbers, the consequence, the snippets
    /// and the run's own narration.
    #[test]
    fn the_frame_places_all_five_panes() {
        let painted = painted(110, 30, &Free, &[]);
        let whole = painted.join("\n");
        assert!(whole.contains("voices  20260819-100000  2/3"), "{whole}");
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
        let whole = painted_with(110, 30, &Free, &[], &said(), sounding, None).join("\n");
        assert!(whole.contains("playing 12s of 1m 47s"), "{whole}");
        assert!(whole.contains("^P restart"), "{whole}");
        assert!(
            !whole.contains("right work on it"),
            "the key list gives way to the position\n{whole}"
        );
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

        let over = painted_with(110, 30, &Free, &[], &said(), sounding, status).join("\n");
        assert!(over.contains("playing 3s of 30s"), "{over}");
        assert!(!over.contains("could not play the clip"), "{over}");

        let stopped = painted_with(110, 30, &Free, &[], &said(), None, status).join("\n");
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
        let no_audio = painted_with(120, 30, &Free, &[], &silent, None, None).join("\n");
        assert!(no_audio.contains("^L no audio"), "{no_audio}");

        let nothing_said = painted_with(120, 30, &Free, &[], &[], None, None).join("\n");
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
        let whole = painted_with(110, 30, &Free, &[], &said(), sounding, None).join("\n");
        assert!(whole.contains("^L restart"), "{whole}");
        assert!(
            !whole.contains("^P restart"),
            "naming ^P would name a key that starts something else\n{whole}"
        );
    }

    /// The minimum this frame claims to work at. Every pane still has a border and a title, which
    /// is what says nothing was laid out at a negative height.
    #[test]
    fn every_pane_survives_eighty_by_twenty_four() {
        let painted = painted(80, 24, &Free, &[]);
        assert_eq!(painted.len(), 24);
        let whole = painted.join("\n");
        for title in [" voices ", " resembles ", " would ", " said ", " run "] {
            assert!(
                whole.contains(title.trim()),
                "{title} missing from\n{whole}"
            );
        }
    }
}
