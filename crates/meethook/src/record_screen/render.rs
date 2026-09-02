//! The record frame, given a [`State`]. Every widget in this interface is built here and nowhere
//! else.
//!
//! Nothing in this module decides anything: it takes the state the notes produced and places
//! it. That is what lets it be exercised through [`ratatui::backend::TestBackend`] without a
//! terminal, and what keeps the part that *is* decidable in `state`. Wording follows the same
//! rule as there: every sentence comes off the composers in `crate::record` -- the meeting line
//! and each selector row included -- verbatim. The only lines invented here are chrome about
//! the frame itself rather than facts about the run: the phase word, the key hint, the guess
//! marker on the offer list and the "by hand" marker on a settled pick.

use std::time::Duration;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::Stylize;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};

use super::state::{Phase, State};
use crate::record::{meeting_clause_line, mic_line, session_id_line, speaker_line};

/// Places the record frame for one draw.
///
/// `elapsed` is a parameter rather than a [`State`] field, for the same reason enroll's draw
/// takes its playback progress: it is derived from a clock, and the state machine is documented
/// as having none in it. `None` outside recording.
pub fn draw(frame: &mut Frame, state: &State, elapsed: Option<Duration>) {
    let whole = frame.area();
    // Too small to mean anything; drawing into it would only produce a wall of borders.
    if whole.height < 5 || whole.width < 20 {
        return;
    }

    // The trouble pane reserves room only when there is trouble: a clean run never pays for a
    // pane it will not fill, capped so a run with repeated failures still leaves room for the
    // rest of the frame.
    let trouble_height = if state.trouble.is_empty() {
        0
    } else {
        state
            .trouble
            .len()
            .saturating_add(2)
            .min(6)
            .min(whole.height.saturating_sub(4) as usize) as u16
    };

    let [header, body, footer] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(1),
        ])
        .areas(whole);

    frame.render_widget(
        Line::from(vec![
            Span::raw("meethook record").bold(),
            Span::raw("  ").dim(),
            Span::raw(format!("· {}", state.phase.word())).dim(),
        ]),
        header,
    );

    let mut body_lines: Vec<Line> = vec![status_line(state, elapsed)];

    if let Some(session) = &state.session {
        body_lines.push(Line::from(Span::raw(session_id_line(&session.id))).bold());
        body_lines.push(Line::from(Span::raw(format!("  {}", session.dir)).dim()));
        body_lines.push(Line::from(Span::raw(mic_line(
            session.mic_rate,
            session.mic_channels,
        ))));
        body_lines.push(Line::from(Span::raw(speaker_line(session.speaker_rate))));
        // The meeting the session is stated to be, placed where the finish summary puts its
        // clause: the settled pick first -- its fit is Confirmed by construction, so the
        // clause reads plainly and the marker says how it got there -- else the automatic
        // guess through the composer, so a weak fit shows its caveat verbatim.
        if let Some(settled) = &state.settled {
            body_lines.push(Line::from(Span::raw(format!(
                "  meeting   {}  (by hand)",
                settled.clause()
            ))));
        } else if let Some(guess) = &state.guess {
            body_lines.push(Line::from(Span::raw(meeting_clause_line(guess))));
        }
    }

    if let Some(last) = &state.last {
        let mut spans = vec![Span::raw(format!("recorded {}", last.id)).dim()];
        if let Some(meeting) = &last.meeting {
            spans.push(Span::raw(format!("  · {}", meeting.clause())).dim());
        }
        body_lines.push(Line::from(spans));
    }

    // The selector takes the notice region rather than growing the frame: at the 80x24 floor
    // the body has exactly this much room to spare. Each row is the projection's own line; the
    // cursor row is highlighted, and the guess is marked the way the correction command marks
    // the attached one -- adapted, since what the rules would attach is not yet attached.
    if state.selector_open {
        body_lines.push(Line::from(Span::raw(" ")));
        if state.offered.is_empty() {
            // The degraded view, in the correction command's own words: no grant, empty
            // calendar or unreadable store all land here, and none of them is an error.
            body_lines.push(Line::from(Span::raw(
                "No meeting is on the calendar around this session, or the calendar could not be read",
            )
            .dim()));
        } else {
            for (nth, offer) in state.offered.iter().enumerate() {
                let marker = if state
                    .guess
                    .as_ref()
                    .is_some_and(|guess| guess.event_id == offer.event_id)
                {
                    "  <- the guess"
                } else {
                    ""
                };
                let line = format!("{:>2}  {}{marker}", nth + 1, offer.line());
                body_lines.push(if nth == state.cursor {
                    Line::from(Span::raw(line).bold())
                } else {
                    Line::from(Span::raw(line))
                });
            }
        }
    } else if let Some(notice) = &state.notice {
        body_lines.push(Line::from(Span::raw(" ")));
        body_lines.push(Line::from(Span::raw(notice.clone())));
    }

    body_lines.push(Line::from(Span::raw(" ")));
    // Contextual bindings: the base stop always, and the meeting keys only while a session is
    // recording -- a key that cannot work in the current context is not advertised, and
    // Enter's wording follows what it would do now.
    let mut hint = vec![
        Span::raw("Ctrl-C / Ctrl-D").bold(),
        Span::raw("  stop and exit").dim(),
    ];
    if state.phase == Phase::Recording {
        if state.selector_open {
            hint.push(Span::raw("  up/down move  enter choose  esc back").dim());
        } else {
            hint.push(Span::raw("  enter choose a meeting").dim());
        }
    }
    body_lines.push(Line::from(hint));

    if trouble_height > 0 {
        let [main, trouble] = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(trouble_height)])
            .areas(body);
        frame.render_widget(Paragraph::new(body_lines).wrap(Wrap { trim: true }), main);

        // Each stashed note arrives composed, trailing newline included; the pane supplies its
        // own line breaks.
        let lines: Vec<Line> = state
            .trouble
            .iter()
            .map(|t| Line::from(t.trim_end().to_string()))
            .collect();
        frame.render_widget(
            Paragraph::new(lines)
                .block(Block::default().borders(Borders::ALL).title(" trouble "))
                .wrap(Wrap { trim: true }),
            trouble,
        );
    } else {
        frame.render_widget(Paragraph::new(body_lines).wrap(Wrap { trim: true }), body);
    }

    frame.render_widget(
        Line::from(
            Span::raw(format!(
                "sessions  {}",
                state.paths.sessions_dir().display()
            ))
            .dim(),
        ),
        footer,
    );
}

/// The first line of the body: what the run is doing right now, and since when if it is
/// recording.
fn status_line(state: &State, elapsed: Option<Duration>) -> Line<'_> {
    match state.phase {
        Phase::Idle => Line::from(Span::raw("watching for the next call").dim()),
        Phase::Beginning => Line::from(Span::raw("opening a session").dim()),
        Phase::Recording => {
            let mut spans = vec![Span::raw("● recording").green().bold()];
            if let Some(elapsed) = elapsed {
                spans.push(Span::raw(format!("  {}", format_elapsed(elapsed))).dim());
            }
            if let Some(session) = &state.session {
                spans.push(Span::raw(format!("  ·  {}", session.id)).dim());
            }
            Line::from(spans)
        }
        Phase::Finalizing => Line::from(Span::raw("finalizing the session").yellow()),
    }
}

/// `mm:ss` under an hour, `h:mm:ss` beyond it: a meeting-length clock, not a stopwatch.
fn format_elapsed(elapsed: Duration) -> String {
    let secs = elapsed.as_secs();
    if secs >= 3600 {
        format!(
            "{:01}:{:02}:{:02}",
            secs / 3600,
            (secs % 3600) / 60,
            secs % 60
        )
    } else {
        format!("{:02}:{:02}", secs / 60, secs % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{Note, STOPPING, WATCHING};
    use meethook_enroll::{MeetingLabel, MeetingOffer};
    use meethook_session::{Attendee, AttendeeStatus, Meeting, MeetingFit, SessionId};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::style::Modifier;

    /// Draws one frame into a buffer and hands back its rows, trimmed: the same read-out the
    /// assertions below need, spelled once.
    fn painted(width: u16, height: u16, state: &State, elapsed: Option<Duration>) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("a test backend");
        terminal
            .draw(|frame| draw(frame, state, elapsed))
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

    fn started(n: u8) -> Note {
        Note::SessionStarted {
            id: SessionId::parse(format!("20260809-05{n:02}00").as_str()).unwrap(),
            dir: std::path::PathBuf::from(format!("/tmp/meethook/sessions/20260809-05{n:02}00")),
            mic_rate: 48_000,
            mic_channels: 1,
            speaker_rate: 44_100,
        }
    }

    /// A candidate meeting with a chosen fit, the way the loop would hand one over.
    fn meeting_of(event_id: &str, title: &str, fit: MeetingFit) -> Meeting {
        Meeting::new(
            event_id.to_owned(),
            title.to_owned(),
            "Work".to_owned(),
            "2026-08-15T10:00:00Z".parse().unwrap(),
            "2026-08-15T11:00:00Z".parse().unwrap(),
        )
        .with_fit(fit)
    }

    /// Draws one frame and hands back each row with whether any of its cells are bold: the
    /// highlight assertions below need the style, which the plain read-out trims away.
    fn rows_highlighted(
        width: u16,
        height: u16,
        state: &State,
        elapsed: Option<Duration>,
    ) -> Vec<(String, bool)> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("a test backend");
        terminal
            .draw(|frame| draw(frame, state, elapsed))
            .expect("drawing into a buffer cannot fail");
        let buffer = terminal.backend().buffer().clone();
        (0..buffer.area.height)
            .map(|y| {
                let mut any_bold = false;
                let text: String = (0..buffer.area.width)
                    .map(|x| {
                        if buffer[(x, y)].modifier.contains(Modifier::BOLD) {
                            any_bold = true;
                        }
                        buffer[(x, y)].symbol().to_string()
                    })
                    .collect();
                (text.trim_end().to_string(), any_bold)
            })
            .collect()
    }

    /// An idle frame says what it is waiting for, where sessions will land, and how to leave.
    #[test]
    fn the_idle_frame_says_what_it_is_waiting_for() {
        let mut state = State::default();
        state.apply(&Note::Watching);
        let painted = painted(80, 24, &state, None).join("\n");
        assert!(painted.contains("meethook record"), "{painted}");
        assert!(painted.contains("watching"), "{painted}");
        assert!(painted.contains(WATCHING), "{painted}");
        assert!(painted.contains("Ctrl-C"), "{painted}");
        assert!(painted.contains("stop and exit"), "{painted}");
        assert!(painted.contains("sessions"), "{painted}");
        // No session is recording, so the meeting keys are not advertised: a key that cannot
        // work in the current context is not offered.
        assert!(!painted.contains("choose a meeting"), "{painted}");
    }

    /// A recording frame shows the live session the way the plain run announced it: id,
    /// directory, both measured rates -- plus the clock the plain run has no business printing.
    #[test]
    fn the_recording_frame_shows_the_live_session_and_its_clock() {
        let mut state = State::default();
        state.apply(&Note::AlreadyActive);
        state.apply(&started(7));
        let painted = painted(80, 24, &state, Some(Duration::from_secs(125))).join("\n");
        assert!(painted.contains("recording"), "{painted}");
        assert!(painted.contains("02:05"), "{painted}");
        assert!(painted.contains("20260809-050700"), "{painted}");
        assert!(
            painted.contains("/tmp/meethook/sessions/20260809-050700"),
            "{painted}"
        );
        assert!(
            painted.contains("48000 Hz, 1 channel(s) reported by the input device"),
            "{painted}"
        );
        assert!(painted.contains("44100 Hz"), "{painted}");
    }

    /// A finalizing frame keeps the notice the run gave, in the composer's words.
    #[test]
    fn the_finalizing_frame_keeps_the_notice() {
        let mut state = State::default();
        state.apply(&started(8));
        state.apply(&Note::Stopping);
        let painted = painted(80, 24, &state, None).join("\n");
        assert!(painted.contains("finalizing"), "{painted}");
        assert!(painted.contains(STOPPING), "{painted}");
    }

    /// Trouble gets its pane only when there is some, and then it says what was stashed.
    #[test]
    fn trouble_gets_a_pane_only_when_there_is_some() {
        let mut state = State::default();
        state.apply(&Note::Watching);
        let clean = painted(80, 24, &state, None).join("\n");
        assert!(!clean.contains("trouble"), "{clean}");

        state.apply(&Note::BeginFailed("could not open the input".to_string()));
        let troubled = painted(80, 24, &state, None).join("\n");
        assert!(troubled.contains("trouble"), "{troubled}");
        assert!(troubled.contains("could not open the input"), "{troubled}");
    }

    /// A finished session stays on screen after the run goes back to watching, with its
    /// meeting clause if it had one.
    #[test]
    fn the_last_finish_stays_on_screen_with_its_clause() {
        let mut state = State::default();
        state.apply(&started(9));
        state.apply(&Note::Recorded {
            id: SessionId::parse("20260809-050900").unwrap(),
            mic_secs: 7.5,
            speaker_secs: 7.5,
            dir: std::path::PathBuf::from("/tmp/meethook/sessions/20260809-050900"),
            meeting: None,
        });
        state.apply(&Note::Watching);
        let painted = painted(80, 24, &state, None).join("\n");
        assert!(painted.contains("recorded 20260809-050900"), "{painted}");
        assert!(painted.contains("watching"), "{painted}");
    }

    /// The clock formats as a meeting length, not a stopwatch.
    #[test]
    fn the_clock_reads_as_a_meeting_length() {
        assert_eq!(format_elapsed(Duration::from_secs(5)), "00:05");
        assert_eq!(format_elapsed(Duration::from_secs(3600)), "1:00:00");
        assert_eq!(format_elapsed(Duration::from_secs(3661)), "1:01:01");
    }

    /// Privacy from day one, constraint 6: a meeting carrying every field that must never reach
    /// a terminal crosses into the frame as title and fit only. The negative list is asserted
    /// against the *pixels* rather than the state, so a projection widened by accident fails
    /// here instead of at a user's screen.
    #[test]
    fn the_frame_never_paints_an_attendee_or_the_invite() {
        use meethook_enroll::MeetingLabel;
        use meethook_session::{Attendee, AttendeeStatus, Meeting};

        let meeting = Meeting::new(
            "EVENT-ABC".to_owned(),
            "Incident review".to_owned(),
            "Work".to_owned(),
            "2026-08-15T10:00:00Z".parse().unwrap(),
            "2026-08-15T11:00:00Z".parse().unwrap(),
        )
        .with_people(
            Some(Attendee {
                name: Some("Alan Turing".to_owned()),
                email: Some("alan@example.com".to_owned()),
                status: AttendeeStatus::Accepted,
                is_you: false,
            }),
            vec![Attendee {
                name: Some("Grace Hopper".to_owned()),
                email: Some("grace@example.com".to_owned()),
                status: AttendeeStatus::Accepted,
                is_you: true,
            }],
        )
        .with_invite(
            None,
            Some("Babbage Room".to_owned()),
            Some("Dial-in 555-0100, passcode 481516".to_owned()),
        );

        let mut state = State::default();
        state.apply(&started(11));
        state.apply(&Note::Recorded {
            id: SessionId::parse("20260809-051100").unwrap(),
            mic_secs: 7.5,
            speaker_secs: 7.5,
            dir: "/tmp/meethook/sessions/20260809-051100".into(),
            meeting: Some(MeetingLabel::from(&meeting)),
        });

        let painted = painted(80, 24, &state, None).join("\n");
        assert!(painted.contains("recorded"), "{painted}");
        assert!(painted.contains("Incident review"), "{painted}");
        for secret in [
            "Turing",
            "Hopper",
            "@example.com",
            "@",
            "Babbage",
            "Dial-in",
            "481516",
        ] {
            assert!(
                !painted.contains(secret),
                "the frame leaks {secret:?}: {painted}"
            );
        }
    }

    /// While a session is live, the frame states the meeting the automatic rule would attach,
    /// through the composer -- so a weak fit carries its caveat verbatim -- and advertises the
    /// key that opens the list.
    #[test]
    fn the_live_meeting_line_carries_the_guess_and_its_caveat() {
        let guess = meeting_of("EVENT-A", "Standup", MeetingFit::JoinedLate);
        let mut state = State::default();
        state.apply(&started(13));
        state.apply(&Note::MeetingOffered {
            offered: vec![MeetingOffer::from(&guess)],
            guess: Some(MeetingLabel::from(&guess)),
        });

        let painted = painted(80, 24, &state, Some(Duration::from_secs(60))).join("\n");
        assert!(painted.contains("recording"), "{painted}");
        assert!(painted.contains("Standup"), "{painted}");
        // The caveat is longer than the 80-column floor once composed onto the line, so it
        // wraps -- assert the halves rather than the whole sentence.
        assert!(
            painted.contains("uncertain:"),
            "the weak fit's caveat is missing: {painted}"
        );
        assert!(
            painted.contains("began after this meeting had"),
            "the weak fit's caveat is missing: {painted}"
        );
        assert!(painted.contains("enter choose a meeting"), "{painted}");
    }

    /// A settled pick supersedes the guess wherever the frame states the meeting: the clause
    /// reads plainly -- Confirmed carries no caveat -- with a marker saying how it got there.
    #[test]
    fn a_settled_pick_replaces_the_guess_on_the_frame() {
        let guess = meeting_of("EVENT-A", "Standup", MeetingFit::JoinedLate);
        let picked = meeting_of("EVENT-B", "Incident review", MeetingFit::Confirmed);
        let mut state = State::default();
        state.apply(&started(14));
        state.apply(&Note::MeetingOffered {
            offered: vec![MeetingOffer::from(&guess), MeetingOffer::from(&picked)],
            guess: Some(MeetingLabel::from(&guess)),
        });
        state.apply(&Note::MeetingSettled {
            label: MeetingLabel::from(&picked),
        });

        let painted = painted(80, 24, &state, None).join("\n");
        assert!(painted.contains("Incident review"), "{painted}");
        assert!(painted.contains("(by hand)"), "{painted}");
        assert!(
            !painted.contains("uncertain:"),
            "the guess's caveat survived the pick: {painted}"
        );
    }

    /// The open selector lists every offer through the projection's own line, marks the guess
    /// the way the correction command marks the attached one, and highlights only the row under
    /// the cursor -- here the second, after one move down.
    #[test]
    fn an_open_selector_lists_marks_and_highlights() {
        let guess = meeting_of("EVENT-A", "Standup", MeetingFit::Started);
        let other = meeting_of("EVENT-B", "Planning", MeetingFit::Unknown);
        let mut state = State::default();
        state.apply(&started(15));
        state.apply(&Note::MeetingOffered {
            offered: vec![MeetingOffer::from(&guess), MeetingOffer::from(&other)],
            guess: Some(MeetingLabel::from(&guess)),
        });
        state.open_selector();
        state.next();

        let rows = rows_highlighted(80, 24, &state, None);
        // The quoted titles discriminate the selector's rows from the session block's
        // meeting line, which states the same guess unquoted.
        let standup = rows
            .iter()
            .find(|(text, _)| text.contains("\"Standup\""))
            .expect("the guess offer is listed: {rows:?}");
        let planning = rows
            .iter()
            .find(|(text, _)| text.contains("\"Planning\""))
            .expect("the second offer is listed: {rows:?}");
        assert!(
            standup.0.contains("<- the guess"),
            "the guess is not marked: {rows:?}"
        );
        assert!(
            !planning.0.contains("<- the guess"),
            "the marker follows the guess, not the cursor: {rows:?}"
        );
        assert!(planning.1, "the cursor row is highlighted: {rows:?}");
        assert!(!standup.1, "only the cursor row is highlighted: {rows:?}");
        assert!(
            rows.iter().any(|(text, _)| text.contains("up/down move")),
            "the open selector advertises its keys: {rows:?}"
        );
        assert!(
            rows.iter().any(|(text, _)| text.contains("esc back")),
            "the open selector advertises its escape: {rows:?}"
        );
    }

    /// The degraded view: no grant, empty calendar or unreadable store all open the selector
    /// to the correction command's own sentence rather than a blank area or an error.
    #[test]
    fn an_empty_offer_list_says_nothing_is_offered() {
        let mut state = State::default();
        state.apply(&started(16));
        state.apply(&Note::MeetingOffered {
            offered: vec![],
            guess: None,
        });
        state.open_selector();

        let painted = painted(80, 24, &state, None).join("\n");
        // The sentence is longer than the 80-column floor, so it wraps -- assert the halves
        // rather than the whole line.
        assert!(
            painted.contains("No meeting is on the calendar around this session"),
            "{painted}"
        );
        assert!(painted.contains("could not be"), "{painted}");
    }

    /// Privacy at the type boundary the live selector crosses: a meeting carrying every field
    /// that must never reach a terminal projects into offers, and the painted pixels carry the
    /// title and count -- and nothing from the secret list. The same promise as the finish
    /// line's negative test, asserted against the selector's rows instead of the summary.
    #[test]
    fn the_selector_never_paints_an_attendee_or_the_invite() {
        let meeting = Meeting::new(
            "EVENT-ABC".to_owned(),
            "Incident review".to_owned(),
            "Work".to_owned(),
            "2026-08-15T10:00:00Z".parse().unwrap(),
            "2026-08-15T11:00:00Z".parse().unwrap(),
        )
        .with_people(
            Some(Attendee {
                name: Some("Alan Turing".to_owned()),
                email: Some("alan@example.com".to_owned()),
                status: AttendeeStatus::Accepted,
                is_you: false,
            }),
            vec![Attendee {
                name: Some("Grace Hopper".to_owned()),
                email: Some("grace@example.com".to_owned()),
                status: AttendeeStatus::Accepted,
                is_you: true,
            }],
        )
        .with_invite(
            Some("https://example.com/j/12345".to_owned()),
            Some("Babbage Room".to_owned()),
            Some("Dial-in 555-0100, passcode 481516".to_owned()),
        );

        let mut state = State::default();
        state.apply(&started(17));
        state.apply(&Note::MeetingOffered {
            offered: vec![MeetingOffer::from(&meeting)],
            guess: Some(MeetingLabel::from(&meeting)),
        });
        state.open_selector();

        let painted = painted(80, 24, &state, None).join("\n");
        assert!(painted.contains("Incident review"), "{painted}");
        assert!(painted.contains("attendee(s)"), "{painted}");
        for secret in [
            "Turing",
            "Hopper",
            "@example.com",
            "@",
            "Babbage",
            "Dial-in",
            "481516",
            "example.com",
        ] {
            assert!(
                !painted.contains(secret),
                "the selector leaks {secret:?}: {painted}"
            );
        }
    }

    /// A terminal too small to mean anything draws nothing rather than a wall of borders: the
    /// guard is total, so every pane survives any size the terminal might report.
    #[test]
    fn every_pane_survives_a_terminal_too_small_to_mean_anything() {
        let mut state = State::default();
        state.apply(&started(12));
        state.apply(&Note::BeginFailed("could not open the input".to_string()));

        // Below the floor: nothing is drawn, and drawing into it cannot fail.
        let blank = painted(19, 4, &state, Some(Duration::from_secs(5)));
        assert!(blank.iter().all(|row| row.is_empty()), "{blank:?}");

        // At the floor: something is drawn, and it still does not panic with a clock running
        // and trouble stashed -- the two things that make a frame the busiest it gets.
        let squeezed = painted(20, 5, &state, Some(Duration::from_secs(3661)));
        assert!(squeezed.iter().any(|row| !row.is_empty()), "{squeezed:?}");
    }
}
