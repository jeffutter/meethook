//! Correcting, or clearing, the meeting a session was labelled with.
//!
//! The calendar guess is right most of the time and cannot be right always. A session that
//! begins twenty minutes into an hour-long invite is either a late join or an unrelated call,
//! and no rule over start and end times separates the two -- which is why `MeetingFit` marks
//! that case rather than deciding it. A session recorded during a double-booked hour resolves
//! to whichever candidate the recorder's `select` prefers, which is the shorter one and not
//! necessarily the one the user actually attended. The remaining answer in both cases is a
//! human who knows which it was.
//!
//! `enroll` exists on exactly that premise for speaker identification: an automated guess, a
//! human correction, and a transcript rewritten in place. This is the same shape aimed at the
//! meeting label, and it lives in this crate for that reason -- plus a practical one. The
//! correction has to re-render a `transcript.md` it did not write, which only `enroll` and
//! `forget` already do, and it must be decidable in `cargo test` on a machine with no calendar
//! grant. Depending on `meethook-record` would put ScreenCaptureKit, Core Audio and EventKit
//! behind every one of these tests; a [`MeetingSource`] instead is the same split
//! `calendar.rs` already makes between fetching candidates and deciding between them.
//!
//! # Preview, then act
//!
//! `meethook meeting <id>` with no flag prints the current label and the numbered candidates
//! and writes nothing. `--event N` attaches the Nth. `--clear` removes the label. That is
//! `forget`'s Preview/Confirmed shape rather than `enroll`'s [`crate::Interviewer`]: an
//! interview exists because a person has to *listen to a clip* before answering, and there is
//! nothing to listen to here.
//!
//! It is deliberately not `--yes`-gated the way `forget` is. A forgotten reference cannot be
//! rebuilt; a mislabelled session is corrected by running this command again, so a second
//! confirmation would be ceremony over a reversible act.
//!
//! # What is printed
//!
//! Times, title, calendar name and an attendee *count*. Never an attendee name or address,
//! never the organizer, the location, the URL or the invite body -- the rule `meethook
//! record`'s own meeting line already prints in front of, applied here where a user is
//! choosing *between* meetings and the temptation to render more of them is strongest.
//! `location` is excluded with the rest because organizers routinely paste a join URL into it,
//! and an invite body is the single most likely field in the tree to carry a dial-in PIN.

use std::io::Write;

use jiff::Timestamp;
use meethook_session::{
    Meeting, Paths, SessionId, SessionMetadata, Transcript, TranscriptContext, TranscriptTemplate,
};

use crate::Result;

/// Where the meetings offered as a correction come from.
///
/// One method, and the whole of what this module needs a calendar for. The live implementation
/// is `meethook_record::meetings_around` and lives in the CLI crate; the test one answers from
/// a list, which is what makes every rule below decidable with no grant, no EventKit and no
/// events anywhere near it.
///
/// Total by construction, exactly as the lookup it wraps is: no grant, no events and a
/// framework failure are all an empty `Vec`. A correction command must not fail because the
/// calendar was unreachable, and its other half -- clearing a label -- must not consult one at
/// all.
pub trait MeetingSource {
    /// Every meeting worth offering for a session that started at `at`, in a stable order.
    fn around(&self, at: Timestamp) -> Vec<Meeting>;
}

/// What one run of the correction was asked to do.
pub enum MeetingChoice {
    /// Print the current label and the candidates, and write nothing.
    Show,
    /// Attach the Nth offered meeting. **1-based, exactly as the listing prints it.**
    Event(usize),
    /// Remove the label: this session was not recorded during any meeting.
    Clear,
}

/// How a correction ended, so a caller can pick an exit status without re-deriving anything.
pub enum Labelled {
    /// The listing was printed. Nothing was written.
    Shown,
    /// `session.json` was rewritten, and the transcript with it where there was one.
    Written(Relabelling),
    /// There is no such offered event, so nothing was written. The count has been printed.
    NoSuchEvent { offered: usize },
}

/// What a correction did, derived once and printed from.
pub struct Relabelling {
    pub session: SessionId,
    /// The title of the meeting now attached, or `None` when the label was cleared.
    pub title: Option<String>,
    /// Whether a `transcript.md` was brought in line. False means there was none to bring --
    /// the session has not been transcribed yet -- rather than that one was left stale.
    pub transcript_rewritten: bool,
}

/// Prints the current label and the meetings around the session, and -- for
/// [`MeetingChoice::Event`] and [`MeetingChoice::Clear`] -- rewrites it.
///
/// # Order of operations, and what each buys
///
/// 1. Read `session.json`. A session id with no directory, or an unreadable file, is an `Err`
///    naming the path, as in every other reader in this crate.
/// 2. **[`MeetingChoice::Clear`] never touches `calendar`.** The seam is a parameter and the
///    clear path simply does not call it, so "clearing works with no calendar access at all"
///    is structural rather than a branch that has to be got right. The test passes a source
///    that panics if consulted.
/// 3. [`MeetingChoice::Show`] and [`MeetingChoice::Event`] ask the calendar **once**, and both
///    the printed numbering and the index address that one `Vec` -- so the listing and the
///    pick cannot disagree about what "2" means.
/// 4. An index outside `1..=len` prints the count and returns [`Labelled::NoSuchEvent`],
///    having written nothing.
/// 5. `session.json` first, then the transcript. [`SessionMetadata::write`] is atomic, so the
///    marker that a session completed is replaced wholly or not at all; and the authoritative
///    file going first means an interrupt leaves a stale transcript that re-running this
///    command brings in line, rather than a transcript no on-disk state justifies. `forget`'s
///    write order, for `forget`'s reason.
pub fn run_meeting(
    paths: &Paths,
    session: &SessionId,
    choice: MeetingChoice,
    calendar: &dyn MeetingSource,
    template: &TranscriptTemplate,
    out: &mut dyn Write,
) -> Result<Labelled> {
    let session_paths = paths.session(session);
    let mut metadata = SessionMetadata::read(&session_paths.session_json())?;

    writeln!(out, "{session}  {}", current_label(&metadata))?;

    let chosen = match choice {
        MeetingChoice::Clear => None,
        MeetingChoice::Show | MeetingChoice::Event(_) => {
            let offered = calendar.around(metadata.start_time);
            let index = match choice {
                MeetingChoice::Event(index) => index,
                _ => {
                    write_offer(out, session, &metadata, &offered)?;
                    return Ok(Labelled::Shown);
                }
            };
            let Some(meeting) = index.checked_sub(1).and_then(|nth| offered.get(nth)) else {
                writeln!(
                    out,
                    "There is no meeting {index} to attach: {} were offered around this session",
                    offered.len()
                )?;
                writeln!(
                    out,
                    "meethook meeting {session} lists them, and --clear removes the label \
                     altogether"
                )?;
                return Ok(Labelled::NoSuchEvent {
                    offered: offered.len(),
                });
            };
            Some(meeting.clone())
        }
    };

    let title = chosen.as_ref().map(|meeting| meeting.title.clone());
    metadata.label_by_hand(chosen);
    metadata.write(&session_paths.session_json())?;
    match &title {
        Some(title) => writeln!(out, "Labelled {session} as {title:?}")?,
        None => writeln!(
            out,
            "Cleared the meeting from {session}: it was not recorded during one"
        )?,
    }

    // Only a transcribed session has a `transcript.md` to bring in line, and `transcript.json`
    // is this tool's "already transcribed" marker everywhere else, so it is the marker here
    // too. An untranscribed session is a success: the label it was just given is what the
    // transcription will render.
    let transcript_json = session_paths.transcript_json();
    if !transcript_json.exists() {
        writeln!(
            out,
            "Not transcribed yet -- its transcript will carry this label when it is"
        )?;
        return Ok(Labelled::Written(Relabelling {
            session: session.clone(),
            title,
            transcript_rewritten: false,
        }));
    }

    let transcript = match Transcript::read(&transcript_json) {
        Ok(transcript) => transcript,
        Err(e) => {
            // The label is already on disk -- it is the authoritative file, and this is
            // exactly the interrupt the write order was chosen for. Saying so, with the remedy
            // `enroll` gives for this same file, is what turns "the transcript still names the
            // old meeting" into something the user can act on.
            writeln!(
                out,
                "The label is written, but the transcript could not be re-rendered: \
                 re-transcribe this session with --force, or run this command again"
            )?;
            return Err(e.into());
        }
    };
    transcript.write(&session_paths, template, &TranscriptContext::now(&metadata))?;
    writeln!(out, "Transcript brought in line")?;

    Ok(Labelled::Written(Relabelling {
        session: session.clone(),
        title,
        transcript_rewritten: true,
    }))
}

/// The label the session carries right now, in one line.
///
/// Three states rather than two: no meeting was ever found, a meeting is attached, or a human
/// said there was none. The third is worth its own wording because it is the state this
/// command exists to produce, and a user re-running it needs to see that their correction
/// stuck rather than that the lookup found nothing again.
fn current_label(metadata: &SessionMetadata) -> String {
    match &metadata.meeting {
        Some(meeting) => {
            let mut line = format!("is labelled {}", describe(meeting));
            if let Some(caveat) = meeting.fit.caveat() {
                line.push_str(&format!("  ({caveat})"));
            }
            line
        }
        None if metadata.meeting_cleared => {
            "has no meeting: cleared by hand, and nothing will guess at it again".to_owned()
        }
        None => "has no meeting".to_owned(),
    }
}

/// The candidates, numbered as `--event` addresses them, plus what to do with them.
///
/// An empty calendar is a sentence rather than blank space, for the reason `forget`'s report
/// gives: an offer whose value is its completeness cannot express "nothing" by omission. It
/// also cannot tell a calendar with no events around this session from one it could not read
/// -- that degradation is deliberate and total -- so the wording covers both and points at the
/// half of this command that needs no calendar.
fn write_offer(
    out: &mut dyn Write,
    session: &SessionId,
    metadata: &SessionMetadata,
    offered: &[Meeting],
) -> Result<()> {
    if offered.is_empty() {
        writeln!(
            out,
            "No meeting is on the calendar around this session, or the calendar could not be read"
        )?;
        writeln!(
            out,
            "meethook meeting {session} --clear records that it was not recorded during one"
        )?;
        return Ok(());
    }

    writeln!(out, "{} meeting(s) around it:", offered.len())?;
    let attached = metadata
        .meeting
        .as_ref()
        .map(|meeting| meeting.event_id.as_str());
    for (nth, meeting) in offered.iter().enumerate() {
        let marker = if attached == Some(meeting.event_id.as_str()) {
            "  <- the one attached"
        } else {
            ""
        };
        writeln!(out, "  {}  {}{marker}", nth + 1, describe(meeting))?;
    }
    writeln!(
        out,
        "meethook meeting {session} --event N attaches one, --clear removes the label"
    )?;
    Ok(())
}

/// One meeting as a person may see it, and the whole of what they may see.
///
/// The title, when it ran, the calendar it is in, and how many people were invited. **Not** the
/// attendees themselves, the organizer, the location, the URL or the invite body: those reach
/// `session.json` because speaker identification and a reader's "what was this about" need
/// them, and they reach nothing else -- no terminal, no log line, no pasted error report.
///
/// One function, so the rule is a property of the code rather than a promise repeated at each
/// call site. `meethook-record`'s own `summarize` is the same arrangement for the same reason,
/// and both are tested against a meeting stuffed with everything that must not appear.
///
/// Local time rather than UTC, on the precedent a transcript's `created:` sets: this is read
/// beside a calendar the user has open, and 10:00 there has to be 10:00 here.
fn describe(meeting: &Meeting) -> String {
    format!(
        "{:?}  {} .. {}  [{}]  {} attendee(s)",
        meeting.title,
        local(meeting.start, "%Y-%m-%d %H:%M"),
        local(meeting.end, "%H:%M"),
        meeting.calendar,
        meeting.attendee_count(),
    )
}

/// An instant in the machine's own zone, which is the one the calendar was written in.
fn local(at: Timestamp, format: &str) -> String {
    at.to_zoned(jiff::tz::TimeZone::system())
        .strftime(format)
        .to_string()
}

/// The whole command, over real session directories on a temporary disk and with no calendar
/// anywhere near it.
#[cfg(test)]
mod tests {
    use meethook_session::{Attendee, AttendeeStatus, MeetingFit};

    use super::*;
    use crate::tests::{files_under, make_session, session_metadata};

    /// The instant `make_session` gives every fixture session.
    fn session_start() -> Timestamp {
        "2026-08-09T05:26:00Z".parse().unwrap()
    }

    /// A calendar that answers with a fixed list, and records that it was asked.
    struct Offering(Vec<Meeting>);

    impl MeetingSource for Offering {
        fn around(&self, at: Timestamp) -> Vec<Meeting> {
            assert_eq!(at, session_start(), "the lookup must use the session start");
            self.0.clone()
        }
    }

    /// A calendar that must not be consulted at all. Clearing a label needs none, and this is
    /// how that is asserted rather than described.
    struct NoCalendar;

    impl MeetingSource for NoCalendar {
        fn around(&self, _at: Timestamp) -> Vec<Meeting> {
            unreachable!("clearing a label must not consult the calendar");
        }
    }

    fn meeting(event_id: &str, title: &str, start: &str, end: &str) -> Meeting {
        Meeting::new(
            event_id.to_owned(),
            title.to_owned(),
            "Work".to_owned(),
            start.parse().unwrap(),
            end.parse().unwrap(),
        )
    }

    /// The two meetings of the ticket's double-booking case, around the fixture session.
    fn double_booked() -> Vec<Meeting> {
        vec![
            meeting(
                "EVENT-1",
                "Standup",
                "2026-08-09T05:20:00Z",
                "2026-08-09T05:50:00Z",
            ),
            meeting(
                "EVENT-2",
                "Incident review",
                "2026-08-09T05:25:00Z",
                "2026-08-09T06:25:00Z",
            ),
        ]
    }

    fn run(
        paths: &Paths,
        id: &str,
        choice: MeetingChoice,
        calendar: &dyn MeetingSource,
    ) -> (Labelled, String) {
        let session = SessionId::parse(id).unwrap();
        let mut out = Vec::new();
        let labelled = run_meeting(
            paths,
            &session,
            choice,
            calendar,
            // Resolved from the root, exactly as the CLI does.
            &TranscriptTemplate::resolve(paths, None).unwrap(),
            &mut out,
        )
        .unwrap();
        (labelled, String::from_utf8(out).unwrap())
    }

    fn metadata_of(paths: &Paths, id: &str) -> SessionMetadata {
        let session = paths.session(&SessionId::parse(id).unwrap());
        SessionMetadata::read(&session.session_json()).unwrap()
    }

    fn frontmatter_of(paths: &Paths, id: &str) -> String {
        let session = paths.session(&SessionId::parse(id).unwrap());
        std::fs::read_to_string(session.transcript_md()).unwrap()
    }

    /// Gives a session the label the recorder's own lookup would have written -- an automatic
    /// match, not a confirmed one -- so a correction has something wrong to correct.
    fn mislabelled(paths: &Paths, id: &str, fit: MeetingFit) {
        let session = paths.session(&SessionId::parse(id).unwrap());
        let metadata = session_metadata(&SessionId::parse(id).unwrap()).with_meeting(Some(
            meeting(
                "EVENT-1",
                "Standup",
                "2026-08-09T05:20:00Z",
                "2026-08-09T05:50:00Z",
            )
            .with_fit(fit),
        ));
        metadata.write(&session.session_json()).unwrap();
        let transcript = crate::tests::transcript_of(&session);
        transcript
            .write(
                &session,
                &TranscriptTemplate::resolve(paths, None).unwrap(),
                &TranscriptContext::now(&metadata),
            )
            .unwrap();
    }

    /// Acceptance criterion #4: the whole clear path, against a calendar that panics if it is
    /// consulted. The seam is a parameter, so this is the strongest form the claim has.
    #[test]
    fn clearing_a_label_never_consults_the_calendar() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        mislabelled(&paths, "20260809-052600", MeetingFit::JoinedLate);

        let (labelled, output) = run(&paths, "20260809-052600", MeetingChoice::Clear, &NoCalendar);

        assert!(matches!(labelled, Labelled::Written(_)), "{output}");
        assert!(
            output.contains("Cleared the meeting from 20260809-052600"),
            "{output}"
        );
    }

    /// Acceptance criterion #1: what is left has to *read* as a session recorded outside any
    /// meeting -- everywhere a meeting is consumed, not merely in a field somebody remembered
    /// to check. The rendered transcript is compared against one that never had a meeting at
    /// all, byte for byte, with the render instant pinned so the comparison means something.
    #[test]
    fn a_cleared_session_reads_as_one_recorded_outside_any_meeting() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        // A second session that never had a meeting, and its rendering, as the reference.
        let never = make_session(&paths, "20260810-101500");
        let never_metadata = session_metadata(&SessionId::parse("20260810-101500").unwrap());
        mislabelled(&paths, "20260809-052600", MeetingFit::JoinedLate);

        let (_, output) = run(&paths, "20260809-052600", MeetingChoice::Clear, &NoCalendar);

        let cleared = metadata_of(&paths, "20260809-052600");
        assert!(cleared.meeting.is_none(), "{output}");
        assert!(cleared.meeting_cleared, "{output}");
        assert!(cleared.meeting_settled_by_hand(), "{output}");

        // The renderings, both pinned to one instant: the only difference between the two
        // sessions is their id, so the frontmatter above the body has to be identical.
        let pinned: Timestamp = "2026-08-09T07:00:00Z".parse().unwrap();
        let template = TranscriptTemplate::resolve(&paths, None).unwrap();
        let rendered = crate::tests::transcript_of(&paths.session(&cleared.session_id))
            .render_markdown(&template, &TranscriptContext::at(&cleared, pinned))
            .unwrap();
        let reference = crate::tests::transcript_of(&never)
            .render_markdown(&template, &TranscriptContext::at(&never_metadata, pinned))
            .unwrap();
        assert_eq!(
            head(&rendered),
            head(&reference),
            "a cleared session must render like one that never had a meeting"
        );
    }

    /// The frontmatter block of a rendering, which is where every meeting key lives.
    fn head(rendered: &str) -> &str {
        let rest = rendered.strip_prefix("---\n").expect(rendered);
        rest.split_once("\n---\n").expect(rendered).0
    }

    /// Acceptance criterion #5, the clearing direction: a `transcript.md` already rendered with
    /// the old meeting in its frontmatter is brought in line in the same run.
    #[test]
    fn clearing_strips_the_meeting_from_a_transcript_already_rendered_with_it() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        mislabelled(&paths, "20260809-052600", MeetingFit::JoinedLate);
        let before = frontmatter_of(&paths, "20260809-052600");
        assert!(before.contains("meeting_title: \"Standup\""), "{before}");
        assert!(
            before.contains("meeting_match: \"joined_late\""),
            "{before}"
        );

        let (_, output) = run(&paths, "20260809-052600", MeetingChoice::Clear, &NoCalendar);

        let after = frontmatter_of(&paths, "20260809-052600");
        for stale in [
            "meeting_title",
            "meeting_description",
            "meeting_match",
            "Standup",
        ] {
            assert!(!after.contains(stale), "{stale} survived: {after}");
        }
        assert!(output.contains("Transcript brought in line"), "{output}");
    }

    /// Acceptance criteria #2 and #5: the double-booked hour. The user picks the meeting they
    /// were actually in from the offered list -- no title typed, no event identifier -- and the
    /// transcript names it afterwards.
    #[test]
    fn attaching_an_offered_meeting_rewrites_the_label_and_the_transcript() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        mislabelled(&paths, "20260809-052600", MeetingFit::Started);

        let (labelled, output) = run(
            &paths,
            "20260809-052600",
            MeetingChoice::Event(2),
            &Offering(double_booked()),
        );

        assert!(matches!(labelled, Labelled::Written(_)), "{output}");
        let metadata = metadata_of(&paths, "20260809-052600");
        let attached = metadata.meeting.as_ref().expect("a meeting is attached");
        assert_eq!(attached.title, "Incident review", "{output}");
        assert_eq!(attached.event_id, "EVENT-2", "{output}");
        assert!(!metadata.meeting_cleared, "{output}");
        assert!(
            output.contains(r#"Labelled 20260809-052600 as "Incident review""#),
            "{output}"
        );

        let frontmatter = frontmatter_of(&paths, "20260809-052600");
        assert!(
            frontmatter.contains(r#"meeting_title: "Incident review""#),
            "{frontmatter}"
        );
        assert!(!frontmatter.contains("Standup"), "{frontmatter}");
        // A confirmed label is strong, so the tentative marker goes with the old one.
        assert!(!frontmatter.contains("meeting_match"), "{frontmatter}");
    }

    /// Acceptance criterion #3: what a human set is distinguishable from what the lookup
    /// produced, and no later pass overwrites it by guessing again -- in both directions.
    #[test]
    fn a_label_settled_by_hand_is_marked_and_is_not_re_guessed() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        make_session(&paths, "20260810-101500");
        mislabelled(&paths, "20260809-052600", MeetingFit::JoinedLate);
        mislabelled(&paths, "20260810-101500", MeetingFit::JoinedLate);

        run(
            &paths,
            "20260809-052600",
            MeetingChoice::Event(2),
            &Offering(double_booked()),
        );
        run(&paths, "20260810-101500", MeetingChoice::Clear, &NoCalendar);

        let attached = metadata_of(&paths, "20260809-052600");
        assert_eq!(
            attached.meeting.as_ref().unwrap().fit,
            MeetingFit::Confirmed,
            "an attached label carries its own provenance"
        );
        assert!(attached.meeting_settled_by_hand());
        let cleared = metadata_of(&paths, "20260810-101500");
        assert!(cleared.meeting_settled_by_hand());

        // The re-guess. `with_meeting` is the one door a later automatic pass would come
        // through, and it refuses both of these.
        let guess = meeting(
            "EVENT-9",
            "Something else",
            "2026-08-09T05:00:00Z",
            "2026-08-09T06:00:00Z",
        )
        .with_fit(MeetingFit::Started);
        let re_guessed = attached.clone().with_meeting(Some(guess.clone()));
        assert_eq!(re_guessed, attached, "a confirmed label was overwritten");
        let re_guessed = cleared.clone().with_meeting(Some(guess));
        assert_eq!(re_guessed, cleared, "a cleared label was overwritten");
    }

    /// The preview: the current label, the candidates and what to type, with nothing written.
    /// "Nothing" is byte-for-byte over every file under the root, rather than over the ones a
    /// listing was expected to leave alone.
    #[test]
    fn showing_the_candidates_writes_nothing() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        mislabelled(&paths, "20260809-052600", MeetingFit::JoinedLate);
        let before = files_under(root.path());

        let (labelled, output) = run(
            &paths,
            "20260809-052600",
            MeetingChoice::Show,
            &Offering(double_booked()),
        );

        assert!(matches!(labelled, Labelled::Shown), "{output}");
        assert_eq!(files_under(root.path()), before, "{output}");
        assert!(output.contains("2 meeting(s) around it:"), "{output}");
        assert!(output.contains("  1  \"Standup\""), "{output}");
        assert!(output.contains("  2  \"Incident review\""), "{output}");
        // The one already attached is marked, so a user can see what they are changing from.
        assert!(output.contains("<- the one attached"), "{output}");
        assert!(
            output.contains("uncertain: the recording began after this meeting had started"),
            "the caveat on the current label is what sent the user here: {output}"
        );
        assert!(
            output.contains("meethook meeting 20260809-052600 --event N"),
            "{output}"
        );
    }

    /// An empty offer is a sentence rather than blank space, and it points at the half of the
    /// command that needs no calendar -- which is also the answer when the calendar simply
    /// could not be read, since the two are deliberately indistinguishable here.
    #[test]
    fn nothing_offered_says_so_and_names_the_way_out() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let (labelled, output) = run(
            &paths,
            "20260809-052600",
            MeetingChoice::Show,
            &Offering(Vec::new()),
        );

        assert!(matches!(labelled, Labelled::Shown), "{output}");
        assert!(
            output.contains("20260809-052600  has no meeting"),
            "{output}"
        );
        assert!(
            output.contains("No meeting is on the calendar around this session"),
            "{output}"
        );
        assert!(output.contains("--clear"), "{output}");
    }

    /// An index nobody offered writes nothing and says how many there were. `--event 0` cannot
    /// reach here -- the CLI refuses it where the range can be named -- but the library still
    /// has to be total over it.
    #[test]
    fn an_index_that_was_not_offered_writes_nothing() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        let before = files_under(root.path());

        for index in [0, 9] {
            let (labelled, output) = run(
                &paths,
                "20260809-052600",
                MeetingChoice::Event(index),
                &Offering(double_booked()),
            );

            assert!(
                matches!(labelled, Labelled::NoSuchEvent { offered: 2 }),
                "{output}"
            );
            assert!(
                output.contains(&format!(
                    "There is no meeting {index} to attach: 2 were offered"
                )),
                "{output}"
            );
            assert_eq!(files_under(root.path()), before, "{output}");
        }
    }

    /// A session that has not been transcribed yet is a success, and says why nothing was
    /// re-rendered: the label it was just given is what its transcription will carry.
    #[test]
    fn a_session_with_no_transcript_is_labelled_and_says_so() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        std::fs::remove_file(session.transcript_json()).unwrap();
        std::fs::remove_file(session.transcript_md()).unwrap();

        let (labelled, output) = run(
            &paths,
            "20260809-052600",
            MeetingChoice::Event(1),
            &Offering(double_booked()),
        );

        match labelled {
            Labelled::Written(relabelling) => {
                assert_eq!(relabelling.title.as_deref(), Some("Standup"), "{output}");
                assert!(!relabelling.transcript_rewritten, "{output}");
            }
            _ => panic!("expected a write: {output}"),
        }
        assert!(output.contains("Not transcribed yet"), "{output}");
        assert!(!session.transcript_md().exists(), "{output}");
    }

    /// Acceptance criterion #6, in the one place this command renders a meeting for a person.
    /// The current label, the offered list and the confirmation line are all checked at once,
    /// against meetings carrying everything that must never appear on a terminal.
    #[test]
    fn nothing_printed_names_an_attendee_or_quotes_the_invite() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");

        let people = |meeting: Meeting| {
            meeting
                .with_people(
                    Some(Attendee {
                        name: Some("Alan Turing".to_owned()),
                        email: Some("alan@example.com".to_owned()),
                        status: AttendeeStatus::Accepted,
                        is_you: false,
                    }),
                    vec![
                        Attendee {
                            name: Some("Grace Hopper".to_owned()),
                            email: Some("grace@example.com".to_owned()),
                            status: AttendeeStatus::Accepted,
                            is_you: false,
                        },
                        Attendee {
                            name: Some("Ada Lovelace".to_owned()),
                            email: Some("ada@example.com".to_owned()),
                            status: AttendeeStatus::Accepted,
                            is_you: true,
                        },
                    ],
                )
                .with_invite(
                    Some("https://example.com/j/12345".to_owned()),
                    Some("Babbage Room, 12 Ada Street".to_owned()),
                    Some("Dial-in 555-0100, passcode 481516".to_owned()),
                )
        };
        let offered: Vec<Meeting> = double_booked().into_iter().map(people).collect();
        // The session starts out labelled with one of them, so the "current label" line is
        // rendering a fully-populated meeting too.
        let session = paths.session(&SessionId::parse("20260809-052600").unwrap());
        let mut metadata = session_metadata(&SessionId::parse("20260809-052600").unwrap());
        metadata.label_by_hand(Some(offered[0].clone()));
        metadata.write(&session.session_json()).unwrap();

        let (_, listing) = run(
            &paths,
            "20260809-052600",
            MeetingChoice::Show,
            &Offering(offered.clone()),
        );
        let (_, attached) = run(
            &paths,
            "20260809-052600",
            MeetingChoice::Event(2),
            &Offering(offered),
        );

        let printed = format!("{listing}{attached}");
        // What makes the listing usable is still there.
        assert!(printed.contains("Standup"), "{printed}");
        assert!(printed.contains("2 attendee(s)"), "{printed}");
        for secret in [
            "Grace",
            "Hopper",
            "grace@example.com",
            "Ada Lovelace",
            "ada@example.com",
            "Alan",
            "Turing",
            "alan@example.com",
            "@",
            "Dial-in",
            "555-0100",
            "passcode",
            "481516",
            "Babbage",
            "12 Ada Street",
            "example.com/j/12345",
        ] {
            assert!(
                !printed.contains(secret),
                "the output leaks {secret:?}: {printed}"
            );
        }
    }
}
