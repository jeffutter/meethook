//! meeting labels on the queue line and across the Interviewer seam.

use super::*;

/// Gives a fixture session the meeting label the recorder's lookup would have written,
/// with the fit given: `make_session` writes sessions without meetings, so the label is
/// attached by rewriting `session.json`, the way the `meeting.rs` fixtures do.
fn labelled_meeting(paths: &Paths, id: &str, fit: MeetingFit) {
    let session = paths.session(&SessionId::parse(id).unwrap());
    let metadata = session_metadata(&SessionId::parse(id).unwrap()).with_meeting(Some(
        Meeting::new(
            "EVENT-1".to_owned(),
            "Incident review".to_owned(),
            "Work".to_owned(),
            "2026-08-09T05:20:00Z".parse().unwrap(),
            "2026-08-09T06:20:00Z".parse().unwrap(),
        )
        .with_fit(fit),
    ));
    metadata.write(&session.session_json()).unwrap();
}

/// The one display shape, pinned byte for byte over every fit: the title alone when the
/// fit states it plainly, the title plus the fit's own caveat otherwise.
#[test]
fn a_meeting_label_states_a_strong_fit_plainly_and_qualifies_the_rest() {
    for fit in [
        MeetingFit::Started,
        MeetingFit::StartedEarly,
        MeetingFit::Confirmed,
    ] {
        let label = MeetingLabel {
            title: "Standup".to_owned(),
            fit,
        };
        assert_eq!(label.clause(), "Standup", "{fit:?}");
    }
    assert_eq!(
        MeetingLabel {
            title: "Standup".to_owned(),
            fit: MeetingFit::JoinedLate,
        }
        .clause(),
        "Standup  (uncertain: the recording began after this meeting had started)"
    );
    assert_eq!(
        MeetingLabel {
            title: "Standup".to_owned(),
            fit: MeetingFit::AfterEnd,
        }
        .clause(),
        "Standup  (uncertain: the recording began after this meeting had ended)"
    );
    assert_eq!(
        MeetingLabel {
            title: "Standup".to_owned(),
            fit: MeetingFit::Unknown,
        }
        .clause(),
        "Standup  (unverified: this session was recorded before meethook scored the match)"
    );
}

/// Acceptance criteria #1, #2 and #5, over every fit: the queue announcement names the
/// meeting once, under the count line, unqualified when the fit is strong and with the
/// same caveat `meethook record` prints when it is not -- and the two voices then
/// prompted about do not repeat it.
#[test]
fn the_queue_line_names_the_meeting_once_and_qualifies_it_as_record_does() {
    for fit in MeetingFit::ALL {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600");
        labelled_meeting(&paths, "20260809-052600", fit);

        let mut interviewer = Scripted::answering(vec![named("Alice"), named("Aaron")]);
        let (_, output) = run(&paths, &[], &mut interviewer);

        let head = "20260809-052600  2 unresolved voice(s)\n    meeting   Incident review";
        if let Some(caveat) = fit.caveat() {
            assert!(
                output.contains(&format!("{head}  ({caveat})\n")),
                "the weak fit must carry its caveat: {fit:?}: {output}"
            );
        } else {
            assert!(
                output.contains(&format!("{head}\n")),
                "the strong fit must be stated plainly: {fit:?}: {output}"
            );
        }
        // Once, with the session, not once per voice: both answers land afterwards and
        // neither may name the meeting again.
        assert_eq!(
            output.matches("Incident review").count(),
            1,
            "the meeting is named once, where the session is announced: {fit:?}: {output}"
        );
    }
}

/// The sub-line sits under the held-back clause too, which is the other shape the count
/// line takes: a meeting is not only named on the plain path.
#[test]
fn the_queue_line_sits_under_the_held_back_clause_as_well() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_fragmented_session(&paths, "20260809-052600");
    labelled_meeting(&paths, "20260809-052600", MeetingFit::Started);

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (_, output) = run(&paths, &[], &mut interviewer);

    assert!(
        output.contains(
            "20260809-052600  1 unresolved voice(s), 3 quieter voice(s) not offered -- \
             meethook enroll --all\n    meeting   Incident review\n"
        ),
        "{output}"
    );
}

/// Acceptance criterion #3: a meeting carrying everything that must not reach a terminal
/// leaks none of it -- the queue line is the title and the fit, and nothing off the
/// roster.
#[test]
fn the_queue_line_leaks_nothing_off_the_roster() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    let session = paths.session(&SessionId::parse("20260809-052600").unwrap());
    let metadata =
        session_metadata(&SessionId::parse("20260809-052600").unwrap()).with_meeting(Some(
            Meeting::new(
                "EVENT-1".to_owned(),
                "Incident review".to_owned(),
                "Work".to_owned(),
                "2026-08-09T05:20:00Z".parse().unwrap(),
                "2026-08-09T06:20:00Z".parse().unwrap(),
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
                Some("Babbage Room, 12 Ada Street".to_owned()),
                Some("Dial-in 555-0100, passcode 481516".to_owned()),
            )
            .with_fit(MeetingFit::JoinedLate),
        ));
    metadata.write(&session.session_json()).unwrap();

    let mut interviewer = Scripted::answering(vec![named("Alice"), named("Aaron")]);
    let (_, output) = run(&paths, &[], &mut interviewer);

    assert!(output.contains("Incident review"), "{output}");
    for secret in [
        "Turing",
        "Hopper",
        "@",
        "Babbage",
        "Ada Street",
        "example.com",
        "Dial-in",
        "555-0100",
        "passcode",
        "481516",
    ] {
        assert!(
            !output.contains(secret),
            "the queue line leaks {secret:?}: {output}"
        );
    }
}

/// Acceptance criterion #4, the absence half: a session with no meeting says nothing
/// about meetings -- no reserved row, no empty label. The word does not appear anywhere
/// else in enroll's output, so its absence is the whole claim; the byte-identity itself
/// is pinned by `one_runs_narration_reads_as_these_lines_in_this_order`, whose fixtures
/// carry no meetings.
#[test]
fn a_session_without_a_meeting_says_nothing_about_meetings() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![named("Alice"), named("Aaron")]);
    let (_, output) = run(&paths, &[], &mut interviewer);

    assert!(!output.contains("meeting"), "{output}");
}

/// TASK-051.02 acceptance criterion #6: the meeting reaches an interface across the
/// Interviewer seam -- every voice of a labelled session is handed the same title and fit,
/// and a session without one is handed `None` rather than a value an interface could only
/// have gotten by reading `session.json` behind the seam's back.
#[test]
fn the_seam_hands_every_voice_the_meeting_it_was_recorded_during() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    labelled_meeting(&paths, "20260809-052600", MeetingFit::JoinedLate);

    let mut interviewer = Scripted::answering(vec![named("Alice"), named("Aaron")]);
    run(&paths, &[], &mut interviewer);

    let expected = Some(MeetingLabel {
        title: "Incident review".to_owned(),
        fit: MeetingFit::JoinedLate,
    });
    assert_eq!(interviewer.seen.len(), 2, "both voices were asked about");
    for seen in &interviewer.seen {
        assert_eq!(
            seen.meeting, expected,
            "the seam carries the label, per voice"
        );
    }

    // The absent half: the common case hands `None`, which is what lets a surface reserve
    // nothing for a title that is not there.
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![named("Alice"), named("Aaron")]);
    run(&paths, &[], &mut interviewer);

    assert_eq!(interviewer.seen.len(), 2, "both voices were asked about");
    for seen in &interviewer.seen {
        assert_eq!(
            seen.meeting, None,
            "no meeting means no label, not an empty one"
        );
    }
}
