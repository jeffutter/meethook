//! the narration a run reads as, and what the queue pane holds.

use super::*;

/// One run's narration, whole and in order, as [`Lines`] renders it.
///
/// Every other test here asserts a substring, which cannot see a line that moved, a blank
/// line that appeared, or a pair that swapped -- and the notes the run now emits are placed
/// by a renderer rather than by the statement that computed them, so line order is exactly
/// what wants pinning. The fixture is built to reach one of each tier in one run: a session
/// passed over before anything is read, a queue header with its held-back clause, a
/// transcript brought into line before a question is asked, an enrollment, a reference taken
/// off somebody else, and an answer refused.
///
/// `--correct` for the whole run, so the second session's already-named voice is asked
/// about; that is also why both headers read "to review" rather than "unresolved", which
/// the tests above cover.
#[test]
fn one_runs_narration_reads_as_these_lines_in_this_order() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());

    // The recorder died mid-session: nothing to read, and the first line of the run.
    let orphan = paths.session(&SessionId::parse("20260809-052500").unwrap());
    std::fs::create_dir_all(orphan.dir()).unwrap();

    // One voice worth a question and three fragments under the floor. Its voices sit on the
    // two axes Nate and the second session's voices do not, so enrolling here changes
    // nothing there and the two sessions' lines stay independent.
    let fragmented = make_fragmented_session(&paths, "20260809-052600");
    with_embeddings(&fragmented, &[voice(2), voice(3), voice(3), voice(3)]);

    let session = make_session(&paths, "20260809-052700");
    heard_at_once(&session, 0, 1);
    enrolled(&[("Nate", voice(0))], &paths);

    let mut interviewer = Scripted::answering(vec![named("Alice"), named("Aaron"), named("Aaron")]);
    let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

    assert_eq!(
        output,
        "20260809-052500  passed over: no session.json (the recorder crashed mid-session)\n\
         20260809-052600  1 voice(s) to review, 0 of them already named, 3 quieter voice(s) \
         not offered -- meethook enroll --all\n\
         20260809-052600  enrolled Alice\n\
         20260809-052700  transcript brought up to date\n\
         20260809-052700  2 voice(s) to review, 1 of them already named\n\
         20260809-052700  Nate no longer has a reference: that voice is Aaron\n\
         20260809-052700  enrolled Aaron\n\
         20260809-052700  refused Aaron for Unknown 2: Unknown 1 already has that name and \
         the two were heard speaking at once, so they are not one person -- meethook enroll \
         --correct --voice Unknown 1 if that is the wrong one\n"
    );
    assert_eq!(
        report,
        EnrollReport {
            named: 2,
            session_only: 0,
            skipped: 0,
            kept: 0,
            held_back: 3,
            refused: 1,
            passed_over: 1,
            failed: 0,
            asserted: 0,
            vetoes_overridden: 0,
        }
    );
}

/// TASK-046.06.01 acceptance criterion #1: a prompt is handed the whole session, not only
/// the voice it is about -- which is what a queue pane is drawn from.
#[test]
fn a_prompt_carries_every_voice_of_its_session() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::default();
    let (_, output) = run(&paths, &[], &mut interviewer);

    // Both voices, in first-appearance order -- the order the transcript reads in -- with
    // the basis and not only the label, and neither of them under the floor.
    assert_eq!(
        interviewer.seen[0].queue,
        vec![
            Row {
                number: "Unknown 1".to_string(),
                attribution: Attribution::Unknown("Unknown 1".to_string()),
                speech_seconds: 10.0,
                below_floor: false,
            },
            Row {
                number: "Unknown 2".to_string(),
                attribution: Attribution::Unknown("Unknown 2".to_string()),
                speech_seconds: 11.0,
                below_floor: false,
            },
        ],
        "{output}"
    );
}

/// TASK-046.06.01 acceptance criterion #1, the half a queue pane needs to explain itself:
/// the voices this run did *not* offer are in the queue, and say why.
#[test]
fn the_queue_holds_the_voices_the_floor_held_back_and_marks_them() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_fragmented_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::default();
    let (report, output) = run(&paths, &[], &mut interviewer);

    // One question, three voices held back -- and all four rows on the prompt.
    assert_eq!(interviewer.seen.len(), 1, "{output}");
    assert_eq!(report.held_back, 3, "{output}");
    assert_eq!(
        interviewer.seen[0].rows(),
        [
            ("Unknown 1", "Unknown 1", false),
            ("Unknown 2", "Unknown 2", true),
            ("Unknown 3", "Unknown 3", true),
            ("Unknown 4", "Unknown 4", true),
        ],
        "{output}"
    );
}

/// TASK-046.06.01 acceptance criterion #2: the queue is rebuilt per question, so it shows
/// what this run has already done rather than what the session looked like when it opened.
#[test]
fn the_queue_shows_a_voice_an_earlier_answer_named() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (_, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(
        interviewer.seen[0].queue[0].attribution,
        Attribution::Unknown("Unknown 1".to_string()),
        "{output}"
    );
    assert_eq!(
        interviewer.seen[1].queue[0].attribution,
        Attribution::Identified {
            name: "Alice".to_string(),
            similarity: 1.0
        },
        "the second question must see the first one's answer: {output}"
    );
    // And the handle did not move with the name -- acceptance criterion #3 in its in-run
    // form, which is the one a cursor depends on.
    assert_eq!(interviewer.seen[1].queue[0].number, "Unknown 1", "{output}");
}

/// TASK-046.06.01 acceptance criterion #3: the handle a state machine keys on is the
/// "Unknown N", and it stays put when the label does not.
#[test]
fn a_voice_carries_a_number_that_a_name_does_not_move() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    enrolled(&[("Alice", voice(0))], &paths);

    let mut interviewer = Scripted::default();
    let (_, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

    assert_eq!(interviewer.seen[0].label(), "Alice", "{output}");
    assert_eq!(
        interviewer.seen[0].number, "Unknown 1",
        "the label is the name and the number is the handle: {output}"
    );
}
