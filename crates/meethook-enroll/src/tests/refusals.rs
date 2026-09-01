//! the heard-at-once veto, theft by argmax, and their overrides.

use super::*;

/// The heard-at-once veto is the one way an answer can still cost an earlier name once
/// references accumulate instead of replacing: segmentation heard these two voices at
/// once, so they are not one person however certain the user is, and the veto has to refuse
/// one of the two answers.
///
/// What this ticket changes is that it is refused *out loud* and the earlier name is what
/// survives. Before, the veto could take the earlier answer instead, and said nothing.
#[test]
fn an_answer_the_heard_at_once_veto_would_take_from_an_earlier_voice_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    heard_at_once(&session, 0, 1);

    let mut interviewer = Scripted::answering(vec![named("Alice"), named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert_eq!(report.refused, 1, "{output}");
    assert!(
        output.contains(
            "refused Alice for Unknown 2: Unknown 1 already has that name and the two \
             were heard speaking at once"
        ),
        "a refusal the user cannot read is a silent revert: {output}"
    );
    let said = transcript_of(&session);
    assert_eq!(
        (
            said.turns[0].speaker.as_str(),
            said.turns[2].speaker.as_str()
        ),
        ("Alice", "Unknown 2"),
        "the first answer is what survives: {output}"
    );
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.references("Alice"), 1, "{:?}", speakers.speakers);
}

/// The heard-at-once veto is unchanged in effect by references accumulating: a person is one
/// contender for one name however many recordings back it, so two references of one person
/// can never be awarded to two voices that overlap in time.
///
/// Alice ends up holding two recordings, and the third session contains a voice matching
/// each of them *exactly* -- at distance 0, so nothing but the veto can separate them --
/// which segmentation heard talking over each other. One of the two gets the name.
#[test]
fn two_references_of_one_person_are_never_awarded_to_two_voices_heard_at_once() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let first = make_session(&paths, "20260809-052600");
    let second = make_session(&paths, "20260809-052700");
    let third = make_session(&paths, "20260809-052800");
    with_embeddings(&first, &[nearly(0.0), voice(3)]);
    with_embeddings(&second, &[nearly(60.0), voice(3)]);
    with_embeddings(&third, &[nearly(0.0), nearly(60.0)]);
    heard_at_once(&third, 0, 1);

    let mut interviewer = Scripted::answering(vec![
        named("Alice"),
        Answer::Skip,
        named("Alice"),
        Answer::Skip,
    ]);
    let (_, output) = run(&paths, &[], &mut interviewer);

    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.references("Alice"), 2, "{:?}", speakers.speakers);
    let said = transcript_of(&third);
    assert_eq!(
        (
            said.turns[0].speaker.as_str(),
            said.turns[2].speaker.as_str()
        ),
        ("Alice", "Unknown 2"),
        "one name cannot land on two voices heard at once, whatever backs it: {output}"
    );
}

/// Theft by argmax: a reference stored for one person can sit nearer to a third voice than
/// that voice's own name's reference does, moving a name the user never asked about. Bob is
/// 40 degrees from cluster 1 and holds it; Alice's new reference would be 20 degrees away
/// and would win it. Refused, and nothing is written at all.
#[test]
fn an_answer_that_would_move_another_voices_name_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
    enrolled(&[("Bob", nearly(60.0))], &paths);
    let before = std::fs::read(paths.speakers_json()).unwrap();

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.refused, 1, "{output}");
    assert_eq!(report.named, 0, "{output}");
    assert!(
        output.contains("refused Alice for Unknown 1: it would take Bob off Unknown 2"),
        "{output}"
    );
    assert_eq!(
        std::fs::read(paths.speakers_json()).unwrap(),
        before,
        "a refused answer writes nothing"
    );
    let said = transcript_of(&session);
    assert_eq!(
        (
            said.turns[0].speaker.as_str(),
            said.turns[2].speaker.as_str()
        ),
        ("Unknown 1", "Bob"),
        "{output}"
    );
}

/// The third path to the same loss, and the one neither TASK-027 nor its plan noticed: a
/// hand-given name beats an identification on a voice it overlaps, so naming a quiet
/// fragment can drop that name off the voice that had it -- without any reference being
/// stored or removed. Refused by the same check, which is why the check is at the label
/// level rather than inside identification.
#[test]
fn naming_a_quiet_fragment_is_refused_when_it_would_unname_the_voice_that_has_that_name() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_speech_seconds(&session, &[40.0, 1.5]);
    heard_at_once(&session, 0, 1);
    enrolled(&[("Alice", voice(0))], &paths);

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    assert_eq!(report.refused, 1, "{output}");
    assert_eq!(report.named, 0, "{output}");
    assert!(
        output.contains("refused Alice for Unknown 2: it would take Alice off Unknown 1"),
        "{output}"
    );
    assert!(
        assigned_in(&session, "20260809-052600").names.is_empty(),
        "a refused answer writes nothing"
    );
    assert_eq!(
        transcript_of(&session).turns[0].speaker,
        "Alice",
        "{output}"
    );
}

/// The other side of theft by argmax: the same answer, insisted on. An interface that showed
/// the user which voice pays and what it loses before a key was pressed has removed the
/// surprise the refusal exists to prevent, so the answer is honoured -- and everything a
/// name ordinarily writes is written, this session's transcript included.
#[test]
fn naming_a_voice_anyway_takes_the_name_off_the_voice_that_had_it() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
    enrolled(&[("Bob", nearly(60.0))], &paths);
    let before = std::fs::read(paths.speakers_json()).unwrap();

    let mut interviewer = Scripted::answering(vec![named_anyway("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.refused, 0, "{output}");
    assert_eq!(report.named, 1, "{output}");
    assert!(
        output.contains(
            "named Alice for Unknown 1 anyway: Unknown 2 no longer reads Bob -- \
             meethook enroll --correct --voice Unknown 2 to give it a name again"
        ),
        "the voice that paid has to be named where the run is read afterwards, not only in \
         the pane that warned about it: {output}"
    );
    assert_ne!(
        std::fs::read(paths.speakers_json()).unwrap(),
        before,
        "an honoured answer writes the name it was given"
    );
    let said = transcript_of(&session);
    assert_eq!(said.turns[0].speaker, "Alice", "{output}");
    assert_ne!(
        said.turns[2].speaker, "Bob",
        "the transcript has to agree with the cost that was accepted: {output}"
    );
}

/// The heard-at-once veto is not reachable from here however insistent the answer is.
/// Segmentation *heard* these two voices at once and so proved they are different people;
/// overriding that is the claim that several voices are one person, which is a different
/// question with a ticket of its own. Byte for byte the refusal an ordinary answer gets.
#[test]
fn the_heard_at_once_veto_is_refused_however_insistent_the_answer_is() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    heard_at_once(&session, 0, 1);

    let mut interviewer = Scripted::answering(vec![named_anyway("Alice"), named_anyway("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert_eq!(report.refused, 1, "{output}");
    assert!(
        output.contains(
            "refused Alice for Unknown 2: Unknown 1 already has that name and the two \
             were heard speaking at once"
        ),
        "insisting must not change the sentence, let alone the outcome: {output}"
    );
    let said = transcript_of(&session);
    assert_eq!(
        (
            said.turns[0].speaker.as_str(),
            said.turns[2].speaker.as_str()
        ),
        ("Alice", "Unknown 2"),
        "{output}"
    );
}

/// The override is at the label level, like the check it overrides: it does not depend on
/// which of the three mechanisms produced the loss. Here no reference is stored or removed
/// at all -- a hand-given name on a quiet fragment simply beats the identification on the
/// voice it overlaps -- and insisting takes Alice off the voice that had her all the same.
#[test]
fn the_quiet_fragment_path_can_be_overridden_too() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_speech_seconds(&session, &[40.0, 1.5]);
    heard_at_once(&session, 0, 1);
    enrolled(&[("Alice", voice(0))], &paths);

    let mut interviewer = Scripted::answering(vec![named_anyway("Alice")]);
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    assert_eq!(report.refused, 0, "{output}");
    assert_eq!(report.named, 1, "{output}");
    assert!(
        output.contains("named Alice for Unknown 2 anyway: Unknown 1 no longer reads Alice"),
        "{output}"
    );
    assert_eq!(
        assigned_in(&session, "20260809-052600").names.len(),
        1,
        "an honoured answer records the name against the session: {output}"
    );
    assert_ne!(
        transcript_of(&session).turns[0].speaker,
        "Alice",
        "the voice that lost the name keeps it in the transcript otherwise: {output}"
    );
}

/// A name supplied up front never overrides anything. `--name` is never shown the voice it
/// lands on -- which is why it needs a selector at all -- so it has certainly not been shown
/// the third voice an override would cost, and the premise the override rests on does not
/// hold for it.
#[test]
fn a_name_given_up_front_cannot_override_a_refusal() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
    enrolled(&[("Bob", nearly(60.0))], &paths);
    let before = std::fs::read(paths.speakers_json()).unwrap();

    let selector = VoiceSelector::from("Unknown 1");
    let (report, output) = run_over(
        &paths,
        // `--voice` needs the one session it is about, exactly as the CLI insists.
        &["20260809-052600"],
        Some(Selection::Voice(selector)),
        Offer::default(),
        Sessions::default(),
        Enrolment::default(),
        &mut GivenName::new("Alice"),
    );

    assert_eq!(report.refused, 1, "{output}");
    assert_eq!(report.named, 0, "{output}");
    assert!(
        output.contains("refused Alice for Unknown 1: it would take Bob off Unknown 2"),
        "{output}"
    );
    assert_eq!(
        std::fs::read(paths.speakers_json()).unwrap(),
        before,
        "a refused answer writes nothing"
    );
    assert_eq!(transcript_of(&session).turns[2].speaker, "Bob", "{output}");
}
