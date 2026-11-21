//! reopening a voice answered earlier in the run, to change its name.

use super::*;

fn stored(paths: &Paths) -> Vec<(String, Vec<f32>)> {
    EnrolledSpeakers::read_or_empty(paths)
        .unwrap()
        .speakers
        .iter()
        .map(|s| (s.name.clone(), s.embedding.clone()))
        .collect()
}

/// Renaming a voice answered earlier replaces what the first name wrote: the database holds
/// the new name's reference and none of the old, the transcript reads the new name, and the
/// voice is counted once.
#[test]
fn reopening_a_named_voice_replaces_its_name_everywhere() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![
        named("Bob"),
        Answer::Later,
        named("Alice"),
        named("Carol"),
    ])
    .steering(&[Some("Unknown 1")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(
        interviewer.labels(),
        ["Unknown 1", "Unknown 2", "Bob", "Unknown 2"],
        "the reopened voice is asked first, under the name it has now: {output}"
    );
    assert_eq!(report.named, 2, "{output}");
    assert_eq!(report.skipped + report.kept + report.refused, 0, "{output}");
    assert_eq!(
        stored(&paths),
        [
            ("Alice".to_string(), voice(0)),
            ("Carol".to_string(), voice(1))
        ],
        "no reference left under Bob: {output}"
    );
    let said = transcript_of(&session);
    assert_eq!(
        (
            said.turns[0].speaker.as_str(),
            said.turns[2].speaker.as_str(),
            said.turns[3].speaker.as_str()
        ),
        ("Alice", "Carol", "Alice"),
        "{output}"
    );
    assert!(assigned_in(&session, "20260809-052600").names.is_empty());
}

/// A voice too quiet for a reference is named for the session alone; reopening it overwrites
/// that row rather than appending, and the session-only count follows.
#[test]
fn reopening_a_session_only_voice_overwrites_its_row() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_speech_seconds(&session, &[1.5, 1.6]);

    let mut interviewer = Scripted::answering(vec![
        named("Bob"),
        Answer::Later,
        named("Alice"),
        named("Carol"),
    ])
    .steering(&[Some("Unknown 1")]);
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    assert_eq!(report.named, 2, "{output}");
    assert_eq!(report.session_only, 2, "{output}");
    assert_eq!(
        assigned_in(&session, "20260809-052600")
            .names
            .iter()
            .map(|row| (row.cluster, row.name.as_str()))
            .collect::<Vec<_>>(),
        [(0, "Alice"), (1, "Carol")],
        "{output}"
    );
    assert!(stored(&paths).is_empty(), "{output}");
}

/// Reopening and answering with the name it already has changes nothing: same files, same
/// report as a run that never reopened.
#[test]
fn reopening_with_the_same_name_changes_nothing() {
    let plain_root = tempfile::tempdir().unwrap();
    let plain_paths = Paths::new(plain_root.path());
    let plain_session = make_session(&plain_paths, "20260809-052600");
    let mut plain = Scripted::answering(vec![named("Bob"), Answer::Skip]);
    let (plain_report, _) = run(&plain_paths, &[], &mut plain);

    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    let mut interviewer = Scripted::answering(vec![
        named("Bob"),
        Answer::Later,
        named("Bob"),
        Answer::Skip,
    ])
    .steering(&[Some("Unknown 1")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(
        interviewer.labels(),
        ["Unknown 1", "Unknown 2", "Bob", "Unknown 2"]
    );
    assert_eq!(
        (report.named, report.skipped, report.kept, report.refused),
        (
            plain_report.named,
            plain_report.skipped,
            plain_report.kept,
            plain_report.refused
        ),
        "{output}"
    );
    assert_eq!(stored(&paths), stored(&plain_paths), "{output}");
    assert_eq!(
        transcript_of(&session).turns,
        transcript_of(&plain_session).turns,
        "{output}"
    );
}

/// A rename the veto refuses leaves the earlier answer, on disk and in the report, exactly
/// as it was.
#[test]
fn a_refused_reopen_keeps_the_earlier_answer() {
    let baseline_root = tempfile::tempdir().unwrap();
    let baseline_paths = Paths::new(baseline_root.path());
    let baseline_session = make_session(&baseline_paths, "20260809-052600");
    heard_at_once(&baseline_session, 0, 1);
    let mut baseline = Scripted::answering(vec![named("Alice"), named("Bob")]);
    let (baseline_report, _) = run(&baseline_paths, &[], &mut baseline);

    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    heard_at_once(&session, 0, 1);
    let mut interviewer = Scripted::answering(vec![named("Alice"), named("Bob"), named("Bob")])
        .steering(&[Some("Unknown 1")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(
        interviewer.labels(),
        ["Unknown 1", "Unknown 2", "Alice"],
        "{output}"
    );
    assert_eq!(report.refused, 1, "{output}");
    assert_eq!(report.named, baseline_report.named, "{output}");
    assert_eq!(stored(&paths), stored(&baseline_paths), "{output}");
    assert_eq!(
        transcript_of(&session).turns,
        transcript_of(&baseline_session).turns,
        "{output}"
    );
}

/// Skipping the reopened voice keeps its answer and counts nothing new.
#[test]
fn skipping_a_reopened_voice_keeps_it_and_counts_nothing() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![
        named("Bob"),
        Answer::Later,
        Answer::Skip,
        named("Carol"),
    ])
    .steering(&[Some("Unknown 1")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(
        (report.named, report.skipped, report.kept),
        (2, 0, 0),
        "{output}"
    );
    assert_eq!(transcript_of(&session).turns[0].speaker, "Bob", "{output}");
    assert_eq!(
        stored(&paths),
        [
            ("Bob".to_string(), voice(0)),
            ("Carol".to_string(), voice(1))
        ],
        "{output}"
    );
}

/// A blank name on the reopened voice is not an answer and does not disturb the old one.
#[test]
fn a_blank_name_on_a_reopened_voice_counts_nothing() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![
        named("Bob"),
        Answer::Later,
        named("  "),
        named("Carol"),
    ])
    .steering(&[Some("Unknown 1")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(
        (report.named, report.skipped, report.kept),
        (2, 0, 0),
        "{output}"
    );
}

/// A voice answered as part of a group is not offered again: re-deciding one member would
/// leave the rest of the group's claim and the counters inconsistent.
#[test]
fn a_group_member_is_not_reopened() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![Answer::Group {
        name: "Bob".to_string(),
        members: vec!["Unknown 1".to_string(), "Unknown 2".to_string()],
    }])
    .steering(&[Some("Unknown 2")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(interviewer.labels(), ["Unknown 1"], "{output}");
    assert_eq!(report.named, 2, "{output}");
}

/// A reopened voice marked into a group with another voice is named by the group: the
/// earlier name's reference is replaced, the transcript reads the group's name, and the voice
/// is counted once.
#[test]
fn a_reopened_voice_answered_as_a_group_is_renamed_and_counted_once() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![
        named("Bob"),
        Answer::Later,
        Answer::Group {
            name: "Carol".to_string(),
            members: vec!["Unknown 1".to_string(), "Unknown 2".to_string()],
        },
    ])
    .steering(&[Some("Unknown 1")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 2, "{output}");
    assert_eq!(report.skipped + report.kept + report.refused, 0, "{output}");
    assert!(
        stored(&paths).iter().all(|(name, _)| name == "Carol"),
        "no reference left under Bob: {output}"
    );
    let said = transcript_of(&session);
    assert_eq!(
        (
            said.turns[0].speaker.as_str(),
            said.turns[2].speaker.as_str(),
            said.turns[3].speaker.as_str()
        ),
        ("Carol", "Carol", "Carol"),
        "{output}"
    );
}

/// After a group answer took over a reopened voice, that voice no longer counts as a solo
/// answer, so a later skip on another voice is counted the way it would be in a run that
/// never reopened anything.
#[test]
fn a_group_answer_over_a_reopened_voice_leaves_later_counts_right() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    let mut interviewer = Scripted::answering(vec![
        named("Bob"),
        Answer::Later,
        Answer::Group {
            name: "Carol".to_string(),
            members: vec!["Unknown 1".to_string()],
        },
        Answer::Skip,
    ])
    .steering(&[Some("Unknown 1")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert_eq!(report.skipped + report.kept, 1, "{output}");
    assert!(
        stored(&paths).iter().all(|(name, _)| name == "Carol"),
        "{output}"
    );
}

/// A target nobody settled is ignored, so the answerer's own bound on steering is what ends
/// it and the loop cannot spin.
#[test]
fn a_target_that_was_never_answered_is_ignored() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![named("Bob"), Answer::Skip]).steering(&[
        Some("Unknown 9"),
        Some("Unknown 9"),
        Some("Unknown 9"),
    ]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
    assert_eq!((report.named, report.skipped), (1, 1), "{output}");
}
