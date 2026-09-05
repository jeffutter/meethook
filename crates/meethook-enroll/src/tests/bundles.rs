//! Fragment bundling: below-floor fragments that cluster close enough to ask about as one
//! question instead of one per fragment. The fixture is the shape real clustering leaves a
//! meeting in -- a loud voice and a quiet tail split across three near-identical clusters --
//! and every test here runs it through [`Interviewer::ask`](super::super::Interviewer::ask)
//! with the same seams the other modules use.

use super::*;

/// The bundling fixture: [`make_fragmented_session`]'s one loud voice plus three sub-floor
/// fragments within degrees of each other -- close enough to fold into a bundle, far under
/// the merge limit, and orthogonal to the loud voice so nothing collapses into it.
fn bundled_session(paths: &Paths, id: &str) -> SessionPaths {
    let session = make_fragmented_session(paths, id);
    // Clusters 1--3 are the same voice the diarisation cut up -- 2 and 4 degrees apart, a
    // tight bundle -- while the loud cluster 0 sits a quarter turn away, so naming it cannot
    // settle the quiet tail. `nearly` measures off the x axis, which is where `voice(0)`
    // points, so the tail must live near the y axis instead: 90, 92 and 94 degrees.
    with_embeddings(
        &session,
        &[voice(0), nearly(90.0), nearly(92.0), nearly(94.0)],
    );
    session
}

/// One answer names every member of the bundle: two questions for four voices, the fan-out
/// commits all three members, and the sub-floor name stores no references.
#[test]
fn one_answer_names_every_member_of_a_bundle() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = bundled_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![
        named("Ivan"),
        Answer::FragmentGroup {
            name: "Pete".to_string(),
            members: vec!["Unknown 2".into(), "Unknown 3".into(), "Unknown 4".into()],
        },
    ])
    .accepting_fragment_groups();
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    // Two questions, not four: the solo voice and the bundle.
    assert_eq!(interviewer.seen.len(), 2, "{output}");
    assert_eq!(interviewer.positions(), ["1/2", "2/2"], "{output}");

    // The picture crosses the seam on both questions -- the pane's rows need it even when
    // the question itself is not the bundle -- while only the bundle carries its own key.
    let group = &interviewer.seen[0].fragment_groups[0];
    assert_eq!(group.members, ["Unknown 2", "Unknown 3", "Unknown 4"]);
    assert_eq!(group.speech_seconds, 1.5 + 0.9 + 2.0);
    assert!(group.best.is_none(), "nothing is enrolled yet");
    assert_eq!(interviewer.seen[0].bundle_members, None);
    let bundle: Vec<&str> = interviewer.seen[1]
        .bundle_members
        .as_deref()
        .expect("the bundle question carries its own key")
        .iter()
        .map(String::as_str)
        .collect();
    assert_eq!(bundle, ["Unknown 2", "Unknown 3", "Unknown 4"]);

    // The anchor keeps the question's number and stacks its members' lines behind it.
    assert_eq!(interviewer.seen[1].number, "Unknown 2");
    assert_eq!(interviewer.seen[1].snippets, ["and from me", "mm", "yes"]);

    // `session_only` is a sub-count of `named`: all four voices were named, and the three
    // bundle members landed in the session's rows rather than `speakers.json` because they sit
    // under the reference floor.
    assert_eq!(report.named, 4, "{output}");
    assert_eq!(report.session_only, 3, "{output}");
    assert_eq!(report.skipped, 0, "{output}");

    // The transcript reads back the fan-out: one name on every fragment.
    let transcript = transcript_of(&session);
    let labels: Vec<&str> = said(&transcript)
        .iter()
        .map(|(speaker, _, _)| *speaker)
        .collect();
    assert_eq!(labels, ["Ivan", SPEAKER_YOU, "Pete", "Pete", "Pete"]);

    // And the database holds Ivan alone: a sub-floor answer never poisons the reference
    // store, however many fragments the single answer committed.
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    let names: Vec<&str> = speakers.speakers.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["Ivan"]);
}

/// The position counts questions while the report counts clusters: deferring everything
/// asks twice and skips four.
#[test]
fn positions_count_questions_but_the_report_counts_clusters() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let _session = bundled_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::default().accepting_fragment_groups();
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    assert_eq!(interviewer.positions(), ["1/2", "2/2"], "{output}");
    assert_eq!(report.skipped, 4, "{output}");
    assert_eq!(report.named, 0, "{output}");
}

/// A deferred bundle comes round again as the same question: the second pass re-asks it
/// under its own number, and answering it there still fans out over all three members.
#[test]
fn a_deferred_bundle_comes_round_as_the_same_question() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = bundled_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![
        Answer::Skip,
        Answer::Later,
        Answer::FragmentGroup {
            name: "Pete".to_string(),
            members: vec!["Unknown 2".into(), "Unknown 3".into(), "Unknown 4".into()],
        },
    ])
    .accepting_fragment_groups();
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    // The bundle was asked first, deferred to the end, and came back under its own number.
    assert_eq!(interviewer.positions(), ["1/2", "2/2", "2/2"], "{output}");
    assert_eq!(report.session_only, 3, "{output}");
    let transcript = transcript_of(&session);
    let labels: Vec<&str> = said(&transcript)
        .iter()
        .map(|(speaker, _, _)| *speaker)
        .collect();
    // Cluster 0 was skipped, so it keeps its placeholder label beside the fan-out.
    assert_eq!(labels, ["Unknown 1", SPEAKER_YOU, "Pete", "Pete", "Pete"]);
}

/// A vetoed member stays out of the fan-out while the rest commit: the bundle carries no
/// override authority, and the refusal shows in the report and the transcript.
#[test]
fn a_vetoed_member_of_a_bundle_stays_unnamed_while_the_rest_commit() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = bundled_session(&paths, "20260809-052600");
    // Cluster 2 was heard at once with the loud one: naming it what the loud one reads as
    // would unname the loud one, so the fan-out must refuse it and commit the others.
    heard_at_once(&session, 0, 2);

    let mut interviewer = Scripted::answering(vec![
        named("Ivan"),
        Answer::FragmentGroup {
            name: "Ivan".to_string(),
            members: vec!["Unknown 2".into(), "Unknown 3".into(), "Unknown 4".into()],
        },
    ])
    .accepting_fragment_groups();
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    // The vetoed member is refused, not named: two of the three bundle members commit
    // session-only beside Ivan's reference, and the third stays out of the fan-out.
    assert_eq!(report.named, 3, "{output}");
    assert_eq!(report.session_only, 2, "{output}");
    assert_eq!(report.refused, 1, "{output}");
    let transcript = transcript_of(&session);
    let labels: Vec<&str> = said(&transcript)
        .iter()
        .map(|(speaker, _, _)| *speaker)
        .collect();
    assert_eq!(labels, ["Ivan", SPEAKER_YOU, "Ivan", "Unknown 3", "Ivan"]);
}

/// Forced-reference enrolment projects no bundles: the bundling is a property of the default
/// run's quiet tail, and a `--force-reference` run must carry none of those keys across the
/// seam, whatever questions it ends up asking.
///
/// It stops after two questions rather than four, and that is why the embeddings stay tight
/// here: naming the first fragment stores a reference, and the rest of the tail identifies
/// against it before their own turn comes up. Under default enrolment the same tail never
/// identifies -- a sub-floor answer stores no reference -- which is what lets the bundle form
/// at all. The invariant under test is the absence of the bundle keys, pinned directly rather
/// than inferred from a count.
#[test]
fn forced_reference_enrolment_projects_no_bundles() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let _session = bundled_session(&paths, "20260809-052600");

    let mut interviewer =
        Scripted::answering(vec![named("Ivan"), named("Ada")]).accepting_fragment_groups();
    let (report, output) = run_enrolling(&paths, &[], ALL, Enrolment::Always, &mut interviewer);

    for seen in &interviewer.seen {
        assert!(seen.fragment_groups.is_empty(), "{output}");
        assert_eq!(seen.bundle_members, None, "{output}");
    }
    // Two references stored, nothing session-only: a forced-reference run writes every name
    // into `speakers.json`, which is exactly what lets the tail identify against itself.
    assert_eq!(report.named, 2, "{output}");
    assert_eq!(report.session_only, 0, "{output}");
}

/// An answerer that does not accept bundles gets one question per fragment: the default
/// behaviour the headless callers rely on, pinned against drift.
#[test]
fn an_answerer_that_does_not_accept_bundles_gets_one_question_per_fragment() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let _session = bundled_session(&paths, "20260809-052600");

    // No `accepting_fragment_groups`: the flag is off and the run must not project any.
    let mut interviewer = Scripted::answering(vec![
        named("Ivan"),
        named("Ada"),
        named("Ben"),
        named("Cyra"),
    ]);
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    assert_eq!(
        interviewer.positions(),
        ["1/4", "2/4", "3/4", "4/4"],
        "{output}"
    );
    for seen in &interviewer.seen {
        assert!(seen.fragment_groups.is_empty(), "{output}");
        assert_eq!(seen.bundle_members, None, "{output}");
    }
    // `session_only` is a sub-count of `named`: all four were named, three of them written to
    // the session's rows alone because they sit under the reference floor.
    assert_eq!(report.named, 4, "{output}");
    assert_eq!(report.session_only, 3, "{output}");
}
