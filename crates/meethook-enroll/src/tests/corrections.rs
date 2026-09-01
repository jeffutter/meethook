//! --correct: renames, re-rendering, and reference repair.

use super::*;

/// The transcript body a re-render left on disk, below its frontmatter.
///
/// Compared rather than the whole file because `updated` is the render instant, and two
/// renderings a few microseconds apart can straddle a second boundary. The body is the
/// half TASK-038 is about.
fn markdown_body(session: &SessionPaths) -> String {
    let markdown = std::fs::read_to_string(session.transcript_md()).unwrap();
    markdown.split_once("\n---\n").unwrap().1.to_string()
}

/// TASK-038 acceptance criterion #6, the half where the rename does *not* merge anything:
/// naming the voice between two runs of another leaves the blocks where they were, and the
/// re-rendered file is what a fresh rendering of the relabelled turns produces.
#[test]
fn a_rename_that_merges_nothing_re_renders_the_same_blocks() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    let (report, output) = run_naming_at(&paths, &["20260809-052600"], "00:03", "Alice");
    assert_eq!(report.named, 1, "{output}");

    let transcript = transcript_of(&session);
    assert_eq!(
        said(&transcript)
            .iter()
            .map(|(speaker, _, _)| *speaker)
            .collect::<Vec<_>>(),
        ["Unknown 1", "You", "Alice", "Unknown 1"]
    );
    let body = markdown_body(&session);
    assert_eq!(
        body,
        transcript
            .render_markdown(
                &TranscriptTemplate::resolve(&paths, None).unwrap(),
                &TranscriptContext::now(&session_metadata(
                    &SessionId::parse("20260809-052600").unwrap()
                )),
            )
            .unwrap()
            .split_once("\n---\n")
            .unwrap()
            .1
    );
    // Four speakers in a row, none of them repeating: four lines, as before collapsing.
    assert_eq!(body.trim_start().lines().count(), 4, "{body}");
    assert!(body.contains("**[00:03] Alice:** and from me\n"), "{body}");
}

/// TASK-038 acceptance criterion #6, the half that only collapsing can get wrong: naming a
/// voice clustering had split in two puts one name on both halves, and where those halves
/// are adjacent the re-render must print them as one block rather than the same name twice
/// in a row -- which is what a fresh `transcribe` of the relabelled turns now produces.
#[test]
fn a_rename_that_makes_two_blocks_adjacent_merges_them_on_re_render() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    // One voice the clusterer did not join up, so naming it once names both halves.
    with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);
    assert_eq!(report.named, 1, "{output}");

    let transcript = transcript_of(&session);
    assert_eq!(
        said(&transcript)
            .iter()
            .map(|(speaker, _, _)| *speaker)
            .collect::<Vec<_>>(),
        ["Alice", "You", "Alice", "Alice"]
    );
    let body = markdown_body(&session);
    assert_eq!(
        body,
        transcript
            .render_markdown(
                &TranscriptTemplate::resolve(&paths, None).unwrap(),
                &TranscriptContext::now(&session_metadata(
                    &SessionId::parse("20260809-052600").unwrap()
                )),
            )
            .unwrap()
            .split_once("\n---\n")
            .unwrap()
            .1
    );
    // The last two turns were two blocks under two names before the rename and are one
    // block under one timestamp after it.
    assert_eq!(body.trim_start().lines().count(), 3, "{body}");
    assert!(
        body.contains("**[00:03] Alice:** and from me let us start\n"),
        "{body}"
    );
    assert!(!body.contains("Unknown"), "{body}");
}

/// TASK-019.03 acceptance criteria #1 and #2, which is the whole ticket in one test: a
/// voice the database has named the wrong person is reached, corrected, and lands in both
/// files -- and a later default run does not ask about it again.
#[test]
fn correcting_a_named_voice_updates_the_database_and_this_transcript() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    // Cluster 0 is enrolled under the wrong name.
    let mut first = Scripted::answering(vec![named("Alice"), named("Carol")]);
    run(&paths, &[], &mut first);

    let mut interviewer = Scripted::answering(vec![named("Bob")]);
    let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

    // The question that was asked: a name, and how confident the claim behind it was.
    assert_eq!(interviewer.labels(), ["Alice", "Carol"], "{output}");
    assert_eq!(interviewer.seen[0].confidence(), Some(1.0), "{output}");
    assert_eq!(report.named, 1, "{output}");
    assert_eq!(report.kept, 1, "{output}");

    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    let stored: Vec<(&str, &[f32])> = speakers
        .speakers
        .iter()
        .map(|s| (s.name.as_str(), s.embedding.as_slice()))
        .collect();
    assert_eq!(
        stored,
        [("Carol", voice(1).as_slice()), ("Bob", voice(0).as_slice())],
        "the corrected name owns this voice, and the wrong one no longer claims it"
    );
    assert_eq!(
        said(&transcript_of(&session))
            .iter()
            .map(|(speaker, _, _)| *speaker)
            .collect::<Vec<_>>(),
        ["Bob", "You", "Carol", "Bob"]
    );

    // ...and the correction sticks: a later default run has nothing to ask about.
    let mut again = Scripted::default();
    let (report, output) = run(&paths, &[], &mut again);
    assert!(again.seen.is_empty(), "{:?}", again.seen);
    assert_eq!(report.passed_over, 1, "{output}");
}

/// Acceptance criterion #3: reaching an already-named voice takes an explicit request. A
/// default run over a half-identified session offers only the half nothing matched.
#[test]
fn a_default_run_still_asks_only_about_unresolved_voices() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    enrolled(&[("Alice", voice(0))], &paths);

    let mut default = Scripted::default();
    let (_, output) = run(&paths, &[], &mut default);
    assert_eq!(default.labels(), ["Unknown 2"], "{output}");
    assert!(output.contains("1 unresolved voice(s)"), "{output}");

    let mut correcting = Scripted::default();
    let (report, output) = run_asking(&paths, &[], CORRECT, &mut correcting);
    assert_eq!(correcting.labels(), ["Alice", "Unknown 2"], "{output}");
    assert_eq!(correcting.seen[0].confidence(), Some(1.0), "{output}");
    assert_eq!(correcting.seen[1].confidence(), None, "{output}");
    assert!(
        output.contains("2 voice(s) to review, 1 of them already named"),
        "{output}"
    );
    assert_eq!(report.kept, 1, "{output}");
    assert_eq!(report.skipped, 1, "{output}");
}

/// Acceptance criterion #4's other half: pressing Enter on an already-named voice keeps
/// that identification. The same nothing a skip writes -- byte for byte -- and counted
/// apart from it, because a kept voice has a name and a skipped one does not.
#[test]
fn keeping_an_identification_writes_nothing_and_is_not_counted_as_a_skip() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    enrolled(&[("Alice", voice(0)), ("Bob", voice(1))], &paths);

    // A default run first, so the snapshot below is of a transcript already in step with
    // the database and any difference is the correcting run's doing.
    run(&paths, &[], &mut Scripted::default());
    let before = (
        std::fs::read(session.transcript_json()).unwrap(),
        std::fs::read(session.transcript_md()).unwrap(),
        std::fs::read(session.speaker_clusters_json()).unwrap(),
        std::fs::read(paths.speakers_json()).unwrap(),
    );

    // Enter, then Enter with a stray space in the buffer.
    let mut interviewer = Scripted::answering(vec![Answer::Skip, named("   ")]);
    let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

    assert_eq!(interviewer.labels(), ["Alice", "Bob"], "{output}");
    assert_eq!(report.kept, 2, "{output}");
    assert_eq!(report.skipped, 0, "{output}");
    assert_eq!(report.named, 0, "{output}");
    assert_eq!(
        (
            std::fs::read(session.transcript_json()).unwrap(),
            std::fs::read(session.transcript_md()).unwrap(),
            std::fs::read(session.speaker_clusters_json()).unwrap(),
            std::fs::read(paths.speakers_json()).unwrap(),
        ),
        before
    );
}

/// Acceptance criterion #5 under `--correct`, which is where it could regress: the in-run
/// guard no longer looks at "is this named" alone, so the split-voice case has to be
/// checked with the flag on as well as off.
#[test]
fn correcting_still_asks_once_about_one_voice_clustering_split_in_two() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

    assert_eq!(
        interviewer.labels(),
        ["Unknown 1"],
        "the second half of one voice must not be asked about: {output}"
    );
    assert_eq!(report.named, 1, "{output}");
}

/// Replaces `a_voice_an_answer_unnamed_is_still_asked_about`, which encoded the un-naming as
/// *intended*: re-affirming cluster 0 re-anchored Alice's only reference onto it, cluster 1
/// fell out of range, and the old test asserted it was re-prompted about as a question the
/// run had created. Under a reference set the situation stops existing, and this one fixture
/// flip -- the same clusters at 0 and 80 degrees, the same Alice at 40 -- is the clearest
/// statement of what this ticket does.
///
/// Adding a reference removes none, so Alice's 40-degree reference is still there and still
/// names cluster 1. Nothing is un-named, so there is no second question.
#[test]
fn an_answer_no_longer_takes_the_name_off_the_other_half_of_a_voice() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    // 80 degrees apart, with Alice's reference sitting between them: inside
    // `IDENTIFY_DISTANCE` of both, and it stays that way now that answering cluster 0
    // appends a reference instead of moving the one that named cluster 1.
    with_embeddings(&session, &[nearly(0.0), nearly(80.0)]);
    enrolled(&[("Alice", nearly(40.0))], &paths);

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

    // `--correct` offers named voices, so cluster 1 is still asked about -- but it is asked
    // about as *Alice*, with the confidence behind that, rather than as the "Unknown 2" the
    // answer used to have turned it into. That is the whole difference.
    assert_eq!(interviewer.labels(), ["Alice", "Alice"], "{output}");
    assert!(interviewer.seen[1].confidence().is_some(), "{output}");
    assert_eq!(
        report.skipped, 0,
        "no voice was left unnamed, so nothing was skipped: {output}"
    );
    assert_eq!(report.kept, 1, "{output}");
    assert_eq!(report.refused, 0, "{output}");
    // Both halves still read Alice in the transcript, and the database holds both
    // recordings of her rather than only the newer one.
    let said = transcript_of(&session);
    assert_eq!(said.turns[0].speaker, "Alice", "{output}");
    assert_eq!(said.turns[2].speaker, "Alice", "{output}");
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.references("Alice"), 2, "{:?}", speakers.speakers);
}

/// One voice cannot be two people's stored reference. Correcting a voice enrolled under
/// the wrong name leaves that name holding a reference built from somebody else's audio,
/// which then competes as an exact tie in every future meeting -- and wins whenever it
/// sorts first. Both orderings are checked, so the fix cannot be about the alphabet.
#[test]
fn correcting_a_voice_removes_the_reference_the_wrong_name_kept_of_it() {
    for correction in ["Ryan", "Aaron"] {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600");
        enrolled(&[("Nate", voice(0))], &paths);

        let mut interviewer = Scripted::answering(vec![named(correction)]);
        let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

        assert_eq!(report.named, 1, "{output}");
        assert!(
            output.contains(&format!(
                "Nate no longer has a reference: that voice is {correction}"
            )),
            "an enrollment must not vanish without a line about it: {output}"
        );
        let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
        let stored: Vec<&str> = speakers.speakers.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(stored, [correction], "{output}");
        assert_eq!(
            transcript_of(&session).turns[0].speaker,
            correction,
            "{output}"
        );
    }
}

/// A reference built from a *different* recording of the same person is a legitimate one
/// and is left alone: only a reference identical to this cluster is a claim about a voice
/// the user has just said is somebody else.
#[test]
fn correcting_a_voice_leaves_the_wrong_names_other_reference_alone() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    // Nate, enrolled from audio that is not either cluster here, matched to cluster 0 by
    // being merely close to it -- which is the false accept this ticket opens with.
    with_embeddings(
        &paths.session(&SessionId::parse("20260809-052600").unwrap()),
        &[nearly(0.0), nearly(80.0)],
    );
    enrolled(&[("Nate", nearly(20.0))], &paths);

    let mut interviewer = Scripted::answering(vec![named("Ryan")]);
    let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert!(!output.contains("no longer has a reference"), "{output}");
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    let stored: Vec<(&str, &[f32])> = speakers
        .speakers
        .iter()
        .map(|s| (s.name.as_str(), s.embedding.as_slice()))
        .collect();
    assert_eq!(
        stored,
        [
            ("Nate", nearly(20.0).as_slice()),
            ("Ryan", nearly(0.0).as_slice())
        ],
        "Nate's own enrollment must survive somebody else's correction"
    );
}

/// The correction guarantee under a reference set, which is where it could have quietly
/// stopped working: the wrong name loses the reference built from *this* voice and keeps the
/// ones built from its own recordings, and the line says how many it has left rather than
/// claiming it has none.
#[test]
fn correcting_one_of_several_references_leaves_the_others_and_says_how_many_remain() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    // Nate, with two recordings: one of them *is* cluster 0, the other is somebody's real
    // second meeting with him.
    enrolled(&[("Nate", voice(0)), ("Nate", voice(3))], &paths);

    let mut interviewer = Scripted::answering(vec![named("Ryan")]);
    let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert!(
        output.contains(
            "Nate no longer has that reference: that voice is Ryan -- Nate keeps 1 other(s)"
        ),
        "a person who lost one of three recordings has not lost their enrollment: {output}"
    );
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    let stored: Vec<(&str, &[f32])> = speakers
        .speakers
        .iter()
        .map(|s| (s.name.as_str(), s.embedding.as_slice()))
        .collect();
    assert_eq!(
        stored,
        [("Nate", voice(3).as_slice()), ("Ryan", voice(0).as_slice())],
        "only the reference built from the corrected voice goes"
    );
    assert_eq!(transcript_of(&session).turns[0].speaker, "Ryan", "{output}");
}

/// The defect TASK-027 was raised for, stated as the smallest case that showed it: two
/// voices in one session given the same name. Under the old replacement rule the second
/// answer overwrote the reference that had named the first, so cluster 0 dropped back to
/// "Unknown 1" and its transcript was rewritten to say so -- silently, because the in-run
/// guard then declined to ask about the voice it had just un-named.
#[test]
fn two_voices_in_one_session_given_one_name_both_keep_it() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![named("Alice"), named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 2, "{output}");
    assert_eq!(report.refused, 0, "{output}");
    let said = transcript_of(&session);
    assert_eq!(
        (
            said.turns[0].speaker.as_str(),
            said.turns[2].speaker.as_str()
        ),
        ("Alice", "Alice"),
        "neither answer may cost the other: {output}"
    );
    // The rendering the user actually reads, checked separately: the defect's visible
    // symptom was an "Unknown N" line in transcript.md about somebody already named, and
    // this session has only these two voices, so neither file may mention a stranger.
    let markdown = std::fs::read_to_string(session.transcript_md()).unwrap();
    assert!(
        !markdown.contains("Unknown"),
        "transcript.md still calls a named voice a stranger: {markdown}"
    );
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.references("Alice"), 2, "{:?}", speakers.speakers);
}

/// The gain the reference set was measured for, and the reason it is worth a schema bump:
/// one person named in two meetings is recognised in a third that neither recording alone
/// would have reached.
///
/// Discriminating by construction. The third voice sits 10 degrees off the first recording
/// and 50 off the second; `IDENTIFY_DISTANCE` is 0.35, and 50 degrees is 0.357 -- outside
/// it. So under the old rule, where the second answer replaced the first, this voice would
/// read "Unknown 1" and be asked about instead.
#[test]
fn a_person_named_in_two_sessions_is_recognised_in_a_third() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let first = make_session(&paths, "20260809-052600");
    let second = make_session(&paths, "20260809-052700");
    let third = make_session(&paths, "20260809-052800");
    // Each session's second voice is orthogonal to every reference, so nothing but Alice is
    // ever in play. The two recordings of Alice are 60 degrees apart -- far enough that the
    // second session asks about her rather than matching her to the first.
    with_embeddings(&first, &[nearly(0.0), voice(3)]);
    with_embeddings(&second, &[nearly(60.0), voice(3)]);
    with_embeddings(&third, &[nearly(10.0), voice(3)]);

    let mut interviewer = Scripted::answering(vec![
        named("Alice"),
        Answer::Skip,
        named("Alice"),
        Answer::Skip,
    ]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 2, "{output}");
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.references("Alice"), 2, "{:?}", speakers.speakers);
    assert_eq!(
        transcript_of(&third).turns[0].speaker,
        "Alice",
        "the third session's voice is only within reach of the first recording: {output}"
    );
    assert!(
        !interviewer
            .seen
            .iter()
            .any(|v| v.session == "20260809-052800" && v.label() == "Unknown 1"),
        "a voice the database can already name must not be asked about: {:?}",
        interviewer.seen
    );
}

/// The walkthrough TASK-027 closes on, and the sharpest statement of the defect: name a
/// voice, name the same person in another session, then go back and run `enroll` over the
/// first session again. Its voice still reads her name, and its transcript is byte-identical
/// -- not merely equivalent, because the bug's visible symptom was a transcript rewritten to
/// say "Unknown 1" about somebody the user had already named.
#[test]
fn naming_a_person_again_elsewhere_leaves_the_first_sessions_transcript_untouched() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let first = make_session(&paths, "20260809-052600");
    let second = make_session(&paths, "20260809-052700");
    with_embeddings(&first, &[nearly(0.0), voice(3)]);
    with_embeddings(&second, &[nearly(60.0), voice(3)]);

    let mut interviewer = Scripted::answering(vec![
        named("Alice"),
        Answer::Skip,
        named("Alice"),
        Answer::Skip,
    ]);
    run(&paths, &[], &mut interviewer);
    let before = (
        std::fs::read(first.transcript_json()).unwrap(),
        std::fs::read(first.transcript_md()).unwrap(),
    );
    assert_eq!(transcript_of(&first).turns[0].speaker, "Alice");

    let mut again = Scripted::default();
    let (report, output) = run(&paths, &["20260809-052600"], &mut again);

    assert_eq!(
        (
            std::fs::read(first.transcript_json()).unwrap(),
            std::fs::read(first.transcript_md()).unwrap()
        ),
        before,
        "a second naming of Alice elsewhere must not rewrite this transcript: {output}"
    );
    assert_eq!(transcript_of(&first).turns[0].speaker, "Alice", "{output}");
    assert_eq!(report.refused, 0, "{output}");
    assert!(
        !output.contains("brought up to date"),
        "nothing changed, so nothing should have been rewritten: {output}"
    );
}
