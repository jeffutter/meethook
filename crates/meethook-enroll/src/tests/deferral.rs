//! deferred voices and leaving sessions mid-run.

use super::*;

/// TASK-046.06.01 acceptance criteria #4 and #6: a deferred voice is asked about again in
/// the same session, with the number it was first offered with.
#[test]
fn a_deferred_voice_comes_back_with_the_position_it_had() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    // Defer the first voice, answer the second, then answer the first on the second pass.
    let mut interviewer = Scripted::answering(vec![Answer::Later, named("Bob"), named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(
        interviewer.labels(),
        ["Unknown 1", "Unknown 2", "Unknown 1"],
        "{output}"
    );
    assert_eq!(
        interviewer.positions(),
        ["1/2", "2/2", "1/2"],
        "a deferred voice is the same question, so it keeps its number: {output}"
    );
    assert_eq!(report.named, 2, "{output}");

    // And the second pass's answer landed on the voice it was asked about.
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    let stored: Vec<(&str, &[f32])> = speakers
        .speakers
        .iter()
        .map(|s| (s.name.as_str(), s.embedding.as_slice()))
        .collect();
    assert_eq!(
        stored,
        [("Bob", voice(1).as_slice()), ("Alice", voice(0).as_slice())],
        "{output}"
    );
}

/// TASK-046.06.01 acceptance criterion #5: a pass that produces no answer at all is where
/// a session ends, and the voices still deferred are the skips they turned out to be.
#[test]
fn deferring_every_voice_ends_the_session_and_counts_skips() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    let before = files_under(root.path());

    let mut interviewer = Scripted::answering(vec![Answer::Later, Answer::Later]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(
        interviewer.labels(),
        ["Unknown 1", "Unknown 2"],
        "one pass, and no second one: nothing moved: {output}"
    );
    assert_eq!(report.skipped, 2, "{output}");
    assert_eq!(report.kept, 0, "{output}");
    assert_eq!(report.named, 0, "{output}");
    assert_eq!(
        files_under(root.path()),
        before,
        "deferring writes nothing, however many times it is answered"
    );
}

/// TASK-046.06.02.01 acceptance criterion #1: an answerer that says it is still working
/// keeps the session open across a pass that produced no answer, and is asked about the
/// same voices again with the same numbers.
///
/// This is the hole a full-screen frame falls into and a line prompt cannot. A frame with a
/// cursor defers a voice in order to *reach* another one, so moving the cursor backwards is
/// a pass in which nothing was answered -- and before this method existed that ended the
/// run, which from the user's side is the frame closing because they pressed Up.
#[test]
fn a_still_working_answerer_is_offered_the_deferred_voices_again() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    // Pass one defers both voices, which on its own is where a session ends. The answerer
    // says otherwise for that one pass, so pass two happens and lands a name.
    let mut interviewer = Scripted::answering(vec![
        Answer::Later,
        Answer::Later,
        named("Alice"),
        Answer::Skip,
    ])
    .working_for(1);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(
        interviewer.labels(),
        ["Unknown 1", "Unknown 2", "Unknown 1", "Unknown 2"],
        "the stalled pass kept the session open, so both voices come back: {output}"
    );
    assert_eq!(
        interviewer.positions(),
        ["1/2", "2/2", "1/2", "2/2"],
        "a re-offered voice is the same question, so it keeps its number: {output}"
    );
    assert_eq!(report.named, 1, "{output}");
    assert_eq!(report.skipped, 1, "{output}");

    // And the second pass's answer landed on the voice it was asked about.
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    let stored: Vec<(&str, &[f32])> = speakers
        .speakers
        .iter()
        .map(|s| (s.name.as_str(), s.embedding.as_slice()))
        .collect();
    assert_eq!(stored, [("Alice", voice(0).as_slice())], "{output}");
}

/// TASK-046.06.02.01 acceptance criterion #4, the half of the termination contract the loop
/// keeps for itself: a pass with nothing left to offer ends the session without asking the
/// answerer, because there is no next prompt through which it could change its mind or
/// reach [`Answer::Quit`].
#[test]
fn an_empty_queue_ends_the_session_without_consulting_the_answerer() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    // Both voices answered on the first pass, so the second pass has nothing to ask about.
    let mut interviewer = Scripted::answering(vec![named("Alice"), named("Bob")]).working_for(1);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
    assert_eq!(report.named, 2, "{output}");
    assert_eq!(
        interviewer.working_passes.get(),
        1,
        "the countdown is untouched, so the empty pass never asked: {output}"
    );
}

/// TASK-046.06.01 acceptance criterion #5, the other bucket: a deferred voice that already
/// had a name is a kept identification, exactly as pressing Enter on it would be.
///
/// And TASK-046.06.02.01 acceptance criterion #2: a session that stayed open over a
/// still-working pass ends by the same counting when the answerer does say it is finished.
#[test]
fn a_deferred_voice_that_was_named_is_kept_rather_than_skipped() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    // Only the first voice is enrolled, so `--correct` offers one named and one unnamed.
    enrolled(&[("Alice", voice(0))], &paths);

    let mut interviewer = Scripted::answering(vec![Answer::Later, Answer::Later]);
    let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

    assert_eq!(interviewer.labels(), ["Alice", "Unknown 2"], "{output}");
    assert_eq!(report.kept, 1, "{output}");
    assert_eq!(report.skipped, 1, "{output}");

    // The same run, but the answerer works through one stalled pass before it agrees it is
    // finished. Deferring writes nothing, so the second run is offered exactly what the
    // first one was.
    let mut later = Scripted::answering(vec![
        Answer::Later,
        Answer::Later,
        Answer::Later,
        Answer::Later,
    ])
    .working_for(1);
    let (report, output) = run_asking(&paths, &[], CORRECT, &mut later);

    assert_eq!(
        later.labels(),
        ["Alice", "Unknown 2", "Alice", "Unknown 2"],
        "{output}"
    );
    assert_eq!(
        (report.kept, report.skipped, report.named),
        (1, 1, 0),
        "the terminal deferred set is counted once, into the same buckets: {output}"
    );
}

/// TASK-046.06.01 acceptance criterion #5, against the in-run guard: a voice somebody
/// else's answer named while it sat deferred is passed over on the next pass rather than
/// asked twice -- and counted in neither bucket, because it was answered.
#[test]
fn a_deferred_voice_another_answer_named_is_not_asked_again() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    // One person clustering split in two: naming either half names the other.
    with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);

    let mut interviewer = Scripted::answering(vec![Answer::Later, named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(
        interviewer.labels(),
        ["Unknown 1", "Unknown 2"],
        "the deferred voice was named by the other answer, so there is nothing to ask: \
         {output}"
    );
    assert_eq!(report.named, 1, "{output}");
    assert_eq!(report.skipped, 0, "{output}");
    assert_eq!(report.kept, 0, "{output}");
}

/// TASK-049 acceptance criteria #1 and #2: one answer ends the session's questions, and the
/// voices left behind are still counted -- so the report accounts for the whole queue
/// without a keypress per voice in it.
#[test]
fn leaving_a_session_ends_it_without_asking_about_the_rest() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_fragmented_session(&paths, "20260809-052600");

    // Four offered under `--all`: one voice worth naming and a tail of fragments.
    let mut interviewer = Scripted::answering(vec![named("Alice"), Answer::Leave]);
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    assert_eq!(
        interviewer.labels(),
        ["Unknown 1", "Unknown 2"],
        "two questions for four voices: the rest are left without being asked: {output}"
    );
    assert_eq!(report.named, 1, "{output}");
    assert_eq!(
        report.skipped, 3,
        "the voice on the screen and the two behind it are all left: {output}"
    );
    assert_eq!(report.kept, 0, "{output}");
    assert!(
        output.contains("left early, 3 voice(s) left as they were"),
        "the run says why the skips outnumber the answers: {output}"
    );
}

/// TASK-049 acceptance criterion #2, the case where the arithmetic can go quietly wrong: a
/// voice this same pass has already named by naming its other half is not also reported as
/// one the run left alone.
#[test]
fn a_left_voice_named_earlier_in_the_pass_is_not_counted_twice() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_fragmented_session(&paths, "20260809-052600");
    // The first and third voices are one person clustering split in two, so the first
    // answer names a voice still sitting in the queue behind the one being asked about.
    with_embeddings(&session, &[nearly(0.0), voice(1), nearly(20.0), voice(3)]);

    let mut interviewer = Scripted::answering(vec![named("Alice"), Answer::Leave]);
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert_eq!(
        report.kept, 0,
        "the split half was named by this run, not left as it was found: {output}"
    );
    assert_eq!(
        report.skipped, 2,
        "the two genuinely unanswered voices, and not the one already counted: {output}"
    );
}

/// TASK-049 acceptance criteria #1 and #4: leaving one session opens the next, which is the
/// whole difference between this and quitting.
#[test]
fn leaving_one_session_opens_the_next() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    make_session(&paths, "20260810-052600");

    let mut interviewer = Scripted::answering(vec![Answer::Leave, named("Bob"), Answer::Skip]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    let asked: Vec<(&str, &str)> = interviewer
        .seen
        .iter()
        .map(|shown| (shown.session.as_str(), shown.number.as_str()))
        .collect();
    assert_eq!(
        asked,
        [
            ("20260809-052600", "Unknown 1"),
            ("20260810-052600", "Unknown 1"),
            ("20260810-052600", "Unknown 2"),
        ],
        "one question in the session that was left, and the next session ran in full: \
         {output}"
    );
    assert_eq!(report.named, 1, "{output}");
    assert_eq!(
        report.skipped, 3,
        "both voices of the first session and the one skipped in the second: {output}"
    );
}

/// TASK-049 acceptance criterion #3: leaving writes nothing of its own, and what was
/// answered before it is already on disk.
#[test]
fn a_name_accepted_before_leaving_stays_on_disk() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![named("Alice"), Answer::Leave]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!((report.named, report.skipped), (1, 1), "{output}");
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(
        speakers
            .speakers
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>(),
        ["Alice"],
        "the accepted name survives the session being left: {output}"
    );
    assert_eq!(
        said(&transcript_of(&session))
            .iter()
            .map(|(speaker, _, _)| *speaker)
            .collect::<Vec<_>>(),
        ["Alice", "You", "Unknown 2", "Alice"],
        "the voice left behind is untouched, and the named one is written: {output}"
    );
}

/// TASK-049 acceptance criterion #4: leaving the last session on disk ends the run rather
/// than looping over it again or erroring.
#[test]
fn leaving_the_last_session_ends_the_run() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    let before = files_under(root.path());

    let mut interviewer = Scripted::answering(vec![Answer::Leave]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(
        interviewer.labels(),
        ["Unknown 1"],
        "one question, and the run returned rather than coming round again: {output}"
    );
    assert_eq!(report.skipped, 2, "{output}");
    assert_eq!(report.named, 0, "{output}");
    assert_eq!(
        files_under(root.path()),
        before,
        "leaving writes nothing at all"
    );
}

/// TASK-049 acceptance criterion #5: leaving is an answer, not a stalled pass, so
/// [`Interviewer::still_working`] is never consulted on this path -- it can neither
/// suppress the exit nor be defeated by it.
#[test]
fn leaving_is_not_a_stalled_pass() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    // An answerer that would keep five further stalled passes open. The session ends anyway.
    let mut interviewer = Scripted::answering(vec![Answer::Leave]).working_for(5);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(interviewer.labels(), ["Unknown 1"], "{output}");
    assert_eq!(report.skipped, 2, "{output}");
    assert_eq!(
        interviewer.working_passes.get(),
        5,
        "the countdown is untouched, so the exit never went through the fixed point: \
         {output}"
    );
}

/// TASK-046.06.01 acceptance criterion #7: which voices a session offers and whether a
/// session with nothing unresolved is visited are two decisions, decidable apart.
#[test]
fn visiting_a_resolved_session_is_decided_apart_from_which_voices_it_offers() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    // Nothing is unresolved: both voices are identified before the run starts.
    enrolled(&[("Alice", voice(0)), ("Bob", voice(1))], &paths);

    // Offer the named voices, but do not visit a session that has nothing unresolved.
    let mut skipping = Scripted::default();
    let (report, output) = run_over(
        &paths,
        &[],
        None,
        CORRECT,
        Sessions::Unresolved,
        Enrolment::default(),
        &mut skipping,
    );
    assert!(skipping.seen.is_empty(), "{output}");
    assert_eq!(report.passed_over, 1, "{output}");

    // Same offer, visited anyway, which is what `--correct` asks for. The pair above and
    // below is the split itself: the frame takes the first combination and `--correct` the
    // second, off the same `Offer`.
    let mut asking = Scripted::default();
    let (report, output) = run_over(
        &paths,
        &[],
        None,
        CORRECT,
        Sessions::Every,
        Enrolment::default(),
        &mut asking,
    );
    assert_eq!(asking.labels(), ["Alice", "Bob"], "{output}");
    assert_eq!(report.passed_over, 0, "{output}");

    // And visiting cannot manufacture a question: with the named voices left out there are
    // no candidates at all, so the session is still passed over.
    let mut empty_handed = Scripted::default();
    let (report, output) = run_over(
        &paths,
        &[],
        None,
        Offer::default(),
        Sessions::Every,
        Enrolment::default(),
        &mut empty_handed,
    );
    assert!(empty_handed.seen.is_empty(), "{output}");
    assert_eq!(report.passed_over, 1, "{output}");
}

/// TASK-046.06.01 acceptance criterion #9: every snippet crosses the seam, so the cap
/// belongs to whatever is displaying them.
#[test]
fn a_voice_carries_every_snippet_it_has() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_turns(
        &paths,
        &session,
        "20260809-052600",
        (0..5)
            .map(|i| speaker_turn(f64::from(i), 0, "Unknown 1", &format!("line {i}")))
            .collect(),
    );

    let mut interviewer = Scripted::default();
    let (_, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(
        interviewer.seen[0].snippets,
        ["line 0", "line 1", "line 2", "line 3", "line 4"],
        "{output}"
    );
}

/// TASK-046.06.01 acceptance criterion #10: the universe `resolve()` requires is carried
/// across the seam, and it is not the ranking -- which is exactly the failure that doc
/// names, reproduced here.
#[test]
fn a_voice_carries_every_enrolled_name_and_not_only_the_ranked_ones() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    // "Stale" holds one reference of a dimension nothing in this session can be compared
    // to, so the ranking drops them -- and a typed "Stale" must still find them.
    enrolled(&[("Alice", voice(0)), ("Stale", vec![1.0; 8])], &paths);

    let mut interviewer = Scripted::default();
    let (_, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(
        interviewer.seen[0].offered(),
        [("Alice", 1)],
        "an incomparable reference cannot be ranked: {output}"
    );
    assert_eq!(
        interviewer.seen[0].enrolled,
        ["Alice", "Stale"],
        "but both people are enrolled, and resolving a name is about who is there: {output}"
    );
}
