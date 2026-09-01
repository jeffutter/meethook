//! positions, prompt contents, clips, and what a prompt offers.

use super::*;

/// TASK-026 acceptance criteria #1 and #2: every prompt says which voice it is of how
/// many, and that total is the number the session line printed just above the questions.
/// Asserted together, because the whole value of the number is that it agrees with what
/// the user was told a moment ago.
#[test]
fn every_prompt_says_which_voice_it_is_of_how_many() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::default();
    let (_, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(interviewer.positions(), ["1/2", "2/2"], "{output}");
    assert!(output.contains("2 unresolved voice(s)"), "{output}");
}

/// TASK-026 acceptance criteria #4 and #6: a run over several sessions counts each session
/// separately, and the session on the same prompt says which one a position belongs to.
///
/// The second session's total is 1 rather than 2 because Alice is identified out of its
/// queue before any question is asked -- which is acceptance criterion #2 from the other
/// direction: the total is whatever that session actually offered.
#[test]
fn positions_restart_in_each_session_of_a_run() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    make_session(&paths, "20260809-052700");

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (_, output) = run(&paths, &[], &mut interviewer);

    let sessions: Vec<&str> = interviewer
        .seen
        .iter()
        .map(|v| v.session.as_str())
        .collect();
    assert_eq!(
        sessions,
        ["20260809-052600", "20260809-052600", "20260809-052700"],
        "{output}"
    );
    assert_eq!(interviewer.positions(), ["1/2", "2/2", "1/1"], "{output}");
    assert!(
        output.contains("20260809-052700  1 unresolved voice(s)"),
        "{output}"
    );
}

/// TASK-026 acceptance criterion #3, and the decision behind it made assertable: a voice an
/// earlier answer in the same run named is passed over, and its number goes with it. The
/// positions read 1/4, 2/4, 4/4 -- a gap in the middle and a total that does not shrink --
/// because the total is what the session line promised and the gap is a question that
/// answered itself.
#[test]
fn a_voice_an_earlier_answer_named_leaves_a_gap_in_the_positions() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_fragmented_session(&paths, "20260809-052600");
    // Clusters 0 and 2 are one person that clustering split in two, so naming the first
    // names the third on the way past.
    with_embeddings(&session, &[nearly(0.0), voice(1), nearly(20.0), voice(3)]);

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (_, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    assert_eq!(
        interviewer.labels(),
        ["Unknown 1", "Unknown 2", "Unknown 4"],
        "Unknown 3 is Alice, already named by the first answer: {output}"
    );
    assert_eq!(interviewer.positions(), ["1/4", "2/4", "4/4"], "{output}");
    assert!(output.contains("4 unresolved voice(s)"), "{output}");
}

/// Acceptance criterion #8: nothing to ask about is passed over silently rather than
/// prompting, and so is a session nobody has transcribed yet.
#[test]
fn sessions_with_nothing_to_ask_about_are_passed_over_without_prompting() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());

    // Already fully identified.
    let resolved = make_session(&paths, "20260809-052600");
    EnrolledSpeakers::new(vec![
        EnrolledSpeaker {
            name: "Alice".to_string(),
            embedding: voice(0),
            clip_seconds: None,
        },
        EnrolledSpeaker {
            name: "Bob".to_string(),
            embedding: voice(1),
            clip_seconds: None,
        },
    ])
    .write(&paths)
    .unwrap();

    // Recorded but never transcribed.
    let untranscribed = paths.session(&SessionId::parse("20260809-052700").unwrap());
    std::fs::create_dir_all(untranscribed.dir()).unwrap();
    std::fs::write(untranscribed.session_json(), b"{}").unwrap();

    // The recorder died mid-session.
    let orphan = paths.session(&SessionId::parse("20260809-052800").unwrap());
    std::fs::create_dir_all(orphan.dir()).unwrap();

    let mut interviewer = Scripted::default();
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert!(interviewer.seen.is_empty(), "{:?}", interviewer.seen);
    assert_eq!(report.passed_over, 3, "{output}");
    assert_eq!(report.failed, 0, "{output}");
    assert!(output.contains("nothing unresolved"), "{output}");
    // A session where everybody is already named is the one somebody is looking at when
    // one of those names is wrong, and this line is all it prints.
    assert!(
        output.contains("2 named voice(s) -- meethook enroll --correct"),
        "a correction nobody is told how to reach is not reachable: {output}"
    );
    assert!(output.contains("not transcribed yet"), "{output}");
    assert!(output.contains("no session.json"), "{output}");
    // Nobody was asked, and the transcript still caught up with the database: a session
    // where everyone is already known is exactly the one that would otherwise be passed
    // over on every future run, keeping its stale labels for good.
    assert_eq!(
        said(&transcript_of(&resolved))
            .iter()
            .map(|(speaker, _, _)| *speaker)
            .collect::<Vec<_>>(),
        ["Alice", "You", "Bob", "Alice"]
    );
    assert!(output.contains("brought up to date"), "{output}");
}

/// Acceptance criterion #3 and the queue order: each prompt carries that voice's own
/// lines and its own clip, and they arrive in "Unknown N" order rather than in talk-time
/// order.
///
/// Cluster 0 is the first to speak and cluster 1 the second, so the labels below are also
/// the assertion that first-appearance order is what the queue follows.
#[test]
fn each_prompt_carries_that_voices_snippets_and_clip_in_unknown_order() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::default();
    run(&paths, &[], &mut interviewer);

    assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"]);
    assert_eq!(
        interviewer.seen[0].snippets,
        ["hi there", "let us start"],
        "only this voice's lines, whitespace trimmed"
    );
    assert_eq!(interviewer.seen[1].snippets, ["and from me"]);
    assert_eq!(interviewer.seen[0].speech_seconds, 10.0);
    // The representative spans 0.5 s to 2.5 s of a 16 kHz track.
    assert_eq!(interviewer.seen[0].clip_samples, 32_000);
    assert_eq!(interviewer.seen[1].clip_samples, 32_000);
}

/// Acceptance criterion #11: no audio is not a failed session. The prompt still happens,
/// still carries the snippets, and an answer still lands on disk.
#[test]
fn a_session_with_no_speaker_wav_is_still_asked_about_with_an_empty_clip() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    std::fs::remove_file(session.speaker_wav()).unwrap();

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert_eq!(report.failed, 0, "{output}");
    assert_eq!(interviewer.seen[0].clip_samples, 0);
    assert_eq!(interviewer.seen[0].snippets, ["hi there", "let us start"]);
    assert_eq!(
        interviewer.seen[0].snippet_samples,
        [0, 0],
        "and no audio under any line either, with the times still saying when they were said"
    );
    assert_eq!(interviewer.seen[0].snippet_times, [(0.0, 1.0), (4.0, 1.0)]);
    assert_eq!(transcript_of(&session).turns[0].speaker, "Alice");
}

/// A representative that runs off the end of the track -- a truncated `speaker.wav` -- is
/// clipped to what is there rather than refused, for the same reason as above.
#[test]
fn a_representative_past_the_end_of_the_track_plays_what_is_there() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    let id = SessionId::parse("20260809-052600").unwrap();
    SpeakerClusters::new(
        id,
        vec![
            cluster(0, 0.0, (5.0, 90.0)),
            cluster(1, 3.0, (600.0, 620.0)),
        ],
    )
    .write(&session)
    .unwrap();

    let mut interviewer = Scripted::default();
    run(&paths, &[], &mut interviewer);

    // The track is six seconds long: one second of the first clip survives, none of the
    // second.
    assert_eq!(interviewer.seen[0].clip_samples, 16_000);
    assert_eq!(interviewer.seen[1].clip_samples, 0);
}

/// Acceptance criterion #2 end to end, which is the half no unit test can see: that the run
/// actually reads the offset out of `session.json` rather than defaulting it to zero.
///
/// The fixture's speaker track starts a second after the microphone's, so every snippet's
/// track time is its turn's timeline second less one -- and the audio under it is a second
/// earlier in `speaker.wav` than a run that ignored the offset would have cut.
#[test]
fn a_session_whose_speaker_track_started_late_still_lines_the_audio_up() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    let metadata = with_speaker_offset(&session, "20260809-052600", 1.0);
    // Every turn after the speaker track's own start, so that what this test measures is
    // the offset and not the clamp that catches a turn from before it.
    write_transcript(
        &Transcript::new(
            SessionId::parse("20260809-052600").unwrap(),
            vec![
                speaker_turn(2.0, 0, "Unknown 1", "hi there"),
                mic_turn(2.5, "morning"),
                speaker_turn(3.0, 1, "Unknown 2", "and from me"),
                speaker_turn(5.0, 0, "Unknown 1", "let us start"),
            ],
        ),
        &paths,
        &session,
        &metadata,
    );

    let mut interviewer = Scripted::default();
    let (_, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
    assert_eq!(
        interviewer.seen[0].snippet_times,
        [(1.0, 1.0), (4.0, 1.0)],
        "the turns are at 2 s and 5 s on the timeline, and the speaker track began at 1 s"
    );
    assert_eq!(interviewer.seen[1].snippet_times, [(2.0, 1.0)]);
    assert_eq!(interviewer.seen[0].snippet_samples, [16_000, 16_000]);
    assert_eq!(interviewer.seen[1].snippet_samples, [16_000]);
    // The clip is untouched by the offset: a representative's seconds are already track
    // time, which is exactly the confusion this ticket exists to keep apart.
    assert_eq!(interviewer.seen[0].clip_samples, 32_000);
}

/// Rewrites a fixture's `session.json` so that its speaker track begins `seconds` after its
/// microphone track, which is what `speaker_offset_seconds` reads.
///
/// Separate from [`session_metadata`] so that the default fixture -- both tracks starting
/// together, which is what every other test here assumes -- does not move.
fn with_speaker_offset(session: &SessionPaths, id: &str, seconds: f64) -> SessionMetadata {
    let id = SessionId::parse(id).unwrap();
    let base = session_metadata(&id);
    let mut speaker = base.speaker;
    // Ticks, not nanoseconds: `session.json` records the machine's rational timebase and
    // the arithmetic that reads it back is exact, so the fixture does the same conversion
    // in reverse rather than guessing at a tick.
    speaker.host_ticks += (seconds * 1e9 * f64::from(speaker.timebase_denom)
        / f64::from(speaker.timebase_numer)) as u64;
    let metadata = SessionMetadata::new(id, base.start_time, base.mic, speaker);
    metadata.write(&session.session_json()).unwrap();
    metadata
}

/// Acceptance criterion #7: the prompt is handed everybody enrolled, nearest first, so an
/// [`Interviewer`] can offer names without ever opening `speakers.json`.
///
/// All three references are outside `IDENTIFY_DISTANCE` of cluster 0 -- 60, 75 and 85
/// degrees, against a cut at 0.40 of cosine distance -- which is exactly the voice
/// identification gave up on and the one whose prompt has something to offer. The names run
/// against the ranking alphabetically, so a list in name order would fail this.
#[test]
fn a_prompt_is_handed_every_enrolled_person_nearest_first() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_embeddings(&session, &[voice(0), voice(1)]);
    enrolled(
        &[
            ("Zoe", nearly(60.0)),
            ("Alice", nearly(75.0)),
            ("Mona", nearly(85.0)),
        ],
        &paths,
    );

    let mut interviewer = Scripted::answering(vec![Answer::Skip]);
    let (_, output) = run(&paths, &[], &mut interviewer);

    // Cluster 1 sits close to the 60-degree reference, so it is identified and not asked
    // about; cluster 0 is the one question this run has.
    assert_eq!(interviewer.labels(), ["Unknown 1"], "{output}");
    let shown = &interviewer.seen[0];
    assert_eq!(
        shown.offered(),
        [("Zoe", 1), ("Alice", 1), ("Mona", 1)],
        "{output}"
    );
    assert!(
        (shown.resembles[0].similarity - 60.0f32.to_radians().cos()).abs() < 1e-6,
        "{:?}",
        shown.resembles
    );
    // Every one of them is past the cut identification applies, and still offered.
    for candidate in &shown.resembles {
        assert!(
            1.0 - candidate.similarity > IDENTIFY_DISTANCE,
            "{candidate:?} should be outside the cut for this test to mean anything"
        );
    }
}

/// The ranking reflects the database as it stands at the prompt, not as it stood when the
/// run began -- so a name given a moment ago is offered for the next voice, which is the
/// case that matters when clustering has split one person in two.
#[test]
fn a_name_given_earlier_in_the_run_is_offered_for_a_later_voice() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![named("Alice"), Answer::Skip]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    // Nobody was enrolled when the first question was asked.
    assert_eq!(interviewer.seen[0].offered(), [], "{output}");
    assert_eq!(interviewer.seen[1].offered(), [("Alice", 1)], "{output}");
}

/// Acceptance criterion #6 at the seam, and the state of every install before anybody has
/// been enrolled: nobody to offer is an empty list, and the question is still asked.
#[test]
fn an_empty_database_offers_nobody_and_still_prompts() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![Answer::Skip, Answer::Skip]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.skipped, 2, "{output}");
    assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
    assert!(
        interviewer.seen.iter().all(|v| v.resembles.is_empty()),
        "{output}"
    );
}

/// A correction prompt shows a name and a ranking on one screen, and the two must not
/// disagree: the first entry is the person the identification already named, carrying the
/// same number the label does.
#[test]
fn an_identified_voices_ranking_leads_with_the_name_it_was_given() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_embeddings(&session, &[voice(0), voice(1)]);
    enrolled(&[("Alice", nearly(10.0)), ("Zoe", nearly(70.0))], &paths);

    let mut interviewer = Scripted::answering(vec![Answer::Skip, Answer::Skip]);
    let (_, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

    assert_eq!(interviewer.labels(), ["Alice", "Zoe"], "{output}");
    for shown in &interviewer.seen {
        assert_eq!(shown.resembles[0].name, shown.label(), "{output}");
        assert_eq!(
            Some(shown.resembles[0].similarity),
            shown.confidence(),
            "{output}"
        );
    }
    // Both people are offered for both voices; only the order differs.
    assert_eq!(
        interviewer.seen[0].offered(),
        [("Alice", 1), ("Zoe", 1)],
        "{output}"
    );
    assert_eq!(
        interviewer.seen[1].offered(),
        [("Zoe", 1), ("Alice", 1)],
        "{output}"
    );
}
