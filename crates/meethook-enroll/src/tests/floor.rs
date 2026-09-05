//! the offer floors: --all, --correct, held-back voices, session-only names.

use super::*;

/// The prompt finds its lines by the cluster the turns came from, not by what they read.
/// Two voices under one enrolled name is exactly the case a correction is for, and keyed
/// on the label text both prompts would show the same person's words.
#[test]
fn each_correction_prompt_carries_only_its_own_voices_lines() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    // One reference matching both clusters: two voices, one name in the transcript.
    with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
    enrolled(&[("Andrew", nearly(10.0))], &paths);

    let mut interviewer = Scripted::default();
    let (_, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

    assert_eq!(interviewer.labels(), ["Andrew", "Andrew"], "{output}");
    assert_eq!(interviewer.seen[0].snippets, ["hi there", "let us start"]);
    assert_eq!(interviewer.seen[1].snippets, ["and from me"]);
}

/// The two flags stay orthogonal: `--correct` reaches the named voices, the floor still
/// decides which are worth a question, and only `--all` lifts it.
#[test]
fn correcting_does_not_lift_the_floor_and_all_does_not_reach_named_voices() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_speech_seconds(&session, &[40.0, 1.5]);
    enrolled(&[("Bob", voice(1))], &paths);

    let mut correcting = Scripted::default();
    let (report, output) = run_asking(&paths, &[], CORRECT, &mut correcting);
    assert_eq!(correcting.labels(), ["Unknown 1"], "{output}");
    assert_eq!(report.held_back, 1, "{output}");
    assert!(output.contains("meethook enroll --all"), "{output}");

    let mut both = Scripted::default();
    let (report, output) = run_asking(
        &paths,
        &[],
        Offer {
            quiet: true,
            named: true,
        },
        &mut both,
    );
    assert_eq!(both.labels(), ["Unknown 1", "Bob"], "{output}");
    assert_eq!(report.held_back, 0, "{output}");
}

/// TASK-021 acceptance criterion #1, at the scale a unit test can hold it: a voice under
/// [`PROMPT_FLOOR_SECONDS`] is not asked about, and the run says both how many it held
/// back and how to get at them.
#[test]
fn a_voice_too_quiet_to_be_worth_a_question_is_not_asked_about() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_speech_seconds(&session, &[40.0, 1.5]);

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(interviewer.labels(), ["Unknown 1"], "{output}");
    assert_eq!(report.held_back, 1, "{output}");
    assert_eq!(report.named, 1, "{output}");
    assert!(
        output.contains("1 unresolved voice(s), 1 quieter voice(s) not offered"),
        "{output}"
    );
    assert!(
        output.contains("meethook enroll --all"),
        "a held-back voice nobody is told how to reach is not reachable: {output}"
    );
}

/// The escape the line above advertises actually reaches them, in the same
/// first-appearance order the queue always follows.
#[test]
fn all_asks_about_the_voices_the_floor_holds_back() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_speech_seconds(&session, &[40.0, 1.5]);

    let mut interviewer = Scripted::default();
    let (report, output) = run_asking(
        &paths,
        &[],
        Offer {
            quiet: true,
            ..Offer::default()
        },
        &mut interviewer,
    );

    assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
    assert_eq!(report.held_back, 0, "{output}");
    assert!(!output.contains("not offered"), "{output}");
}

/// TASK-021 acceptance criterion #2, which is the one that matters: the floor filters
/// *questions*. Nothing is merged, deleted, renumbered or re-attributed, so the clusters
/// file is byte-identical and every held-back voice still reads the "Unknown N" it was
/// written with -- while the voice that was named reads their name.
#[test]
fn holding_a_voice_back_changes_no_cluster_and_no_unknown_numbering() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_fragmented_session(&paths, "20260809-052600");
    let before = std::fs::read(session.speaker_clusters_json()).unwrap();

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(interviewer.labels(), ["Unknown 1"], "{output}");
    assert_eq!(report.held_back, 3, "{output}");
    assert_eq!(
        std::fs::read(session.speaker_clusters_json()).unwrap(),
        before,
        "the floor must not touch the clustering"
    );
    assert_eq!(
        said(&transcript_of(&session))
            .iter()
            .map(|(speaker, _, _)| *speaker)
            .collect::<Vec<_>>(),
        ["Alice", "You", "Unknown 2", "Unknown 3", "Unknown 4"],
        "held-back voices keep the labels transcribe gave them"
    );
}

/// The proof that the floor is a filter on questions and not on labelling: one person
/// clustering split into a large half and a fragment is named once, from the half that
/// was offered, and the held-back half is relabelled with them -- exactly as a `--force`
/// re-transcribe would do it.
#[test]
fn naming_an_offered_voice_still_relabels_its_held_back_half() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
    with_speech_seconds(&session, &[40.0, 1.5]);

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(interviewer.labels(), ["Unknown 1"], "{output}");
    assert_eq!(report.named, 1, "{output}");
    assert_eq!(
        said(&transcript_of(&session))
            .iter()
            .map(|(speaker, _, _)| *speaker)
            .collect::<Vec<_>>(),
        ["Alice", "You", "Alice", "Alice"],
        "the floor decides which voices are asked about, not which turns are labelled"
    );
}

/// A floor that hides every voice in a session would be a command that does nothing, so
/// a recording where nobody clears it offers everybody. This is what keeps the
/// end-to-end tests -- three seconds of synthesised audio apiece -- meaningful.
#[test]
fn a_session_where_nobody_clears_the_floor_offers_everybody() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_speech_seconds(&session, &[1.0, 2.0]);

    let mut interviewer = Scripted::default();
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
    assert_eq!(report.held_back, 0, "{output}");
    assert!(output.contains("2 unresolved voice(s)"), "{output}");
    assert!(!output.contains("not offered"), "{output}");
}

/// TASK-019 acceptance criteria #1 and #2: an answer about a voice with 1.5 s of speech is
/// kept, and kept *here* -- the transcript reads as the person the user named, and the
/// database that every future meeting is matched against is byte-for-byte what it was.
///
/// The two acts the floor separates. Before it, this answer wrote a reference built from a
/// fragment; now it writes a row in this session's own file and says so.
#[test]
fn naming_a_voice_under_the_reference_floor_names_the_session_and_not_the_database() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_speech_seconds(&session, &[40.0, 1.5]);
    // Somebody unrelated is already enrolled, so "unchanged" is a real claim about a real
    // file rather than about one that was never created.
    enrolled(&[("Bob", voice(3))], &paths);
    let before = std::fs::read(paths.speakers_json()).unwrap();

    let mut interviewer = Scripted::answering(vec![Answer::Skip, named("Silas")]);
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    assert_eq!(interviewer.labels(), ["Unknown 1", "Unknown 2"], "{output}");
    assert_eq!(report.named, 1, "{output}");
    assert_eq!(
        report.session_only, 1,
        "a name given to a voice under the floor is a session-scoped one: {output}"
    );
    assert_eq!(
        std::fs::read(paths.speakers_json()).unwrap(),
        before,
        "a voice this quiet must not change the enrolled database at all"
    );

    let assigned = assigned_in(&session, "20260809-052600");
    assert_eq!(
        assigned
            .names
            .iter()
            .map(|row| (row.cluster, row.name.as_str(), &row.embedding))
            .collect::<Vec<_>>(),
        [(1, "Silas", &voice(1))]
    );

    // Only that voice's turns move, and they carry no confidence: nothing was matched.
    assert_eq!(
        said(&transcript_of(&session)),
        [
            ("Unknown 1", "  hi there  ", None),
            ("You", "morning", None),
            ("Silas", "and from me", None),
            ("Unknown 1", "let us start", None),
        ]
    );

    // Which of the two it did is not something a user should have to infer from a file.
    assert!(
        output.contains("named Silas in this session only"),
        "{output}"
    );
    assert!(output.contains("1.5 s of speech"), "{output}");
    assert!(output.contains("--force-reference"), "{output}");
    assert!(!output.contains("enrolled Silas"), "{output}");
}

/// The override the line above advertises: `--force-reference` writes the reference the
/// floor would have withheld, and then there is nothing session-scoped to record.
#[test]
fn force_reference_stores_the_reference_the_floor_would_have_withheld() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_speech_seconds(&session, &[40.0, 1.5]);

    let mut interviewer = Scripted::answering(vec![Answer::Skip, named("Silas")]);
    let (report, output) = run_enrolling(&paths, &[], ALL, Enrolment::Always, &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert_eq!(report.session_only, 0, "{output}");
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.speakers.len(), 1);
    assert_eq!(speakers.speakers[0].name, "Silas");
    assert_eq!(speakers.speakers[0].embedding, voice(1));
    assert!(
        !session.speaker_names_json().exists(),
        "an enrolled voice is not also a session-scoped name: {output}"
    );
    assert!(output.contains("enrolled Silas"), "{output}");

    // And the turns now carry a similarity, because this is an identification.
    assert_eq!(
        said(&transcript_of(&session))[2],
        ("Silas", "and from me", Some(1.0))
    );
}

/// TASK-019 acceptance criterion #5: an answer is an answer. A voice named for its session
/// is not asked about again -- not even by `--all`, which is what reached it in the first
/// place -- and `--correct` is the way back to it, with the prompt saying what it knows.
#[test]
fn a_voice_named_for_its_session_is_asked_about_again_only_by_correct() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = named_for_its_session(&paths, "20260809-052600");

    let mut again = Scripted::default();
    let (report, output) = run_asking(&paths, &[], ALL, &mut again);
    assert_eq!(
        again.labels(),
        ["Unknown 1"],
        "only the voice nobody named should still be asked about: {output}"
    );
    assert_eq!(report.skipped, 1, "{output}");

    let mut correcting = Scripted::default();
    let (_, output) = run_asking(&paths, &[], ALL_AND_CORRECT, &mut correcting);
    assert_eq!(correcting.labels(), ["Unknown 1", "Silas"], "{output}");
    assert_eq!(
        correcting.seen[1].attribution,
        Attribution::Assigned {
            name: "Silas".to_string()
        },
        "the prompt has to say this name was given to this session, not matched"
    );
    assert_eq!(correcting.seen[1].confidence(), None, "{output}");
    assert_eq!(
        transcript_of(&session).turns[2].speaker,
        "Silas",
        "a run that answered nothing must leave the name where it was"
    );
}

/// Correcting one: the row is replaced rather than appended to, so a voice answered twice
/// is one claim about one voice and not two rows racing to label it.
#[test]
fn re_answering_a_voice_named_for_its_session_replaces_its_row() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = named_for_its_session(&paths, "20260809-052600");

    let mut correcting = Scripted::answering(vec![Answer::Skip, named("Alex")]);
    let (report, output) = run_asking(&paths, &[], ALL_AND_CORRECT, &mut correcting);

    assert_eq!(report.session_only, 1, "{output}");
    assert_eq!(
        assigned_in(&session, "20260809-052600")
            .names
            .iter()
            .map(|row| (row.cluster, row.name.as_str()))
            .collect::<Vec<_>>(),
        [(1, "Alex")]
    );
    assert_eq!(transcript_of(&session).turns[2].speaker, "Alex");
}

/// One voice, one record. The same fragment reached again with `--force-reference` is a
/// promotion: the reference is written and the session-scoped row it replaces is dropped,
/// so the two can never be made to disagree about who this voice is.
#[test]
fn enrolling_a_voice_that_was_named_for_its_session_drops_its_row() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = named_for_its_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![Answer::Skip, named("Silas")]);
    let (report, output) = run_enrolling(
        &paths,
        &[],
        ALL_AND_CORRECT,
        Enrolment::Always,
        &mut interviewer,
    );

    assert_eq!(report.session_only, 0, "{output}");
    assert!(
        assigned_in(&session, "20260809-052600").names.is_empty(),
        "an enrolled voice must stop being an assignment too: {output}"
    );
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.speakers.len(), 1);
    assert_eq!(speakers.speakers[0].embedding, voice(1));
    assert_eq!(
        said(&transcript_of(&session))[2],
        ("Silas", "and from me", Some(1.0)),
        "the same name, now on the basis of a match"
    );
}

// A standing denial written into `speaker_names.json` before the run, exactly as an earlier
// reject would have left it: cluster `cluster` is not `name`, matched by this embedding.
fn with_denial(session: &SessionPaths, id: &str, cluster: u32, name: &str, embedding: Vec<f32>) {
    let parsed = SessionId::parse(id).unwrap();
    let mut names = SpeakerNames::read_or_empty(session, &parsed).unwrap();
    names.deny(cluster, name, &embedding);
    names.write(session).unwrap();
}

// Alice against the x-axis, so a cluster at `nearly(0.0)` identifies strictly and one at
// `nearly(54.0)` -- cosine distance 0.41, past the strict cut, inside the tentative window --
// can only be guessed.
fn alice_enrolled(paths: &Paths) {
    enrolled(&[("Alice", voice(0))], paths);
}

/// A session whose every voice is already spoken for: the main identified, the fragment
/// below the floor carrying a standing guess. The default offer has nothing left to ask, so
/// the run passes over instead of falling back to offering everybody -- which is what the
/// floor used to do when its only candidates were settled, nagging the tail on every run.
#[test]
fn a_session_whose_every_voice_is_already_settled_is_passed_over_counting_the_tail() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());

    // Guessed: the fragment reads "Alice?" in the transcript, and that guess is what settles
    // it for the queue.
    let guessed = make_session(&paths, "20260809-052600");
    with_speech_seconds(&guessed, &[40.0, 1.5]);
    with_embeddings(&guessed, &[nearly(0.0), nearly(54.0)]);
    alice_enrolled(&paths);

    // Denied: the same shape with a standing denial pre-written, so the guess never appears
    // and the fragment reads an ordinary "Unknown 2" -- indistinguishable from an open
    // fragment except through the denial ids the queue was given.
    let denied = make_session(&paths, "20260809-052700");
    with_speech_seconds(&denied, &[40.0, 1.5]);
    with_embeddings(&denied, &[nearly(0.0), nearly(54.0)]);
    with_denial(&denied, "20260809-052700", 1, "Alice", nearly(54.0));

    let mut interviewer = Scripted::default();
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert!(interviewer.seen.is_empty(), "{output}");
    assert_eq!(report.passed_over, 2, "{output}");
    // Both sessions say why they were passed over and how to reach the tail anyway: the
    // guess and the denial settle identically, because both are deliberate state the user
    // can see in the file.
    assert!(
        output.contains(
            "20260809-052600  passed over: nothing unresolved \
             (1 named voice(s), 1 guessed or dismissed -- meethook enroll --all --correct)"
        ),
        "{output}"
    );
    assert!(
        output.contains(
            "20260809-052700  passed over: nothing unresolved \
             (1 named voice(s), 1 guessed or dismissed -- meethook enroll --all --correct)"
        ),
        "{output}"
    );
    // And the denial did its job upstream too: the suppressed guess is not in the file.
    let markdown = std::fs::read_to_string(denied.transcript_md()).unwrap();
    assert!(!markdown.contains("Alice?"), "{markdown}");
    assert!(markdown.contains("Unknown 2"), "{markdown}");
}

/// A pass-over with both halves at once names both escapes: `--all` alone cannot resurface
/// the named voice -- `queue()` still excludes it unless `offer.named` is set -- so the
/// hint carries `--correct` beside it rather than letting the settled tail outrank it.
#[test]
fn a_pass_over_with_both_named_and_settled_voices_hints_at_both_flags() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());

    // One strict match (named) and one guessable fragment (settled): the same shape the
    // settled-tail test above holds, asserted here only on the escape it prints.
    let session = make_session(&paths, "20260809-052600");
    with_speech_seconds(&session, &[40.0, 1.5]);
    with_embeddings(&session, &[nearly(0.0), nearly(54.0)]);
    alice_enrolled(&paths);

    let mut interviewer = Scripted::default();
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert!(interviewer.seen.is_empty(), "{output}");
    assert_eq!(report.passed_over, 1, "{output}");
    assert!(
        output.contains(
            "20260809-052600  passed over: nothing unresolved \
             (1 named voice(s), 1 guessed or dismissed -- meethook enroll --all --correct)"
        ),
        "{output}"
    );
}

/// The mixed case: an open voice above the floor plus a settled tail. The run asks about
/// the open voice only -- the settled fragment is in neither the offer nor the held-back
/// count, so the queue line carries no `--all` clause at all.
#[test]
fn a_mixed_session_offers_only_its_open_voices_and_counts_nothing_held_back() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    let id = SessionId::parse("20260809-052600").unwrap();

    // Main identified, fragment guessed, third voice open above the floor.
    let mut clusters = vec![
        cluster(0, 0.0, (0.5, 2.5)),
        cluster(1, 3.0, (3.0, 3.5)),
        cluster(2, 6.0, (6.0, 7.0)),
    ];
    clusters[1].embedding = nearly(54.0);
    for (c, seconds) in clusters.iter_mut().zip([40.0, 1.5, 8.0]) {
        c.speech_seconds = seconds;
    }
    clusters[0].embedding = nearly(0.0);
    clusters[2].embedding = voice(2);
    SpeakerClusters::new(id.clone(), clusters)
        .write(&session)
        .unwrap();
    write_transcript(
        &Transcript::new(
            id.clone(),
            vec![
                speaker_turn(0.0, 0, "Unknown 1", "hi there"),
                speaker_turn(3.0, 1, "Unknown 2", "mm"),
                speaker_turn(6.0, 2, "Unknown 3", "over here"),
            ],
        ),
        &paths,
        &session,
        &session_metadata(&id),
    );
    alice_enrolled(&paths);

    let mut interviewer = Scripted::default();
    let (report, output) = run(&paths, &[], &mut interviewer);

    // One question, about the open voice; the settled tail is invisible to the offer.
    assert_eq!(interviewer.labels(), ["Unknown 3"], "{output}");
    assert_eq!(report.held_back, 0, "{output}");
    assert!(
        !output.contains("quieter voice(s) not offered"),
        "the settled tail must not advertise --all: {output}"
    );
    assert!(
        output.contains("20260809-052600  1 unresolved voice(s)"),
        "{output}"
    );
}

/// `--all` lifts the settled exclusion exactly as it lifts the floor: the guessed fragment
/// comes back into the offer, marked, beside the open voices.
#[test]
fn all_still_reaches_a_settled_fragment_marked_as_a_guess() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    let id = SessionId::parse("20260809-052600").unwrap();

    let mut clusters = vec![
        cluster(0, 0.0, (0.5, 2.5)),
        cluster(1, 3.0, (3.0, 3.5)),
        cluster(2, 6.0, (6.0, 7.0)),
    ];
    clusters[0].embedding = nearly(0.0);
    clusters[1].embedding = nearly(54.0);
    clusters[2].embedding = voice(2);
    for (c, seconds) in clusters.iter_mut().zip([40.0, 1.5, 8.0]) {
        c.speech_seconds = seconds;
    }
    SpeakerClusters::new(id.clone(), clusters)
        .write(&session)
        .unwrap();
    write_transcript(
        &Transcript::new(
            id.clone(),
            vec![
                speaker_turn(0.0, 0, "Unknown 1", "hi there"),
                speaker_turn(3.0, 1, "Unknown 2", "mm"),
                speaker_turn(6.0, 2, "Unknown 3", "over here"),
            ],
        ),
        &paths,
        &session,
        &session_metadata(&id),
    );
    alice_enrolled(&paths);

    let mut interviewer = Scripted::default();
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    // The guess first (it spoke first), still marked as a guess: reaching it is the user's
    // choice, and the mark is what says the answer costs nothing but a label.
    assert_eq!(interviewer.labels(), ["Alice?", "Unknown 3"], "{output}");
    assert_eq!(report.held_back, 0, "{output}");
}

/// `--correct` is byte-identical to what it printed before settledness existed: the settled
/// fragment is still held back by the floor and counted as such, because the widening flag
/// lifted the exclusion the moment it lifted everything else.
#[test]
fn correct_treats_a_settled_fragment_exactly_as_it_did_before() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    let id = SessionId::parse("20260809-052600").unwrap();

    let mut clusters = vec![
        cluster(0, 0.0, (0.5, 2.5)),
        cluster(1, 3.0, (3.0, 3.5)),
        cluster(2, 6.0, (6.0, 7.0)),
    ];
    clusters[0].embedding = nearly(0.0);
    clusters[1].embedding = nearly(54.0);
    clusters[2].embedding = voice(2);
    for (c, seconds) in clusters.iter_mut().zip([40.0, 1.5, 8.0]) {
        c.speech_seconds = seconds;
    }
    SpeakerClusters::new(id.clone(), clusters)
        .write(&session)
        .unwrap();
    write_transcript(
        &Transcript::new(
            id.clone(),
            vec![
                speaker_turn(0.0, 0, "Unknown 1", "hi there"),
                speaker_turn(3.0, 1, "Unknown 2", "mm"),
                speaker_turn(6.0, 2, "Unknown 3", "over here"),
            ],
        ),
        &paths,
        &session,
        &session_metadata(&id),
    );
    alice_enrolled(&paths);

    let mut interviewer = Scripted::default();
    let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

    assert_eq!(interviewer.labels(), ["Alice", "Unknown 3"], "{output}");
    assert_eq!(report.held_back, 1, "{output}");
    assert!(
        output.contains(
            "20260809-052600  2 voice(s) to review, 1 of them already named, \
             1 quieter voice(s) not offered -- meethook enroll --all"
        ),
        "{output}"
    );
}

/// Targeting bypasses the queue and its gates alike, settledness included: the fragment is
/// reachable by its marked label and by its number, while the bare name reaches the voice
/// that reads it -- the main -- not the guess.
#[test]
fn targeting_reaches_a_settled_fragment_by_label_or_number_but_not_by_bare_name() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_speech_seconds(&session, &[40.0, 1.5]);
    with_embeddings(&session, &[nearly(0.0), nearly(54.0)]);
    alice_enrolled(&paths);

    // By the marked label the transcript prints.
    let mut by_label = Scripted::default();
    let (_, output) = run_targeting(&paths, &["20260809-052600"], "Alice?", &mut by_label);
    assert_eq!(by_label.labels(), ["Alice?"], "{output}");

    // By the number, which keeps pointing at the same voice whatever it reads as.
    let mut by_number = Scripted::default();
    let (_, output) = run_targeting(&paths, &["20260809-052600"], "2", &mut by_number);
    assert_eq!(by_number.labels(), ["Alice?"], "{output}");

    // By the moment it spoke: `--at` resolves through the transcript rather than the queue,
    // so it reaches the guess the same way the number does.
    let mut by_time = Scripted::default();
    let (_, output) = run_at(&paths, &["20260809-052600"], "00:03", &mut by_time);
    assert_eq!(by_time.labels(), ["Alice?"], "{output}");

    // The bare name matches the voice that reads it: the identified main, not the guess.
    let mut by_name = Scripted::default();
    let (_, output) = run_targeting(&paths, &["20260809-052600"], "Alice", &mut by_name);
    assert_eq!(by_name.labels(), ["Alice"], "{output}");
}
