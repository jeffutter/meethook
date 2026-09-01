//! targeting by number, name, selector, and timestamp.

use super::*;

/// TASK-025 acceptance criterion #1: `--voice` asks about the voice it names and about
/// nobody else, in both the forms the number can be written in.
#[test]
fn a_voice_selected_by_number_is_the_only_one_asked_about() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![named("Bob")]);
    let (report, output) = run_targeting(&paths, &["20260809-052600"], "2", &mut interviewer);

    assert_eq!(interviewer.labels(), ["Unknown 2"], "{output}");
    assert_eq!(report.named, 1, "{output}");
    // TASK-026: a targeted run says `1/1` rather than suppressing the position. It is true,
    // and it says the useful thing -- this is the only question, the run ends after this
    // answer. Suppressing it would put a rule about when a position is worth showing inside
    // the terminal, where no test can see what the user was shown.
    assert_eq!(interviewer.positions(), ["1/1"], "{output}");
    assert_eq!(
        said(&transcript_of(&session))
            .iter()
            .map(|(speaker, _, _)| *speaker)
            .collect::<Vec<_>>(),
        ["Unknown 1", "You", "Bob", "Unknown 1"],
        "the voice that was not asked about must be left exactly as it was"
    );

    // The written-out label is the same selector: a user reading "Unknown 1" off a prompt
    // header should not have to work out which part of it to type.
    let mut spelled_out = Scripted::default();
    let (_, output) = run_targeting(&paths, &["20260809-052600"], "Unknown 1", &mut spelled_out);
    assert_eq!(spelled_out.labels(), ["Unknown 1"], "{output}");
}

/// Acceptance criteria #2 and #3: a voice the database has already named is reachable by
/// that name, with no `--correct` -- which is the state somebody is in when the name is
/// the thing that is wrong.
#[test]
fn a_voice_can_be_selected_by_the_name_it_currently_reads_as() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    enrolled(&[("Bob", voice(1))], &paths);

    let mut interviewer = Scripted::answering(vec![named("Robert Chen")]);
    let (report, output) = run_targeting(&paths, &["20260809-052600"], "Bob", &mut interviewer);

    assert_eq!(interviewer.labels(), ["Bob"], "{output}");
    assert_eq!(
        interviewer.seen[0].attribution,
        Attribution::Identified {
            name: "Bob".to_string(),
            similarity: 1.0
        },
        "the prompt has to ask whether this identification is right, not who this is"
    );
    assert_eq!(report.named, 1, "{output}");
    assert_eq!(transcript_of(&session).turns[2].speaker, "Robert Chen");
}

/// Acceptance criterion #3 for the other filter, and the reason `held_back` stays at zero:
/// a run aimed at one voice is not holding anything back, so it must not end on a line
/// offering `--all`.
#[test]
fn a_targeted_voice_under_the_prompt_floor_is_asked_about_without_all() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_speech_seconds(&session, &[40.0, 1.5]);

    let mut interviewer = Scripted::default();
    let (report, output) = run_targeting(&paths, &["20260809-052600"], "2", &mut interviewer);

    assert_eq!(interviewer.labels(), ["Unknown 2"], "{output}");
    assert_eq!(report.held_back, 0, "{output}");
    assert!(!output.contains("not offered"), "{output}");
}

/// TASK-025 acceptance criterion #4, written as the comparison it actually is rather than
/// as literal expectations: the prompt a targeted voice gets is the prompt that voice gets
/// in a full run -- same header, same snippets, same clip -- because it is produced by the
/// same code from the same cluster.
///
/// Everything but the position, which is a fact about the *run* rather than about the
/// voice, and a run aimed at one voice genuinely is a different run: it has one question in
/// it. Destructured exhaustively, no `..`, so that a field added to [`Voice`] later cannot
/// quietly fall out of this comparison -- the compiler makes the author name it and say
/// which side of the line it is on.
#[test]
fn a_targeted_prompt_is_what_the_full_run_would_have_shown() {
    let id = "20260809-052600";

    let queued_root = tempfile::tempdir().unwrap();
    let queued_paths = Paths::new(queued_root.path());
    make_session(&queued_paths, id);
    let mut queued = Scripted::default();
    let (_, output) = run_asking(&queued_paths, &[], ALL_AND_CORRECT, &mut queued);
    assert_eq!(queued.labels(), ["Unknown 1", "Unknown 2"], "{output}");

    let targeted_root = tempfile::tempdir().unwrap();
    let targeted_paths = Paths::new(targeted_root.path());
    make_session(&targeted_paths, id);
    let mut aimed = Scripted::default();
    let (_, output) = run_targeting(&targeted_paths, &[id], "2", &mut aimed);

    assert_eq!(aimed.seen.len(), 1, "{output}");
    let Shown {
        session,
        meeting,
        position,
        attribution,
        number,
        speech_seconds,
        queue,
        snippets,
        snippet_times,
        snippet_samples,
        clip_samples,
        resembles,
        enrolled,
    } = &aimed.seen[0];
    let queued = &queued.seen[1];
    assert_eq!(session, &queued.session);
    assert_eq!(meeting, &queued.meeting);
    assert_eq!(attribution, &queued.attribution);
    assert_eq!(number, &queued.number);
    assert_eq!(speech_seconds, &queued.speech_seconds);
    // A targeted prompt sees the whole session too: narrowing decides which voices are
    // *asked about*, and the queue is what the session holds.
    assert_eq!(queue, &queued.queue);
    assert_eq!(snippets, &queued.snippets);
    assert_eq!(snippet_times, &queued.snippet_times);
    assert_eq!(snippet_samples, &queued.snippet_samples);
    assert_eq!(clip_samples, &queued.clip_samples);
    assert_eq!(resembles, &queued.resembles);
    assert_eq!(enrolled, &queued.enrolled);
    assert_eq!(
        (position.to_string(), queued.position.to_string()),
        ("1/1".to_string(), "2/2".to_string()),
        "the position is the one thing that differs, because it counts the run's questions \
         and the targeted run has one"
    );
}

/// Acceptance criterion #5: reaching a voice differently does not write differently. A
/// targeted answer about a 1.5 s voice takes the same session-only path, and the same
/// `--force-reference` override lifts it.
#[test]
fn naming_a_targeted_quiet_voice_still_writes_only_a_session_name() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_speech_seconds(&session, &[40.0, 1.5]);
    // Somebody unrelated is already enrolled, so "unchanged" is a claim about a real file.
    enrolled(&[("Bob", voice(3))], &paths);
    let before = std::fs::read(paths.speakers_json()).unwrap();

    let mut interviewer = Scripted::answering(vec![named("Silas")]);
    let (report, output) = run_targeting(&paths, &["20260809-052600"], "2", &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert_eq!(report.session_only, 1, "{output}");
    assert_eq!(
        std::fs::read(paths.speakers_json()).unwrap(),
        before,
        "a targeted answer about a voice this quiet must not touch the database either"
    );
    assert_eq!(
        assigned_in(&session, "20260809-052600")
            .names
            .iter()
            .map(|row| (row.cluster, row.name.as_str()))
            .collect::<Vec<_>>(),
        [(1, "Silas")]
    );
    assert!(
        output.contains("named Silas in this session only"),
        "{output}"
    );

    // And the override composes with a selector exactly as it does with the queue: it is
    // the other axis, and the targeted path never touches it.
    let forced_root = tempfile::tempdir().unwrap();
    let forced_paths = Paths::new(forced_root.path());
    let forced = make_session(&forced_paths, "20260809-052600");
    with_speech_seconds(&forced, &[40.0, 1.5]);

    let mut forcing = Scripted::answering(vec![named("Silas")]);
    let second = VoiceSelector::from("2");
    let (report, output) = run_over(
        &forced_paths,
        &["20260809-052600"],
        Some(Selection::Voice(second)),
        Offer::default(),
        Sessions::default(),
        Enrolment::Always,
        &mut forcing,
    );

    assert_eq!(report.session_only, 0, "{output}");
    let speakers = EnrolledSpeakers::read_or_empty(&forced_paths).unwrap();
    assert_eq!(speakers.speakers.len(), 1);
    assert_eq!(speakers.speakers[0].name, "Silas");
    assert_eq!(speakers.speakers[0].embedding, voice(1));
}

/// Acceptance criterion #6, the miss half: a selector that names nobody asks nothing, says
/// so, and lists what the session does have -- quiet voices included, since those are what
/// somebody is reaching for when they miss. `failed` is what turns that into a non-zero
/// exit at the CLI.
#[test]
fn a_selector_matching_nothing_reports_what_the_session_has_and_fails() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_fragmented_session(&paths, "20260809-052600");

    for missed in ["Nobody", "9"] {
        let mut interviewer = Scripted::default();
        let (report, output) =
            run_targeting(&paths, &["20260809-052600"], missed, &mut interviewer);

        assert!(interviewer.seen.is_empty(), "{missed}: {output}");
        assert_eq!(report.failed, 1, "{missed}: {output}");
        assert!(output.contains("no voice matched"), "{missed}: {output}");
        for label in ["Unknown 1", "Unknown 2", "Unknown 3", "Unknown 4"] {
            assert!(
                output.contains(label),
                "a miss has to say what the session contains, including the voices under \
                 the floor -- {label} missing from: {output}"
            );
        }
    }
}

/// Acceptance criterion #6, the ambiguous half. Two clusters under one enrolled name is
/// exactly the false accept `--correct` exists to fix, so the message has to hand back the
/// thing that tells them apart rather than picking one of them.
#[test]
fn an_ambiguous_selector_names_both_voices_and_the_numbers_that_split_them() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
    enrolled(&[("Alice", nearly(0.0))], &paths);

    let mut interviewer = Scripted::answering(vec![named("Someone")]);
    let (report, output) = run_targeting(&paths, &["20260809-052600"], "Alice", &mut interviewer);

    assert!(interviewer.seen.is_empty(), "{output}");
    assert_eq!(report.failed, 1, "{output}");
    assert!(output.contains("matches 2 voices"), "{output}");
    assert!(output.contains("Unknown 1"), "{output}");
    assert!(output.contains("Unknown 2"), "{output}");

    // ...and the number it handed back does reach one of them.
    let mut disambiguated = Scripted::answering(vec![named("Someone")]);
    let (report, output) = run_targeting(&paths, &["20260809-052600"], "2", &mut disambiguated);
    assert_eq!(disambiguated.labels(), ["Alice"], "{output}");
    assert_eq!(report.named, 1, "{output}");
}

/// A voice number means nothing across sessions and a name would fan out over every
/// recording on disk, so a selector without exactly one session id is refused before
/// anything is read -- and refused loudly, since the alternative is a run that asks about
/// somebody else's Unknown 2.
#[test]
fn a_selector_without_exactly_one_session_id_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    make_session(&paths, "20260809-052700");

    for ids in [&[][..], &["20260809-052600", "20260809-052700"][..]] {
        let mut interviewer = Scripted::default();
        let (report, output) = run_targeting(&paths, ids, "2", &mut interviewer);

        assert!(interviewer.seen.is_empty(), "{ids:?}: {output}");
        assert_eq!(report.failed, 1, "{ids:?}: {output}");
        assert!(
            output.contains("--voice needs exactly one session id"),
            "{ids:?}: {output}"
        );
    }
}

/// Why the number is the "Unknown N" and not the cluster id, at the level a user meets it:
/// naming a voice does not renumber anybody, so the number that reached it still reaches
/// it afterwards -- and the second visit is a correction.
#[test]
fn a_number_keeps_pointing_at_a_voice_after_it_has_been_named() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    let mut first = Scripted::answering(vec![named("Bob")]);
    let (report, output) = run_targeting(&paths, &["20260809-052600"], "2", &mut first);
    assert_eq!(first.labels(), ["Unknown 2"], "{output}");
    assert_eq!(report.named, 1, "{output}");

    let mut again = Scripted::answering(vec![named("Robert Chen")]);
    let (report, output) = run_targeting(&paths, &["20260809-052600"], "2", &mut again);

    assert_eq!(
        again.labels(),
        ["Bob"],
        "the same number must reach the same voice, now under its name: {output}"
    );
    assert_eq!(report.named, 1, "{output}");
    assert_eq!(transcript_of(&session).turns[2].speaker, "Robert Chen");
}

/// TASK-033 acceptance criteria #1 and #7: a session id, a timestamp and a name are the
/// whole command. Nothing is prompted -- [`GivenName`] has no terminal to prompt with --
/// and both transcript files come out of it reading the new name.
#[test]
fn a_timestamp_and_a_name_name_the_voice_speaking_then() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    let (report, output) = run_naming_at(&paths, &["20260809-052600"], "00:03", "Alice");

    assert_eq!(report.named, 1, "{output}");
    assert_eq!(report.failed, 0, "{output}");
    assert!(output.contains("1 voice selected at 00:03"), "{output}");

    // 00:03 is cluster 1's turn, and it is that whole voice that gets enrolled.
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.speakers.len(), 1);
    assert_eq!(speakers.speakers[0].name, "Alice");
    assert_eq!(speakers.speakers[0].embedding, voice(1));

    assert_eq!(
        said(&transcript_of(&session)),
        [
            ("Unknown 1", "  hi there  ", None),
            ("You", "morning", None),
            ("Alice", "and from me", Some(1.0)),
            ("Unknown 1", "let us start", None),
        ]
    );
    let md = std::fs::read_to_string(session.transcript_md()).unwrap();
    assert!(md.contains("**[00:03] Alice:** and from me"), "{md}");
    assert!(!md.contains("Unknown 2"), "{md}");
}

/// Acceptance criterion #2. Minutes are not wrapped at 60 on the way out, so `90:05` is what
/// the user has in front of them for a turn an hour and a half in -- and it has to be what
/// reaches that turn.
#[test]
fn a_timestamp_past_fifty_nine_minutes_reaches_its_turn() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_turns(
        &paths,
        &session,
        "20260809-052600",
        vec![
            speaker_turn(0.0, 0, "Unknown 1", "hi there"),
            speaker_turn(5405.0, 1, "Unknown 2", "still here"),
        ],
    );
    // The label that turn prints, which is what the user copies.
    let md = std::fs::read_to_string(session.transcript_md()).unwrap();
    assert!(md.contains("**[90:05] Unknown 2:**"), "{md}");

    let (report, output) = run_naming_at(&paths, &["20260809-052600"], "90:05", "Alice");

    assert_eq!(report.named, 1, "{output}");
    assert_eq!(said(&transcript_of(&session))[1].0, "Alice", "{output}");
}

/// Acceptance criterion #3. Naming a voice renames every turn it spoke, which is what naming
/// a voice means everywhere else in this tool -- so the command says how far that reached
/// rather than leaving a user who pointed at one line to infer it.
#[test]
fn renaming_through_a_timestamp_reports_the_turns_and_the_speech_it_covered() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    // Cluster 0 speaks twice, a second each, and the moment pointed at is only one of them.
    let (report, output) = run_naming_at(&paths, &["20260809-052600"], "00:00", "Alice");
    assert_eq!(report.named, 1, "{output}");
    assert!(
        output.contains("renamed 2 turn(s), 2s of speech, to Alice"),
        "{output}"
    );

    // And when clustering split one person in two, the count covers both halves: the claim
    // is about what changed, not about the voice that was selected.
    let split_root = tempfile::tempdir().unwrap();
    let split_paths = Paths::new(split_root.path());
    let split = make_session(&split_paths, "20260809-052600");
    with_embeddings(&split, &[nearly(0.0), nearly(20.0)]);

    let (report, output) = run_naming_at(&split_paths, &["20260809-052600"], "00:00", "Alice");
    assert_eq!(report.named, 1, "{output}");
    assert!(
        output.contains("renamed 3 turn(s), 3s of speech, to Alice"),
        "naming one half of a split voice renames both: {output}"
    );

    // Answering a voice with the name it already reads as changes nothing, and says that
    // rather than reporting zero turns.
    let (report, output) = run_naming_at(&split_paths, &["20260809-052600"], "00:00", "Alice");
    assert_eq!(report.failed, 0, "{output}");
    assert!(
        output.contains("no turns changed: that voice already read as Alice"),
        "{output}"
    );
}

/// Acceptance criterion #4. Four ways a timestamp lands on nothing nameable, and each one
/// says which it was: only one of them is the user's mistake, and the others each suggest a
/// different next move.
#[test]
fn a_timestamp_that_lands_on_nothing_nameable_says_which_of_the_four_it_was() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    // The fixture's turns are 0-1 and 3-5 on the speaker track with the mic at 1-2, so it
    // already has a hole and an end.
    for (at, expected) in [
        ("00:01", "is on the microphone track"),
        ("00:02", "nobody was speaking at 00:02"),
        (
            "00:30",
            "is past the end of this session, which ends at 00:05",
        ),
    ] {
        let (report, output) = run_naming_at(&paths, &["20260809-052600"], at, "Alice");
        assert_eq!(report.failed, 1, "{at}: {output}");
        assert_eq!(report.named, 0, "{at}: {output}");
        assert!(output.contains(expected), "{at}: {output}");
    }
    // The silence line hands back the nearest turn, because a miss here is usually a second
    // or two off and the right timestamp is on the page the user is reading.
    let (_, output) = run_naming_at(&paths, &["20260809-052600"], "00:02", "Alice");
    assert!(
        output.contains("the nearest turn is You at 00:01"),
        "{output}"
    );

    // The fourth: a transcript whose speech belongs to no cluster at all, which is what
    // diarization finding no voices leaves behind.
    let bare_root = tempfile::tempdir().unwrap();
    let bare_paths = Paths::new(bare_root.path());
    let bare = make_session(&bare_paths, "20260809-052600");
    with_turns(
        &bare_paths,
        &bare,
        "20260809-052600",
        vec![Turn {
            speaker: unknown_speaker(1),
            start: 0.0,
            end: 4.0,
            text: "hi there".to_string(),
            source_track: SourceTrack::Speaker,
            cluster: None,
            speaker_id_confidence: None,
        }],
    );

    let (report, output) = run_naming_at(&bare_paths, &["20260809-052600"], "00:00", "Alice");
    assert_eq!(report.failed, 1, "{output}");
    assert!(
        output.contains("the turn at 00:00 records no voice"),
        "{output}"
    );
}

/// Acceptance criterion #5. What an answer writes is the other axis entirely, and pointing
/// at a timestamp does not touch it: the reference floor applies exactly as it does to the
/// queue, and `--force-reference` overrides it exactly as it does there.
#[test]
fn a_timestamp_follows_the_same_reference_floor_and_the_same_override() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_speech_seconds(&session, &[40.0, 1.5]);

    let (report, output) = run_naming_at(&paths, &["20260809-052600"], "00:03", "Silas");
    assert_eq!(report.named, 1, "{output}");
    assert_eq!(report.session_only, 1, "{output}");
    assert!(
        EnrolledSpeakers::read_or_empty(&paths)
            .unwrap()
            .speakers
            .is_empty(),
        "a voice this quiet must not reach the database however it was selected: {output}"
    );
    assert_eq!(
        assigned_in(&session, "20260809-052600")
            .names
            .iter()
            .map(|row| (row.cluster, row.name.as_str()))
            .collect::<Vec<_>>(),
        [(1, "Silas")]
    );

    // ... and the override the line above advertises writes the reference the floor withheld.
    let forced_root = tempfile::tempdir().unwrap();
    let forced_paths = Paths::new(forced_root.path());
    let forced = make_session(&forced_paths, "20260809-052600");
    with_speech_seconds(&forced, &[40.0, 1.5]);

    let (report, output) = run_over(
        &forced_paths,
        &["20260809-052600"],
        Some(Selection::At("00:03".parse().unwrap())),
        Offer::default(),
        Sessions::default(),
        Enrolment::Always,
        &mut GivenName::new("Silas"),
    );

    assert_eq!(report.session_only, 0, "{output}");
    let speakers = EnrolledSpeakers::read_or_empty(&forced_paths).unwrap();
    assert_eq!(speakers.speakers.len(), 1);
    assert_eq!(speakers.speakers[0].name, "Silas");
    assert_eq!(speakers.speakers[0].embedding, voice(1));
    assert!(!forced.speaker_names_json().exists(), "{output}");
}

/// Acceptance criterion #6, both halves: naming somebody already enrolled adds a recording
/// of them rather than replacing one, and an answer that would take a name off a voice the
/// user was not pointing at is refused. The safeguards are downstream of the selection, so a
/// timestamp reaches exactly the same ones.
#[test]
fn a_name_that_already_exists_is_reused_and_never_taken_off_another_voice() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    // Alice, enrolled from a voice that matches neither cluster here.
    enrolled(&[("Alice", voice(3))], &paths);

    let (report, output) = run_naming_at(&paths, &["20260809-052600"], "00:03", "Alice");

    assert_eq!(report.named, 1, "{output}");
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(
        speakers
            .speakers
            .iter()
            .map(|s| (s.name.as_str(), s.embedding.as_slice()))
            .collect::<Vec<_>>(),
        [
            ("Alice", voice(3).as_slice()),
            ("Alice", voice(1).as_slice())
        ],
        "the first recording must survive the second: {output}"
    );

    // The refusal: cluster 1 reads Bob, and naming its near neighbour Alice would move that
    // name off it. Nothing is written and the voice keeps what it read.
    let taken_root = tempfile::tempdir().unwrap();
    let taken_paths = Paths::new(taken_root.path());
    let taken = make_session(&taken_paths, "20260809-052600");
    with_embeddings(&taken, &[nearly(0.0), nearly(20.0)]);
    enrolled(&[("Bob", nearly(60.0))], &taken_paths);
    let before = std::fs::read(taken_paths.speakers_json()).unwrap();

    let (report, output) = run_naming_at(&taken_paths, &["20260809-052600"], "00:00", "Alice");

    assert_eq!(report.refused, 1, "{output}");
    assert_eq!(report.named, 0, "{output}");
    assert!(
        output.contains("refused Alice for Unknown 1: it would take Bob off Unknown 2"),
        "{output}"
    );
    assert_eq!(
        std::fs::read(taken_paths.speakers_json()).unwrap(),
        before,
        "a refused answer writes nothing"
    );
    assert_eq!(said(&transcript_of(&taken))[2].0, "Bob", "{output}");
}

/// Two turns a fraction of a second apart print the same label, and then the timestamp names
/// neither voice on its own. That is a question this command cannot answer for the user, so
/// it hands back what tells them apart -- exactly as an ambiguous `--voice` does.
#[test]
fn a_label_two_voices_share_is_refused_with_the_numbers_that_split_them() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_turns(
        &paths,
        &session,
        "20260809-052600",
        vec![
            speaker_turn(10.1, 0, "Unknown 1", "one word"),
            speaker_turn(10.6, 1, "Unknown 2", "another"),
        ],
    );

    let (report, output) = run_naming_at(&paths, &["20260809-052600"], "00:10", "Alice");

    assert_eq!(report.failed, 1, "{output}");
    assert_eq!(report.named, 0, "{output}");
    assert!(output.contains("is the label of 2 turns"), "{output}");
    assert!(output.contains("--voice \"Unknown 1\""), "{output}");
    assert!(output.contains("--voice \"Unknown 2\""), "{output}");
}

/// A timestamp is an offset into one recording, so it lands somewhere different in each of
/// several -- refused before anything is read, like `--voice`, and with the reason that
/// belongs to the flag that was passed.
#[test]
fn a_timestamp_without_exactly_one_session_id_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    make_session(&paths, "20260809-052700");

    for ids in [&[][..], &["20260809-052600", "20260809-052700"][..]] {
        let (report, output) = run_naming_at(&paths, ids, "00:03", "Alice");
        assert_eq!(report.failed, 1, "{ids:?}: {output}");
        assert!(
            output.contains("--at needs exactly one session id"),
            "{ids:?}: {output}"
        );
        assert!(
            output.contains("offset into one recording"),
            "the reason has to be the one that belongs to --at: {ids:?}: {output}"
        );
    }
}

/// A name supplied up front is never shown the voice it lands on, so a queue would put one
/// name on everybody in it. Refused in the library, which is the only place that can see both
/// the answerer and the selection.
#[test]
fn a_name_given_up_front_without_a_selector_is_refused() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    let before = files_under(root.path());

    let (report, output) = run_over(
        &paths,
        &["20260809-052600"],
        None,
        Offer::default(),
        Sessions::default(),
        Enrolment::default(),
        &mut GivenName::new("Alice"),
    );

    assert_eq!(report.failed, 1, "{output}");
    assert_eq!(report.named, 0, "{output}");
    assert!(output.contains("--name needs a voice"), "{output}");
    assert_eq!(files_under(root.path()), before, "{output}");
}
