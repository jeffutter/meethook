//! naming a voice, what a rewrite writes, skips, and the session-failure guards.

use super::*;

/// Acceptance criteria #5 and #6, at the level a user meets them: one answer puts a
/// person in the database and their name on their own turns, and on nobody else's.
///
/// It also pins the thing a rename must never do, which is change the *shape* of a
/// transcript it did not write. The root carries a template here, in place before the
/// session is, so the rewrite below has something other than the built-in default to revert
/// to if it ever resolved the template from anywhere but the root.
#[test]
fn naming_a_voice_enrolls_them_and_rewrites_that_sessions_transcript() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    std::fs::create_dir_all(paths.root()).unwrap();
    std::fs::write(paths.transcript_template(), USER_TEMPLATE).unwrap();
    let session = make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.speakers.len(), 1);
    assert_eq!(speakers.speakers[0].name, "Alice");
    assert_eq!(speakers.speakers[0].embedding, voice(0));

    assert_eq!(
        said(&transcript_of(&session)),
        [
            ("Alice", "  hi there  ", Some(1.0)),
            ("You", "morning", None),
            ("Unknown 2", "and from me", None),
            ("Alice", "let us start", Some(1.0)),
        ]
    );
    // The rendering is rewritten from the turns, not patched line by line.
    let markdown = std::fs::read_to_string(session.transcript_md()).unwrap();
    assert_eq!(
        markdown,
        transcript_of(&session)
            .render_markdown(
                &TranscriptTemplate::resolve(&paths, None).unwrap(),
                &TranscriptContext::now(&session_metadata(
                    &SessionId::parse("20260809-052600").unwrap()
                )),
            )
            .unwrap()
    );
    assert!(markdown.contains("Alice"), "{markdown}");
    assert!(!markdown.contains("Unknown 1"), "{markdown}");
    // Acceptance criterion #5: the rewrite went through the root's template, not the
    // built-in default. Both marks, because either alone would pass on a default rendering
    // that happened to be a prefix or a suffix of this one.
    assert!(
        markdown.starts_with("---\nvault: mine\n---\n"),
        "{markdown}"
    );
    assert!(markdown.contains("Alice> let us start\n"), "{markdown}");
    assert!(!markdown.contains("**["), "{markdown}");
    // The captions are rewritten by the same call, so they cannot be left naming a voice
    // the transcript beside them no longer calls a stranger. The user's template has no
    // say here: `transcript.vtt` is a machine format.
    let vtt = std::fs::read_to_string(session.transcript_vtt()).unwrap();
    assert_eq!(vtt, transcript_of(&session).render_vtt());
    assert!(vtt.contains("<v Alice>let us start\n"), "{vtt}");
    assert!(!vtt.contains("Unknown 1"), "{vtt}");
}

/// A template that is nothing like the built-in default in either half: different
/// frontmatter, and a body line no default rendering could produce.
const USER_TEMPLATE: &str = "---\nvault: mine\n---\n\
    {% for turn in turns %}{{ turn.speaker }}> {{ turn.text }}\n{% endfor %}";

/// Acceptance criterion #6's actual claim, which the assertion above only illustrates:
/// the rewritten transcript is what `transcribe --force` would now produce. Checked by
/// deriving the labels the way `merge` does -- `unknown_labels` over the clusters,
/// `identify_clusters` against the database -- rather than by restating the expected
/// strings, so the two paths cannot drift without this failing.
#[test]
fn the_rewritten_transcript_is_what_a_force_re_transcribe_would_produce() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![named("Alice"), named("Bob")]);
    run(&paths, &[], &mut interviewer);

    let clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    let identified = identify_clusters(&clusters.clusters, &speakers);
    let unknown = unknown_labels(
        clusters
            .clusters
            .iter()
            .map(|c| (c.id, c.first_spoke_seconds)),
    );
    // The transcript's speaker turns, in order, are cluster 0, 1, 0.
    let expected: Vec<(String, Option<f32>)> = [0u32, 1, 0]
        .iter()
        .map(|id| match identified.get(id) {
            Some(who) => (who.name.clone(), Some(who.similarity)),
            None => (unknown[id].clone(), None),
        })
        .collect();

    let written: Vec<(String, Option<f32>)> = transcript_of(&session)
        .turns
        .iter()
        .filter(|t| t.source_track == SourceTrack::Speaker)
        .map(|t| (t.speaker.clone(), t.speaker_id_confidence))
        .collect();
    assert_eq!(written, expected);
}

/// The same invariant where the tentative band has something to say: the fragment sits at
/// cosine distance 0.41 from Alice -- past the strict cut, inside the tentative window -- so
/// neither process may name it unmarked, and whatever the pass-over relabel wrote must be
/// what a `--force` re-transcribe would now derive through the same two passes.
#[test]
fn the_rewritten_transcript_of_a_tentative_session_is_what_a_force_re_transcribe_would_produce() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    let id = SessionId::parse("20260809-052600").unwrap();

    // Alice is enrolled against cluster 0's exact voice before the run, which is what puts
    // her in the in-session pool the band may guess from.
    let mut speakers = EnrolledSpeakers::new(Vec::new());
    speakers.store_reference("Alice", voice(0), 40.0);
    speakers.write(&paths).unwrap();

    // The third voice exists only so the fragment is not the sole unresolved one: a floor that
    // hides every voice in a session offers them all instead, and then nothing is held back
    // for this test to pin. Above the floor and far from everything, it is asked about once
    // and skipped.
    let mut clusters = vec![
        cluster(0, 0.0, (0.5, 2.5)),
        cluster(1, 3.0, (3.0, 3.5)),
        cluster(2, 6.0, (6.0, 7.0)),
    ];
    clusters[1].embedding = nearly(54.0);
    for (cluster, seconds) in clusters.iter_mut().zip([40.0, 1.5, 8.0]) {
        cluster.speech_seconds = seconds;
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

    // One question, about the voice above the floor; the fragment is held back, not asked.
    let mut interviewer = Scripted::default();
    let (report, output) = run(&paths, &[], &mut interviewer);
    assert_eq!(interviewer.seen.len(), 1, "{output}");
    assert_eq!(report.held_back, 1, "{output}");

    // What `transcribe --force` would now derive: the strict pass, then the band over its
    // image, then the tier rule both processes share.
    let clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    let identified = identify_clusters(&clusters.clusters, &speakers);
    let tentative = tentative_identifications(&clusters.clusters, &speakers, &identified);
    let unknown = unknown_labels(
        clusters
            .clusters
            .iter()
            .map(|c| (c.id, c.first_spoke_seconds)),
    );
    let labels = attributions(
        &unknown,
        Naming::new(&clusters.clusters, &identified, &[]).with_tentative(&tentative),
    );
    let expected: Vec<(String, Option<f32>)> = [0u32, 1, 2]
        .iter()
        .map(|id| {
            let label = &labels[id];
            (label.label().to_string(), label.confidence())
        })
        .collect();
    let written: Vec<(String, Option<f32>)> = transcript_of(&session)
        .turns
        .iter()
        .filter(|t| t.source_track == SourceTrack::Speaker)
        .map(|t| (t.speaker.clone(), t.speaker_id_confidence))
        .collect();
    assert_eq!(written, expected);
    // And the guess is marked, not asserted: the band's whole cost model is that a wrong
    // one costs a visibly-questionable line rather than an unmarked misfiling.
    let markdown = std::fs::read_to_string(session.transcript_md()).unwrap();
    assert!(markdown.contains("Alice?"), "{markdown}");
}

/// The guard on `merge` staying the sole producer of a turn's provenance: `enroll` changes
/// what a cluster is called and never which cluster a turn came from. That is what keeps
/// a rewritten transcript identical to a `--force` re-transcribe, since the field would
/// otherwise be one `enroll` could drift.
#[test]
fn a_rewrite_leaves_every_turns_cluster_exactly_as_it_was() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    let before: Vec<Option<u32>> = transcript_of(&session)
        .turns
        .iter()
        .map(|t| t.cluster)
        .collect();

    let mut interviewer = Scripted::answering(vec![named("Alice"), named("Bob")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 2, "{output}");
    let after: Vec<Option<u32>> = transcript_of(&session)
        .turns
        .iter()
        .map(|t| t.cluster)
        .collect();
    assert_eq!(after, before);
    assert_eq!(before, [Some(0), None, Some(1), Some(0)]);
}

/// The compatibility decision on `TRANSCRIPT_SCHEMA_VERSION`, at the level a user meets
/// it: a transcript written before turns recorded their cluster is refused rather than
/// read with that provenance fabricated, it says how to fix it, and the session after it
/// is still asked about.
#[test]
fn a_transcript_without_clusters_fails_its_session_without_ending_the_queue() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let stale = make_session(&paths, "20260809-052600");
    make_session(&paths, "20260809-052700");
    std::fs::write(
        stale.transcript_json(),
        br#"{
          "schema_version": 1,
          "session_id": "20260809-052600",
          "turns": [
            {
              "speaker": "Unknown 1",
              "start": 0.0,
              "end": 1.0,
              "text": "hi there",
              "source_track": "speaker",
              "speaker_id_confidence": null
            }
          ]
        }"#,
    )
    .unwrap();

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.failed, 1, "{output}");
    assert!(output.contains("--force"), "{output}");
    assert_eq!(report.named, 1, "{output}");
    for voice in &interviewer.seen {
        assert_eq!(voice.session, "20260809-052700", "{voice:?}");
    }
}

/// Acceptance criterion #7: a skip changes nothing, and "nothing" is byte-for-byte. A
/// rewrite that happened to produce equivalent turns would still churn the files.
#[test]
fn skipping_every_voice_leaves_the_files_byte_identical() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    let before = (
        std::fs::read(session.transcript_json()).unwrap(),
        std::fs::read(session.transcript_md()).unwrap(),
        std::fs::read(session.speaker_clusters_json()).unwrap(),
    );

    let mut interviewer = Scripted::answering(vec![Answer::Skip, Answer::Skip]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.skipped, 2, "{output}");
    assert_eq!(report.named, 0);
    assert_eq!(
        (
            std::fs::read(session.transcript_json()).unwrap(),
            std::fs::read(session.transcript_md()).unwrap(),
            std::fs::read(session.speaker_clusters_json()).unwrap(),
        ),
        before
    );
    assert!(
        !paths.speakers_json().exists(),
        "a run that named nobody must not create a database"
    );
    assert!(
        !session.speaker_names_json().exists(),
        "a run that named nobody must not create a names file either"
    );
}

/// Acceptance criterion #4, and the boundary the clusters file exists to defend: enroll
/// reads it and never writes it, so nothing here can start depending on a name being in
/// there.
#[test]
fn a_run_that_names_everybody_still_leaves_the_clusters_file_untouched() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    let before = std::fs::read(session.speaker_clusters_json()).unwrap();

    let mut interviewer = Scripted::answering(vec![named("Alice"), named("Bob")]);
    run(&paths, &[], &mut interviewer);

    assert_eq!(
        std::fs::read(session.speaker_clusters_json()).unwrap(),
        before
    );
}

/// Acceptance criterion #1, and the deduplication rule: the same person in two sessions is
/// asked about once, because the second session identifies them from the answer given in
/// the first. Sessions are worked through in id order.
#[test]
fn a_person_named_in_one_session_is_matched_rather_than_asked_about_again() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let first = make_session(&paths, "20260809-052600");
    let second = make_session(&paths, "20260809-052700");

    // One name, then skips: whoever is asked about after Alice is somebody else.
    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    let sessions: Vec<&str> = interviewer
        .seen
        .iter()
        .map(|v| v.session.as_str())
        .collect();
    assert_eq!(
        sessions,
        ["20260809-052600", "20260809-052600", "20260809-052700"],
        "expected both voices of the first session, then the second session's other voice"
    );
    assert_eq!(
        interviewer.labels(),
        ["Unknown 1", "Unknown 2", "Unknown 2"],
        "the second session's Alice must not be asked about again"
    );

    // ...and her name reaches the second session's transcript anyway, on the way past.
    for session in [&first, &second] {
        assert_eq!(
            transcript_of(session).turns[0].speaker,
            "Alice",
            "in {}",
            session.dir().display()
        );
    }
}

/// Acceptance criterion #2: ids scope the run, and one that is not on disk is named
/// rather than quietly doing less than was asked.
#[test]
fn ids_scope_the_run_and_an_unknown_id_is_reported() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    let untouched = make_session(&paths, "20260809-052700");

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run(
        &paths,
        &["20260809-052600", "20260809-999999"],
        &mut interviewer,
    );

    assert!(output.contains("20260809-999999  not found"), "{output}");
    assert_eq!(report.failed, 1);
    assert_eq!(report.named, 1);
    for voice in &interviewer.seen {
        assert_eq!(voice.session, "20260809-052600", "{voice:?}");
    }
    assert_eq!(transcript_of(&untouched).turns[0].speaker, "Unknown 1");
}

/// Acceptance criterion #9: ending the run early keeps everything already answered. The
/// name given before the quit is on disk in both files, and nothing after it was asked.
#[test]
fn quitting_keeps_every_name_accepted_so_far() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let first = make_session(&paths, "20260809-052600");
    let second = make_session(&paths, "20260809-052700");

    let mut interviewer = Scripted::answering(vec![named("Alice"), Answer::Quit]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert_eq!(interviewer.seen.len(), 2, "{:?}", interviewer.seen);

    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.speakers.len(), 1);
    assert_eq!(transcript_of(&first).turns[0].speaker, "Alice");
    assert!(
        std::fs::read_to_string(first.transcript_md())
            .unwrap()
            .contains("Alice")
    );
    // The queue stopped where it was told to, rather than carrying on to the next session.
    assert_eq!(transcript_of(&second).turns[0].speaker, "Unknown 1");
}

/// A session transcribed by a build that did not record first appearances cannot be
/// mapped from "Unknown 2" back to a voice, so it is reported and counted -- and the
/// session after it is still asked about.
#[test]
fn a_stale_clusters_file_fails_its_session_without_ending_the_queue() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let stale = make_session(&paths, "20260809-052600");
    make_session(&paths, "20260809-052700");
    std::fs::write(
        stale.speaker_clusters_json(),
        br#"{
          "schema_version": 1,
          "session_id": "20260809-052600",
          "clusters": [
            {
              "id": 0,
              "embedding": [1.0, 0.0, 0.0, 0.0],
              "speech_seconds": 42.5,
              "representatives": [{ "start": 1.0, "end": 3.0 }]
            }
          ]
        }"#,
    )
    .unwrap();

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.failed, 1, "{output}");
    assert!(output.contains("--force"), "{output}");
    assert_eq!(report.named, 1, "{output}");
    for voice in &interviewer.seen {
        assert_eq!(voice.session, "20260809-052700", "{voice:?}");
    }
}

/// A blank answer is a skip, not an entry called "". Somebody pressing Enter with a stray
/// space in the buffer must not end up in the database.
#[test]
fn a_blank_name_is_a_skip_rather_than_an_empty_entry() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![named("   "), named("  Bob  ")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.skipped, 1, "{output}");
    assert_eq!(report.named, 1, "{output}");
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.speakers.len(), 1);
    // Trimmed, so the transcript does not read "**[00:03]   Bob  :**".
    assert_eq!(speakers.speakers[0].name, "Bob");
}

/// One person clustering split in two is named once and lands on both halves, because
/// that is what a `--force` re-transcribe would do with the reference this answer just
/// stored.
#[test]
fn naming_a_split_voice_names_its_other_half_without_asking_twice() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    // Two clusters a few degrees apart: one voice the clusterer did not join up.
    with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert_eq!(
        interviewer.labels(),
        ["Unknown 1"],
        "the second half of one voice must not be asked about"
    );
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.speakers.len(), 1);
    assert_eq!(
        said(&transcript_of(&session))
            .iter()
            .map(|(speaker, _, _)| *speaker)
            .collect::<Vec<_>>(),
        ["Alice", "You", "Alice", "Alice"]
    );
}

/// The transcript's schema version survives a rewrite: `enroll` edits turns, it does not
/// re-stamp the file as something it is not.
#[test]
fn a_rewritten_transcript_keeps_its_schema_version_and_session_id() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    run(&paths, &[], &mut interviewer);

    let transcript = transcript_of(&session);
    assert_eq!(transcript.schema_version, TRANSCRIPT_SCHEMA_VERSION);
    assert_eq!(transcript.session_id.as_str(), "20260809-052600");
}

/// An empty meethook directory is a first run, not an error.
#[test]
fn no_sessions_at_all_is_reported_rather_than_failing() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());

    let mut interviewer = Scripted::default();
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report, EnrollReport::default());
    assert!(output.contains("No sessions found"), "{output}");
}
