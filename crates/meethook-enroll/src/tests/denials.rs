//! The reject answer: what denying a tentative guess writes, and what it says as it does so.
//!
//! The fixtures everywhere below are the shape the band leaves behind once the regulars are
//! enrolled: a main voice the strict pass identifies, and a below-floor fragment the tentative
//! band marks as a guess. That pair is what the seam between "meethook guessed this" and "the
//! user decided about it" is for.

use super::*;
use meethook_session::DeniedName;
use meethook_transcribe::resolve_denials;

/// The fixture every test here starts from: `id`'s main voice is enrolled as Ivan strictly,
/// and its fragment sits at cosine distance 0.41 from his reference -- past the strict cut,
/// inside the tentative window -- so it reads "Ivan?" until somebody decides about it.
fn guessed_session(paths: &Paths, id: &str) -> SessionPaths {
    let session = make_session(paths, id);
    with_speech_seconds(&session, &[40.0, 1.5]);
    with_embeddings(&session, &[nearly(0.0), nearly(54.0)]);
    enrolled(&[("Ivan", voice(0))], paths);
    session
}

/// Denying a guess commits through the same preview and fixed-order writes a naming takes:
/// the suppression row lands in `speaker_names.json`, the database is untouched, and the
/// transcript moves the guess back to the number its turns were written with.
#[test]
fn denying_a_guess_writes_the_denial_and_demotes_the_label() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = guessed_session(&paths, "20260809-052600");
    let id = SessionId::parse("20260809-052600").unwrap();

    // What the database holds before the answer, so the run cannot quietly change it.
    let speakers_before = EnrolledSpeakers::read_or_empty(&paths).unwrap();

    let mut interviewer = Scripted::answering(vec![Answer::Deny {
        name: "Ivan".to_string(),
    }]);
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    // The narration, whole and in order: the bring-up-to-date, the queue, and the denial's
    // own line saying both halves of the demotion plus the durable half.
    assert_eq!(
        output,
        "20260809-052600  transcript brought up to date\n\
         20260809-052600  1 unresolved voice(s)\n\
         20260809-052600  not Ivan: Ivan? reads Unknown 2 again -- meethook will not guess \
         Ivan for this voice again\n"
    );
    assert_eq!(interviewer.labels(), ["Ivan?"], "{output}");
    assert_eq!(report.denied, 1, "{output}");
    assert_eq!(report.named, 0, "{output}");
    assert_eq!(report.refused, 0, "{output}");

    // The row itself: the cluster, the name as displayed, and the embedding the denial is
    // resolved through -- the handle, not the number.
    let names = assigned_in(&session, "20260809-052600");
    assert_eq!(
        names.denied,
        vec![DeniedName {
            cluster: 1,
            name: "Ivan".to_string(),
            embedding: nearly(54.0),
        }]
    );
    // A denial records the opposite of a naming: no assignment row rides along with it.
    assert!(names.names.is_empty());

    // The database is the one file a denial never touches.
    let speakers_after = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers_before.speakers, speakers_after.speakers);

    // The transcript moved the guess back to its number, and only the guess: the main keeps
    // its identification and its confidence, and the fragment's confidence goes with the
    // mark, because a number is not a similarity.
    let transcript = transcript_of(&session);
    let written: Vec<(&str, Option<f32>)> = transcript
        .turns
        .iter()
        .filter(|t| t.source_track == SourceTrack::Speaker)
        .map(|t| (t.speaker.as_str(), t.speaker_id_confidence))
        .collect();
    assert_eq!(written.len(), 3);
    assert_eq!(written[0].0, "Ivan");
    assert!(written[0].1.is_some());
    assert_eq!(written[1], ("Unknown 2", None));
    assert_eq!(written[2].0, "Ivan");
    assert!(written[2].1.is_some());
    let _ = id;
}

/// A denial is a decision, and decisions end the question: the cluster is committed, so the
/// same run offers the denied guess exactly once even though `--all` put it in the queue.
#[test]
fn a_denied_guess_is_not_offered_again_in_the_same_run() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_fragmented_session(&paths, "20260809-052600");
    // Main identified, first fragment guessed, the other two open fragments far from anyone.
    with_embeddings(&session, &[nearly(0.0), nearly(54.0), voice(2), voice(3)]);
    enrolled(&[("Ivan", voice(0))], &paths);

    let mut interviewer = Scripted::answering(vec![
        Answer::Deny {
            name: "Ivan".to_string(),
        },
        Answer::Skip,
        Answer::Skip,
    ]);
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    // All three voices were offered -- the denial did not hide the others -- and each was
    // offered exactly once, in first-appearance order.
    assert_eq!(
        interviewer.labels(),
        ["Ivan?", "Unknown 3", "Unknown 4"],
        "{output}"
    );
    assert_eq!(report.denied, 1, "{output}");
    assert_eq!(report.skipped, 2, "{output}");
}

/// Acceptance criterion #4: a rejected voice is out of the left-behind count, because
/// `left_unanswered` excludes the committed and the denial commits. Counting it would have
/// the run report a voice it spoke for as one it left behind.
#[test]
fn a_denied_guess_is_excluded_from_the_left_behind_count() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_fragmented_session(&paths, "20260809-052600");
    with_embeddings(&session, &[nearly(0.0), nearly(54.0), voice(2), voice(3)]);
    enrolled(&[("Ivan", voice(0))], &paths);

    let mut interviewer = Scripted::answering(vec![
        Answer::Deny {
            name: "Ivan".to_string(),
        },
        Answer::Leave,
    ]);
    let (report, output) = run_asking(&paths, &[], ALL, &mut interviewer);

    // Two voices left as they were: the one the leave was pressed on and the one after it.
    // The denied guess is neither -- it was answered, and its answer wrote something.
    assert!(
        output.contains("20260809-052600  left early, 2 voice(s) left as they were"),
        "{output}"
    );
    assert_eq!(report.denied, 1, "{output}");
    assert_eq!(report.skipped, 2, "{output}");
}

/// The durable half: a fresh default run over the denied session finds the fragment reading
/// its number again -- the guess suppressed by the standing row -- and passes the session
/// over counting the fragment as settled, asking nothing.
#[test]
fn a_second_run_honors_the_denial_and_passes_the_session_over() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = guessed_session(&paths, "20260809-052600");

    let mut denying = Scripted::answering(vec![Answer::Deny {
        name: "Ivan".to_string(),
    }]);
    let (_, output) = run_asking(&paths, &[], ALL, &mut denying);
    assert_eq!(denying.labels(), ["Ivan?"], "{output}");

    // And the guess is gone from the file, where the user will look for it.
    let markdown = std::fs::read_to_string(session.transcript_md()).unwrap();
    assert!(!markdown.contains("Ivan?"), "{markdown}");
    assert!(markdown.contains("Unknown 2"), "{markdown}");

    // Through the rule `transcribe --force` itself applies: the strict pass, then the band
    // over its image with the standing denial resolved, then the tier rule both processes
    // share. The fragment comes back as its number, not the guess -- which is what makes the
    // demotion durable rather than a rewrite of one file.
    let clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    let identified = identify_clusters(&clusters.clusters, &speakers);
    let tentative = tentative_identifications(&clusters.clusters, &speakers, &identified);
    let names = assigned_in(&session, "20260809-052600");
    let unknown = unknown_labels(
        clusters
            .clusters
            .iter()
            .map(|c| (c.id, c.first_spoke_seconds)),
    );
    let labels = attributions(
        &unknown,
        Naming::new(&clusters.clusters, &identified, &names.names)
            .with_tentative(&tentative)
            .with_denials(&resolve_denials(&clusters.clusters, &names.denied)),
    );
    assert_eq!(labels[&1].label(), "Unknown 2", "{output}");

    // And the next run has nothing left to ask: the fragment is settled by its denial even
    // though it now reads as an ordinary unknown.
    let mut second = Scripted::default();
    let (report, output) = run(&paths, &[], &mut second);
    assert!(second.seen.is_empty(), "{output}");
    assert_eq!(report.passed_over, 1, "{output}");
    assert_eq!(report.denied, 0, "{output}");
    assert!(
        output.contains(
            "20260809-052600  passed over: nothing unresolved \
             (1 named voice(s), 1 guessed or dismissed -- meethook enroll --all)"
        ),
        "{output}"
    );
}

/// Re-denying the same guess through `--all` is a no-op the second time around: the row
/// already stands, so every file stays byte-identical, and the run still says what it did.
#[test]
fn re_denying_a_denied_guess_leaves_every_file_byte_identical() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    guessed_session(&paths, "20260809-052600");

    let mut first = Scripted::answering(vec![Answer::Deny {
        name: "Ivan".to_string(),
    }]);
    let (_, output) = run_asking(&paths, &[], ALL, &mut first);
    assert_eq!(first.labels(), ["Ivan?"], "{output}");
    let before = files_under(root.path());

    // The fragment now reads its number, and reaching it takes the same widening flag.
    let mut again = Scripted::answering(vec![Answer::Deny {
        name: "Ivan".to_string(),
    }]);
    let (report, output) = run_asking(&paths, &[], ALL, &mut again);
    assert_eq!(again.labels(), ["Unknown 2"], "{output}");
    assert_eq!(report.denied, 1, "{output}");
    assert!(
        output.contains(
            "20260809-052600  not Ivan: Unknown 2 reads Unknown 2 again -- meethook will \
             not guess Ivan for this voice again"
        ),
        "{output}"
    );
    assert_eq!(before, files_under(root.path()));
}

/// The outcome a preview reported is the outcome the write produced, for the denial's own
/// shape: the demotion measured over the dry run's clones is what lands in the transcript,
/// and the row the candidate state carries is what stands in `speaker_names.json`. Agreement
/// is structural -- the commit takes the copies the dry run built -- so what this pins is
/// that a later refactor cannot go back to deriving the two separately.
#[test]
fn a_preview_of_a_denial_is_what_the_answer_then_writes() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = guessed_session(&paths, "20260809-052600");

    // An answerer that previews the denial through the seam before answering it and keeps
    // the consequence: no `Paths`, no database, no session directory -- the same claim the
    // naming agreement test makes, which is what makes this the standing drift check for
    // the denial's write path too.
    struct Denying {
        name: String,
        preview: Option<Consequence>,
    }
    impl Interviewer for Denying {
        fn identify(&mut self, voice: &Voice<'_>) -> Answer {
            if self.preview.is_none() {
                self.preview = Some(voice.preview.deny_to(&self.name));
            }
            Answer::Deny {
                name: self.name.clone(),
            }
        }
    }
    let mut interviewer = Denying {
        name: "Ivan".to_string(),
        preview: None,
    };
    let (report, output) = run_over(
        &paths,
        &[],
        None,
        ALL,
        Sessions::Unresolved,
        Enrolment::default(),
        &mut interviewer,
    );
    let preview = interviewer
        .preview
        .as_ref()
        .expect("the one voice was shown")
        .clone();

    // Nothing about the denial can be refused or stored, so the whole of its consequence is
    // the demotion -- and the demotion says exactly what the transcript now reads.
    assert_eq!(preview.refused, None, "{output}");
    assert_eq!(preview.stored, None, "{output}");
    assert!(preview.displaced.is_empty(), "{output}");
    assert!(preview.stale.is_empty(), "{output}");
    let Demotion { from, to } = preview
        .demoted
        .expect("a denial always carries its demotion");
    assert_eq!(from.as_str(), "Ivan?", "{output}");

    let transcript = transcript_of(&session);
    let written: Vec<&str> = transcript
        .turns
        .iter()
        .filter(|t| t.source_track == SourceTrack::Speaker)
        .map(|t| t.speaker.as_str())
        .collect();
    assert_eq!(written, ["Ivan", "Unknown 2", "Ivan"], "{output}");
    assert_eq!(to.as_str(), written[1], "{output}");

    // And the row: what the candidate state carried is what the file holds, bit for bit.
    let names = assigned_in(&session, "20260809-052600");
    assert_eq!(names.denied, preview.assigned.denied, "{output}");
    assert_eq!(report.denied, 1, "{output}");
}

/// A denial whose embedding matches no cluster resolves to nothing: re-clustering moved the
/// voice the claim was about, so the row stays in the file untouched but settles no
/// question -- the fragment is open again, held back by the floor like any open quiet voice
/// rather than counted as spoken for.
#[test]
fn a_stale_denial_resolves_to_nothing_and_the_fragment_is_open_again() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    let id = SessionId::parse("20260809-052600").unwrap();

    // Main identified strictly, fragment far from anyone -- outside the tentative window, so
    // no guess can settle it either -- and a third voice open above the floor, so the offer
    // does not fall back and the floor alone decides the fragment's fate.
    let mut clusters = vec![
        cluster(0, 0.0, (0.5, 2.5)),
        cluster(1, 3.0, (3.0, 3.5)),
        cluster(2, 6.0, (6.0, 7.0)),
    ];
    clusters[0].embedding = nearly(0.0);
    clusters[1].embedding = voice(2);
    clusters[2].embedding = voice(3);
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
    enrolled(&[("Ivan", voice(0))], &paths);

    // The stale row: the right cluster and the right name, but an embedding no cluster
    // holds, so the resolution's exact match finds nothing rather than another voice.
    let mut names = SpeakerNames::read_or_empty(&session, &id).unwrap();
    names.deny(1, "Ivan", &[0.5f32; 4]);
    names.write(&session).unwrap();

    let mut interviewer = Scripted::default();
    let (report, output) = run(&paths, &[], &mut interviewer);

    // The fragment is open: offered nothing, held back, and advertised with `--all` -- the
    // exact reading a session with no denial at all would give, which is the point of the
    // staleness rule: the claim was about a voice that no longer exists.
    assert_eq!(interviewer.labels(), ["Unknown 3"], "{output}");
    assert_eq!(report.held_back, 1, "{output}");
    assert_eq!(report.denied, 0, "{output}");
    assert!(
        output.contains("1 quieter voice(s) not offered -- meethook enroll --all"),
        "{output}"
    );
    // Dropped at resolution, not deleted: the row still stands in the file.
    let names = assigned_in(&session, "20260809-052600");
    assert_eq!(names.denied.len(), 1, "{output}");
}
