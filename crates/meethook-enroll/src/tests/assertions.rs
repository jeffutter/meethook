//! one-remote-speaker assertions and group consequences.

use super::*;

// --- The one-remote-speaker assertion ----------------------------------------------------

/// A session with `n` voices, each on its own orthogonal axis and first speaking in id
/// order, so "Unknown N" is the cluster with id N - 1. All but the last eleven clear the
/// reference floor at distinct lengths; those sit below it. The shape real clustering leaves
/// when one person is split into many fragments.
fn make_many_cluster_session(paths: &Paths, id: &str, n: usize) -> SessionPaths {
    let parsed = SessionId::parse(id).unwrap();
    let session = paths.session(&parsed);
    std::fs::create_dir_all(session.dir()).unwrap();
    let metadata = session_metadata(&parsed);
    metadata.write(&session.session_json()).unwrap();
    write_speaker_wav(&session.speaker_wav());

    let clusters: Vec<SpeakerCluster> = (0..n as u32)
        .map(|i| {
            let mut cluster = cluster(i, i as f64 * 0.1, (0.5, 2.5));
            cluster.embedding = axis(i as usize, n);
            cluster.speech_seconds = if (i as usize) < n - 11 {
                5.0 + i as f64 * 0.5
            } else {
                0.5 + (n - i as usize) as f64 * 0.1
            };
            cluster
        })
        .collect();
    SpeakerClusters::new(parsed.clone(), clusters)
        .write(&session)
        .unwrap();

    let turns: Vec<Turn> = (0..n as u32)
        .map(|i| speaker_turn(i as f64, i, &format!("Unknown {}", i + 1), "one word"))
        .collect();
    write_transcript(
        &Transcript::new(parsed.clone(), turns),
        paths,
        &session,
        &metadata,
    );
    session
}

/// `run_enroll` with the assertion half of the rules filled in, returning the result rather
/// than unwrapping it -- the interrupt test needs to see the failure.
fn run_asserting_raw(
    paths: &Paths,
    ids: &[&str],
    name: Option<&str>,
    interviewer: &mut dyn Interviewer,
) -> Result<(EnrollReport, String)> {
    let requested: Vec<SessionId> = ids.iter().map(|id| SessionId::parse(id).unwrap()).collect();
    let mut out = Vec::new();
    let report = run_enroll(
        paths,
        &requested,
        EnrollRules {
            selector: None,
            offer: Offer::default(),
            sessions: Sessions::Unresolved,
            enrolment: Enrolment::default(),
            one_speaker: name,
            relabel_transcript: true,
            template: &TranscriptTemplate::resolve(paths, None).unwrap(),
        },
        interviewer,
        &mut Lines::new(&mut out),
    )?;
    Ok((report, String::from_utf8(out).unwrap()))
}

/// `run_asserting_raw`, for the tests where the run is expected to come back whole.
fn run_asserting(
    paths: &Paths,
    ids: &[&str],
    name: Option<&str>,
    interviewer: &mut Scripted,
) -> (EnrollReport, String) {
    run_asserting_raw(paths, ids, name, interviewer).unwrap()
}

/// Acceptance criterion #1 and #2: the user asserts one remote speaker and gives that
/// person a name, and every voice on the track reads it afterwards -- the quiet ones
/// included, which no queue offers by default -- without anything being asked about any of
/// them.
#[test]
fn an_asserted_name_reaches_every_voice_including_the_quiet_ones() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_fragmented_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(Vec::new());
    let (report, output) = run_asserting(
        &paths,
        &["20260809-052600"],
        Some("Grace"),
        &mut interviewer,
    );

    assert!(
        interviewer.seen.is_empty(),
        "nothing may be asked under an assertion: {output}"
    );
    assert_eq!(report.asserted, 4, "{output}");
    assert_eq!(report.named, 4, "{output}");
    assert_eq!(report.session_only, 3, "{output}");

    // Every voice reads the asserted name, and the mic track is untouched.
    let transcript = transcript_of(&session);
    let said = said(&transcript);
    assert_eq!(
        said.iter().filter(|(who, _, _)| *who == "Grace").count(),
        4,
        "every speaker-track turn should read as the asserted person: {said:?}"
    );
    assert!(
        said.iter().any(|(who, _, _)| *who == SPEAKER_YOU),
        "the local speaker keeps their own label: {said:?}"
    );

    // The three quiet voices are named against the session alone; the loud one holds the
    // only reference.
    let assigned = assigned_in(&session, "20260809-052600");
    assert_eq!(assigned.names.len(), 3, "{:?}", assigned.names);
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.references("Grace"), 1, "{output}");
    assert!(output.contains("one remote speaker settled"), "{output}");
}

/// Acceptance criterion #3: the voices the heard-at-once veto would have refused are named
/// anyway, and each one is reported -- naming the voice it was heard at once with, which is
/// the evidence the veto acted on -- rather than silently overridden.
#[test]
fn the_heard_at_once_veto_is_overridden_and_reported_for_each_voice_it_reached() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    heard_at_once(&session, 0, 1);

    let mut interviewer = Scripted::answering(Vec::new());
    let (report, output) = run_asserting(
        &paths,
        &["20260809-052600"],
        Some("Grace"),
        &mut interviewer,
    );

    // The second voice was heard at once with the first, and that pair is what the veto
    // would have refused; the first was committed before the second, so exactly one veto
    // is overridden and it is the second voice that reports it.
    assert_eq!(report.vetoes_overridden, 1, "{output}");
    assert!(output.contains("named Grace for Unknown 2"), "{output}");
    assert!(output.contains("heard at once with Unknown 1"), "{output}");
    assert!(
        output.contains("the one-remote-speaker assertion says this track is one person"),
        "{output}"
    );
    // Both keep the name regardless.
    let transcript = transcript_of(&session);
    let said = said(&transcript);
    assert!(
        said.iter().filter(|(who, _, _)| *who == "Grace").count() == 3,
        "both voices keep the asserted name: {said:?}"
    );
    assert!(output.contains("1 veto(s) overridden"), "{output}");
}

/// Acceptance criterion #4, the plan's D4 rule made mechanical: a hundred and one above-
/// and-below-floor clusters do not become a hundred and one references. The existing cap
/// does the bounding, and the ten held are the ten longest above-floor clips -- the
/// selection is a stated rule, not a property of how many clusters the session happens to
/// hold.
#[test]
fn a_hundred_and_one_voice_session_stores_ten_references_the_ten_longest_above_the_floor() {
    const VOICES: usize = 101;
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_many_cluster_session(&paths, "20260820-140414", VOICES);

    let mut interviewer = Scripted::answering(Vec::new());
    let (report, output) = run_asserting(
        &paths,
        &["20260820-140414"],
        Some("Grace"),
        &mut interviewer,
    );

    assert_eq!(report.asserted, VOICES, "{output}");
    assert_eq!(report.session_only, 11, "{output}");
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(
        speakers.references("Grace"),
        MAX_REFERENCES_PER_SPEAKER,
        "{output}"
    );

    // The ten longest above-floor clips are ids 80..=89, at 45.0 s up to 49.5 s: everything
    // else held is shorter, so nothing else survives the cap.
    let held: Vec<Vec<f32>> = speakers
        .speakers
        .iter()
        .map(|s| s.embedding.clone())
        .collect();
    for i in 0..VOICES {
        let expected = (80..=89).contains(&(i as u32));
        assert_eq!(
            held.contains(&axis(i, VOICES)),
            expected,
            "voice {i} should be held iff it is among the ten longest: {output}"
        );
    }

    // And every voice, quiet included, reads the name in the transcript.
    let transcript = transcript_of(&session);
    let said = said(&transcript);
    assert_eq!(said.len(), VOICES);
    assert!(said.iter().all(|(who, _, _)| *who == "Grace"), "{output}");
    assert!(
        output.contains(&format!("{VOICES} voice(s) read as Grace")),
        "{output}"
    );
}

/// Acceptance criterion #7, first half: the fact lands on disk before the first per-voice
/// commit, so an interrupt between the two leaves a state that explains itself -- the
/// assertion present, nothing derived from it yet -- and a re-run converges onto the whole.
#[test]
fn an_interrupt_before_the_first_commit_leaves_the_assertion_on_disk_and_a_rerun_converges() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    // Make the database unwritable: the assertion itself lives in the session directory,
    // which stays writable, so it survives while the first commit cannot reach
    // `speakers.json`.
    std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o555)).unwrap();
    let interrupted = run_asserting_raw(
        &paths,
        &["20260809-052600"],
        Some("Grace"),
        &mut Scripted::default(),
    );
    std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(
        interrupted.is_err(),
        "the first commit must fail while the database is unwritable"
    );

    // What survived is self-consistent: the fact is on disk, and nothing derived from it
    // is.
    let metadata = SessionMetadata::read(&session.session_json()).unwrap();
    assert_eq!(metadata.one_remote_speaker.as_deref(), Some("Grace"));
    assert!(!session.speaker_names_json().exists());
    assert!(!paths.speakers_json().exists());
    let transcript = transcript_of(&session);
    let before = said(&transcript);
    assert!(
        before.iter().any(|(who, _, _)| *who == "Unknown 1"),
        "no label may have moved before the first commit: {before:?}"
    );

    // And a re-run converges onto the complete state.
    let (_, output) = run_asserting(
        &paths,
        &["20260809-052600"],
        Some("Grace"),
        &mut Scripted::default(),
    );
    let transcript = transcript_of(&session);
    let after = said(&transcript);
    assert!(
        after
            .iter()
            .all(|(who, _, _)| *who == "Grace" || *who == SPEAKER_YOU),
        "the re-run must complete what the interrupt left behind: {after:?}\n{output}"
    );
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.references("Grace"), 2, "{output}");
}

/// Acceptance criteria #6 and #7, second half: a re-run over the state a killed run would
/// have left -- the fact on disk, some voices already named, the transcript still carrying
/// the old labels -- converges onto the same state a fresh run produces, and a further
/// pass writes nothing at all.
#[test]
fn a_rerun_converges_from_a_partial_state_and_then_writes_nothing_new() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_fragmented_session(&paths, "20260809-052600");
    let id = SessionId::parse("20260809-052600").unwrap();

    // Two of four voices named, one reference stored, the transcript still unlabelled: a
    // run interrupted after its second commit.
    let mut metadata = SessionMetadata::read(&session.session_json()).unwrap();
    metadata.assert_one_remote_speaker("Grace".to_string());
    metadata.write(&session.session_json()).unwrap();
    let clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
    let mut names = SpeakerNames::read_or_empty(&session, &id).unwrap();
    names.assign(0, "Grace", clusters.clusters[0].embedding.clone());
    names.assign(1, "Grace", clusters.clusters[1].embedding.clone());
    names.write(&session).unwrap();
    enrolled(&[("Grace", clusters.clusters[0].embedding.clone())], &paths);

    let (report, output) = run_asserting(
        &paths,
        &["20260809-052600"],
        Some("Grace"),
        &mut Scripted::default(),
    );
    assert_eq!(report.asserted, 4, "{output}");
    let transcript = transcript_of(&session);
    let said = said(&transcript);
    assert!(
        said.iter()
            .all(|(who, _, _)| *who == "Grace" || *who == SPEAKER_YOU),
        "the re-run must complete the transcript: {said:?}\n{output}"
    );

    // A further pass is a no-op on disk: converged means byte-identical, not merely
    // equivalent.
    let before = files_under(root.path());
    let (_, output) = run_asserting(
        &paths,
        &["20260809-052600"],
        Some("Grace"),
        &mut Scripted::default(),
    );
    assert_eq!(
        files_under(root.path()),
        before,
        "a converged assertion rewrote a file: {output}"
    );
}

/// The displacement D4 states: references another name built from this very track are
/// withdrawn when the assertion names the track's one person, because the user has just
/// said the evidence belongs to somebody else.
#[test]
fn an_assertion_displaces_references_that_another_name_built_from_this_track() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    let clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
    // Both voices were previously enrolled as Bob from this track.
    enrolled(
        &[
            ("Bob", clusters.clusters[0].embedding.clone()),
            ("Bob", clusters.clusters[1].embedding.clone()),
        ],
        &paths,
    );

    let (report, output) = run_asserting(
        &paths,
        &["20260809-052600"],
        Some("Grace"),
        &mut Scripted::default(),
    );
    assert_eq!(report.asserted, 2, "{output}");

    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.references("Grace"), 2, "{output}");
    assert!(
        speakers.speakers.iter().all(|s| s.name == "Grace"),
        "Bob's evidence from this track is withdrawn: {:?}",
        speakers.speakers
    );
}

/// Acceptance criterion #9, across sessions: asserting one session's track leaves every
/// other session's files byte-identical.
#[test]
fn asserting_one_session_leaves_the_other_sessions_byte_identical() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let asserted = make_session(&paths, "20260809-052600");
    let bystander = make_session(&paths, "20260810-052600");

    let before = files_under(bystander.dir());
    let (_report, output) = run_asserting(
        &paths,
        &["20260809-052600"],
        Some("Grace"),
        &mut Scripted::default(),
    );
    assert_eq!(
        files_under(bystander.dir()),
        before,
        "an assertion about one session must not touch another: {output}"
    );
    let _ = asserted;
}

/// The frame's half of acceptance criterion #5, at the seam: answering one voice with the
/// assertion switches the rest of the session to it -- the quiet voices included, which the
/// queue never offered -- and the headless flag and this answer land the same state.
#[test]
fn answering_a_voice_with_the_assertion_switches_the_rest_of_the_run_to_it() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_fragmented_session(&paths, "20260809-052600");

    let mut interviewer = Scripted::answering(vec![Answer::OneSpeaker("Grace".to_string())]);
    let (report, output) = run(&paths, &["20260809-052600"], &mut interviewer);

    assert_eq!(
        interviewer.seen.len(),
        1,
        "only the voice the key was pressed on may be asked: {output}"
    );
    assert_eq!(
        report.asserted, 4,
        "the assertion reaches the quiet voices too: {output}"
    );

    let metadata = SessionMetadata::read(&session.session_json()).unwrap();
    assert_eq!(metadata.one_remote_speaker.as_deref(), Some("Grace"));
    let transcript = transcript_of(&session);
    let said = said(&transcript);
    assert!(
        said.iter()
            .all(|(who, _, _)| *who == "Grace" || *who == SPEAKER_YOU),
        "every voice reads the asserted name: {said:?}\n{output}"
    );
    assert!(output.contains("one remote speaker asserted"), "{output}");
    assert!(output.contains("one remote speaker settled"), "{output}");
}

/// The guards at the edge of the mode: a name of nothing but spaces is a request not
/// served rather than a silent no-op, and the assertion needs exactly one session id.
#[test]
fn the_assertion_guards_refuse_without_writing_anything() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    make_session(&paths, "20260810-052600");
    let before = files_under(root.path());

    let (_, output) = run_asserting(
        &paths,
        &["20260809-052600"],
        Some("   "),
        &mut Scripted::default(),
    );
    assert!(output.contains("nothing but spaces"), "{output}");

    let (_, output) = run_asserting(
        &paths,
        &["20260809-052600", "20260810-052600"],
        Some("Grace"),
        &mut Scripted::default(),
    );
    assert!(output.contains("exactly one session id"), "{output}");

    assert_eq!(
        files_under(root.path()),
        before,
        "a refused guard writes nothing"
    );
}

/// An up-front name beside the assertion is not refused: the assertion selects every voice
/// in the session itself, so a name waiting for a voice has all of them at once, and the
/// answerer is never consulted at all.
#[test]
fn an_upfront_name_is_not_refused_beside_an_assertion() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");

    let mut given = GivenName::new("Unused");
    let (report, output) =
        run_asserting_raw(&paths, &["20260809-052600"], Some("Grace"), &mut given).unwrap();
    assert_eq!(
        report.asserted, 2,
        "the assertion outranks the up-front name: {output}"
    );
    assert!(!output.contains("needs a voice"), "{output}");
}

/// TASK-050.01 acceptance criterion #4: the same assertion triggered from the full-screen
/// frame (`Answer::OneSpeaker`) and from the headless flag leaves byte-identical on-disk
/// state. One commit loop, two doors into it -- the frame contributes exactly one value,
/// the answer, and everything else is shared.
#[test]
fn the_frame_door_and_the_headless_door_leave_byte_identical_state() {
    // Two identically seeded fresh roots: the same fixture builder, the same session id,
    // and a heard-at-once pair so the veto evidence is present for both doors.
    let headless_root = tempfile::tempdir().unwrap();
    let headless = Paths::new(headless_root.path());
    let headless_session = make_fragmented_session(&headless, "20260809-052600");
    heard_at_once(&headless_session, 0, 1);

    let frame_root = tempfile::tempdir().unwrap();
    let frame = Paths::new(frame_root.path());
    let frame_session = make_fragmented_session(&frame, "20260809-052600");
    heard_at_once(&frame_session, 0, 1);

    // The headless door: the flag. The frame door: the answer the key produces.
    let mut headless_interviewer = Scripted::answering(Vec::new());
    let (headless_report, _) = run_asserting(
        &headless,
        &["20260809-052600"],
        Some("Grace"),
        &mut headless_interviewer,
    );
    let mut frame_interviewer = Scripted::answering(vec![Answer::OneSpeaker("Grace".to_string())]);
    let (frame_report, _) = run(&frame, &["20260809-052600"], &mut frame_interviewer);

    // The write-relevant counts agree. The full reports are not compared field by field:
    // the frame door builds the queue before the assertion arrives mid-run, so it counts
    // the below-floor voices as held back, while the headless door never reaches the queue
    // at all. Prompting bookkeeping differs; what the runs leave behind must not.
    assert_eq!(headless_report.named, frame_report.named);
    assert_eq!(headless_report.session_only, frame_report.session_only);
    assert_eq!(headless_report.asserted, frame_report.asserted);
    assert_eq!(
        headless_report.vetoes_overridden,
        frame_report.vetoes_overridden
    );
    // The trees each run left are identical file by file. `files_under` returns absolute
    // paths, so strip the root first -- the claim is about the tree, not about where the
    // tempdir happened to live. And the transcript header carries the wall clock of the
    // run that rewrote it (`updated:`), which sits outside either door's control: two runs
    // straddling a second boundary would differ there and nowhere else, so that one line
    // is normalised and every other byte compared as written.
    let normalise = |path: &Path, bytes: &[u8]| -> Vec<u8> {
        // The transcript alone carries the clock; every other file compares byte for byte
        // as written.
        if path.file_name().is_some_and(|name| name == "transcript.md")
            && let Ok(text) = std::str::from_utf8(bytes)
        {
            text.lines()
                .map(|line| {
                    if line.starts_with("updated:") {
                        "updated: <the clock>".to_string()
                    } else {
                        line.to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
                .into_bytes()
        } else {
            bytes.to_vec()
        }
    };
    let tree = |root: &Path| {
        files_under(root)
            .into_iter()
            .map(|(path, bytes)| {
                (
                    path.strip_prefix(root).unwrap().to_path_buf(),
                    normalise(&path, &bytes),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(tree(headless_root.path()), tree(frame_root.path()));
}

/// TASK-050.01: the preview's counts are the run's own numbers, not a re-derivation of them
/// -- on the fragmented fixture with its heard-at-once pair, what `Preview::one_speaker`
/// promises is what the run reports once it has run.
#[test]
fn the_assertion_preview_counts_match_the_run_s_override_report() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_fragmented_session(&paths, "20260809-052600");
    heard_at_once(&session, 0, 1);

    let clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
    let unknown = unknown_labels(
        clusters
            .clusters
            .iter()
            .map(|c| (c.id, c.first_spoke_seconds)),
    );
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    let assigned =
        SpeakerNames::read_or_empty(&session, &SessionId::parse("20260809-052600").unwrap())
            .unwrap();
    let preview = Preview::new(
        &clusters.clusters,
        &unknown,
        &speakers,
        &assigned,
        &clusters.clusters[0],
        Enrolment::default(),
        None,
        &[],
    );
    let assertion = preview.one_speaker("Grace").unwrap();

    let mut interviewer = Scripted::answering(Vec::new());
    let (report, output) = run_asserting(
        &paths,
        &["20260809-052600"],
        Some("Grace"),
        &mut interviewer,
    );

    assert_eq!(assertion.voices, report.asserted);
    assert_eq!(assertion.vetoes_overridden, report.vetoes_overridden);
    assert!(output.contains("4 voice(s) will read as Grace"));
    assert!(output.contains("1 veto(s) overridden"));
}

/// The frame door's interrupt rule, TASK-050.01 acceptance criterion #4: the fact lands in
/// `session.json` before the first commit on this door too, so a failure between the two
/// leaves a state that explains itself -- the assertion present, no partial labels -- and
/// a re-run converges.
#[test]
fn an_interrupt_after_the_frame_door_fact_leaves_the_assertion_and_a_rerun_converges() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");

    // The frame door writes the fact after the answer comes back, so the database goes
    // unwritable at the moment the answer is given: the fact (an existing file rewritten in
    // place) still lands, and the first commit, which creates `speakers.json`, cannot.
    let mut interviewer = UnwritableAfterFact(root.path().to_path_buf());
    let mut out = Vec::new();
    let interrupted = run_enroll(
        &paths,
        &[SessionId::parse("20260809-052600").unwrap()],
        EnrollRules {
            selector: None,
            offer: Offer::default(),
            sessions: Sessions::Unresolved,
            enrolment: Enrolment::default(),
            one_speaker: None,
            relabel_transcript: true,
            template: &TranscriptTemplate::resolve(&paths, None).unwrap(),
        },
        &mut interviewer,
        &mut Lines::new(&mut out),
    );
    assert!(
        interrupted.is_err(),
        "the failed commit must surface as an error"
    );

    // Survivors: the fact is on disk; nothing was written into the label stores; the
    // transcript still shows the unknowns.
    let metadata = SessionMetadata::read(&session.session_json()).unwrap();
    assert_eq!(metadata.one_remote_speaker.as_deref(), Some("Grace"));
    assert!(!session.speaker_names_json().exists());
    assert!(!paths.speakers_json().exists());
    let first_transcript = transcript_of(&session);
    let before = said(&first_transcript);
    assert!(
        before.iter().any(|(who, _, _)| *who == "Unknown 1"),
        "no label may have moved before the first commit: {before:?}"
    );

    // A re-run through the same door converges: the fact is already there, so the switch
    // skips the write and commits every voice against it.
    std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut rerun = Scripted::answering(vec![Answer::OneSpeaker("Grace".to_string())]);
    let (report, output) = run(&paths, &["20260809-052600"], &mut rerun);
    assert_eq!(report.asserted, 2, "{output}");
    let second_transcript = transcript_of(&session);
    let after = said(&second_transcript);
    assert!(
        after
            .iter()
            .all(|(who, _, _)| *who == "Grace" || *who == SPEAKER_YOU),
        "the re-run must complete what the interrupt left behind: {after:?}\n{output}"
    );
}

/// The frame door's answer, with the database made unwritable the moment it is given: the
/// fact lands and the first commit fails, exactly as the headless flag's interrupt test
/// arranges it for its own door.
struct UnwritableAfterFact(PathBuf);

impl Interviewer for UnwritableAfterFact {
    fn identify(&mut self, _voice: &Voice<'_>) -> Answer {
        std::fs::set_permissions(self.0.as_path(), std::fs::Permissions::from_mode(0o555)).unwrap();
        Answer::OneSpeaker("Grace".to_string())
    }
}

// --- TASK-046.09.01: a user-chosen group of voices named under one name --------------

/// A group answer from a script: the name plus the "Unknown N" handles the user marked.
fn group(name: &str, members: &[&str]) -> Answer {
    Answer::Group {
        name: name.to_string(),
        members: members.iter().map(|m| m.to_string()).collect(),
    }
}

/// Every file under `root` keyed by its path relative to `root`, with the transcript's
/// wall-clock line normalised away -- the one byte a run controls that sits outside the
/// behaviour under test, so two runs straddling a second boundary still compare equal.
fn tree_normalised(root: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    files_under(root)
        .into_iter()
        .map(|(path, bytes)| {
            let rel = path.strip_prefix(root).unwrap().to_path_buf();
            let bytes = if path.file_name().is_some_and(|name| name == "transcript.md")
                && let Ok(text) = std::str::from_utf8(&bytes)
            {
                text.lines()
                    .map(|line| {
                        if line.starts_with("updated:") {
                            "updated: <the clock>".to_string()
                        } else {
                            line.to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
                    .into_bytes()
            } else {
                bytes
            };
            (rel, bytes)
        })
        .collect()
}

/// The group's half of the frame door's interrupt rule: the database goes unwritable the
/// moment the group answer is given, so the first member's commit -- the first write the
/// group path ever makes -- cannot land, and nothing the group would write has.
struct UnwritableBeforeGroup(PathBuf);

impl Interviewer for UnwritableBeforeGroup {
    fn identify(&mut self, _voice: &Voice<'_>) -> Answer {
        std::fs::set_permissions(self.0.as_path(), std::fs::Permissions::from_mode(0o555)).unwrap();
        Answer::Group {
            name: "Grace".to_string(),
            members: vec!["Unknown 1".to_string(), "Unknown 2".to_string()],
        }
    }
}

/// TASK-046.09.01 acceptance criterion #1: a user-chosen group of voices commits under one
/// name past the heard-at-once veto that would refuse them singly, and the override is
/// counted and narrated -- naming the voice the member was heard at once with, the evidence
/// the veto acted on, and saying it was the user's grouping rather than an assertion.
#[test]
fn a_group_commits_past_the_heard_at_once_veto_and_narrates_the_override() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    heard_at_once(&session, 0, 1);

    let mut interviewer = Scripted::answering(vec![group("Grace", &["Unknown 1", "Unknown 2"])]);
    let (report, output) = run(&paths, &["20260809-052600"], &mut interviewer);

    assert_eq!(
        interviewer.seen.len(),
        1,
        "only the anchor voice may be asked: {output}"
    );
    assert_eq!(report.named, 2, "{output}");
    assert_eq!(report.refused, 0, "{output}");
    assert_eq!(report.vetoes_overridden, 1, "{output}");

    // Both voices read the group's name; the mic track keeps its own label.
    let transcript = transcript_of(&session);
    let said = said(&transcript);
    assert_eq!(
        said.iter().filter(|(who, _, _)| *who == "Grace").count(),
        3,
        "every speaker-track turn reads the group's name: {said:?}"
    );
    assert!(
        said.iter().any(|(who, _, _)| *who == SPEAKER_YOU),
        "{said:?}"
    );

    // The override line names the overlapping voice and says it was the user's grouping.
    assert!(output.contains("named Grace for Unknown 2"), "{output}");
    assert!(output.contains("heard at once with Unknown 1"), "{output}");
    assert!(
        output.contains("you chose these voices as one person"),
        "the group's authority must be told apart from the assertion's: {output}"
    );

    // References stored per the floor rule: both above the floor, so both are held.
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.references("Grace"), 2, "{:?}", speakers.speakers);
}

/// Acceptance criterion #1, the quiet half (D5): a group reaches members below the offer
/// floor -- the queue never offers them -- and names them against the session alone, the
/// way the assertion reaches the quiet voices alike. Only the loud member touches the
/// database; the rest are recorded against this session's names file.
#[test]
fn a_group_reaches_the_below_floor_members_and_names_them_against_the_session() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_fragmented_session(&paths, "20260809-052600");
    heard_at_once(&session, 0, 1);

    // Three-way group: the loud voice (id 0) plus two below-floor members (id 1, id 2). The
    // heard-at-once pair is 0-1, so committing id 1 overrides one veto.
    let mut interviewer = Scripted::answering(vec![group(
        "Grace",
        &["Unknown 1", "Unknown 2", "Unknown 3"],
    )]);
    let (report, output) = run(&paths, &["20260809-052600"], &mut interviewer);

    assert_eq!(
        interviewer.seen.len(),
        1,
        "the quiet members are not asked: {output}"
    );
    assert_eq!(report.named, 3, "{output}");
    assert_eq!(report.refused, 0, "{output}");
    assert_eq!(report.vetoes_overridden, 1, "{output}");
    // Two of the three sit below the reference floor: named against the session alone.
    assert_eq!(report.session_only, 2, "{output}");

    // All three members read the group's name; the fourth voice (not a member) keeps its own.
    let transcript = transcript_of(&session);
    let said = said(&transcript);
    assert_eq!(
        said.iter().filter(|(who, _, _)| *who == "Grace").count(),
        3,
        "the three members read the group's name: {said:?}"
    );
    assert!(
        said.iter().any(|(who, _, _)| *who == "Unknown 4"),
        "{said:?}"
    );

    // Only the loud member stores a reference; the two quiet ones leave the database alone.
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.references("Grace"), 1, "{:?}", speakers.speakers);
    // All three members keep a names-file row -- the declaration stands in both stores for
    // the above-floor member and in the session alone for the two below it -- so every
    // later run reads all three as Grace without re-asking.
    let assigned = assigned_in(&session, "20260809-052600");
    assert_eq!(assigned.names.len(), 3, "{:?}", assigned.names);
}

/// Acceptance criterion #2, first half: an interrupt between the group answer and the first
/// member's commit leaves the on-disk state exactly as it stood -- no reference, no names-
/// file delta, transcript and session.json untouched -- and a re-run with the same answer
/// converges onto the whole.
#[test]
fn an_interrupt_before_the_first_group_commit_writes_nothing_and_a_rerun_converges() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    heard_at_once(&session, 0, 1);
    let before = files_under(root.path());

    let mut interviewer = UnwritableBeforeGroup(root.path().to_path_buf());
    let interrupted = run_asserting_raw(&paths, &["20260809-052600"], None, &mut interviewer);
    std::fs::set_permissions(root.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(
        interrupted.is_err(),
        "the failed commit must surface as an error"
    );

    // Nothing the group would write has landed: byte-for-byte the same tree as before.
    assert_eq!(
        files_under(root.path()),
        before,
        "an interrupted group wrote something"
    );

    // A re-run with the same answer converges onto the complete state.
    let mut rerun = Scripted::answering(vec![group("Grace", &["Unknown 1", "Unknown 2"])]);
    let (report, output) = run(&paths, &["20260809-052600"], &mut rerun);
    assert_eq!(report.named, 2, "{output}");
    assert_eq!(report.vetoes_overridden, 1, "{output}");
    let transcript = transcript_of(&session);
    let said = said(&transcript);
    assert_eq!(
        said.iter().filter(|(who, _, _)| *who == "Grace").count(),
        3,
        "the re-run must complete what the interrupt left behind: {said:?}\n{output}"
    );
}

/// Acceptance criterion #2, second half: a re-run over the state a killed run would have
/// left -- the first member fully committed, the second untouched -- converges onto the same
/// state a fresh run produces, and a further pass writes nothing at all.
#[test]
fn a_rerun_over_a_partial_group_converges_and_then_writes_nothing_new() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    heard_at_once(&session, 0, 1);

    // A run interrupted after its first member's commit: Unknown 1 named Grace (reference,
    // names file, transcript), Unknown 2 left where it stood.
    let _ = run(
        &paths,
        &["20260809-052600"],
        &mut Scripted::answering(vec![named("Grace")]),
    );
    let mid = transcript_of(&session);
    assert_eq!(mid.turns[0].speaker.as_str(), "Grace");
    assert_eq!(mid.turns[2].speaker.as_str(), "Unknown 2");

    // Re-running the whole group completes the second member against the first's evidence,
    // overriding the veto the first member's name now holds.
    let mut rerun = Scripted::answering(vec![group("Grace", &["Unknown 1", "Unknown 2"])]);
    let (report, out) = run(&paths, &["20260809-052600"], &mut rerun);
    assert_eq!(report.vetoes_overridden, 1, "{out}");
    let transcript = transcript_of(&session);
    let after = said(&transcript);
    assert_eq!(
        after.iter().filter(|(who, _, _)| *who == "Grace").count(),
        3,
        "the re-run must complete the transcript: {after:?}\n{out}"
    );
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.references("Grace"), 2, "{:?}", speakers.speakers);

    // A further pass is a no-op on disk: converged means byte-identical, not merely
    // equivalent.
    let before = files_under(root.path());
    let (_, out) = run(
        &paths,
        &["20260809-052600"],
        &mut Scripted::answering(vec![group("Grace", &["Unknown 1", "Unknown 2"])]),
    );
    assert_eq!(
        files_under(root.path()),
        before,
        "a converged group rewrote a file: {out}"
    );
}

/// A commit interrupted between a member's `speakers.json` write and its
/// `speaker_names.json` write leaves that member holding a database reference under the
/// name but no names-file row. Confirming it plainly -- the natural recovery gesture, not
/// the group door -- must stand the declaration up in both stores so the transcript label
/// agrees with the database on every later pass, and the rest of the group still converges
/// through the door onto the same state.
#[test]
fn a_plain_confirmation_of_a_stranded_voice_repairs_its_row_and_the_group_converges() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    heard_at_once(&session, 0, 1);
    // The post-SIGKILL state: a Grace reference built from cluster 0's exact embedding,
    // with no `speaker_names.json` row for it.
    enrolled(&[("Grace", voice(0))], &paths);

    // Pass 1 (the natural gesture): confirm 'is Unknown 1 Grace?' with Enter. Before the
    // fix this confirmation forgot the row and left the voice demoted on every later pass.
    let (_, out) = run_asking(
        &paths,
        &[],
        CORRECT,
        &mut Scripted::answering(vec![named("Grace")]),
    );
    let transcript = transcript_of(&session);
    assert_eq!(transcript.turns[0].speaker.as_str(), "Grace", "{out}");
    assert_eq!(transcript.turns[3].speaker.as_str(), "Grace", "{out}");
    // The heart of the fix: a standing row now exists for the stranded member.
    let assigned = assigned_in(&session, "20260809-052600");
    assert!(
        assigned
            .names
            .iter()
            .any(|row| row.cluster == 0 && row.name == "Grace"),
        "no names-file row for the confirmed voice: {:?}\n{out}",
        assigned.names
    );

    // Pass 2 (the rest of the group through the door): completes the second member against
    // the first's evidence, overriding the veto the first member's name now holds.
    let (report, out) = run(
        &paths,
        &["20260809-052600"],
        &mut Scripted::answering(vec![group("Grace", &["Unknown 1", "Unknown 2"])]),
    );
    assert_eq!(report.vetoes_overridden, 1, "{out}");
    let transcript = transcript_of(&session);
    let after = said(&transcript);
    assert_eq!(after[0].0, "Grace", "{after:?}\n{out}");
    assert_eq!(after[1].0, "You", "{after:?}\n{out}");
    assert_eq!(after[2].0, "Grace", "{after:?}\n{out}");
    assert_eq!(after[3].0, "Grace", "{after:?}\n{out}");
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.references("Grace"), 2, "{:?}", speakers.speakers);
    // Both members hold a standing row: the one the plain confirmation wrote and the one
    // the group door wrote.
    let assigned = assigned_in(&session, "20260809-052600");
    assert_eq!(assigned.names.len(), 2, "{:?}", assigned.names);

    // Pass 3: a further pass writes nothing at all -- converged means byte-identical.
    let before = files_under(root.path());
    let (_, out) = run(
        &paths,
        &["20260809-052600"],
        &mut Scripted::answering(vec![]),
    );
    assert_eq!(
        files_under(root.path()),
        before,
        "a converged pass rewrote a file: {out}"
    );
}

/// Acceptance criterion #3: the aggregate preview equals the sequential application of the
/// members' individual previews. On a fixture with a colliding pre-enrolment on each member
/// -- one displaced by the correction, one left stale below the floor -- every field of the
/// group's consequence is what the run's own report and the on-disk state say once it has run.
#[test]
fn the_group_preview_equals_the_sequential_commit() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_fragmented_session(&paths, "20260809-052600");
    heard_at_once(&session, 0, 1);

    // Bob holds a reference built from the loud member's exact voice (a correction will
    // displace it); Alice one built from the quiet member's exact voice (a session-only
    // naming leaves it behind as stale).
    let clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
    enrolled(
        &[
            ("Bob", clusters.clusters[0].embedding.clone()),
            ("Alice", clusters.clusters[1].embedding.clone()),
        ],
        &paths,
    );

    // The aggregate dry run, off the database as it stands before the run.
    let unknown = unknown_labels(
        clusters
            .clusters
            .iter()
            .map(|c| (c.id, c.first_spoke_seconds)),
    );
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    let assigned =
        SpeakerNames::read_or_empty(&session, &SessionId::parse("20260809-052600").unwrap())
            .unwrap();
    let preview = Preview::new(
        &clusters.clusters,
        &unknown,
        &speakers,
        &assigned,
        &clusters.clusters[0],
        Enrolment::default(),
        None,
        &[],
    );
    let consequence = preview.group("Grace", &["Unknown 1", "Unknown 2"]).unwrap();

    let mut interviewer = Scripted::answering(vec![group("Grace", &["Unknown 1", "Unknown 2"])]);
    let (report, output) = run(&paths, &["20260809-052600"], &mut interviewer);

    // Applied and refused agree with the run, in queue order.
    assert_eq!(
        consequence.applied,
        vec!["Unknown 1".to_string(), "Unknown 2".to_string()],
        "{output}"
    );
    assert!(consequence.refused.is_empty(), "{output}");
    assert_eq!(report.refused, 0, "{output}");

    // The vetoes the preview counts are the run's own number.
    assert_eq!(
        consequence.vetoes_overridden, report.vetoes_overridden,
        "{output}"
    );
    assert_eq!(consequence.vetoes_overridden, 1, "{output}");

    // The total reference count the preview promises is what the database ends up holding --
    // one, since only the loud member is above the floor.
    let final_speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(
        consequence.references_after,
        final_speakers.references("Grace"),
        "{output}"
    );
    assert_eq!(
        final_speakers.references("Grace"),
        1,
        "{:?}",
        final_speakers.speakers
    );

    // Displaced: Bob's reference, built from the loud member's exact voice, is withdrawn.
    assert!(
        consequence
            .displaced
            .iter()
            .any(|d| d.name == "Bob" && d.remaining == 0),
        "{:?}",
        consequence.displaced
    );
    assert_eq!(
        final_speakers.references("Bob"),
        0,
        "{:?}",
        final_speakers.speakers
    );

    // Stale: Alice's reference, built from the quiet member's exact voice, survives --
    // nothing below the floor touches the database -- and the preview flags it.
    assert!(
        consequence.stale.contains(&"Alice".to_string()),
        "{:?}",
        consequence.stale
    );
    assert_eq!(
        final_speakers.references("Alice"),
        1,
        "{:?}",
        final_speakers.speakers
    );

    // Both members read the name; the two non-members keep their own labels.
    let transcript = transcript_of(&session);
    let said = said(&transcript);
    assert_eq!(
        said.iter().filter(|(who, _, _)| *who == "Grace").count(),
        2,
        "{said:?}"
    );
}

/// Acceptance criterion #3, the refusal half: a member whose naming would take a name off a
/// non-member is refused by the standard check while the walk carries on, and the preview
/// reports exactly which member was refused and what the applied members leave behind.
#[test]
fn the_group_preview_reports_the_member_the_run_refuses() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    with_embeddings(&session, &[nearly(0.0), nearly(20.0)]);
    enrolled(&[("Bob", nearly(60.0))], &paths);

    let clusters = SpeakerClusters::read(&session.speaker_clusters_json()).unwrap();
    let unknown = unknown_labels(
        clusters
            .clusters
            .iter()
            .map(|c| (c.id, c.first_spoke_seconds)),
    );
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    let assigned =
        SpeakerNames::read_or_empty(&session, &SessionId::parse("20260809-052600").unwrap())
            .unwrap();
    let preview = Preview::new(
        &clusters.clusters,
        &unknown,
        &speakers,
        &assigned,
        &clusters.clusters[0],
        Enrolment::default(),
        None,
        &[],
    );
    let consequence = preview.group("Grace", &["Unknown 1", "Unknown 2"]).unwrap();

    let mut interviewer = Scripted::answering(vec![group("Grace", &["Unknown 1", "Unknown 2"])]);
    let (report, output) = run(&paths, &["20260809-052600"], &mut interviewer);

    // The first member's naming would take Bob off the second voice, so the group refuses it
    // and carries on: the preview says exactly that, and the run agrees.
    assert_eq!(consequence.refused.len(), 1, "{output}");
    assert_eq!(consequence.refused[0].0, "Unknown 1");
    assert!(matches!(consequence.refused[0].1, Refusal::Taken { .. }));
    assert_eq!(
        consequence.applied,
        vec!["Unknown 2".to_string()],
        "{output}"
    );
    assert_eq!(report.refused, 1, "{output}");
    assert_eq!(report.named, 1, "{output}");

    // The total reference count is over the applied members only.
    let final_speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(
        consequence.references_after,
        final_speakers.references("Grace"),
        "{output}"
    );
    assert_eq!(
        final_speakers.references("Grace"),
        1,
        "{:?}",
        final_speakers.speakers
    );
}

/// Acceptance criterion #4: a group commit writes nothing to session.json. A subset names
/// voices; it does not claim the track, so the one-remote-speaker fact never lands and the
/// file's bytes do not move.
#[test]
fn a_group_commit_writes_nothing_to_session_json() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    heard_at_once(&session, 0, 1);

    let before_meta = SessionMetadata::read(&session.session_json()).unwrap();
    assert!(before_meta.one_remote_speaker.is_none());
    let before_bytes = std::fs::read(session.session_json()).unwrap();

    let mut interviewer = Scripted::answering(vec![group("Grace", &["Unknown 1", "Unknown 2"])]);
    let (report, output) = run(&paths, &["20260809-052600"], &mut interviewer);
    assert_eq!(report.named, 2, "{output}");

    // No one-remote-speaker fact, and the file is byte-for-byte unchanged.
    let after_meta = SessionMetadata::read(&session.session_json()).unwrap();
    assert!(
        after_meta.one_remote_speaker.is_none(),
        "a group must not write the assertion fact: {output}"
    );
    assert_eq!(
        std::fs::read(session.session_json()).unwrap(),
        before_bytes,
        "session.json changed: {output}"
    );
}

/// D2, made mechanical: a one-member group carries no veto authority, so it behaves exactly
/// like today's plain naming of that member -- including the heard-at-once veto refusing the
/// second voice. Two identically seeded roots, one answered with plain names and one with
/// one-member groups, leave identical counts and byte-identical trees.
#[test]
fn a_one_member_group_behaves_like_plain_naming_and_the_veto_still_refuses() {
    let plain_root = tempfile::tempdir().unwrap();
    let plain = Paths::new(plain_root.path());
    let plain_session = make_session(&plain, "20260809-052600");
    heard_at_once(&plain_session, 0, 1);

    let group_root = tempfile::tempdir().unwrap();
    let grp = Paths::new(group_root.path());
    let group_session = make_session(&grp, "20260809-052600");
    heard_at_once(&group_session, 0, 1);

    // Plain: name both voices Alice; the veto refuses the second.
    let mut plain_iv = Scripted::answering(vec![named("Alice"), named("Alice")]);
    let (plain_report, plain_out) = run(&plain, &["20260809-052600"], &mut plain_iv);

    // Group: the same two namings as one-member groups; the veto must refuse the second alike.
    let mut group_iv = Scripted::answering(vec![
        group("Alice", &["Unknown 1"]),
        group("Alice", &["Unknown 2"]),
    ]);
    let (group_report, group_out) = run(&grp, &["20260809-052600"], &mut group_iv);

    // Same counts: one named, one refused by the veto, no overrides either way.
    assert_eq!(
        plain_report.named, group_report.named,
        "{plain_out}\n{group_out}"
    );
    assert_eq!(
        plain_report.refused, group_report.refused,
        "{plain_out}\n{group_out}"
    );
    assert_eq!(
        plain_report.vetoes_overridden, group_report.vetoes_overridden,
        "{plain_out}\n{group_out}"
    );
    assert_eq!(
        plain_report.session_only, group_report.session_only,
        "{plain_out}\n{group_out}"
    );
    assert_eq!(
        group_report.refused, 1,
        "the veto still refuses a one-member group: {group_out}"
    );

    // And identical trees, byte for byte.
    assert_eq!(
        tree_normalised(plain_root.path()),
        tree_normalised(group_root.path())
    );
}
