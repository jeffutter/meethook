//! what an answer stores: references, the cap, and database versions.

use super::*;

/// Replaces `naming_someone_already_enrolled_replaces_their_reference`, which asserted the
/// v1 rule this ticket retires: one row per name, the second recording overwriting the
/// first. Overwriting is what made naming a second voice cost the first one its name, and
/// TASK-027.01 measured it as the *worst* of the three candidate policies on both corpora.
///
/// Typing a name already in the database now adds another recording of that person: both
/// rows survive, in enrollment order, and the line says how many they hold.
#[test]
fn naming_someone_already_enrolled_adds_a_reference_rather_than_replacing_one() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    // Alice, enrolled from a voice that matches neither cluster here.
    EnrolledSpeakers::new(vec![EnrolledSpeaker {
        name: "Alice".to_string(),
        embedding: voice(3),
        clip_seconds: None,
    }])
    .write(&paths)
    .unwrap();

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert!(
        output.contains("enrolled another recording of Alice: 2 reference(s) now"),
        "{output}"
    );
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    let stored: Vec<(&str, &[f32])> = speakers
        .speakers
        .iter()
        .map(|s| (s.name.as_str(), s.embedding.as_slice()))
        .collect();
    assert_eq!(
        stored,
        [
            ("Alice", voice(3).as_slice()),
            ("Alice", voice(0).as_slice())
        ],
        "the first recording must survive the second"
    );
}

/// Re-answering the same voice with the same name must not spend a capped reference slot on
/// information already held -- the common way to reach this being a second `--correct` pass
/// over a session that was enrolled from it in the first place.
#[test]
fn re_answering_a_voice_with_the_name_it_already_gave_stores_nothing_new() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    enrolled(&[("Alice", voice(0))], &paths);

    let mut interviewer = Scripted::answering(vec![named("Alice")]);
    let (report, output) = run_asking(&paths, &[], CORRECT, &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert!(
        output.contains("Alice already has a reference built from this voice"),
        "{output}"
    );
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.references("Alice"), 1, "{:?}", speakers.speakers);
}

/// The growth cap. A person met in more rooms than meethook keeps recordings of gets the
/// name in that transcript and no new reference -- and, crucially, loses none of the ones
/// they have, because this recording is no better than any of them. Dropping the oldest
/// would un-name a voice in some earlier session, which is the defect this whole ticket
/// exists to end; only a *longer* recording displaces anything, which is the companion test.
///
/// Every session here holds the same 10.0 s in the answered cluster, so the offer past the
/// cap ties with the shortest held rather than beating it. Every voice is on its own axis,
/// so no two are ever within reach of each other and each session really does have to ask.
#[test]
fn at_the_reference_cap_the_name_is_recorded_against_the_session_instead() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let axes = MAX_REFERENCES_PER_SPEAKER + 2;
    let sessions: Vec<SessionPaths> = (0..=MAX_REFERENCES_PER_SPEAKER)
        .map(|i| {
            let session = make_session(&paths, &format!("20260809-0526{i:02}"));
            with_embeddings(&session, &[axis(i, axes), axis(axes - 1, axes)]);
            session
        })
        .collect();

    let mut interviewer = Scripted::answering(
        sessions
            .iter()
            .flat_map(|_| [named("Alice"), Answer::Skip])
            .collect(),
    );
    let (report, output) = run(&paths, &[], &mut interviewer);

    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(
        speakers.references("Alice"),
        MAX_REFERENCES_PER_SPEAKER,
        "nothing already stored may be dropped to make room: {output}"
    );
    // The answer past the cap is still an answer: the transcript reads Alice, the name is
    // in that session's own file, and the line says why no reference was stored.
    let last = sessions.last().unwrap();
    assert_eq!(transcript_of(last).turns[0].speaker, "Alice", "{output}");
    let assigned = assigned_in(
        last,
        &format!("20260809-0526{MAX_REFERENCES_PER_SPEAKER:02}"),
    );
    assert_eq!(assigned.names.len(), 1, "{:?}", assigned.names);
    assert_eq!(assigned.names[0].name, "Alice");
    assert_eq!(report.named, sessions.len(), "{output}");
    assert_eq!(report.session_only, 1, "{output}");
    assert!(
        output.contains(&format!(
            "Alice already holds {MAX_REFERENCES_PER_SPEAKER} reference(s)"
        )),
        "{output}"
    );
    // The remedy is two commands rather than a file path: this is the line that used to send
    // people to a text editor, and both halves have to be on it -- `speakers` because the
    // line cannot know which reference should go, `forget` because that is what removes it.
    assert!(
        output.contains("meethook speakers shows what each of them is naming"),
        "the line has to say how to see what each recording is naming: {output}"
    );
    assert!(
        output.contains("meethook forget Alice --reference N removes the one you pick"),
        "the line has to name the command that makes room: {output}"
    );
    assert!(
        !output.contains(&paths.speakers_json().display().to_string()),
        "no remedy in this tool is a hand-edit of speakers.json any more: {output}"
    );
}

/// The companion to the cap: a *longer* recording of somebody full displaces the shortest
/// one they hold, and says so. This is what makes a person's references get better with use
/// rather than merely being whichever ten meethook happened to meet first.
///
/// The last session carries 90.0 s where the ten before it carried 10.0 s, so the offer past
/// the cap beats the shortest held rather than tying with it. The line has to name both
/// lengths: something stored was dropped, and an enrollment that vanishes without a line
/// about it is worse than the bug.
#[test]
fn past_the_cap_a_longer_recording_displaces_the_shortest_and_says_what_went() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let axes = MAX_REFERENCES_PER_SPEAKER + 2;
    let sessions: Vec<SessionPaths> = (0..=MAX_REFERENCES_PER_SPEAKER)
        .map(|i| {
            let session = make_session(&paths, &format!("20260809-0526{i:02}"));
            with_embeddings(&session, &[axis(i, axes), axis(axes - 1, axes)]);
            with_speech_seconds(
                &session,
                &[
                    if i == MAX_REFERENCES_PER_SPEAKER {
                        90.0
                    } else {
                        10.0
                    },
                    10.0,
                ],
            );
            session
        })
        .collect();

    let mut interviewer = Scripted::answering(
        sessions
            .iter()
            .flat_map(|_| [named("Alice"), Answer::Skip])
            .collect(),
    );
    let (report, output) = run(&paths, &[], &mut interviewer);

    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(
        speakers.references("Alice"),
        MAX_REFERENCES_PER_SPEAKER,
        "the cap still holds: {output}"
    );
    assert!(
        speakers
            .speakers
            .iter()
            .any(|s| s.clip_seconds == Some(90.0)),
        "the longer recording is what is now stored: {:?}",
        speakers.speakers
    );
    assert_eq!(
        speakers
            .speakers
            .iter()
            .filter(|s| s.clip_seconds == Some(10.0))
            .count(),
        MAX_REFERENCES_PER_SPEAKER - 1,
        "exactly one of the ten should have gone: {:?}",
        speakers.speakers
    );
    assert_eq!(
        report.session_only, 0,
        "the answer stored a reference, so it is not a session-only name: {output}"
    );
    assert!(
        output.contains("enrolled a better recording of Alice: 90.0 s replaces the shortest"),
        "{output}"
    );
    assert!(
        output.contains("which was 10.0 s"),
        "the line has to say what was dropped, not just what was kept: {output}"
    );
}

/// A database written before this schema bump. References cannot be regenerated -- the audio
/// they were built from may be long deleted -- so a v1 file must be migrated rather than
/// refused: the names in it still identify their voices, and the file is upgraded by the
/// next write rather than left claiming a version its contents no longer match.
#[test]
fn a_v1_database_still_names_its_voices_and_is_upgraded_by_the_next_write() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    // Written by hand at version 1, which is the only way to produce one now: the raw bytes
    // rather than a serialized struct, so that bumping the constant cannot quietly turn this
    // fixture into a current-version file and stop testing the migration.
    std::fs::write(
        paths.speakers_json(),
        b"{\n  \"schema_version\": 1,\n  \"speakers\": [\
          {\"name\": \"Alice\", \"embedding\": [1.0, 0.0, 0.0, 0.0]}]\n}\n"
            .as_slice(),
    )
    .unwrap();
    assert_eq!(voice(0), [1.0, 0.0, 0.0, 0.0], "the fixture is cluster 0");

    let mut interviewer = Scripted::answering(vec![named("Bob")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert_eq!(
        transcript_of(&session).turns[0].speaker,
        "Alice",
        "a v1 name must survive the upgrade: {output}"
    );
    let on_disk = std::fs::read_to_string(paths.speakers_json()).unwrap();
    assert!(
        on_disk.contains(&format!(
            "\"schema_version\": {}",
            meethook_session::ENROLLED_SPEAKERS_SCHEMA_VERSION
        )),
        "{on_disk}"
    );
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(speakers.references("Alice"), 1, "{:?}", speakers.speakers);
    assert_eq!(speakers.references("Bob"), 1, "{:?}", speakers.speakers);
}

/// The v2 -> v3 half of the same guarantee. A v2 row carries no clip length, and the
/// migration must leave it that way rather than inventing one: an unmeasured reference is
/// never the row an eviction picks, and a zero written here would make it the first to go.
#[test]
fn a_v2_reference_keeps_its_name_and_gains_no_invented_clip_length() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let session = make_session(&paths, "20260809-052600");
    std::fs::write(
        paths.speakers_json(),
        b"{\n  \"schema_version\": 2,\n  \"speakers\": [\
          {\"name\": \"Alice\", \"embedding\": [1.0, 0.0, 0.0, 0.0]}]\n}\n"
            .as_slice(),
    )
    .unwrap();

    let mut interviewer = Scripted::answering(vec![named("Bob")]);
    let (report, output) = run(&paths, &[], &mut interviewer);

    assert_eq!(report.named, 1, "{output}");
    assert_eq!(
        transcript_of(&session).turns[0].speaker,
        "Alice",
        "a v2 name must survive the upgrade: {output}"
    );
    let speakers = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    let alice = speakers
        .speakers
        .iter()
        .find(|s| s.name == "Alice")
        .expect("Alice survives the migration");
    assert_eq!(alice.clip_seconds, None, "{:?}", speakers.speakers);
    let bob = speakers
        .speakers
        .iter()
        .find(|s| s.name == "Bob")
        .expect("Bob was just enrolled");
    assert!(
        bob.clip_seconds.is_some(),
        "a reference written now records what it was built from: {bob:?}"
    );
}

/// A database from a *newer* meethook cannot be read as though it were this one, and a run
/// that ignored it would silently un-name everybody. Reported by name against the session,
/// like every other unreadable file on this path, and the queue does not carry on into a
/// second session naming nobody.
#[test]
fn a_database_from_a_newer_meethook_fails_the_run_by_name() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    make_session(&paths, "20260809-052600");
    std::fs::write(
        paths.speakers_json(),
        b"{\n  \"schema_version\": 99,\n  \"speakers\": []\n}\n",
    )
    .unwrap();

    let mut interviewer = Scripted::default();
    let mut out = Vec::new();
    let error = run_enroll(
        &paths,
        &[],
        EnrollRules {
            selector: None,
            offer: Offer::default(),
            sessions: Sessions::default(),
            enrolment: Enrolment::default(),
            relabel_transcript: true,
            one_speaker: None,
            template: &TranscriptTemplate::builtin(),
        },
        &mut interviewer,
        &mut Lines::new(&mut out),
    )
    .unwrap_err();

    assert!(error.to_string().contains("speakers.json"), "{error}");
    assert!(error.to_string().contains("upgrade meethook"), "{error}");
    assert!(
        interviewer.seen.is_empty(),
        "nothing may be asked against a database that could not be read"
    );
}
