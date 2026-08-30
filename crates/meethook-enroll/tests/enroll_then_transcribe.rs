//! The join between the two commands: a person named once is recognised ever after.
//!
//! Everything either command does on its own is covered by unit tests inside its own crate.
//! What no test in either crate can decide is whether the file they meet over -- the enrolled
//! database at `~/meethook/speakers.json` -- is written by `enroll` in the shape `transcribe`
//! reads it in. That is the whole point of the enrollment path, and it has exactly one
//! mechanically checkable half:
//!
//! *Name a voice in one session, and a **different** session containing that voice comes back
//! named, without anybody being asked again.*
//!
//! So this drives the real `run_enroll` and the real `run_batch` in sequence, over real
//! session directories on a temporary disk, with only the two model-backed engines faked.
//! Nothing is handed between them in memory: the second command is told nothing but where the
//! root directory is, exactly as the CLI tells it.
//!
//! The other half -- whether one human voice recorded in two different meetings actually lands
//! inside `IDENTIFY_DISTANCE` -- is a question about the embedding model and a real recording
//! rather than about this code. It is TASK-014, and it needs a microphone.

use meethook_enroll::{
    Answer, EnrollReport, EnrollRules, Enrolment, Interviewer, Offer, Sessions, Voice, run_enroll,
    write_clip,
};
use meethook_session::{
    Paths, RepresentativeSegment, SessionId, SessionMetadata, SpeakerCluster, SpeakerNames,
    TrackSync, Transcript, TranscriptTemplate,
};
use meethook_transcribe::{
    AsrSegment, BatchReport, Diarization, Diarize, Engines, Result, SpeakerTurn, SpeechToText,
    TARGET_RATE, run_batch,
};

/// Apple Silicon's timebase, matching the recorder's.
const NUMER: u32 = 125;
const DENOM: u32 = 3;

/// A recognizer that says the same thing about every track it is handed.
///
/// *What* was recognised does not matter here -- this is about who the words are attributed to
/// -- so one canned segment per call keeps the fixture out of the way of the assertion.
struct FakeAsr;

impl SpeechToText for FakeAsr {
    fn transcribe(&mut self, _audio_16k_mono: &[f32]) -> Result<Vec<AsrSegment>> {
        Ok(vec![AsrSegment {
            start_s: 0.0,
            end_s: 1.0,
            text: "hello".to_string(),
        }])
    }
}

/// Diarization without a model: one voice, whichever one the test says was in the room.
struct FakeDiarizer {
    embedding: Vec<f32>,

    /// How much that voice spoke. Which side of `enroll`'s reference floor this sits on is
    /// what decides whether naming it writes a reference or a session-scoped name, so it is a
    /// parameter rather than a constant even though the audio is a quarter second either way.
    speech_seconds: f64,
}

impl Diarize for FakeDiarizer {
    fn diarize(&mut self, _speaker_16k_mono: &[f32]) -> Result<Diarization> {
        Ok(Diarization {
            clusters: vec![SpeakerCluster {
                id: 0,
                embedding: self.embedding.clone(),
                speech_seconds: self.speech_seconds,
                first_spoke_seconds: 0.0,
                heard_at_once_with: Vec::new(),
                representatives: vec![RepresentativeSegment {
                    start: 0.0,
                    end: 0.2,
                }],
            }],
            turns: vec![SpeakerTurn {
                start_s: 0.0,
                end_s: 1.0,
                cluster: 0,
            }],
        })
    }
}

/// A distinct unit vector per person, so enrolling one of these voices matches it and nobody
/// else's -- the convention both crates' own tests already use.
fn voice(id: u32) -> Vec<f32> {
    let mut embedding = vec![0.0f32; 4];
    embedding[id as usize % 4] = 1.0;
    embedding
}

/// A recordable session: both tracks, sample-synchronised, a quarter second of audio in each.
/// Long enough to be real audio and short enough to be free.
fn make_session(paths: &Paths, id: &str) -> SessionId {
    let id = SessionId::parse(id).unwrap();
    let session = paths.session(&id);
    std::fs::create_dir_all(session.dir()).unwrap();

    let samples = TARGET_RATE as usize / 4;
    let quiet = vec![0.0f32; samples];
    let tone: Vec<f32> = (0..samples)
        .map(|i| (i as f32 / TARGET_RATE as f32 * 440.0 * std::f32::consts::TAU).sin() * 0.3)
        .collect();
    write_clip(&session.mic_wav(), &quiet).unwrap();
    write_clip(&session.speaker_wav(), &tone).unwrap();

    let sync = TrackSync {
        host_ticks: 900_000_000_000,
        timebase_numer: NUMER,
        timebase_denom: DENOM,
    };
    SessionMetadata::new(
        id.clone(),
        jiff::Timestamp::from_second(1_770_000_000).unwrap(),
        sync,
        sync,
    )
    .write(&session.session_json())
    .unwrap();

    id
}

/// Transcribes exactly one session, as the CLI would, with `who` the only voice on its far
/// end -- a voice that spoke long enough for a name given to it to become a reference.
fn transcribe(paths: &Paths, id: &SessionId, who: u32) -> BatchReport {
    transcribe_speaking(paths, id, who, 10.0, false)
}

/// `transcribe`, with the far-end voice's talk time and `--force` exposed. Separate so the
/// tests about the enrolled database carry neither.
fn transcribe_speaking(
    paths: &Paths,
    id: &SessionId,
    who: u32,
    speech_seconds: f64,
    force: bool,
) -> BatchReport {
    let mut factory = || {
        Ok(Engines {
            asr: Box::new(FakeAsr),
            diarizer: Box::new(FakeDiarizer {
                embedding: voice(who),
                speech_seconds,
            }),
        })
    };
    run_batch(
        paths,
        std::slice::from_ref(id),
        force,
        &TranscriptTemplate::resolve(paths, None).unwrap(),
        meethook_transcribe::mixdown::Settings::default(),
        &mut factory,
        &mut std::io::sink(),
    )
    .unwrap()
}

/// An interviewer holding at most one answer, which records every voice it was shown so a test
/// can assert on what was asked as well as on what was written.
struct AsksOnce {
    answer: Option<String>,
    asked: Vec<String>,
}

impl AsksOnce {
    fn answering(name: Option<&str>) -> AsksOnce {
        AsksOnce {
            answer: name.map(str::to_string),
            asked: Vec::new(),
        }
    }
}

impl Interviewer for AsksOnce {
    fn identify(&mut self, voice: &Voice<'_>) -> Answer {
        self.asked
            .push(format!("{} {}", voice.session, voice.attribution.label()));
        match self.answer.take() {
            Some(name) => Answer::Named {
                name,
                anyway: false,
            },
            None => Answer::Skip,
        }
    }
}

/// `--all`: the only way a voice under `PROMPT_FLOOR_SECONDS` is offered at all, and so the
/// only way the tests below reach one quiet enough to be named for its session alone.
const QUIET: Offer = Offer {
    quiet: true,
    named: false,
};

fn enroll(paths: &Paths, interviewer: &mut dyn Interviewer) -> EnrollReport {
    enroll_offering(paths, Offer::default(), interviewer)
}

/// `enroll`, reaching the quiet voices too, which is the only way a voice under the reference
/// floor is asked about in a session that also holds a louder one.
fn enroll_offering(paths: &Paths, offer: Offer, interviewer: &mut dyn Interviewer) -> EnrollReport {
    run_enroll(
        paths,
        &[],
        EnrollRules {
            selector: None,
            offer,
            // Mirrors the CLI, where both halves come off `--correct`.
            sessions: if offer.named {
                Sessions::Every
            } else {
                Sessions::Unresolved
            },
            enrolment: Enrolment::default(),
            one_speaker: None,
            template: &TranscriptTemplate::resolve(paths, None).unwrap(),
        },
        interviewer,
        &mut meethook_enroll::Lines::new(&mut std::io::sink()),
    )
    .unwrap()
}

/// Who the transcript says spoke, in order -- what a reader of the file sees.
fn speakers_in(paths: &Paths, id: &SessionId) -> Vec<String> {
    Transcript::read(&paths.session(id).transcript_json())
        .unwrap()
        .turns
        .iter()
        .map(|turn| turn.speaker.clone())
        .collect()
}

/// TASK-009 acceptance criterion #8, across both commands.
///
/// January is transcribed before anybody is enrolled, so its far-end voice comes out
/// "Unknown 1" and `enroll` asks about it. February is the same person in a different meeting,
/// transcribed *after* that answer -- and must come back named without a second prompt, purely
/// because `transcribe` read the file `enroll` wrote.
#[test]
fn a_person_named_in_one_session_is_named_by_transcribe_in_the_next() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let january = make_session(&paths, "20260105-090000");
    let february = make_session(&paths, "20260209-090000");

    // Before anybody is enrolled, a voice is a number.
    assert_eq!(transcribe(&paths, &january, 0).transcribed, 1);
    assert!(
        speakers_in(&paths, &january).contains(&"Unknown 1".to_string()),
        "an unenrolled voice should be numbered, was {:?}",
        speakers_in(&paths, &january)
    );

    let mut interviewer = AsksOnce::answering(Some("Alice"));
    let report = enroll(&paths, &mut interviewer);
    assert_eq!(report.named, 1);
    assert_eq!(interviewer.asked, vec![format!("{january} Unknown 1")]);
    assert!(
        speakers_in(&paths, &january).contains(&"Alice".to_string()),
        "the session that surfaced the unknown voice should be rewritten"
    );

    // The claim. February is transcribed after that answer, by a command that prompts for
    // nothing and was handed nothing but the root directory.
    assert_eq!(transcribe(&paths, &february, 0).transcribed, 1);
    let february_says = speakers_in(&paths, &february);
    assert!(
        february_says.contains(&"Alice".to_string()),
        "February should be named from the database enroll wrote, was {february_says:?}"
    );
    assert!(
        !february_says.iter().any(|who| who.starts_with("Unknown")),
        "nothing in February should still be unknown, was {february_says:?}"
    );

    // And nobody is asked again: a second `enroll` over everything finds nothing unresolved in
    // either session.
    let mut nobody = AsksOnce::answering(None);
    let report = enroll(&paths, &mut nobody);
    assert_eq!(
        nobody.asked,
        Vec::<String>::new(),
        "an enrolled person should never be asked about again"
    );
    assert_eq!(report.named, 0);
    assert_eq!(report.passed_over, 2);
}

/// The other join, for a voice too short to build a reference from: `enroll` writes the name
/// into the session, and `transcribe --force` over that same session reads it back.
///
/// This is the pair of commands the session-scoped name exists for. Naming somebody costs
/// nothing in `speakers.json` -- the file is never even created here -- and the name still
/// survives a re-transcribe, which is the thing that used to silently revert it.
#[test]
fn a_voice_too_quiet_for_a_reference_is_named_in_its_own_session_and_survives_a_re_transcribe() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let january = make_session(&paths, "20260105-090000");

    // Two seconds of speech: a real participant, and far too little to fingerprint.
    transcribe_speaking(&paths, &january, 0, 2.0, false);

    let mut interviewer = AsksOnce::answering(Some("Alex"));
    let report = enroll_offering(&paths, QUIET, &mut interviewer);

    assert_eq!(report.named, 1);
    assert_eq!(
        report.session_only, 1,
        "a voice under the reference floor should be named against its session"
    );
    assert!(
        !paths.speakers_json().exists(),
        "naming a voice this quiet must not write the enrolled database at all"
    );
    let names = SpeakerNames::read_or_empty(&paths.session(&january), &january).unwrap();
    assert_eq!(
        names
            .names
            .iter()
            .map(|row| row.name.as_str())
            .collect::<Vec<&str>>(),
        ["Alex"]
    );
    assert!(
        speakers_in(&paths, &january).contains(&"Alex".to_string()),
        "the transcript should read as the person the user named, was {:?}",
        speakers_in(&paths, &january)
    );

    // The claim. Re-transcribing from the audio is what used to throw a name like this away.
    transcribe_speaking(&paths, &january, 0, 2.0, true);
    assert!(
        speakers_in(&paths, &january).contains(&"Alex".to_string()),
        "a forced re-transcribe should keep the name, was {:?}",
        speakers_in(&paths, &january)
    );

    // And a later default run has nothing to ask: the name is an answer, even without a
    // similarity behind it.
    let mut again = AsksOnce::answering(None);
    enroll_offering(&paths, QUIET, &mut again);
    assert_eq!(
        again.asked,
        Vec::<String>::new(),
        "a voice already named must not be asked about again"
    );
}

/// The negative that keeps that test honest, and the point of the whole file being per
/// session: a name given to one meeting's voice is a claim about that meeting. The same
/// person, recorded again next month, is still a stranger -- because nothing was enrolled.
#[test]
fn a_name_given_to_one_session_does_not_name_that_voice_in_the_next() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let january = make_session(&paths, "20260105-090000");
    let february = make_session(&paths, "20260209-090000");

    transcribe_speaking(&paths, &january, 0, 2.0, false);
    enroll_offering(&paths, QUIET, &mut AsksOnce::answering(Some("Alex")));

    transcribe_speaking(&paths, &february, 0, 2.0, false);

    let february_says = speakers_in(&paths, &february);
    assert!(
        february_says.contains(&"Unknown 1".to_string()),
        "a session-scoped name must not name that voice elsewhere, was {february_says:?}"
    );
    // And February is the only thing left to ask about.
    let mut interviewer = AsksOnce::answering(None);
    enroll_offering(&paths, QUIET, &mut interviewer);
    assert_eq!(interviewer.asked, vec![format!("{february} Unknown 1")]);
}

/// A name is a claim about the voice it was given to, not about a cluster number. When
/// re-clustering redraws that voice the claim no longer has anything to attach to, and the
/// turns go back to their number rather than carrying the name onto whoever inherited the id.
#[test]
fn a_name_recorded_against_a_clustering_that_changed_is_ignored() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let january = make_session(&paths, "20260105-090000");

    transcribe_speaking(&paths, &january, 0, 2.0, false);
    enroll_offering(&paths, QUIET, &mut AsksOnce::answering(Some("Alex")));
    assert!(speakers_in(&paths, &january).contains(&"Alex".to_string()));

    // One element of the recorded centroid moved, which is what a re-clustering that redrew
    // this voice would look like from here.
    let session = paths.session(&january);
    let mut names = SpeakerNames::read_or_empty(&session, &january).unwrap();
    names.names[0].embedding[0] += 0.000_1;
    names.write(&session).unwrap();

    transcribe_speaking(&paths, &january, 0, 2.0, true);

    let says = speakers_in(&paths, &january);
    assert!(
        says.contains(&"Unknown 1".to_string()),
        "a stale name should leave the voice numbered, was {says:?}"
    );
    assert!(
        !says.contains(&"Alex".to_string()),
        "a stale name must not be applied by cluster id, was {says:?}"
    );
}

/// The negative that keeps the test above honest: what joins the two sessions is the database,
/// not the mere fact that one process transcribed both. A second meeting with a *different*
/// voice on the far end is still numbered, and still asked about.
#[test]
fn a_different_voice_in_the_next_session_is_still_unknown() {
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());
    let january = make_session(&paths, "20260105-090000");
    let february = make_session(&paths, "20260209-090000");

    transcribe(&paths, &january, 0);
    enroll(&paths, &mut AsksOnce::answering(Some("Alice")));

    transcribe(&paths, &february, 1);
    let february_says = speakers_in(&paths, &february);
    assert!(
        february_says.contains(&"Unknown 1".to_string()),
        "a voice nobody enrolled should stay numbered, was {february_says:?}"
    );

    let mut interviewer = AsksOnce::answering(None);
    enroll(&paths, &mut interviewer);
    assert_eq!(interviewer.asked, vec![format!("{february} Unknown 1")]);
}
