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

use meethook_enroll::{Answer, EnrollReport, Interviewer, UnknownVoice, run_enroll, write_clip};
use meethook_session::{
    Paths, RepresentativeSegment, SessionId, SessionMetadata, SpeakerCluster, TrackSync, Transcript,
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
}

impl Diarize for FakeDiarizer {
    fn diarize(&mut self, _speaker_16k_mono: &[f32]) -> Result<Diarization> {
        Ok(Diarization {
            clusters: vec![SpeakerCluster {
                id: 0,
                embedding: self.embedding.clone(),
                speech_seconds: 1.0,
                first_spoke_seconds: 0.0,
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

/// Transcribes exactly one session, as the CLI would, with `who` the only voice on its far end.
fn transcribe(paths: &Paths, id: &SessionId, who: u32) -> BatchReport {
    let mut factory = || {
        Ok(Engines {
            asr: Box::new(FakeAsr),
            diarizer: Box::new(FakeDiarizer {
                embedding: voice(who),
            }),
        })
    };
    run_batch(
        paths,
        std::slice::from_ref(id),
        false,
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
    fn identify(&mut self, voice: &UnknownVoice<'_>) -> Answer {
        self.asked
            .push(format!("{} {}", voice.session, voice.label));
        match self.answer.take() {
            Some(name) => Answer::Named(name),
            None => Answer::Skip,
        }
    }
}

fn enroll(paths: &Paths, interviewer: &mut dyn Interviewer) -> EnrollReport {
    run_enroll(paths, &[], interviewer, &mut std::io::sink()).unwrap()
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
