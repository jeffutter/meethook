//! From a wav file nobody recorded with meethook to a name in `speakers.json`.
//!
//! This is the chain TASK-014's calibration has to walk, with only the two model-backed
//! engines faked:
//!
//! *`build_session` over ordinary wav files, then the real `transcribe`, then the real
//! `enroll` -- and the reference that lands in `speakers.json` is derived from the audio that
//! was supplied, not from anything a test handed across in memory.*
//!
//! Every step after the build is the production one. Nothing is passed between the commands
//! except the root directory, exactly as the CLI passes it, and the session directory is read
//! back off disk by each in turn. What is faked is the pair of ONNX graphs, and the diarizer's
//! stand-in still *computes* its embedding from the samples it was handed -- see
//! [`fingerprint`] -- so "the stored reference derives from the supplied audio" is a claim this
//! file can actually decide rather than assert about a constant.
//!
//! What it deliberately does not decide is whether one *human* voice recorded twice lands
//! inside `IDENTIFY_DISTANCE`. That is a question about the embedding model and real speech,
//! and it is the measurement the tooling here exists to make possible.
//!
//! No audio file is checked in. Every fixture is synthesised into a `tempfile::tempdir()` at
//! run time: third-party corpus audio and real meeting recordings are both other people
//! speaking.

use std::f32::consts::TAU;
use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavSpec, WavWriter};
use meethook_enroll::{
    Answer, EnrollRules, Enrolment, Interviewer, Offer, Sessions, Voice, run_enroll,
};
use meethook_session::{
    EnrolledSpeakers, Paths, RepresentativeSegment, SessionId, SpeakerCluster, SpeakerClusters,
    Transcript, TranscriptTemplate,
};
use meethook_transcribe::{
    AsrSegment, Diarization, Diarize, Engines, Result, SpeakerTurn, SpeechToText, TARGET_RATE,
    build_session, read_track_16k_mono, run_batch,
};

/// The two tones the fixtures are built from, and the frequencies [`fingerprint`] probes at.
/// Sharing the list is what makes two different sources land in near-orthogonal directions
/// rather than merely at different lengths.
const PROBE_HZ: [f32; 2] = [220.0, 440.0];

/// A voice fingerprint with no model in it: the track projected onto a quadrature pair of
/// sinusoids at each of [`PROBE_HZ`], normalised to the unit length every consumer of an
/// embedding assumes.
///
/// The point is that it is a *function of the audio*. A fake diarizer returning a canned
/// vector -- which is what every other test in this crate wants, and rightly -- could not tell
/// a session built from the supplied wav files apart from one built from silence.
fn fingerprint(audio: &[f32]) -> Vec<f32> {
    let mut raw = Vec::with_capacity(PROBE_HZ.len() * 2);
    for hz in PROBE_HZ {
        let (mut cosine, mut sine) = (0.0f32, 0.0f32);
        for (i, sample) in audio.iter().enumerate() {
            let phase = i as f32 / TARGET_RATE as f32 * hz * TAU;
            cosine += sample * phase.cos();
            sine += sample * phase.sin();
        }
        raw.push(cosine);
        raw.push(sine);
    }

    let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        // A silent track has no direction. Any fixed unit vector will do; what matters is
        // that it is not NaN, which would poison every comparison downstream.
        return vec![1.0, 0.0, 0.0, 0.0];
    }
    raw.iter().map(|v| v / norm).collect()
}

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

/// One cluster, whose embedding is computed from the track it was handed.
struct FingerprintDiarizer;

impl Diarize for FingerprintDiarizer {
    fn diarize(&mut self, speaker_16k_mono: &[f32]) -> Result<Diarization> {
        let seconds = speaker_16k_mono.len() as f64 / f64::from(TARGET_RATE);
        Ok(Diarization {
            clusters: vec![SpeakerCluster {
                id: 0,
                embedding: fingerprint(speaker_16k_mono),
                speech_seconds: seconds,
                first_spoke_seconds: 0.0,
                heard_at_once_with: Vec::new(),
                representatives: vec![RepresentativeSegment {
                    start: 0.0,
                    end: seconds.min(2.0),
                }],
            }],
            turns: vec![SpeakerTurn {
                start_s: 0.0,
                end_s: seconds,
                cluster: 0,
            }],
        })
    }
}

/// Answers the first voice it is shown and skips the rest.
struct AsksOnce {
    answer: Option<String>,
    asked: Vec<String>,
}

impl Interviewer for AsksOnce {
    fn identify(&mut self, voice: &Voice<'_>) -> Answer {
        self.asked.push(voice.attribution.label().to_string());
        // A voice a user cannot hear is a voice they cannot name, so the clip reaching the
        // prompt is part of what "enroll accepts this session" has to mean.
        assert!(
            !voice.clip.is_empty(),
            "a constructed session must still yield a playable clip"
        );
        match self.answer.take() {
            Some(name) => Answer::Named {
                name,
                anyway: false,
            },
            None => Answer::Skip,
        }
    }
}

/// 48 kHz stereo 16-bit PCM: what a corpus ships, and the furthest a source header gets from
/// the 16 kHz mono float the recorder writes.
fn write_source(path: &Path, hz: f32, seconds: f64) {
    let spec = WavSpec {
        channels: 2,
        sample_rate: 48_000,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut writer = WavWriter::create(path, spec).unwrap();
    for i in 0..(48_000.0 * seconds) as usize {
        let sample = ((i as f32 / 48_000.0 * hz * TAU).sin() * 0.5 * 32_767.0) as i16;
        writer.write_sample(sample).unwrap();
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
}

/// Transcribes one session exactly as the CLI would.
fn transcribe(paths: &Paths, id: &SessionId) {
    let mut factory = || {
        Ok(Engines {
            asr: Box::new(FakeAsr),
            diarizer: Box::new(FingerprintDiarizer),
        })
    };
    let report = run_batch(
        paths,
        std::slice::from_ref(id),
        false,
        &TranscriptTemplate::resolve(paths, None).unwrap(),
        meethook_transcribe::mixdown::Settings::default(),
        &mut factory,
        &mut std::io::sink(),
    )
    .unwrap();
    assert_eq!(report.failed, 0, "{id} should transcribe cleanly");
    assert_eq!(report.transcribed, 1);
}

/// Builds a session under `root` from one synthesised source, and returns it.
///
/// Six seconds rather than a token three: `FingerprintDiarizer` reports the track's own
/// duration as the voice's talk time, and `enroll` stores a reference in `speakers.json` only
/// for a voice above its reference floor -- which is exactly what these tests are about. A
/// three-second source would have the name recorded against the session instead, and the
/// assertions below would be checking a different path than the one they describe.
fn build(root: &Paths, dir: &Path, name: &str, hz: f32) -> SessionId {
    let source = dir.join(format!("{name}.wav"));
    write_source(&source, hz, 6.0);
    build_session(root, &[source], &[]).unwrap().id
}

/// Acceptance criteria #1 and #3, as far as they can be reached without a model: a wav file
/// meethook never recorded goes through the real `transcribe` and the real `enroll` without
/// error, and what lands in `speakers.json` is a reference computed from that audio.
#[test]
fn a_session_built_from_a_wav_file_transcribes_enrolls_and_stores_that_audio() {
    let sources = tempfile::tempdir().unwrap();
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());

    let session = build(&paths, sources.path(), "alice", PROBE_HZ[1]);
    transcribe(&paths, &session);

    // `transcribe` accepted it: both files `enroll` needs are there, and the transcript names
    // the constructed session's one voice.
    let transcript = Transcript::read(&paths.session(&session).transcript_json()).unwrap();
    assert!(
        transcript
            .turns
            .iter()
            .any(|turn| turn.speaker == "Unknown 1"),
        "the far-end voice should be numbered before anybody is enrolled, was {:?}",
        transcript.turns
    );

    let mut interviewer = AsksOnce {
        answer: Some("Alice".to_string()),
        asked: Vec::new(),
    };
    let report = run_enroll(
        &paths,
        &[],
        EnrollRules {
            selector: None,
            offer: Offer::default(),
            sessions: Sessions::default(),
            enrolment: Enrolment::default(),
            one_speaker: None,
            template: &TranscriptTemplate::resolve(&paths, None).unwrap(),
        },
        &mut interviewer,
        &mut meethook_enroll::Lines::new(&mut std::io::sink()),
    )
    .unwrap();
    assert_eq!(report.failed, 0);
    assert_eq!(report.named, 1);
    assert_eq!(interviewer.asked, ["Unknown 1"]);

    // The claim. The stored reference is what `fingerprint` computes over the audio that was
    // written into the session -- recomputed here from `speaker.wav` on disk, so the whole
    // chain (source wav -> build -> transcribe -> enroll -> speakers.json) is what is being
    // checked, not a value this test handed anybody.
    let stored = EnrolledSpeakers::read_or_empty(&paths).unwrap();
    assert_eq!(stored.speakers.len(), 1);
    assert_eq!(stored.speakers[0].name, "Alice");

    let written = read_track_16k_mono(&paths.session(&session).speaker_wav()).unwrap();
    let expected = fingerprint(&written);
    assert_eq!(stored.speakers[0].embedding.len(), expected.len());
    for (got, want) in stored.speakers[0].embedding.iter().zip(&expected) {
        assert!(
            (got - want).abs() < 1e-6,
            "stored {:?} should be the fingerprint of speaker.wav {expected:?}",
            stored.speakers[0].embedding
        );
    }

    // The other half of the same claim, and the one the trial-list runner rests on: the stored
    // reference is element for element *one cluster's* embedding, copied rather than averaged
    // or renormalized on the way through. `speaker-trials` takes an item's dominant cluster as
    // that item's voice for exactly this reason -- if enrollment did anything else to the
    // vector, the distances it measures would not be the distances identification decides on.
    let clusters = SpeakerClusters::read(&paths.session(&session).speaker_clusters_json()).unwrap();
    assert_eq!(clusters.clusters.len(), 1);
    assert_eq!(
        stored.speakers[0].embedding, clusters.clusters[0].embedding,
        "enroll must store the cluster's own embedding, unmodified"
    );

    // And the transcript is rewritten in that name, which is what a reader sees.
    let transcript = Transcript::read(&paths.session(&session).transcript_json()).unwrap();
    assert!(
        transcript.turns.iter().any(|turn| turn.speaker == "Alice"),
        "{:?}",
        transcript.turns
    );
}

/// The negative that keeps the test above honest: what is stored tracks the audio, so a second
/// session built from *different* audio embeds differently.
///
/// This is the shape TASK-014's measurement takes -- a reference from one recording, a distance
/// to another - reduced to the part that holds without a model. Whether two recordings of one
/// person land inside `IDENTIFY_DISTANCE` is exactly what no fake can answer.
#[test]
fn two_sessions_built_from_different_audio_embed_differently() {
    let sources = tempfile::tempdir().unwrap();
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());

    let low = build(&paths, sources.path(), "low", PROBE_HZ[0]);
    let high = build(&paths, sources.path(), "high", PROBE_HZ[1]);
    assert_ne!(low, high, "two builds must not collide on one directory");

    transcribe(&paths, &low);
    transcribe(&paths, &high);

    let embedding = |id: &SessionId| -> Vec<f32> {
        let clusters = SpeakerClusters::read(&paths.session(id).speaker_clusters_json()).unwrap();
        assert_eq!(clusters.clusters.len(), 1);
        clusters.clusters[0].embedding.clone()
    };

    let cosine: f32 = embedding(&low)
        .iter()
        .zip(&embedding(&high))
        .map(|(a, b)| a * b)
        .sum();
    assert!(
        1.0 - cosine > 0.5,
        "two different sources should not embed to the same direction, distance was {}",
        1.0 - cosine
    );
}

/// Acceptance criterion #4's precondition, and the one thing a scratch root exists for: the
/// build writes nothing outside the root it was handed.
#[test]
fn building_and_transcribing_touches_only_the_root_it_was_given() {
    let sources = tempfile::tempdir().unwrap();
    let root = tempfile::tempdir().unwrap();
    let paths = Paths::new(root.path());

    let session = build(&paths, sources.path(), "alice", PROBE_HZ[1]);
    transcribe(&paths, &session);
    run_enroll(
        &paths,
        &[],
        EnrollRules {
            selector: None,
            offer: Offer::default(),
            sessions: Sessions::default(),
            enrolment: Enrolment::default(),
            one_speaker: None,
            template: &TranscriptTemplate::resolve(&paths, None).unwrap(),
        },
        &mut AsksOnce {
            answer: Some("Alice".to_string()),
            asked: Vec::new(),
        },
        &mut meethook_enroll::Lines::new(&mut std::io::sink()),
    )
    .unwrap();

    let mut found: Vec<PathBuf> = std::fs::read_dir(root.path())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    found.sort();
    assert_eq!(
        found,
        [paths.sessions_dir(), paths.speakers_json()]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    );

    // The source directory is left exactly as it was found: one wav, unmodified.
    let sources: Vec<PathBuf> = std::fs::read_dir(sources.path())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(sources.len(), 1);
}
