//! Mic isolation, end to end, with the real voice isolator and the real speech detector and
//! no listening step.
//!
//! The earlier Python trial of the isolator removed about 0.006% of the energy on already-clean
//! tracks, so a stage that silently skipped the model would look fine to a casual check. Every
//! test here therefore feeds the isolator a track with broadband noise added on top of the
//! Apache-2.0 sample `sample_14.wav` (from `inworld-ai/voice-isolator`) and asserts that the
//! noise is measurably *gone* from what recognition receives.
//!
//! All tests are `#[ignore]`: they load the 260 MB isolator (downloaded on first use into
//! `$MEETHOOK_ROOT/models`, default `~/meethook/models`) and each isolation run takes about a
//! minute on the CPU. Run them with
//!
//! ```text
//! cargo test --release -p meethook-transcribe --test mic_isolation_e2e -- --ignored --nocapture
//! ```

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use jiff::Timestamp;
use meethook_models::ensure_model;
use meethook_session::{
    Classification, CleaningRecord, DiscoveredSession, EnrolledSpeakers, IsolationOutcome, Paths,
    SessionId, SessionMetadata, TrackSync,
};
use meethook_transcribe::{
    AsrSegment, Diarization, Diarize, FIRERED_VAD_MODEL, FireRedVad, GateSpeechFinder, Isolation,
    IsolationStage, SpeechToText, TARGET_RATE, VoiceIsolator, mixdown, read_track_16k_mono,
    transcribe_session_isolating,
};

const RATE_48K: u32 = 48_000;
/// Root-mean-square level of the broadband noise added to the sample. The sample's own speech
/// sits well above this, so the speech regions stay detectable while the noise floor between
/// words is clearly the noise's.
const NOISE_RMS: f32 = 0.02;
/// How much lower the noise floor must be in what recognition heard.
const REQUIRED_REDUCTION_DB: f64 = 6.0;

type Heard = Rc<RefCell<Vec<Vec<f32>>>>;

/// Records every slice of audio it is given, then returns no text.
struct RecordingAsr(Heard);

impl SpeechToText for RecordingAsr {
    fn transcribe(&mut self, audio: &[f32]) -> meethook_transcribe::Result<Vec<AsrSegment>> {
        self.0.borrow_mut().push(audio.to_vec());
        Ok(Vec::new())
    }
}

struct NoDiarizer;

impl Diarize for NoDiarizer {
    fn diarize(&mut self, _: &[f32]) -> meethook_transcribe::Result<Diarization> {
        Ok(Diarization::default())
    }
}

fn models_dir() -> PathBuf {
    let root = match std::env::var_os("MEETHOOK_ROOT") {
        Some(root) => PathBuf::from(root),
        None => std::env::home_dir()
            .expect("no home directory; set MEETHOOK_ROOT")
            .join("meethook"),
    };
    root.join("models")
}

fn sample_48k() -> Vec<f32> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/voice-isolator/sample_14.wav");
    let mut reader = hound::WavReader::open(&path).expect("sample_14.wav must be readable");
    let spec = reader.spec();
    assert_eq!((spec.channels, spec.sample_rate), (1, RATE_48K));
    match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / scale)
                .collect()
        }
    }
}

/// Deterministic white noise at `rms`: xorshift64, so the fixture is the same on every run and
/// a marginal threshold fails every time rather than once a month.
fn noise(len: usize, rms: f32) -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    // Uniform on [-1, 1) has an rms of 1/sqrt(3).
    let scale = rms * 3.0f32.sqrt();
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / 8_388_608.0 - 1.0) * scale
        })
        .collect()
}

fn write_wav_48k(path: &Path, samples: &[f32]) {
    let mut writer = hound::WavWriter::create(
        path,
        hound::WavSpec {
            channels: 1,
            sample_rate: RATE_48K,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap();
    for &s in samples {
        writer.write_sample(s).unwrap();
    }
    writer.finalize().unwrap();
}

/// A session whose mic track is the sample plus broadband noise and whose speaker track is a
/// few seconds of a quiet tone: unrelated to the mic, so the echo canceller has nothing to
/// remove and `mic.cleaned.flac` is the noisy mic.
fn make_session(root: &Path) -> DiscoveredSession {
    let id = SessionId::parse("20260809-052600").unwrap();
    let paths = Paths::new(root).session(&id);
    std::fs::create_dir_all(paths.dir()).unwrap();

    let speech = sample_48k();
    let mic: Vec<f32> = speech
        .iter()
        .zip(noise(speech.len(), NOISE_RMS))
        .map(|(s, n)| s + n)
        .collect();
    write_wav_48k(&paths.mic_wav(), &mic);

    let tone: Vec<f32> = (0..RATE_48K as usize * 4)
        .map(|i| 0.05 * (std::f32::consts::TAU * 900.0 * i as f32 / RATE_48K as f32).sin())
        .collect();
    write_wav_48k(&paths.speaker_wav(), &tone);

    let sync = TrackSync {
        host_ticks: 900_000_000_000,
        timebase_numer: 125,
        timebase_denom: 3,
    };
    SessionMetadata::new(
        id.clone(),
        Timestamp::from_second(1_770_000_000).unwrap(),
        sync,
        sync,
    )
    .write(&paths.session_json())
    .unwrap();

    DiscoveredSession {
        id,
        paths,
        classification: Classification::Valid,
    }
}

fn real_stage() -> IsolationStage {
    let models = models_dir();
    ensure_model(&models, VoiceIsolator::required_model(), &mut |_, _| {})
        .expect("the voice isolator must download");
    let vad = ensure_model(&models, &FIRERED_VAD_MODEL, &mut |_, _| {})
        .expect("the VAD model must download");
    IsolationStage {
        isolator: Box::new(VoiceIsolator::open(&models).expect("the isolator must load")),
        finder: Box::new(GateSpeechFinder(FireRedVad::load(&vad).unwrap())),
    }
}

/// Runs `transcribe` on a fresh session and returns what recognition heard on the mic track,
/// the cleaned mic track on disk, and the recorded isolation outcome.
fn run(isolation: &mut Isolation) -> (Vec<f32>, Vec<f32>, IsolationOutcome) {
    let dir = tempfile::tempdir().unwrap();
    let session = make_session(dir.path());
    let heard: Heard = Rc::default();
    transcribe_session_isolating(
        &session,
        &mut RecordingAsr(heard.clone()),
        &mut NoDiarizer,
        isolation,
        &EnrolledSpeakers::new(Vec::new()),
        mixdown::Settings::default(),
        &mut std::io::stderr(),
    )
    .expect("transcribe must succeed");

    let mic_heard = heard.borrow()[0].clone();
    let cleaned = read_track_16k_mono(&session.paths.mic_cleaned_flac()).unwrap();
    let outcome = CleaningRecord::read_if_present(&session.paths)
        .unwrap()
        .expect("cleaning.json must exist")
        .isolation
        .expect("cleaning.json must record the isolation outcome");
    (mic_heard, cleaned, outcome)
}

/// The noise floor of `audio` inside `regions`, in dB: the mean square over the quietest 20%
/// of the 20 ms frames that lie wholly inside a region. The quietest frames are the gaps
/// between words, which is where added broadband noise is the only thing left to hear; speech
/// frames would measure the speech instead.
fn noise_floor_db(audio: &[f32], regions: &[meethook_transcribe::SpeechRegion]) -> f64 {
    let frame = (TARGET_RATE as usize) * 20 / 1000;
    let mut energies: Vec<f64> = Vec::new();
    for region in regions {
        let start = (region.start_s * f64::from(TARGET_RATE)).ceil() as usize;
        let end = ((region.end_s * f64::from(TARGET_RATE)).floor() as usize).min(audio.len());
        let mut at = start;
        while at + frame <= end {
            let e = audio[at..at + frame]
                .iter()
                .map(|&s| f64::from(s).powi(2))
                .sum::<f64>()
                / frame as f64;
            energies.push(e);
            at += frame;
        }
    }
    assert!(!energies.is_empty(), "no complete frame inside any region");
    energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let quiet = &energies[..(energies.len() / 5).max(1)];
    10.0 * (quiet.iter().sum::<f64>() / quiet.len() as f64)
        .max(1e-20)
        .log10()
}

fn speech_regions_of(audio: &[f32]) -> Vec<meethook_transcribe::SpeechRegion> {
    real_stage().finder.speech_regions(audio).unwrap()
}

#[test]
#[ignore = "downloads the 260 MB voice isolator; about a minute per isolation run"]
fn isolation_lowers_the_noise_floor_and_the_flag_turns_it_off() {
    // On: the model ran, and recognition heard a quieter noise floor than the cleaned track
    // holds.
    let (heard, cleaned, outcome) = run(&mut Isolation::Ready(real_stage()));
    assert!(
        matches!(outcome, IsolationOutcome::Ran { regions, .. } if regions > 0),
        "isolation was recorded as {outcome:?}"
    );
    assert_eq!(heard.len(), cleaned.len(), "same length and timeline");
    assert_ne!(heard, cleaned, "recognition heard the un-isolated track");

    let regions = speech_regions_of(&cleaned);
    assert!(!regions.is_empty(), "the detector found no speech");
    let before = noise_floor_db(&cleaned, &regions);
    let after = noise_floor_db(&heard, &regions);
    eprintln!(
        "noise floor inside speech regions: cleaned {before:.1} dB, heard {after:.1} dB, \
         reduction {:.1} dB over {} regions",
        before - after,
        regions.len()
    );
    assert!(
        after <= before - REQUIRED_REDUCTION_DB,
        "noise floor fell only {:.1} dB ({before:.1} -> {after:.1})",
        before - after
    );

    // Off: recognition heard exactly the cleaned track, and the record says disabled.
    let (heard_off, cleaned_off, outcome_off) = run(&mut Isolation::Off);
    assert_eq!(outcome_off, IsolationOutcome::DisabledByFlag);
    assert_eq!(
        heard_off, cleaned_off,
        "with --no-isolate recognition must hear mic.cleaned.flac unchanged"
    );
    // The cleaned track itself does not depend on the flag.
    assert_eq!(cleaned_off, cleaned);
}

#[test]
#[ignore = "downloads the 260 MB voice isolator; about a minute per isolation run"]
fn two_isolation_runs_over_the_same_input_are_identical() {
    let (first, ..) = run(&mut Isolation::Ready(real_stage()));
    let (second, ..) = run(&mut Isolation::Ready(real_stage()));
    assert_eq!(first, second, "isolation is not deterministic");
}
