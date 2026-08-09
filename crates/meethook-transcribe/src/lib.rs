//! Turning recorded sessions into transcripts.
//!
//! This slice transcribes the microphone track only, so every turn is labelled `You`; the
//! participants on the speaker track arrive with diarization in a later slice.
//!
//! The batch rules live here rather than in the CLI because they are the part with teeth --
//! never redo work silently, never let one bad session take down the rest, never fetch a
//! 1.6 GB model in order to do nothing -- and they need to be testable against a fake
//! recognizer rather than a real one.

mod aec;
mod align;
mod asr;
mod audio;

use std::io::Write;
use std::path::PathBuf;

pub use align::{Alignment, NotMeasurable, measure_reference_lag};
pub use asr::{AsrSegment, SpeechToText, WhisperEngine};
pub use audio::TARGET_RATE;

use meethook_models::ModelSpec;
use meethook_session::{
    Classification, DiscoveredSession, Paths, SPEAKER_YOU, SessionId, SessionMetadata, SourceTrack,
    Transcript, Turn, discover_sessions,
};

/// The Whisper checkpoint this tool transcribes with.
///
/// large-v3-turbo: close to large-v3's accuracy at several times the speed, which is the
/// right trade for turning a finished meeting around rather than for streaming.
///
/// The URL pins an immutable revision, not `main`. `main` is a moving pointer, and a
/// republished checkpoint would turn a working install into a hash mismatch nobody asked
/// for. Both values below come from the git-LFS pointer Hugging Face serves at the `raw/`
/// path (`curl .../raw/<rev>/<file>` prints `oid sha256:...` and `size`), which is how to
/// get them again without downloading 1.6 GB.
///
/// If download size or memory ever becomes a problem, the quantized build of the same model
/// is `ggml-large-v3-turbo-q5_0.bin`, sha256
/// `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`, 574041195 bytes.
pub const WHISPER_MODEL: ModelSpec = ModelSpec {
    file_name: "ggml-large-v3-turbo.bin",
    url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/\
          5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo.bin",
    sha256: "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
    size_bytes: 1_624_555_275,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Session(#[from] meethook_session::Error),

    #[error("could not read {path} as audio: {source}")]
    Wav {
        path: PathBuf,
        #[source]
        source: hound::Error,
    },

    #[error("{path} is not a track this tool can read: {detail}")]
    UnsupportedAudio { path: PathBuf, detail: String },

    #[error("resampling failed: {0}")]
    Resample(String),

    #[error("speech recognition failed: {source}")]
    Asr {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// The recorder asserts a sane timebase before it writes one, so this means a
    /// hand-edited or corrupted `session.json` rather than a real machine.
    #[error("session.json for {session} reports an unusable clock timebase {numer}/{denom}")]
    DegenerateTimebase {
        session: SessionId,
        numer: u32,
        denom: u32,
    },

    #[error("could not load the speech recognition model: {source}")]
    Engine {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("could not write output: {0}")]
    Output(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    fn wav(path: impl Into<PathBuf>, source: hound::Error) -> Self {
        Error::Wav {
            path: path.into(),
            source,
        }
    }
}

/// Opens the recognizer. Boxed and fallible so the caller owns model acquisition, and so
/// the batch can decide whether opening one is worth doing at all.
pub type EngineFactory<'a> = dyn FnMut() -> std::result::Result<Box<dyn SpeechToText>, Box<dyn std::error::Error + Send + Sync>>
    + 'a;

/// What a batch did, so the caller can pick an exit status without re-deriving it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatchReport {
    pub transcribed: usize,
    pub skipped: usize,
    pub failed: usize,
}

/// Transcribes one session, returning the transcript without writing it.
///
/// Not writing is deliberate: the caller decides, which is what keeps skip and `--force`
/// logic in one place instead of spread between here and there.
pub fn transcribe_session(
    session: &DiscoveredSession,
    asr: &mut dyn SpeechToText,
) -> Result<Transcript> {
    let metadata = session.load_metadata()?;
    let offset = mic_offset_seconds(&metadata)?;

    let audio = audio::read_track_16k_mono(&session.paths.mic_wav())?;
    let segments = asr.transcribe(&audio)?;

    let mut turns: Vec<Turn> = segments
        .into_iter()
        .map(|segment| Turn {
            speaker: SPEAKER_YOU.to_string(),
            start: offset + segment.start_s,
            end: offset + segment.end_s,
            // Whisper's segmentation is used exactly as emitted: no re-splitting, no merging
            // of neighbours. Any re-segmentation here would have to be undone by the slice
            // that interleaves speaker-track turns.
            text: segment.text,
            source_track: SourceTrack::Mic,
            // Known, not inferred: the mic track is this machine's user by construction.
            speaker_id_confidence: None,
        })
        .collect();

    // Sorted here so every later slice inherits the invariant rather than each one having to
    // establish it when it merges in speaker-track turns.
    turns.sort_by(|a, b| a.start.total_cmp(&b.start));

    Ok(Transcript::new(session.id.clone(), turns))
}

/// Runs transcription over a selection of sessions, reporting progress to `out`.
///
/// Reads nothing from stdin and prompts for nothing, on every path. That is not incidental:
/// this command is meant to be pointed at a directory of meetings and left alone, which is
/// why naming unrecognized speakers lives in a separate `enroll` command instead.
///
/// `open_engine` is called at most once, and only when there is real work -- a rerun over an
/// already-transcribed directory must not trigger a multi-gigabyte model download in order
/// to then do nothing.
pub fn run_batch(
    paths: &Paths,
    requested: &[SessionId],
    force: bool,
    open_engine: &mut EngineFactory<'_>,
    out: &mut dyn Write,
) -> Result<BatchReport> {
    let discovered = discover_sessions(paths)?;
    let mut report = BatchReport::default();

    // An id the user named that is not on disk is worth reporting individually; transcribing
    // three of four requested sessions and exiting 0 would look like success.
    for id in requested {
        if !discovered.iter().any(|session| &session.id == id) {
            writeln!(out, "{id}  not found")?;
            report.failed += 1;
        }
    }

    let selected: Vec<&DiscoveredSession> = if requested.is_empty() {
        discovered.iter().collect()
    } else {
        discovered
            .iter()
            .filter(|session| requested.contains(&session.id))
            .collect()
    };

    if selected.is_empty() && requested.is_empty() {
        writeln!(
            out,
            "No sessions found in {}",
            paths.sessions_dir().display()
        )?;
        return Ok(report);
    }

    // Partitioned before the model is touched, so the "nothing to do" case costs nothing.
    let mut work = Vec::new();
    for session in selected {
        match session.classification {
            Classification::Orphaned => {
                writeln!(
                    out,
                    "{}  skipped: no session.json (the recorder crashed mid-session)",
                    session.id
                )?;
                report.skipped += 1;
            }
            Classification::Transcribed if !force => {
                writeln!(
                    out,
                    "{}  skipped: already transcribed (use --force to redo)",
                    session.id
                )?;
                report.skipped += 1;
            }
            _ => work.push(session),
        }
    }

    if work.is_empty() {
        writeln!(out, "Nothing to transcribe.")?;
        return Ok(report);
    }

    let mut engine = open_engine().map_err(|source| Error::Engine { source })?;

    for session in work {
        writeln!(out, "{}  transcribing...", session.id)?;
        match transcribe_and_write(session, engine.as_mut()) {
            Ok(turns) => {
                writeln!(out, "{}  {turns} turn(s)", session.id)?;
                report.transcribed += 1;
            }
            // One unreadable session must not cost the user the rest of the batch.
            Err(e) => {
                writeln!(out, "{}  failed: {e}", session.id)?;
                report.failed += 1;
            }
        }
    }

    Ok(report)
}

fn transcribe_and_write(session: &DiscoveredSession, asr: &mut dyn SpeechToText) -> Result<usize> {
    let transcript = transcribe_session(session, asr)?;
    transcript.write(&session.paths)?;
    Ok(transcript.turns.len())
}

/// Seconds from session start to the microphone track's first sample.
///
/// Session start is the earlier of the two tracks' first samples, not `session.json`'s
/// `start_time`: that field is a wall-clock instant captured when the directory was created,
/// with no recorded pairing to mach tick space, so it cannot be compared to either track's
/// `host_ticks`. Using the earliest track instead keeps every turn non-negative once
/// speaker-track turns join the same timeline.
///
/// The conversion is exact -- integer ticks scaled by the machine's rational timebase in
/// `u128`, rounded once at the end. Going through `f64` first would lose the low bits of a
/// mach tick count within a day of uptime.
///
/// This is metadata alignment only. Correcting the acoustic offset between the two capture
/// APIs has to be measured from the signals themselves and is not this function's business.
fn mic_offset_seconds(metadata: &SessionMetadata) -> Result<f64> {
    let mic = metadata.mic;
    if mic.timebase_numer == 0 || mic.timebase_denom == 0 {
        return Err(Error::DegenerateTimebase {
            session: metadata.session_id.clone(),
            numer: mic.timebase_numer,
            denom: mic.timebase_denom,
        });
    }

    let origin = mic.host_ticks.min(metadata.speaker.host_ticks);
    let delta = u128::from(mic.host_ticks - origin);
    let nanos = delta * u128::from(mic.timebase_numer) / u128::from(mic.timebase_denom);
    Ok(nanos as f64 / 1e9)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use hound::{SampleFormat, WavSpec, WavWriter};
    use jiff::Timestamp;
    use meethook_session::{SessionPaths, TrackSync};

    use super::*;

    /// Apple Silicon's timebase. 125/3 rather than Intel's 1/1 is exactly the ratio that
    /// makes an unscaled tick count look plausible while being 41x wrong.
    const NUMER: u32 = 125;
    const DENOM: u32 = 3;

    struct FakeAsr {
        segments: Vec<AsrSegment>,
    }

    impl SpeechToText for FakeAsr {
        fn transcribe(&mut self, _audio: &[f32]) -> Result<Vec<AsrSegment>> {
            Ok(self.segments.clone())
        }
    }

    fn fake_engine() -> Box<dyn SpeechToText> {
        Box::new(FakeAsr {
            segments: vec![segment(0.0, 1.0, "hello")],
        })
    }

    fn segment(start: f64, end: f64, text: &str) -> AsrSegment {
        AsrSegment {
            start_s: start,
            end_s: end,
            text: text.to_string(),
        }
    }

    fn write_silence(path: &Path, seconds: f32) {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        for _ in 0..(48_000.0 * seconds) as usize {
            writer.write_sample(0.0f32).unwrap();
        }
        writer.finalize().unwrap();
    }

    fn metadata(id: &SessionId, mic_ticks: u64, speaker_ticks: u64) -> SessionMetadata {
        let sync = |ticks| TrackSync {
            host_ticks: ticks,
            timebase_numer: NUMER,
            timebase_denom: DENOM,
        };
        SessionMetadata::new(
            id.clone(),
            Timestamp::from_second(1_770_000_000).unwrap(),
            sync(mic_ticks),
            sync(speaker_ticks),
        )
    }

    /// Builds a valid session directory. `mic_lag_ticks` is how much later than the speaker
    /// track the mic track delivered its first sample.
    fn make_session(paths: &Paths, id: &str, mic_lag_ticks: u64) -> DiscoveredSession {
        let id = SessionId::parse(id).unwrap();
        let session_paths = paths.session(&id);
        std::fs::create_dir_all(session_paths.dir()).unwrap();
        write_silence(&session_paths.mic_wav(), 0.25);
        write_silence(&session_paths.speaker_wav(), 0.25);

        // A tick magnitude large enough that an accidental f64 round-trip would lose bits.
        let base = 900_000_000_000u64;
        metadata(&id, base + mic_lag_ticks, base)
            .write(&session_paths.session_json())
            .unwrap();

        DiscoveredSession {
            id,
            paths: session_paths,
            classification: Classification::Valid,
        }
    }

    /// WAVs but no `session.json`: the recorder died mid-session.
    fn make_orphan(paths: &Paths, id: &str) {
        let session_paths = paths.session(&SessionId::parse(id).unwrap());
        std::fs::create_dir_all(session_paths.dir()).unwrap();
        write_silence(&session_paths.mic_wav(), 0.25);
    }

    /// Runs a batch against a single fake engine, reporting how many times the engine was
    /// opened so "no work means no model" can be asserted.
    fn run(paths: &Paths, ids: &[&str], force: bool) -> (BatchReport, usize, String) {
        let requested: Vec<SessionId> =
            ids.iter().map(|id| SessionId::parse(id).unwrap()).collect();
        let mut opened = 0usize;
        let mut out = Vec::new();
        let report = {
            let mut factory = || {
                opened += 1;
                Ok(fake_engine())
            };
            run_batch(paths, &requested, force, &mut factory, &mut out).unwrap()
        };
        (report, opened, String::from_utf8(out).unwrap())
    }

    #[test]
    fn a_mic_track_that_started_later_is_offset_onto_the_session_timeline() {
        // 1_000_000 ticks at 125/3 ns per tick is 41.666... ms.
        let id = SessionId::parse("20260809-052600").unwrap();
        let offset =
            mic_offset_seconds(&metadata(&id, 900_000_001_000_000, 900_000_000_000_000)).unwrap();
        assert!(
            (offset - 0.041_666_666).abs() < 1e-9,
            "offset was {offset} s"
        );
    }

    #[test]
    fn a_mic_track_that_started_first_defines_time_zero() {
        let id = SessionId::parse("20260809-052600").unwrap();
        let offset =
            mic_offset_seconds(&metadata(&id, 900_000_000_000_000, 900_000_005_000_000)).unwrap();
        assert_eq!(offset, 0.0);
    }

    #[test]
    fn every_turn_is_labelled_you_on_the_mic_track_with_no_confidence() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600", 1_000_000);

        let mut asr = FakeAsr {
            segments: vec![segment(2.0, 3.0, "second"), segment(0.5, 1.0, "first")],
        };
        let transcript = transcribe_session(&session, &mut asr).unwrap();

        assert_eq!(transcript.turns.len(), 2);
        // Sorted, and every timestamp shifted by the mic's 41.67 ms late start.
        assert_eq!(transcript.turns[0].text, "first");
        assert!((transcript.turns[0].start - 0.541_666_6).abs() < 1e-5);
        assert!((transcript.turns[1].start - 2.041_666_6).abs() < 1e-5);
        for turn in &transcript.turns {
            assert_eq!(turn.speaker, SPEAKER_YOU);
            assert_eq!(turn.source_track, SourceTrack::Mic);
            assert_eq!(turn.speaker_id_confidence, None);
        }
    }

    #[test]
    fn a_session_with_no_recognized_speech_still_produces_both_files() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600", 0);

        let mut asr = FakeAsr {
            segments: Vec::new(),
        };
        let transcript = transcribe_session(&session, &mut asr).unwrap();
        assert!(transcript.turns.is_empty());

        transcript.write(&session.paths).unwrap();
        assert!(session.paths.transcript_json().is_file());
        assert!(session.paths.transcript_md().is_file());
    }

    #[test]
    fn multiple_sessions_transcribe_in_one_invocation() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600", 0);
        make_session(&paths, "20260809-052700", 0);

        let (report, opened, _) = run(&paths, &["20260809-052600", "20260809-052700"], false);

        assert_eq!(report.transcribed, 2);
        assert_eq!(report.failed, 0);
        // One model load for the whole batch, not one per session.
        assert_eq!(opened, 1);
    }

    #[test]
    fn an_already_transcribed_session_is_skipped_until_force_is_passed() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600", 0);

        let (first, _, _) = run(&paths, &[], false);
        assert_eq!(first.transcribed, 1);

        let (second, opened, output) = run(&paths, &[], false);
        assert_eq!(second.transcribed, 0);
        assert_eq!(second.skipped, 1);
        assert!(output.contains("already transcribed"), "{output}");
        // The sharp edge: a no-op rerun must not fetch a 1.6 GB model in order to do nothing.
        assert_eq!(opened, 0, "a fully skipped batch must not open the model");

        let (third, _, _) = run(&paths, &[], true);
        assert_eq!(third.transcribed, 1);
        assert_eq!(third.skipped, 0);
    }

    #[test]
    fn an_orphaned_session_is_warned_about_while_its_neighbours_still_transcribe() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_orphan(&paths, "20260809-052500");
        make_session(&paths, "20260809-052600", 0);

        let (report, _, output) = run(&paths, &[], false);

        assert!(output.contains("no session.json"), "{output}");
        assert_eq!(report.skipped, 1);
        assert_eq!(report.transcribed, 1);
        // An orphan is a normal state, not a failure.
        assert_eq!(report.failed, 0);

        assert!(
            SessionPaths::new(paths.sessions_dir().join("20260809-052600"))
                .transcript_json()
                .is_file()
        );
        assert!(
            !SessionPaths::new(paths.sessions_dir().join("20260809-052500"))
                .transcript_json()
                .exists(),
            "an orphan must never get a transcript built from unverifiable audio"
        );
    }

    #[test]
    fn a_session_that_fails_does_not_take_the_batch_down_with_it() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let broken = make_session(&paths, "20260809-052500", 0);
        std::fs::write(broken.paths.mic_wav(), b"not a wav file at all").unwrap();
        make_session(&paths, "20260809-052600", 0);

        let (report, _, output) = run(&paths, &[], false);

        assert_eq!(report.failed, 1, "{output}");
        assert_eq!(report.transcribed, 1, "{output}");
    }

    #[test]
    fn a_requested_session_that_does_not_exist_is_named_rather_than_ignored() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600", 0);

        let (report, _, output) = run(&paths, &["20260809-052600", "20260809-999999"], false);

        assert!(output.contains("20260809-999999  not found"), "{output}");
        assert_eq!(report.transcribed, 1);
        assert_eq!(report.failed, 1);
    }
}
