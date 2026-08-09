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
mod fbank;
mod onnx;
mod segmentation;
mod speakers;

use std::io::Write;
use std::path::PathBuf;

pub use aec::{Cleaned, Cleaning, PassThrough, cancel_bleed};
pub use align::{Alignment, NotMeasurable, measure_reference_lag};
pub use asr::{AsrSegment, SpeechToText, WhisperEngine};
pub use audio::{TARGET_RATE, read_track_16k_mono};
pub use onnx::{Loaded, open_session};
pub use segmentation::{LocalTurn, segment_speaker_track};
pub use speakers::{Clustering, cluster_speaker_turns};

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

/// The speaker-segmentation graph diarization runs over the speaker track.
///
/// pyannote segmentation 3.0, exported to ONNX. It consumes a fixed-length window of raw
/// audio and emits, per frame, a distribution over the *powerset* of up to three concurrent
/// speakers -- which is why the output's last dimension is 7 (silence, three singles, three
/// pairs) rather than a speaker count.
///
/// Graph contract, asserted in [`onnx`]'s smoke test:
/// input `input_values` f32 `[batch_size, num_channels, num_samples]`;
/// output `logits` f32 `[batch_size, num_frames, 7]`.
///
/// The file name is the repository's, not the repository's `model.onnx`: the models
/// directory is flat and shared, so a generic name would collide with the next ONNX model
/// added.
///
/// Like [`WHISPER_MODEL`], the URL pins an immutable revision and the hash and size come
/// from the git-LFS pointer Hugging Face serves at the `raw/` path, so bumping the revision
/// does not require downloading the weights to re-derive them.
pub const SEGMENTATION_MODEL: ModelSpec = ModelSpec {
    file_name: "pyannote-segmentation-3.0.onnx",
    url: "https://huggingface.co/onnx-community/pyannote-segmentation-3.0/resolve/\
          733a93b6473d019a773298e08cefa686894b1854/onnx/model.onnx",
    sha256: "057ee564753071c0b09b5b611648b50ac188d50846bff5f01e9f7bbf1591ea25",
    size_bytes: 5_986_908,
};

/// The speaker-embedding graph that turns a segment of speech into a voice fingerprint.
///
/// WeSpeaker's VoxCeleb ResNet34-LM. It takes fbank features rather than raw audio -- 80
/// mel bins per frame -- and returns one 256-dimensional embedding per utterance, which is
/// what clustering and enrollment compare.
///
/// Graph contract, asserted in [`onnx`]'s smoke test:
/// input `feats` f32 `[B, T, 80]`; output `embs` f32 `[B, 256]`.
pub const EMBEDDING_MODEL: ModelSpec = ModelSpec {
    file_name: "wespeaker-voxceleb-resnet34-LM.onnx",
    url: "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/\
          f0c48c298fd835726c27956a5d617bad7115627e/voxceleb_resnet34_LM.onnx",
    sha256: "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068",
    size_bytes: 26_530_309,
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

    /// Everything that can go wrong once the segmentation graph is already loaded:
    /// inference itself, and an output that does not match the shape the decoder reads.
    /// Loading is [`Error::Onnx`].
    #[error("speaker segmentation failed: {0}")]
    Segmentation(String),

    /// The same, one graph later: inference through the embedding model, or an output that
    /// is not the 256-dimensional vector clustering compares.
    #[error("speaker embedding failed: {0}")]
    Embedding(String),

    #[error("could not load the ONNX model at {path}: {source}")]
    Onnx {
        path: PathBuf,
        #[source]
        source: Box<ort::Error>,
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
/// Not writing the *transcript* is deliberate: the caller decides, which is what keeps skip
/// and `--force` logic in one place instead of spread between here and there.
/// `mic.cleaned.wav` is a different matter and is written here, because it is an input to
/// the recognition that happens here rather than a result the caller chooses to keep.
///
/// `progress` receives one line about the echo-cancellation pre-pass. A user whose sessions
/// quietly stopped being cleaned should be able to see that from normal output rather than
/// from a mysteriously worse transcript.
pub fn transcribe_session(
    session: &DiscoveredSession,
    asr: &mut dyn SpeechToText,
    progress: &mut dyn Write,
) -> Result<Transcript> {
    let metadata = session.load_metadata()?;
    let offset = mic_offset_seconds(&metadata)?;

    let audio = clean_mic_track(session, mic_minus_speaker_seconds(&metadata)?, progress)?;
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
        match transcribe_and_write(session, engine.as_mut(), out) {
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

fn transcribe_and_write(
    session: &DiscoveredSession,
    asr: &mut dyn SpeechToText,
    progress: &mut dyn Write,
) -> Result<usize> {
    let transcript = transcribe_session(session, asr, progress)?;
    transcript.write(&session.paths)?;
    Ok(transcript.turns.len())
}

/// Removes speaker bleed from the mic track, writes the result to `mic.cleaned.wav`, and
/// returns it for recognition.
///
/// The cleaned track is written on every path, including the ones where nothing was
/// cancelled. That is what lets everything downstream read one file with no branch to get
/// wrong, and it means `mic.wav` is never the thing being transcribed -- so nobody can
/// accidentally re-point ASR at the uncleaned track.
///
/// An unreadable or absent `speaker.wav` is not fatal here, unlike `mic.wav`. A session with
/// no reference is a session with nothing to cancel, which is a normal recording, not a
/// broken one.
fn clean_mic_track(
    session: &DiscoveredSession,
    mic_minus_speaker_s: f64,
    progress: &mut dyn Write,
) -> Result<Vec<f32>> {
    let mic = audio::read_track_16k_mono(&session.paths.mic_wav())?;
    let speaker = audio::read_track_16k_mono(&session.paths.speaker_wav()).unwrap_or_default();

    let cleaned = aec::cancel_bleed(&mic, &speaker, mic_minus_speaker_s);
    writeln!(progress, "{}  {}", session.id, cleaned.cleaning)?;
    aec::write_cleaned_track(&session.paths.mic_cleaned_wav(), &cleaned.audio)?;

    Ok(cleaned.audio)
}

/// Seconds from session start to the microphone track's first sample.
///
/// Session start is the earlier of the two tracks' first samples, not `session.json`'s
/// `start_time`: that field is a wall-clock instant captured when the directory was created,
/// with no recorded pairing to mach tick space, so it cannot be compared to either track's
/// `host_ticks`. Using the earliest track instead keeps every turn non-negative once
/// speaker-track turns join the same timeline.
fn mic_offset_seconds(metadata: &SessionMetadata) -> Result<f64> {
    Ok(mic_minus_speaker_seconds(metadata)?.max(0.0))
}

/// How much later the microphone track's first sample is than the speaker track's, negative
/// if the microphone started first.
///
/// The conversion is exact -- integer ticks scaled by the machine's rational timebase in
/// `i128`, rounded once at the end. Going through `f64` first would lose the low bits of a
/// mach tick count within a day of uptime.
///
/// This is metadata alignment only, and the sign is the whole reason it exists separately
/// from [`mic_offset_seconds`]: the echo canceller's delay search needs to know which track
/// actually started first, and clamping that to zero would centre the search in the wrong
/// place. Correcting the *acoustic* offset between the two capture APIs is a different
/// problem again, measured from the signals themselves in [`align`].
fn mic_minus_speaker_seconds(metadata: &SessionMetadata) -> Result<f64> {
    let mic = metadata.mic;
    if mic.timebase_numer == 0 || mic.timebase_denom == 0 {
        return Err(Error::DegenerateTimebase {
            session: metadata.session_id.clone(),
            numer: mic.timebase_numer,
            denom: mic.timebase_denom,
        });
    }

    let delta = i128::from(mic.host_ticks) - i128::from(metadata.speaker.host_ticks);
    let nanos = delta * i128::from(mic.timebase_numer) / i128::from(mic.timebase_denom);
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

    #[derive(Default)]
    struct FakeAsr {
        segments: Vec<AsrSegment>,
        /// Whatever audio the recognizer was handed, so a test can assert on *which* track
        /// reached it rather than trusting that the right file was opened.
        heard: Vec<f32>,
    }

    impl SpeechToText for FakeAsr {
        fn transcribe(&mut self, audio: &[f32]) -> Result<Vec<AsrSegment>> {
            self.heard = audio.to_vec();
            Ok(self.segments.clone())
        }
    }

    fn fake_engine() -> Box<dyn SpeechToText> {
        Box::new(FakeAsr {
            segments: vec![segment(0.0, 1.0, "hello")],
            heard: Vec::new(),
        })
    }

    /// Progress output no test is asserting on.
    fn quiet() -> std::io::Sink {
        std::io::sink()
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

    /// A session whose microphone genuinely heard its own speakers: 30 s of far-end speech
    /// played into the room 120 ms late, over a local talker on a different schedule.
    ///
    /// Written at 16 kHz so what lands on disk is exactly what the pre-pass will work with,
    /// leaving nothing between the fixture and the assertion but the code under test.
    fn make_bleeding_session(paths: &Paths, id: &str) -> DiscoveredSession {
        let rate = TARGET_RATE as usize;
        let len = rate * 30;
        let lag = rate * 120 / 1000;

        // Deterministic band-limited noise, gated into bursts: speech-shaped enough that the
        // delay estimator has something to lock onto, and reproducible so a marginal
        // threshold fails every time rather than once a month.
        let speech = |seed: u64, cycle_s: usize, talk_s: usize| -> Vec<f32> {
            let mut state = seed;
            let (mut low, mut previous, mut high) = (0.0f32, 0.0f32, 0.0f32);
            (0..len)
                .map(|i| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    low = 0.6 * low + 0.4 * ((state >> 40) as f32 / 8_388_608.0 - 1.0);
                    high = 0.99 * high + low - previous;
                    previous = low;
                    if i % (rate * cycle_s) < rate * talk_s {
                        high * 0.3
                    } else {
                        0.0
                    }
                })
                .collect()
        };

        let speaker = speech(11, 4, 3);
        let near = speech(29, 5, 2);
        let mut mic: Vec<f32> = near.iter().map(|n| n * 0.7).collect();
        for (tap, gain) in [(0usize, 0.55f32), (rate * 3 / 1000, 0.30)] {
            for (i, played) in speaker.iter().enumerate() {
                if let Some(sample) = mic.get_mut(i + lag + tap) {
                    *sample += played * gain;
                }
            }
        }

        let id = SessionId::parse(id).unwrap();
        let session_paths = paths.session(&id);
        std::fs::create_dir_all(session_paths.dir()).unwrap();
        aec::write_cleaned_track(&session_paths.mic_wav(), &mic).unwrap();
        aec::write_cleaned_track(&session_paths.speaker_wav(), &speaker).unwrap();
        metadata(&id, 900_000_000_000, 900_000_000_000)
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
            ..FakeAsr::default()
        };
        let transcript = transcribe_session(&session, &mut asr, &mut quiet()).unwrap();

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

        let mut asr = FakeAsr::default();
        let transcript = transcribe_session(&session, &mut asr, &mut quiet()).unwrap();
        assert!(transcript.turns.is_empty());

        transcript.write(&session.paths).unwrap();
        assert!(session.paths.transcript_json().is_file());
        assert!(session.paths.transcript_md().is_file());
    }

    /// Acceptance criterion #1. The derived track appears; the recording it was derived from
    /// is not touched, compared byte for byte rather than by size or mtime.
    #[test]
    fn transcribing_produces_a_cleaned_track_and_leaves_the_raw_mic_track_byte_identical() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_bleeding_session(&paths, "20260809-052600");
        let before = std::fs::read(session.paths.mic_wav()).unwrap();

        let mut asr = FakeAsr::default();
        transcribe_session(&session, &mut asr, &mut quiet()).unwrap();

        assert!(session.paths.mic_cleaned_wav().is_file());
        assert_eq!(
            std::fs::read(session.paths.mic_wav()).unwrap(),
            before,
            "mic.wav must survive transcription unmodified"
        );
    }

    /// Acceptance criterion #2. Not "a cleaned file exists" -- that the samples the
    /// recognizer actually saw are the cleaned ones, which is the only version of this claim
    /// that a future refactor cannot quietly break.
    #[test]
    fn the_recognizer_is_handed_the_cleaned_track_rather_than_the_raw_one() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_bleeding_session(&paths, "20260809-052600");

        let mut asr = FakeAsr::default();
        transcribe_session(&session, &mut asr, &mut quiet()).unwrap();

        let raw = audio::read_track_16k_mono(&session.paths.mic_wav()).unwrap();
        let cleaned = audio::read_track_16k_mono(&session.paths.mic_cleaned_wav()).unwrap();
        assert_eq!(asr.heard, cleaned, "ASR must read mic.cleaned.wav");
        assert_ne!(
            asr.heard, raw,
            "on a session with real bleed the cleaned track must differ from the raw one, \
             otherwise this test would pass without any cancellation happening at all"
        );
    }

    /// Acceptance criterion #3, from the outside: the pre-pass is reference-based against
    /// `speaker.wav`, so removing that file has to change the outcome to a pass-through.
    #[test]
    fn without_a_speaker_track_there_is_nothing_to_cancel_and_the_mic_passes_through() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_bleeding_session(&paths, "20260809-052600");
        std::fs::remove_file(session.paths.speaker_wav()).unwrap();

        let mut asr = FakeAsr::default();
        let mut progress = Vec::new();
        transcribe_session(&session, &mut asr, &mut progress).unwrap();

        let raw = audio::read_track_16k_mono(&session.paths.mic_wav()).unwrap();
        assert_eq!(asr.heard, raw);
        // Still written, and still what ASR read: one input file, no branch downstream.
        assert_eq!(
            audio::read_track_16k_mono(&session.paths.mic_cleaned_wav()).unwrap(),
            raw
        );
        let progress = String::from_utf8(progress).unwrap();
        assert!(progress.contains("no echo cancellation"), "{progress}");
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
