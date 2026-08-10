//! Turning recorded sessions into transcripts.
//!
//! Both captured tracks end up in one transcript. The microphone track is recognised and
//! labelled `You` -- there is exactly one local speaker, so it is never diarized -- while the
//! speaker track is diarized, recognised, and merged in chronologically with each
//! participant turn attributed to a distinct voice.
//!
//! The batch rules live here rather than in the CLI because they are the part with teeth --
//! never redo work silently, never let one bad session take down the rest, never fetch a
//! 1.6 GB model in order to do nothing -- and they need to be testable against a fake
//! recognizer rather than a real one.

mod aec;
mod align;
mod asr;
mod audio;
mod diarize;
mod fbank;
mod gpu;
mod identify;
mod import;
mod levels;
mod merge;
mod onnx;
mod segmentation;
mod speakers;
mod trials;
mod vad;

use std::io::Write;
use std::path::PathBuf;

pub use aec::{Cleaned, Cleaning, PassThrough, cancel_bleed};
pub use align::{Alignment, NotMeasurable, measure_reference_lag};
pub use asr::{AsrSegment, SpeechToText, WhisperEngine};
pub use audio::{TARGET_RATE, read_track_16k_mono};
pub use diarize::{Diarization, Diarize, OnnxDiarizer, SpeakerTurn};
pub use gpu::NoMetalDevice;
pub use identify::{IDENTIFY_DISTANCE, Identification, identify_clusters};
pub use import::{BuiltSession, ImportedSource, MIC_SILENCE_S, SPLICE_GAP_S, build_session};
pub use levels::{LevelSummary, RUN_BRIDGE_S, SILENCE_FLOOR};
pub use merge::merge;
pub use onnx::{Loaded, open_session};
pub use segmentation::{LocalTurn, segment_speaker_track};
pub use speakers::{Clustering, cluster_speaker_turns};
pub use trials::{EqualError, Spread, Trial, TrialReport, ZeroFalseAccept, score_trials};
pub use vad::{SileroVad, SpeechRegion, VadTuning};

use meethook_models::ModelSpec;
use meethook_session::{
    Classification, DiscoveredSession, EnrolledSpeakers, Paths, SessionId, SessionMetadata,
    SpeakerClusters, Transcript, discover_sessions,
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
/// Graph contract, asserted in the `onnx` module's smoke test:
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
/// Graph contract, asserted in the `onnx` module's smoke test:
/// input `feats` f32 `[B, T, 80]`; output `embs` f32 `[B, 256]`.
pub const EMBEDDING_MODEL: ModelSpec = ModelSpec {
    file_name: "wespeaker-voxceleb-resnet34-LM.onnx",
    url: "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/\
          f0c48c298fd835726c27956a5d617bad7115627e/voxceleb_resnet34_LM.onnx",
    sha256: "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068",
    size_bytes: 26_530_309,
};

/// The voice-activity detector that says which stretches of a track hold speech.
///
/// Silero v5.1.2, in whisper.cpp's own ggml format. 885 KB, and no new crate dependency at
/// all: whisper.cpp 1.8.3 is already vendored by `whisper-rs-sys`, and `whisper-rs` 0.16
/// exposes a safe wrapper around its standalone VAD. See [`SileroVad`] for why a separate
/// detector rather than the pyannote graph already installed.
///
/// v5.1.2 rather than the v6.2.0 that also sits in that repository: v5.1.2 is what
/// whisper.cpp's own documentation and default tooling use, so it is the version its
/// thresholds and post-processing were tuned against.
///
/// Like [`WHISPER_MODEL`], the URL pins an immutable revision rather than `main`, and the hash
/// and size come from the git-LFS pointer Hugging Face serves at the `raw/` path -- so bumping
/// the revision does not require downloading the weights to re-derive them.
pub const SILERO_VAD_MODEL: ModelSpec = ModelSpec {
    file_name: "ggml-silero-v5.1.2.bin",
    url: "https://huggingface.co/ggml-org/whisper-vad/resolve/\
          9ffd54a1e1ee413ddf265af9913beaf518d1639b/ggml-silero-v5.1.2.bin",
    sha256: "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf",
    size_bytes: 885_098,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Session(#[from] meethook_session::Error),

    /// `transparent` rather than `"{0}"`, matching `meethook_record::Error::Permissions`:
    /// the latter makes `anyhow` print this long, multi-line message twice -- once as the
    /// error and once as its own cause.
    #[error(transparent)]
    NoMetalDevice(#[from] NoMetalDevice),

    #[error("could not read {path} as audio: {source}")]
    Wav {
        path: PathBuf,
        #[source]
        source: hound::Error,
    },

    #[error("{path} is not a track this tool can read: {detail}")]
    UnsupportedAudio { path: PathBuf, detail: String },

    /// Audio that cannot produce a turn, a cluster or an embedding: a wav whose data chunk
    /// holds nothing, or no source files at all. Raised by [`build_session`] rather than
    /// tolerated, because a session assembled from it would be indistinguishable from a
    /// successful recording in which nobody spoke.
    #[error("cannot build a session: {detail}")]
    NoAudio { detail: String },

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

    /// Loading or running the voice-activity detector. A `String` rather than a `#[source]`
    /// because whisper-rs's VAD errors carry no information at all -- `NullPointer` on load
    /// and on segmentation, `GenericError(-1)` on detection -- with the real diagnosis going
    /// to the logging hooks. Since the crate will not say what went wrong, the message
    /// [`SileroVad`] builds has to.
    #[error("voice activity detection failed: {0}")]
    Vad(String),

    #[error("could not load the ONNX model at {path}: {source}")]
    Onnx {
        path: PathBuf,
        #[source]
        source: Box<ort::Error>,
    },

    /// No `{source}` in the message even though one is attached: `anyhow` already prints the
    /// cause under "Caused by", and interpolating it here as well prints it twice. Harmless
    /// for a one-line ort failure, unreadable for the several-line [`NoMetalDevice`] one that
    /// also arrives through here.
    #[error("could not load the speech recognition model")]
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

/// Everything transcription needs a model for, opened together.
///
/// One value rather than two factories because the laziness has to be a single decision.
/// Three separate "have we opened this yet" states would each need the same guard, and the
/// invariant that matters -- a batch with nothing to do downloads nothing at all -- would
/// then be three invariants, any one of which could quietly stop holding.
pub struct Engines {
    pub asr: Box<dyn SpeechToText>,
    pub diarizer: Box<dyn Diarize>,
}

/// Opens both engines. Boxed and fallible so the caller owns model acquisition, and so the
/// batch can decide whether opening them is worth doing at all.
pub type EngineFactory<'a> =
    dyn FnMut() -> std::result::Result<Engines, Box<dyn std::error::Error + Send + Sync>> + 'a;

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
/// and `--force` logic in one place instead of spread between here and there. The two files
/// that *are* written here -- `mic.cleaned.wav` and `speaker_clusters.json` -- are not
/// results the caller chooses to keep but by-products of the work: one is the audio
/// recognition actually ran on, the other is what spares `enroll` from re-running diarization
/// over the whole meeting later. `speaker_clusters.json` lands before this returns, so a
/// crash between here and the caller's write leaves clusters with no transcript -- which the
/// next run simply overwrites -- rather than a transcript naming clusters nobody stored.
///
/// The speaker track is read once and used three times: as the echo canceller's reference,
/// as diarization's input, and as the second recognition pass. A long meeting resampled
/// three times over would cost seconds for nothing.
///
/// `progress` receives one line about the echo-cancellation pre-pass. A user whose sessions
/// quietly stopped being cleaned should be able to see that from normal output rather than
/// from a mysteriously worse transcript.
///
/// A session with no `speaker.wav`, or an empty one, degrades to a mic-only transcript
/// rather than failing. A recording with nothing on the far end is a normal recording.
///
/// `speakers` is the enrolled database, handed in already loaded rather than read from
/// `paths` here. The read belongs to the batch -- one file, one read, however many sessions
/// -- and passing the values keeps this function testable against a hand-built database with
/// no filesystem in the way.
pub fn transcribe_session(
    session: &DiscoveredSession,
    asr: &mut dyn SpeechToText,
    diarizer: &mut dyn Diarize,
    speakers: &EnrolledSpeakers,
    progress: &mut dyn Write,
) -> Result<Transcript> {
    let metadata = session.load_metadata()?;

    // An unreadable or absent `speaker.wav` is not fatal here, unlike `mic.wav`: it is the
    // far end of the call, and a call with no far end recorded is still a call.
    let speaker_track =
        audio::read_track_16k_mono(&session.paths.speaker_wav()).unwrap_or_default();
    let mic_track = clean_mic_track(
        session,
        &speaker_track,
        mic_minus_speaker_seconds(&metadata)?,
        progress,
    )?;

    let mic_segments = asr.transcribe(&mic_track)?;

    // Diarize before recognising the speaker track, so a diarization failure costs a
    // Whisper pass over the meeting that would have been thrown away anyway.
    let (diarization, speaker_segments) = if speaker_track.is_empty() {
        (Diarization::default(), Vec::new())
    } else {
        let diarization = diarizer.diarize(&speaker_track)?;
        let segments = asr.transcribe(&speaker_track)?;
        (diarization, segments)
    };

    // Names are decided here rather than inside `merge`, and are never written back into
    // `speaker_clusters.json`: that file is what diarization honestly knows about the audio,
    // and `enroll` reads it expecting to find no names in it.
    let identified = identify::identify_clusters(&diarization.clusters, speakers);

    SpeakerClusters::new(session.id.clone(), diarization.clusters).write(&session.paths)?;

    let turns = merge::merge(
        mic_segments,
        mic_offset_seconds(&metadata)?,
        speaker_segments,
        speaker_offset_seconds(&metadata)?,
        &diarization.turns,
        &identified,
    );

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

    // After the partition, so a batch with nothing to do reads nothing it does not need, and
    // once for the batch rather than once per session: `enroll` never runs concurrently with
    // `transcribe`, so re-reading between sessions would only make a batch's output depend on
    // when each session happened to be reached.
    let speakers = EnrolledSpeakers::read_or_empty(paths)?;

    let mut engines = open_engine().map_err(|source| Error::Engine { source })?;

    for session in work {
        writeln!(out, "{}  transcribing...", session.id)?;
        match transcribe_and_write(session, &mut engines, &speakers, out) {
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
    engines: &mut Engines,
    speakers: &EnrolledSpeakers,
    progress: &mut dyn Write,
) -> Result<usize> {
    let transcript = transcribe_session(
        session,
        engines.asr.as_mut(),
        engines.diarizer.as_mut(),
        speakers,
        progress,
    )?;
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
/// An empty `speaker` is not fatal here, unlike an unreadable `mic.wav`. A session with no
/// reference is a session with nothing to cancel, which is a normal recording, not a broken
/// one.
fn clean_mic_track(
    session: &DiscoveredSession,
    speaker: &[f32],
    mic_minus_speaker_s: f64,
    progress: &mut dyn Write,
) -> Result<Vec<f32>> {
    let mic = audio::read_track_16k_mono(&session.paths.mic_wav())?;

    let cleaned = aec::cancel_bleed(&mic, speaker, mic_minus_speaker_s);
    writeln!(progress, "{}  {}", session.id, cleaned.cleaning)?;
    audio::write_track_16k_mono(&session.paths.mic_cleaned_wav(), &cleaned.audio)?;

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

/// Seconds from session start to the speaker track's first sample: the mirror of
/// [`mic_offset_seconds`], so exactly one of the two is non-zero for any session.
///
/// Both come from `session.json`'s recorded ticks, and deliberately *not* from
/// [`align::measure_reference_lag`]. That measurement is the acoustic path -- how long after
/// the system rendered a sample the microphone heard it come back out of a speaker in a room
/// -- and it bundles output latency and air propagation, neither of which has anything to do
/// with when the far end actually spoke. Applying it here would shift every participant turn
/// late by up to a few hundred milliseconds. The tick delta is honest to well under the
/// accuracy merge ordering needs.
fn speaker_offset_seconds(metadata: &SessionMetadata) -> Result<f64> {
    Ok((-mic_minus_speaker_seconds(metadata)?).max(0.0))
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
    use meethook_session::{
        EnrolledSpeaker, SPEAKER_YOU, SessionPaths, SourceTrack, TrackSync, Turn,
    };

    use super::*;

    /// Apple Silicon's timebase. 125/3 rather than Intel's 1/1 is exactly the ratio that
    /// makes an unscaled tick count look plausible while being 41x wrong.
    const NUMER: u32 = 125;
    const DENOM: u32 = 3;

    /// A recognizer that answers from a script and remembers what it was asked.
    ///
    /// One session means two calls -- the cleaned mic track, then the speaker track, in that
    /// order -- so both sides of the fake are per-call. Recording every buffer rather than
    /// only the last is what lets a test assert on *which* audio reached the recognizer,
    /// which is the only version of "ASR reads the cleaned track" that a refactor cannot
    /// quietly break.
    #[derive(Default)]
    struct FakeAsr {
        /// One canned response per call, in call order. Calls past the end answer nothing.
        responses: Vec<Vec<AsrSegment>>,
        heard: Vec<Vec<f32>>,
    }

    impl FakeAsr {
        fn saying(mic: Vec<AsrSegment>, speaker: Vec<AsrSegment>) -> FakeAsr {
            FakeAsr {
                responses: vec![mic, speaker],
                heard: Vec::new(),
            }
        }
    }

    impl SpeechToText for FakeAsr {
        fn transcribe(&mut self, audio: &[f32]) -> Result<Vec<AsrSegment>> {
            self.heard.push(audio.to_vec());
            Ok(self
                .responses
                .get(self.heard.len() - 1)
                .cloned()
                .unwrap_or_default())
        }
    }

    /// Diarization without a model: whatever the test says was on the speaker track.
    ///
    /// Every buffer it was handed is kept, so "the mic track is never diarized" is
    /// assertable against the samples themselves rather than trusted.
    #[derive(Default)]
    struct FakeDiarizer {
        diarization: Diarization,
        heard: Vec<Vec<f32>>,
    }

    impl Diarize for FakeDiarizer {
        fn diarize(&mut self, speaker_16k_mono: &[f32]) -> Result<Diarization> {
            self.heard.push(speaker_16k_mono.to_vec());
            Ok(self.diarization.clone())
        }
    }

    /// A distinct unit vector per cluster id, so a test can enroll one of these voices and
    /// have it match that cluster and nobody else's.
    fn voice(id: u32) -> Vec<f32> {
        let mut embedding = vec![0.0f32; 4];
        embedding[id as usize % 4] = 1.0;
        embedding
    }

    /// `first_spoke` is seconds into the speaker track, and every caller below keeps it
    /// consistent with the turns it hands the same fake diarizer -- that agreement is what
    /// production computes, so a fixture that broke it would be testing nothing.
    fn cluster(id: u32, seconds: f64, first_spoke: f64) -> meethook_session::SpeakerCluster {
        meethook_session::SpeakerCluster {
            id,
            embedding: voice(id),
            speech_seconds: seconds,
            first_spoke_seconds: first_spoke,
            representatives: vec![meethook_session::RepresentativeSegment {
                start: 0.0,
                end: 2.0,
            }],
        }
    }

    fn fake_engines() -> Engines {
        Engines {
            asr: Box::new(FakeAsr::saying(
                vec![segment(0.0, 1.0, "hello")],
                vec![segment(0.0, 1.0, "hi")],
            )),
            diarizer: Box::new(FakeDiarizer {
                diarization: Diarization {
                    clusters: vec![cluster(0, 1.0, 0.0)],
                    turns: vec![SpeakerTurn {
                        start_s: 0.0,
                        end_s: 1.0,
                        cluster: 0,
                    }],
                },
                heard: Vec::new(),
            }),
        }
    }

    /// Progress output no test is asserting on.
    fn quiet() -> std::io::Sink {
        std::io::sink()
    }

    /// Every install before anybody has run `enroll`, which is what all the tests that are
    /// not about identification want.
    fn nobody_enrolled() -> EnrolledSpeakers {
        EnrolledSpeakers::new(Vec::new())
    }

    /// A database naming each person after the cluster whose voice they were enrolled from,
    /// so `enrolled(&[("Alice", 0)])` is "Alice is cluster 0".
    fn enrolled(people: &[(&str, u32)]) -> EnrolledSpeakers {
        EnrolledSpeakers::new(
            people
                .iter()
                .map(|&(name, from_cluster)| EnrolledSpeaker {
                    name: name.to_string(),
                    embedding: voice(from_cluster),
                })
                .collect(),
        )
    }

    fn transcript_of(paths: &Paths, id: &str) -> Transcript {
        let session = paths.session(&SessionId::parse(id).unwrap());
        Transcript::read(&session.transcript_json()).unwrap()
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
        audio::write_track_16k_mono(&session_paths.mic_wav(), &mic).unwrap();
        audio::write_track_16k_mono(&session_paths.speaker_wav(), &speaker).unwrap();
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

    /// Runs a batch against a single set of fake engines, reporting how many times they were
    /// opened so "no work means no model" can be asserted.
    fn run(paths: &Paths, ids: &[&str], force: bool) -> (BatchReport, usize, String) {
        let requested: Vec<SessionId> =
            ids.iter().map(|id| SessionId::parse(id).unwrap()).collect();
        let mut opened = 0usize;
        let mut out = Vec::new();
        let report = {
            let mut factory = || {
                opened += 1;
                Ok(fake_engines())
            };
            run_batch(paths, &requested, force, &mut factory, &mut out).unwrap()
        };
        (report, opened, String::from_utf8(out).unwrap())
    }

    /// Transcribes one session with nobody on the far end: the shape every test that is
    /// about the microphone side wants.
    fn mic_only(session: &DiscoveredSession, asr: &mut FakeAsr) -> Result<Transcript> {
        transcribe_session(
            session,
            asr,
            &mut FakeDiarizer::default(),
            &nobody_enrolled(),
            &mut quiet(),
        )
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

    /// The two offsets are mirrors: whichever track started second is the one that gets
    /// pushed down the timeline, and exactly one of them is ever non-zero. A sign error here
    /// would put every participant turn on the wrong side of the meeting.
    #[test]
    fn exactly_one_track_is_offset_and_it_is_the_one_that_started_second() {
        let id = SessionId::parse("20260809-052600").unwrap();
        let base = 900_000_000_000_000u64;
        // 1_000_000 ticks at 125/3 ns per tick is 41.666... ms.
        let expected = 0.041_666_666;

        for (mic_ticks, speaker_ticks, mic_offset, speaker_offset) in [
            (base + 1_000_000, base, expected, 0.0),
            (base, base + 1_000_000, 0.0, expected),
            (base, base, 0.0, 0.0),
        ] {
            let metadata = metadata(&id, mic_ticks, speaker_ticks);
            let mic = mic_offset_seconds(&metadata).unwrap();
            let speaker = speaker_offset_seconds(&metadata).unwrap();
            assert!((mic - mic_offset).abs() < 1e-9, "mic offset was {mic} s");
            assert!(
                (speaker - speaker_offset).abs() < 1e-9,
                "speaker offset was {speaker} s"
            );
            assert!(mic == 0.0 || speaker == 0.0, "both tracks were offset");
        }
    }

    #[test]
    fn every_turn_is_labelled_you_on_the_mic_track_with_no_confidence() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600", 1_000_000);

        let mut asr = FakeAsr::saying(
            vec![segment(2.0, 3.0, "second"), segment(0.5, 1.0, "first")],
            Vec::new(),
        );
        let transcript = mic_only(&session, &mut asr).unwrap();

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
        let transcript = mic_only(&session, &mut asr).unwrap();
        assert!(transcript.turns.is_empty());

        transcript.write(&session.paths).unwrap();
        assert!(session.paths.transcript_json().is_file());
        assert!(session.paths.transcript_md().is_file());
    }

    /// Two clusters and a diarized speaker track: the fixture the two-track tests below run
    /// against, with the mic starting 41.67 ms after the speaker track so session time zero
    /// is the *speaker* track's first sample.
    fn two_party(paths: &Paths, id: &str) -> (DiscoveredSession, FakeAsr, FakeDiarizer) {
        let session = make_session(paths, id, 1_000_000);
        let asr = FakeAsr::saying(
            vec![
                segment(1.0, 2.0, "morning"),
                segment(5.0, 6.0, "sounds good"),
            ],
            vec![
                segment(0.0, 0.9, "hi there"),
                segment(3.0, 4.0, "and from me"),
                segment(7.0, 8.0, "let us start"),
            ],
        );
        let diarizer = FakeDiarizer {
            diarization: Diarization {
                clusters: vec![cluster(0, 1.9, 0.0), cluster(1, 1.0, 3.0)],
                turns: vec![
                    SpeakerTurn {
                        start_s: 0.0,
                        end_s: 0.9,
                        cluster: 0,
                    },
                    SpeakerTurn {
                        start_s: 3.0,
                        end_s: 4.0,
                        cluster: 1,
                    },
                    SpeakerTurn {
                        start_s: 7.0,
                        end_s: 8.0,
                        cluster: 0,
                    },
                ],
            },
            heard: Vec::new(),
        };
        (session, asr, diarizer)
    }

    /// Acceptance criteria #1, #2, #3 and #9, at the level a user meets them: one transcript
    /// holding both tracks in order, the local speaker labelled "You" and never diarized,
    /// and two participants told apart rather than pooled into one "Unknown".
    #[test]
    fn a_two_party_session_produces_one_chronological_transcript_of_both_tracks() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let (session, mut asr, mut diarizer) = two_party(&paths, "20260809-052600");

        let transcript = transcribe_session(
            &session,
            &mut asr,
            &mut diarizer,
            &nobody_enrolled(),
            &mut quiet(),
        )
        .unwrap();

        let said: Vec<(&str, &str)> = transcript
            .turns
            .iter()
            .map(|turn| (turn.speaker.as_str(), turn.text.as_str()))
            .collect();
        assert_eq!(
            said,
            [
                ("Unknown 1", "hi there"),
                ("You", "morning"),
                ("Unknown 2", "and from me"),
                ("You", "sounds good"),
                ("Unknown 1", "let us start"),
            ]
        );
        assert!(
            transcript
                .turns
                .windows(2)
                .all(|w| w[0].start <= w[1].start),
            "{:?}",
            transcript.turns
        );

        // "You" and the mic track are the same set of turns, in both directions.
        for turn in &transcript.turns {
            assert_eq!(
                turn.speaker == SPEAKER_YOU,
                turn.source_track == SourceTrack::Mic,
                "{turn:?}"
            );
        }
        // The other half of AC #2, and the one a label check cannot make: the microphone
        // samples never reached diarization at all.
        let speaker_track = audio::read_track_16k_mono(&session.paths.speaker_wav()).unwrap();
        assert_eq!(
            diarizer.heard,
            [speaker_track],
            "diarization must be run on the speaker track and on nothing else"
        );

        // AC #9: one line per turn, in transcript order.
        let markdown = transcript.render_markdown();
        assert_eq!(markdown.lines().count(), transcript.turns.len());
        assert!(
            markdown.starts_with("**[00:00] Unknown 1:** hi there\n"),
            "{markdown}"
        );
    }

    /// Acceptance criterion #5, and the reason this file is written here rather than by the
    /// caller: `enroll` has to be able to name these speakers later without re-running
    /// diarization over the whole meeting.
    #[test]
    fn a_transcribed_session_gets_a_clusters_file_enroll_can_work_from() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let (session, mut asr, mut diarizer) = two_party(&paths, "20260809-052600");

        transcribe_session(
            &session,
            &mut asr,
            &mut diarizer,
            &nobody_enrolled(),
            &mut quiet(),
        )
        .unwrap();

        let stored =
            meethook_session::SpeakerClusters::read(&session.paths.speaker_clusters_json())
                .unwrap();
        assert_eq!(stored.session_id, session.id);
        assert_eq!(stored.clusters.len(), 2);
        for cluster in &stored.clusters {
            assert!(!cluster.embedding.is_empty(), "{cluster:?}");
            assert!(!cluster.representatives.is_empty(), "{cluster:?}");
        }

        // The one thing in here that cannot be recovered from the rest of the file, and the
        // whole reason `enroll` can map an "Unknown N" back to a cluster: `unknown_labels`
        // over these has to reproduce the labels the transcript was just written with.
        assert_eq!(
            stored
                .clusters
                .iter()
                .map(|c| (c.id, c.first_spoke_seconds))
                .collect::<Vec<_>>(),
            [(0, 0.0), (1, 3.0)]
        );
        let recovered = meethook_session::unknown_labels(
            stored
                .clusters
                .iter()
                .map(|c| (c.id, c.first_spoke_seconds)),
        );
        assert_eq!(recovered[&0], "Unknown 1");
        assert_eq!(recovered[&1], "Unknown 2");
    }

    /// The file is written on every path, including the one where there was nobody to find.
    /// One file with no branch downstream beats an absent file every consumer has to handle.
    #[test]
    fn a_session_with_no_participants_still_gets_an_empty_clusters_file() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let session = make_session(&paths, "20260809-052600", 0);

        mic_only(&session, &mut FakeAsr::default()).unwrap();

        let stored =
            meethook_session::SpeakerClusters::read(&session.paths.speaker_clusters_json())
                .unwrap();
        assert!(stored.clusters.is_empty());
    }

    /// Re-running must produce the same transcript, byte for byte. Labels that moved between
    /// runs would mean "Unknown 2" is not a thing a user can act on -- and `--force` exists
    /// precisely so a transcript can be regenerated.
    ///
    /// Run with one speaker enrolled as well as with none, because substituting a name is a
    /// new way for a rerun to stop being byte-identical: a tie between two references, or a
    /// number handed out in a different order, would show up here and nowhere else.
    #[test]
    fn re_transcribing_the_same_session_produces_an_identical_transcript() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());

        let transcribe = |id: &str, speakers: &EnrolledSpeakers| {
            let (session, mut asr, mut diarizer) = two_party(&paths, id);
            let transcript =
                transcribe_session(&session, &mut asr, &mut diarizer, speakers, &mut quiet())
                    .unwrap();
            transcript.write(&session.paths).unwrap();
            (
                std::fs::read(session.paths.transcript_json()).unwrap(),
                std::fs::read(session.paths.transcript_md()).unwrap(),
            )
        };

        // Compared as bytes rather than as parsed turns, because what a user diffs is the
        // file: a re-ordered tie or a renumbered label would show up here and nowhere else.
        for speakers in [nobody_enrolled(), enrolled(&[("Alice", 0)])] {
            let first = transcribe("20260809-052600", &speakers);
            let second = transcribe("20260809-052600", &speakers);
            assert_eq!(first, second);
        }
    }

    /// Acceptance criteria #1 and #5, at the level a user meets them: a hand-written
    /// `speakers.json` turns "Unknown 1" into a name, the similarity that decided it lands on
    /// those turns, and nothing that makes no identity claim carries a number.
    ///
    /// The vacated "Unknown 1" is the visible half of numbering over every voice and
    /// substituting afterwards: the second speaker stays "Unknown 2" rather than being
    /// promoted, so naming Alice does not silently relabel the person nobody has named.
    #[test]
    fn an_enrolled_speaker_is_named_in_the_transcript_with_the_similarity_that_matched_them() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let (session, mut asr, mut diarizer) = two_party(&paths, "20260809-052600");

        let transcript = transcribe_session(
            &session,
            &mut asr,
            &mut diarizer,
            &enrolled(&[("Alice", 0)]),
            &mut quiet(),
        )
        .unwrap();

        let said: Vec<(&str, &str, Option<f32>)> = transcript
            .turns
            .iter()
            .map(|t| (t.speaker.as_str(), t.text.as_str(), t.speaker_id_confidence))
            .collect();
        assert_eq!(
            said,
            [
                ("Alice", "hi there", Some(1.0)),
                ("You", "morning", None),
                ("Unknown 2", "and from me", None),
                ("You", "sounds good", None),
                ("Alice", "let us start", Some(1.0)),
            ]
        );
    }

    /// `speaker_clusters.json` is what diarization honestly knows about the audio, and
    /// `enroll` reads it back expecting to find no names in it -- so identification must not
    /// leak into the file even when it succeeded.
    #[test]
    fn naming_a_speaker_leaves_the_clusters_file_anonymous() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let (session, mut asr, mut diarizer) = two_party(&paths, "20260809-052600");

        transcribe_session(
            &session,
            &mut asr,
            &mut diarizer,
            &enrolled(&[("Alice", 0)]),
            &mut quiet(),
        )
        .unwrap();

        let raw = std::fs::read_to_string(session.paths.speaker_clusters_json()).unwrap();
        assert!(!raw.contains("Alice"), "{raw}");
        let stored =
            meethook_session::SpeakerClusters::read(&session.paths.speaker_clusters_json())
                .unwrap();
        assert_eq!(
            stored.clusters.iter().map(|c| c.id).collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(stored.clusters[0].embedding, voice(0));
    }

    /// Acceptance criterion #6, at the level a first-run user meets it: a fresh install has no
    /// `speakers.json` at all, and that has to be an ordinary all-Unknown transcript rather
    /// than a failed batch.
    #[test]
    fn a_first_run_with_no_speakers_file_transcribes_everyone_as_unknown() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600", 0);
        assert!(!paths.speakers_json().exists());

        let (report, _, output) = run(&paths, &[], false);

        assert_eq!(report.failed, 0, "{output}");
        assert_eq!(report.transcribed, 1, "{output}");
        let transcript = transcript_of(&paths, "20260809-052600");
        assert!(
            transcript.turns.iter().any(|t| t.speaker == "Unknown 1"),
            "{:?}",
            transcript.turns
        );
        assert!(
            transcript
                .turns
                .iter()
                .all(|t| t.speaker_id_confidence.is_none()),
            "{:?}",
            transcript.turns
        );
    }

    /// The ticket's own manual check, mechanized: a `speakers.json` written by hand at the
    /// root, an ordinary batch over an ordinary session, and the cluster comes back named.
    /// This is the only test that proves `run_batch` reads the file at all.
    #[test]
    fn a_batch_reads_speakers_json_from_the_root_and_names_the_matching_cluster() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600", 0);
        enrolled(&[("Alice", 0)]).write(&paths).unwrap();

        let (report, _, output) = run(&paths, &[], false);

        assert_eq!(report.transcribed, 1, "{output}");
        let transcript = transcript_of(&paths, "20260809-052600");
        let alice: Vec<&Turn> = transcript
            .turns
            .iter()
            .filter(|t| t.speaker == "Alice")
            .collect();
        assert_eq!(alice.len(), 1, "{:?}", transcript.turns);
        assert_eq!(alice[0].speaker_id_confidence, Some(1.0));
    }

    /// A database that exists and does not parse is not the first-run case and must not be
    /// silently downgraded into one: ten enrolled people quietly coming back as ten Unknowns
    /// is the failure worth interrupting a batch for. The message has to name the file, or the
    /// user has no way to find what to fix.
    #[test]
    fn a_malformed_speakers_file_fails_the_batch_by_name() {
        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        make_session(&paths, "20260809-052600", 0);
        std::fs::write(paths.speakers_json(), b"{ not json at all").unwrap();

        let mut out = Vec::new();
        let error =
            run_batch(&paths, &[], false, &mut || Ok(fake_engines()), &mut out).unwrap_err();

        assert!(error.to_string().contains("speakers.json"), "{error}");
        assert!(
            !paths
                .session(&SessionId::parse("20260809-052600").unwrap())
                .transcript_json()
                .exists(),
            "a batch that could not read the database must not write half-named transcripts"
        );
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
        mic_only(&session, &mut asr).unwrap();

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
        mic_only(&session, &mut asr).unwrap();

        let raw = audio::read_track_16k_mono(&session.paths.mic_wav()).unwrap();
        let cleaned = audio::read_track_16k_mono(&session.paths.mic_cleaned_wav()).unwrap();
        assert_eq!(asr.heard[0], cleaned, "ASR must read mic.cleaned.wav");
        assert_ne!(
            asr.heard[0], raw,
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
        let mut diarizer = FakeDiarizer::default();
        let mut progress = Vec::new();
        transcribe_session(
            &session,
            &mut asr,
            &mut diarizer,
            &nobody_enrolled(),
            &mut progress,
        )
        .unwrap();

        let raw = audio::read_track_16k_mono(&session.paths.mic_wav()).unwrap();
        assert_eq!(
            asr.heard,
            std::slice::from_ref(&raw),
            "only the mic track exists to hear"
        );
        assert!(
            diarizer.heard.is_empty(),
            "there is no speaker track to diarize, so the models must not be run at all"
        );
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
        // The sharp edge: a no-op rerun must not fetch 1.7 GB of models in order to do
        // nothing. One factory for all three is what keeps this a single decision.
        assert_eq!(opened, 0, "a fully skipped batch must not open any model");

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
