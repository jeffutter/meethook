//! The one thing transcription needs from diarization, and the ONNX pipeline behind it.
//!
//! Segmentation windows, powerset logits, filterbanks and distance matrices all live one
//! level down, in [`crate::segmentation`] and [`crate::speakers`]. What the transcript needs
//! from all of that is much smaller: who spoke, when, and a fingerprint per voice so
//! `enroll` can name them later. That is [`Diarization`], and [`Diarize`] is the seam it
//! comes through.
//!
//! The seam exists for the same reason [`crate::SpeechToText`] does. Batch behaviour --
//! skipping, `--force`, orphan handling, the merge itself -- has to be decidable in
//! `cargo test`, and requiring 32 MB of ONNX weights to assert that two turns come out in
//! chronological order would put that out of reach.

use std::path::Path;

use meethook_session::SpeakerCluster;
use ort::session::Session;

use crate::segmentation::segment_speaker_track;
use crate::speakers::cluster_speaker_turns;
use crate::{Result, onnx};

/// One stretch of speech on the speaker track, attributed to one of the session's voices.
///
/// Times are seconds **into the speaker track**, not seconds from session start: diarization
/// never sees the other track, so it cannot know where this one sits on the shared timeline.
/// Putting them there is the merge's job.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerTurn {
    pub start_s: f64,
    pub end_s: f64,
    /// Which [`SpeakerCluster`] this turn's voice belongs to, by `id`.
    pub cluster: u32,
}

/// The voices on one speaker track, and when each of them spoke.
///
/// Turns that could not be attributed -- a fragment too short for the embedding model to
/// describe -- are simply absent. There is no cluster to name them with, and carrying an
/// unattributed span through the merge would only invite a caller to invent one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Diarization {
    /// One entry per distinct voice, most talkative first, `id` matching the index. This is
    /// what lands in `speaker_clusters.json`.
    pub clusters: Vec<SpeakerCluster>,
    /// Attributed speech, in start order.
    pub turns: Vec<SpeakerTurn>,
}

/// The one thing transcription needs from diarization.
///
/// One method wide, deliberately: it is a seam, not a plugin framework.
pub trait Diarize {
    /// Finds the distinct voices in `speaker_16k_mono`, which must be 16 kHz mono `f32`.
    ///
    /// A track with no speech in it yields no clusters and no turns rather than an error. A
    /// meeting where nobody but the user spoke is a normal meeting.
    fn diarize(&mut self, speaker_16k_mono: &[f32]) -> Result<Diarization>;
}

/// The real pipeline: pyannote segmentation, then WeSpeaker embeddings, then clustering.
///
/// Both graphs are loaded once and reused across a whole batch, for the same reason the
/// Whisper context is: loading is the expensive part.
pub struct OnnxDiarizer {
    segmenter: Session,
    embedder: Session,
    accelerated: bool,
}

impl OnnxDiarizer {
    /// Loads both graphs from already-fetched weights.
    pub fn load(segmentation_model: &Path, embedding_model: &Path) -> Result<OnnxDiarizer> {
        let segmenter = onnx::open_session(segmentation_model)?;
        let embedder = onnx::open_session(embedding_model)?;
        Ok(OnnxDiarizer {
            accelerated: segmenter.accelerated && embedder.accelerated,
            segmenter: segmenter.session,
            embedder: embedder.session,
        })
    }

    /// False when either graph fell back to CPU because CoreML would not take it, and false
    /// by construction off macOS, where the graphs run on the CPU execution provider and
    /// there is nothing to decline them.
    ///
    /// Worth reporting: a CPU-only diarization pass is perfectly correct and several times
    /// slower, and a user who is not told will only see a transcribe that mysteriously takes
    /// minutes.
    pub fn accelerated(&self) -> bool {
        self.accelerated
    }
}

impl Diarize for OnnxDiarizer {
    fn diarize(&mut self, speaker_16k_mono: &[f32]) -> Result<Diarization> {
        let local = segment_speaker_track(speaker_16k_mono, &mut self.segmenter)?;
        let clustering = cluster_speaker_turns(speaker_16k_mono, &local, &mut self.embedder)?;

        // `assignment` is positional against `local`, so zipping is what turns
        // window-local turns into turns owned by a speaker who persists across the meeting.
        let turns = local
            .iter()
            .zip(&clustering.assignment)
            .filter_map(|(turn, assigned)| {
                Some(SpeakerTurn {
                    start_s: turn.start_s,
                    end_s: turn.end_s,
                    cluster: (*assigned)?,
                })
            })
            .collect();

        Ok(Diarization {
            clusters: clustering.clusters,
            turns,
        })
    }
}
