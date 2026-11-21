//! The one thing transcription needs from diarization.
//!
//! Mel frames, speaker-cache state, slot probabilities and filterbanks all live one level
//! down, in the `nemotron_*` modules and [`crate::speakers`]. What the transcript needs from
//! all of that is much smaller: who spoke, when, and a fingerprint per voice so
//! `enroll` can name them later. That is [`Diarization`], and [`Diarize`] is the seam it
//! comes through.
//!
//! The seam exists for the same reason [`crate::SpeechToText`] does. Batch behaviour --
//! skipping, `--force`, orphan handling, the merge itself -- has to be decidable in
//! `cargo test`, and requiring 230 MB of ONNX weights to assert that two turns come out in
//! chronological order would put that out of reach.

use meethook_session::SpeakerCluster;

use crate::Result;

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
