//! Turning the voices on the speaker track into fingerprints `enroll` and identification use.
//!
//! Diarization (see [`crate::NemotronDiarizer`]) says who spoke when. What it cannot say is
//! whose voice that is across meetings, so each speaker's turns are embedded with WeSpeaker
//! here and averaged into the cluster's reference vector.
//!
//! Filterbanks and ONNX tensors stay private. What the diarizer gets is [`WeSpeaker`] as a
//! [`TurnEmbedder`], the [`slice()`] of audio a [`Turn`] covers, [`reference_embedding`] to
//! average a group, and [`representatives`] to pick the clips `enroll` plays.

use meethook_session::{MIN_REPRESENTATIVE_SECONDS, RepresentativeSegment};
use ort::session::Session;
use ort::value::TensorRef;

use crate::audio::TARGET_RATE;
use crate::fbank::{Fbank, MEL_BINS};
use crate::voice_vectors::{group_mean, normalize};
use crate::{Error, Result};

/// How much speech a cluster must hold to count as a solid speaker rather than a fragment.
///
/// The convention is `speech_seconds < floor` is a fragment and `>= floor` is a speaker, so a
/// cluster sitting exactly on the floor is a speaker. `enroll` and identification read it.
///
/// Measured on a real session (`20260810-093047`): the six clusters the user confirmed as the
/// six people in the room held 426.8 / 372.3 / 124.8 / 105.0 / 66.0 / 47.0 s and the largest
/// leftover held 12.3 s, so any floor in `(12.3, 47.0]` gives the same partition and 30 s sits
/// near the middle. Absolute seconds rather than a share of the meeting, because the quantity
/// is how much evidence a mean rests on, which does not change because somebody else talked
/// more. Measured on clusters from the previous diarizer; TASK-088.09 re-checks it on Nemotron slots.
pub const SPEAKER_FLOOR_SECONDS: f64 = 30.0;

/// The shortest turn worth embedding.
///
/// The filterbank floor is a single 25 ms frame, which is not the real limit: a fifth of a
/// second of speech produces an embedding that describes a phoneme rather than a person.
/// Half a second is the point where the embedding starts to be about the voice.
///
/// Turns below it are skipped, not failed.
pub(crate) const MIN_EMBEDDABLE_SECONDS: f64 = 0.5;

/// Clips offered per cluster. Three gives `enroll` a second and a third opinion when the
/// first is unclear, without turning naming one speaker into a listening exercise.
const MAX_REPRESENTATIVES: usize = 3;

/// One stretch of speech on the speaker track, in seconds into that track.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Turn {
    pub start_s: f64,
    pub end_s: f64,
}

/// The samples one turn covers, clipped to the track.
pub(crate) fn slice<'a>(samples: &'a [f32], turn: &Turn) -> &'a [f32] {
    let at = |seconds: f64| ((seconds * TARGET_RATE as f64) as usize).min(samples.len());
    let (start, end) = (at(turn.start_s), at(turn.end_s));
    &samples[start..end.max(start)]
}

/// Something that turns a stretch of 16 kHz audio into a unit-length voice embedding, or `None`
/// when there is too little audio to describe. A seam so callers can be tested without weights; the only real implementation is [`WeSpeaker`].
pub(crate) trait TurnEmbedder {
    fn embed(&mut self, samples: &[f32]) -> Result<Option<Vec<f32>>>;
}

/// The WeSpeaker session and its filterbank, as a [`TurnEmbedder`].
pub(crate) struct WeSpeaker {
    fbank: Fbank,
    session: Session,
}

impl WeSpeaker {
    pub(crate) fn new(session: Session) -> Self {
        Self {
            fbank: Fbank::new(),
            session,
        }
    }
}

impl TurnEmbedder for WeSpeaker {
    fn embed(&mut self, samples: &[f32]) -> Result<Option<Vec<f32>>> {
        embed(&mut self.fbank, &mut self.session, samples)
    }
}

/// The longest stretch fed to the embedding network in one run.
///
/// CoreML fails the run with "Error in dynamically resizing for sequence length (error: -1)"
/// once the input passes roughly 16,000 fbank frames (160 s; 16,000 runs, 17,000 does not,
/// measured on the ONNX Runtime CoreML provider this build links). The diarizer can hand over
/// a turn that long: a slot that latches onto a noisy track holds one run of many minutes.
/// 120 s keeps clear of the limit, and a voice does not need more audio than that to be
/// described.
const MAX_WINDOW_SECONDS: f64 = 120.0;

/// One turn's voice as a unit vector, or `None` if there was not enough audio to describe.
///
/// Turns are embedded one at a time rather than batched. A meeting has hundreds of them,
/// not millions, and batching would mean padding every turn to a common length and telling
/// the network to ignore the padding -- a mask to get wrong in exchange for a saving nobody
/// has measured.
///
/// A turn longer than [`MAX_WINDOW_SECONDS`] is cut into equal windows, each embedded alone,
/// and the result is the normalized mean of those -- the same combination a cluster's
/// reference uses. Equal rather than "full windows plus a remainder" so that no window is a
/// scrap too short to describe anyone.
pub(crate) fn embed(
    fbank: &mut Fbank,
    session: &mut Session,
    samples: &[f32],
) -> Result<Option<Vec<f32>>> {
    let max_window = (MAX_WINDOW_SECONDS * TARGET_RATE as f64) as usize;
    let windows = samples.len().div_ceil(max_window).max(1);
    if windows == 1 {
        return embed_window(fbank, session, samples);
    }

    let size = samples.len().div_ceil(windows);
    let mut embeddings = Vec::with_capacity(windows);
    for window in samples.chunks(size) {
        if let Some(embedding) = embed_window(fbank, session, window)? {
            embeddings.push(embedding);
        }
    }
    let vectors: Vec<&[f32]> = embeddings.iter().map(Vec::as_slice).collect();
    Ok(group_mean(&vectors).map(|(unit, _)| unit))
}

/// One run of the network over `samples`, which must fit in [`MAX_WINDOW_SECONDS`].
fn embed_window(
    fbank: &mut Fbank,
    session: &mut Session,
    samples: &[f32],
) -> Result<Option<Vec<f32>>> {
    let features = fbank.compute(samples);
    if features.is_empty() {
        return Ok(None);
    }
    let frames = features.len() / MEL_BINS;

    let input = TensorRef::from_array_view(([1usize, frames, MEL_BINS], &features[..]))
        .map_err(inference_failed)?;
    let outputs = session.run(ort::inputs![input]).map_err(inference_failed)?;
    let (shape, embedding) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(inference_failed)?;

    let [1, dimensions] = shape[..] else {
        return Err(Error::Embedding(format!(
            "the embedding model returned a tensor of shape {shape:?}, not [1, dimensions]"
        )));
    };
    let mut embedding = embedding[..dimensions as usize].to_vec();
    normalize(&mut embedding);
    Ok(Some(embedding))
}

/// The cluster's voice: the mean of its members, normalized afterwards.
///
/// The order is the contract, not a detail -- `speakers.json` stores enrolled speakers this
/// way and compares with a dot product, so averaging after normalizing, or forgetting to
/// normalize at all, produces a vector that is never quite equal to anything and matches
/// nobody. The arithmetic is [`group_mean`]'s; this only maps member indices to the borrowed
/// slices it takes and drops the length that step also returns.
pub(crate) fn reference_embedding(members: &[usize], embeddings: &[Vec<f32>]) -> Vec<f32> {
    let vectors: Vec<&[f32]> = members.iter().map(|&i| embeddings[i].as_slice()).collect();
    group_mean(&vectors)
        .map(|(unit, _)| unit)
        .expect("a cluster's reference needs members with a non-zero mean direction")
}

/// Clips for `enroll` to play, longest speech first.
///
/// A turn shorter than [`MIN_REPRESENTATIVE_SECONDS`] is widened around its middle rather
/// than passed over, so every cluster has something playable. The widening reaches into
/// audio the diarizer did not call speech, which in practice is this speaker's own breath
/// or the room -- a slightly loose clip is a far better outcome than a speaker who cannot
/// be named at all.
pub(crate) fn representatives(members: &[Turn], track_end_s: f64) -> Vec<RepresentativeSegment> {
    let mut longest = members.to_vec();
    longest.sort_by(|a, b| (b.end_s - b.start_s).total_cmp(&(a.end_s - a.start_s)));
    longest.truncate(MAX_REPRESENTATIVES);

    longest
        .into_iter()
        .map(|turn| {
            let short_by = MIN_REPRESENTATIVE_SECONDS - (turn.end_s - turn.start_s);
            if short_by <= 0.0 {
                return RepresentativeSegment {
                    start: turn.start_s,
                    end: turn.end_s,
                };
            }
            // Widen both ways, then slide back inside the track if that ran off an end.
            let mut start = (turn.start_s - short_by / 2.0).max(0.0);
            let mut end = start + MIN_REPRESENTATIVE_SECONDS;
            if end > track_end_s {
                end = track_end_s;
                start = (end - MIN_REPRESENTATIVE_SECONDS).max(0.0);
            }
            RepresentativeSegment { start, end }
        })
        .collect()
}

fn inference_failed(source: ort::Error) -> Error {
    Error::Embedding(source.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Which side of the average the normalization happens on. The mean of these two is
    /// (0.5, 0.5), whose length is 0.707; normalizing after averaging is what makes it a
    /// unit vector, and `speakers.json` will compare against it with a plain dot product.
    #[test]
    fn a_reference_embedding_is_the_normalized_mean_of_its_members() {
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let reference = reference_embedding(&[0, 1], &embeddings);

        let expected = 0.5f32.sqrt();
        assert!((reference[0] - expected).abs() < 1e-6, "{reference:?}");
        assert!((reference[1] - expected).abs() < 1e-6, "{reference:?}");
        let length: f32 = reference.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((length - 1.0).abs() < 1e-6, "length {length}");
    }

    fn turn(start_s: f64, end_s: f64) -> Turn {
        Turn { start_s, end_s }
    }

    #[test]
    fn representatives_are_the_longest_turns_capped_at_three() {
        let members = [
            turn(0.0, 2.0),
            turn(10.0, 20.0),
            turn(30.0, 33.0),
            turn(40.0, 45.0),
        ];

        let picked = representatives(&members, 60.0);

        assert_eq!(picked.len(), MAX_REPRESENTATIVES);
        assert_eq!(picked[0].start, 10.0);
        assert_eq!(picked[1].start, 40.0);
        assert_eq!(picked[2].start, 30.0);
    }

    /// The guarantee `enroll` depends on: never a fragment too short to recognise a voice
    /// from, and never a segment that runs off either end of the track.
    #[test]
    fn a_short_turn_is_widened_to_the_minimum_without_leaving_the_track() {
        for (members, track_end_s) in [
            (turn(10.0, 10.6), 60.0), // room on both sides
            (turn(0.1, 0.7), 60.0),   // up against the start
            (turn(59.4, 59.9), 60.0), // up against the end
        ] {
            let picked = representatives(&[members], track_end_s);
            let only = picked[0];
            assert!(
                only.seconds() >= MIN_REPRESENTATIVE_SECONDS - 1e-9,
                "{only:?} is {} s",
                only.seconds()
            );
            assert!(only.start >= 0.0 && only.end <= track_end_s, "{only:?}");
        }
    }

    /// A track shorter than the minimum clip cannot satisfy it, and must still produce
    /// something playable rather than a segment running past the end of the audio.
    #[test]
    fn a_track_shorter_than_the_minimum_clip_yields_the_whole_track() {
        let picked = representatives(&[turn(0.2, 0.8)], 1.0);
        assert_eq!(picked[0].start, 0.0);
        assert_eq!(picked[0].end, 1.0);
    }

    /// A turn with no audio behind it produces no filterbank frames, and must be skipped
    /// rather than handed to the network as an empty tensor.
    #[test]
    fn a_turn_with_no_frames_is_skipped_before_it_reaches_the_model() {
        // `embed` bails before touching the session, so there is nothing to load here: an
        // empty slice is exactly what a turn past the end of the track produces.
        let samples = vec![0.0f32; TARGET_RATE as usize];
        assert!(slice(&samples, &turn(5.0, 6.0)).is_empty());
        assert!(Fbank::new().compute(&[]).is_empty());
    }

    /// Opens the embedding weights, or `None` if they are not installed. Same bargain as
    /// the graph-contract tests: `cargo test` never reaches for a 26 MB download.
    fn model(spec: &meethook_models::ModelSpec) -> Option<Session> {
        let root = match std::env::var_os("MEETHOOK_ROOT") {
            Some(root) => std::path::PathBuf::from(root),
            None => std::env::home_dir()?.join("meethook"),
        };
        let path = root.join("models").join(spec.file_name);
        if !path.is_file() {
            eprintln!(
                "skipping: {} is not installed; \
                 run `cargo run --example fetch-onnx-models` to fetch it",
                path.display()
            );
            return None;
        }
        Some(
            crate::open_session(&path)
                .expect("an installed model must load")
                .session,
        )
    }

    /// A voice: a glottal buzz at `f0` through three fixed formants, gated on and off at a
    /// syllable rate. Not speech, but harmonic and modulated the way speech is, and -- the
    /// point here -- the *same* signal every time it is asked for, so two utterances of one
    /// `voice` are the same speaker as far as anything downstream can tell.
    ///
    /// What it is *not* is two people. Measured against this checkpoint, synthetic voices
    /// across a wide range of pitches and formants all land within 0.17 of each other, well
    /// inside [`crate::IDENTIFY_DISTANCE`]: WeSpeaker was trained on human speech and hears all of
    /// these as one unremarkable speaker. So the tests below use it to ask "does one voice
    /// stay one voice", never "do two voices stay apart" -- that second question needs real
    /// recordings, and the `speaker-trials` example is how it gets asked.
    fn voice(seconds: f64, f0: f64, formants: [f64; 3]) -> Vec<f32> {
        let n = (seconds * TARGET_RATE as f64) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / TARGET_RATE as f64;
                let gate = if (t * 4.0).fract() < 0.7 { 1.0 } else { 0.0 };
                let buzz: f64 = (1..=40)
                    .map(|h| {
                        let hz = f0 * h as f64;
                        let gain: f64 = formants
                            .iter()
                            .map(|f| 1.0 / (1.0 + ((hz - f) / 100.0).powi(2)))
                            .sum();
                        gain * (t * hz * std::f64::consts::TAU).sin()
                    })
                    .sum();
                (0.2 * gate * buzz) as f32
            })
            .collect()
    }

    /// The guard on the feature extractor, which is the part of this pipeline that fails
    /// silently. Drop the int16 scaling, or the mean normalization, or use the wrong window
    /// function, and nothing errors -- inference still runs and still returns 256 plausible
    /// floats. The only symptom is that the embeddings stop describing the voice, and every
    /// clip starts looking like every other clip.
    ///
    /// So: two clips of one voice must come back closer together than clips of two different
    /// ones. A weak claim against real speech and a demanding one here, because these
    /// synthetic voices are barely distinguishable to this checkpoint to begin with -- if
    /// the features degenerate, even this ordering stops holding.
    #[test]
    fn one_voice_embeds_closer_to_itself_than_to_another() {
        let Some(mut embedder) = model(&crate::EMBEDDING_MODEL) else {
            return;
        };
        let mut fbank = Fbank::new();
        let mut embedding = |samples: Vec<f32>| {
            embed(&mut fbank, &mut embedder, &samples)
                .unwrap()
                .expect("six seconds of audio must produce an embedding")
        };
        let distance =
            |a: &[f32], b: &[f32]| 1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();

        let long = embedding(voice(6.0, 130.0, [730.0, 1090.0, 2440.0]));
        let short = embedding(voice(4.0, 130.0, [730.0, 1090.0, 2440.0]));
        let other = embedding(voice(6.0, 300.0, [270.0, 2300.0, 3000.0]));

        let same = distance(&long, &short);
        let different = distance(&long, &other);
        assert!(
            same < different,
            "one voice is {same} from itself but {different} from another voice"
        );
    }

    /// The failure this window cap exists for: a 20-minute turn made the CoreML run fail
    /// outright, which aborted the whole transcription. It must come back as an embedding, and
    /// one that still matches the same voice heard for a few seconds.
    #[test]
    fn a_turn_longer_than_the_network_can_take_in_one_run_still_embeds() {
        let Some(mut embedder) = model(&crate::EMBEDDING_MODEL) else {
            return;
        };
        let mut fbank = Fbank::new();
        let mut embedding = |samples: Vec<f32>| {
            embed(&mut fbank, &mut embedder, &samples)
                .unwrap()
                .expect("audio this long must produce an embedding")
        };

        let formants = [730.0, 1090.0, 2440.0];
        let long = embedding(voice(1200.0, 130.0, formants));
        let short = embedding(voice(6.0, 130.0, formants));
        let other = embedding(voice(6.0, 300.0, [270.0, 2300.0, 3000.0]));

        let distance =
            |a: &[f32], b: &[f32]| 1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(distance(&long, &short) < distance(&long, &other));
    }
}
