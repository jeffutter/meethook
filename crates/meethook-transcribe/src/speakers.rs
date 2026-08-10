//! Giving the voices on the speaker track identities that last the whole meeting.
//!
//! Segmentation can only say "someone is talking, and it is not whoever else is talking
//! right now, in this ten-second window". Nothing about a window index survives into the
//! next window, so identity cannot come from segmentation -- it has to come from the voice
//! itself. That is what this module does: one embedding per turn, then clustering across
//! the whole meeting, so a person who speaks at minute 0 and again at minute 40 comes back
//! as the same speaker.
//!
//! Filterbanks, ONNX tensors and the pairwise distance matrix stay private. What the caller
//! gets is [`cluster_speaker_turns`], the clusters it returns, and -- so that a grouping can
//! be checked rather than taken on trust -- the per-turn embeddings those clusters are means
//! of. See [`Clustering::turn_embeddings`] for why that last one is handed back.

use meethook_session::{MIN_REPRESENTATIVE_SECONDS, RepresentativeSegment, SpeakerCluster};
use ort::session::Session;
use ort::value::TensorRef;

use crate::audio::TARGET_RATE;
use crate::fbank::{Fbank, MEL_BINS};
use crate::segmentation::LocalTurn;
use crate::{Error, Result};

/// How far apart two turns' voices may be and still be one person.
///
/// Cosine distance, so 0 is the same vector and 1 is orthogonal. For this checkpoint two
/// clips of one speaker typically land inside 0.3 and two different speakers outside 0.6,
/// with the crossover around 0.5 -- which is where sherpa-onnx's diarization puts its
/// default, and this sits deliberately below it.
///
/// The reason to be below rather than at the crossover is that the two mistakes are not
/// symmetric. Too low and one person splits into two clusters: the user sees two Unknowns,
/// hears both, and can name them the same thing or ignore one. Too high and two people
/// merge: one person's words are attributed to the other, in a transcript that looks
/// perfectly ordinary and that nobody will re-check. A visible extra speaker is a cheap
/// error and a silent misattribution is an expensive one, so the threshold is biased toward
/// splitting.
///
/// Measured on a real 43-minute meeting (session `20260810-093047`, six dominant speakers on
/// the speaker track, 87.5% of all speech) rather than inherited from the checkpoint's
/// published behaviour:
///
/// | population                                          | min   | median      | max   |
/// |-----------------------------------------------------|-------|-------------|-------|
/// | two turns of one speaker                            | 0.077 | 0.232-0.396 | 0.683 |
/// | two speakers heard in one window, so known different | 0.270 | 0.837       | 1.118 |
///
/// The populations overlap across `[0.270, 0.683]`, so no threshold separates them and 0.45
/// buys one kind of mistake with the other. It is kept because it is the value that got the
/// six real speakers right: the closest two of them are 0.304 apart at their nearest turns
/// and 0.604 apart at the median, so the nearest wrong merge is a long way above the cut,
/// while every dominant speaker's own turns average below it.
///
/// The cut is not what strands short turns in clusters of their own -- that is average
/// linkage against a large group, and raising this constant to absorb them would need 0.6-0.8
/// and would merge the people above. See TASK-018.
const MERGE_DISTANCE: f32 = 0.45;

/// The shortest turn worth embedding.
///
/// The filterbank floor is a single 25 ms frame, which is not the real limit: a fifth of a
/// second of speech produces an embedding that describes a phoneme rather than a person,
/// and dropping one of those into the clustering is worse than leaving it out, because it
/// lands somewhere arbitrary and can drag a merge with it. Half a second is the point where
/// the embedding starts to be about the voice.
///
/// Turns below it are skipped, not failed, and counted -- see [`Clustering::skipped`].
const MIN_EMBEDDABLE_SECONDS: f64 = 0.5;

/// Clips offered per cluster. Three gives `enroll` a second and a third opinion when the
/// first is unclear, without turning naming one speaker into a listening exercise.
const MAX_REPRESENTATIVES: usize = 3;

/// The speakers found on a track, and which turn belongs to which.
pub struct Clustering {
    /// One entry per distinct voice, most talkative first, with `id` matching the index.
    pub clusters: Vec<SpeakerCluster>,

    /// Parallel to the turns handed in: the cluster each was assigned, or `None` for a turn
    /// too short to embed.
    ///
    /// This is how the caller attributes recognised text to a speaker, which is why it is
    /// positional rather than a map -- there is no turn identity to key on.
    pub assignment: Vec<Option<u32>>,

    /// Parallel to the turns handed in: each turn's unit-length voice embedding, `None` in
    /// exactly the positions [`Clustering::assignment`] is `None`.
    ///
    /// Named for the turns rather than shortened to `embeddings`, because
    /// [`SpeakerCluster::embedding`] is one field away and is a different vector: this is
    /// the population, that is its normalized mean.
    ///
    /// Handed back because the mean is not enough to calibrate anything on. A distance
    /// between two cluster references only ever describes groups that already survived
    /// the merge threshold, so it can say how far apart the decisions were but not how close
    /// the evidence came to going the other way -- how nearly one speaker split, or how
    /// nearly two merged. Those are turn-to-turn questions, and answering them outside this
    /// module is what keeps the answer honest: the diagnostic reads the same vectors
    /// clustering grouped instead of re-deriving its own, which could quietly disagree.
    ///
    /// Production callers pay no inference and no arithmetic for this -- these vectors are
    /// computed either way and were previously just dropped. The whole cost is that they
    /// stay alive until the `Clustering` does, `4 * dimensions` bytes per embedded turn:
    /// about 1 KB per turn, so ~2 MB for a two-hour meeting, against a speaker track for
    /// that meeting already resident at ~460 MB.
    pub turn_embeddings: Vec<Option<Vec<f32>>>,
}

impl Clustering {
    /// How many turns were dropped for being too short to embed.
    ///
    /// Worth reporting if it ever gets large: every skipped turn is speech that will end up
    /// unattributed, and a track that is mostly one-word interjections would otherwise look
    /// like a track with no speakers.
    pub fn skipped(&self) -> usize {
        self.assignment.iter().filter(|a| a.is_none()).count()
    }
}

/// Embeds every turn and clusters them into per-meeting speakers.
///
/// `samples_16k` is the same 16 kHz mono track that produced `turns`, and `embedder` is a
/// loaded WeSpeaker graph. Times in the returned representatives are offsets into that
/// track.
///
/// No speaker count is asked for or guessed at: how many people are in a meeting is not
/// something the user should have to say, so clustering stops on a distance threshold
/// rather than at a target number of clusters.
///
/// An empty track, no turns, or turns that are all too short all yield no clusters rather
/// than an error. A meeting where nobody but the user spoke is a normal meeting.
pub fn cluster_speaker_turns(
    samples_16k: &[f32],
    turns: &[LocalTurn],
    embedder: &mut Session,
) -> Result<Clustering> {
    let track_end_s = samples_16k.len() as f64 / TARGET_RATE as f64;
    let mut fbank = Fbank::new();

    // Embedded turns, and where each came from in `turns`.
    let mut embeddings = Vec::new();
    let mut sources = Vec::new();
    for (index, turn) in turns.iter().enumerate() {
        if turn.end_s - turn.start_s < MIN_EMBEDDABLE_SECONDS {
            continue;
        }
        let Some(embedding) = embed(&mut fbank, embedder, slice(samples_16k, turn))? else {
            continue;
        };
        embeddings.push(embedding);
        sources.push(index);
    }

    let constraints: Vec<(usize, usize)> = sources
        .iter()
        .map(|&i| (turns[i].window, turns[i].local_speaker))
        .collect();
    let mut groups = agglomerate(&embeddings, &constraints);

    // Most talkative first, so cluster 0 is the person the meeting was mostly with and the
    // ids mean something to a human reading the file.
    let spoken = |group: &Vec<usize>| -> f64 {
        group
            .iter()
            .map(|&e| turns[sources[e]].end_s - turns[sources[e]].start_s)
            .sum()
    };
    groups.sort_by(|a, b| spoken(b).total_cmp(&spoken(a)));

    let mut assignment = vec![None; turns.len()];
    let mut clusters = Vec::with_capacity(groups.len());
    for (id, group) in groups.iter().enumerate() {
        for &member in group {
            assignment[sources[member]] = Some(id as u32);
        }
        let members: Vec<LocalTurn> = group.iter().map(|&e| turns[sources[e]]).collect();
        clusters.push(SpeakerCluster {
            id: id as u32,
            embedding: reference_embedding(group, &embeddings),
            speech_seconds: spoken(group),
            // `agglomerate` never returns an empty group, but the type allows one and the
            // alternative fold would leave an infinity here -- which serializes as `null`
            // and makes the whole clusters file unreadable rather than merely odd.
            first_spoke_seconds: members
                .iter()
                .map(|turn| turn.start_s)
                .min_by(f64::total_cmp)
                .unwrap_or(0.0),
            representatives: representatives(&members, track_end_s),
        });
    }

    // Scattered by `sources` exactly the way `assignment` is, so the two vectors cannot
    // drift into disagreeing about which turns were embedded. `embeddings` is finished with
    // by here, so this moves rather than copies.
    let mut turn_embeddings = vec![None; turns.len()];
    for (embedding, &index) in embeddings.into_iter().zip(&sources) {
        turn_embeddings[index] = Some(embedding);
    }

    Ok(Clustering {
        clusters,
        assignment,
        turn_embeddings,
    })
}

/// The samples one turn covers, clipped to the track.
fn slice<'a>(samples: &'a [f32], turn: &LocalTurn) -> &'a [f32] {
    let at = |seconds: f64| ((seconds * TARGET_RATE as f64) as usize).min(samples.len());
    let (start, end) = (at(turn.start_s), at(turn.end_s));
    &samples[start..end.max(start)]
}

/// One turn's voice as a unit vector, or `None` if there was not enough audio to describe.
///
/// Turns are embedded one at a time rather than batched. A meeting has hundreds of them,
/// not millions, and batching would mean padding every turn to a common length and telling
/// the network to ignore the padding -- a mask to get wrong in exchange for a saving nobody
/// has measured.
fn embed(fbank: &mut Fbank, session: &mut Session, samples: &[f32]) -> Result<Option<Vec<f32>>> {
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

/// Scales a vector to unit length, leaving a zero vector alone.
///
/// Every distance below is a dot product, which is only a cosine if both sides are unit
/// length; normalizing here is what lets the rest of this module stop thinking about it.
fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.iter_mut().for_each(|v| *v /= norm);
    }
}

/// The cluster's voice: the mean of its members, normalized afterwards.
///
/// The order is the contract, not a detail -- `speakers.json` stores enrolled speakers this
/// way and compares with a dot product, so averaging after normalizing, or forgetting to
/// normalize at all, produces a vector that is never quite equal to anything and matches
/// nobody.
fn reference_embedding(members: &[usize], embeddings: &[Vec<f32>]) -> Vec<f32> {
    let mut mean = vec![0.0f32; embeddings[members[0]].len()];
    for &member in members {
        for (m, v) in mean.iter_mut().zip(&embeddings[member]) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= members.len() as f32;
    }
    normalize(&mut mean);
    mean
}

/// Groups unit-length embeddings into speakers, returning each group's member indices.
///
/// Agglomerative with average linkage: start with every turn alone, repeatedly merge the
/// two closest groups, stop when the closest pair is further apart than [`MERGE_DISTANCE`].
/// Average linkage rather than single (which chains one speaker into the next through a
/// single ambiguous turn) or complete (which shatters a speaker over one bad clip).
///
/// The implementation is the naive one -- recompute every group pair's average distance on
/// every merge -- because a meeting has hundreds of turns and a cleverer one would trade
/// something that is obviously correct for time nobody is waiting on.
///
/// `constraints` carries each embedding's `(window, local_speaker)` from segmentation, and
/// it is free supervision: two turns the model heard *in the same window* with different
/// local speaker indices are definitely different people, so no merge is allowed to put
/// them together however close their embeddings look.
fn agglomerate(embeddings: &[Vec<f32>], constraints: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let n = embeddings.len();
    let mut distances = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..i {
            // Both sides are unit length, so the dot product is the cosine.
            let cosine: f32 = embeddings[i]
                .iter()
                .zip(&embeddings[j])
                .map(|(a, b)| a * b)
                .sum();
            let distance =
                if constraints[i].0 == constraints[j].0 && constraints[i].1 != constraints[j].1 {
                    // Heard at once: no evidence about their voices can make them one person.
                    f32::INFINITY
                } else {
                    1.0 - cosine
                };
            distances[i * n + j] = distance;
            distances[j * n + i] = distance;
        }
    }

    let mut groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    loop {
        let mut best = None;
        for a in 0..groups.len() {
            for b in 0..a {
                let sum: f32 = groups[a]
                    .iter()
                    .flat_map(|&i| groups[b].iter().map(move |&j| (i, j)))
                    .map(|(i, j)| distances[i * n + j])
                    .sum();
                let average = sum / (groups[a].len() * groups[b].len()) as f32;
                if average < MERGE_DISTANCE && best.is_none_or(|(d, _, _)| average < d) {
                    best = Some((average, a, b));
                }
            }
        }
        let Some((_, a, b)) = best else { break };

        let merged = groups.swap_remove(a);
        groups[b].extend(merged);
        groups[b].sort_unstable();
    }
    groups
}

/// Clips for `enroll` to play, longest speech first.
///
/// A turn shorter than [`MIN_REPRESENTATIVE_SECONDS`] is widened around its middle rather
/// than passed over, so every cluster has something playable. The widening reaches into
/// audio segmentation did not call speech, which in practice is this speaker's own breath
/// or the room -- a slightly loose clip is a far better outcome than a speaker who cannot
/// be named at all.
fn representatives(members: &[LocalTurn], track_end_s: f64) -> Vec<RepresentativeSegment> {
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

    /// A unit vector pointing `degrees` away from the first axis, in the first two
    /// dimensions. Cosine distance between two of these is `1 - cos(difference)`, so the
    /// tests below can name the distance they mean instead of a pile of decimals.
    fn at(degrees: f32) -> Vec<f32> {
        let radians = degrees.to_radians();
        vec![radians.cos(), radians.sin(), 0.0, 0.0]
    }

    /// Constraints that never forbid a merge: every turn in its own window.
    fn unconstrained(n: usize) -> Vec<(usize, usize)> {
        (0..n).map(|i| (i, 0)).collect()
    }

    fn group_sizes(groups: &[Vec<usize>]) -> Vec<usize> {
        let mut sizes: Vec<usize> = groups.iter().map(|g| g.len()).collect();
        sizes.sort_unstable();
        sizes
    }

    #[test]
    fn identical_voices_become_one_speaker() {
        let embeddings = vec![at(0.0), at(0.0), at(0.0)];
        let groups = agglomerate(&embeddings, &unconstrained(3));
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], [0, 1, 2]);
    }

    /// Orthogonal voices are as different as this representation gets: distance 1.0, well
    /// past any threshold worth having.
    #[test]
    fn clearly_separated_voices_stay_apart() {
        let embeddings = vec![at(0.0), at(90.0), at(0.0), at(90.0)];
        let groups = agglomerate(&embeddings, &unconstrained(4));
        assert_eq!(group_sizes(&groups), [2, 2]);
        let together = |i: usize, j: usize| groups.iter().any(|g| g.contains(&i) && g.contains(&j));
        assert!(together(0, 2) && together(1, 3));
        assert!(!together(0, 1));
    }

    /// The common case for this tool -- a meeting with one other person in it, or a talk --
    /// and the one an over-eager threshold ruins most visibly. Turns of one voice vary;
    /// varying by up to 30 degrees (distance 0.13) must not be enough to split them.
    #[test]
    fn one_speaker_with_varying_turns_yields_exactly_one_cluster() {
        let embeddings: Vec<Vec<f32>> = [0.0, 12.0, -9.0, 30.0, 5.0]
            .iter()
            .map(|d| at(*d))
            .collect();
        assert_eq!(agglomerate(&embeddings, &unconstrained(5)).len(), 1);
    }

    #[test]
    fn no_turns_yields_no_clusters() {
        assert!(agglomerate(&[], &[]).is_empty());
    }

    /// Where the threshold actually sits. 60 degrees is a distance of 0.5, above
    /// `MERGE_DISTANCE`; 50 degrees is 0.357, below it.
    #[test]
    fn the_threshold_splits_where_it_says_it_does() {
        assert_eq!(
            agglomerate(&[at(0.0), at(60.0)], &unconstrained(2)).len(),
            2
        );
        assert_eq!(
            agglomerate(&[at(0.0), at(50.0)], &unconstrained(2)).len(),
            1
        );
    }

    /// Two people talking over each other in one window are two people, no matter how
    /// similar the model thinks they sound. Identical embeddings is the strongest possible
    /// case for merging them, which is why the test uses it.
    #[test]
    fn turns_heard_simultaneously_are_never_merged() {
        let embeddings = vec![at(0.0), at(0.0)];
        let groups = agglomerate(&embeddings, &[(7, 0), (7, 1)]);
        assert_eq!(group_sizes(&groups), [1, 1]);
    }

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

    fn turn(start_s: f64, end_s: f64) -> LocalTurn {
        LocalTurn {
            start_s,
            end_s,
            window: 0,
            local_speaker: 0,
        }
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
    fn model(spec: &crate::ModelSpec) -> Option<Session> {
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
    /// inside [`MERGE_DISTANCE`]: WeSpeaker was trained on human speech and hears all of
    /// these as one unremarkable speaker. So the tests below use it to ask "does one voice
    /// stay one voice", never "do two voices stay apart" -- that second question needs real
    /// recordings, and the `cluster-speaker-track` example is how it gets asked.
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

    /// AC#4, and the reason this module exists: identity has to outlive a segmentation
    /// window. One voice speaks at the top of a forty-second track and again at the bottom;
    /// segmentation puts those in different windows and has no way to connect them, and
    /// clustering has to be what puts the two ends back together.
    ///
    /// Runs both real graphs, because a fake embedder would only be testing itself.
    #[test]
    fn a_voice_heard_at_both_ends_of_a_long_track_is_one_speaker() {
        let (Some(mut segmenter), Some(mut embedder)) = (
            model(&crate::SEGMENTATION_MODEL),
            model(&crate::EMBEDDING_MODEL),
        ) else {
            return;
        };

        let mut track = vec![0.0f32; 40 * TARGET_RATE as usize];
        let mut speak = |at_s: f64, samples: Vec<f32>| {
            let start = (at_s * TARGET_RATE as f64) as usize;
            track[start..start + samples.len()].copy_from_slice(&samples);
        };
        speak(1.0, voice(6.0, 130.0, [730.0, 1090.0, 2440.0]));
        speak(15.0, voice(6.0, 210.0, [400.0, 1900.0, 2600.0]));
        speak(32.0, voice(6.0, 130.0, [730.0, 1090.0, 2440.0]));

        let turns = crate::segment_speaker_track(&track, &mut segmenter).unwrap();
        let at = |at_s: f64| {
            turns
                .iter()
                .position(|t| t.start_s <= at_s && at_s <= t.end_s)
                .unwrap_or_else(|| panic!("nothing was heard at {at_s} s: {turns:?}"))
        };
        let (first, last) = (at(3.0), at(34.0));

        // Without this the test would pass on a `cluster_speaker_turns` that simply echoed
        // segmentation's local speaker index: the two ends have to be unconnectable by
        // anything except the voice for the assertion below to mean what it says.
        assert_ne!(
            turns[first].window, turns[last].window,
            "the fixture must put the two ends in different segmentation windows: {turns:?}"
        );

        let clustering = cluster_speaker_turns(&track, &turns, &mut embedder).unwrap();

        let opening = clustering.assignment[first].expect("the opening voice must be clustered");
        let closing = clustering.assignment[last].expect("the closing voice must be clustered");
        assert_eq!(
            opening, closing,
            "the same voice at 3 s and 34 s split into clusters {opening} and {closing}"
        );

        for embedding in clustering.turn_embeddings.iter().flatten() {
            assert_eq!(embedding.len(), 256);
            let length: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((length - 1.0).abs() < 1e-5, "length {length}");
        }

        // The claim that makes the handed-back vectors worth calibrating on: they are the
        // ones clustering actually grouped, not a plausible re-embedding. A refactor that
        // recomputed them -- with different fbank settings, or without normalizing -- would
        // still produce 256 unit-length floats and would fail only here.
        for cluster in &clustering.clusters {
            let mine: Vec<usize> = clustering
                .assignment
                .iter()
                .enumerate()
                .filter(|(_, assigned)| **assigned == Some(cluster.id))
                .map(|(index, _)| index)
                .collect();
            let members: Vec<Vec<f32>> = mine
                .iter()
                .map(|&i| clustering.turn_embeddings[i].clone().expect("assigned"))
                .collect();
            let rebuilt = reference_embedding(&(0..members.len()).collect::<Vec<_>>(), &members);
            for (a, b) in rebuilt.iter().zip(&cluster.embedding) {
                assert!((a - b).abs() < 1e-5, "cluster {} drifted", cluster.id);
            }
        }

        for cluster in &clustering.clusters {
            assert_eq!(cluster.embedding.len(), 256);
            let length: f32 = cluster.embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((length - 1.0).abs() < 1e-5, "length {length}");
            assert!(!cluster.representatives.is_empty());
            for clip in &cluster.representatives {
                assert!(
                    clip.seconds() >= MIN_REPRESENTATIVE_SECONDS - 1e-9,
                    "{clip:?}"
                );
                assert!(clip.start >= 0.0 && clip.end <= 40.0, "{clip:?}");
            }
        }
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
        assert!(same < MERGE_DISTANCE, "one voice split from itself: {same}");
    }

    /// A word too short to describe a voice must drop out of clustering rather than land
    /// somewhere arbitrary and drag a merge with it -- and must be counted on the way out,
    /// because every skipped turn is speech that ends up unattributed.
    #[test]
    fn a_turn_too_short_to_embed_is_skipped_and_counted() {
        let Some(mut embedder) = model(&crate::EMBEDDING_MODEL) else {
            return;
        };
        let mut track = vec![0.0f32; 10 * TARGET_RATE as usize];
        let speech = voice(3.0, 130.0, [730.0, 1090.0, 2440.0]);
        track[..speech.len()].copy_from_slice(&speech);

        let turns = [turn(0.0, 3.0), turn(5.0, 5.2)];
        let clustering = cluster_speaker_turns(&track, &turns, &mut embedder).unwrap();

        assert_eq!(clustering.assignment, [Some(0), None]);
        assert_eq!(clustering.skipped(), 1);
        assert_eq!(clustering.clusters.len(), 1);

        // The invariant `turn_embeddings` documents, and the one every caller zipping the
        // two vectors together relies on: a vector exists in exactly the positions a cluster
        // assignment does.
        let embedded: Vec<bool> = clustering
            .turn_embeddings
            .iter()
            .map(Option::is_some)
            .collect();
        let assigned: Vec<bool> = clustering.assignment.iter().map(Option::is_some).collect();
        assert_eq!(embedded, assigned);
    }

    /// What `enroll` will read out of `speaker_clusters.json` to recover the "Unknown N"
    /// numbering: each cluster's first appearance is the earliest turn it actually holds.
    ///
    /// The turns are handed over latest-first, because that is the mistake worth guarding --
    /// taking the group's first member rather than its earliest one would answer 12 s here,
    /// and the only symptom downstream would be two voices numbered the wrong way round.
    #[test]
    fn a_clusters_first_appearance_is_the_earliest_turn_it_holds() {
        let Some(mut embedder) = model(&crate::EMBEDDING_MODEL) else {
            return;
        };
        let mut track = vec![0.0f32; 20 * TARGET_RATE as usize];
        let mut speak = |at_s: f64, samples: Vec<f32>| {
            let start = (at_s * TARGET_RATE as f64) as usize;
            track[start..start + samples.len()].copy_from_slice(&samples);
        };
        speak(1.0, voice(3.0, 130.0, [730.0, 1090.0, 2440.0]));
        speak(12.0, voice(3.0, 300.0, [270.0, 2300.0, 3000.0]));

        let turns = [turn(12.0, 15.0), turn(1.0, 4.0)];
        let clustering = cluster_speaker_turns(&track, &turns, &mut embedder).unwrap();

        for cluster in &clustering.clusters {
            let earliest = turns
                .iter()
                .zip(&clustering.assignment)
                .filter(|(_, assigned)| **assigned == Some(cluster.id))
                .map(|(turn, _)| turn.start_s)
                .fold(f64::INFINITY, f64::min);
            assert_eq!(cluster.first_spoke_seconds, earliest, "{cluster:?}");
        }
        assert!(
            clustering
                .clusters
                .iter()
                .any(|c| c.first_spoke_seconds == 1.0),
            "{:?}",
            clustering.clusters
        );
    }
}
