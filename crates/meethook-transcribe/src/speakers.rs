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
//!
//! For the same reason it also gets [`group_distance`]: a way to ask how far two groups of
//! turns sit apart under either of the two criteria that disagree about it -- the average
//! linkage clustering merges on, and the centroid distance a later pass would threshold --
//! together with the factor that relates them. Asking that here rather than re-deriving it
//! outside is what keeps a diagnostic from quietly disagreeing with the code it diagnoses.

use std::collections::BTreeMap;

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
/// Measured on a real 43-minute meeting (session `20260810-093047`, six dominant clusters
/// holding 87.5% of the speech on a track the user confirms carried six people) rather than
/// inherited from the checkpoint's published behaviour:
///
/// | population                                          | min   | median      | max   |
/// |-----------------------------------------------------|-------|-------------|-------|
/// | two turns of one speaker                            | 0.077 | 0.232-0.396 | 0.683 |
/// | two speakers heard in one window, so known different | 0.270 | 0.837       | 1.118 |
///
/// Both populations are ground truth from segmentation co-occurrence, not from assuming who
/// is who. They overlap across `[0.270, 0.683]`, so no threshold separates them and 0.45 buys
/// one kind of mistake with the other. It is kept because every dominant speaker's own turns
/// average below it while the closest two dominant clusters average 0.604 apart, so the
/// nearest merge this constant declined is a long way above the cut.
///
/// The closest of those two clusters were confirmed by ear to be two different people, so
/// this constant is known to have separated them and not merely assumed to have. It is also
/// the case that they sit only 0.429 apart by *centroid* -- close enough that
/// [`crate::IDENTIFY_DISTANCE`], which thresholds that other distance at this same number,
/// files one of them under the other's name. Averaging distances is not the distance of
/// averages, and the two constants are not interchangeable despite being equal. See TASK-020.
///
/// The cut is not what strands short turns in clusters of their own -- that is average
/// linkage against a large group, and raising this constant to absorb them would need 0.6-0.8
/// and would merge the people above. See TASK-018.
///
/// Public so that a diagnostic reporting [`GroupDistance::average_linkage`] can say which
/// side of the decision each number fell on. A linkage column without the cut beside it asks
/// its reader to remember the threshold, and a reader who misremembers it reads the whole
/// report backwards.
pub const MERGE_DISTANCE: f32 = 0.45;

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
/// Agglomerative with average linkage: start from segmentation's own grouping, repeatedly merge
/// the two closest groups, stop when the closest pair is further apart than [`MERGE_DISTANCE`].
/// Average linkage rather than single (which chains one speaker into the next through a
/// single ambiguous turn) or complete (which shatters a speaker over one bad clip).
///
/// The implementation is the naive one -- recompute every group pair's average distance on
/// every merge -- because a meeting has hundreds of turns and a cleverer one would trade
/// something that is obviously correct for time nobody is waiting on.
///
/// `constraints` carries each embedding's `(window, local_speaker)` from segmentation, and it is
/// free supervision in *both* directions, because a local speaker index is an assertion about
/// who was talking inside one window:
///
///   - Two turns the model heard in one window under **different** indices are definitely
///     different people, so no merge may ever put them together however close their embeddings
///     look. [`pairwise_distances`] encodes that as an infinity.
///   - Two turns heard in one window under the **same** index are one person on exactly the same
///     authority. Windows do not overlap and segmentation reopens a turn for one index whenever
///     the silence inside it runs past `MAX_GAP_IN_TURN_S`, so such a pair is two turns only
///     because real silence separated them -- not because anything doubted it was one voice.
///
/// The second direction is applied by *seeding*: the initial partition is one group per distinct
/// `(window, local_speaker)` rather than one group per turn, so turns segmentation already called
/// one person start together and no sequence of merges can pull them apart. Seeding rather than
/// substituting a zero -- or a negative -- distance for a must-link pair, because a substitution
/// would change the criterion the loop merges on and poison every average running through that
/// pair; this changes only where the loop starts.
///
/// **The two directions cannot conflict.** Every turn carries exactly one
/// `(window, local_speaker)`, so the seeds are the equivalence classes of that key rather than a
/// transitive closure over pairs, and every pair inside one seed shares *both* halves of it. A
/// cannot-link pair needs the same window and *different* indices, which no intra-seed pair has.
/// So a seed is always internally finite, and the cannot-link constraint keeps working untouched:
/// a forbidden pair still makes every average linkage spanning it infinite, so no merge can bring
/// the two groups holding it together.
///
/// Seeding is not free, and the cost is a merge rather than a misattribution: a seed holding two
/// turns that sound different has a spread-out mean, and average linkage is centroid distance
/// inflated by that spread (see [`group_distance`]), so forcing turns together *raises* the
/// group's distance to everything else and can cost a merge elsewhere.
fn agglomerate(embeddings: &[Vec<f32>], constraints: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let n = embeddings.len();
    let distances = pairwise_distances(embeddings, constraints);

    // A `BTreeMap` rather than a hash map because the seed order decides which of two equally
    // close pairs the greedy search below takes first, so a randomized iteration order would make
    // clustering irreproducible -- and not even stable within one run, since two hash maps in one
    // process are seeded differently. Ordering by `(window, local_speaker)` is close to
    // chronological, a window index being the turn's start divided by the window length.
    let mut seeds: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    // `take(n)` because a group may only hold turns the distance matrix has rows for; `turn`
    // ascends, so every seed comes out sorted, which is the invariant the merge step below
    // maintains with `sort_unstable`.
    for (turn, &key) in constraints.iter().enumerate().take(n) {
        seeds.entry(key).or_default().push(turn);
    }
    let mut groups: Vec<Vec<usize>> = seeds.into_values().collect();

    // The argument above, asserted against the matrix that actually encodes it: a seed can never
    // hold a forbidden pair, so no starting group is infinitely far from itself.
    debug_assert!(
        groups.iter().all(|seed| {
            seed.iter()
                .all(|&i| seed.iter().all(|&j| distances[i * n + j].is_finite()))
        }),
        "a seed held a cannot-link pair, which one key per turn makes impossible"
    );

    loop {
        let mut best = None;
        for a in 0..groups.len() {
            for b in 0..a {
                let average = average_linkage(&groups[a], &groups[b], &distances, n);
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

/// Every pair of embeddings' cosine distance, with the cannot-link constraint substituted in.
///
/// Row-major and `n * n`, symmetric, zero down the diagonal. A pair segmentation heard in one
/// window under different local speaker indices is [`f32::INFINITY`] rather than its cosine,
/// which is how the constraint propagates without anything downstream re-checking it: an
/// infinite pair makes every [`average_linkage`] spanning it infinite too, so no group can
/// ever come to hold both turns.
fn pairwise_distances(embeddings: &[Vec<f32>], constraints: &[(usize, usize)]) -> Vec<f32> {
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
            let distance = if heard_at_once(constraints[i], constraints[j]) {
                // Heard at once: no evidence about their voices can make them one person.
                f32::INFINITY
            } else {
                1.0 - cosine
            };
            distances[i * n + j] = distance;
            distances[j * n + i] = distance;
        }
    }
    distances
}

/// Whether segmentation heard these two turns at once under different local speaker indices,
/// which makes them different people whatever their embeddings look like.
///
/// Each argument is one turn's `(window, local_speaker)`. The same window means the model was
/// asked who was talking during one ten-second stretch, and two different indices mean it
/// answered with two of them -- the cannot-link direction of the free supervision described on
/// [`agglomerate`].
///
/// One spelling of the rule for the two places that need it. [`pairwise_distances`] encodes it
/// as an infinity so that no merge can span it; `adoption.rs` reads it as a *label* on a pair
/// clustering never made, which is a question about a distance rather than about a merge and so
/// cannot be answered by reading the matrix. A second `==` and `!=` written out in the other
/// place is how a report comes to disagree with the clustering it reports on.
pub(crate) fn heard_at_once(a: (usize, usize), b: (usize, usize)) -> bool {
    a.0 == b.0 && a.1 != b.1
}

/// The average distance between two groups' members -- the criterion [`agglomerate`] merges on.
///
/// Reads the matrix [`pairwise_distances`] built, so a forbidden pair contributes an infinity
/// and makes the whole average infinite. That is deliberate and it is also why this is not the
/// same quantity as [`GroupDistance::average_linkage`], which is computed from the embeddings
/// with no constraint in it: one is a merge decision, the other is a distance.
fn average_linkage(a: &[usize], b: &[usize], distances: &[f32], n: usize) -> f32 {
    let sum: f32 = a
        .iter()
        .flat_map(|&i| b.iter().map(move |&j| (i, j)))
        .map(|(i, j)| distances[i * n + j])
        .sum();
    sum / (a.len() * b.len()) as f32
}

/// How far apart two groups of unit-length embeddings are, under both criteria at once.
///
/// See [`group_distance`] for the identity relating the three fields, which is the whole
/// reason they are returned together rather than measured separately.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupDistance {
    /// The mean of every cross-group pairwise cosine distance: the criterion clustering merges
    /// on, against [`MERGE_DISTANCE`].
    ///
    /// Computed from the embeddings alone, *without* the cannot-link substitution, so it is a
    /// distance rather than a decision and the identity below holds for it. Whether a merge is
    /// forbidden is a separate question with a separate answer, and folding the two into one
    /// number would erase both.
    pub average_linkage: f32,

    /// The cosine distance between the two groups' normalized means.
    ///
    /// This is what a leftover-adoption pass would threshold, and the quantity
    /// [`crate::IDENTIFY_DISTANCE`] is measured in -- the distance between a cluster's
    /// reference vector and another's.
    pub centroid: f32,

    /// `|a| * |b|`: the product of the lengths of the two *unnormalized* group means.
    ///
    /// At most 1, reached only when a group's members all point one way, and falling as either
    /// group grows or spreads out. It is the factor by which averaging distances exceeds the
    /// distance of averages.
    pub shrinkage: f32,
}

/// Both group distances and the shrinkage tying them together, or [`None`] if either group is
/// empty or has no direction.
///
/// Members must be unit length and of one dimensionality -- which is what one embedding model
/// produces, and what [`Clustering::turn_embeddings`] hands over.
///
/// For unit-length members these are not two independent measurements but one. With `a` and
/// `b` the *unnormalized* group means, the mean of the cross-group pairwise cosine distances
/// is `1 - a.b`, while centroid distance is `1 - (a/|a|).(b/|b|)`. So:
///
/// ```text
/// average_linkage = 1 - |a| * |b| * (1 - centroid)
/// ```
///
/// Average linkage is centroid distance inflated by the shrinkage of the two means, and a
/// group mean shrinks as its group grows and spreads. That inflation is the size bias TASK-018
/// is about: a two-second fragment offered to sixty-seven turns of one speaker is charged for
/// the spread of the group it is being compared to, so the cluster most likely to own it is the
/// hardest one for it to join. On clusters 1 and 3 of session `20260810-093047` the two read
/// 0.604 and 0.429, putting the shrinkage at 0.693.
///
/// `average_linkage` is computed as the honest mean over pairs rather than from the identity,
/// so that `the_two_group_distances_are_one_identity` is asserting a claim about this code and
/// not rearranging its own algebra.
///
/// [`None`] rather than zeroes, matching [`crate::Spread::of`]: the distance between a group
/// and nothing does not exist, and a fabricated 0.000 reads as two identical voices. The other
/// [`None`] case -- a group whose members cancel to a zero mean -- has no direction to compare,
/// so its centroid distance would be arbitrary rather than merely uncertain.
pub fn group_distance(a: &[&[f32]], b: &[&[f32]]) -> Option<GroupDistance> {
    let (unit_a, length_a) = group_mean(a)?;
    let (unit_b, length_b) = group_mean(b)?;

    let sum: f32 = a
        .iter()
        .flat_map(|x| b.iter().map(move |y| (x, y)))
        .map(|(x, y)| 1.0 - dot(x, y))
        .sum();

    Some(GroupDistance {
        average_linkage: sum / (a.len() * b.len()) as f32,
        centroid: 1.0 - dot(&unit_a, &unit_b),
        shrinkage: length_a * length_b,
    })
}

/// A group's mean direction, and the length of its mean before normalizing.
///
/// Same order of operations as [`reference_embedding`] -- average, then normalize -- so the
/// direction returned here is the vector a cluster stores. What this adds is the length that
/// step throws away, which for unit-length members is the group's coherence: 1 when they all
/// point one way, falling toward 0 as they spread.
///
/// [`None`] for an empty group, and for one whose members cancel exactly. The second is
/// unreachable for real voices but trivially reachable in a test, and a zero vector normalizes
/// to itself, which would otherwise be reported as a confident distance of 1.0 to everything.
fn group_mean(members: &[&[f32]]) -> Option<(Vec<f32>, f32)> {
    let mut mean = vec![0.0f32; members.first()?.len()];
    for member in members {
        for (m, v) in mean.iter_mut().zip(*member) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= members.len() as f32;
    }

    let length = mean.iter().map(|v| v * v).sum::<f32>().sqrt();
    if length == 0.0 {
        return None;
    }
    normalize(&mut mean);
    Some((mean, length))
}

/// Cosine of two unit-length vectors, which for them is just the dot product.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
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

    /// Constraints that say nothing in either direction: every turn alone in its own window, so
    /// no pair is forbidden from merging and no pair is forced to start together either. The
    /// distances are then the only thing grouping these turns, which is what every test using
    /// this is about.
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

    /// The must-link direction, and the strongest possible case against it: two turns
    /// segmentation heard in one window under one local speaker index, whose embeddings are
    /// orthogonal -- distance 1.0 against a cut of 0.45. Distance does not get a vote, because
    /// segmentation already said this is one person talking with a pause in it.
    #[test]
    fn one_local_speaker_stays_one_speaker_however_far_apart_the_turns_sound() {
        let groups = agglomerate(&[at(0.0), at(90.0)], &[(3, 1), (3, 1)]);
        assert_eq!(groups, vec![vec![0, 1]]);
    }

    /// A whole seed class rather than a chain of pairwise merges: three turns under one
    /// `(window, local_speaker)` are one group even though no two of them would have merged on
    /// their own.
    #[test]
    fn every_turn_of_one_local_speaker_lands_in_one_group() {
        let embeddings = vec![at(0.0), at(85.0), at(-80.0)];
        let groups = agglomerate(&embeddings, &[(5, 2), (5, 2), (5, 2)]);
        assert_eq!(groups, vec![vec![0, 1, 2]]);
    }

    /// Where the two directions of the constraint meet, and the case that says which one wins.
    /// Turn 2 is *identical* to turn 0, so nothing about their voices argues for keeping them
    /// apart -- but turn 2 was heard at once with turn 1, and turns 0 and 1 are one local
    /// speaker, so admitting turn 2 would put a forbidden pair in one cluster. It stays out.
    #[test]
    fn a_must_link_class_never_absorbs_a_turn_heard_at_once_with_one_of_its_members() {
        let embeddings = vec![at(0.0), at(90.0), at(0.0)];
        let groups = agglomerate(&embeddings, &[(3, 1), (3, 1), (3, 0)]);
        // Seeded in `(window, local_speaker)` order, so local speaker 0 comes back first.
        assert_eq!(groups, vec![vec![2], vec![0, 1]]);
    }

    /// AC#4 at the unit level: the same turns and constraints produce the same clusters in the
    /// same order, twice. Seeding puts a map between segmentation and the greedy merge loop, and
    /// the loop breaks ties by group position, so a randomized iteration order would make
    /// clustering irreproducible -- this is what fails if anyone reaches for a hash map.
    ///
    /// The expected value also pins the documented seed order, `(window, local_speaker)`
    /// ascending, so that tie-break order is a stated contract rather than an accident. These
    /// four classes are mutually orthogonal, so no merge happens and the groups returned are the
    /// seeds themselves.
    #[test]
    fn clustering_is_reproducible_and_seeded_in_window_order() {
        let embeddings = vec![
            at(0.0),
            at(4.0),
            at(90.0),
            at(94.0),
            at(180.0),
            at(184.0),
            at(270.0),
        ];
        let constraints = [(0, 1), (0, 1), (1, 0), (1, 0), (1, 2), (1, 2), (2, 0)];

        let groups = agglomerate(&embeddings, &constraints);

        assert_eq!(groups, agglomerate(&embeddings, &constraints));
        assert_eq!(
            groups,
            vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6]],
            "seeds must come out ordered by (window, local_speaker)"
        );
    }

    /// Borrowed views of the members of one group, the shape [`group_distance`] takes.
    fn vectors<'a>(embeddings: &'a [Vec<f32>], group: &[usize]) -> Vec<&'a [f32]> {
        group.iter().map(|&i| embeddings[i].as_slice()).collect()
    }

    /// Group shapes worth measuring a distance over: two singletons, a singleton against a
    /// spread group, two spread groups, and lopsided sizes -- the last because shrinkage is
    /// where group size enters, so a pair of groups with very different spreads is the case
    /// most likely to expose an arithmetic slip.
    fn shapes() -> Vec<(Vec<usize>, Vec<usize>)> {
        vec![
            (vec![0], vec![4]),
            (vec![0], vec![4, 5, 6]),
            (vec![0, 1, 2], vec![4, 5, 6]),
            (vec![0, 1, 2, 3], vec![6]),
        ]
    }

    /// A cloud of one voice around 0 degrees and another around 70, spread enough that the two
    /// group means are visibly shorter than their members.
    fn two_clouds() -> Vec<Vec<f32>> {
        [0.0, 18.0, -14.0, 7.0, 70.0, 88.0, 55.0]
            .iter()
            .map(|d| at(*d))
            .collect()
    }

    /// AC#2 of TASK-018.01, and the reason [`GroupDistance`] returns three numbers rather than
    /// two: average linkage is centroid distance inflated by the shrinkage of the two group
    /// means. If a refactor ever computes one of the columns differently from the others, this
    /// is what says so, and it is the claim the whole stranded-cluster report rests on.
    #[test]
    fn the_two_group_distances_are_one_identity() {
        let embeddings = two_clouds();
        for (left, right) in shapes() {
            let measured =
                group_distance(&vectors(&embeddings, &left), &vectors(&embeddings, &right))
                    .expect("neither group is empty");
            let from_identity = 1.0 - measured.shrinkage * (1.0 - measured.centroid);
            assert!(
                (measured.average_linkage - from_identity).abs() < 1e-5,
                "{left:?} vs {right:?}: linkage {} but the identity says {from_identity} \
                 (centroid {}, shrinkage {})",
                measured.average_linkage,
                measured.centroid,
                measured.shrinkage
            );
        }
    }

    /// The claim that makes reporting [`GroupDistance::average_linkage`] honest: it is the same
    /// number `agglomerate` compares against [`MERGE_DISTANCE`], not a plausible relative of
    /// it. Unconstrained embeddings, because the constraint substitutes an infinity and the two
    /// then deliberately disagree.
    #[test]
    fn average_linkage_matches_what_agglomerate_merges_on() {
        let embeddings = two_clouds();
        let n = embeddings.len();
        let distances = pairwise_distances(&embeddings, &unconstrained(n));

        for (left, right) in shapes() {
            let merged_on = average_linkage(&left, &right, &distances, n);
            let reported =
                group_distance(&vectors(&embeddings, &left), &vectors(&embeddings, &right))
                    .expect("neither group is empty")
                    .average_linkage;
            assert!(
                (merged_on - reported).abs() < 1e-6,
                "{left:?} vs {right:?}: agglomerate merges on {merged_on}, the report says \
                 {reported}"
            );
        }
    }

    /// The boundary that makes the size bias legible. Two one-member groups have means equal to
    /// their members, so there is no shrinkage and the two criteria agree exactly -- which says
    /// that every gap between the two columns elsewhere is group size and spread, and nothing
    /// else.
    #[test]
    fn two_single_turn_groups_have_no_shrinkage() {
        let embeddings = two_clouds();
        let measured = group_distance(&vectors(&embeddings, &[0]), &vectors(&embeddings, &[4]))
            .expect("neither group is empty");

        assert!((measured.shrinkage - 1.0).abs() < 1e-6, "{measured:?}");
        assert!(
            (measured.average_linkage - measured.centroid).abs() < 1e-6,
            "{measured:?}"
        );
    }

    /// Two groups nothing can be said about: one with no members, and one whose members cancel
    /// to a mean of zero. Both are [`None`] rather than a fabricated distance -- a zero vector
    /// normalizes to itself and would otherwise report a confident 1.000 to everything.
    #[test]
    fn a_group_with_no_direction_has_no_distance() {
        let voice: &[f32] = &[1.0, 0.0, 0.0, 0.0];
        let opposite: &[f32] = &[-1.0, 0.0, 0.0, 0.0];

        assert_eq!(group_distance(&[voice], &[]), None);
        assert_eq!(group_distance(&[], &[voice]), None);
        assert_eq!(group_distance(&[voice, opposite], &[voice]), None);
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

    /// A turn as segmentation would report it: in the window its start falls inside.
    ///
    /// The window is derived rather than fixed at 0 because it is not decoration -- turns sharing
    /// a `(window, local_speaker)` are must-linked by [`agglomerate`], so a helper that filed
    /// every turn in window 0 would silently force every turn a test built into one cluster. Real
    /// windows are [`crate::segmentation::WINDOW_SECONDS`] long and laid end to end, so a turn
    /// starts inside its own window; tests that need two local speakers set that field
    /// themselves.
    fn turn(start_s: f64, end_s: f64) -> LocalTurn {
        LocalTurn {
            start_s,
            end_s,
            window: (start_s / crate::segmentation::WINDOW_SECONDS) as usize,
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
