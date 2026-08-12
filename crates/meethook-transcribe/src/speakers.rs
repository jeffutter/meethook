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
//! linkage clustering merges on, and the centroid distance `adopt_below_floor` thresholds --
//! together with the factor that relates them. Asking that here rather than re-deriving it
//! outside is what keeps a diagnostic from quietly disagreeing with the code it diagnoses.

use std::collections::BTreeMap;

use meethook_session::{MIN_REPRESENTATIVE_SECONDS, RepresentativeSegment, SpeakerCluster};
use ort::session::Session;
use ort::value::TensorRef;

use crate::audio::TARGET_RATE;
use crate::fbank::{Fbank, MEL_BINS};
use crate::progress::Phase;
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
/// nearest merge this constant declined is a long way above the cut. That 0.604 was measured
/// before TASK-018 changed the grouping; on the clustering that ships now the same two voices
/// average **0.656** apart, so the margin above the cut grew rather than shrank.
///
/// The closest of those two clusters were confirmed by ear to be two different people, so
/// this constant is known to have separated them and not merely assumed to have. It is also
/// the case that they sit only 0.429 apart by *centroid*, which is the distance
/// [`crate::IDENTIFY_DISTANCE`] thresholds -- and while that constant was also 0.45 it filed
/// one of them under the other's name. Averaging distances is not the distance of averages,
/// which is why one value cannot serve both: see [`group_distance`] for the identity relating
/// them, and `IDENTIFY_DISTANCE` for the populations that have since moved it to its own,
/// lower value. That constant's value is not this one's and neither follows the other.
/// See TASK-020.
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

/// How close a stranded fragment has to sit to a speaker before `adopt_below_floor` hands it
/// over.
///
/// # What this thresholds
///
/// Cosine distance between **two group means**: a below-floor group's normalized mean on one
/// side, an above-floor cluster's on the other -- [`GroupDistance::centroid`], 0 for the same
/// direction and 1 for orthogonal. Not turn-to-turn, and *not* [`GroupDistance::average_linkage`],
/// which is a different number on the same pair of groups.
///
/// # Where the value comes from
///
/// From two populations segmentation labelled by itself on session `20260810-093047`
/// (TASK-018.02.02.01). Neither needed an enrolment and neither was settled by ear:
///
/// | population                                    | pairs | min   | median | p95   | max   |
/// |-----------------------------------------------|-------|-------|--------|-------|-------|
/// | same speaker, leave-one-class-out             | 39    | 0.060 | 0.137  | 0.221 | 0.284 |
/// | different speakers, cannot-link               | 20    | 0.173 | 0.731  | 1.012 | 1.051 |
///
/// A **positive** is a must-link class -- two or more embedded turns sharing one
/// `(window, local_speaker)`, which segmentation heard as one person -- that landed wholly inside
/// an above-floor cluster, measured against the rest of that cluster with the class excluded. That
/// is this pass's own shape: a few seconds of one voice against a speaker estimated from minutes.
/// A **negative** is a below-floor cluster the same-window cannot-link constraint bars from an
/// above-floor one.
///
/// Scored through [`crate::score_trials`], so the boundary convention below is the one that module
/// states and tests -- accept is *strictly* below the cut. The two populations **overlap across
/// `[0.173, 0.284]`**, so no cut separates them; equal error is 10.1% at 0.196, and the largest cut
/// that misattributes nobody is 0.173, which rejects 30.8% of the labelled same-speaker pairs.
///
/// # Why the value is not that misattribution-free 0.173
///
/// Two halves, and neither softens the other.
///
/// **The negatives price a rule this pass does not use.** Every one of those 20 pairs is a pair the
/// same-window constraint *already refuses*, and `adopt_below_floor` takes its argmax **among
/// permitted targets only** -- a barred target is never a candidate, so its distance is never
/// looked at. A cut at 0.25 does not adopt the 0.173 pair; that pair is vetoed before any distance
/// is compared. Reading 0.173 as the safe cut double-counts a protection the pass already has.
/// What the negatives legitimately bound is trust in the *unblocked* offers, where nothing
/// protects anybody.
///
/// **And 20 pairs is one observation.** The misattribution-free cut *is* the minimum of the
/// different-speaker side, so it moves wherever that single closest pair moves, and nothing says
/// the closest pair seen is near the closest pair possible. The report prints it as a bound to
/// check by ear rather than a number to ship, and this constant treats it that way.
///
/// The population that does share this pass's shape is the positives, and all 39 sit at or below
/// 0.284. So the admissible window is `[0.196, 0.296)` -- bounded below by where rejecting real
/// same-speaker fragments begins to bite, above by the ceiling in the next paragraph -- and
/// **0.25** is the low side of it: it rejects only the upper tail of the labelled positives, and
/// on that session it adopts 15 fragments holding 100.4 s. Every cut in `[0.254, 0.296)` adopts
/// the same 16, so the extra 0.04 of width buys exactly one fragment.
///
/// # The ceiling, and the fragment that lowers it
///
/// Hard-capped strictly below **0.429**: the centroid gap between clusters 1 and 3 of that session,
/// Andrew and Ryan, confirmed by ear to be two different people. A cut at or above that is
/// measuring a gap two speakers fit inside.
///
/// Capped further below **0.296**, which is where a 7.8 s fragment (turns 100.4-101.6 and
/// 103.3-110.0) sits from Andrew. That clip may be Alex, a real seventh participant, and
/// the enrolled evidence cannot settle it -- Alex's stored reference *is* the centroid of the old
/// blended cluster that clip came out of, so distances to it are circular for exactly this
/// question. Adopting it costs a silent 7.8 s misattribution of a real participant if it is his;
/// declining it costs one more visible `Unknown N` if it is not. 0.25 keeps 0.046 of margin under
/// it where 0.29 would keep 0.006.
///
/// # Why it is not [`MERGE_DISTANCE`], and this is not a calibration difference
///
/// Because it thresholds a **different quantity**, not because it is a different pass.
/// `MERGE_DISTANCE` governs average linkage -- the mean over every cross-group pair of turn
/// distances -- and for unit-length members [`group_distance`] gives the exact relation:
///
/// ```text
/// average_linkage = 1 - shrinkage * (1 - centroid)
/// ```
///
/// so average linkage is centroid distance inflated by the shrinkage of the two means, and the two
/// numbers diverge further the larger and more spread out either group is. The worked pair is the
/// ceiling above: clusters 1 and 3 read **0.604** linkage and **0.429** centroid, putting the
/// shrinkage at **0.693**. Two constants of equal value would therefore be two different cuts.
/// TASK-020 is the live bug that came of describing two quantities in one set of words; that is
/// the mistake this comment exists to not repeat.
///
/// Public for the same reason [`MERGE_DISTANCE`] is: `cluster-speaker-track` scores its two
/// populations at this cut and prints the sweep around it, and a calibration constant a
/// diagnostic has to keep its own copy of is a constant that drifts out of agreement with the
/// code it claims to describe.
pub const ADOPTION_DISTANCE: f32 = 0.25;

/// How much speech a cluster's centroid has to rest on before `adopt_below_floor` will offer
/// fragments to it.
///
/// The quantity is not "how small a fragment is". It is how much evidence a mean is estimated
/// from before it is trustworthy enough to own somebody else's speech: below the floor a cluster
/// is a fragment looking for an owner, at or above it a cluster is a speaker that could be one.
/// The convention is `speech_seconds < floor` is a fragment and `>= floor` is a speaker, so a
/// cluster sitting exactly *on* the floor is a speaker -- which is the same convention
/// `crate::adoption_populations` partitions on, and the two must agree or the report describes a
/// different pass from the one that ships.
///
/// # Where 30 s comes from
///
/// On session `20260810-093047` the six clusters the user confirms are the six people in the room
/// hold 426.8 / 372.3 / 124.8 / 105.0 / 66.0 / 47.0 s, and the largest of the 65 leftovers holds
/// 12.3 s. **Any floor `f` with `12.3 < f <= 47.0` gives exactly that partition** -- 6 speakers and
/// 65 fragments -- so this value is insensitive across a 34.7 s band rather than fitted to one
/// recording. 30 s sits near the middle of that band.
///
/// Both edges are consequences rather than curiosities. Below 12.3 s the largest leftover becomes
/// an adoption *target*, so a fragment could be adopted into 12.3 s of speech -- which is the
/// failure the floor exists to prevent. Above 47.0 s the smallest real speaker stops being one and
/// 47.0 s of a participant goes looking for an owner.
///
/// # Why absolute seconds rather than a share of the meeting
///
/// The quantity is how much evidence a centroid rests on, and that does not change because
/// somebody else talked more. The consequence is that on a short recording where nobody clears the
/// floor nothing is adopted at all, which is the conservative direction and is exactly the no-op
/// the model-gated tests in this module see.
///
/// Public because `cluster-speaker-track` partitions its report on this and defaults `--floor` to
/// it, so that the report describes the pass that ships rather than a neighbouring one. The flag
/// stays, because reading the report at another floor is how the band above gets re-measured on a
/// recording whose gap sits elsewhere.
pub const SPEAKER_FLOOR_SECONDS: f64 = 30.0;

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

    // Embedded turns, where each came from in `turns`, and how long each one is. The durations
    // are built here rather than derived where they are needed, so that the talk time the
    // adoption pass partitions on and the talk time that reaches
    // [`SpeakerCluster::speech_seconds`] cannot become two numbers.
    let mut embeddings = Vec::new();
    let mut sources = Vec::new();
    let mut seconds = Vec::new();
    // One filterbank pass and one embedding inference per turn, and a long meeting has
    // hundreds of turns -- the second half of the diarization stretch that used to be silent.
    let mut phase = Phase::start("diarize: embedding voices");
    for (index, turn) in turns.iter().enumerate() {
        phase.at(index, turns.len());
        if turn.end_s - turn.start_s < MIN_EMBEDDABLE_SECONDS {
            continue;
        }
        let Some(embedding) = embed(&mut fbank, embedder, slice(samples_16k, turn))? else {
            continue;
        };
        embeddings.push(embedding);
        sources.push(index);
        seconds.push(turn.end_s - turn.start_s);
    }
    phase.done();

    let constraints: Vec<(usize, usize)> = sources
        .iter()
        .map(|&i| (turns[i].window, turns[i].local_speaker))
        .collect();
    let mut groups = adopt_below_floor(
        agglomerate(&embeddings, &constraints),
        &embeddings,
        &constraints,
        &seconds,
    );

    // Most talkative first, so cluster 0 is the person the meeting was mostly with and the
    // ids mean something to a human reading the file.
    let spoken = |group: &Vec<usize>| -> f64 { group.iter().map(|&e| seconds[e]).sum() };
    groups.sort_by(|a, b| spoken(b).total_cmp(&spoken(a)));

    // After the sort, so the indices are cluster ids; and after every pass that rewrites
    // `groups`, which is the only placement that stays correct. `adopt_below_floor` runs
    // between `agglomerate` and this sort, because it moves members between groups and empties
    // the ones it adopts -- so a relation computed before it would describe a grouping that no
    // longer exists, while one computed here describes the grouping that ships.
    let exclusions = heard_at_once_between(&groups, &constraints);

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
            heard_at_once_with: exclusions[id].clone(),
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

    // Every pass looks at every surviving pair, and each pass merges at most one of them, so
    // the work is quadratic in seeds -- a silent minute of its own on a meeting with a
    // four-figure turn count. One merge is the natural tick, and the seed count is the most
    // merges that can ever happen.
    let seeded = groups.len();
    let mut phase = Phase::start("diarize: clustering voices");

    loop {
        phase.at(seeded - groups.len(), seeded);
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
    phase.done();
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

/// Lifts [`heard_at_once`] from turns to groups: whether any turn of one was heard at once with
/// any turn of the other, which makes the two groups two different people.
///
/// One witnessing pair is proof and the search stops there. Counting them would invite a
/// threshold on evidence that is already categorical.
fn heard_apart(a: &[usize], b: &[usize], constraints: &[(usize, usize)]) -> bool {
    a.iter().any(|&i| {
        b.iter()
            .any(|&j| heard_at_once(constraints[i], constraints[j]))
    })
}

/// Hands each group holding less than [`SPEAKER_FLOOR_SECONDS`] of speech to the nearest speaker
/// that is allowed to have it, and leaves it alone when there is no such speaker.
///
/// Clustering strands short turns: a two-second fragment is charged for the spread of any large
/// group it is compared to (see [`group_distance`]), so the cluster most likely to own it is the
/// hardest one for it to join, and it ends up a cluster of one. On session `20260810-093047` that
/// is 65 leftover clusters beside 6 real speakers. This is the second pass that sweeps them up,
/// and it is settled practice -- pyannote's clustering prunes clusters under a `min_cluster_size`
/// and reassigns them to the nearest retained centroid. What is specific here is that it can
/// **decline**, and that it honours the cannot-link constraint pyannote has no equivalent of.
///
/// `groups` is what [`agglomerate`] returned, `embeddings` the unit-length vector per embedded
/// turn, `constraints` each turn's `(window, local_speaker)`, and `seconds` each turn's duration
/// -- all four indexed the same way. Returns the same members regrouped: every input turn appears
/// in exactly one output group, and no output group is empty.
///
/// # The decisions, rather than the mechanics
///
/// **Targets are above-floor groups only.** A fragment is never adopted into another fragment, and
/// two above-floor groups are never merged with each other however close they sit -- so the six
/// speakers a recording has stay six by construction rather than by luck. Whether two speakers
/// should have merged was `agglomerate`'s question and it has already been answered.
///
/// **Argmax among permitted targets, then the cut** -- not argmax then veto. A fragment whose
/// nearest speaker is barred by the constraint is offered to its next-nearest *permitted* one; a
/// fragment every speaker bars is declined. `identify_clusters` applies the same constraint the
/// other way round and its documentation says why the two differ: adoption's constraint holds
/// between a fragment and each candidate, so it is known before the choice; identification's does
/// not exist until some other cluster has taken the name. Adoption vetoes candidates.
/// Identification vetoes decisions.
///
/// **Permitted is judged against the target as it stands, not as it arrived.** Two fragments that
/// were heard at once with *each other* can both be nearest to one speaker, and adopting both
/// would put a forbidden pair in one cluster by the back door. So the second one is offered its
/// next-nearest permitted speaker instead, exactly as if the first had always been part of that
/// group. This is the only order-dependent step, and fragments are visited in ascending group
/// index, which [`agglomerate`] fixes deterministically.
///
/// **Centroids are frozen and there is one pass.** Every distance is measured against the grouping
/// as it arrived, so what one fragment is given cannot move the mean another is measured against.
/// That makes the result independent of the order adoptions are applied in -- determinism for
/// free rather than argued -- and makes it the same operation the sweep in TASK-018.02.02 measured.
/// Iterative re-centroiding is a different pass and deliberately not this one.
///
/// **Strictly below [`ADOPTION_DISTANCE`]**, which is how `crate::score_trials` and
/// `identify_clusters` spell the same comparison, so the report the constant was chosen from and
/// the pass that ships it cannot differ by one `<`.
///
/// **An exact tie goes to the lowest group index**, which after `agglomerate` is
/// `(window, local_speaker)` order -- stated so that a rerun cannot move a fragment between two
/// equidistant speakers with nothing having changed.
///
/// # What happens to the fragments it declines: they stay clusters, and are not suppressed
///
/// A declined fragment comes back as its own cluster and reaches the user as another `Unknown N`.
/// Suppressing it instead -- dropping its assignment -- is tempting, because the residue is large:
/// on the session above this pass leaves roughly 56 clusters, not 7. It is refused for three
/// reasons.
///
/// This module is on record that the two mistakes are not symmetric: a visible extra speaker is an
/// error the user fixes in `enroll` in ten seconds, and a silent misattribution lands in a
/// transcript nobody will re-read. See [`MERGE_DISTANCE`].
///
/// And suppression is **not** the absence of a misattribution. `merge::attribute` falls back to
/// the nearest diarized turn in time when no turn overlaps a span, so a suppressed fragment is
/// still attributed -- to whoever happened to be speaking beside it, invisibly, with no cluster to
/// inspect. That is already what the turns under [`MIN_EMBEDDABLE_SECONDS`] get, and it is a worse
/// answer here because these turns are long enough to have an opinion about.
///
/// Mechanically, too: [`Clustering::skipped`] counts `assignment.is_none()` and is documented as
/// how many turns were too short to embed. Routing declined fragments through the same `None`
/// would make that counter conflate two unrelated reasons, and it is the first line of every
/// report. Suppression would have to split that count in two before it could be honest.
fn adopt_below_floor(
    groups: Vec<Vec<usize>>,
    embeddings: &[Vec<f32>],
    constraints: &[(usize, usize)],
    seconds: &[f64],
) -> Vec<Vec<usize>> {
    let vectors = |group: &[usize]| -> Vec<&[f32]> {
        group.iter().map(|&e| embeddings[e].as_slice()).collect()
    };
    let is_speaker = |group: &[usize]| -> bool {
        group.iter().map(|&e| seconds[e]).sum::<f64>() >= SPEAKER_FLOOR_SECONDS
    };

    let above: Vec<usize> = (0..groups.len())
        .filter(|&g| is_speaker(&groups[g]))
        .collect();
    let below: Vec<usize> = (0..groups.len())
        .filter(|&g| !is_speaker(&groups[g]))
        .collect();
    // Nobody to adopt into, or nobody to adopt: both are the no-op, and both are ordinary. A
    // recording where one person talked for four minutes has no fragments; one where nobody
    // cleared the floor has no speaker whose mean is worth trusting.
    if above.is_empty() || below.is_empty() {
        return groups;
    }

    // Every offer, measured before any adoption is applied. This is what "frozen centroids"
    // means concretely: the grid is complete before a single member moves.
    let offers: Vec<Vec<Option<f32>>> = below
        .iter()
        .map(|&small| {
            above
                .iter()
                .map(|&large| {
                    group_distance(&vectors(&groups[small]), &vectors(&groups[large]))
                        .map(|distance| distance.centroid)
                })
                .collect()
        })
        .collect();

    let mut groups = groups;
    for (nth, &small) in below.iter().enumerate() {
        let mut nearest: Option<(f32, usize)> = None;
        for (mth, &large) in above.iter().enumerate() {
            if heard_apart(&groups[small], &groups[large], constraints) {
                continue;
            }
            // No direction to compare -- unreachable for real voices, and a decline rather
            // than an arbitrary distance if it ever happens.
            let Some(distance) = offers[nth][mth] else {
                continue;
            };
            // Strictly nearer, and `above` ascends, so an exact tie keeps the lower group.
            if nearest.is_none_or(|(held, _)| distance < held) {
                nearest = Some((distance, large));
            }
        }

        if let Some((_, large)) = nearest.filter(|&(distance, _)| distance < ADOPTION_DISTANCE) {
            let adopted = std::mem::take(&mut groups[small]);
            groups[large].extend(adopted);
            // Members ascend, which is the invariant `agglomerate` maintains and which
            // `reference_embedding` and the report order downstream both read.
            groups[large].sort_unstable();
        }
    }

    // An adopted group is empty now, and an empty group is not merely untidy: it would reach
    // `reference_embedding`, which indexes `members[0]`.
    groups.retain(|group| !group.is_empty());
    groups
}

/// Lifts [`heard_apart`] to a whole grouping: for each group, the ids of the groups it
/// holds a turn heard at once with.
///
/// `groups` arrives in cluster-id order -- sorted by talk time -- so a position in the
/// returned outer vector *is* the cluster id, and so are the ids inside it. Each inner list
/// is ascending and holds no duplicates, and the relation is emitted in both directions,
/// which is the symmetry [`SpeakerCluster::heard_at_once_with`] promises.
///
/// `constraints` is indexed by embedded-turn index, the way the members of `groups` are.
///
/// Pure, so the rule can be tested without a model or an audio track.
fn heard_at_once_between(groups: &[Vec<usize>], constraints: &[(usize, usize)]) -> Vec<Vec<u32>> {
    let mut exclusions = vec![Vec::new(); groups.len()];
    for a in 0..groups.len() {
        // `agglomerate` seeds one group per turn and only ever merges pairs whose distance
        // is finite, and `adopt_below_floor` only adopts into a group nothing in the fragment
        // was heard at once with -- so a group can never contain two turns heard at once and no
        // group can exclude itself. Cheap to keep honest rather than assumed.
        debug_assert!(
            !heard_apart(&groups[a], &groups[a], constraints),
            "group {a} excludes itself"
        );
        for b in (a + 1)..groups.len() {
            if heard_apart(&groups[a], &groups[b], constraints) {
                exclusions[a].push(b as u32);
                exclusions[b].push(a as u32);
            }
        }
    }
    exclusions
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
    /// This is what [`ADOPTION_DISTANCE`] thresholds, and the quantity
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
///
/// Visible to the crate because [`crate::reference_duration_sweep`] builds references out of
/// parts of a cluster and has to build them the way [`reference_embedding`] does, or it would be
/// measuring an algorithm meethook does not run. Reaching for this rather than re-deriving a
/// mean over there is what makes that a fact about the code instead of a comment.
pub(crate) fn group_mean(members: &[&[f32]]) -> Option<(Vec<f32>, f32)> {
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

    /// The relation the clusters carry to disk, at the level that decides it. Groups 0 and 2
    /// each hold a turn from window 5 under a different local speaker, so they are two people;
    /// group 1 shares no window with either.
    ///
    /// Both directions must be present, and each list ascending, because that is the shape
    /// `SpeakerCluster::heard_at_once_with` promises its readers.
    #[test]
    fn groups_sharing_a_window_under_different_local_speakers_exclude_each_other() {
        // turn:              0       1       2       3       4
        let constraints = [(5, 0), (7, 0), (5, 1), (9, 0), (5, 1)];
        let groups = vec![vec![0, 1], vec![3], vec![2, 4]];

        assert_eq!(
            heard_at_once_between(&groups, &constraints),
            vec![vec![2], vec![], vec![0]]
        );
    }

    /// The same local speaker index in the same window is the *must*-link direction, and two
    /// turns in different windows say nothing at all. Neither is an exclusion, and a rule that
    /// keyed on the window alone would report both.
    #[test]
    fn groups_that_never_overlapped_exclude_nobody() {
        let constraints = [(5, 0), (6, 1), (5, 0)];
        let groups = vec![vec![0], vec![1], vec![2]];

        assert_eq!(
            heard_at_once_between(&groups, &constraints),
            vec![Vec::<u32>::new(), Vec::new(), Vec::new()]
        );
    }

    /// A group of one turn has nothing to be excluded from, and the empty case is a meeting
    /// where nobody spoke rather than an error.
    #[test]
    fn a_lone_group_and_no_groups_at_all_yield_no_exclusions() {
        assert_eq!(
            heard_at_once_between(&[vec![0]], &[(5, 0)]),
            vec![Vec::<u32>::new()]
        );
        assert!(heard_at_once_between(&[], &[]).is_empty());
    }

    /// Borrowed views of the members of one group, the shape [`group_distance`] takes.
    fn vectors<'a>(embeddings: &'a [Vec<f32>], group: &[usize]) -> Vec<&'a [f32]> {
        group.iter().map(|&i| embeddings[i].as_slice()).collect()
    }

    // [`adopt_below_floor`]. These pure tests are the pass's only coverage, and that is worth
    // saying out loud rather than assuming: every model-gated test in this module runs on a
    // track holding well under [`SPEAKER_FLOOR_SECONDS`] of speech -- 18 s in the longest --
    // so at the shipped floor none of them has a speaker to adopt into and none of them
    // exercises this pass at all. A green `MEETHOOK_ROOT` run says the pass is a no-op on a
    // short track, which is a real guarantee and is not the same one.

    /// A speaker with 40 s of speech and a two-second fragment 10 degrees away -- a centroid
    /// distance of 0.015, well inside the cut. The ordinary case the pass exists for.
    #[test]
    fn a_fragment_near_a_speaker_is_adopted_into_it() {
        let embeddings = vec![at(0.0), at(0.0), at(10.0)];

        let groups = adopt_below_floor(
            vec![vec![0, 1], vec![2]],
            &embeddings,
            &unconstrained(3),
            &[20.0, 20.0, 2.0],
        );

        assert_eq!(groups, vec![vec![0, 1, 2]]);
    }

    /// The pass declining, which is the half pyannote's equivalent does not have. 60 degrees is
    /// a centroid distance of 0.500, twice the cut: this fragment is nobody in the room, and
    /// filing it under the only speaker present would be the misattribution the constant exists
    /// to prevent.
    #[test]
    fn a_fragment_that_belongs_to_nobody_present_stays_a_cluster_of_its_own() {
        let embeddings = vec![at(0.0), at(0.0), at(60.0)];

        let groups = adopt_below_floor(
            vec![vec![0, 1], vec![2]],
            &embeddings,
            &unconstrained(3),
            &[20.0, 20.0, 2.0],
        );

        assert_eq!(groups, vec![vec![0, 1], vec![2]]);
    }

    /// AC#2 at the level that decides it, and the strongest possible case against the
    /// constraint: the fragment is *identical* to the speaker, distance 0.000, so nothing about
    /// their voices argues for keeping them apart. Segmentation heard turn 2 while turn 0 was
    /// speaking, under a different local speaker index, so it is somebody else whatever the
    /// embedding says.
    #[test]
    fn a_fragment_the_constraint_forbids_is_declined_however_close_it_sounds() {
        let embeddings = vec![at(0.0), at(0.0), at(0.0)];

        let groups = adopt_below_floor(
            vec![vec![0, 1], vec![2]],
            &embeddings,
            &[(0, 0), (1, 0), (0, 1)],
            &[20.0, 20.0, 2.0],
        );

        assert_eq!(groups, vec![vec![0, 1], vec![2]]);
    }

    /// Argmax among permitted targets rather than argmax then veto, as an assertion. The
    /// fragment's nearest speaker is the first (0.001) and its second choice the other (0.049),
    /// both far inside the cut; the first is barred. "Argmax then veto" would decline this
    /// fragment, which is the other rule and is not this one.
    #[test]
    fn a_fragment_whose_nearest_speaker_is_barred_is_offered_the_next_permitted_one() {
        let embeddings = vec![at(0.0), at(0.0), at(20.0), at(20.0), at(2.0)];
        // Turn 4 was heard at once with turn 0, and with nothing in the second speaker.
        let constraints = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1)];

        let groups = adopt_below_floor(
            vec![vec![0, 1], vec![2, 3], vec![4]],
            &embeddings,
            &constraints,
            &[20.0, 20.0, 20.0, 20.0, 2.0],
        );

        assert_eq!(groups, vec![vec![0, 1], vec![2, 3, 4]]);
    }

    /// And when every speaker is barred there is no next choice: the fragment is declined
    /// rather than handed to the least-bad option.
    #[test]
    fn a_fragment_every_speaker_bars_is_declined_rather_than_adopted_anywhere() {
        let embeddings = vec![at(0.0), at(0.0), at(5.0), at(5.0), at(2.0)];
        // Window 0 held three local speakers: one turn of each speaker, and the fragment.
        let constraints = [(0, 0), (1, 0), (0, 2), (3, 0), (0, 1)];

        let groups = adopt_below_floor(
            vec![vec![0, 1], vec![2, 3], vec![4]],
            &embeddings,
            &constraints,
            &[20.0, 20.0, 20.0, 20.0, 2.0],
        );

        assert_eq!(groups, vec![vec![0, 1], vec![2, 3], vec![4]]);
    }

    /// The back door into AC#2, and the reason the constraint is read against the target as it
    /// stands rather than as it arrived. Both fragments are inside the cut of the only speaker,
    /// and segmentation heard the two of *them* at once -- so adopting both would put a
    /// forbidden pair in one cluster without either adoption having crossed a barred pair.
    ///
    /// The second call is the counterfactual that stops this passing vacuously: with nothing
    /// forbidden, both are adopted.
    #[test]
    fn two_fragments_heard_at_once_with_each_other_do_not_both_join_one_speaker() {
        let embeddings = vec![at(0.0), at(0.0), at(5.0), at(8.0)];
        let seconds = [20.0, 20.0, 2.0, 2.0];

        let groups = adopt_below_floor(
            vec![vec![0, 1], vec![2], vec![3]],
            &embeddings,
            &[(0, 0), (1, 0), (9, 0), (9, 1)],
            &seconds,
        );
        assert_eq!(groups, vec![vec![0, 1, 2], vec![3]]);

        let groups = adopt_below_floor(
            vec![vec![0, 1], vec![2], vec![3]],
            &embeddings,
            &unconstrained(4),
            &seconds,
        );
        assert_eq!(groups, vec![vec![0, 1, 2, 3]], "both are inside the cut");
    }

    /// Frozen centroids as an assertion rather than a sentence. The speaker sits at 0 degrees;
    /// the first fragment is 40 degrees away (0.234, inside the cut) and the second 45 degrees
    /// away (0.293, outside it). Adopting the first would swing a *recomputed* centroid to 13.1
    /// degrees and pull the second inside, so a pass that re-centroided would take both -- and
    /// would take a different pair if the fragments arrived in the other order.
    ///
    /// The `merged` assertion is what keeps this from passing vacuously: it says the
    /// re-centroided answer really would have differed.
    #[test]
    fn centroids_are_frozen_so_one_adoption_cannot_change_another_s_answer() {
        let embeddings = vec![at(0.0), at(0.0), at(40.0), at(45.0)];

        let groups = adopt_below_floor(
            vec![vec![0, 1], vec![2], vec![3]],
            &embeddings,
            &unconstrained(4),
            &[20.0, 20.0, 2.0, 2.0],
        );

        assert_eq!(groups, vec![vec![0, 1, 2], vec![3]]);

        let merged = group_distance(
            &vectors(&embeddings, &[0, 1, 2]),
            &vectors(&embeddings, &[3]),
        )
        .expect("neither group is empty");
        assert!(
            merged.centroid < ADOPTION_DISTANCE,
            "re-centroiding would not have changed the second answer, so this proves nothing: \
             {merged:?}"
        );
    }

    /// AC#7's guarantee by construction: two speakers 5 degrees apart -- a centroid distance of
    /// 0.004, far inside the cut -- stay two speakers, because an above-floor group is never a
    /// candidate for adoption into anything. Whether those two should have merged was
    /// `agglomerate`'s question and this pass does not reopen it.
    ///
    /// Also the emptied group: the fragment's own group must not come back as an empty `Vec`,
    /// which `reference_embedding` would index into.
    #[test]
    fn speakers_are_never_merged_into_each_other_and_no_empty_group_survives() {
        let embeddings = vec![at(0.0), at(0.0), at(5.0), at(5.0), at(1.0)];

        let groups = adopt_below_floor(
            vec![vec![0, 1], vec![2, 3], vec![4]],
            &embeddings,
            &unconstrained(5),
            &[20.0, 20.0, 20.0, 20.0, 2.0],
        );

        assert_eq!(groups, vec![vec![0, 1, 4], vec![2, 3]]);
        assert!(groups.iter().all(|group| !group.is_empty()));
    }

    /// An exact tie has to resolve the same way on every run, or a `--force` re-transcribe
    /// could move a fragment between two equidistant speakers with nothing having changed.
    /// This fragment sits 10 degrees from each; the lower group keeps it.
    #[test]
    fn an_exact_tie_goes_to_the_lower_group() {
        let embeddings = vec![at(-10.0), at(-10.0), at(10.0), at(10.0), at(0.0)];

        let groups = adopt_below_floor(
            vec![vec![0, 1], vec![2, 3], vec![4]],
            &embeddings,
            &unconstrained(5),
            &[20.0, 20.0, 20.0, 20.0, 2.0],
        );

        assert_eq!(groups, vec![vec![0, 1, 4], vec![2, 3]]);
    }

    /// The floor's convention, which has to match the instrument that measured it exactly:
    /// `speech_seconds < floor` is a fragment and `>= floor` is a speaker, so a cluster sitting
    /// *on* the floor can be adopted into. A hair under it and there is no speaker anywhere, so
    /// the same two groups come back untouched.
    #[test]
    fn a_cluster_sitting_exactly_on_the_floor_is_a_speaker() {
        let embeddings = vec![at(0.0), at(2.0)];
        let offered = |speech: f64| {
            adopt_below_floor(
                vec![vec![0], vec![1]],
                &embeddings,
                &unconstrained(2),
                &[speech, 2.0],
            )
        };

        assert_eq!(offered(SPEAKER_FLOOR_SECONDS), vec![vec![0, 1]]);
        assert_eq!(
            offered(SPEAKER_FLOOR_SECONDS - 0.001),
            vec![vec![0], vec![1]]
        );
    }

    /// Every degenerate shape is the input back rather than a panic: no turns at all, nothing
    /// above the floor, and nothing below it. The last is the common one -- a meeting where one
    /// person talked and nothing was stranded.
    #[test]
    fn degenerate_shapes_are_no_ops_rather_than_panics() {
        assert!(adopt_below_floor(Vec::new(), &[], &[], &[]).is_empty());

        let embeddings = vec![at(0.0), at(3.0)];
        assert_eq!(
            adopt_below_floor(
                vec![vec![0, 1]],
                &embeddings,
                &unconstrained(2),
                &[20.0, 20.0]
            ),
            vec![vec![0, 1]],
            "one speaker and nothing looking for an owner"
        );
        assert_eq!(
            adopt_below_floor(
                vec![vec![0], vec![1]],
                &embeddings,
                &unconstrained(2),
                &[2.0, 2.0]
            ),
            vec![vec![0], vec![1]],
            "two fragments and no speaker either of them could join"
        );
    }

    /// AC#11 without a model: a track carrying one voice comes back as one cluster, and the
    /// pass leaves it alone whether that voice cleared the floor or not.
    #[test]
    fn one_voice_stays_one_cluster_through_the_pass() {
        let embeddings: Vec<Vec<f32>> = [0.0, 12.0, -9.0, 30.0, 5.0]
            .iter()
            .map(|d| at(*d))
            .collect();
        let constraints = unconstrained(5);
        let one = agglomerate(&embeddings, &constraints);
        assert_eq!(one.len(), 1);

        for speech in [1.0, 20.0] {
            assert_eq!(
                adopt_below_floor(one.clone(), &embeddings, &constraints, &[speech; 5]),
                one
            );
        }
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
