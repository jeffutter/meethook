//! Prints the speakers clustering finds in a recorded session, and how far apart they are.
//!
//! ```text
//! cargo run --release --example cluster-speaker-track ~/meethook/sessions/20260809-021730
//! cargo run --release --example cluster-speaker-track some-other-recording.wav
//! cargo run --release --example cluster-speaker-track ~/meethook/sessions/20260809-021730 --write
//! cargo run --release --example cluster-speaker-track ~/meethook/sessions/20260810-093047 --floor 20
//! ```
//!
//! A session directory means its `speaker.wav`; any other path is used as given. `--write`
//! saves `speaker_clusters.json` into the session directory, which is how `enroll` gets
//! something real to develop against without re-running transcription. `--floor` sets the
//! talk-time below which a cluster is treated as a fragment rather than a speaker, defaulting to
//! the [`SPEAKER_FLOOR_SECONDS`] the adoption pass ships with.
//!
//! Output runs to thousands of lines on a long meeting, so redirect it to a file.
//!
//! Why this exists: the unit tests can prove that clustering separates vectors that are far
//! apart and joins vectors that are close, but not that a *voice* produces vectors in the
//! right places. Synthetic audio cannot settle that either -- measured against this
//! checkpoint, buzzes across a wide range of pitches and formants all land within 0.17 of
//! each other, well inside the merge threshold, because the network was trained on people.
//! So the two questions that matter are answered here, by a person, on a real recording:
//!
//!   1. Does each participant get exactly one cluster, rather than three?
//!   2. Does the person who spoke at the start and again at the end get one cluster?
//!
//! The distance matrix is printed for the same reason. If two clusters that are the same
//! person sit at 0.5, or two people sit at 0.4, that is the number to move `MERGE_DISTANCE`
//! against -- and it is far better to learn it here than from a transcript that quietly
//! attributed one person's words to another.
//!
//! That matrix on its own cannot move the threshold, though, which is why the turn-to-turn
//! blocks are here as well. It compares cluster *means*, and a mean only exists because the
//! grouping already happened -- every distance in it is a distance that survived the cut, so
//! it reports how far apart the decisions ended up and never how close they came to going
//! the other way. The two numbers that decide whether 0.45 has room are underneath it: the
//! largest distance between two turns *inside* one cluster, which is how near one person came
//! to splitting in two, and the smallest distance between turns of two *different* clusters,
//! which is how near two people came to merging. Means always look safer than the clouds they
//! summarize, so both are printed next to each other.
//!
//! Those two are still conditional on the grouping being right, and only a person with the
//! recording can settle that -- hence the turn timings beside every extreme, so the pair that
//! produced a number can be played before the number is believed. The last block needs no such
//! trust: two turns the segmentation model heard in the *same* window under different local
//! speaker indices are two people as a matter of what was in the audio, whatever any threshold
//! later decides, so their distances are different-speaker evidence that no grouping decision
//! stands behind.
//!
//! The two blocks after that are about the other failure this clustering has, which is not a
//! speaker split in two but a speaker shattered into ninety: a handful of real voices plus a
//! long tail of one- and two-second fragments, each in a cluster of its own. Both blocks exist
//! to say how much of that tail a sweep can take and at what cost. **They now run on a
//! clustering `adopt_below_floor` has already swept**, so what they describe is the residue that
//! pass declined and what a *further* sweep would do to it -- not the material
//! `ADOPTION_DISTANCE` was chosen from, which was measured before the pass existed and cannot be
//! re-derived from a run that includes it. The first prints every remaining fragment's distance
//! to every real speaker under both criteria -- the average linkage clustering merged on and the
//! centroid distance `ADOPTION_DISTANCE` thresholds -- plus the shrinkage factor that makes the
//! two disagree, which is the mechanism stranding the fragments in the first place. The second
//! sweeps candidate thresholds and says how much of what is left each one would adopt.
//!
//! The last block is the other half of the free supervision. The known-different block above
//! uses one direction of segmentation's local speaker index; two turns in one window under the
//! *same* index are the same person on exactly that same authority. Where clustering split such
//! a pair it is wrong, provably, with no embedding and no threshold involved -- which is what
//! this block measured before `agglomerate` read that direction, and 73 of 206 such pairs were
//! split. `agglomerate` now seeds its groups by `(window, local_speaker)`, so the block has
//! become a regression check on the constraint the library applies rather than a report of an
//! opportunity going unused: its "split across two" count should read 0 on every session, and
//! any other number means seeding is not doing what it claims. The pairs with a turn too short
//! to embed stay out of reach either way -- the constraint only covers turns that were embedded.
//!
//! The enrolled-reference table at the end answers the other threshold's version of that
//! question, `IDENTIFY_DISTANCE`. `transcript.json` records a similarity only for clusters
//! that matched, so the single most useful measurement -- how far one person's stored
//! reference sits from a *different* person's voice -- is invisible on every normal path
//! precisely because it was rejected. Here every pair is printed, accepted or not, so the
//! margin either side of the cut is readable rather than inferred.

use std::path::PathBuf;

use meethook_session::{
    EnrolledSpeakers, Paths, SessionId, SessionPaths, SpeakerCluster, SpeakerClusters,
};
use meethook_transcribe::{
    ADOPTION_DISTANCE, AdoptionPopulations, CentroidPair, Clustering, EMBEDDING_MODEL,
    IDENTIFY_DISTANCE, LocalTurn, MERGE_DISTANCE, PairLabel, SEGMENTATION_MODEL,
    SPEAKER_FLOOR_SECONDS, TARGET_RATE, TrialReport, adoption_populations, identify_clusters,
    open_session, score_trials,
};

/// Pairs below which the adoption-population block reports a population as too thin to choose a
/// threshold from.
///
/// Not a statistical criterion and not pretending to be one. It is a count at which one pair
/// moving takes the answer with it: the largest cut that misattributes nobody *is* the minimum of
/// the different-speaker side, so its uncertainty is the uncertainty of a single observation
/// however many others sit behind it, and at a couple of dozen pairs there is no reason to think
/// the closest pair observed is near the closest pair possible. A report that printed a
/// separation point from twenty pairs without saying that would be inviting a constant to be read
/// off it.
const THIN_POPULATION: usize = 100;

fn main() {
    let usage =
        "usage: cluster-speaker-track <session-dir | wav-file> [--write] [--floor <s>] [--cut <d>]";
    let mut target: Option<PathBuf> = None;
    let mut write = false;
    // The floor the shipped pass partitions on, so that this report describes the clustering
    // that ships rather than a neighbouring one. `SPEAKER_FLOOR_SECONDS` carries the evidence
    // the value came from -- the 12.3-47.0 s band of floors that give the same partition on
    // session `20260810-093047`. `--floor` stays, because re-measuring that band on a recording
    // whose gap sits elsewhere is exactly what it is for.
    let mut floor = SPEAKER_FLOOR_SECONDS;
    // The cut the adoption trial list's false-accept and false-reject counts are taken at.
    // `ADOPTION_DISTANCE` because it is the constant the pass this section measures actually
    // thresholds, over exactly this quantity -- a small group's centroid against a speaker's --
    // and not `MERGE_DISTANCE`, which thresholds average linkage and would be a number printed
    // beside a decision it does not govern. `--cut` is there so that asking "what would 0.30
    // have done" costs a re-score rather than a re-run of the embedding.
    let mut cut = ADOPTION_DISTANCE;

    // Scanned rather than taken positionally, because `--floor` carries a value and the old
    // `args.any(|a| a == "--write")` consumed the whole remaining iterator to find its flag.
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.to_str().unwrap_or_default() {
            "--write" => write = true,
            "--floor" => {
                floor = rest
                    .next()
                    .and_then(|value| value.to_str())
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_else(|| fail("--floor takes a number of seconds"));
            }
            "--cut" => {
                cut = rest
                    .next()
                    .and_then(|value| value.to_str())
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_else(|| fail("--cut takes a cosine distance"));
            }
            flag if flag.starts_with("--") => fail(&format!("unknown flag {flag}\n{usage}")),
            _ if target.is_none() => target = Some(PathBuf::from(arg)),
            _ => fail(&format!("only one target is accepted\n{usage}")),
        }
    }
    let Some(target) = target else {
        eprintln!("{usage}");
        std::process::exit(2);
    };

    let session_dir = target.is_dir().then(|| target.clone());
    let track = match &session_dir {
        Some(dir) => dir.join("speaker.wav"),
        None => target,
    };
    let audio =
        meethook_transcribe::read_track_16k_mono(&track).unwrap_or_else(|e| fail(&format!("{e}")));

    let mut segmenter = load(SEGMENTATION_MODEL.file_name);
    let mut embedder = load(EMBEDDING_MODEL.file_name);

    let started = std::time::Instant::now();
    let turns = meethook_transcribe::segment_speaker_track(&audio, &mut segmenter)
        .unwrap_or_else(|e| fail(&format!("{e}")));
    let clustering = meethook_transcribe::cluster_speaker_turns(&audio, &turns, &mut embedder)
        .unwrap_or_else(|e| fail(&format!("{e}")));
    let elapsed = started.elapsed();

    let seconds = audio.len() as f64 / TARGET_RATE as f64;
    println!(
        "{}: {seconds:.1} s of audio, {} turns, {} speakers, {} turns too short to embed, \
         diarized in {:.1} s",
        track.display(),
        turns.len(),
        clustering.clusters.len(),
        clustering.skipped(),
        elapsed.as_secs_f64()
    );

    // Turn indices per cluster, positional against `turns`. Computed once because every
    // block below groups the same way, and two loops that each rebuilt this grouping could
    // disagree about it.
    let members: Vec<Vec<usize>> = clustering
        .clusters
        .iter()
        .map(|cluster| {
            clustering
                .assignment
                .iter()
                .enumerate()
                .filter(|(_, assigned)| **assigned == Some(cluster.id))
                .map(|(index, _)| index)
                .collect()
        })
        .collect();
    let span = |turn: usize| turn_span(&turns, turn);

    for (index, cluster) in clustering.clusters.iter().enumerate() {
        let mine = &members[index];
        let spoken: Vec<String> = mine.iter().map(|&turn| span(turn)).collect();
        println!(
            "\nspeaker {}: {:.1} s of speech over {} turns",
            cluster.id,
            cluster.speech_seconds,
            spoken.len()
        );
        println!("  turns: {}", spoken.join(" "));

        // How close this speaker came to being split in two. The max is the number that
        // matters: `MERGE_DISTANCE` has headroom only insofar as it sits above this.
        match spread(pairs_within(mine, &clustering.turn_embeddings)) {
            Some(within) => {
                println!(
                    "  within speaker {}: {} pairs  min {:.3}  median {:.3}  max {:.3}",
                    cluster.id, within.count, within.min, within.median, within.max
                );
                let (a, b) = within.furthest;
                println!(
                    "    furthest apart: {} vs {}  ({:.3})",
                    span(a),
                    span(b),
                    within.max
                );
            }
            None => println!(
                "  within speaker {}: {} embedded turn, no pairs",
                cluster.id,
                mine.len()
            ),
        }

        for clip in &cluster.representatives {
            println!(
                "  play:  {:>8.2} -> {:>8.2}  ({:.2} s)",
                clip.start,
                clip.end,
                clip.seconds()
            );
        }
    }

    if clustering.clusters.len() > 1 {
        println!("\ncosine distance between speakers:");
        print!("     ");
        for cluster in &clustering.clusters {
            print!("{:>7}", cluster.id);
        }
        println!();
        for row in &clustering.clusters {
            print!("{:>5}", row.id);
            for column in &clustering.clusters {
                let cosine: f32 = row
                    .embedding
                    .iter()
                    .zip(&column.embedding)
                    .map(|(a, b)| a * b)
                    .sum();
                print!("{:>7.3}", 1.0 - cosine);
            }
            println!();
        }
    }

    // Deliberately adjacent to the matrix above: the comparison between a mean-to-mean
    // distance and the closest approach of the two clouds behind those means is the whole
    // reason both are printed.
    println!("\nturn-to-turn distance between speakers:");
    if clustering.clusters.len() < 2 {
        // Both degenerate shapes say so out loud rather than rendering as zero lines, which
        // would be indistinguishable from the block being broken.
        println!(
            "  {} in this recording; no cluster pairs to compare",
            match clustering.clusters.len() {
                0 => "no speakers",
                _ => "only one speaker",
            }
        );
    }
    for left in 0..clustering.clusters.len() {
        for right in 0..left {
            let cross = pairs_between(&members[left], &members[right], &clustering.turn_embeddings);
            let Some(cross) = spread(cross) else {
                println!(
                    "  {} vs {}:  no pairs (one of them has no embedded turns)",
                    clustering.clusters[right].id, clustering.clusters[left].id
                );
                continue;
            };
            let (a, b) = cross.closest;
            println!(
                "  {} vs {}:  {} pairs  min {:.3}  median {:.3}   closest: {} vs {}",
                clustering.clusters[right].id,
                clustering.clusters[left].id,
                cross.count,
                cross.min,
                cross.median,
                span(a),
                span(b)
            );
        }
    }

    // The one block no threshold stands behind. Segmentation heard these two turns at once
    // under different local speaker indices, so they are different people whatever the
    // clustering decided -- and `agglomerate` refuses to merge such a pair for that reason,
    // meaning these distances are the only different-speaker evidence here that is not
    // conditional on a grouping a reader has to trust.
    println!("\nknown-different speakers (heard in one window, different local speakers):");
    match spread(pairs_heard_at_once(&turns, &clustering.turn_embeddings)) {
        Some(known) => {
            let (a, b) = known.closest;
            println!(
                "  {} pairs  min {:.3}  median {:.3}  max {:.3}   closest: {} vs {}",
                known.count,
                known.min,
                known.median,
                known.max,
                span(a),
                span(b)
            );
        }
        None => println!("  none: no two speakers were ever heard in the same window"),
    }

    // Built once and read by both of the blocks below, so the grid one prints and the population
    // the other scores are the same numbers rather than two computations that could disagree.
    let populations = adoption_populations(&turns, &clustering, floor);

    print_stranded_clusters(&clustering, &members, &populations, floor);
    print_must_link_splits(&clustering, &turns, &members);
    print_adoption_populations(&clustering, &turns, &populations, floor, cut);
    print_enrolled_distances(&clustering.clusters);

    if write {
        let Some(dir) = session_dir else {
            fail("--write needs a session directory, not a bare wav file");
        };
        let id = dir
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| SessionId::parse(n).ok())
            .unwrap_or_else(|| fail("that directory is not named like a session"));
        let paths = SessionPaths::new(&dir);
        SpeakerClusters::new(id, clustering.clusters)
            .write(&paths)
            .unwrap_or_else(|e| fail(&format!("{e}")));
        println!("\nwrote {}", paths.speaker_clusters_json().display());
    }
}

/// The shape of a set of turn-to-turn distances, and which pair sits at each end.
struct Spread {
    count: usize,
    min: f32,
    median: f32,
    max: f32,
    /// Turn indices of the pair at `min`, and of the pair at `max`. Printed as timings so
    /// the reader can play them: a distribution nobody can check against the audio is a
    /// number to believe rather than evidence.
    closest: (usize, usize),
    furthest: (usize, usize),
}

/// Summarizes distances between turn pairs, or `None` when there are no pairs at all.
///
/// The arithmetic -- and with it the conventions that matter, that the median of an even count
/// is the mean of the two middle values and that an empty population is `None` rather than
/// zeroes or NaN -- belongs to [`meethook_transcribe::Spread`], which states and tests them in
/// one place. `None` here still means what it always did: a cluster holding one turn has no
/// within-cluster distances, callers print "no pairs", and a fabricated 0.000 would read as two
/// identical turns.
///
/// What stays local is the pair of turn indices at each end, which a summary of bare numbers
/// cannot carry and should not try to: they are what turns a distance into a clip somebody can
/// play.
fn spread(pairs: Vec<(usize, usize, f32)>) -> Option<Spread> {
    let mut pairs = pairs;
    pairs.sort_by(|a, b| a.2.total_cmp(&b.2));

    let distances: Vec<f32> = pairs.iter().map(|pair| pair.2).collect();
    let summary = meethook_transcribe::Spread::of(&distances)?;

    let (first, last) = (pairs[0], pairs[pairs.len() - 1]);
    Some(Spread {
        count: summary.count,
        min: summary.min,
        median: summary.median,
        max: summary.max,
        closest: (first.0, first.1),
        furthest: (last.0, last.1),
    })
}

/// Cosine distance between two unit-length turn embeddings.
///
/// Raw, unlike the distance clustering merges on, which substitutes infinity for a pair
/// segmentation heard at once. Within a cluster the difference cannot arise -- an infinite
/// pair makes its groups' average infinite, so no cluster can hold one -- but between
/// clusters and in the known-different block it is the entire point: infinity there would
/// erase precisely the closest approaches being measured.
fn distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

/// Every unordered pair of turns within one cluster, with its distance.
///
/// Turns too short to embed carry no vector and take part in nothing; they are already
/// reported as the "turns too short to embed" count on the first line.
fn pairs_within(members: &[usize], embeddings: &[Option<Vec<f32>>]) -> Vec<(usize, usize, f32)> {
    let mut pairs = Vec::new();
    for (nth, &i) in members.iter().enumerate() {
        for &j in &members[nth + 1..] {
            if let (Some(a), Some(b)) = (&embeddings[i], &embeddings[j]) {
                pairs.push((i, j, distance(a, b)));
            }
        }
    }
    pairs
}

/// Every turn of one cluster against every turn of another.
fn pairs_between(
    left: &[usize],
    right: &[usize],
    embeddings: &[Option<Vec<f32>>],
) -> Vec<(usize, usize, f32)> {
    let mut pairs = Vec::new();
    for &i in left {
        for &j in right {
            if let (Some(a), Some(b)) = (&embeddings[i], &embeddings[j]) {
                pairs.push((i, j, distance(a, b)));
            }
        }
    }
    pairs
}

/// Pairs of turns segmentation heard in one window under different local speaker indices.
///
/// Different people by construction: the model was asked who was speaking during a single
/// ten-second window and answered with two of them. Nothing about voice embeddings or merge
/// thresholds is assumed, which is what makes these distances worth more than the rest.
fn pairs_heard_at_once(
    turns: &[LocalTurn],
    embeddings: &[Option<Vec<f32>>],
) -> Vec<(usize, usize, f32)> {
    let mut pairs = Vec::new();
    for i in 0..turns.len() {
        for j in 0..i {
            if !known_different(turns, i, j) {
                continue;
            }
            if let (Some(a), Some(b)) = (&embeddings[i], &embeddings[j]) {
                pairs.push((i, j, distance(a, b)));
            }
        }
    }
    pairs
}

/// Every cluster still below the talk-time floor after `adopt_below_floor` has run, its distance
/// to every cluster above it, and what a further sweep of those distances would adopt.
///
/// Two blocks in one function because they share the same scaffolding -- which clusters are
/// below the floor, which are above, each one's embeddings, and whether the constraint forbids
/// the pair -- and a second function that rebuilt any of that could sweep a different population
/// from the one it printed.
///
/// Three distance columns, because the two criteria are two numbers and the third says why.
/// `linkage` is the average of the cross-pair cosine distances, which is what `agglomerate`
/// compared against [`MERGE_DISTANCE`] and declined. `centroid` is the distance between the two
/// clusters' reference vectors, which is what [`ADOPTION_DISTANCE`] thresholds here and what
/// [`IDENTIFY_DISTANCE`] thresholds against an enrolled reference. `shrinkage` is the factor
/// between them,
/// and it is the mechanism of the bug rather than a curiosity: it is at most 1 and falls as a
/// group grows and spreads, so a fragment is charged for the spread of whatever group it is
/// offered to, and the cluster most likely to own it resists it hardest.
///
/// `blocked` is a separate column and not folded into the distances. A merge the same-window
/// constraint forbids cannot be adopted however close the two look, and a merge that is merely
/// far might be adoptable under a threshold nobody has chosen yet. One number could not tell a
/// reader which of those it was looking at.
///
/// Every distance printed here is read off [`AdoptionPopulations::offers`] rather than measured
/// again. The grid is the same one [`print_adoption_populations`] scores, and two computations of
/// it could quietly diverge -- which for a report whose whole job is to be believed instead of a
/// threshold picked by ear would be the one failure that matters.
fn print_stranded_clusters(
    clustering: &Clustering,
    members: &[Vec<usize>],
    populations: &AdoptionPopulations,
    floor: f64,
) {
    let clusters = &clustering.clusters;
    // Folded from 0.0 rather than `sum()`, whose identity for floats is `-0.0`: an empty row of
    // the sweep below would otherwise report "-0.0 s" adopted, which reads as a bug in the
    // instrument and costs the reader more attention than the whole line is worth.
    let seconds = |group: &[u32]| -> f64 {
        group
            .iter()
            .fold(0.0, |total, &c| total + clusters[c as usize].speech_seconds)
    };
    let below = &populations.below;
    let above = &populations.above;

    println!(
        "\nstranded clusters, and where each would go (floor {floor:.1} s of speech). These are \
         what the shipped adoption pass DECLINED -- it has already run on this clustering, so \
         every row below is residue rather than the material its constant was chosen from:"
    );

    // Every degenerate shape says so in a sentence. A block that renders as no lines is
    // indistinguishable from a block that is broken, and two of these -- a recording with one
    // speaker, a recording with no speech on the speaker track -- are ordinary rather than odd.
    if clusters.is_empty() {
        println!("  no speakers in this recording; nothing is stranded");
        return;
    }
    if below.is_empty() {
        println!(
            "  nothing is stranded: every cluster ({} of them) holds at least {floor:.1} s",
            clusters.len()
        );
        return;
    }
    if above.is_empty() {
        println!(
            "  every cluster ({} of them) is under {floor:.1} s, so there is nothing to adopt \
             into -- lower --floor, or accept that this track has no dominant speaker",
            clusters.len()
        );
        return;
    }

    println!(
        "  {} of {} clusters are below the floor, holding {:.1} s of the {:.1} s that got \
         clustered; {} are above it, holding {:.1} s",
        below.len(),
        clusters.len(),
        seconds(below),
        seconds(below) + seconds(above),
        above.len(),
        seconds(above)
    );
    println!(
        "  linkage is what agglomerate merged on and declined, against its cut of \
         {MERGE_DISTANCE:.3}; centroid is what ADOPTION_DISTANCE ({ADOPTION_DISTANCE:.3}) \
         thresholds; shrinkage is the \
         factor between them, 1 - linkage = shrinkage * (1 - centroid); blocked means the \
         same-window constraint forbids this merge whatever the distances say"
    );

    if populations.declined.offers_without_distance > 0 {
        println!(
            "  {} below/above pair(s) have no distance at all -- one side has no embedded turns \
             -- and are missing from every row below",
            populations.declined.offers_without_distance
        );
    }

    // Where each fragment would go, kept for the sweep below so it thresholds the same numbers
    // it just printed. `None` is a fragment every above-floor cluster is barred from adopting.
    let mut nearest: Vec<(u32, Option<f32>)> = Vec::with_capacity(below.len());

    for &small in below {
        println!(
            "\n  cluster {} ({:.1} s, {}, first at {:.1})",
            clusters[small as usize].id,
            clusters[small as usize].speech_seconds,
            plural(members[small as usize].len(), "turn"),
            clusters[small as usize].first_spoke_seconds
        );

        let mut best: Option<(f32, u32, f32)> = None;
        for offer in offers_from(populations, small) {
            let blocked = is_blocked(offer);
            println!(
                "     -> {:<4} linkage {:.3}   centroid {:.3}   shrinkage {:.3}   {}",
                offer.large.cluster,
                offer.distance.average_linkage,
                offer.distance.centroid,
                offer.distance.shrinkage,
                if blocked { "blocked" } else { "-" }
            );
            if !blocked && best.is_none_or(|(closest, _, _)| offer.distance.centroid < closest) {
                best = Some((
                    offer.distance.centroid,
                    offer.large.cluster,
                    offer.distance.average_linkage,
                ));
            }
        }

        match best {
            Some((centroid, id, linkage)) => println!(
                "     nearest permitted: {id} at centroid {centroid:.3} (linkage {linkage:.3})"
            ),
            None => {
                println!("     nearest permitted: none -- every cluster above the floor is blocked")
            }
        }
        nearest.push((small, best.map(|(centroid, _, _)| centroid)));
    }

    // The sweep. Centroid distance, because that is what `ADOPTION_DISTANCE` thresholds and
    // saying so is the only thing keeping the two criteria from being confused for each other.
    // Argmax among permitted targets and then the cut, matching `adopt_below_floor`; centroids
    // frozen, because that pass adopts in one pass and does not re-centroid as it goes. An
    // iterative pass would adopt more and would need its own sweep.
    //
    // Read as a *further* sweep: the shipped pass has already taken everything under its own cut,
    // so the row at that cut adopts nothing and the rows above it price widening it.
    println!(
        "\n  further adoption sweep over centroid distance, one pass, argmax among permitted \
         targets then the cut. The shipped pass has already run at {ADOPTION_DISTANCE:.3}, so \
         this prices widening it rather than choosing it:"
    );
    println!("    threshold   adopted              remaining            clusters after");
    for step in 4..=16 {
        let threshold = step as f32 * 0.05;
        let adopted: Vec<u32> = nearest
            .iter()
            .filter(|(_, centroid)| centroid.is_some_and(|centroid| centroid < threshold))
            .map(|&(cluster, _)| cluster)
            .collect();
        let remaining: Vec<u32> = below
            .iter()
            .copied()
            .filter(|cluster| !adopted.contains(cluster))
            .collect();
        println!(
            "    {threshold:>9.3}   {:>3} ({:>6.1} s)      {:>3} ({:>6.1} s)      {:>3}",
            adopted.len(),
            seconds(&adopted),
            remaining.len(),
            seconds(&remaining),
            above.len() + remaining.len()
        );
    }

    let unadoptable: Vec<u32> = nearest
        .iter()
        .filter(|(_, centroid)| centroid.is_none())
        .map(|&(cluster, _)| cluster)
        .collect();
    println!(
        "  no permitted target at any threshold: {} ({:.1} s)",
        plural(unadoptable.len(), "cluster"),
        seconds(&unadoptable)
    );

    // The ceiling on any centroid threshold, and the reason the sweep is printed with one.
    // Two above-floor clusters are two speakers a person confirmed; a cut wider than the gap
    // between the closest of them is measuring a distance two different people fit inside.
    match populations.ceiling() {
        Some((centroid, left, right)) => println!(
            "  the closest two clusters above the floor ({left} and {right}) sit {centroid:.3} \
             apart by centroid; a cut at or above that is measuring a gap two separate speakers \
             fit inside"
        ),
        None => {
            println!("  only one cluster above the floor, so there is no gap to compare against")
        }
    }
}

/// One below-floor cluster's row of the offer grid, in the grid's own order.
fn offers_from(
    populations: &AdoptionPopulations,
    small: u32,
) -> impl Iterator<Item = &CentroidPair> {
    populations
        .offers
        .iter()
        .filter(move |offer| offer.small.cluster == small)
}

/// Whether the same-window constraint bars this offer, which is also what labels it a
/// different-speaker pair. One question, so one spelling of it.
fn is_blocked(offer: &CentroidPair) -> bool {
    matches!(offer.label, PairLabel::CannotLink { .. })
}

/// The two segmentation-labelled populations of the distance [`ADOPTION_DISTANCE`] thresholds,
/// scored, with the pairs behind every number.
///
/// Measured on the clustering that ships, which is the one `adopt_below_floor` has already swept:
/// the positives are now classes inside a cluster the pass may have added fragments to, and the
/// offer grid holds only what it declined. So this verifies the constant, and the numbers it was
/// *chosen* from are the ones recorded on TASK-018.02.02 from before the pass existed.
///
/// The blocks above print the raw material and never join it: every below-floor cluster against
/// every above-floor one with a `blocked` column, and every must-link pair with its cluster ids.
/// This is the join -- **two labelled populations of one quantity**, which is what a threshold can
/// be chosen from and a table read by eye cannot be.
///
/// The quantity is centroid distance, a small group's normalized mean against a larger group's,
/// and the section says so on every line that prints a number. `MERGE_DISTANCE` thresholds the
/// other quantity, average linkage; [`ADOPTION_DISTANCE`] thresholds this one, for this decision,
/// and [`IDENTIFY_DISTANCE`] thresholds it for another. The first two were once the
/// same number, and TASK-020 is the bug that cost -- one confirmed pair of speakers landing on
/// opposite sides of the two criteria at a shared 0.45 -- which is why the insistence is worth
/// the words even now that the values differ.
///
/// Nothing here decides anything. The construction and its exclusions live in
/// `meethook_transcribe::adoption_populations`, where they are unit-tested, and the scoring lives
/// in [`score_trials`], which states and tests the boundary convention -- accept is *strictly*
/// below the cut -- so that this report cannot state a different one than the decision it informs
/// would use. What is local is the presentation, deliberately following
/// `examples/speaker-trials.rs` word for word wherever the two print the same arithmetic.
fn print_adoption_populations(
    clustering: &Clustering,
    turns: &[LocalTurn],
    populations: &AdoptionPopulations,
    floor: f64,
    cut: f32,
) {
    let clusters = &clustering.clusters;
    let declined = &populations.declined;
    let negatives: Vec<&CentroidPair> = populations.negatives().collect();
    let ceiling = populations.ceiling();

    println!("\nadoption populations: the two labelled populations of one distance");
    println!(
        "  the quantity, everywhere in this section: CENTROID distance -- a small group's \
         normalized mean against a larger group's, cosine, 0 for the same direction. Not \
         turn-to-turn, and not the average linkage agglomerate merged on. MERGE_DISTANCE \
         ({MERGE_DISTANCE:.3}) governs that other number and governs nothing below; linkage and \
         shrinkage are printed beside each pair so that neither can be read as the other."
    );

    if populations.positives.is_empty() && negatives.is_empty() {
        println!(
            "  neither population exists on this recording, so there is nothing here to choose a \
             threshold from. Both need clusters on either side of the floor and segmentation \
             judgements joining them; a track with one speaker has neither."
        );
        return;
    }

    // ------------------------------------------------------------------------------------
    // What was built, and what was left out of it
    // ------------------------------------------------------------------------------------
    println!("\n  construction:");
    println!(
        "    positives, labelled same-speaker by segmentation alone: leave-one-class-out. A \
         must-link class -- two or more embedded turns sharing one (window, local speaker), which \
         segmentation heard as one person -- that landed wholly inside a cluster above the \
         {floor:.1} s floor, measured against the REST of that cluster. The class is excluded \
         from the residual; left in, it would bias the mean it is being compared to."
    );
    println!(
        "    the weak leg of that label, and it belongs beside every number derived from it: \
         segmentation says the class's own turns are one person. That the class and the residual \
         are one person is CLUSTERING's claim -- seeding forced the class together, and the merge \
         loop is what put it in that cluster. The within-class cross-check at the end of this \
         section is the population with no such leg."
    );
    println!(
        "    negatives, labelled different-speaker by the same-window cannot-link constraint: a \
         below-floor cluster the constraint bars from an above-floor cluster, because \
         segmentation heard one turn of each at once under different local speaker indices. The \
         witness turn pair is printed with each one so it can be played rather than believed."
    );
    println!(
        "    the caveat that belongs beside any cut derived from the negatives: every one of them \
         is a pair the constraint ALREADY refuses, so a cut read off them prices a distance-only \
         rule the pass does not use. What they do measure is how close two provably different \
         people's centroids come at this granularity -- which is what bounds trust in a cut on \
         the unblocked pairs, where no constraint protects anybody."
    );
    println!(
        "    excluded from the positives, counted rather than dropped, of {} (window, local \
         speaker) classes in all: {} of a single embedded turn (one turn is no pair, so no \
         must-link assertion at all -- which cluster it joined is clustering's decision), {} \
         inside a below-floor cluster (no speaker-scale residual to measure against), {} that are \
         their whole cluster (no residual left), {} split across clusters (seeding makes that \
         impossible, so this must read 0), {} with no direction. {} positives.",
        populations.classes,
        declined.single_turn,
        declined.below_floor,
        declined.whole_cluster,
        declined.split_across_clusters,
        declined.no_direction,
        populations.positives.len()
    );

    // ------------------------------------------------------------------------------------
    // The two populations
    // ------------------------------------------------------------------------------------
    let positive_distances = centroids(populations.positives.iter());
    let negative_distances = centroids(negatives.iter().copied());

    println!("\n  centroid distance, the two labelled populations:");
    print_centroid_spread(
        "same speaker      (leave-one-class-out)",
        &positive_distances,
    );
    print_centroid_spread(
        "different speakers (cannot-link)       ",
        &negative_distances,
    );
    println!(
        "    {} below-floor/above-floor offers in all, {} of them blocked and so labelled; the \
         other {} carry no label in either direction and are scored as neither",
        populations.offers.len(),
        negatives.len(),
        populations.offers.len() - negatives.len()
    );

    println!("\n  the same-speaker pairs, each class against the rest of its own cluster:");
    for pair in &populations.positives {
        let (window, local_speaker) = pair
            .label
            .class()
            .expect("a positive is labelled by its must-link class");
        println!(
            "    cluster {:<3} window {:>4} local speaker {}   centroid {:.3}   (linkage {:.3}, \
             shrinkage {:.3})   class {:.1} s over {} vs residual {:.1} s over {}",
            pair.large.cluster,
            window,
            local_speaker,
            pair.distance.centroid,
            pair.distance.average_linkage,
            pair.distance.shrinkage,
            pair.small.seconds,
            plural(pair.small.turns.len(), "turn"),
            pair.large.seconds,
            plural(pair.large.turns.len(), "turn")
        );
        println!("        class turns: {}", spans(turns, &pair.small.turns));
    }

    println!("\n  the different-speaker pairs, each fragment against a speaker it is barred from:");
    for pair in &negatives {
        let PairLabel::CannotLink { witness } = pair.label else {
            continue;
        };
        let (earlier, later) = witness;
        println!(
            "    {:>3} vs {:<3} centroid {:.3}   (linkage {:.3}, shrinkage {:.3})   fragment \
             {:.1} s over {}, {}",
            pair.small.cluster,
            pair.large.cluster,
            pair.distance.centroid,
            pair.distance.average_linkage,
            pair.distance.shrinkage,
            pair.small.seconds,
            plural(pair.small.turns.len(), "turn"),
            purity(pair.small.classes)
        );
        println!(
            "        heard at once in window {}: {} (local speaker {}) / {} (local speaker {});  \
             fragment turns: {}",
            turns[earlier].window,
            turn_span(turns, earlier),
            turns[earlier].local_speaker,
            turn_span(turns, later),
            turns[later].local_speaker,
            spans(turns, &pair.small.turns)
        );
    }

    // ------------------------------------------------------------------------------------
    // Scored, through the one place that states the conventions
    // ------------------------------------------------------------------------------------
    let trials = populations.trials();
    let report = score_trials(&trials, cut);

    println!(
        "\n  scored through score_trials, which states and tests the conventions: accept is \
         STRICTLY below the cut, a false accept is a different-speaker pair below it and a false \
         reject a same-speaker pair at or above it, percentiles are nearest-rank."
    );
    println!(
        "    at the cut {:.3} -- ADOPTION_DISTANCE unless --cut moved it, so these two counts \
         price the shipped pass on the populations it was chosen from. Read them with the \
         negatives' caveat above: every different-speaker pair here is one the constraint \
         already refuses and the pass never offers, so a false accept in this column is a cut a \
         DISTANCE-ONLY rule would have got wrong, not one this pass does:",
        report.threshold
    );
    print_costs("      ", &report);

    println!("\n  separation:");
    match report.overlap {
        Some((min_different, max_same)) => println!(
            "    NO SINGLE THRESHOLD SEPARATES THESE TWO POPULATIONS. They overlap between \
             {min_different:.3} (the closest different-speaker pair) and {max_same:.3} (the \
             furthest-apart same-speaker pair); every cut inside that band trades one kind of \
             mistake for the other."
        ),
        None => match (report.same, report.different) {
            (Some(same), Some(different)) => println!(
                "    the two populations do not overlap: every same-speaker pair is below {:.3} \
                 and every different-speaker pair is at or above {:.3}, so any cut in between \
                 makes no mistakes on this list",
                same.max, different.min
            ),
            _ => println!("    not measurable: one side of the trial list is empty"),
        },
    }
    match report.equal_error {
        Some(equal_error) => println!(
            "    equal error rate {:.1}% at a cut of {:.3}  (the mean of the two rates where they \
             come closest to crossing)",
            equal_error.rate * 100.0,
            equal_error.threshold
        ),
        None => println!("    equal error rate: not measurable, one side of the list is empty"),
    }
    match report.zero_false_accept {
        Some(zero) => {
            println!(
                "    the largest cut that misattributes nobody is {:.3}, and it rejects {:.1}% of \
                 same-speaker pairs",
                zero.threshold,
                zero.false_reject_rate * 100.0
            );
            // Priced as well as named. The asymmetry argument the adoption constant will rest on
            // -- a silent misattribution is expensive and a visible extra Unknown N is cheap --
            // is only an argument until somebody says what the cheap error costs in pairs.
            let priced = score_trials(&trials, zero.threshold);
            println!("    what refusing to misattribute anybody costs, at that cut:");
            print_costs("      ", &priced);
        }
        None => println!("    no misattribution-free cut is measurable from this list"),
    }

    // The thinness statement, immediately under the cut it is about, because the cut is the
    // minimum of one population and therefore rests on one pair however many are behind it.
    let thin = populations.positives.len() < THIN_POPULATION || negatives.len() < THIN_POPULATION;
    if thin {
        println!(
            "\n    TOO THIN TO CHOOSE A CONSTANT FROM: {} same-speaker and {} different-speaker \
             pairs, against the {THIN_POPULATION} this report treats as the point where one pair \
             stops moving the answer. The misattribution-free cut IS the minimum of the \
             different-speaker side, so it is a single observation: it moves wherever that one \
             pair moves, and nothing here says the closest pair seen is near the closest pair \
             possible. Read it as a bound to check by ear, not as a number to ship.",
            populations.positives.len(),
            negatives.len()
        );
    }
    if !negative_distances.is_empty() {
        println!("    the three closest different-speaker pairs, which are the ones it rests on:");
        let mut closest: Vec<&&CentroidPair> = negatives.iter().collect();
        closest.sort_by(|a, b| a.distance.centroid.total_cmp(&b.distance.centroid));
        for pair in closest.iter().take(3) {
            println!(
                "      {:>3} vs {:<3} centroid {:.3}   fragment {:.1} s over {}, {}   {}",
                pair.small.cluster,
                pair.large.cluster,
                pair.distance.centroid,
                pair.small.seconds,
                plural(pair.small.turns.len(), "turn"),
                purity(pair.small.classes),
                spans(turns, &pair.small.turns)
            );
        }
    }

    // ------------------------------------------------------------------------------------
    // The pairs a cut would be deciding, whether or not any label covers them
    // ------------------------------------------------------------------------------------
    print_contested_targets(clusters, turns, populations, ceiling);

    // ------------------------------------------------------------------------------------
    // The floor, which decides what counts as a speaker before any distance is looked at
    // ------------------------------------------------------------------------------------
    println!("\n  the talk-time floor, and the band of floors giving this same partition:");
    println!(
        "    the convention is speech_seconds < floor is a fragment and >= floor is a speaker, so \
         a cluster sitting exactly ON the floor is a speaker"
    );
    for cluster in clusters {
        println!(
            "    cluster {:<3} {:>7.1} s   {}",
            cluster.id,
            cluster.speech_seconds,
            if populations.above.contains(&cluster.id) {
                "speaker"
            } else {
                "fragment"
            }
        );
    }
    match populations.floor_band {
        Some((max_below, min_above)) => {
            println!(
                "    any floor f with {max_below:.1} < f <= {min_above:.1} produces exactly this \
                 partition -- {} speakers holding the talk time and {} fragments -- so the \
                 --floor {floor:.1} this ran at is insensitive across a {:.1} s band rather than \
                 tuned to the answer",
                populations.above.len(),
                populations.below.len(),
                min_above - max_below
            );
            println!(
                "    both edges are consequences, not trivia: at f <= {max_below:.1} the largest \
                 fragment becomes an adoption TARGET on {max_below:.1} s of speech, which is the \
                 failure the floor exists to prevent; above {min_above:.1} the smallest speaker \
                 stops being one and its {min_above:.1} s go looking for an owner"
            );
        }
        None => println!(
            "    no band: one side of the floor is empty on this recording, so there is no \
             partition for a range of floors to preserve"
        ),
    }

    // ------------------------------------------------------------------------------------
    // The two populations that are the right label or the right scale, but never both
    // ------------------------------------------------------------------------------------
    println!("\n  cross-check -- purest label, WRONG SHAPE: leave-one-TURN-out inside a class.");
    println!(
        "    One embedded turn of a must-link class against the rest of that class, in any \
         cluster, above the floor or below it. Nothing but segmentation stands behind the label, \
         which is what the leave-one-class-out positives cannot say. Its shape is wrong: a class \
         fits inside one ten-second window, so neither side is a speaker-scale mean and the \
         shrinkage differs. It says whether the positives LOOK like a same-speaker population. It \
         is not a substitute for them and is not in the trial list above."
    );
    print_centroid_spread(
        "within one class  (turn vs rest of class)",
        &centroids(populations.within_class.iter()),
    );

    println!("\n  auxiliary negatives -- right scale, label from a person rather than the model:");
    println!(
        "    every pair of clusters above the floor. Large against large is a third shape and \
         folding it into the trial list would flatter the separation, so it is scored on its own. \
         What it licenses is the ceiling: two clusters above the floor are two people, so a cut at \
         or above the gap between the closest of them is measuring a distance two speakers fit \
         inside."
    );
    print_centroid_spread(
        "speaker vs speaker (above the floor)   ",
        &centroids(populations.above_floor.iter()),
    );
    for pair in &populations.above_floor {
        println!(
            "      {:>3} vs {:<3} centroid {:.3}   (linkage {:.3}, shrinkage {:.3})   {:.1} s vs \
             {:.1} s",
            pair.small.cluster,
            pair.large.cluster,
            pair.distance.centroid,
            pair.distance.average_linkage,
            pair.distance.shrinkage,
            pair.small.seconds,
            pair.large.seconds
        );
    }
    match ceiling {
        Some((centroid, left, right)) => {
            println!("    the ceiling is {centroid:.3}, clusters {left} and {right}")
        }
        None => println!(
            "    no ceiling: fewer than two clusters above the floor, so nothing here bounds a cut"
        ),
    }
}

/// Every below-floor cluster a cut is actually deciding about: those closer to some speaker than
/// the two closest speakers are to each other.
///
/// The selection rule is structural rather than a margin heuristic, and rather than a hardcoded
/// cluster id, so it stays true on a session where the ids are different. Under the ceiling means
/// the fragment sits nearer a speaker than two known-different speakers sit to each other, so no
/// cut can take it without also being wide enough to have merged those two -- which is exactly
/// the set of pairs a threshold choice is a choice about.
///
/// On session `20260810-093047` cluster 12 lands here at 0.296 under column 1 with 0.550 as its
/// second choice. It is not ambiguous by margin, which is why a margin rule would have missed it:
/// what makes it the pair the ceiling turns on is external knowledge -- it may be a seventh
/// speaker whose stored reference is the old blended cluster's centroid, so the enrolled distances
/// that would settle it are circular. No report can settle that. This one only has to make it
/// impossible to miss.
fn print_contested_targets(
    clusters: &[SpeakerCluster],
    turns: &[LocalTurn],
    populations: &AdoptionPopulations,
    ceiling: Option<(f32, u32, u32)>,
) {
    println!("\n  contested targets: the fragments a cut is really deciding about.");
    match ceiling {
        Some((centroid, left, right)) => println!(
            "    Rows are every below-floor cluster whose nearest PERMITTED speaker sits under the \
             ceiling of {centroid:.3} -- the gap between clusters {left} and {right}, two \
             different people -- ordered by that distance. A fragment in this table is closer to \
             some speaker than those two are to each other, so no cut adopts it without being \
             wide enough to have merged them."
        ),
        None => println!(
            "    Fewer than two clusters above the floor, so there is no ceiling to select on and \
             every fragment with a permitted target is listed."
        ),
    }
    println!(
        "    Cells are centroid distance; 'b' marks a pair the same-window constraint bars \
         whatever the distance says, and blocked pairs are not eligible to be a nearest choice. \
         'nearest' and 'second' are the two closest permitted speakers and 'margin' the gap \
         between them."
    );

    // Rows, and the two closest permitted targets for each.
    let mut rows: Vec<(u32, f32, Option<f32>, u32)> = Vec::new();
    for &small in &populations.below {
        let mut permitted: Vec<(f32, u32)> = offers_from(populations, small)
            .filter(|offer| !is_blocked(offer))
            .map(|offer| (offer.distance.centroid, offer.large.cluster))
            .collect();
        permitted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let Some(&(nearest, target)) = permitted.first() else {
            continue;
        };
        if ceiling.is_some_and(|(ceiling, _, _)| nearest >= ceiling) {
            continue;
        }
        rows.push((small, nearest, permitted.get(1).map(|&(d, _)| d), target));
    }
    rows.sort_by(|a, b| a.1.total_cmp(&b.1));

    if rows.is_empty() {
        println!(
            "    none: every fragment's nearest permitted speaker is further away than two \
             speakers are from each other, so no cut under the ceiling adopts anything"
        );
        return;
    }

    print!(
        "    {:>7}  {:>8}  {:>8}  {:>7}  {:>7}  {:>7} |",
        "cluster", "seconds", "first at", "nearest", "second", "margin"
    );
    for &large in &populations.above {
        print!("{large:>8}");
    }
    println!();
    for &(small, nearest, second, target) in &rows {
        let cluster = &clusters[small as usize];
        print!(
            "    {:>7}  {:>8.1}  {:>8.1}  {:>7.3}  {:>7}  {:>7} |",
            small,
            cluster.speech_seconds,
            cluster.first_spoke_seconds,
            nearest,
            match second {
                Some(second) => format!("{second:.3}"),
                None => "-".to_string(),
            },
            match second {
                Some(second) => format!("{:.3}", second - nearest),
                None => "-".to_string(),
            }
        );
        for &large in &populations.above {
            let cell = offers_from(populations, small)
                .find(|offer| offer.large.cluster == large)
                .map(|offer| {
                    format!(
                        "{:.3}{}",
                        offer.distance.centroid,
                        if is_blocked(offer) { "b" } else { " " }
                    )
                })
                .unwrap_or_else(|| "-  ".to_string());
            print!("{cell:>8}");
        }
        println!();
        println!(
            "        cluster {small} nearest permitted {target} at {nearest:.3}; turns: {}",
            spans(
                turns,
                offers_from(populations, small)
                    .next()
                    .map(|offer| offer.small.turns.as_slice())
                    .unwrap_or_default()
            )
        );
    }
}

/// Every pair's centroid distance, which is the only field a spread of these is ever over.
fn centroids<'a>(pairs: impl Iterator<Item = &'a CentroidPair>) -> Vec<f32> {
    pairs.map(|pair| pair.distance.centroid).collect()
}

/// A population's shape, labelled `centroid` because that is the only quantity here.
fn print_centroid_spread(label: &str, distances: &[f32]) {
    match meethook_transcribe::Spread::of(distances) {
        Some(s) => println!(
            "    {label}: {} centroid pair(s)  min {:.3}  p05 {:.3}  median {:.3}  p95 {:.3}  \
             max {:.3}  mean {:.3}",
            s.count, s.min, s.p05, s.median, s.p95, s.max, s.mean
        ),
        None => println!("    {label}: no pairs"),
    }
}

/// The two error counts at one cut, in the wording `speaker-trials` uses for the same numbers.
fn print_costs(indent: &str, report: &TrialReport) {
    println!(
        "{indent}false accepts: {} different-speaker pair(s) below the cut{}",
        report.false_accepts,
        percent(report.false_accept_rate)
    );
    println!(
        "{indent}false rejects: {} same-speaker pair(s) at or above it{}",
        report.false_rejects,
        percent(report.false_reject_rate)
    );
}

fn percent(rate: Option<f32>) -> String {
    match rate {
        Some(rate) => format!("  ({:.1}%)", rate * 100.0),
        None => "  (no such pairs, so no rate)".to_string(),
    }
}

/// Whether segmentation grouped this fragment itself, or embedding assembled it across windows.
///
/// The distinction decides what a different-speaker label on it is worth. One class means
/// segmentation heard these turns in one window under one index, so the fragment is a small group
/// of one person. Several means the clustering built it out of turns from different windows, and
/// it may be a blend belonging to nobody -- in which case its label is only as good as the
/// clustering, which is the one thing the labelled populations exist to avoid depending on.
fn purity(classes: usize) -> String {
    match classes {
        1 => "one (window, local speaker) class, so segmentation itself grouped it".to_string(),
        several => format!(
            "{several} classes across windows, assembled by embedding -- may be a blend belonging \
             to nobody"
        ),
    }
}

/// Several turns' timings on one line, so a group of numbers can be played rather than believed.
fn spans(turns: &[LocalTurn], held: &[usize]) -> String {
    held.iter()
        .map(|&turn| turn_span(turns, turn))
        .collect::<Vec<String>>()
        .join(" ")
}

/// Turn pairs segmentation heard in one window under the same local speaker index, and what
/// clustering did with them.
///
/// The must-link direction of the constraint. Different indices in one window are different
/// people; the *same* index in one window is one person, on exactly the same authority. Windows do
/// not overlap and segmentation closes and reopens a turn for one index whenever the silence
/// inside it runs past a quarter second, so a pair like this is two turns of one speaker rather
/// than an artefact of the decoder -- and wherever clustering put such a pair in two clusters, it
/// is wrong, with no embedding, no threshold and no model standing behind the claim.
///
/// `agglomerate` reads that direction now: it seeds its groups by `(window, local_speaker)`, so
/// every embedded pair this block finds is in one cluster by construction. That makes the block a
/// regression check rather than the opportunity report it was written as -- **the line to read is
/// "split across two", and it must be 0.** Anything else means seeding is broken, and no unit test
/// says so as directly, because this is the constraint measured on the audio it came from.
///
/// The other two counts still say something. "Already in one cluster" is the population the
/// seeding covers, and the pairs with a turn too short to embed are out of its reach in principle:
/// `constraints` only carries embedded turns, so lowering `MIN_EMBEDDABLE_SECONDS` is the only
/// thing that would reach them, and that is a separate judgement.
///
/// The last line -- how many clusters transitive cluster-level merging of these pairs would
/// leave -- is kept because it is not the same operation the library performs. It merges whole
/// clusters, which can drag turns from other windows along with the pair; seeding merges turns.
/// Now that the split count should be 0 it will normally have nothing to do, and a session where
/// it does is a session where the two operations disagree.
fn print_must_link_splits(clustering: &Clustering, turns: &[LocalTurn], members: &[Vec<usize>]) {
    println!("\nmust-link pairs (heard in one window under the same local speaker index):");

    let mut together = 0usize;
    let mut unembedded = 0usize;
    let mut split: Vec<(usize, usize)> = Vec::new();
    for i in 0..turns.len() {
        for j in 0..i {
            if !same_local_speaker(turns, i, j) {
                continue;
            }
            match (clustering.assignment[j], clustering.assignment[i]) {
                (Some(left), Some(right)) if left == right => together += 1,
                (Some(_), Some(_)) => split.push((j, i)),
                // No cluster on at least one side, so nothing here can sweep it either way.
                // Counted rather than dropped: it is still one speaker's speech going
                // unattributed, and the first line of this report already totals those turns.
                _ => unembedded += 1,
            }
        }
    }

    let population = together + unembedded + split.len();
    if population == 0 {
        println!("  none: no local speaker was ever heard twice in one window");
        return;
    }
    println!(
        "  {}: {together} already in one cluster, {} split across two, {unembedded} with a turn \
         too short to embed",
        plural(population, "pair"),
        split.len()
    );
    if split.is_empty() {
        println!("  clustering split none of them, so there is nothing to gain here");
        return;
    }

    let mut involved: Vec<usize> = split.iter().flat_map(|&(j, i)| [j, i]).collect();
    involved.sort_unstable();
    involved.dedup();
    let speech: f64 = involved
        .iter()
        .map(|&turn| turns[turn].end_s - turns[turn].start_s)
        .sum();
    println!(
        "  the split ones, covering {} and {speech:.1} s of speech:",
        plural(involved.len(), "distinct turn")
    );
    for &(earlier, later) in &split {
        println!(
            "    window {:>4}  local speaker {}  clusters {} and {}   {} / {}",
            turns[earlier].window,
            turns[earlier].local_speaker,
            cluster_of(clustering, earlier),
            cluster_of(clustering, later),
            turn_span(turns, earlier),
            turn_span(turns, later)
        );
    }

    let mut pairs: Vec<(usize, usize)> = split
        .iter()
        .map(|&(earlier, later)| {
            let (left, right) = (
                cluster_of(clustering, earlier) as usize,
                cluster_of(clustering, later) as usize,
            );
            (left.min(right), left.max(right))
        })
        .collect();
    pairs.sort_unstable();
    pairs.dedup();

    // Applied transitively, because merging A into B and B into C says A and C are one speaker
    // too, and with the cannot-link constraint winning any conflict: a component that has come
    // to hold two turns heard at once cannot absorb another, and segmentation is the authority
    // on both sides of that. Cluster ids index `members` -- `cluster_speaker_turns` numbers them
    // by position -- so the two are the same key throughout.
    let mut parent: Vec<usize> = (0..clustering.clusters.len()).collect();
    let (mut merges, mut vetoed) = (0usize, 0usize);
    for &(left, right) in &pairs {
        let (left, right) = (root(&mut parent, left), root(&mut parent, right));
        if left == right {
            continue;
        }
        let (mine, theirs) = (
            component_turns(&mut parent, left, members),
            component_turns(&mut parent, right, members),
        );
        if heard_apart(&mine, &theirs, turns) {
            vetoed += 1;
            continue;
        }
        parent[right] = left;
        merges += 1;
    }
    let mut left_over = 0usize;
    for cluster in 0..clustering.clusters.len() {
        if root(&mut parent, cluster) == cluster {
            left_over += 1;
        }
    }

    println!(
        "  {} implicated; applied transitively with cannot-link winning any conflict, {} applied \
         and {vetoed} refused, leaving {} of {} -- before any distance is looked at",
        plural(pairs.len(), "distinct cluster pair"),
        plural(merges, "merge"),
        plural(left_over, "cluster"),
        clustering.clusters.len()
    );
}

/// The cluster a turn was assigned, which the caller has already established it has.
fn cluster_of(clustering: &Clustering, turn: usize) -> u32 {
    clustering.assignment[turn].expect("this turn was clustered")
}

/// The representative of a cluster's merge component, flattening the path walked to reach it.
fn root(parent: &mut [usize], mut cluster: usize) -> usize {
    while parent[cluster] != cluster {
        parent[cluster] = parent[parent[cluster]];
        cluster = parent[cluster];
    }
    cluster
}

/// Every turn held by every cluster currently merged into one component.
fn component_turns(parent: &mut [usize], component: usize, members: &[Vec<usize>]) -> Vec<usize> {
    let mut held = Vec::new();
    for (cluster, mine) in members.iter().enumerate() {
        if root(parent, cluster) == component {
            held.extend_from_slice(mine);
        }
    }
    held
}

/// Whether segmentation heard these two turns at once under different local speaker indices,
/// which makes them different people whatever their embeddings look like.
fn known_different(turns: &[LocalTurn], i: usize, j: usize) -> bool {
    turns[i].window == turns[j].window && turns[i].local_speaker != turns[j].local_speaker
}

/// Whether segmentation heard these two turns in one window under the *same* local speaker
/// index, which makes them one person on exactly the same authority.
fn same_local_speaker(turns: &[LocalTurn], i: usize, j: usize) -> bool {
    turns[i].window == turns[j].window && turns[i].local_speaker == turns[j].local_speaker
}

/// Whether the cannot-link constraint forbids holding these two sets of turns in one cluster.
///
/// The same test `agglomerate` encodes as an infinity in its distance matrix, asked directly
/// rather than read off a distance: once a group holds a forbidden pair every average linkage
/// through it is infinite, so clustering never needs to ask again, but a report weighing a merge
/// that clustering did not make does.
fn heard_apart(left: &[usize], right: &[usize], turns: &[LocalTurn]) -> bool {
    left.iter()
        .any(|&i| right.iter().any(|&j| known_different(turns, i, j)))
}

/// "1 pair", "206 pairs".
///
/// Counts here run from zero to hundreds on the same line of the same run, and a report that
/// says "1 pairs" spends its reader's attention on wondering what else it is not being careful
/// about. Every verb near one of these is a participle for the same reason.
fn plural(count: usize, noun: &str) -> String {
    format!("{count} {noun}{}", if count == 1 { "" } else { "s" })
}

/// One turn's timing, so a number printed beside it can be played rather than believed.
fn turn_span(turns: &[LocalTurn], turn: usize) -> String {
    format!("{:.1}-{:.1}", turns[turn].start_s, turns[turn].end_s)
}

/// Every cluster measured against every enrolled reference, accepted or not.
///
/// This is the number TASK-014 has to calibrate `IDENTIFY_DISTANCE` on, and it is not
/// obtainable from anything else meethook writes: identification records a similarity only
/// where it matched, so the rejections -- the half of the evidence that says how much room
/// there is above the cut -- are exactly the half that gets dropped.
///
/// The accept/reject column comes from [`identify_clusters`] rather than from comparing each
/// distance to the threshold here. Identification is argmax *then* threshold, so a reference
/// that clears the cut while a nearer one wins is not a match, and a hand-rolled comparison
/// would print it as one -- a diagnostic that disagrees with the decision it is diagnosing is
/// worse than no diagnostic.
fn print_enrolled_distances(clusters: &[SpeakerCluster]) {
    let paths = Paths::new(meethook_root());
    let enrolled =
        EnrolledSpeakers::read_or_empty(&paths).unwrap_or_else(|e| fail(&format!("{e}")));

    println!(
        "\ncosine distance to enrolled references in {}",
        paths.speakers_json().display()
    );
    println!("identification accepts a cluster's nearest reference below {IDENTIFY_DISTANCE:.3}");

    // Both degenerate shapes say so out loud. A section that renders as nothing is
    // indistinguishable from the feature being broken, and both of these are the *normal*
    // state of a fresh install rather than a failure.
    if enrolled.speakers.is_empty() {
        println!("  no enrolled speakers yet; run `meethook enroll` on a session first");
        return;
    }
    if clusters.is_empty() {
        println!("  no clusters in this recording; nothing to compare against");
        return;
    }

    let identified = identify_clusters(clusters, &enrolled);
    for cluster in clusters {
        let matched = identified.get(&cluster.id);
        match matched {
            Some(id) => println!("\n  speaker {} -> {}", cluster.id, id.name),
            None => println!("\n  speaker {} -> unidentified", cluster.id),
        }
        for speaker in &enrolled.speakers {
            // A reference of a different length came from a different embedding model, and
            // `best_match` skips it for that reason. Printing a truncated `zip` of the two as
            // a distance would invent evidence about an entry identification is ignoring.
            if speaker.embedding.len() != cluster.embedding.len() {
                println!(
                    "    {:<20} not comparable ({} dims vs the cluster's {})",
                    speaker.name,
                    speaker.embedding.len(),
                    cluster.embedding.len()
                );
                continue;
            }

            // Both sides are unit vectors by contract, so the dot product is the cosine --
            // the same arithmetic `best_match` does, so the two cannot disagree.
            let cosine: f32 = speaker
                .embedding
                .iter()
                .zip(&cluster.embedding)
                .map(|(a, b)| a * b)
                .sum();
            let accepted = matched.is_some_and(|id| id.name == speaker.name);
            println!(
                "    {:<20} {:>7.3}  {}",
                speaker.name,
                1.0 - cosine,
                if accepted { "accepted" } else { "rejected" }
            );
        }
    }
}

/// `--root`'s equivalent here: `$MEETHOOK_ROOT`, else `~/meethook`.
///
/// One resolution for the models this example runs and the `speakers.json` it reads against
/// them, so a diagnostic can never report distances from one install's database to another
/// install's embeddings.
fn meethook_root() -> PathBuf {
    match std::env::var_os("MEETHOOK_ROOT") {
        Some(root) => PathBuf::from(root),
        None => std::env::home_dir()
            .expect("could not determine the home directory; set MEETHOOK_ROOT")
            .join("meethook"),
    }
}

fn load(file_name: &str) -> ort::session::Session {
    let model = meethook_root().join("models").join(file_name);

    let loaded = open_session(&model).unwrap_or_else(|e| {
        fail(&format!(
            "{e}\nrun `cargo run --example fetch-onnx-models` first"
        ))
    });
    if !loaded.accelerated {
        eprintln!("note: CoreML declined {file_name}; running on CPU");
    }
    loaded.session
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}
