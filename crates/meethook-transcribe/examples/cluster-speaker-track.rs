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
//! talk-time below which a cluster is treated as a fragment rather than a speaker; see
//! [`DEFAULT_FLOOR_SECONDS`].
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
//! to say whether a second pass could sweep that tail up, and at what cost, *before* the pass
//! is written. The first prints every fragment's distance to every real speaker under both
//! criteria -- the average linkage clustering merged on and the centroid distance a second pass
//! would threshold -- plus the shrinkage factor that makes the two disagree, which is the
//! mechanism stranding the fragments in the first place. The second sweeps candidate thresholds
//! and says how much of the tail each one would actually adopt.
//!
//! The last block is the other half of the free supervision. The known-different block above
//! uses one direction of segmentation's local speaker index; two turns in one window under the
//! *same* index are the same person on exactly that same authority, and nothing in meethook
//! reads that direction today. Where clustering split such a pair it is wrong, provably, with
//! no embedding and no threshold involved -- so how much speech sits in that population is a
//! number worth having before reaching for a distance.
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
    Clustering, EMBEDDING_MODEL, IDENTIFY_DISTANCE, LocalTurn, MERGE_DISTANCE, SEGMENTATION_MODEL,
    TARGET_RATE, group_distance, identify_clusters, open_session,
};

/// Talk-time below which the stranded-cluster blocks treat a cluster as a fragment looking for
/// an owner rather than as a speaker that could own one.
///
/// A flag rather than a constant in the library on purpose. Whatever floor a leftover-adoption
/// pass eventually ships has to be chosen from evidence, and a number baked in here would be a
/// number picked by eye that the pass then inherited. This default is only for reading the
/// report: 30 s sits inside the gap that separates the two populations on session
/// `20260810-093047` -- smallest of the six dominant clusters 47.8 s, largest of the eighty-nine
/// fragments 8.7 s -- so it reproduces the split without being load-bearing anywhere. Move it
/// with `--floor` on any recording where that gap sits elsewhere.
const DEFAULT_FLOOR_SECONDS: f64 = 30.0;

fn main() {
    let usage = "usage: cluster-speaker-track <session-dir | wav-file> [--write] [--floor <s>]";
    let mut target: Option<PathBuf> = None;
    let mut write = false;
    let mut floor = DEFAULT_FLOOR_SECONDS;

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

    print_stranded_clusters(&clustering, &turns, &members, floor);
    print_must_link_splits(&clustering, &turns, &members);
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

/// Every cluster below the talk-time floor, its distance to every cluster above it, and what a
/// second pass thresholding those distances would adopt.
///
/// Two blocks in one function because they share the same scaffolding -- which clusters are
/// below the floor, which are above, each one's embeddings, and whether the constraint forbids
/// the pair -- and a second function that rebuilt any of that could sweep a different population
/// from the one it printed.
///
/// Three distance columns, because the two criteria are two numbers and the third says why.
/// `linkage` is the average of the cross-pair cosine distances, which is what `agglomerate`
/// compared against [`MERGE_DISTANCE`] and declined. `centroid` is the distance between the two
/// clusters' reference vectors, which is what a leftover-adoption pass would threshold and what
/// [`IDENTIFY_DISTANCE`] already thresholds elsewhere. `shrinkage` is the factor between them,
/// and it is the mechanism of the bug rather than a curiosity: it is at most 1 and falls as a
/// group grows and spreads, so a fragment is charged for the spread of whatever group it is
/// offered to, and the cluster most likely to own it resists it hardest.
///
/// `blocked` is a separate column and not folded into the distances. A merge the same-window
/// constraint forbids cannot be adopted however close the two look, and a merge that is merely
/// far might be adoptable under a threshold nobody has chosen yet. One number could not tell a
/// reader which of those it was looking at.
fn print_stranded_clusters(
    clustering: &Clustering,
    turns: &[LocalTurn],
    members: &[Vec<usize>],
    floor: f64,
) {
    let clusters = &clustering.clusters;
    // Folded from 0.0 rather than `sum()`, whose identity for floats is `-0.0`: an empty row of
    // the sweep below would otherwise report "-0.0 s" adopted, which reads as a bug in the
    // instrument and costs the reader more attention than the whole line is worth.
    let seconds = |group: &[usize]| -> f64 {
        group
            .iter()
            .fold(0.0, |total, &c| total + clusters[c].speech_seconds)
    };
    let below: Vec<usize> = (0..clusters.len())
        .filter(|&c| clusters[c].speech_seconds < floor)
        .collect();
    let above: Vec<usize> = (0..clusters.len())
        .filter(|&c| clusters[c].speech_seconds >= floor)
        .collect();

    println!("\nstranded clusters, and where each would go (floor {floor:.1} s of speech):");

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
        seconds(&below),
        seconds(&below) + seconds(&above),
        above.len(),
        seconds(&above)
    );
    println!(
        "  linkage is what agglomerate merged on and declined, against its cut of \
         {MERGE_DISTANCE:.3}; centroid is what a second pass would threshold; shrinkage is the \
         factor between them, 1 - linkage = shrinkage * (1 - centroid); blocked means the \
         same-window constraint forbids this merge whatever the distances say"
    );

    let vectors = |group: &[usize]| -> Vec<&[f32]> {
        group
            .iter()
            .filter_map(|&turn| clustering.turn_embeddings[turn].as_deref())
            .collect()
    };

    // Where each fragment would go, kept for the sweep below so it thresholds the same numbers
    // it just printed. `None` is a fragment every above-floor cluster is barred from adopting.
    let mut nearest: Vec<(usize, Option<f32>)> = Vec::with_capacity(below.len());

    for &small in &below {
        println!(
            "\n  cluster {} ({:.1} s, {}, first at {:.1})",
            clusters[small].id,
            clusters[small].speech_seconds,
            plural(members[small].len(), "turn"),
            clusters[small].first_spoke_seconds
        );

        let mut best: Option<(f32, u32, f32)> = None;
        for &large in &above {
            let blocked = heard_apart(&members[small], &members[large], turns);
            let Some(distance) =
                group_distance(&vectors(&members[small]), &vectors(&members[large]))
            else {
                println!(
                    "     -> {:<4} no distance: one side has no embedded turns",
                    clusters[large].id
                );
                continue;
            };
            println!(
                "     -> {:<4} linkage {:.3}   centroid {:.3}   shrinkage {:.3}   {}",
                clusters[large].id,
                distance.average_linkage,
                distance.centroid,
                distance.shrinkage,
                if blocked { "blocked" } else { "-" }
            );
            if !blocked && best.is_none_or(|(closest, _, _)| distance.centroid < closest) {
                best = Some((
                    distance.centroid,
                    clusters[large].id,
                    distance.average_linkage,
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

    // The sweep. Centroid distance, because that is what a second pass would threshold and
    // saying so is the only thing keeping the two criteria from being confused for each other.
    // Argmax among permitted targets and then the cut, matching `identify_clusters`; centroids
    // frozen, because the pass being measured here adopts in one pass and does not re-centroid
    // as it goes. An iterative pass would adopt more and would need its own sweep.
    println!(
        "\n  adoption sweep over centroid distance, one pass, argmax among permitted targets \
         then the cut:"
    );
    println!("    threshold   adopted              remaining            clusters after");
    for step in 4..=16 {
        let threshold = step as f32 * 0.05;
        let adopted: Vec<usize> = nearest
            .iter()
            .filter(|(_, centroid)| centroid.is_some_and(|centroid| centroid < threshold))
            .map(|&(cluster, _)| cluster)
            .collect();
        let remaining: Vec<usize> = below
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

    let unadoptable: Vec<usize> = nearest
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
    let mut closest_above: Option<(f32, u32, u32)> = None;
    for (nth, &left) in above.iter().enumerate() {
        for &right in &above[nth + 1..] {
            let Some(distance) =
                group_distance(&vectors(&members[left]), &vectors(&members[right]))
            else {
                continue;
            };
            if closest_above.is_none_or(|(closest, _, _)| distance.centroid < closest) {
                closest_above = Some((distance.centroid, clusters[left].id, clusters[right].id));
            }
        }
    }
    match closest_above {
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

/// Turn pairs segmentation heard in one window under the same local speaker index, and what
/// clustering did with them.
///
/// The other direction of the constraint `agglomerate` already uses, and the only one nothing in
/// meethook reads. Different indices in one window are different people; the *same* index in one
/// window is one person, on exactly the same authority. Windows do not overlap and segmentation
/// closes and reopens a turn for one index whenever the silence inside it runs past a quarter
/// second, so a pair like this is two turns of one speaker rather than an artefact of the
/// decoder -- and wherever clustering put such a pair in two clusters, it is wrong, with no
/// embedding, no threshold and no model standing behind the claim.
///
/// So the number this block exists for is the last one it prints: how many clusters would be
/// left if those pairs were applied as merges. That is free supervision, and it is the lever
/// that costs nothing to pull, so its size decides whether a distance threshold has to carry the
/// whole tail on its own.
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
