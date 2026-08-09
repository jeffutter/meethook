//! Prints the speakers clustering finds in a recorded session, and how far apart they are.
//!
//! ```text
//! cargo run --release --example cluster-speaker-track ~/meethook/sessions/20260809-021730
//! cargo run --release --example cluster-speaker-track some-other-recording.wav
//! cargo run --release --example cluster-speaker-track ~/meethook/sessions/20260809-021730 --write
//! ```
//!
//! A session directory means its `speaker.wav`; any other path is used as given. `--write`
//! saves `speaker_clusters.json` into the session directory, which is how `enroll` gets
//! something real to develop against without re-running transcription.
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
    EMBEDDING_MODEL, IDENTIFY_DISTANCE, LocalTurn, SEGMENTATION_MODEL, TARGET_RATE,
    identify_clusters, open_session,
};

fn main() {
    let mut args = std::env::args_os().skip(1);
    let Some(target) = args.next().map(PathBuf::from) else {
        eprintln!("usage: cluster-speaker-track <session-dir | wav-file> [--write]");
        std::process::exit(2);
    };
    let write = args.any(|a| a == "--write");

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
    let span = |turn: usize| format!("{:.1}-{:.1}", turns[turn].start_s, turns[turn].end_s);

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
            if turns[i].window != turns[j].window
                || turns[i].local_speaker == turns[j].local_speaker
            {
                continue;
            }
            if let (Some(a), Some(b)) = (&embeddings[i], &embeddings[j]) {
                pairs.push((i, j, distance(a, b)));
            }
        }
    }
    pairs
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
