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

#[path = "../support/mod.rs"]
mod support;

mod adoption;
mod clustering;
mod must_link;
mod reference;

use std::path::PathBuf;

use adoption::{print_adoption_populations, print_stranded_clusters};
use clustering::{
    print_cross_speaker_distances, print_inter_speaker_matrix, print_known_different,
    print_within_cluster_spreads,
};
use meethook_session::{SessionId, SessionPaths, SpeakerClusters};
use meethook_transcribe::{
    ADOPTION_DISTANCE, EMBEDDING_MODEL, SEGMENTATION_MODEL, SPEAKER_FLOOR_SECONDS, TARGET_RATE,
    adoption_populations,
};
use must_link::print_must_link_splits;
use reference::{print_enrolled_distances, print_reference_durations};
use support::{fail, load};

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

    let mut segmenter = load(&meethook_root(), SEGMENTATION_MODEL.file_name);
    let mut embedder = load(&meethook_root(), EMBEDDING_MODEL.file_name);

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
    print_within_cluster_spreads(&clustering, &members, &turns);
    print_inter_speaker_matrix(&clustering);
    print_cross_speaker_distances(&clustering, &members, &turns);
    print_known_different(&clustering, &turns);

    // Built once and read by both of the blocks below, so the grid one prints and the population
    // the other scores are the same numbers rather than two computations that could disagree.
    let populations = adoption_populations(&turns, &clustering, floor);

    print_stranded_clusters(&clustering, &members, &populations, floor);
    print_must_link_splits(&clustering, &turns, &members);
    print_adoption_populations(&clustering, &turns, &populations, floor, cut);
    print_enrolled_distances(&clustering.clusters);
    print_reference_durations(&clustering, &turns, floor);

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

/// `--root`'s equivalent here: `$MEETHOOK_ROOT`, else `~/meethook`.
///
/// One resolution for the models this example runs and the `speakers.json` it reads against
/// them, so a diagnostic can never report distances from one install's database to another
/// install's embeddings.
pub(crate) fn meethook_root() -> PathBuf {
    match std::env::var_os("MEETHOOK_ROOT") {
        Some(root) => PathBuf::from(root),
        None => std::env::home_dir()
            .expect("could not determine the home directory; set MEETHOOK_ROOT")
            .join("meethook"),
    }
}
