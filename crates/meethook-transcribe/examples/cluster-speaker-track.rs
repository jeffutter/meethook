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
    EMBEDDING_MODEL, IDENTIFY_DISTANCE, SEGMENTATION_MODEL, TARGET_RATE, identify_clusters,
    open_session,
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

    for cluster in &clustering.clusters {
        let spoken: Vec<String> = turns
            .iter()
            .zip(&clustering.assignment)
            .filter(|(_, assigned)| **assigned == Some(cluster.id))
            .map(|(turn, _)| format!("{:.1}-{:.1}", turn.start_s, turn.end_s))
            .collect();
        println!(
            "\nspeaker {}: {:.1} s of speech over {} turns",
            cluster.id,
            cluster.speech_seconds,
            spoken.len()
        );
        println!("  turns: {}", spoken.join(" "));
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
