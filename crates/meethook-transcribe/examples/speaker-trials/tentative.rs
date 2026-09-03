//! What the tentative band would have guessed at, over the same corpus.
//!
//! The strict simulation above scores each session's *dominant* cluster against the enrolled
//! population. This block asks what happens to the rest: every fragment under
//! [`TENTATIVE_FLOOR_SECONDS`] that the strict pass did not award, scored only against the
//! names this session itself strictly identified -- the band's pool, nothing else.
//!
//! Every item is re-measured, cache or no cache: a cached voice holds the dominant cluster
//! alone, and fragments are exactly what the cache does not keep. That is also why the flag
//! is off by default -- an extra block must not change the bytes of any earlier calibration
//! re-run.

use std::collections::{BTreeMap, BTreeSet};

use meethook_session::{EnrolledSpeaker, EnrolledSpeakers, Paths, SpeakerCluster};
use meethook_transcribe::{
    Spread, TARGET_RATE, TENTATIVE_DISTANCE, TENTATIVE_FLOOR_SECONDS, build_session,
    cluster_speaker_turns, identify_clusters, rank_enrolled, read_track_16k_mono,
    segment_speaker_track, tentative_identifications, tentative_pairs,
};

use super::Args;
use super::manifest::Item;
use super::voices::{Models, Voice};

/// One fragment's verdict for the report below.
enum Verdict {
    /// The band awarded it a name from the pool.
    Guessed { name: String, similarity: f32 },
    /// The pool was non-empty but nothing cleared the cut.
    Rejected { nearest_distance: f32 },
    /// Nobody was strictly identified in this session, so there was nothing to guess from.
    NoPool,
}

/// The block: per-session fragment tables, then the totals a threshold decision needs.
pub fn report_tentative(paths: &Paths, args: &Args, items: &[Item], voices: &[Voice]) {
    println!("\nwhat the tentative band would have guessed at");
    println!(
        "  window: cosine distance < {TENTATIVE_DISTANCE:.3} (TENTATIVE_DISTANCE), \
         fragments under {TENTATIVE_FLOOR_SECONDS:.0} s of speech"
    );

    // The enrolled population mirrors the identification simulation: one reference per
    // person, first session in manifest order. The pool below is whatever THIS session
    // strictly named against that population, which is the band's rule verbatim.
    let mut reference_sessions: BTreeMap<String, &Voice> = BTreeMap::new();
    for voice in voices {
        reference_sessions
            .entry(voice.speaker.clone())
            .or_insert(voice);
    }
    let enrolled = EnrolledSpeakers::new(
        reference_sessions
            .values()
            .map(|voice| EnrolledSpeaker {
                name: voice.speaker.clone(),
                embedding: voice.embedding.clone(),
                clip_seconds: None,
            })
            .collect(),
    );

    let mut models = Models::new(args.root.clone());
    let mut fragments = 0usize;
    let mut guessed = 0usize;
    let mut right = 0usize;
    let mut rejected = 0usize;
    let mut no_pool = 0usize;

    for item in items {
        println!("\n{} / {}", item.speaker, item.session);
        let built = match build_session(paths, &item.wavs, &[]) {
            Ok(built) => built,
            Err(e) => {
                println!("  skipped: {e}");
                continue;
            }
        };
        let track = match read_track_16k_mono(&built.paths.speaker_wav()) {
            Ok(track) => track,
            Err(e) => {
                println!("  skipped: {e}");
                drop(built);
                continue;
            }
        };
        let wanted = (args.seconds * f64::from(TARGET_RATE)) as usize;
        let audio = &track[..wanted.min(track.len())];

        let (segmenter, embedder) = models.graphs();
        let turns = match segment_speaker_track(audio, segmenter) {
            Ok(turns) => turns,
            Err(e) => {
                println!("  skipped: {e}");
                drop(built);
                continue;
            }
        };
        let clustering = match cluster_speaker_turns(audio, &turns, embedder) {
            Ok(clustering) => clustering,
            Err(e) => {
                println!("  skipped: {e}");
                drop(built);
                continue;
            }
        };
        if !args.keep_sessions {
            let _ = std::fs::remove_dir_all(built.paths.dir());
        }

        let clusters = clustering.clusters;
        // The populations the cut is priced from, measured through the same function the
        // constant's documentation names: labelled negatives, and the unlabelled side carried
        // beside them so its shape shows up instead of hiding.
        let pairs = tentative_pairs(&clusters);
        let negatives: Vec<f32> = pairs
            .iter()
            .filter(|pair| pair.different_speaker)
            .map(|pair| pair.distance)
            .collect();
        let mut nearest_unlabelled: BTreeMap<u32, f32> = BTreeMap::new();
        for pair in &pairs {
            if pair.different_speaker {
                continue;
            }
            nearest_unlabelled
                .entry(pair.fragment)
                .and_modify(|nearest| *nearest = (*nearest).min(pair.distance))
                .or_insert(pair.distance);
        }
        println!("  heard-at-once negatives:   {}", spread_line(&negatives));
        println!(
            "  nearest non-overlap partner: {}",
            spread_line(&nearest_unlabelled.values().copied().collect::<Vec<_>>())
        );

        let identified = identify_clusters(&clusters, &enrolled);
        let pool: BTreeSet<&str> = identified.values().map(|who| who.name.as_str()).collect();
        let tentative = tentative_identifications(&clusters, &enrolled, &identified);

        let fragments_here: Vec<_> = clusters
            .iter()
            .filter(|cluster| {
                cluster.speech_seconds < TENTATIVE_FLOOR_SECONDS
                    && !identified.contains_key(&cluster.id)
            })
            .collect();
        if fragments_here.is_empty() {
            println!("  no fragments under {TENTATIVE_FLOOR_SECONDS:.0} s");
            continue;
        }
        println!(
            "  pool: {}",
            if pool.is_empty() {
                "(empty -- nobody strictly identified here)".to_string()
            } else {
                pool.iter().cloned().collect::<Vec<_>>().join(", ")
            }
        );

        for cluster in fragments_here {
            fragments += 1;
            let verdict = match tentative.get(&cluster.id) {
                Some(who) => {
                    guessed += 1;
                    if who.name == item.speaker {
                        right += 1;
                    }
                    Verdict::Guessed {
                        name: who.name.clone(),
                        similarity: who.similarity,
                    }
                }
                None if pool.is_empty() => {
                    no_pool += 1;
                    Verdict::NoPool
                }
                None => {
                    rejected += 1;
                    Verdict::Rejected {
                        nearest_distance: nearest_in_pool(cluster, &pool, &enrolled),
                    }
                }
            };
            println!(
                "  cluster {} ({:.1} s): {}",
                cluster.id,
                cluster.speech_seconds,
                render_verdict(verdict, item.speaker.as_str())
            );
        }
    }

    println!("\ntentative band totals, {fragments} fragment(s):");
    let wrong = guessed - right;
    println!("  guessed:  {guessed} ({right} on their own words, {wrong} on somebody else's)");
    if guessed > 0 {
        println!(
            "    false guesses: {:.1}% of guesses",
            100.0 * wrong as f32 / guessed as f32
        );
    }
    println!("  rejected: {rejected} (a pool existed, nothing cleared the cut)");
    println!("  no pool:  {no_pool} (nobody strictly identified in that session)");
}

/// One population's shape on one line, in the harness's usual vocabulary -- nearest-rank
/// percentiles, a value that occurred rather than an interpolation -- or the honest answer
/// when the population is empty.
fn spread_line(distances: &[f32]) -> String {
    match Spread::of(distances) {
        Some(spread) => format!(
            "{} sample(s)  min {:.3}  p05 {:.3}  median {:.3}  p95 {:.3}",
            spread.count, spread.min, spread.p05, spread.median, spread.p95
        ),
        None => "no samples".to_string(),
    }
}

fn render_verdict(verdict: Verdict, owner: &str) -> String {
    match verdict {
        Verdict::Guessed { name, similarity } => {
            let mark = if name == owner { "right" } else { "WRONG" };
            format!("guessed {name}? at {similarity:.2} similarity ({mark})")
        }
        Verdict::Rejected { nearest_distance } => format!(
            "rejected (nearest pooled distance {nearest_distance:.3}, \
             the cut admits {TENTATIVE_DISTANCE:.3})"
        ),
        Verdict::NoPool => "no pool".to_string(),
    }
}

/// How close the fragment got to its pool, through the same `rank_enrolled` the band walks,
/// so a rejection reads as evidence rather than a shrug. Printed as a distance because the
/// cut is one.
fn nearest_in_pool(
    cluster: &SpeakerCluster,
    pool: &BTreeSet<&str>,
    enrolled: &EnrolledSpeakers,
) -> f32 {
    rank_enrolled(&cluster.embedding, enrolled)
        .into_iter()
        .find(|entry| pool.contains(entry.name.as_str()))
        .map(|entry| 1.0 - entry.similarity)
        .unwrap_or(f32::INFINITY)
}
