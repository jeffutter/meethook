//! Items to voices.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use meethook_session::Paths;
use meethook_transcribe::{
    EMBEDDING_MODEL, SEGMENTATION_MODEL, SPLICE_GAP_S, TARGET_RATE, build_session,
    cluster_speaker_turns, read_track_16k_mono, segment_speaker_track,
};
use serde::{Deserialize, Serialize};

use super::Args;
use super::cache::read_cache;
use super::manifest::Item;
use super::support::session_prep::{converted, levels};
use super::support::{fail, load};

/// One item's voice: the dominant cluster of the session built from its audio.
#[derive(Clone, Serialize, Deserialize)]
pub struct Voice {
    pub speaker: String,
    pub session: String,
    /// Carried so a cache written before an embedding-model change and topped up after it
    /// refuses rather than scoring: two lengths mean two incomparable spaces, and a
    /// truncating `zip` would return plausible cosines from unrelated ones.
    pub dimensions: usize,
    pub speech_seconds: f64,
    pub embedding: Vec<f32>,
}

/// The two ONNX graphs, opened at most once and only if some item actually needs measuring.
///
/// A run that re-scores a cache at a new threshold should not need the models on disk at all
/// -- that is the whole point of the cache -- so loading is lazy rather than up front.
struct Models {
    root: PathBuf,
    segmenter: Option<ort::session::Session>,
    embedder: Option<ort::session::Session>,
}

impl Models {
    fn graphs(&mut self) -> (&mut ort::session::Session, &mut ort::session::Session) {
        let root = &self.root;
        let segmenter = self
            .segmenter
            .get_or_insert_with(|| load(root, SEGMENTATION_MODEL.file_name));
        let embedder = self
            .embedder
            .get_or_insert_with(|| load(root, EMBEDDING_MODEL.file_name));
        (segmenter, embedder)
    }
}

/// Turns every item into one voice, printing what it measured and dropping -- by name -- the
/// items that could not produce one.
///
/// A run that quietly discarded a third of the manifest is a run whose error rates describe a
/// different population than the one they name, so the drop list is printed even when it is
/// empty.
pub fn embed_items(paths: &Paths, args: &Args, items: &[Item]) -> Vec<Voice> {
    let cached = match (&args.embeddings, args.fresh) {
        (Some(path), false) => read_cache(path),
        _ => BTreeMap::new(),
    };

    let mut models = Models {
        root: args.root.clone(),
        segmenter: None,
        embedder: None,
    };
    let mut voices = Vec::new();
    let mut dropped: Vec<(String, String)> = Vec::new();

    for item in items {
        println!("\n{} / {}", item.speaker, item.session);
        let key = (item.speaker.clone(), item.session.clone());
        if let Some(voice) = cached.get(&key) {
            // Stated as re-use, per item: a cache is keyed on `(speaker, session)` and cannot
            // know that the wav files behind that key changed. A number nobody can tell came
            // from today's audio is not evidence.
            println!(
                "  reused from the embedding cache: {} dims, {:.1} s of speech \
                 (pass --fresh to re-measure)",
                voice.dimensions, voice.speech_seconds
            );
            voices.push(voice.clone());
            continue;
        }

        match measure(paths, args, item, &mut models) {
            Ok(voice) => voices.push(voice),
            Err(reason) => {
                println!("  dropped: {reason}");
                dropped.push((format!("{} / {}", item.speaker, item.session), reason));
            }
        }
    }

    println!("\ndropped {} of {} item(s)", dropped.len(), items.len());
    for (item, reason) in &dropped {
        println!("  {item}: {reason}");
    }

    // One embedding model per run, or the dot products below compare unrelated spaces. The
    // cache is where a mixture arrives from, and it is checked on load as well; this catches
    // the other half, a cache from an older model topped up with today's measurements.
    let dimensions: BTreeSet<usize> = voices.iter().map(|voice| voice.dimensions).collect();
    if dimensions.len() > 1 {
        fail(&format!(
            "these voices are not comparable: {dimensions:?} dimensions in one run, which \
             means more than one embedding model. Re-run with --fresh."
        ));
    }
    voices
}

/// Builds one item's session, diarizes it, and takes the dominant cluster as that item's
/// voice.
///
/// The session is built with the production [`build_session`] rather than a re-implementation
/// of it, so the audio measured here is bit-for-bit the audio a real session would hold, and
/// the conversion and level lines come along for free -- which is what turns "this item
/// produced no clusters" into "this item was silent" without a second investigation.
fn measure(paths: &Paths, args: &Args, item: &Item, models: &mut Models) -> Result<Voice, String> {
    let built = build_session(paths, &item.wavs, &[]).map_err(|e| e.to_string())?;
    for source in &built.speaker_sources {
        println!("  {}", converted(source));
    }
    if built.speaker_sources.len() > 1 {
        println!("  spliced with {SPLICE_GAP_S:.2} s of silence between sources");
    }
    levels("speaker.wav", &built.speaker);

    let measured = measure_built(args, &built.paths.speaker_wav(), item, models);

    if !args.keep_sessions {
        // The disk cost of a manifest is transient by construction: one built session at a
        // time, removed as soon as its embedding is out.
        let _ = std::fs::remove_dir_all(built.paths.dir());
    } else {
        println!("  kept {}", built.paths.dir().display());
    }
    measured
}

fn measure_built(
    args: &Args,
    speaker_wav: &Path,
    item: &Item,
    models: &mut Models,
) -> Result<Voice, String> {
    let track = read_track_16k_mono(speaker_wav).map_err(|e| e.to_string())?;
    let wanted = (args.seconds * f64::from(TARGET_RATE)) as usize;
    let audio = &track[..wanted.min(track.len())];
    let measured_seconds = audio.len() as f64 / f64::from(TARGET_RATE);
    println!(
        "  measured {measured_seconds:.1} s of {:.1} s",
        track.len() as f64 / f64::from(TARGET_RATE)
    );

    let (segmenter, embedder) = models.graphs();
    let turns = segment_speaker_track(audio, segmenter).map_err(|e| e.to_string())?;
    let clustering = cluster_speaker_turns(audio, &turns, embedder).map_err(|e| e.to_string())?;
    println!(
        "  {} turns, {} cluster(s), {} turn(s) too short to embed",
        turns.len(),
        clustering.clusters.len(),
        clustering.skipped()
    );

    // Most talk time wins. `cluster_speaker_turns` already sorts groups that way, so this is
    // cluster 0 in practice -- taken by value rather than by position so that it stays the
    // documented rule rather than a dependency on an ordering stated elsewhere.
    let dominant = clustering
        .clusters
        .iter()
        .max_by(|a, b| a.speech_seconds.total_cmp(&b.speech_seconds))
        .ok_or_else(|| {
            "no clusters at all (silence, or every turn too short to embed)".to_string()
        })?;

    println!(
        "  voice: cluster {}, {:.1} s of speech ({:.0}% of the measured audio)",
        dominant.id,
        dominant.speech_seconds,
        100.0 * dominant.speech_seconds / measured_seconds.max(f64::MIN_POSITIVE)
    );
    if dominant.speech_seconds < args.min_speech {
        return Err(format!(
            "dominant cluster holds {:.1} s of speech, under the {:.1} s floor",
            dominant.speech_seconds, args.min_speech
        ));
    }

    Ok(Voice {
        speaker: item.speaker.clone(),
        session: item.session.clone(),
        dimensions: dominant.embedding.len(),
        speech_seconds: dominant.speech_seconds,
        embedding: dominant.embedding.clone(),
    })
}
