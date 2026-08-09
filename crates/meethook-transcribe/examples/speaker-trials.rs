//! Scores a whole speaker trial list: many cross-session pairs measured at once.
//!
//! ```text
//! cargo run --release --example speaker-trials -- \
//!   --root /tmp/calibration --embeddings /tmp/calibration/embeddings.json trials.tsv
//!
//! # re-print at a different cut, from the cache, with no audio and no models
//! cargo run --release --example speaker-trials -- \
//!   --root /tmp/calibration --embeddings /tmp/calibration/embeddings.json \
//!   --threshold 0.55 trials.tsv
//! ```
//!
//! Why this exists: `cluster-speaker-track` prints every cluster-to-reference distance for
//! *one* session, which is the right arithmetic at the wrong scale. Deciding whether
//! `IDENTIFY_DISTANCE` is in the right place needs error rates over hundreds of pairs -- a
//! false-accept rate, a false-reject rate, an equal-error rate, and whether the two
//! distributions overlap at all -- and none of those are obtainable by reading a table by eye.
//!
//! # The manifest
//!
//! Tab-separated, one line per source wav. `#` comments and blank lines are ignored:
//!
//! ```text
//! speaker_id <TAB> session_id <TAB> /path/to/audio.wav
//! ```
//!
//! Lines sharing a `(speaker_id, session_id)` become one **item**, spliced in file order. TSV
//! rather than JSON because the thing that writes one is a shell pipeline over a corpus
//! directory listing, and a person deciding what to include has to be able to write one by
//! hand.
//!
//! The item -- one person as they sounded in one recording session -- is the unit here because
//! that is what an enrolled reference *is*. Each becomes exactly one voice, through the same
//! `build_session` -> `segment_speaker_track` -> `cluster_speaker_turns` path a real session
//! takes, and the voice is the dominant cluster rather than a mean over clusters: `enroll`
//! copies one cluster's embedding verbatim, so averaging here would be calibrating an
//! algorithm meethook does not run.
//!
//! # Two rules that make the numbers mean what they claim
//!
//! **No pair drawn from a single session is a trial.** That is `MERGE_DISTANCE`'s question,
//! and including such pairs would flatter the same-speaker side with exactly the within-session
//! variation `IDENTIFY_DISTANCE` does not govern. Excluded pairs are counted and printed, so
//! the rule is visible rather than trusted.
//!
//! **The identification simulation goes through the real [`identify_clusters`].** The
//! trial-list rates above it are the standard speaker-verification quantities and they are
//! *not* what meethook decides: identification is argmax over every enrolled reference and
//! *then* the cut, so a reference that clears the threshold while a nearer one wins is not a
//! match. Both are reported, because only one of them is about this code.
//!
//! No audio is written into the repository by any path here, and the only artifact worth
//! carrying away from a corpus run is `--embeddings`, which holds derived vectors and no
//! audio.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use meethook_session::{
    EnrolledSpeaker, EnrolledSpeakers, Paths, RepresentativeSegment, SpeakerCluster,
};
use meethook_transcribe::{
    EMBEDDING_MODEL, IDENTIFY_DISTANCE, ImportedSource, LevelSummary, SEGMENTATION_MODEL,
    SPLICE_GAP_S, Spread, TARGET_RATE, Trial, TrialReport, build_session, cluster_speaker_turns,
    identify_clusters, open_session, read_track_16k_mono, score_trials, segment_speaker_track,
};
use serde::{Deserialize, Serialize};

/// How much of each item's audio is measured, in seconds, unless `--seconds` says otherwise.
///
/// An enrolled reference is one meeting's cluster -- minutes of speech, not half an hour -- so
/// a thirty-minute corpus track measured whole would be a more flattering reference than
/// meethook ever holds.
const DEFAULT_SECONDS: f64 = 120.0;

/// How much speech an item's dominant cluster must hold to be scored, unless `--min-speech`
/// says otherwise.
///
/// This tool's own floor, deliberately not `speakers::MIN_EMBEDDABLE_SECONDS`: that constant
/// is private, it governs whether a *turn* can be embedded at all, and a copy of it here would
/// be a number that drifts out of agreement with the one it claims to be. Printed in the
/// header for the same reason the threshold is: a population nobody can reconstruct is not
/// evidence.
const DEFAULT_MIN_SPEECH: f64 = 3.0;

fn main() {
    let args = parse().unwrap_or_else(|message| {
        eprintln!("{message}");
        eprintln!(
            "usage: speaker-trials [--root <dir>] [--seconds <n>] [--min-speech <n>]\n       \
             [--threshold <distance>] [--embeddings <file>] [--fresh] [--keep-sessions] \
             <manifest.tsv>\n       \
             the root may also come from $MEETHOOK_ROOT; there is no default"
        );
        std::process::exit(2);
    });

    let items = read_manifest(&args.manifest).unwrap_or_else(|e| fail(&e));
    let paths = Paths::new(&args.root);

    println!("root:      {}", paths.root().display());
    println!(
        "manifest:  {} ({} items)",
        args.manifest.display(),
        items.len()
    );
    println!(
        "threshold: {:.3}{}",
        args.threshold,
        if args.threshold == IDENTIFY_DISTANCE {
            "  (IDENTIFY_DISTANCE)"
        } else {
            "  (--threshold; IDENTIFY_DISTANCE is not this)"
        }
    );
    println!(
        "limits:    at most {:.0} s measured per item, dominant cluster must hold {:.1} s of \
         speech",
        args.seconds, args.min_speech
    );

    let voices = embed_items(&paths, &args, &items);
    if let Some(path) = &args.embeddings {
        write_cache(path, &voices);
        println!("\nwrote {} voice(s) to {}", voices.len(), path.display());
    }

    let trials = pair_up(&voices);
    report_shape(&voices, &trials);

    let report = score_trials(&trials.trials, args.threshold);
    report_scores(&report);
    report_identification(&voices, args.threshold);
}

// ---------------------------------------------------------------------------------------
// Arguments and manifest
// ---------------------------------------------------------------------------------------

struct Args {
    root: PathBuf,
    manifest: PathBuf,
    seconds: f64,
    min_speech: f64,
    threshold: f32,
    embeddings: Option<PathBuf>,
    /// Ignore whatever `--embeddings` already holds and re-measure every item.
    fresh: bool,
    keep_sessions: bool,
}

/// Hand-rolled rather than clap, matching the other examples in this crate: a diagnostic must
/// never be the reason a build breaks.
///
/// There is deliberately no `~/meethook` fallback for the root, unlike `cluster-speaker-track`
/// and like `build-session`: this tool *creates* session directories, and it must not be able
/// to create a hundred of them among somebody's real recordings because a variable was unset.
fn parse() -> Result<Args, String> {
    let mut root = std::env::var_os("MEETHOOK_ROOT").map(PathBuf::from);
    let mut manifest: Option<PathBuf> = None;
    let mut seconds = DEFAULT_SECONDS;
    let mut min_speech = DEFAULT_MIN_SPEECH;
    let mut threshold = IDENTIFY_DISTANCE;
    let mut embeddings = None;
    let mut fresh = false;
    let mut keep_sessions = false;

    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |flag: &str| {
            args.next()
                .and_then(|v| v.into_string().ok())
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        let number = |flag: &str, raw: String| {
            raw.parse::<f64>()
                .map_err(|_| format!("{flag} needs a number, not {raw:?}"))
        };
        match arg.to_str() {
            Some("--root") => root = Some(PathBuf::from(value("--root")?)),
            Some("--embeddings") => embeddings = Some(PathBuf::from(value("--embeddings")?)),
            Some("--seconds") => seconds = number("--seconds", value("--seconds")?)?,
            Some("--min-speech") => min_speech = number("--min-speech", value("--min-speech")?)?,
            Some("--threshold") => {
                threshold = number("--threshold", value("--threshold")?)? as f32;
            }
            Some("--fresh") => fresh = true,
            Some("--keep-sessions") => keep_sessions = true,
            Some(flag) if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            _ if manifest.is_none() => manifest = Some(PathBuf::from(arg)),
            _ => return Err("only one manifest may be given".to_string()),
        }
    }

    Ok(Args {
        root: root.ok_or("no root: pass --root or set $MEETHOOK_ROOT")?,
        manifest: manifest.ok_or("no manifest was given")?,
        seconds,
        min_speech,
        threshold,
        embeddings,
        fresh,
        keep_sessions,
    })
}

/// One person as they sounded in one recording session: the unit everything here measures.
struct Item {
    speaker: String,
    session: String,
    wavs: Vec<PathBuf>,
}

/// Parses the manifest, grouping lines that share a `(speaker, session)` into one item.
///
/// Items come back in manifest order, and so do the wav files within each -- which is what
/// makes "each speaker's *first* session is the enrolled reference" a reproducible statement
/// about a file the operator wrote rather than about whatever order a hash map felt like.
fn read_manifest(path: &Path) -> Result<Vec<Item>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;

    let mut items: Vec<Item> = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
        let [speaker, session, wav] = fields.as_slice() else {
            return Err(format!(
                "{}:{}: expected three tab-separated fields \
                 (speaker, session, wav), found {}",
                path.display(),
                number + 1,
                fields.len()
            ));
        };
        if speaker.is_empty() || session.is_empty() || wav.is_empty() {
            return Err(format!(
                "{}:{}: no field may be empty",
                path.display(),
                number + 1
            ));
        }

        match items
            .iter_mut()
            .find(|item| item.speaker == *speaker && item.session == *session)
        {
            Some(item) => item.wavs.push(PathBuf::from(wav)),
            None => items.push(Item {
                speaker: (*speaker).to_string(),
                session: (*session).to_string(),
                wavs: vec![PathBuf::from(wav)],
            }),
        }
    }

    if items.is_empty() {
        return Err(format!("{} named no audio at all", path.display()));
    }
    Ok(items)
}

// ---------------------------------------------------------------------------------------
// Items to voices
// ---------------------------------------------------------------------------------------

/// One item's voice: the dominant cluster of the session built from its audio.
#[derive(Clone, Serialize, Deserialize)]
struct Voice {
    speaker: String,
    session: String,
    /// Carried so a cache written before an embedding-model change and topped up after it
    /// refuses rather than scoring: two lengths mean two incomparable spaces, and a
    /// truncating `zip` would return plausible cosines from unrelated ones.
    dimensions: usize,
    speech_seconds: f64,
    embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct EmbeddingCache {
    schema_version: u32,
    items: Vec<Voice>,
}

const CACHE_SCHEMA_VERSION: u32 = 1;

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

fn load(root: &Path, file_name: &str) -> ort::session::Session {
    let model = root.join("models").join(file_name);
    let loaded = open_session(&model).unwrap_or_else(|e| {
        fail(&format!(
            "{e}\nrun `cargo run --release --example fetch-onnx-models` first"
        ))
    });
    if !loaded.accelerated {
        eprintln!("note: CoreML declined {file_name}; running on CPU");
    }
    loaded.session
}

/// Turns every item into one voice, printing what it measured and dropping -- by name -- the
/// items that could not produce one.
///
/// A run that quietly discarded a third of the manifest is a run whose error rates describe a
/// different population than the one they name, so the drop list is printed even when it is
/// empty.
fn embed_items(paths: &Paths, args: &Args, items: &[Item]) -> Vec<Voice> {
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
    levels(&built.speaker);

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

fn converted(source: &ImportedSource) -> String {
    format!(
        "{}: {} Hz, {} ch -> {TARGET_RATE} Hz mono ({:.2} s)",
        source.path.display(),
        source.sample_rate,
        source.channels,
        source.samples as f64 / f64::from(TARGET_RATE)
    )
}

/// The same line `build-session` prints, and for the same reason: reading it takes a second
/// and turns "the measurement came back empty" into "the input was dead".
fn levels(summary: &LevelSummary) {
    let dbfs = summary.peak_dbfs();
    let peak = if dbfs.is_infinite() {
        "0.0 (digital silence)".to_string()
    } else {
        format!("{:.4} ({dbfs:.1} dBFS)", summary.peak)
    };
    println!(
        "  speaker.wav  {:.2} s, peak {peak}, {:.1}% above floor, longest run {:.3} s",
        summary.duration_s(),
        summary.above_fraction() * 100.0,
        summary.longest_run_s()
    );
}

// ---------------------------------------------------------------------------------------
// The embedding cache
// ---------------------------------------------------------------------------------------

/// Reads `--embeddings`, or an empty map if it does not exist yet.
///
/// A file that exists and does not parse is fatal rather than ignored: silently re-measuring
/// everything would be slow and confusing, and silently scoring half a cache would be worse.
fn read_cache(path: &Path) -> BTreeMap<(String, String), Voice> {
    if !path.exists() {
        return BTreeMap::new();
    }
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| fail(&format!("could not read {}: {e}", path.display())));
    let cache: EmbeddingCache = serde_json::from_str(&text)
        .unwrap_or_else(|e| fail(&format!("could not parse {}: {e}", path.display())));

    if cache.schema_version != CACHE_SCHEMA_VERSION {
        fail(&format!(
            "{} is schema version {}, this tool writes {CACHE_SCHEMA_VERSION}",
            path.display(),
            cache.schema_version
        ));
    }

    // Two dimensions in one file means two embedding models and two spaces that cannot be
    // compared. `best_match` skips such a reference for exactly this reason; here it would
    // poison a whole run's worth of rates instead of one row, so it stops the run.
    let dimensions: BTreeSet<usize> = cache.items.iter().map(|item| item.dimensions).collect();
    if dimensions.len() > 1 {
        fail(&format!(
            "{} mixes {dimensions:?} dimensions, so it was written by more than one \
             embedding model. Delete it or re-run with --fresh.",
            path.display()
        ));
    }
    for item in &cache.items {
        if item.embedding.len() != item.dimensions {
            fail(&format!(
                "{}: {} / {} says {} dimensions but carries {}",
                path.display(),
                item.speaker,
                item.session,
                item.dimensions,
                item.embedding.len()
            ));
        }
    }

    println!(
        "cache:     {} ({} voice(s) available for re-use)",
        path.display(),
        cache.items.len()
    );
    cache
        .items
        .into_iter()
        .map(|item| ((item.speaker.clone(), item.session.clone()), item))
        .collect()
}

fn write_cache(path: &Path, voices: &[Voice]) {
    let cache = EmbeddingCache {
        schema_version: CACHE_SCHEMA_VERSION,
        items: voices.to_vec(),
    };
    let json = serde_json::to_string_pretty(&cache)
        .unwrap_or_else(|e| fail(&format!("could not serialize the embedding cache: {e}")));
    std::fs::write(path, json)
        .unwrap_or_else(|e| fail(&format!("could not write {}: {e}", path.display())));
}

// ---------------------------------------------------------------------------------------
// Pairing and reporting
// ---------------------------------------------------------------------------------------

struct TrialList {
    trials: Vec<Trial>,
    /// Pairs refused because both items came from one recording session.
    within_session: usize,
}

/// Every legal unordered pair of items.
///
/// The one rule with teeth: **two items from the same session are never a trial**, whoever
/// they are. For two items of one speaker that is `MERGE_DISTANCE`'s question rather than this
/// one; for two speakers recorded in a single session it is a pair that shares a microphone,
/// a room and a codec, which is exactly the variation a cross-session threshold has to survive
/// and would therefore be measured too favourably.
///
/// Unordered and counted once, because cosine distance is symmetric.
fn pair_up(voices: &[Voice]) -> TrialList {
    let mut trials = Vec::new();
    let mut within_session = 0;
    for (index, a) in voices.iter().enumerate() {
        for b in &voices[..index] {
            if a.session == b.session {
                within_session += 1;
                continue;
            }
            trials.push(Trial {
                same_speaker: a.speaker == b.speaker,
                distance: distance(&a.embedding, &b.embedding),
            });
        }
    }
    TrialList {
        trials,
        within_session,
    }
}

/// Cosine distance between two unit-length voices: the same dot product `best_match` does, so
/// this cannot disagree with the decision it is measuring.
fn distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
}

/// The shape of the trial list, printed before any statistic taken over it.
///
/// A trial list whose shape is not stated cannot be checked, and every published number of
/// this kind is quoted alongside its trial count.
fn report_shape(voices: &[Voice], trials: &TrialList) {
    let mut sessions: BTreeMap<&str, usize> = BTreeMap::new();
    for voice in voices {
        *sessions.entry(voice.speaker.as_str()).or_default() += 1;
    }
    let per_speaker: Vec<f32> = sessions.values().map(|count| *count as f32).collect();

    let same = trials
        .trials
        .iter()
        .filter(|trial| trial.same_speaker)
        .count();

    println!("\ntrial list");
    println!(
        "  {} item(s) over {} speaker(s)",
        voices.len(),
        sessions.len()
    );
    match Spread::of(&per_speaker) {
        Some(spread) => println!(
            "  sessions per speaker: min {:.0}  median {:.0}  max {:.0}",
            spread.min, spread.median, spread.max
        ),
        None => println!("  sessions per speaker: no items at all"),
    }
    println!("  {same} same-speaker pair(s)");
    println!("  {} different-speaker pair(s)", trials.trials.len() - same);
    println!(
        "  {} pair(s) refused for sharing one session (MERGE_DISTANCE's question, not this one)",
        trials.within_session
    );

    if same == 0 {
        println!(
            "  note: no same-speaker pairs. Every speaker in this manifest was recorded in one \
             session only, so nothing here measures whether one voice matches itself."
        );
    }
    if trials.trials.len() == same {
        println!(
            "  note: no different-speaker pairs. This manifest names one speaker, so nothing \
             here measures whether two voices are told apart."
        );
    }
}

fn report_scores(report: &TrialReport) {
    println!("\ndistances");
    print_spread("same speaker     ", report.same.as_ref());
    print_spread("different speaker", report.different.as_ref());

    println!("\nat threshold {:.3}", report.threshold);
    println!(
        "  false accepts: {} different-speaker pair(s) below the cut{}",
        report.false_accepts,
        percent(report.false_accept_rate)
    );
    println!(
        "  false rejects: {} same-speaker pair(s) at or above it{}",
        report.false_rejects,
        percent(report.false_reject_rate)
    );

    println!("\nseparation");
    match report.overlap {
        Some((min_different, max_same)) => println!(
            "  NO SINGLE THRESHOLD SEPARATES THESE TWO POPULATIONS. They overlap between \
             {min_different:.3} (the closest different-speaker pair) and {max_same:.3} (the \
             furthest-apart same-speaker pair); every cut inside that band trades one kind of \
             mistake for the other."
        ),
        None => match (report.same, report.different) {
            (Some(same), Some(different)) => println!(
                "  the two populations do not overlap: every same-speaker pair is below \
                 {:.3} and every different-speaker pair is at or above {:.3}, so any cut in \
                 between makes no mistakes on this list",
                same.max, different.min
            ),
            _ => println!("  not measurable: one side of the trial list is empty"),
        },
    }

    match report.equal_error {
        Some(equal_error) => println!(
            "  equal error rate {:.1}% at a cut of {:.3}  (the mean of the two rates where \
             they come closest to crossing)",
            equal_error.rate * 100.0,
            equal_error.threshold
        ),
        None => println!("  equal error rate: not measurable, one side of the list is empty"),
    }
    match report.zero_false_accept {
        Some(zero) => println!(
            "  the largest cut that misattributes nobody is {:.3}, and it rejects {:.1}% of \
             same-speaker pairs",
            zero.threshold,
            zero.false_reject_rate * 100.0
        ),
        None => println!("  no misattribution-free cut is measurable from this list"),
    }
}

fn print_spread(label: &str, spread: Option<&Spread>) {
    match spread {
        Some(s) => println!(
            "  {label}: {} pair(s)  min {:.3}  p05 {:.3}  median {:.3}  p95 {:.3}  max {:.3}  \
             mean {:.3}",
            s.count, s.min, s.p05, s.median, s.p95, s.max, s.mean
        ),
        None => println!("  {label}: no pairs"),
    }
}

fn percent(rate: Option<f32>) -> String {
    match rate {
        Some(rate) => format!("  ({:.1}%)", rate * 100.0),
        None => "  (no such pairs, so no rate)".to_string(),
    }
}

// ---------------------------------------------------------------------------------------
// What meethook itself would have decided
// ---------------------------------------------------------------------------------------

/// Runs the real identification decision over the same items, and separates its three
/// outcomes.
///
/// The rates above are the standard speaker-verification quantities and they are not what this
/// code does. [`identify_clusters`] is argmax over every enrolled reference and *then* the cut,
/// so a reference that clears the threshold while a nearer one wins is not a match, and a bare
/// trial-list false-accept rate would be a number about a decision rule meethook does not use.
///
/// Each speaker's **first session in manifest order** is their enrolled reference -- mirroring
/// `enroll`, which stores one session's cluster and replaces rather than averages -- and every
/// other session of theirs is a probe. The chosen reference is printed per speaker, because
/// "first" is only reproducible if it names an order somebody can re-read.
///
/// `threshold` is accepted but not applied: it exists so the caller can *say* whether the
/// simulation below is running at the same cut as the trial-list rates. Identification uses
/// [`IDENTIFY_DISTANCE`] internally and nothing here can override it, which is the point.
fn report_identification(voices: &[Voice], threshold: f32) {
    println!("\nwhat meethook would have decided (identify_clusters, argmax then threshold)");
    if threshold != IDENTIFY_DISTANCE {
        println!(
            "  note: --threshold {threshold:.3} moved the rates above; this block is the real \
             decision, which is fixed at IDENTIFY_DISTANCE {IDENTIFY_DISTANCE:.3}"
        );
    }

    // First-seen wins, and manifest order is preserved all the way from `read_manifest`.
    let mut references: Vec<&Voice> = Vec::new();
    let mut probes: Vec<&Voice> = Vec::new();
    for voice in voices {
        if references
            .iter()
            .any(|reference| reference.speaker == voice.speaker)
        {
            probes.push(voice);
        } else {
            references.push(voice);
        }
    }

    println!(
        "  enrolled {} speaker(s) from their first session, probing with {} other session(s)",
        references.len(),
        probes.len()
    );
    for reference in &references {
        println!("    {} <- {}", reference.speaker, reference.session);
    }
    if probes.is_empty() {
        println!(
            "  no probes: every speaker has exactly one session, so there is nothing to \
             identify. A closed-set simulation needs at least one speaker recorded twice."
        );
        return;
    }

    let enrolled = database(&references, None);
    let (mut correct, mut misattributed, mut rejected) = (0usize, 0usize, 0usize);
    for probe in &probes {
        match identify(probe, &enrolled) {
            Some(name) if name == probe.speaker => correct += 1,
            Some(name) => {
                misattributed += 1;
                println!(
                    "    MISATTRIBUTED: {} / {} named as {name}",
                    probe.speaker, probe.session
                );
            }
            None => rejected += 1,
        }
    }

    let of_probes = |count: usize| format!("{:.1}%", 100.0 * count as f32 / probes.len() as f32);
    println!("  closed set, {} probe(s):", probes.len());
    println!("    correct:        {correct} ({})", of_probes(correct));
    println!(
        "    misattributed:  {misattributed} ({}) -- one person's words under another \
         person's name",
        of_probes(misattributed)
    );
    println!(
        "    rejected:       {rejected} ({}) -- an enrolled speaker left as Unknown N",
        of_probes(rejected)
    );

    // The same sweep with the probe's own speaker taken out of the database: how often a voice
    // meethook has never enrolled is given somebody else's name anyway. One filtered database
    // and no new code, and it is the number that maps most directly onto user-visible harm.
    if references.len() < 2 {
        println!(
            "  open set: needs at least two enrolled speakers to be meaningful, this run has {}",
            references.len()
        );
        return;
    }
    let mut false_alarms = 0usize;
    for probe in &probes {
        let strangers = database(&references, Some(probe.speaker.as_str()));
        if let Some(name) = identify(probe, &strangers) {
            false_alarms += 1;
            println!(
                "    OPEN-SET FALSE ALARM: {} / {} named as {name} with {} not enrolled",
                probe.speaker, probe.session, probe.speaker
            );
        }
    }
    println!(
        "  open set, same {} probe(s) with their own speaker removed from the database:",
        probes.len()
    );
    println!(
        "    false alarms:   {false_alarms} ({}) -- an unenrolled voice given a name",
        of_probes(false_alarms)
    );
}

/// The enrolled database these references would have produced, optionally without one person.
fn database(references: &[&Voice], without: Option<&str>) -> EnrolledSpeakers {
    EnrolledSpeakers::new(
        references
            .iter()
            .filter(|reference| Some(reference.speaker.as_str()) != without)
            .map(|reference| EnrolledSpeaker {
                name: reference.speaker.clone(),
                embedding: reference.embedding.clone(),
            })
            .collect(),
    )
}

/// The name the real decision would have put on this voice, if any.
fn identify(probe: &Voice, enrolled: &EnrolledSpeakers) -> Option<String> {
    let cluster = SpeakerCluster {
        id: 0,
        embedding: probe.embedding.clone(),
        speech_seconds: probe.speech_seconds,
        first_spoke_seconds: 0.0,
        representatives: vec![RepresentativeSegment {
            start: 0.0,
            end: probe.speech_seconds.min(2.0),
        }],
    };
    identify_clusters(std::slice::from_ref(&cluster), enrolled)
        .remove(&0)
        .map(|identification| identification.name)
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}
