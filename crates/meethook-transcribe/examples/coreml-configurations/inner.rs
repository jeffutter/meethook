//! The macOS body of the `coreml-configurations` example; see the sibling
//! `main.rs` for what the tool does and why it exists here at all.
//!
//! A module rather than an inline `#[cfg]` on the example itself because an example crate
//! must still define `main` off macOS, and `#![cfg(...)]` would leave none behind.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use meethook_transcribe::{
    Clustering, EMBEDDING_MODEL, LocalTurn, SEGMENTATION_MODEL, cluster_speaker_turns,
    read_track_16k_mono, segment_speaker_track,
};
use ort::ep::CoreML;
use ort::ep::coreml::ModelFormat;
use ort::logging::LogLevel;
use ort::session::Session;

/// The three literal labels ONNX Runtime's CoreML capability line is built from.
///
/// Read out of the linked `libonnxruntime` itself rather than from upstream source, because it is
/// the linked copy whose wording this tool depends on. The whole line reads:
///
/// ```text
/// CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: 1 \
///   number of nodes in the graph: 245 number of nodes supported by CoreML: 245
/// ```
const PARTITIONS_LABEL: &str = "number of partitions supported by CoreML:";
const NODES_LABEL: &str = "number of nodes in the graph:";
const SUPPORTED_LABEL: &str = "number of nodes supported by CoreML:";

/// The substring every line worth keeping from a successful load contains.
const COREML_MARKER: &str = "CoreMLExecutionProvider";

pub fn run() {
    let usage =
        "usage: coreml-configurations <session-dir | wav-file> [--cache-dir <path>] [--log]";
    let mut target: Option<PathBuf> = None;
    let mut cache_root: Option<PathBuf> = None;
    let mut mirror = false;

    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.to_str().unwrap_or_default() {
            "--log" => mirror = true,
            "--cache-dir" => {
                cache_root = Some(
                    rest.next()
                        .map(PathBuf::from)
                        .unwrap_or_else(|| fail("--cache-dir takes a path")),
                );
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

    // Before any session exists: the first session built with no committed options locks in the
    // default, logger-less environment for the whole process, and a `commit()` afterwards
    // silently does nothing.
    let capture = Capture::install(mirror);

    let track = if target.is_dir() {
        target.join("speaker.wav")
    } else {
        target
    };
    let audio = read_track_16k_mono(&track).unwrap_or_else(|e| fail(&format!("{e}")));

    let models = meethook_root().join("models");
    let segmentation = models.join(SEGMENTATION_MODEL.file_name);
    let embedding = models.join(EMBEDDING_MODEL.file_name);
    for model in [&segmentation, &embedding] {
        if !model.is_file() {
            fail(&format!(
                "{} is not installed\nrun `cargo run --example fetch-onnx-models` first",
                model.display()
            ));
        }
    }

    println!("track   {}", track.display());
    println!(
        "audio   {:.1} s",
        audio.len() as f64 / f64::from(meethook_transcribe::TARGET_RATE)
    );
    match &cache_root {
        Some(root) => println!(
            "cache   {} -- kept between runs, so the cold column may already be warm",
            root.display()
        ),
        None => println!("cache   a temporary directory per cell, removed on exit"),
    }

    // The first CoreML session in a process pays a one-off framework initialisation that belongs
    // to no configuration. Charging it to cell 1 would make the baseline look slow, so it is paid
    // here and printed -- it is a real number, part of what a single-shot `meethook transcribe`
    // pays today -- rather than quietly discarded.
    let started = Instant::now();
    match open(&segmentation, ModelFormat::NeuralNetwork, None) {
        Ok(session) => {
            let elapsed = started.elapsed();
            drop(session);
            println!(
                "warm-up {} loaded in {} (excluded from the grid)",
                SEGMENTATION_MODEL.file_name,
                secs(elapsed)
            );
        }
        Err(e) => println!("warm-up failed: {e}"),
    }
    capture.drain();

    let rig = Rig {
        audio: &audio,
        segmentation: &segmentation,
        embedding: &embedding,
        capture: &capture,
        cache_root: cache_root.as_deref(),
    };

    let mut baseline: Option<Pass> = None;
    let mut rows: Vec<Row> = Vec::new();
    for spec in GRID {
        println!();
        println!("=== {}/{}  {} ===", spec.index, GRID.len(), spec.label());
        match rig.run(spec) {
            Err(reason) => {
                println!("  refused: {reason}");
                rows.push(Row::refused(spec, &reason));
            }
            Ok(cell) => {
                for graph in &cell.graphs {
                    graph.print();
                }
                println!(
                    "  segment {}   cluster {}   total {}",
                    secs(cell.pass.segment),
                    secs(cell.pass.cluster),
                    secs(cell.pass.segment + cell.pass.cluster)
                );
                println!(
                    "  {} turns, {} clusters",
                    cell.pass.turns.len(),
                    cell.pass.clustering.clusters.len()
                );
                match &cell.control {
                    Control::NotRun => {}
                    Control::Deterministic => println!(
                        "  control: pass A and pass B agree, \
                         so a divergence below describes the configuration"
                    ),
                    Control::Diverged(detail) => {
                        println!(
                            "  control: pass A and pass B DISAGREE on the same sessions, \
                             so every divergence below is run-to-run noise"
                        );
                        print_divergence(detail);
                    }
                }
                let verdict = match &baseline {
                    None => Verdict::Baseline,
                    Some(base) => match first_divergence(base, &cell.pass) {
                        None => {
                            println!("  vs baseline: identical");
                            Verdict::Identical
                        }
                        Some(detail) => {
                            println!("  vs baseline: differs");
                            print_divergence(&detail);
                            Verdict::Differs
                        }
                    },
                };
                rows.push(Row::ran(spec, &cell, verdict));
                if baseline.is_none() {
                    baseline = Some(cell.pass);
                }
            }
        }
    }

    print_summary(&rows);
}

/// Everything a cell needs, bundled so that running one is a single call.
struct Rig<'a> {
    audio: &'a [f32],
    segmentation: &'a Path,
    embedding: &'a Path,
    capture: &'a Capture,
    cache_root: Option<&'a Path>,
}

/// One cell of the grid.
#[derive(Clone, Copy)]
struct CellSpec {
    index: usize,
    format: ModelFormat,
    cached: bool,
}

impl CellSpec {
    fn label(self) -> String {
        let format = match self.format {
            ModelFormat::NeuralNetwork => "NeuralNetwork",
            ModelFormat::MLProgram => "MLProgram",
        };
        let cache = if self.cached { "cache" } else { "no cache" };
        let note = if self.index == 1 { "  (baseline)" } else { "" };
        format!("{format}, {cache}{note}")
    }

    /// The cache directory's leaf name under `--cache-dir`. One per format, because the two
    /// formats' compiled artefacts are different things.
    fn cache_leaf(self) -> &'static str {
        match self.format {
            ModelFormat::NeuralNetwork => "nn",
            ModelFormat::MLProgram => "mlprogram",
        }
    }
}

/// Baseline first, so every later cell has something to compare against.
const GRID: [CellSpec; 4] = [
    CellSpec {
        index: 1,
        format: ModelFormat::NeuralNetwork,
        cached: false,
    },
    CellSpec {
        index: 2,
        format: ModelFormat::NeuralNetwork,
        cached: true,
    },
    CellSpec {
        index: 3,
        format: ModelFormat::MLProgram,
        cached: false,
    },
    CellSpec {
        index: 4,
        format: ModelFormat::MLProgram,
        cached: true,
    },
];

/// A finished cell: what each graph cost, and what the pass on them produced.
struct Cell {
    graphs: Vec<GraphReport>,
    pass: Pass,
    control: Control,
}

/// Whether the pipeline gave the same answer twice on the same sessions.
///
/// Run in cell 1 only. Without it the divergence column is a guess: a difference between two
/// cells only says something about the configuration if the pipeline is deterministic in the
/// first place.
enum Control {
    NotRun,
    Deterministic,
    Diverged(String),
}

impl Rig<'_> {
    /// Runs one cell, or returns why CoreML refused it.
    ///
    /// A refusal is a result, not a crash: the caller prints it and moves to the next cell. There
    /// is deliberately no CPU fallback here -- `open_session`'s retry is right for the library and
    /// wrong for this tool, where a cell quietly reporting CPU numbers under an `MLProgram`
    /// heading would be worse than no cell at all.
    fn run(&self, spec: CellSpec) -> Result<Cell, String> {
        // Held for the whole cell: dropping a `TempDir` deletes the directory, which would pull
        // the cache out from under the sessions still using it.
        let cache = self.cache_for(spec)?;

        // Segmentation first, then the embedder: the order the pass runs them in, so the load
        // column reads in the same direction as the wall clock below it.
        let (segmentation, mut segmenter) =
            self.load_twice(self.segmentation, spec, cache.path())?;
        let (embedding, mut embedder) = self.load_twice(self.embedding, spec, cache.path())?;
        let graphs = vec![segmentation, embedding];

        let pass = run_pass(self.audio, &mut segmenter, &mut embedder)?;
        self.capture.drain();

        let control = if spec.index == 1 {
            let second = run_pass(self.audio, &mut segmenter, &mut embedder)?;
            self.capture.drain();
            match first_divergence(&pass, &second) {
                None => Control::Deterministic,
                Some(detail) => Control::Diverged(detail),
            }
        } else {
            Control::NotRun
        };

        Ok(Cell {
            graphs,
            pass,
            control,
        })
    }

    /// Where this cell's compiled models go, if anywhere.
    fn cache_for(&self, spec: CellSpec) -> Result<CacheDir, String> {
        if !spec.cached {
            return Ok(CacheDir::None);
        }
        match self.cache_root {
            Some(root) => {
                let dir = root.join(spec.cache_leaf());
                std::fs::create_dir_all(&dir)
                    .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
                Ok(CacheDir::Persistent(dir))
            }
            // Its own directory, not one shared with the other cached cell: it makes this cell's
            // cold load unambiguously cold without depending on CoreML's on-disk layout keeping
            // the two formats apart.
            None => tempfile::TempDir::new()
                .map(CacheDir::Temporary)
                .map_err(|e| format!("could not create a temporary cache directory: {e}")),
        }
    }

    /// Loads one graph cold, then warm against the same cache, and hands back the warm session.
    ///
    /// The cold session is dropped before the warm one is timed, so the second load is a load and
    /// not a no-op. A cold load that succeeds and a warm one that fails is named rather than
    /// folded into a generic refusal -- that is the corrupt-cache failure mode, and seeing it
    /// here is free information.
    fn load_twice(
        &self,
        model: &Path,
        spec: CellSpec,
        cache: Option<&Path>,
    ) -> Result<(GraphReport, Session), String> {
        let file_name = model
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("<unnamed graph>")
            .to_string();

        let started = Instant::now();
        let cold_session =
            open(model, spec.format, cache).map_err(|e| format!("{file_name} cold load: {e}"))?;
        let cold = started.elapsed();
        drop(cold_session);
        let cold_lines = self.capture.drain();

        let started = Instant::now();
        let session = open(model, spec.format, cache)
            .map_err(|e| format!("{file_name} warm load, after a cold load that succeeded: {e}"))?;
        let warm = started.elapsed();
        let warm_lines = self.capture.drain();

        // The warm load's rounds are the ones reported: the warm sessions are what the pass runs
        // on, and the cold load's rounds are the same partitioning decided a second earlier.
        let mut capability_rounds = Vec::new();
        let mut unparsed = Vec::new();
        let mut notes = Vec::new();
        for (phase, lines) in [("cold", &cold_lines), ("warm", &warm_lines)] {
            for line in lines.iter() {
                if line.message.contains(PARTITIONS_LABEL) {
                    match capability(&line.message) {
                        Some(round) if phase == "warm" => capability_rounds.push(round),
                        Some(_) => {}
                        None => unparsed.push(format!("{phase}: {}", line.message)),
                    }
                } else {
                    notes.push(format!("{phase}: {:?}: {}", line.level, line.message));
                }
            }
        }

        Ok((
            GraphReport {
                file_name,
                cold,
                warm,
                capability: capability_rounds,
                unparsed,
                notes,
            },
            session,
        ))
    }
}

/// A cell's cache directory, and its lifetime.
enum CacheDir {
    None,
    /// Deleted when this is dropped, which is what keeps a default run from reading a cache an
    /// earlier run left or leaving one behind.
    Temporary(tempfile::TempDir),
    Persistent(PathBuf),
}

impl CacheDir {
    fn path(&self) -> Option<&Path> {
        match self {
            CacheDir::None => None,
            CacheDir::Temporary(dir) => Some(dir.path()),
            CacheDir::Persistent(dir) => Some(dir),
        }
    }
}

/// Builds one session under one configuration.
///
/// `with_log_level` is not optional and not only about raising the level: a custom environment
/// logger makes ONNX Runtime create the environment at VERBOSE, and a session with no severity of
/// its own inherits it. INFO both admits the capability line and caps everything above it.
fn open(model: &Path, format: ModelFormat, cache: Option<&Path>) -> ort::Result<Session> {
    let mut coreml = CoreML::default().with_model_format(format);
    if let Some(dir) = cache {
        coreml = coreml.with_model_cache_dir(dir.display());
    }
    Session::builder()?
        .with_log_level(LogLevel::Info)?
        .with_execution_providers([coreml.build()])?
        .commit_from_file(model)
}

/// What one graph cost, and what CoreML did with it.
struct GraphReport {
    file_name: String,
    cold: Duration,
    warm: Duration,
    /// Every capability round ONNX Runtime reported for the warm load, in order.
    capability: Vec<Capability>,
    /// Capability lines whose wording this tool could not parse. Printed raw rather than dropped:
    /// an upstream rewording must show up as an unfamiliar line, never as a silent "CoreML took
    /// nothing".
    unparsed: Vec<String>,
    /// Anything else CoreML or ONNX Runtime said, which is where a partial refusal explains
    /// itself.
    notes: Vec<String>,
}

impl GraphReport {
    fn print(&self) {
        println!("  {}", self.file_name);
        println!(
            "    load    cold {}   warm {}",
            secs(self.cold),
            secs(self.warm)
        );
        if self.capability.is_empty() && self.unparsed.is_empty() {
            println!("    coreml  no capability report -- CoreML took none of this graph");
        }
        let rounds = self.capability.len();
        for (i, capability) in self.capability.iter().enumerate() {
            let round = if rounds > 1 {
                format!("round {} ", i + 1)
            } else {
                String::new()
            };
            let final_marker = if rounds > 1 && i + 1 == rounds {
                "   (final)"
            } else {
                ""
            };
            let partitions = if capability.partitions == 1 {
                "1 partition".to_string()
            } else {
                format!("{} partitions", capability.partitions)
            };
            let taken = if capability.supported == 0 {
                "  -- CoreML took nothing".to_string()
            } else {
                String::new()
            };
            println!(
                "    coreml  {round}{partitions}, {} of {} nodes{taken}{final_marker}",
                capability.supported, capability.nodes
            );
        }
        for line in &self.unparsed {
            println!("    coreml  unrecognised capability line: {line}");
        }
        for note in &self.notes {
            println!("    note    {note}");
        }
    }
}

/// One `GetCapability` round: how ONNX Runtime split the graph, and how much CoreML took.
struct Capability {
    partitions: usize,
    nodes: usize,
    supported: usize,
}

/// Reads a capability line, or `None` if this is not one or its wording has changed.
fn capability(message: &str) -> Option<Capability> {
    let (_, rest) = message.split_once(PARTITIONS_LABEL)?;
    let (partitions, rest) = rest.split_once(NODES_LABEL)?;
    let (nodes, supported) = rest.split_once(SUPPORTED_LABEL)?;
    Some(Capability {
        partitions: partitions.trim().parse().ok()?,
        nodes: nodes.trim().parse().ok()?,
        supported: supported.trim().parse().ok()?,
    })
}

/// One diarization pass, and what it cost.
struct Pass {
    segment: Duration,
    cluster: Duration,
    turns: Vec<LocalTurn>,
    clustering: Clustering,
}

fn run_pass(
    audio: &[f32],
    segmenter: &mut Session,
    embedder: &mut Session,
) -> Result<Pass, String> {
    let started = Instant::now();
    let turns = segment_speaker_track(audio, segmenter).map_err(|e| format!("segmenting: {e}"))?;
    let segment = started.elapsed();

    let started = Instant::now();
    let clustering =
        cluster_speaker_turns(audio, &turns, embedder).map_err(|e| format!("clustering: {e}"))?;
    let cluster = started.elapsed();

    Ok(Pass {
        segment,
        cluster,
        turns,
        clustering,
    })
}

/// The first turn on which two passes disagree, named on both sides, or `None` if they agree.
///
/// Timings are compared exactly rather than within a tolerance: both are derived from integer
/// frame indices, so bit equality is meaningful, and printing the two values lets the reader judge
/// the size of a difference for themselves.
fn first_divergence(base: &Pass, other: &Pass) -> Option<String> {
    let shared = base.turns.len().min(other.turns.len());
    for index in 0..shared {
        let left = describe_turn(base, index);
        let right = describe_turn(other, index);
        if left != right {
            // "this     " is padded to the width of "baseline " so the two descriptions line up
            // under each other and the differing field is the one that catches the eye.
            let head = format!("turn {index} differs:");
            return Some(format!(
                "{head} baseline {left}\n{:pad$}this     {right}",
                "",
                pad = head.len() + 1
            ));
        }
    }
    if base.turns.len() != other.turns.len() {
        return Some(format!(
            "turn {shared} is past the end of the shorter side: baseline produced {} turns, this produced {}",
            base.turns.len(),
            other.turns.len()
        ));
    }
    None
}

fn describe_turn(pass: &Pass, index: usize) -> String {
    let turn = &pass.turns[index];
    let cluster = match pass.clustering.assignment.get(index).copied().flatten() {
        Some(id) => id.to_string(),
        None => "none".to_string(),
    };
    format!(
        "{:.2} -> {:.2} s window {} local {} cluster {cluster}",
        turn.start_s, turn.end_s, turn.window, turn.local_speaker
    )
}

fn print_divergence(detail: &str) {
    for line in detail.lines() {
        println!("    {line}");
    }
}

/// How a cell's turns compared with the baseline's.
#[derive(Clone, Copy)]
enum Verdict {
    Baseline,
    Identical,
    Differs,
}

impl Verdict {
    fn as_str(self) -> &'static str {
        match self {
            Verdict::Baseline => "baseline",
            Verdict::Identical => "identical",
            Verdict::Differs => "differs",
        }
    }
}

/// One line of the closing table, so the decision is readable without scrolling back.
struct Row {
    label: String,
    outcome: String,
}

impl Row {
    fn ran(spec: CellSpec, cell: &Cell, verdict: Verdict) -> Row {
        let cold: Duration = cell.graphs.iter().map(|g| g.cold).sum();
        let warm: Duration = cell.graphs.iter().map(|g| g.warm).sum();
        Row {
            label: format!("{}  {}", spec.index, spec.label()),
            outcome: format!(
                "{:>9} {:>9} {:>9} {:>9} {:>9} {:>7} {:>9}  {}",
                secs(cold),
                secs(warm),
                secs(cell.pass.segment),
                secs(cell.pass.cluster),
                secs(cell.pass.segment + cell.pass.cluster),
                cell.pass.turns.len(),
                cell.pass.clustering.clusters.len(),
                verdict.as_str()
            ),
        }
    }

    fn refused(spec: CellSpec, reason: &str) -> Row {
        Row {
            label: format!("{}  {}", spec.index, spec.label()),
            outcome: format!("refused: {reason}"),
        }
    }
}

fn print_summary(rows: &[Row]) {
    let header = "configuration";
    let width = rows
        .iter()
        .map(|row| row.label.len())
        .chain([header.len()])
        .max()
        .unwrap_or(0);
    println!();
    println!("=== summary ===");
    println!(
        "  {header:<width$} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7} {:>9}  vs baseline",
        "cold", "warm", "segment", "cluster", "total", "turns", "clusters"
    );
    for row in rows {
        println!("  {:<width$} {}", row.label, row.outcome);
    }
}

/// One captured ONNX Runtime line.
struct Line {
    level: LogLevel,
    message: String,
}

/// The process-wide log sink, installed on the ONNX Runtime environment before any session exists.
///
/// The environment logger rather than the per-session one, and not as a matter of taste:
/// `SessionBuilder::with_logger` keeps the closure in a builder field that is never moved into the
/// committed session, so the pointer ONNX Runtime holds dangles the moment the session is built.
/// The environment's builder is parked in a process-lifetime `OnceLock`, so its pointer stays
/// valid.
struct Capture {
    lines: Arc<Mutex<Vec<Line>>>,
    mirror: bool,
}

impl Capture {
    fn install(mirror: bool) -> Capture {
        let lines: Arc<Mutex<Vec<Line>>> = Arc::default();
        let sink = Arc::clone(&lines);
        // Called from ONNX Runtime's own threads across an `extern "system"` boundary, so it must
        // not panic and must not print: its whole job is to push a string and return. Filtering
        // happens here because INFO is still hundreds of lines per session.
        let committed = ort::init()
            .with_logger(Arc::new(
                move |level: LogLevel,
                      _category: &str,
                      _id: &str,
                      _location: &str,
                      message: &str| {
                    if !(mirror || interesting(level, message)) {
                        return;
                    }
                    if let Ok(mut lines) = sink.lock() {
                        lines.push(Line {
                            level,
                            message: message.to_string(),
                        });
                    }
                },
            ))
            .commit();
        if !committed {
            fail(
                "ort::init().commit() returned false: an ONNX Runtime environment already \
                 existed, so nothing would be captured and every coverage column would read \
                 as though CoreML took nothing",
            );
        }
        Capture { lines, mirror }
    }

    /// Takes everything logged since the last drain, mirroring it under `--log`.
    ///
    /// The main loop is single-threaded and drains between operations, so which operation a line
    /// belongs to is decided by when this is called.
    fn drain(&self) -> Vec<Line> {
        let mut held = match self.lines.lock() {
            Ok(held) => held,
            Err(poisoned) => poisoned.into_inner(),
        };
        let taken = std::mem::take(&mut *held);
        drop(held);
        if self.mirror {
            for line in &taken {
                eprintln!("[ort {:?}] {}", line.level, line.message);
            }
        }
        taken
            .into_iter()
            .filter(|line| interesting(line.level, &line.message))
            .collect()
    }
}

/// Whether a line belongs in the report rather than only in `--log`'s mirror.
fn interesting(level: LogLevel, message: &str) -> bool {
    level >= LogLevel::Error || message.contains(COREML_MARKER)
}

fn secs(duration: Duration) -> String {
    format!("{:.2} s", duration.as_secs_f64())
}

/// `$MEETHOOK_ROOT`, else `~/meethook`, resolved the way the sibling diagnostics resolve it.
fn meethook_root() -> PathBuf {
    match std::env::var_os("MEETHOOK_ROOT") {
        Some(root) => PathBuf::from(root),
        None => std::env::home_dir()
            .expect("could not determine the home directory; set MEETHOOK_ROOT")
            .join("meethook"),
    }
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}
