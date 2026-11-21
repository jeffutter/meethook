//! Opening ONNX Runtime sessions.
//!
//! Two graphs run through here -- pyannote segmentation and the WeSpeaker embedder -- and
//! both want the same thing: CoreML where the ops allow it (macOS), CPU everywhere else.
//! That policy lives in one function rather than at each call site, so a future change to
//! how this machine is driven is one edit rather than two.
//!
//! The ONNX Runtime itself is not vendored. `ort-sys` probes pkg-config for `libonnxruntime` and
//! links that when it finds one: the flake's copy in the sandboxed `nix build` and on Linux, a
//! package such as `libonnxruntime-dev` elsewhere. When it finds none (the macOS dev shell, which
//! has none on purpose) it downloads pyke's hash-pinned prebuilt and links it statically, which is
//! what makes the macOS release a single file; see the `ort` entry in the workspace `Cargo.toml`
//! and MLX.md for why that path was chosen over the (now deleted) dynamic-load one.

use std::path::Path;

#[cfg(target_os = "macos")]
use ort::ep::CoreML;
use ort::session::Session;

use crate::{Error, Result};

/// A loaded graph, and whether it got the accelerator.
///
/// The flag is not decoration. A CPU-only load is perfectly usable and several times
/// slower, so a caller that reports nothing leaves the user with a diarization pass that
/// mysteriously takes minutes instead of seconds. Reporting is the caller's to do -- loading
/// happens inside a batch command whose per-session output must not be interleaved with
/// chatter from here, so this function prints on no path at all.
pub struct Loaded {
    pub session: Session,
    /// False when the whole graph is running on CPU because CoreML would not take it.
    pub accelerated: bool,
}

/// One ort environment for the process. rc.15 made it an explicit argument to
/// `Session::builder`, and it owns the runtime's global state, so building one per session
/// would be wasteful. A failed build is not cached: the next call retries.
fn environment() -> ort::Result<&'static ort::environment::Environment> {
    static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();
    if let Some(env) = ENV.get() {
        return Ok(env);
    }
    let env = ort::init().build()?;
    Ok(ENV.get_or_init(|| env))
}

/// Loads an ONNX graph. On macOS it prefers CoreML and falls back to CPU; on other
/// platforms there is no accelerator compiled into the runtime, so it goes straight to CPU.
///
/// On macOS the fallback is a real retry, not `error_on_failure(false)`. That flag governs
/// only whether *registering* the provider may fail; the interesting failures happen later,
/// when ONNX Runtime hands a partition to CoreML and Apple's compiler refuses it -- which
/// surfaces from `commit_from_file` as a hard error and takes the whole load down with it.
/// pyannote's segmentation graph is exactly the kind of model that provokes this, so the
/// second attempt is what keeps a working slow path from becoming no path at all.
///
/// Nothing about the successful CoreML path is assumed to be all-or-nothing: ONNX Runtime
/// partitions the graph and runs the ops CoreML declines on CPU regardless. `accelerated`
/// distinguishes "some of this is on the ANE" from "none of it could be". Off macOS it is
/// always false, and callers word their notes accordingly rather than implying a decline.
pub fn open_session(model_path: &Path) -> Result<Loaded> {
    // Split out so the several ort failures -- environment, provider registration, model
    // parse, CoreML compile -- collapse into one `?`-chain per attempt.
    #[cfg(target_os = "macos")]
    fn load_with_coreml(model_path: &Path) -> ort::Result<Session> {
        Session::builder(environment()?)?
            .with_execution_providers([CoreML::default().build()])?
            .commit_from_file(model_path)
    }

    #[cfg(target_os = "macos")]
    if let Ok(session) = load_with_coreml(model_path) {
        return Ok(Loaded {
            session,
            accelerated: true,
        });
    }

    // Only the CPU attempt's error is reported. A CoreML error here would name a compiler
    // that the user cannot act on, in place of the parse or permission failure that is the
    // actual reason nothing loaded.
    open_cpu_session(model_path).map(|session| Loaded {
        session,
        accelerated: false,
    })
}

/// Loads an ONNX graph on the CPU, whatever accelerators the platform has.
///
/// For graphs where an accelerator is a loss rather than a missing nicety: the speech
/// detector is a 2 MB network called once per 0.5 s of audio with a state tensor passed in and
/// out each time, and a CoreML partition boundary on every call would cost more than the
/// network does.
pub fn open_cpu_session(model_path: &Path) -> Result<Session> {
    open_cpu_session_with_threads(model_path, None)
}

/// [`open_cpu_session`] with an intra-op thread count (`None` is ONNX Runtime's default, every
/// logical core). For a large graph, like the voice isolator, that is dominated by matrix
/// work: on an M1 Pro the default 10 threads include the 2 efficiency cores, and the
/// performance cores wait on them at every operator. 8 threads measured about 20% faster
/// (30-33 s against 40 s for the same two windows).
pub fn open_cpu_session_with_threads(model_path: &Path, threads: Option<usize>) -> Result<Session> {
    fn load(model_path: &Path, threads: Option<usize>) -> ort::Result<Session> {
        let mut builder = Session::builder(environment()?)?;
        if let Some(n) = threads {
            builder = builder.with_intra_threads(n)?;
        }
        builder.commit_from_file(model_path)
    }
    load(model_path, threads).map_err(|source| Error::Onnx {
        path: model_path.to_path_buf(),
        source: Box::new(source),
    })
}

/// Cores worth giving to a throughput-bound graph: the performance cores on Apple silicon,
/// else `None` (let ONNX Runtime decide).
pub fn performance_cores() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        let mut value: libc::c_int = 0;
        let mut size = std::mem::size_of::<libc::c_int>();
        let name = c"hw.perflevel0.physicalcpu";
        // SAFETY: `name` is NUL-terminated and `value`/`size` describe a c_int.
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                (&mut value as *mut libc::c_int).cast(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        return (rc == 0 && value > 0).then_some(value as usize);
    }
    #[allow(unreachable_code)]
    None
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ort::value::{TensorElementType, ValueType};

    use super::*;
    use crate::{EMBEDDING_MODEL, SEGMENTATION_MODEL};
    use meethook_models::ModelSpec;

    /// One end of a graph: the name, element type and rank the rest of this slice decodes
    /// against. Concrete dimensions are deliberately absent -- every axis these two graphs
    /// care about is symbolic (batch, samples, frames), so pinning numbers would pin
    /// nothing real. The one fixed extent that *is* meaningful (7 powerset classes, 256
    /// embedding dimensions) is checked separately below.
    struct Contract {
        name: &'static str,
        rank: usize,
    }

    const SEGMENTATION_INPUT: Contract = Contract {
        name: "input_values",
        rank: 3,
    };
    const SEGMENTATION_OUTPUT: Contract = Contract {
        name: "logits",
        rank: 3,
    };
    const EMBEDDING_INPUT: Contract = Contract {
        name: "feats",
        rank: 3,
    };
    const EMBEDDING_OUTPUT: Contract = Contract {
        name: "embs",
        rank: 2,
    };

    /// Where `meethook transcribe` would have put the weights. Resolved the same way the
    /// CLI resolves it so a developer who has run the tool once already has the files.
    fn models_dir() -> Option<PathBuf> {
        let root = match std::env::var_os("MEETHOOK_ROOT") {
            Some(root) => PathBuf::from(root),
            None => std::env::home_dir()?.join("meethook"),
        };
        Some(root.join("models"))
    }

    /// Opens `spec`'s weights, or returns `None` if they are not installed.
    ///
    /// Returning rather than downloading is the point: these two files are 32 MB, and a
    /// `cargo test` that silently reaches for them fails on a plane, in CI, and on a machine
    /// that has never run the tool -- none of which are the failure this test exists to
    /// catch.
    fn open_if_installed(spec: &ModelSpec) -> Option<Session> {
        let path = models_dir()?.join(spec.file_name);
        if !path.is_file() {
            eprintln!(
                "skipping: {} is not installed; \
                 run `cargo run --example fetch-onnx-models` to fetch it",
                path.display()
            );
            return None;
        }
        let loaded = open_session(&path).expect("an installed model must load");
        if !loaded.accelerated {
            // Not a failure -- the fallback working is the point -- but worth saying out
            // loud, because on a normal machine this line should not appear.
            eprintln!(
                "{}: CoreML declined this graph; running on CPU",
                spec.file_name
            );
        }
        Some(loaded.session)
    }

    /// Asserts one outlet matches its contract, returning the shape so a caller can check
    /// the fixed extents that matter to it.
    fn check(outlet: &ort::value::Outlet, contract: &Contract) -> Vec<i64> {
        assert_eq!(outlet.name(), contract.name);
        let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
            panic!("{} is not a tensor: {:?}", contract.name, outlet.dtype());
        };
        assert_eq!(
            *ty,
            TensorElementType::Float32,
            "{} element type",
            contract.name
        );
        assert_eq!(shape.len(), contract.rank, "{} rank", contract.name);
        shape.to_vec()
    }

    /// The graph contract both sibling slices decode against. Cheapest possible place to
    /// pin it: a rename or a re-export upstream shows up here rather than as a confusing
    /// tensor-shape error three layers into inference.
    #[test]
    fn the_segmentation_graph_takes_raw_audio_and_emits_powerset_logits() {
        let Some(session) = open_if_installed(&SEGMENTATION_MODEL) else {
            return;
        };

        assert_eq!(session.inputs().len(), 1);
        assert_eq!(session.outputs().len(), 1);
        check(&session.inputs()[0], &SEGMENTATION_INPUT);
        let logits = check(&session.outputs()[0], &SEGMENTATION_OUTPUT);

        // Silence, three single speakers, three pairs. Powerset decoding in the sibling
        // slice reads these seven columns positionally.
        assert_eq!(logits[2], 7, "powerset classes");
    }

    /// The Nemotron graph finds its weights by the literal name `model_fp16.onnx_data` beside
    /// it, so a successful load of the installed pair proves the reference resolves.
    #[test]
    fn the_installed_nemotron_pair_loads() {
        let Some(dir) = models_dir() else { return };
        let all_there = crate::NEMOTRON_DIARIZATION_MODELS
            .iter()
            .all(|s| dir.join(s.file_name).is_file());
        if !all_there {
            eprintln!("skipping: Nemotron pair not installed; run fetch-onnx-models");
            return;
        }
        let graph = dir.join(crate::NEMOTRON_DIARIZATION_GRAPH.file_name);
        let loaded = open_session(&graph).expect("the installed Nemotron pair must load");
        assert!(!loaded.session.inputs().is_empty());
    }

    #[test]
    fn the_embedding_graph_takes_fbank_features_and_emits_256_dimensions() {
        let Some(session) = open_if_installed(&EMBEDDING_MODEL) else {
            return;
        };

        assert_eq!(session.inputs().len(), 1);
        assert_eq!(session.outputs().len(), 1);
        let feats = check(&session.inputs()[0], &EMBEDDING_INPUT);
        let embs = check(&session.outputs()[0], &EMBEDDING_OUTPUT);

        assert_eq!(feats[2], 80, "mel bins per frame");
        assert_eq!(embs[1], 256, "embedding dimensions");
    }
}
