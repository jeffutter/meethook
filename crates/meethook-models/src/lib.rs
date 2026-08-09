//! Lazy, hash-verified acquisition of model weights.
//!
//! Model weights are deliberately not part of the Nix closure: they are hundreds of
//! megabytes to gigabytes of data that change on a different schedule from the code, and
//! baking them into the build environment would make every `nix develop` pay for them.
//! Instead they are fetched on first need into the meethook data directory and checked
//! against a sha256 embedded in source, so "not in the closure" never becomes "whatever
//! bytes the network happened to hand us".
//!
//! This crate knows nothing about whisper, ONNX, sessions, or transcripts. It takes a
//! [`ModelSpec`] and returns a path to a verified local copy -- which is why `transcribe`
//! and `enroll` can both use it without either depending on the other.
//!
//! Nothing here reads stdin or prints. Acquisition happens inside batch commands that must
//! never prompt, so all user-facing output goes through the caller's progress callback.

use std::fmt::Write as _;
use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Read/write chunk size for a download. Large enough that a multi-gigabyte transfer is not
/// dominated by syscall overhead, small enough that progress still looks continuous.
const CHUNK_BYTES: usize = 1 << 20;

/// Prefix for the in-flight download. Distinct from the real file name so a temp file left
/// by a killed process can never be mistaken for an installed model.
const TMP_PREFIX: &str = ".meethook-download-";

/// One model file, described by exactly what is needed to fetch it and prove it arrived
/// intact.
///
/// `url` must point at an immutable revision rather than a moving branch pointer: a
/// repository that republished its weights would otherwise turn a working install into a
/// hash mismatch the user can do nothing about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    /// File name inside the models directory. Also the cache key.
    pub file_name: &'static str,
    pub url: &'static str,
    /// Lowercase hex sha256 of the complete file.
    pub sha256: &'static str,
    pub size_bytes: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("i/o error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("could not download {file_name} from {url}: {source}")]
    Download {
        file_name: &'static str,
        url: &'static str,
        #[source]
        source: Box<ureq::Error>,
    },

    #[error(
        "{file_name} failed verification and was discarded: expected sha256 {expected_sha256} \
         ({expected_bytes} bytes), got {actual_sha256} ({actual_bytes} bytes)"
    )]
    Corrupt {
        file_name: &'static str,
        expected_sha256: &'static str,
        actual_sha256: String,
        expected_bytes: u64,
        actual_bytes: u64,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}

/// Returns a path to a verified local copy of `spec`, downloading it if it is not already
/// installed.
///
/// `progress` is called with `(bytes_so_far, total_bytes)` while downloading and not at all
/// on a cache hit, so a caller can print a progress line without first having to ask
/// whether a download is going to happen.
///
/// A cache hit is decided by name and byte length rather than by re-hashing. The realistic
/// failure mode for an installed file is a truncated download, which the length catches;
/// re-reading gigabytes on every invocation would cost seconds of every run to defend
/// against bit rot the filesystem already checksums.
pub fn ensure_model(
    models_dir: &Path,
    spec: &ModelSpec,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<PathBuf> {
    let dest = models_dir.join(spec.file_name);

    match fs::metadata(&dest) {
        Ok(meta) if meta.is_file() && meta.len() == spec.size_bytes => return Ok(dest),
        Ok(_) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => return Err(Error::io(&dest, e)),
    }

    fs::create_dir_all(models_dir).map_err(|e| Error::io(models_dir, e))?;

    let mut request = ureq::get(spec.url);
    // A bandwidth allowance, nothing more: every model meethook uses is publicly ungated,
    // so an absent token is not an error and there is no gating branch to write. The token
    // appears here and nowhere else -- never in a log line, a progress message, or an error.
    if let Some(token) = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()) {
        request = request.header("Authorization", &format!("Bearer {token}"));
    }

    let response = request.call().map_err(|e| Error::Download {
        file_name: spec.file_name,
        url: spec.url,
        source: Box::new(e),
    })?;

    install_verified(response.into_body().into_reader(), &dest, spec, progress)?;
    Ok(dest)
}

/// Streams `source` into a temp file beside `dest` and promotes it only if it matches
/// `spec`.
///
/// Split out from [`ensure_model`] so the part with consequences -- verify, then promote or
/// discard -- is testable without a network.
///
/// The bytes are hashed as they are written, so even a multi-gigabyte file is read once and
/// never read back. Every failure path removes the temp file first: a partial download left
/// under the real name would be indistinguishable from a finished one on the next run.
pub fn install_verified(
    mut source: impl Read,
    dest: &Path,
    spec: &ModelSpec,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<()> {
    let parent = dest.parent().unwrap_or_else(|| Path::new("."));
    let tmp = parent.join(format!("{TMP_PREFIX}{}", spec.file_name));

    let (digest, written) = match stream_and_hash(&mut source, &tmp, spec, progress) {
        Ok(pair) => pair,
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            return Err(e);
        }
    };

    if digest != spec.sha256 || written != spec.size_bytes {
        let _ = fs::remove_file(&tmp);
        return Err(Error::Corrupt {
            file_name: spec.file_name,
            expected_sha256: spec.sha256,
            actual_sha256: digest,
            expected_bytes: spec.size_bytes,
            actual_bytes: written,
        });
    }

    // Same directory, therefore same filesystem, therefore an atomic rename: a concurrent
    // second invocation sees either no file or the complete, verified one.
    fs::rename(&tmp, dest).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        Error::io(dest, e)
    })
}

/// Writes everything `source` yields to `tmp`, returning the hex digest and byte count of
/// what was written.
fn stream_and_hash(
    source: &mut impl Read,
    tmp: &Path,
    spec: &ModelSpec,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<(String, u64)> {
    let mut file = fs::File::create(tmp).map_err(|e| Error::io(tmp, e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK_BYTES];
    let mut written: u64 = 0;

    loop {
        let n = source.read(&mut buf).map_err(|e| Error::io(tmp, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n]).map_err(|e| Error::io(tmp, e))?;
        written += n as u64;
        progress(written, spec.size_bytes);
    }

    file.sync_all().map_err(|e| Error::io(tmp, e))?;
    Ok((hex(&hasher.finalize()), written))
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const PAYLOAD: &[u8] = b"a tiny stand-in for a very large model";

    /// A spec describing [`PAYLOAD`]. The URL is `.invalid`, which RFC 2606 guarantees
    /// never resolves, so any test that accidentally reaches the network fails fast rather
    /// than depending on one.
    const SPEC: ModelSpec = ModelSpec {
        file_name: "test-model.bin",
        url: "https://meethook.invalid/test-model.bin",
        sha256: "03c94e38d98c7e2d5a60093889bf2eb4229fbb268145a53f7cbea865f1a7d060",
        size_bytes: PAYLOAD.len() as u64,
    };

    fn ignore(_: u64, _: u64) {}

    #[test]
    fn the_declared_digest_describes_the_test_payload() {
        assert_eq!(hex(&Sha256::digest(PAYLOAD)), SPEC.sha256);
    }

    #[test]
    fn a_matching_payload_is_promoted_and_leaves_no_temp_file() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(SPEC.file_name);

        install_verified(Cursor::new(PAYLOAD), &dest, &SPEC, &mut ignore).unwrap();

        assert_eq!(fs::read(&dest).unwrap(), PAYLOAD);
        assert!(temp_files(dir.path()).is_empty());
    }

    /// Right length, wrong bytes -- which the length check alone would wave through. This
    /// is the case the hash exists for.
    #[test]
    fn a_corrupt_payload_fails_loudly_and_installs_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(SPEC.file_name);
        let corrupted: Vec<u8> = PAYLOAD.iter().map(|b| b ^ 0x01).collect();

        let err = install_verified(Cursor::new(corrupted), &dest, &SPEC, &mut ignore).unwrap_err();

        let message = err.to_string();
        assert!(
            message.contains(SPEC.sha256),
            "expected-vs-actual missing: {message}"
        );
        assert!(!dest.exists(), "a failed download must not be installed");
        assert!(temp_files(dir.path()).is_empty());
    }

    #[test]
    fn a_truncated_payload_fails_even_though_it_is_well_formed() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(SPEC.file_name);

        let err =
            install_verified(Cursor::new(&PAYLOAD[..8]), &dest, &SPEC, &mut ignore).unwrap_err();

        assert!(err.to_string().contains("got"), "{err}");
        assert!(!dest.exists());
        assert!(temp_files(dir.path()).is_empty());
    }

    /// A cache hit must not touch the network. A progress callback that panics is what
    /// proves the download path was never entered, rather than merely that it succeeded.
    #[test]
    fn a_correctly_sized_cached_file_short_circuits_before_any_download() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join(SPEC.file_name);
        fs::write(&dest, PAYLOAD).unwrap();

        let mut progress = |_: u64, _: u64| panic!("a cache hit must not report progress");
        let found = ensure_model(dir.path(), &SPEC, &mut progress).unwrap();

        assert_eq!(found, dest);
    }

    #[test]
    fn a_wrong_length_cached_file_is_not_treated_as_a_hit() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join(SPEC.file_name), &PAYLOAD[..4]).unwrap();

        // Getting past the length check means attempting a real fetch, which cannot succeed
        // against `.invalid`; a download error therefore proves the short file was rejected.
        let err = ensure_model(dir.path(), &SPEC, &mut ignore).unwrap_err();
        assert!(matches!(err, Error::Download { .. }), "{err}");
    }

    fn temp_files(dir: &Path) -> Vec<String> {
        fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with(TMP_PREFIX))
            .collect()
    }
}
