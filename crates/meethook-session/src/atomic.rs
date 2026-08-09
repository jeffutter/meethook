use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::{Error, Result};

/// Prefix for in-flight temp files. It starts with a dot and shares no stem with any real
/// session file, so a leftover temp file from a crash can never be mistaken for
/// `session.json` or `transcript.json` by the classifier.
const TMP_PREFIX: &str = ".meethook-tmp-";

/// Writes `bytes` to `path` so that a reader sees either the previous contents or the
/// complete new contents, never a partial write.
///
/// Byte-oriented rather than JSON-oriented on purpose: `transcript.md` needs the same
/// guarantee, and a second near-identical writer is exactly the duplication this crate
/// exists to avoid.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    write_atomic_with(path, |file| {
        file.write_all(bytes).map_err(|e| Error::io(path, e))
    })
}

/// The same all-or-nothing guarantee as [`write_atomic`], for content too large to hand
/// over as a `&[u8]`.
///
/// `fill` is given an empty file positioned at its start and may write and seek freely; an
/// hour of `mic.cleaned.wav` is ~230 MB, so buffering it in memory just to reuse the byte
/// API would be a waste this exists to avoid. The file is not visible under `path` until
/// `fill` returns `Ok`, so a `fill` that fails partway leaves whatever was already there
/// untouched.
///
/// The error type is the caller's, because the caller's failures are its own: hound's WAV
/// errors have nothing to do with this crate. Any I/O failure in the atomic machinery
/// itself arrives as this crate's [`Error`], hence the `From` bound.
///
/// The `sync_all` before `persist` is load-bearing. `NamedTempFile::persist` renames but
/// does not flush, so without it a crash between rename and writeback can leave a
/// zero-length `session.json` -- i.e. a corrupt session that classifies as valid. The
/// second `sync_all`, on the parent directory, makes the rename itself durable.
pub fn write_atomic_with<E, F>(path: &Path, fill: F) -> std::result::Result<(), E>
where
    F: FnOnce(&mut File) -> std::result::Result<(), E>,
    E: From<Error>,
{
    let parent = path.parent().unwrap_or_else(|| Path::new("."));

    // Same directory, therefore same filesystem, therefore `rename` is atomic.
    let mut tmp = tempfile::Builder::new()
        .prefix(TMP_PREFIX)
        .tempfile_in(parent)
        .map_err(|e| Error::io(parent, e))?;

    fill(tmp.as_file_mut())?;
    tmp.as_file()
        .sync_all()
        .map_err(|e| Error::io(tmp.path(), e))?;
    tmp.persist(path).map_err(|e| Error::io(path, e.error))?;

    File::open(parent)
        .and_then(|dir| dir.sync_all())
        .map_err(|e| Error::io(parent, e))?;

    Ok(())
}
