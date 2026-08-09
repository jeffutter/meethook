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
///
/// The `sync_all` before `persist` is load-bearing. `NamedTempFile::persist` renames but
/// does not flush, so without it a crash between rename and writeback can leave a
/// zero-length `session.json` -- i.e. a corrupt session that classifies as valid. The
/// second `sync_all`, on the parent directory, makes the rename itself durable.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));

    // Same directory, therefore same filesystem, therefore `rename` is atomic.
    let mut tmp = tempfile::Builder::new()
        .prefix(TMP_PREFIX)
        .tempfile_in(parent)
        .map_err(|e| Error::io(parent, e))?;

    tmp.write_all(bytes).map_err(|e| Error::io(tmp.path(), e))?;
    tmp.as_file()
        .sync_all()
        .map_err(|e| Error::io(tmp.path(), e))?;
    tmp.persist(path).map_err(|e| Error::io(path, e.error))?;

    File::open(parent)
        .and_then(|dir| dir.sync_all())
        .map_err(|e| Error::io(parent, e))?;

    Ok(())
}
