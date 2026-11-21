//! The MLX shaders, embedded in the binary and unpacked on first use.
//!
//! MLX will not start its GPU stack without a `mlx.metallib` file, and it can only read one from a
//! path: it has no way to load the library from memory. To keep the release a single file, the
//! build script (`build.rs`) compresses the metallib it just built with zstd and this crate embeds
//! the result. [`install`] writes it out once, as `mlx-<id>.metallib` beside the model weights, and
//! returns the path for `mlx_rs::metal::set_metallib_path`, which MLX consults before its own
//! lookup. Nothing needs to ship next to the executable.
//!
//! A cache file is reused when its length is the expected one. It is written to a temporary name
//! and renamed into place, so a crash cannot leave a short file under the final name, and the
//! length check is only there to catch a file someone truncated. Shaders from older builds (other
//! ids) are removed after a successful unpack, so updates do not leave 136 MB each behind.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

static COMPRESSED: &[u8] = include_bytes!(env!("MLX_METALLIB_ZST"));
const LENGTH: u64 = match u64::from_str_radix(env!("MLX_METALLIB_LEN"), 10) {
    Ok(length) => length,
    Err(_) => panic!("MLX_METALLIB_LEN is not a number"),
};
const ID: &str = env!("MLX_METALLIB_ID");

/// Unpacks the embedded shaders into `dir` if they are not already there, and returns the file.
pub(crate) fn install(dir: &Path) -> io::Result<PathBuf> {
    unpack(dir, COMPRESSED, ID, LENGTH)
}

/// [`install`] for any blob, so the behaviour can be tested without the 136 MB one.
fn unpack(dir: &Path, compressed: &[u8], id: &str, length: u64) -> io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("mlx-{id}.metallib"));
    if path.metadata().is_ok_and(|m| m.len() == length) {
        return Ok(path);
    }

    // The process id keeps two meethooks starting together from writing the same temporary file;
    // whichever renames last wins, and the content is identical.
    let temporary = dir.join(format!("mlx-{id}.metallib.{}.tmp", std::process::id()));
    let written = (|| {
        let mut file = File::create(&temporary)?;
        zstd::stream::copy_decode(compressed, &mut file)?;
        file.sync_all()?;
        let actual = file.metadata()?.len();
        if actual != length {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("the embedded shaders unpack to {actual} bytes, expected {length}"),
            ));
        }
        fs::rename(&temporary, &path)
    })();
    if let Err(e) = written {
        let _ = fs::remove_file(&temporary);
        return Err(e);
    }

    remove_other_versions(dir, &path);
    Ok(path)
}

/// Best effort: deletes `mlx-<16 hex>.metallib` files other than `keep`. A process that has one
/// open keeps reading it (the file stays until closed), and one that starts later unpacks its own.
fn remove_other_versions(dir: &Path, keep: &Path) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let is_a_cache_file = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix("mlx-"))
            .and_then(|name| name.strip_suffix(".metallib"))
            .is_some_and(|id| id.len() == 16 && id.bytes().all(|b| b.is_ascii_hexdigit()));
        if is_a_cache_file && path != keep {
            let _ = fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 MB of compressible bytes standing in for the real shaders.
    fn shaders() -> Vec<u8> {
        (0..3_000_000u32).map(|i| (i % 251) as u8).collect()
    }

    fn compressed(raw: &[u8]) -> Vec<u8> {
        zstd::stream::encode_all(raw, 3).unwrap()
    }

    #[test]
    fn the_embedded_blob_is_the_size_the_build_script_recorded() {
        // The real one, which is why this runs only in builds that have it: the compressed bytes
        // are far smaller than the raw length, and the id is a 64-bit hex string.
        assert!(
            COMPRESSED.len() < LENGTH as usize / 4,
            "{}",
            COMPRESSED.len()
        );
        assert_eq!(ID.len(), 16);
        assert!(ID.bytes().all(|b| b.is_ascii_hexdigit()));
        let unpacked = zstd::stream::decode_all(COMPRESSED).unwrap();
        assert_eq!(unpacked.len() as u64, LENGTH);
    }

    #[test]
    fn unpacking_writes_the_shaders_and_a_second_call_reuses_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let raw = shaders();
        let path = unpack(
            dir.path(),
            &compressed(&raw),
            "00000000000000aa",
            raw.len() as u64,
        )
        .unwrap();
        assert_eq!(path.file_name().unwrap(), "mlx-00000000000000aa.metallib");
        assert_eq!(fs::read(&path).unwrap(), raw);

        let modified = path.metadata().unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let again = unpack(
            dir.path(),
            &compressed(&raw),
            "00000000000000aa",
            raw.len() as u64,
        )
        .unwrap();
        assert_eq!(again, path);
        assert_eq!(
            path.metadata().unwrap().modified().unwrap(),
            modified,
            "not rewritten"
        );
    }

    #[test]
    fn a_truncated_cache_file_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let raw = shaders();
        let path = dir.path().join("mlx-00000000000000aa.metallib");
        fs::write(&path, b"short").unwrap();
        unpack(
            dir.path(),
            &compressed(&raw),
            "00000000000000aa",
            raw.len() as u64,
        )
        .unwrap();
        assert_eq!(fs::read(&path).unwrap(), raw);
    }

    #[test]
    fn a_blob_that_unpacks_to_the_wrong_length_leaves_nothing_behind() {
        let dir = tempfile::tempdir().unwrap();
        let raw = shaders();
        let error = unpack(dir.path(), &compressed(&raw), "00000000000000aa", 7).unwrap_err();
        assert!(error.to_string().contains("expected 7"), "{error}");
        assert_eq!(
            fs::read_dir(dir.path()).unwrap().count(),
            0,
            "no final or temporary file"
        );
    }

    #[test]
    fn a_corrupt_blob_is_an_error_and_leaves_nothing_behind() {
        let dir = tempfile::tempdir().unwrap();
        assert!(unpack(dir.path(), b"not zstd at all", "00000000000000aa", 15).is_err());
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn older_versions_are_removed_and_unrelated_files_are_not() {
        let dir = tempfile::tempdir().unwrap();
        let old = dir.path().join("mlx-1111111111111111.metallib");
        let unrelated = dir.path().join("mlx-notes.metallib");
        let model = dir.path().join("voice-isolator-mlx-fp16.safetensors");
        for file in [&old, &unrelated, &model] {
            fs::write(file, b"x").unwrap();
        }
        let raw = shaders();
        let path = unpack(
            dir.path(),
            &compressed(&raw),
            "00000000000000aa",
            raw.len() as u64,
        )
        .unwrap();
        assert!(!old.exists(), "an older shader version is deleted");
        assert!(unrelated.exists() && model.exists() && path.exists());
    }

    #[test]
    fn an_unwritable_directory_is_an_error_not_a_panic() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o500)).unwrap();
        let raw = shaders();
        let result = unpack(
            dir.path(),
            &compressed(&raw),
            "00000000000000aa",
            raw.len() as u64,
        );
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o700)).unwrap();
        assert!(result.is_err());
    }
}
