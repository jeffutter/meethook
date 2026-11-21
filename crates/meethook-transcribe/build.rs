//! Defines the `mlx_engine` cfg and, when it is on, embeds the compiled MLX shaders.
//!
//! The voice isolator's MLX engine is compiled in when the `mlx` feature (on by default) is on
//! *and* the target is macOS, because the `mlx-rs` dependency is macOS-only. The feature alone is
//! not enough: `--all-features` on Linux turns it on, and the code behind it must still vanish
//! there. Spelling the pair out at every use is how a bare `cfg(feature = "mlx")` slipped into an
//! example and broke Linux CI; use `mlx_engine`.
//!
//! # The embedded shaders
//!
//! MLX loads its precompiled Metal kernels from a file, `mlx.metallib`, and will not start its GPU
//! stack without one. To make the release a single file, this script finds the metallib `mlx-sys`
//! just built, compresses it with zstd (136 MB to about 10 MB) and exposes the result to the crate:
//!
//! - `MLX_METALLIB_ZST`: path of the compressed bytes, for `include_bytes!`;
//! - `MLX_METALLIB_LEN`: the uncompressed length in bytes;
//! - `MLX_METALLIB_ID`: a content hash, which names the unpacked cache file.
//!
//! `mlx-sys` does not publish the metallib's location. It does publish `DEP_MLX_INCLUDE` to its
//! direct dependents, a path list whose first entry is `<cmake out>/build/_deps/mlx-src`, and the
//! finished file is `<cmake out>/build/lib/mlx.metallib` (the same bytes as the copy under
//! `~/.mlx/lib/<key>/`). The layout is `mlx-sys`'s, not a contract, so a mismatch stops the build
//! with a message that says what to look at. See MLX.md.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(mlx_engine)");
    println!("cargo::rerun-if-changed=build.rs");
    let feature = std::env::var_os("CARGO_FEATURE_MLX").is_some();
    let macos = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos");
    if feature && macos {
        println!("cargo::rustc-cfg=mlx_engine");
        embed_metallib();
    }
}

#[cfg(all(target_os = "macos", feature = "mlx"))]
fn embed_metallib() {
    use std::path::PathBuf;

    let include = std::env::var_os("DEP_MLX_INCLUDE").expect(
        "DEP_MLX_INCLUDE is not set: meethook-transcribe must depend on mlx-sys directly \
         (see MLX.md)",
    );
    let mlx_src = std::env::split_paths(&include)
        .next()
        .expect("DEP_MLX_INCLUDE is empty");
    // <cmake out>/build/_deps/mlx-src -> <cmake out>
    let cmake_out = mlx_src
        .ancestors()
        .nth(3)
        .expect("DEP_MLX_INCLUDE is shorter than mlx-sys's layout");
    let metallib: PathBuf = cmake_out.join("build/lib/mlx.metallib");
    println!("cargo::rerun-if-changed={}", metallib.display());

    let raw = std::fs::read(&metallib).unwrap_or_else(|e| {
        panic!(
            "cannot read {} ({e}): mlx-sys's output layout may have changed; DEP_MLX_INCLUDE \
             starts with {}. See MLX.md.",
            metallib.display(),
            mlx_src.display()
        )
    });
    assert!(
        raw.len() > 1 << 20,
        "{} is only {} bytes; the MLX shaders did not build",
        metallib.display(),
        raw.len()
    );

    // FNV-1a 64: a name for the cache file, not a security boundary.
    let id = raw.iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    });
    // Level 19: 4 s here, 9.9 MB, and decompression stays about 0.05 s.
    let compressed = zstd::bulk::compress(&raw, 19).expect("zstd cannot compress mlx.metallib");

    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let zst = out.join("mlx.metallib.zst");
    std::fs::write(&zst, compressed).expect("cannot write the compressed metallib");
    println!("cargo::rustc-env=MLX_METALLIB_ZST={}", zst.display());
    println!("cargo::rustc-env=MLX_METALLIB_LEN={}", raw.len());
    println!("cargo::rustc-env=MLX_METALLIB_ID={id:016x}");
}

/// Never reached: `main` calls `embed_metallib` only for a macOS target with the feature on, and
/// those build scripts have the real one (and its `zstd` build dependency). A non-macOS host has
/// no `mlx` build dependencies at all.
#[cfg(not(all(target_os = "macos", feature = "mlx")))]
fn embed_metallib() {
    unreachable!("the mlx_engine cfg needs the mlx feature and a macOS target");
}
