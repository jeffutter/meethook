//! Defines the `mlx_engine` cfg: the voice isolator's MLX engine is compiled in. That needs the
//! `mlx` feature (on by default) *and* macOS, because the `mlx-rs` dependency is macOS-only. The
//! feature alone is not enough: `--all-features` on Linux turns it on, and the code behind it
//! must still vanish there. Spelling the pair out at every use is how a bare
//! `cfg(feature = "mlx")` slipped into an example and broke Linux CI; use `mlx_engine`.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(mlx_engine)");
    println!("cargo::rerun-if-changed=build.rs");
    let feature = std::env::var_os("CARGO_FEATURE_MLX").is_some();
    let macos = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos");
    if feature && macos {
        println!("cargo::rustc-cfg=mlx_engine");
    }
}
