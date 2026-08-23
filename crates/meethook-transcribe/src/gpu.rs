//! Whether this process can actually reach a Metal device, on the platforms that have one.
//!
//! whisper.cpp is built with the `metal` feature on macOS, so `WhisperContext::new_with_params`
//! goes straight to the GPU. In an environment where `MTLCreateSystemDefaultDevice()` returns
//! NULL -- a seatbelt sandbox, CI, a headless SSH session, a VM without GPU passthrough --
//! ggml's Metal buffer allocation fails, and ggml does not null-check that failure: it hands
//! the NULL buffer to `ggml_metal_buffer_is_shared`, which dereferences it. The process dies
//! with SIGSEGV. That happens inside C, so it cannot be caught from Rust; the only fix is to
//! not get there. Hence a probe, before the load.
//!
//! **The decision: hard failure by default, with `MEETHOOK_CPU=1` as an explicit opt-in.**
//!
//! The sibling [`crate::open_session`] falls back CoreML -> CPU silently, and the two cases
//! deliberately do *not* behave the same way. CoreML declining a graph partition is a normal
//! outcome on healthy hardware, and the fallback is what keeps the intended target working. A
//! missing Metal device is never a property of the intended target -- an unsandboxed Apple
//! Silicon Mac always has one -- so it always means the environment is not what the user
//! thinks it is. Silently moving a 1.6 GB Whisper checkpoint onto the CPU inside an
//! unattended batch turns that into a ~20x slowdown nobody chose and nobody can see.
//!
//! Off macOS there is no Metal at all: whisper.cpp is compiled for the CPU, `use_gpu` reports
//! that, and neither the probe nor the error exists in the build. `MEETHOOK_CPU=1` still
//! parses and still means "run on the CPU" -- it is just the only path rather than an opt-out,
//! which keeps the variable meaningful on every platform rather than half of them.
//!
//! The env var is what keeps the error from being a dead end: it names a way forward, and it
//! already has a consumer -- verification work under the agent sandbox has to run Whisper on
//! the CPU for exactly this reason. `MEETHOOK_TIMING_DEBUG` and `MEETHOOK_ACTIVITY_DEBUG` in
//! meethook-record set the naming convention.

use std::ffi::OsStr;
use std::fmt;
#[cfg(target_os = "macos")]
use std::sync::OnceLock;

// objc2-metal's own docs say `MTLCreateSystemDefaultDevice` lives in CoreGraphics and that
// the caller must link it. That was checked here rather than assumed, and on this SDK the
// symbol resolves from Metal alone -- link-tested by dropping objc2-core-graphics from this
// crate and running the probe. If a future SDK or toolchain reinstates the requirement, the
// fix is to depend on objc2-core-graphics (already in the workspace) rather than to
// hand-write a `#[link]` stanza, matching how meethook-record reaches that framework.
#[cfg(target_os = "macos")]
use objc2_metal::MTLCreateSystemDefaultDevice;

/// No usable Metal device, and the GPU is the only path compiled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoMetalDevice(());

impl std::error::Error for NoMetalDevice {}

impl fmt::Display for NoMetalDevice {
    /// Names the cause, the environments that produce it, and the one thing the user can do
    /// about it.
    ///
    /// Without the last part this is a dead end: the machine almost certainly *has* a GPU,
    /// and the user has no way to guess that it is their sandbox and not their hardware.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "no usable Metal device: this process cannot reach the GPU."
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "meethook transcribe runs Whisper on Metal. Some environments hide the GPU even on"
        )?;
        writeln!(
            f,
            "a machine that has one -- a sandbox, CI, an SSH session with no window server, or"
        )?;
        writeln!(f, "a VM without GPU passthrough are the usual causes.")?;
        writeln!(f)?;
        write!(
            f,
            "Set MEETHOOK_CPU=1 to transcribe on the CPU instead. It produces the same \
             transcript, many times slower."
        )
    }
}

/// Decides whether Whisper should be loaded onto the GPU, refusing rather than crashing when
/// it cannot be.
///
/// One function rather than a probe plus an env-var reader for the caller to combine: the
/// three outcomes are one decision, and a caller that assembled them itself could get the
/// order wrong -- probing first would refuse a run that `MEETHOOK_CPU=1` had already opted
/// out of needing a device at all.
///
/// Returns `Ok(true)` to run on the GPU, `Ok(false)` to run on the CPU by request -- or,
/// off macOS, because the CPU is the only path compiled in -- and [`NoMetalDevice`] when
/// there is no device and the user has not asked for the CPU.
pub fn use_gpu() -> std::result::Result<bool, NoMetalDevice> {
    if cpu_requested() {
        return Ok(false);
    }
    #[cfg(target_os = "macos")]
    {
        if metal_device_available() {
            Ok(true)
        } else {
            Err(NoMetalDevice(()))
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        // No GPU backend is compiled in on this platform; the CPU is not a fallback, it is
        // the build. Nothing to probe and nothing to refuse.
        Ok(false)
    }
}

/// True when this process can create a Metal device.
///
/// Cached: one framework call per process, however many callers ask. The probe is far too
/// cheap to matter next to a 1.6 GB model load, and caching keeps that true by construction
/// if it ever gains a second caller.
#[cfg(target_os = "macos")]
fn metal_device_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    // Device presence, not a trial allocation. NULL from this call is the condition actually
    // observed in every environment where the segfault reproduces, and it is the necessary
    // condition for ggml's allocation to have any chance. A trial buffer would be a closer
    // proxy for `ggml_metal_buffer_init` and still would not cover every allocation ggml
    // makes; revisit only if a machine that passes this probe still crashes.
    *AVAILABLE.get_or_init(|| MTLCreateSystemDefaultDevice().is_some())
}

/// Whether the user asked to transcribe on the CPU.
fn cpu_requested() -> bool {
    opted_in(std::env::var_os("MEETHOOK_CPU").as_deref())
}

/// Split from the read purely so it can be tested. `std::env::set_var` races every other
/// thread in the test binary -- several of which read `MEETHOOK_ROOT` -- which is exactly why
/// Rust 2024 made it `unsafe`.
///
/// An empty value counts as unset: that is how a variable exported without one arrives from a
/// shell or a CI config, and it should not silently give up the GPU.
fn opted_in(value: Option<&OsStr>) -> bool {
    value.is_some_and(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The message's whole job is to turn "segmentation fault" into something actionable. A
    /// refactor that drops either half quietly undoes that, which is why both are pinned.
    #[test]
    fn the_message_names_both_the_cause_and_the_way_forward() {
        let message = NoMetalDevice(()).to_string();
        assert!(message.contains("Metal"));
        assert!(message.contains("sandbox"));
        assert!(message.contains("MEETHOOK_CPU=1"));
    }

    /// Giving up the GPU has to be something the user actually asked for. An exported-but-
    /// empty variable is the case worth pinning: it is common, and reading it as an opt-in
    /// would put a whole batch on the CPU by accident.
    ///
    /// Deliberately no test asserting what `use_gpu` returns with the variable unset -- that
    /// would be asserting the hardware the test happens to run on, not the code.
    #[test]
    fn only_a_non_empty_value_opts_in_to_the_cpu() {
        assert!(!opted_in(None));
        assert!(!opted_in(Some(OsStr::new(""))));
        assert!(opted_in(Some(OsStr::new("1"))));
    }
}
