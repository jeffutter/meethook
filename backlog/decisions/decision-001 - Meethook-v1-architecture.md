---
id: decision-001
title: Meethook v1 architecture
date: '2026-08-27 04:38'
status: accepted
---
Meethook is a single `meethook` binary with `record`, `transcribe`, and `enroll` subcommands sharing one on-disk session contract, rather than the two independent binaries the original spec called for — the "no IPC, no shared process" intent that motivated two binaries is preserved because the subcommands share nothing but the contract. That contract — session directory layout, `session.json`'s fields, atomic writes, and session classification (Valid/Orphaned/Transcribed) — is owned by one crate, `meethook-session`, so no consumer crate can drift from it. Sample rate, channel count, and bit depth are deliberately excluded from `session.json`; the WAV header is the sole authority for those, because letting two representations of the same fact exist invites them to disagree. Track sync timestamps are stored as raw `mach_absolute_time` ticks plus timebase info rather than pre-converted nanoseconds, so no precision is lost before transcribe does the real alignment math downstream.

Audio capture uses ScreenCaptureKit for system-wide speaker audio and a fully separate `AVAudioEngine` input tap for the microphone, built on the low-level `objc2-screen-capture-kit` bindings rather than the higher-level `screencapturekit` crate. Both choices were forced: a documented Apple bug corrupts output when a single `SCStream` captures both microphone and system audio together, and the higher-level crate's build script shells out to `swift build`/`xcrun` in a way that can't run cleanly inside the Nix devShell. Both tracks are written as native-format mono WAV with no resampling in the recorder — any resampling would be an irreversible lossy transform on the raw signal, so rate handling is deferred entirely to transcribe. Mono WAVs needed a further fix: `hound`, the WAV-writing crate, always tags mono output as front-left rather than center, so most players and CoreAudio itself route a mono recording into one ear. Rather than patch or fork `hound`, a small `Write`-wrapping shim rewrites just the four channel-mask bytes of the header as they stream past — a fix that also self-heals a crash-checkpointed recording, since `hound` never rewrites that header after the initial write.

The project ships as a Nix flake with `devShells` only — no flake packages — since this is a personal, non-distributed tool with no deployment story to serve. Model weights are kept out of the Nix closure entirely and fetched lazily to disk with embedded sha256 verification on first use, since baking multi-gigabyte, license-restricted weights into every Nix store copy has no upside for a single-user tool.

## Considered options

- Two independent binaries (original spec) — dropped once subcommands were shown to satisfy the same process-isolation intent without the ceremony.
- Unified single-`SCStream` capture including the microphone — ruled out by a documented Apple corruption bug.
- The `screencapturekit` crate over raw `objc2` bindings — its build couldn't run inside the Nix devShell.
- Forking or patching `hound` for the channel-mask bug, or reopening files after finalize to patch them — both cost more than a streaming shim and the latter leaves crash-checkpointed recordings still broken.
- Flake packages via `crane`/`buildRustPackage` — no consumer exists for a distributable package.

## Consequences

Later extended, not reversed: `meethook-record` (pure Apple frameworks) was pulled out of the shared Cargo workspace into its own standalone workspace so the rest of the tool — `transcribe` and `enroll` — can build, test, and run on Linux, accepting real costs (a second lockfile, duplicated dependency pins) rather than asking a non-macOS toolchain to parse Apple-only bindings.

