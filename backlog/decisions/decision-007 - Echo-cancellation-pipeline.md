---
id: decision-007
title: Echo cancellation pipeline
date: '2026-08-27 04:40'
status: accepted
---
Speaker-bleed on the microphone track — audible on this tool's normal external-speaker setup, without headphones — is removed by real reference-based echo cancellation (WebRTC's AEC3, via the `webrtc-audio-processing` crate dynamically linked against nixpkgs), using the synchronized speaker track as the far-end reference against the mic track, run offline in `transcribe` before ASR. `AVAudioEngine`'s built-in voice-processing AEC was ruled out because its reference is scoped to audio the same process renders — it cannot cancel bleed from another app's Zoom or Meet window. The output is written to a new derived file, `mic.cleaned.wav`; the original `mic.wav` is never touched, and if no usable reference exists (headphones, a missing or silent speaker track) the cleaned file is simply an unprocessed copy rather than an error, since that's the common case for a careful user and must never become a hard failure.

Making AEC3 work correctly required several corrections discovered only against real recordings. The session's recorded start-time offset between tracks is measurably wrong by up to several hundred milliseconds — CoreAudio's own reported device latency is unreliable enough (measured under-reporting Bluetooth output latency by nearly a quarter second) that correcting from it would introduce a larger error than it fixes — so the true lag is instead measured acoustically, via phase-transform correlation over several separated, energy-ranked windows of the actual audio, band-limited to the speech range so mic self-noise doesn't get weighted as heavily as real bleed. The aligned reference is deliberately laid down with roughly 20 milliseconds of headroom ahead of the measured lag rather than flush with it, reversing an initial assumption that exact alignment was correct — flush alignment was measured to collapse cancellation from 33 dB down to near zero within about a second and a half, because AEC3's own internal delay estimator needs slack to work with. Separately, on a long meeting the true lag between tracks was found to drift measurably over time — a real session's ~1.9 parts-per-million clock drift was previously misread as disagreement between measurement windows and caused AEC to be skipped for an entire session that was in fact cleanly alignable — fixed by fitting the per-window lag measurements to a line (using a repeated-median regression robust to the small window counts involved) instead of treating them as a single constant value.

The AEC configuration deliberately disables noise suppression and automatic gain control, since both rewrite speech in ways Whisper wasn't trained to expect and neither was needed to solve the actual bleed problem. Whether and how well a session's mic track was cleaned is now recorded to disk as a durable, versioned record (`cleaning.json`) — added after diagnosing the clock-drift bug required rebuilding that information from scratch, when it should have simply been on disk already.

## Considered options

- `AVAudioEngine`'s built-in voice-processing AEC — can't cancel bleed originating from another app's audio output.
- A pure-Rust or speexdsp-based echo canceller instead of linking WebRTC's own APM — both are independent reimplementations rather than the production algorithm; kept as documented fallbacks, never needed.
- Correcting the track offset from CoreAudio's reported device latency — measurably unreliable, would introduce a larger error than it removes.
- Treating measured per-window lag as one constant value rather than fitting a drift line — mistook a real, smoothly drifting clock offset for noisy disagreement and skipped AEC on a cleanly alignable session.
- Enabling noise suppression and automatic gain control — deferred deliberately as a separate, evidence-backed change; neither was needed to fix bleed.

