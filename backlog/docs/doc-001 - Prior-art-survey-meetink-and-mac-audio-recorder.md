---
id: doc-001
title: 'Prior art survey: meetink and mac-audio-recorder'
type: specification
created_date: '2026-08-09 02:06'
updated_date: '2026-08-09 02:07'
---
## Purpose

Reference doc for TASK-001.01. Surveys two existing macOS meeting-recorder projects for lessons/pitfalls before designing meethook's own diarization, speaker-ID, activity-detection, and speaker-bleed handling. Not an implementation guide — cite this from the deeper ASR/diarization tickets on this map.

## 1. sservaes/meetink (github.com/sservaes/meetink)

Source: README at https://github.com/sservaes/meetink (fetched 2026-08-08).

### Diarization / speaker ID

- **Stack:** Python sidecar (`src/diarize/server.py`, local HTTP on `:8179`) running **sherpa-onnx** with a **WeSpeaker ResNet34 ONNX** embedding model (CoreML-accelerated), *not* sherpa-onnx's own end-to-end diarization pipeline (segmentation + clustering) — meetink only uses sherpa-onnx for the embedding extractor and hand-rolls its own matching/clustering logic on top.
- **Approach:** per-~3s-chunk embedding → two-stage `/identify`: (1) cosine match against enrolled-profile centroids (k-means, up to 3 centroids/profile to capture multimodal voices — different mic/mood/accent), gated by a THRESHOLD (min absolute score) **and** a MARGIN (must beat runner-up by a delta) — both required, tuned via three presets (`focused`/`default`/`strict`) chosen from calendar attendee count; (2) unmatched voices fall back to **online clustering** (`THEM-A`, `THEM-B`, …), reconciled after the fact via `/profile assign`.
- **Notable pitfalls it had to engineer around (i.e., lessons for us):**
  - *Single-candidate false positive*: with only one enrolled profile in a session, "top beats runner-up" is meaningless (no runner-up) — a vaguely-similar voice will match by default unless a higher absolute floor is applied.
  - *Close-pair speakers*: when two enrolled voices are acoustically close (centroid-vs-centroid cosine ≥0.80), the standard margin becomes unsatisfiable even for genuine matches — needed a separate smaller-margin/same-threshold path, and explicitly does NOT let close-pair matches auto-train (drift risk).
  - *Auto-train drift*: folding high-confidence live matches back into a profile's centroid can runaway (one wrong sample skews the centroid → more wrong samples match → skews further). Mitigated with a "tightness hysteresis" that suspends auto-train once a profile's samples start spreading, plus per-sample outlier rejection (cosine-to-nearest-centroid floor) on every mutation path.
  - *Cross-session contamination*: the same voice can score high against multiple enrolled profiles depending on who else is in the room — solved with a per-session whitelist derived from calendar attendees, not global matching.
  - Mic stream is **never diarized** — it's always labelled as the local user (`ME`/`/me` name) since there's only one local speaker; only the system-audio (remote) stream goes through the embedder. Reduces diarization scope but also means it can't tell if a second in-room person spoke into the same physical mic.
  - Whisper's built-in `*-tdrz` (tinydiarize) models are kept as a same-process fallback speaker-turn signal when the diarize sidecar isn't running — cheap but coarse (turn detection, not identity).

### "Mic activity" / auto-record trigger — important nuance

meetink does **not** trigger recording on raw mic activity/VAD. Its `/watch` feature is calendar-driven (polls Calendar.app via a Swift EventKit sidecar every 60s, fires a pre-meeting notification, auto-starts at the scheduled time) layered with a **conferencing-app-active** detector polled every 30s using three signals combined: (1) process names (`zoom.us`, `MSTeams`, `Webex`, Meet PWA host), (2) camera-in-use (treated as the strongest signal), (3) browser tab URL regex-matching real meeting-room paths (not just the bare domain, and specifically excluding post-call landing pages so "End" is detected quickly). Unscheduled ("instant") calls are caught by the same app-detection loop firing a confirm-or-skip notification. End-of-call detection is adaptive: cadence tightens from 30s to 5s the moment one inactive poll is seen, for a ~10–15s stop latency.

**Lesson:** this is a materially more complex and more brittle trigger than mic-activity/VAD — it depends on process names and browser URL patterns that break on app updates, and calendar integration that meethook's simpler personal-use design (auto-start on mic activity) sidesteps entirely. Nothing here suggests mic-activity/VAD-based triggering is inferior; meetink's complexity here is largely in service of *silent, no-button-press* capture of ad hoc calls, which is a different design goal. Worth noting as "don't copy this" complexity rather than a pattern to reuse.

### Other pitfalls worth carrying forward

- Track/label state is per-*session*, not persisted mid-recording — a hot mic-swap event (`AVAudioEngineConfigurationChange`) requires rebuilding the tap without dropping in-flight audio.
- Whisper hallucinations on silence are common enough to need an explicit filter list (`(soft music)`, `[typing]`, "thanks for watching", repetition loops, bracketed-short-string heuristic).
- Custom vocabulary prompt file ships empty by default specifically to avoid whisper regurgitating the prompt text during silence (prompt leakage).

## 2. jftuga/mac-audio-recorder (github.com/jftuga/mac-audio-recorder)

Source: README at https://github.com/jftuga/mac-audio-recorder (fetched 2026-08-08), "Speaker bleed" section.

### Speaker-bleed handling

- **Root cause named explicitly:** recording on open speakers (not headphones) means the mic acoustically picks up a delayed, room-filtered copy of the far-end system audio; summing mic + system tracks in a mixdown produces audible doubling/slap-back echo. The separately-written track files themselves are unaffected — only the post-hoc `--mix` step is.
- **Technique used:** *not* acoustic echo cancellation. It's a **sidechain ducker keyed by the system track**, applied only at mixdown time (`--reduce-bleed light|normal|aggressive`), attenuating the mic during passages where it appears to carry only bleed (e.g. -40dB at "normal") and a smaller attenuation during overlapping speech (-12dB at "normal") since a single gain can't separate the local voice from the bleed underneath it. Before processing, it correlates the two envelopes to confirm bleed is actually present (harmless no-op on headphone recordings). Detection is deliberately conservative — under-attenuating (leaving bleed audible) is preferred over risking cutting off real speech.
- **Explicit limitation acknowledged by the author:** this cannot remove bleed from underneath the local speaker's own voice, so overlapping speech keeps some doubling. Headphones are called out as the actual fix; the ducker is a mitigation for when that's not possible.
- Related but separate concern the same tool solves: **track alignment** — mic capture starts a fraction of a second after system capture (device spin-up lag), timestamped from the same `SCStream` and corrected via a measured-offset sidecar file at mixdown, with a >5s-difference sanity check that disables the correction rather than trusting a bogus shift.

**Lesson for meethook:** since meethook already plans to keep mic and system audio as separate tracks (not live-mixed), the actual bleed problem only matters if/when tracks are combined for playback or if diarization runs on a combined signal — feeding raw mic+system into an ASR/diarization pipeline without addressing bleed will duplicate the remote speaker's words into the mic-track transcript during open-speaker use. A ducker is a post-hoc patch; the more robust fix is preventing the bleed at capture time (see alternatives below).

## 3. More modern/robust alternatives worth flagging for deeper research tickets

### Diarization / speaker embedding
- **sherpa-onnx's own offline speaker-diarization pipeline** (segmentation model `sherpa-onnx-pyannote-segmentation-3-0` + embedding model, e.g. 3D-Speaker/WeSpeaker/NeMo + clustering) — meetink only used sherpa-onnx for embedding extraction and hand-rolled the rest; the built-in pipeline adds a proper VAD/overlap-aware segmentation stage before embedding, which meetink's simpler per-fixed-chunk approach lacks. (github.com/k2-fsa/sherpa-onnx)
- **sherpa-rs** (github.com/thewh1teagle/sherpa-rs) — Rust bindings to sherpa-onnx (STT, TTS, VAD, speaker embedding/diarization), relevant because meethook is Rust-native and wants no external Python dependency; this avoids the Python-sidecar-over-HTTP architecture meetink uses entirely.
- **FluidAudio** (github.com/FluidInference/FluidAudio) — Swift-native, CoreML/ANE-only framework (no Python, no PyTorch/ONNX-Runtime-on-CPU) offering VAD (Silero), speaker embedding/clustering, and a pyannote-based diarization pipeline (including a newer "Pyannote Community-1" pipeline: powerset segmentation + WeSpeaker + VBx clustering) purpose-built for on-device Apple Silicon. Benchmarks ~60x real-time on M1, DER within ~5 points of pyannote 3.0. Worth evaluating directly against sherpa-rs since it's built specifically for this platform.
- **pyannote.audio** itself remains the accuracy reference point both of the above are benchmarked against, but is Python/PyTorch and out of scope given the no-external-Python constraint — useful only as a ceiling to measure against, or as an offline model-export source.

### Mic-activity / voice-activity detection
- **Silero VAD** (via sherpa-rs or FluidAudio, or directly through `candle` — Hugging Face's Rust ML framework) is a more robust and still-lightweight alternative to naive amplitude-threshold triggering, and is already what FluidAudio and sherpa-onnx both ship for VAD. Worth using instead of a raw RMS/amplitude gate for the mic-activity auto-start trigger, since it's speech-aware rather than just level-aware (won't false-trigger on room noise/fan noise, won't miss soft speech).

### Speaker bleed / echo
- **AVAudioEngine's built-in voice-processing mode** (`inputNode.setVoiceProcessingEnabled(true)`, backed by `AUVoiceProcessingIO`) performs real acoustic echo cancellation at capture time — it removes device-originated audio (i.e., the system/speaker output) from the mic tap before it's ever written, rather than post-hoc ducking a captured recording. This is a materially more robust fix than mac-audio-recorder's sidechain ducker, since it works on overlapping speech too (mac-audio-recorder's admitted blind spot). Constraints to research further: requires input and output nodes on the same engine and in voice-processing mode simultaneously, cannot be toggled while the engine is running, and historically has had reported volume/ducking side effects worth validating for meethook's use case (recording, not VoIP transmission).
- `AVAudioSession.setPrefersEchoCancelledInput` is a lighter-weight, hardware-assisted alternative/companion worth checking for applicability on macOS (this API line is more established on iOS; verify macOS parity).

## Sources
- https://github.com/sservaes/meetink (README, fetched 2026-08-08)
- https://github.com/jftuga/mac-audio-recorder (README, fetched 2026-08-08)
- https://github.com/k2-fsa/sherpa-onnx
- https://github.com/thewh1teagle/sherpa-rs
- https://github.com/FluidInference/FluidAudio
- https://developer.apple.com/forums/thread/733733 (AVAudioEngine voice processing / echo cancellation)
- https://developer.apple.com/videos/play/wwdc2023/10235/ (What's new in voice processing, WWDC23)
