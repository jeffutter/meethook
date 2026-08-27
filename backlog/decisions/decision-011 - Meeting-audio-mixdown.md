---
id: decision-011
title: Meeting audio mixdown
date: '2026-08-27 04:41'
status: accepted
---
Each session's two tracks are mixed down into one compressed `meeting.opus` file for listening back, encoded with the pure-Rust `ropus` and `ogg` crates (verified, not merely assumed, to introduce no C build dependency once actually built from crates.io) rather than any C-backed Opus library, keeping the project's standing no-C-dependency posture. The mix uses the echo-cancelled mic track rather than the raw one — mixing the raw track would sum the far end's audio twice, once from the speaker track and once again as uncancelled bleed through the mic, producing audible echo — panned to a partial rather than hard left/right separation (constant-power pan, measured and kept at roughly a third of full width after a real listening comparison against wider settings), at a bitrate chosen with one notch of headroom above the lowest setting that sounded clean in a listening test rather than at a bitrate chosen from a table. Clipping after the two tracks are mixed and panned is prevented by a single static gain applied to the whole file if its peak would otherwise exceed a fixed ceiling, rather than a dynamic or per-window limiter (which measurably introduces audible pumping) or Opus's own output-gain metadata field (which is meant to restore gain a player already expects to be missing, and would simply cancel the attenuation out again).

Before mixing, each track's level is independently corrected toward a fixed target using gated integrated loudness (ITU-R BS.1770, hand-implemented directly rather than taking the general-purpose `ebur128` crate, whose streaming multi-mode design is built for a considerably larger job than measuring one number from one track) — gating chosen specifically over plain peak or RMS measurement because the two tracks are asymmetrically silence-heavy by nature (the far end typically talks more than the local mic), and an ungated measure reads pure talk-time as loudness, over-correcting whichever side talks less. The absolute target, −16 LUFS, follows the podcast/streaming convention rather than the broadcast industry's quieter −23 standard, since the listener here is a person on headphones or laptop speakers rather than a broadcast chain. How far a quiet track can be boosted toward that target is capped, since boosting too aggressively raises noise floor rather than useful signal — the cap was raised from an initial 12 dB to 18 dB once measurement against real sessions showed the lower cap binding on every session tested, contradicting the assumption it had originally been set against.

The mixdown's bitrate, pan width, loudness target, and boost cap are all exposed as CLI flags, but two of the four were only added after their original design deliberately excluded them: the loudness target and boost cap were first treated as decisions the algorithm itself should own rather than taste a listener could reasonably disagree with, and were only opened up to the CLI after the tool's owner explicitly asked for that control and was walked through the original reasoning first — recorded here as a deliberate reversal, not a silent scope change.

## Considered options

- Mixing the raw (non-echo-cancelled) mic track — sums the far end's audio twice, producing audible echo.
- A C-backed Opus encoder — the pure-Rust path was verified to work and keeps a standing no-C-dependency posture.
- Peak or plain RMS loudness measurement instead of gated integrated loudness — both were shown to mismeasure this specific pair of tracks because of their asymmetric silence.
- The `ebur128` crate for loudness measurement — built for a considerably larger job (streaming, multi-mode, true-peak) than one number from one slice of samples.
- Opus's own output-gain metadata for clip protection — meant to restore gain, not remove it; would cancel out the attenuation being applied.
- Keeping the loudness target and boost cap CLI-inaccessible — reversed on the owner's explicit request after the original reasoning was reviewed.

