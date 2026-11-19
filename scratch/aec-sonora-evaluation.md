# AEC backend evaluation: C++ webrtc-audio-processing vs pure-Rust sonora

Date: 2026-11-19. Decision: switched to sonora anyway (the owner judged it close enough on the samples and
will try real recordings on another machine); the C++ crate and its build plumbing are removed. Status: measurements only, no listening verdict yet (see "Open"). Headline: with echo present, sonora attenuates an overlapping near-end voice about 8-9 dB more than C++ (correlation with the clean voice 0.25-0.28 against 0.60-0.75), at equal echo removal. With no echo, neither touches the voice.

## Why

Linux CI broke because nixpkgs abseil-cpp 20260817 needs C++20 while
`webrtc-audio-processing-sys` 2.1.0 hard-codes `-std=c++17` (`build.rs:403,435`; the vendored
`meson.build:7,52` also says `cpp_std=c++17`). Upstream (tonarino master, freedesktop master) still
has this and no issue reports it. CI was unblocked by pinning nixpkgs (d9441aa). This evaluates
replacing the C++ dependency with `sonora` 0.3.0 (github.com/dignifiedquire/sonora, 99 stars,
BSD-3-Clause, rust 1.91). The runner-up `aec3` 0.4.0 (RubyBit/aec3-rs, 62 stars) was not tested.

## What sonora is

An AI-assisted pure-Rust port of WebRTC M145 audio processing. Its README says the 2400+ test C++
suite passes against it through its C API. It is a behavioural port, not bit-exact: on the
synthetic fixture only 64 of 960,000 output samples were bit-identical to the C++ library.

## Method

Both backends run through the real `cancel_bleed` path (lag measurement, `shift_reference`,
128-sample `PROCESSOR_DELAY` compensation, frame loop), switched inside `subtract`. Settings:
AEC3 default plus full-band high-pass. Inputs converted to 16 kHz mono f32 with ffmpeg first.

Data: 6 recordings from microsoft/AEC-Challenge `datasets/test_set_icassp2022` (real, crowdsourced,
Windows PCs): far-end single talk x2 (+1 "with movement"), double talk x3. Downloaded via
media.githubusercontent.com (Git LFS). Licence unchecked; not stored in the repo.

## Results

Synthetic 60 s bleed fixture (`aec.rs` tests), output level in dB:

| | bleed-only | near-only |
|---|---|---|
| input | -25.8 | -24.4 |
| C++ | -28.4 | -24.6 |
| sonora | -57.7 | -24.9 |

The C++ figure looks too weak against what the existing test accepts, so this run probably handicapped
C++ (no reference shift in that harness). Treat this table as unreliable; the real-recording run
below used the full path.

Real recordings through `cancel_bleed`:

| clip | outcome | mean per-second change vs input (C++ / sonora) | median ERLE (C++ / sonora) |
|---|---|---|---|
| far-end only 00113069 | cancelled, lag 184 | -32.0 / -37.5 dB | 8.2 / 8.3 |
| far-end only 02462e78 | cancelled, lag 499 | -56.7 / -60.6 dB | 13.9 / 14.6 |
| double talk 00113069 | cancelled, lag 2628 | -15.1 / -15.7 dB | 0.2 / 6.2 |
| double talk 041fc379 | cancelled, lag 450 | -12.2 / -14.7 dB | 8.3 / 8.7 |
| far-end only with movement 00dPPLGk | passed through (alignment gate) | n/a | n/a |
| double talk 02462e78 | passed through (alignment gate) | n/a | n/a |

- Same lags, same pass-through decisions for both backends: the alignment step is shared, so it
  does not distinguish them.
- Sonora's output was equal or quieter in every cancelled clip. On far-end-only clips that is
  residual echo, so lower is better (it removed 4-5 dB more on average, quieter by more than 3 dB in
  13/24 and 17/24 seconds against 2/24 and 1/24 for C++).
- On double talk, lower overall level could also mean a clipped near-end voice. The test set has no
  clean near-end reference for these clips, so this cannot be measured; only listening can say.
- Speed: ~150-240 ms per clip for both, indistinguishable.
- Delay: both outputs are delayed 128 samples relative to input (synthetic fixture cross-correlation),
  so `PROCESSOR_DELAY` holds. Not re-checked on the real clips.
- C++ reported median ERLE 0.2 dB on double talk 00113069 against 6.2 dB for sonora, with nearly
  identical output level. Not understood; ERLE is read from each library's own stats and may be
  computed differently.

## Findings that are not about the AEC backend

2 of 6 real clips (movement; one double talk) were passed through uncancelled by the alignment gate
("5 windows disagreed by 35 ms" and "0 of 7 windows correlated well enough"). Those are
outside this comparison but would matter in production: echo path changes defeat the single-lag assumption.

## Correction (after first listening attempt)

The shootout page first described the "double talk" clips as a near-end voice talking over the echo.
The level traces show otherwise for both clips used: the far end plays for roughly 0-12 s and 27-39 s,
and the near-end voice is mostly on its own between about 13 s and 26 s, where the output equals the
input (checked in the MP3s: -21.4 dB against -21.3 dB). So the far-end-only clips and the far-end
stretches of the double-talk clips are expected to come out as near-silence plus residue. Whether a
near-end voice also sits under the echo in the far-end stretches is not established: coherence between
mic and loopback there is 0.76-0.79 for 041fc379 and 0.51 for 00113069, which leaves room for
non-echo content or a non-linear echo path. Real overlapping double talk is therefore still untested.

## Constructed double talk (answer key available)

The dataset has no real overlap with a known clean voice, so overlap was built: a clean
near-end-only recording (`nearend-singletalk`, normalised to -20 dB RMS) was added onto a real
far-end-only echo recording from 6 s to about 21 s, with the far-end loopback as the reference. The
clean voice is the target. 2 echo recordings x 2 voices = 4 mixes; signal-to-echo ratio -5 to -7 dB.
Same `cancel_bleed` path, same lags as before.

| mix | SDR in near span, dB (mic / C++ / sonora) | correlation with clean voice (mic / C++ / sonora) | voice gain in output (C++ / sonora) | echo-only tail level, dB (mic / C++ / sonora) |
|---|---|---|---|---|
| fA+n1 | -6.8 / 1.5 / -0.7 | 0.42 / 0.57 / 0.28 | -7.1 / -14.9 | -13.5 / -30.9 / -32.2 |
| fA+n2 | -6.9 / 1.8 / -0.6 | 0.41 / 0.61 / 0.29 | -6.4 / -14.8 | -13.5 / -30.9 / -32.7 |
| fB+n1 | -5.3 / 1.7 / -0.4 | 0.48 / 0.57 / 0.27 | -9.6 / -18.3 | -15.5 / -38.3 / -41.8 |
| fB+n2 | -5.4 / 1.7 / -0.3 | 0.47 / 0.57 / 0.29 | -9.6 / -17.4 | -15.5 / -39.7 / -40.2 |

("voice gain" is the least-squares gain of the clean voice inside the output. Correlation is best
over +-400 samples, which came out at lag 0 for the mic and C++ and -2 for sonora, so alignment does not
explain it.)

Reading: this reverses the far-end-only impression. Sonora removes slightly more echo when only echo
is present (tail column), but when a voice overlaps the echo it attenuates the voice by roughly
15-18 dB, against 6-10 dB for C++, and its output correlates with the clean voice less than the raw mic
does. C++ also loses voice (6-10 dB) but improves on the raw mic. That is the "silences the near-end
talker" failure mode, and it matters because the product transcribes the mic track. Limits: 4 mixes
from 2 echoes and 2 voices; voice normalised to one level; voice recorded in other rooms; -5 to -7 dB
echo ratio is harsh. The difference may come from sonora tracking a newer WebRTC (M145) with different
suppression defaults, or from my sonora config (`EchoCanceller::default()`); neither was investigated,
so the cause is unknown.

## Controls and a level confound (supersedes the table above where they disagree)

The "constructed double talk" mixes above were too loud (mic -13 dB to -3 dB RMS, the +12 dB one near
clipping). Controls on the voice alone (no echo in the mic):
- loud (-3 dB RMS), silent reference: C++ -5 to -6 dB and sonora -6 to -7 dB across 150-2400 Hz. Not a
  high-pass filter effect (the voice has no energy below 150 Hz).
- scaled to -25 dB RMS, silent reference: C++ -0.1 dB, sonora -0.3 dB. So neither library touches a clean
  voice at a normal level, and the loss above was a level effect (probably AEC3's saturated-mic
  handling; not verified). The first "silent reference" measurements also used a broadband projection
  that overstated the backend gap.

Rerun with every mix (mild 0/+6/+12 dB and the four original mixes) scaled to -25 dB RMS, 150-4000 Hz
band, mean of 4 mixes per row, 16 of 16 cancelled with the same lags for both backends:

| voice over echo | voice gain dB, C++ / sonora | corr with clean voice, mic / C++ / sonora | echo-only tail change dB, C++ / sonora |
|---|---|---|---|
| -6 | -6.3 / -14.4 | 0.45 / 0.60 / 0.25 | -16.9 / -17.7 |
| 0 | -5.1 / -13.5 | 0.70 / 0.70 / 0.28 | -17.1 / -16.8 |
| +6 | -5.4 / -14.1 | 0.85 / 0.75 / 0.28 | -14.7 / -15.1 |
| +12 | -8.2 / -17.3 | 0.86 / 0.73 / 0.27 | -15.7 / -14.9 |

Reading: at a proper level the backend gap holds. Echo removal is equal; with echo present sonora
loses about 8-9 dB more of an overlapping voice, and its output barely tracks the clean voice
(correlation about 0.27 even when the voice is 12 dB over the echo). Both libraries lose voice
relative to the raw mic once the voice is louder than the echo (corr below the mic's), C++ less.
Not explained: why sonora suppresses more; its config is `EchoCanceller::default()` and nothing was tuned.
Listening: clips 7 and 8 on the shootout page are at this level; clips 5 and 6 are the loud mixes.

## Open

- Listening: `scratchpad/shootout/index.html` (blind A/B, shuffled per clip). Verdict pending.
- Sonora's near-end attenuation under overlap: cause unknown; its suppression parameters were not tuned.
- Only 4 real clips actually exercised the AECs. Not enough to call it "better".
- Rest of the `aec.rs` test set (delay-constant test, stats-throttling test, frame-error test) has not
  been run against sonora. `build_processor`/`Processor` API would need replacing.
- Spike code: `sonora-spike.diff` beside the page. Reverted from the working tree.

## Reproduce

Apply `sonora-spike.diff`, then
`SPIKE_IN=<dir of 16k f32 *_mic.wav/_lpb.wav> SPIKE_OUT=<dir> cargo test -p meethook-transcribe --lib sonora_spike -- --ignored --nocapture`.
