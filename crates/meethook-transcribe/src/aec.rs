//! Acoustic echo cancellation over the recorded mic track.
//!
//! The mic track of a meeting contains the other participants twice: once through the
//! network as the speaker track, and once acoustically as whatever the laptop's speakers
//! played back into its own microphone. Whisper transcribes that leakage as if it were the
//! local user, so it has to be removed before the mic track is transcribed.
//!
//! The remover is WebRTC's AEC3, reached through the tonarino `webrtc-audio-processing`
//! crate. It is linked dynamically against the flake's `webrtc-audio-processing` package
//! rather than built here: the crate's `bundled` feature compiles the vendored C++ itself
//! and does not cross-compile on Apple (tonarino/webrtc-audio-processing#102).
//!
//! This is real reference-based cancellation, and it is possible only because the recorder
//! captured the speaker output as an independent synced track. `AVAudioEngine`'s built-in
//! `isVoiceProcessingEnabled` was ruled out precisely because its reference is scoped to
//! audio the same process renders, so it cannot touch bleed originating in Zoom or a
//! browser.
//!
//! # Two decisions worth knowing before reading the code
//!
//! *Everything happens at 16 kHz.* Reference-based AEC needs both streams at one rate and
//! the two tracks routinely differ -- ScreenCaptureKit delivers 48 kHz while a Bluetooth
//! headset switches the default input to a 16 kHz HFP mic. 16 kHz is the rate to converge
//! on: it is already the ASR target, so [`crate::audio::read_track_16k_mono`] gets both
//! tracks there with no new code; it is a native WebRTC APM rate, so nothing is resampled
//! again inside the processor; and at 16 kHz the APM runs single-band, so there is no
//! band-splitting filter delay shifting the output against the input.
//!
//! *Only the reference moves.* The returned audio is sample-for-sample on `mic`'s own
//! timeline, because the caller adds a metadata offset describing `mic.wav` to every ASR
//! timestamp. Shifting the mic to meet the reference would silently skew every timestamp in
//! the transcript, so the measured lag is applied to the reference instead.

use meethook_session::write_atomic_with;
use std::path::Path;
use webrtc_audio_processing::{Config, Processor, config::EchoCanceller};

use crate::align::{self, Alignment, NotMeasurable};
use crate::audio::TARGET_RATE;
use crate::{Error, Result};

/// AEC3's frame size is fixed at 10 ms, and feeding it any other length is a panic rather
/// than an error, so the 16 kHz frame size is pinned here rather than left to be discovered
/// by a crash mid-meeting.
const SAMPLES_PER_FRAME: usize = 160;

/// How much later a sample comes out of the capture path than it went in, with the echo
/// canceller enabled at 16 kHz.
///
/// Measured, not documented anywhere upstream: an impulse fed through this exact
/// configuration comes back 128 samples -- 8 ms -- late. The high-pass filter adds nothing
/// to it and it is zero with the echo canceller off, so it belongs to AEC3's own capture
/// buffering.
///
/// It is compensated rather than tolerated. `mic.cleaned.wav` must sit sample-for-sample on
/// `mic.wav`'s timeline, because the caller adds an offset describing `mic.wav` to every ASR
/// timestamp; letting the track slip 8 ms late would skew the whole transcript by an amount
/// small enough that nobody would ever catch it. `the_processor_delay_constant_still_matches
/// _the_library` re-measures it, so a library update that changes it fails loudly here
/// instead.
const PROCESSOR_DELAY: usize = 128;

/// How far the reference is left leading the mic track after alignment, in samples (20 ms).
///
/// The obvious move is to align the reference to a lag of exactly zero and let AEC3's own
/// delay estimator handle what remains. Measured on the bleed fixture, that is the one
/// setting that does not work: cancellation reaches 33 dB for the first second and a half,
/// then collapses to 0.2 dB for the remaining 50 s. AEC3 aligns by *delaying* the render
/// stream into its own buffer, and it keeps deliberate headroom in front of the estimate, so
/// a render stream that arrives already flush with the capture leaves it nothing to give
/// back and the alignment never recovers once the near-end talker first disturbs it.
///
/// Anything from 5 ms upwards produces identical output -- 26 dB median ERLE, sample for
/// sample the same at 5, 10, 20 and 40 ms -- because the delay estimator simply buffers the
/// surplus away. 20 ms is chosen as the widest of those, so that an alignment measurement
/// that is a few milliseconds optimistic still lands on the working side of zero.
const RENDER_HEADROOM: i64 = TARGET_RATE as i64 * 20 / 1000;

/// A mic track with the speaker bleed taken out of it, and an account of what happened.
#[derive(Debug, Clone, PartialEq)]
pub struct Cleaned {
    /// Exactly as long as the mic track handed in, on the same timeline.
    pub audio: Vec<f32>,
    pub cleaning: Cleaning,
}

/// What the pre-pass did, in enough detail to say something truthful to the user without
/// reaching back into the audio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cleaning {
    /// The reference lined up and AEC3 ran over the whole track.
    Cancelled {
        /// How much later the mic track heard the far end than the speaker track recorded it,
        /// in samples. This is the measurement, not the shift: the reference is actually laid
        /// down [`RENDER_HEADROOM`] samples ahead of it.
        lag_samples: i64,
        /// How far the windows the lag was measured over disagreed. A wide spread on an
        /// accepted measurement means the delay drifted or the correlation was marginal.
        spread_samples: i64,
        /// Median echo return loss enhancement across the track, in dB, or `None` if AEC3
        /// never converged far enough to report one.
        erle_db: Option<f64>,
    },
    /// The mic track was passed through untouched, for this reason.
    PassedThrough(PassThrough),
}

/// Why cancellation did not run. Every variant is a normal outcome of a real recording.
///
/// None of them is an error, and none of them means a missing file: `mic.cleaned.wav` is
/// written on every path, so the rest of the pipeline has exactly one input to read and no
/// branch to get wrong. That also protects the headphones case, where there is nothing to
/// cancel and AEC3 handed a reference it cannot align is reported to flatten or mute the
/// near-end talker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassThrough {
    /// No speaker track, or one that is digital silence throughout -- nothing was playing,
    /// so nothing bled.
    NoReference,
    /// The two tracks would not align, so there is no reference to subtract.
    Unalignable(NotMeasurable),
}

impl std::fmt::Display for Cleaning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Cleaning::Cancelled {
                lag_samples,
                spread_samples,
                erle_db,
            } => {
                write!(
                    f,
                    "speaker bleed cancelled (reference lag {:+.0} ms, spread {:.0} ms",
                    samples_to_ms(*lag_samples),
                    samples_to_ms(*spread_samples)
                )?;
                match erle_db {
                    Some(db) => write!(f, ", {db:.1} dB echo reduction)"),
                    None => write!(f, ", echo reduction not reported)"),
                }
            }
            Cleaning::PassedThrough(reason) => write!(f, "no echo cancellation: {reason}"),
        }
    }
}

impl std::fmt::Display for PassThrough {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PassThrough::NoReference => {
                write!(f, "the speaker track is missing or silent, so nothing bled")
            }
            PassThrough::Unalignable(reason) => write!(f, "{reason}"),
        }
    }
}

/// Removes speaker bleed from `mic_16k`, using `speaker_16k` as the far-end reference.
///
/// Both tracks must already be 16 kHz mono, as [`crate::audio::read_track_16k_mono`]
/// returns them. `metadata_offset_s` is `session.json`'s account of how much later the mic
/// track's first sample is than the speaker track's; it only centres the delay search, and
/// the answer comes from the signals.
///
/// The returned audio is always exactly `mic_16k.len()` samples long, whether or not
/// cancellation ran.
pub fn cancel_bleed(mic_16k: &[f32], speaker_16k: &[f32], metadata_offset_s: f64) -> Cleaned {
    // A reference of pure digital zero is not something to measure against, it is the
    // headphones case. Covers an absent speaker track too, which the caller hands over as an
    // empty slice rather than branching.
    if speaker_16k.iter().all(|sample| *sample == 0.0) {
        return passed_through(mic_16k, PassThrough::NoReference);
    }

    let (lag, spread) = match align::measure_reference_lag(mic_16k, speaker_16k, metadata_offset_s)
    {
        Alignment::Measured {
            lag_samples,
            spread_samples,
            ..
        } => (lag_samples, spread_samples),
        Alignment::NotMeasurable { reason } => {
            return passed_through(mic_16k, PassThrough::Unalignable(reason));
        }
    };

    let reference = shift_reference(speaker_16k, lag - RENDER_HEADROOM, mic_16k.len());
    let (audio, erle_db) = subtract(mic_16k, &reference);

    Cleaned {
        audio,
        cleaning: Cleaning::Cancelled {
            lag_samples: lag,
            spread_samples: spread,
            erle_db,
        },
    }
}

/// Writes 16 kHz mono float audio to `path`, all or nothing.
///
/// Atomic for the same reason `session.json` is: a half-written cleaned track that ASR then
/// reads is a corrupt transcript that looks like a model failure.
pub fn write_cleaned_track(path: &Path, audio: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    write_atomic_with(path, |file| {
        // Buffered: hound writes each sample straight through, and an hour of audio is 57
        // million of them.
        let mut writer = hound::WavWriter::new(std::io::BufWriter::new(file), spec)
            .map_err(|e| Error::wav(path, e))?;
        for sample in audio {
            writer
                .write_sample(*sample)
                .map_err(|e| Error::wav(path, e))?;
        }
        writer.finalize().map_err(|e| Error::wav(path, e))
    })
}

/// The mic track unchanged, with the reason recorded.
fn passed_through(mic_16k: &[f32], reason: PassThrough) -> Cleaned {
    Cleaned {
        audio: mic_16k.to_vec(),
        cleaning: Cleaning::PassedThrough(reason),
    }
}

/// The speaker track resampled onto the mic track's index space.
///
/// [`Alignment`] states that speaker sample `i` is the same instant in the room as mic
/// sample `i + lag`, so the reference the canceller needs at mic index `j` is speaker sample
/// `j - lag`. Where the speaker track does not reach -- before it started, after it ended --
/// the reference is zero, which is the truth: nothing was playing.
fn shift_reference(speaker: &[f32], lag: i64, mic_len: usize) -> Vec<f32> {
    let mut reference = vec![0.0f32; mic_len];

    // The first mic index whose reference sample exists, and the speaker index it reads.
    let from_mic = (lag.max(0) as usize).min(mic_len);
    let from_speaker = (lag.saturating_neg().max(0) as usize).min(speaker.len());
    let count = (mic_len - from_mic).min(speaker.len() - from_speaker);

    reference[from_mic..from_mic + count]
        .copy_from_slice(&speaker[from_speaker..from_speaker + count]);
    reference
}

/// Runs the aligned pair through AEC3, returning the cleaned mic track and the median echo
/// return loss enhancement AEC3 reported along the way.
///
/// The config is the echo canceller plus the high-pass filter the crate documents as
/// strongly recommended alongside it, and nothing else. Noise suppression and automatic gain
/// control stay off deliberately: both rewrite speech in ways Whisper was not trained to
/// expect, and AGC pumping across a meeting is a plausible way to make transcripts worse
/// while looking like an improvement.
///
/// `stream_delay_ms` is left unset rather than filled in. The signed part of the alignment
/// already happened in the sample domain -- it had to, since that field is unsigned and the
/// measured range reaches negative -- and what is left is a known-positive [`RENDER_HEADROOM`]
/// plus the acoustic path itself, which is exactly what AEC3's internal delay estimator and
/// adaptive filter exist to track.
fn subtract(mic: &[f32], reference: &[f32]) -> (Vec<f32>, Option<f64>) {
    debug_assert_eq!(mic.len(), reference.len());

    let processor = match Processor::new(TARGET_RATE) {
        Ok(processor) => processor,
        // Constructing the processor is the linkage check, and it succeeded once at startup
        // for the whole workspace's tests. If it fails here the honest move is still to hand
        // back the mic track rather than lose the session.
        Err(_) => return (mic.to_vec(), None),
    };
    assert_eq!(
        processor.num_samples_per_frame(),
        SAMPLES_PER_FRAME,
        "AEC3 panics rather than erroring on a frame-length mismatch"
    );

    processor.set_config(Config {
        echo_canceller: Some(EchoCanceller::Full {
            stream_delay_ms: None,
        }),
        high_pass_filter: Some(Default::default()),
        ..Default::default()
    });

    // The track is run through with `PROCESSOR_DELAY` samples of silence appended, so the
    // last real sample gets pushed out of the canceller rather than being left inside it,
    // and the same number of leading output samples is dropped afterwards. Together those
    // two moves put the output back on the input's own timeline.
    let processed_len = mic.len() + PROCESSOR_DELAY;
    let mut cleaned: Vec<f32> = Vec::with_capacity(processed_len);
    let mut erle: Vec<f64> = Vec::new();
    // Reused across frames so an hour of audio is two allocations, not two per 10 ms.
    let mut render_frame = vec![vec![0.0f32; SAMPLES_PER_FRAME]];
    let mut capture_frame = vec![vec![0.0f32; SAMPLES_PER_FRAME]];

    // A whole frame's worth of `track`, starting at `start`, zero-filled past its end.
    let frame_from = |track: &[f32], start: usize, into: &mut [f32]| {
        let from = start.min(track.len());
        let available = (track.len() - from).min(into.len());
        into[..available].copy_from_slice(&track[from..from + available]);
        into[available..].fill(0.0);
    };

    for start in (0..processed_len).step_by(SAMPLES_PER_FRAME) {
        // A track whose length is not a multiple of 160 leaves a partial final frame. It is
        // zero-padded to a full frame and trimmed back afterwards; this is the one place a
        // sample could be silently gained or lost.
        let taken = SAMPLES_PER_FRAME.min(processed_len - start);
        frame_from(reference, start, &mut render_frame[0]);
        frame_from(mic, start, &mut capture_frame[0]);

        // Render before capture, always: the canceller has to have seen what was played
        // before it can be asked what of it survives in what was heard.
        if processor.process_render_frame(&mut render_frame).is_err()
            || processor.process_capture_frame(&mut capture_frame).is_err()
        {
            // Mid-track failure would leave a half-cancelled track, which is worse than an
            // uncancelled one because nothing downstream could tell.
            return (mic.to_vec(), None);
        }

        cleaned.extend_from_slice(&capture_frame[0][..taken]);
        if let Some(db) = processor.get_stats().echo_return_loss_enhancement {
            erle.push(db);
        }
    }

    cleaned.drain(..PROCESSOR_DELAY.min(cleaned.len()));
    cleaned.truncate(mic.len());

    // Median, not the last reading: the final frames of a meeting are usually silence, where
    // there is no echo to enhance the loss of and the figure says nothing.
    erle.sort_by(f64::total_cmp);
    let median = erle.get(erle.len() / 2).copied();
    (cleaned, median)
}

fn samples_to_ms(samples: i64) -> f64 {
    samples as f64 * 1000.0 / f64::from(TARGET_RATE)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: usize = TARGET_RATE as usize;

    /// Deterministic noise, so a threshold that is marginal fails always rather than once a
    /// month.
    struct Noise(u64);

    impl Noise {
        fn sample(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / 8_388_608.0 - 1.0
        }
    }

    /// A continuous voice-band signal: noise through a speech-like spectral tilt and a DC
    /// blocker, since a loudspeaker reproduces neither the top octave nor the rumble.
    ///
    /// Deliberately ungated. Whoever is talking and when is the caller's business here, and
    /// a generator that also gated on a schedule of its own would leave the caller's idea of
    /// "the far end is speaking now" quietly false wherever the two schedules disagreed.
    fn voice_band(seed: u64, samples: usize) -> Vec<f32> {
        let mut noise = Noise(seed);
        let (mut low, mut previous_low, mut high) = (0.0f32, 0.0f32, 0.0f32);
        (0..samples)
            .map(|_| {
                low = 0.6 * low + 0.4 * noise.sample();
                high = 0.99 * high + low - previous_low;
                previous_low = low;
                high * 0.3
            })
            .collect()
    }

    /// Root-mean-square level of a stretch, in dB.
    fn level_db(samples: &[f32]) -> f64 {
        let energy: f64 = samples.iter().map(|s| f64::from(*s) * f64::from(*s)).sum();
        10.0 * (energy / samples.len().max(1) as f64).max(1e-30).log10()
    }

    fn cancelled(cleaning: Cleaning) -> (i64, Option<f64>) {
        match cleaning {
            Cleaning::Cancelled {
                lag_samples,
                erle_db,
                ..
            } => (lag_samples, erle_db),
            Cleaning::PassedThrough(reason) => panic!("expected cancellation, got: {reason}"),
        }
    }

    /// The shape of the real problem: the user talks over some stretches, the far end bleeds
    /// in through the speakers over others, and there are stretches of each alone.
    struct Bleed {
        mic: Vec<f32>,
        speaker: Vec<f32>,
        /// Half-second stretches of the mic track containing bleed and no local talker.
        bleed_only: Vec<std::ops::Range<usize>>,
        /// Half-second stretches containing the local talker and no bleed.
        near_only: Vec<std::ops::Range<usize>>,
    }

    /// The far end talks for the first two seconds in every five; the local user for the last
    /// two in every seven. The periods are coprime, so the two schedules drift against each
    /// other and produce double-talk, bleed alone, and near-end alone in turn -- which is what
    /// makes this fixture able to distinguish cancellation from muting. The gaps are wide
    /// because a stretch only counts as clean when the other signal has been absent either
    /// side of it too.
    const FAR_CYCLE_S: usize = 5;
    const FAR_TALKS_S: usize = 2;
    const NEAR_CYCLE_S: usize = 7;
    const NEAR_SILENT_S: usize = 5;

    fn bleed_fixture(lag: i64) -> Bleed {
        let seconds = 60;
        let len = RATE * seconds;
        let far_speaking = |i: usize| i % (RATE * FAR_CYCLE_S) < RATE * FAR_TALKS_S;
        let near_speaking = |i: usize| i % (RATE * NEAR_CYCLE_S) >= RATE * NEAR_SILENT_S;

        let speaker: Vec<f32> = voice_band(101, len)
            .iter()
            .enumerate()
            .map(|(i, s)| if far_speaking(i) { *s } else { 0.0 })
            .collect();
        let near: Vec<f32> = voice_band(202, len)
            .iter()
            .enumerate()
            .map(|(i, s)| if near_speaking(i) { *s } else { 0.0 })
            .collect();

        // A room: direct sound, a desk bounce, an opposite-polarity wall reflection, a late
        // one. Loud enough that Whisper would read the bleed as speech.
        let ms = |n: i64| RATE as i64 * n / 1000;
        let room: &[(i64, f32)] = &[(0, 0.55), (ms(3), 0.30), (ms(8), -0.20), (ms(15), 0.12)];
        let mut floor = Noise(303);
        let mut mic: Vec<f32> = near
            .iter()
            .map(|n| n * 0.7 + floor.sample() * 0.003)
            .collect();
        for (tap, gain) in room {
            for i in 0..len as i64 {
                let heard_at = i + lag + tap;
                if heard_at >= 0 && (heard_at as usize) < mic.len() {
                    mic[heard_at as usize] += speaker[i as usize] * gain;
                }
            }
        }

        // Stretches classified by what is in them. The first ten seconds are skipped so
        // AEC3's filter has converged.
        //
        // Presence of the wanted signal is judged over the stretch itself; absence of the
        // other one over a quarter second either side as well. The asymmetry is the point:
        // the room has a tail and a suppressor has a hangover, so a stretch that merely
        // starts after the other signal stopped is not yet free of it -- while widening the
        // window for the wanted signal too would label a silent stretch as bleeding just
        // because bleed starts a moment after it ends.
        let guard = RATE / 4;
        let bleeding = |t: usize| {
            room.iter().any(|(tap, _)| {
                let source = t as i64 - lag - tap;
                source >= 0 && (source as usize) < len && far_speaking(source as usize)
            })
        };
        let (mut bleed_only, mut near_only) = (Vec::new(), Vec::new());
        let mut at = RATE * 10;
        while at + RATE < len {
            let range = at..at + RATE / 2;
            let mut padded = at - guard..at + RATE / 2 + guard;
            let bleed_inside = range.clone().any(bleeding);
            let near_inside = range.clone().any(near_speaking);
            let bleed_nearby = padded.clone().any(bleeding);
            let near_nearby = padded.any(near_speaking);
            if bleed_inside && !near_nearby {
                bleed_only.push(range);
            } else if near_inside && !bleed_nearby {
                near_only.push(range);
            }
            at += RATE / 2;
        }

        Bleed {
            mic,
            speaker,
            bleed_only,
            near_only,
        }
    }

    /// Acceptance criterion #5, both halves of it. Attenuation alone is not the claim:
    /// a function that returned silence would pass that trivially, and silencing the
    /// near-end talker is exactly the AEC3 failure mode worth guarding against.
    #[test]
    fn bleed_is_attenuated_while_the_near_end_talker_survives() {
        let lag = (RATE as i64) * 120 / 1000;
        let fixture = bleed_fixture(lag);
        assert!(
            !fixture.bleed_only.is_empty() && !fixture.near_only.is_empty(),
            "the fixture must contain stretches of each signal alone"
        );

        let cleaned = cancel_bleed(&fixture.mic, &fixture.speaker, 0.0);
        let (measured_lag, erle) = cancelled(cleaned.cleaning);
        assert!(
            (measured_lag - lag).abs() <= (RATE as i64) / 1000,
            "recovered a lag of {measured_lag} against {lag}"
        );

        let change = |ranges: &[std::ops::Range<usize>]| -> Vec<f64> {
            ranges
                .iter()
                .map(|range| {
                    level_db(&fixture.mic[range.clone()]) - level_db(&cleaned.audio[range.clone()])
                })
                .collect()
        };

        let attenuation = change(&fixture.bleed_only);
        let worst = attenuation.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(
            worst >= 10.0,
            "bleed-only stretches attenuated by {attenuation:?} dB, worst {worst:.1}; \
             reported ERLE {erle:?}"
        );

        let damage = change(&fixture.near_only);
        let worst = damage.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            worst <= 3.0,
            "the local talker lost {damage:?} dB where nothing bled, worst {worst:.1}"
        );
    }

    /// Acceptance criterion #4. A track whose length is deliberately not a multiple of the
    /// 160-sample frame comes back exactly as long, and -- the part a length check alone
    /// would miss -- with its content still at the sample index it went in at.
    #[test]
    fn no_sample_is_dropped_duplicated_or_shifted_across_frame_boundaries() {
        let fixture = bleed_fixture((RATE as i64) * 60 / 1000);
        // 73 samples past a frame boundary, so a truncating loop would silently lose them.
        let mut mic = fixture.mic;
        mic.truncate(mic.len() - SAMPLES_PER_FRAME + 73);

        let cleaned = cancel_bleed(&mic, &fixture.speaker, 0.0);
        cancelled(cleaned.cleaning);
        assert_eq!(cleaned.audio.len(), mic.len(), "length must be preserved");

        // Where nothing bled, the cleaned track is the mic track with very little done to
        // it, so the two must correlate best at exactly zero lag. This is a stronger claim
        // than any length check: a dropped frame, a duplicated one, or an uncompensated
        // processor delay all show up as a non-zero best lag, and they show up at points
        // spread across the whole recording rather than at one marker.
        let reach = 2 * SAMPLES_PER_FRAME as i64;
        let inside: Vec<_> = fixture
            .near_only
            .iter()
            .filter(|range| range.end + reach as usize <= mic.len())
            .collect();
        assert!(
            !inside.is_empty(),
            "nothing left to compare after truncation"
        );
        for range in inside {
            let mut best = (f64::NEG_INFINITY, i64::MAX);
            for lag in -reach..=reach {
                let score: f64 = range
                    .clone()
                    .map(|i| {
                        let shifted = (i as i64 + lag) as usize;
                        f64::from(cleaned.audio[i]) * f64::from(mic[shifted])
                    })
                    .sum();
                if score > best.0 {
                    best = (score, lag);
                }
            }
            assert_eq!(
                best.1, 0,
                "over samples {range:?} the cleaned track lines up with the mic track at a \
                 lag of {} samples rather than 0",
                best.1
            );
        }
    }

    /// Pins the measured capture-path delay the code compensates for. If a library update
    /// moves it, `mic.cleaned.wav` would silently drift off `mic.wav`'s timeline and take
    /// every transcript timestamp with it, so it is worth re-measuring rather than trusting.
    #[test]
    fn the_processor_delay_constant_still_matches_the_library() {
        let processor = Processor::new(TARGET_RATE).unwrap();
        processor.set_config(Config {
            echo_canceller: Some(EchoCanceller::Full {
                stream_delay_ms: None,
            }),
            high_pass_filter: Some(Default::default()),
            ..Default::default()
        });

        let marker_at = SAMPLES_PER_FRAME * 100 + 37;
        let total = SAMPLES_PER_FRAME * 200;
        let mut out: Vec<f32> = Vec::with_capacity(total);
        for start in (0..total).step_by(SAMPLES_PER_FRAME) {
            let mut render = vec![vec![0.0f32; SAMPLES_PER_FRAME]];
            let mut capture = vec![vec![0.0f32; SAMPLES_PER_FRAME]];
            if (start..start + SAMPLES_PER_FRAME).contains(&marker_at) {
                capture[0][marker_at - start] = 0.5;
            }
            processor.process_render_frame(&mut render).unwrap();
            processor.process_capture_frame(&mut capture).unwrap();
            out.extend_from_slice(&capture[0]);
        }

        let (peak_at, _) = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap();
        assert_eq!(
            peak_at - marker_at,
            PROCESSOR_DELAY,
            "the capture path's delay changed"
        );
    }

    /// Acceptance criterion #5's other side, and the headphones case. Nothing bled, so the
    /// mic track must come back untouched rather than mangled by a guessed reference.
    #[test]
    fn an_unusable_reference_yields_the_mic_track_byte_for_byte() {
        let mic = voice_band(2, RATE * 30);

        // Absent, and digital silence: both are "nothing was playing".
        for reference in [Vec::new(), vec![0.0f32; mic.len()]] {
            let cleaned = cancel_bleed(&mic, &reference, 0.0);
            assert_eq!(cleaned.audio, mic);
            assert_eq!(
                cleaned.cleaning,
                Cleaning::PassedThrough(PassThrough::NoReference)
            );
        }

        // Playing, but not into this microphone -- the user is wearing headphones.
        let uncorrelated = voice_band(3, mic.len());
        let cleaned = cancel_bleed(&mic, &uncorrelated, 0.0);
        assert_eq!(cleaned.audio, mic);
        assert!(
            matches!(
                cleaned.cleaning,
                Cleaning::PassedThrough(PassThrough::Unalignable(_))
            ),
            "{}",
            cleaned.cleaning
        );
    }

    /// An empty mic track is a real session -- one that ended the instant it started -- and
    /// has to survive every arithmetic path here rather than panic on a zero length.
    #[test]
    fn an_empty_mic_track_is_not_a_panic() {
        let cleaned = cancel_bleed(&[], &voice_band(4, RATE * 30), 0.0);
        assert!(cleaned.audio.is_empty());
    }

    /// The reference is only ever shifted, never the mic, and it is padded with the truth --
    /// silence -- where the speaker track does not reach.
    #[test]
    fn the_reference_is_shifted_onto_the_mic_timeline_and_zero_filled_elsewhere() {
        let speaker: Vec<f32> = (1..=5).map(|i| i as f32).collect();

        // The speaker led: its sample 0 is mic sample 2.
        assert_eq!(
            shift_reference(&speaker, 2, 8),
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0]
        );
        // The mic led: the first two speaker samples happened before the mic existed.
        assert_eq!(
            shift_reference(&speaker, -2, 5),
            vec![3.0, 4.0, 5.0, 0.0, 0.0]
        );
        // Shifted clean off either end.
        assert_eq!(shift_reference(&speaker, 99, 4), vec![0.0; 4]);
        assert_eq!(shift_reference(&speaker, -99, 4), vec![0.0; 4]);
    }

    #[test]
    fn a_cleaned_track_round_trips_through_the_wav_writer_at_16_khz() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mic.cleaned.wav");
        let audio = voice_band(5, RATE + 37);

        write_cleaned_track(&path, &audio).unwrap();

        let reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.spec().sample_rate, TARGET_RATE);
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(crate::audio::read_track_16k_mono(&path).unwrap(), audio);
    }
}
