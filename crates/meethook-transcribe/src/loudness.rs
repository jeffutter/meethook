//! How loud the *speech* in one mono track is, as one number.
//!
//! Two tracks recorded from different places have no reason to arrive at the same level: the
//! mic is a close local microphone at whatever gain CoreAudio hands over, and the speaker
//! track is the far end already through a conferencing codec and that application's own AGC.
//! Summing them as they arrive lets whichever side is hotter dominate the mixdown. Answering
//! "how loud is this track" is the first half of fixing that; [`crate::mixdown`] spends the
//! answer.
//!
//! # Why gated loudness, and not peak or RMS
//!
//! Peak is set by the loudest transient -- a cough, a door, a keyboard knock -- so matching
//! peaks pulls a whole track down around one accident. Speech level is a property of the
//! distribution, not the maximum.
//!
//! Ungated RMS gets it wrong in the other direction, and does so specifically here. Both
//! meethook tracks are mostly silence, and asymmetrically so: in a multi-party call the far
//! end talks far more than the local person does. An ungated mean would read the local mic as
//! quiet purely because its owner spoke less, and then boost it hard to compensate --
//! amplifying that track's room tone to match the other track's speech. The two gates in
//! ITU-R BS.1770 exist to throw the silence out before the mean is taken, which is what makes
//! the number mean "how loud is this person when they talk". **The gates are the whole point
//! of using this measure rather than a five-line RMS, and are not a formality to simplify
//! away.**
//!
//! # Why this is hand-rolled rather than `ebur128`
//!
//! The candidate was checked rather than dismissed. `ebur128` is pure Rust and its `cc`
//! build-dependency is optional and off by default, so taking it would *not* have violated
//! TASK-032's rule against adding a C dependency; that is not the argument. The argument is
//! fit. It brings a streaming, channel-mapped, multi-mode analyser with true-peak, loudness
//! range, a histogram history mode and a C ABI surface, plus two new crates and a `build.rs`,
//! and what this crate wants is one number for one mono `&[f32]`. That is two biquads and two
//! gates -- smaller and more legible than the RFC 7845 granule arithmetic [`crate::mixdown`]
//! already hand-rolls beside it.

/// Duration of one loudness block, in milliseconds (BS.1770-4 §3, "gating block").
const BLOCK_MS: usize = 400;

/// Blocks overlap by 75%, so each block advances by a quarter of its length (BS.1770-4 §3).
///
/// Expressed as the number of steps a block spans rather than as a fraction, because the step
/// is what the implementation counts in and deriving the block from it keeps the two exactly
/// commensurate at every sample rate.
const STEPS_PER_BLOCK: usize = 4;

/// The offset that turns mean square into LKFS/LUFS (BS.1770-4 §2.1, eq. 2).
const LOUDNESS_OFFSET_LU: f64 = -0.691;

/// The absolute gate (BS.1770-4 §3): blocks quieter than this are silence, not speech, and
/// are dropped before any mean is taken.
const ABSOLUTE_GATE_LUFS: f64 = -70.0;

/// The relative gate (BS.1770-4 §3): after the absolute gate, blocks more than this far below
/// the mean of what survived are dropped too.
///
/// This is the one that does the work on a meeting track. Pauses between sentences are far
/// too loud to fall under -70 LUFS but are still not speech, and -10 LU below the running
/// mean is what separates them from it.
const RELATIVE_GATE_LU: f64 = -10.0;

/// Integrated loudness of one mono track, in LUFS, under ITU-R BS.1770-4 / EBU R 128.
///
/// `samples` is treated as a single channel with weight G = 1.0, at `rate` Hz. The return is
/// the gated integrated loudness: K-weighted, measured over 400 ms blocks at 75% overlap,
/// with the absolute and relative gates applied.
///
/// `None` means there is nothing here to measure -- no block survived the gates. That covers
/// digital silence, a track shorter than one block, and a track whose only content is below
/// the absolute gate. It is `None` rather than `f64::NEG_INFINITY` because the caller's answer
/// to it is "leave this track alone", and an `Option` says that where an infinity invites
/// arithmetic on it and yields an unbounded gain.
pub fn integrated_lufs(samples: &[f32], rate: u32) -> Option<f64> {
    let step = rate as usize * BLOCK_MS / 1000 / STEPS_PER_BLOCK;
    if step == 0 {
        return None;
    }
    let block = step * STEPS_PER_BLOCK;

    // Squared K-weighted energy per 100 ms step, rather than the filtered signal itself: a
    // block is four consecutive steps, so the per-block mean square falls out of a window of
    // four sums. An hour of 16 kHz mono is 36,000 of these rather than 57.6 million samples of
    // `f64` state.
    let mut shelf = Biquad::high_shelf(rate);
    let mut rlb = Biquad::rlb(rate);
    let mut steps: Vec<f64> = Vec::with_capacity(samples.len() / step + 1);
    for chunk in samples.chunks_exact(step) {
        let mut energy = 0.0;
        for sample in chunk {
            let weighted = rlb.apply(shelf.apply(f64::from(*sample)));
            energy += weighted * weighted;
        }
        steps.push(energy);
    }

    // Only whole blocks count. The trailing partial block is dropped, which is also what makes
    // a track shorter than 400 ms measure as `None` without a case of its own.
    let blocks: Vec<f64> = steps
        .windows(STEPS_PER_BLOCK)
        .map(|window| window.iter().sum::<f64>() / block as f64)
        .collect();

    let absolute: Vec<f64> = blocks
        .into_iter()
        .filter(|z| loudness_of(*z) > ABSOLUTE_GATE_LUFS)
        .collect();
    let relative_gate = loudness_of(mean(&absolute)?) + RELATIVE_GATE_LU;

    let gated: Vec<f64> = absolute
        .into_iter()
        .filter(|z| loudness_of(*z) > relative_gate)
        .collect();
    Some(loudness_of(mean(&gated)?))
}

/// The mean of a set of block energies, or `None` when the set is empty.
fn mean(blocks: &[f64]) -> Option<f64> {
    (!blocks.is_empty()).then(|| blocks.iter().sum::<f64>() / blocks.len() as f64)
}

/// One mean-square value as a loudness in LUFS.
///
/// Digital silence gives `-inf` rather than a NaN, which every comparison below then treats as
/// "under the gate" -- the right answer, arrived at without a branch.
fn loudness_of(mean_square: f64) -> f64 {
    LOUDNESS_OFFSET_LU + 10.0 * mean_square.log10()
}

/// One biquad section of the K-weighting filter, in direct form II.
///
/// `f64` state deliberately, though the samples arrive as `f32`: the RLB section's pole pair
/// sits very close to z = 1 at 16 kHz, and that is exactly where an `f32` accumulator would
/// drift over an hour of audio.
struct Biquad {
    b: [f64; 3],
    a: [f64; 2],
    w1: f64,
    w2: f64,
}

impl Biquad {
    /// Stage 1 of K-weighting: the +4 dB high shelf that stands in for the acoustic effect of
    /// a head in a sound field (BS.1770-4 §2.1).
    ///
    /// Derived from the analog prototype by bilinear transform at `rate` rather than read from
    /// the standard's coefficient table, because that table is tabulated at 48 kHz alone and
    /// this mix runs at 16 kHz. Pasting the 48 kHz numbers in puts both filter corners three
    /// times too high and the measurement is quietly, unfalsifiably wrong -- which is why
    /// there is a test below pinning the derivation against the published table.
    fn high_shelf(rate: u32) -> Self {
        const F0: f64 = 1681.974450955533;
        const GAIN_DB: f64 = 3.999843853973347;
        const Q: f64 = 0.7071752369554196;

        let k = (std::f64::consts::PI * F0 / f64::from(rate)).tan();
        let vh = 10.0f64.powf(GAIN_DB / 20.0);
        let vb = vh.powf(0.4996667741545416);
        let a0 = 1.0 + k / Q + k * k;
        Biquad::new(
            [
                (vh + vb * k / Q + k * k) / a0,
                2.0 * (k * k - vh) / a0,
                (vh - vb * k / Q + k * k) / a0,
            ],
            [2.0 * (k * k - 1.0) / a0, (1.0 - k / Q + k * k) / a0],
        )
    }

    /// Stage 2 of K-weighting: the RLB high-pass that discards the rumble below speech
    /// (BS.1770-4 §2.2).
    fn rlb(rate: u32) -> Self {
        const F0: f64 = 38.13547087602444;
        const Q: f64 = 0.5003270373238773;

        let k = (std::f64::consts::PI * F0 / f64::from(rate)).tan();
        let a0 = 1.0 + k / Q + k * k;
        Biquad::new(
            [1.0, -2.0, 1.0],
            [2.0 * (k * k - 1.0) / a0, (1.0 - k / Q + k * k) / a0],
        )
    }

    /// `a` is the two feedback coefficients; the leading one is normalised to 1.0 by both
    /// constructors above and is not stored.
    fn new(b: [f64; 3], a: [f64; 2]) -> Self {
        Biquad {
            b,
            a,
            w1: 0.0,
            w2: 0.0,
        }
    }

    fn apply(&mut self, x: f64) -> f64 {
        let w = x - self.a[0] * self.w1 - self.a[1] * self.w2;
        let y = self.b[0] * w + self.b[1] * self.w1 + self.b[2] * self.w2;
        self.w2 = self.w1;
        self.w1 = w;
        y
    }
}

/// Signals that look enough like someone talking for a gated measurement to have an opinion
/// about them.
///
/// Here rather than in either test module because [`crate::mixdown`]'s tests need the same
/// shapes -- talk time and speaking level varied independently -- and two copies of a fixture
/// this load-bearing would drift.
#[cfg(test)]
pub(crate) mod fixtures {
    /// One sample of speech-like content: a few harmonically unrelated tones. Deterministic,
    /// because there is no rng in this crate's dependency graph and none wanted in a test that
    /// has to reproduce.
    pub(crate) fn voiced(index: usize, rate: u32, amplitude: f32) -> f32 {
        let t = index as f32 / rate as f32;
        let tau = std::f32::consts::TAU;
        amplitude
            * (0.5 * (tau * 180.0 * t).sin()
                + 0.3 * (tau * 430.0 * t).sin()
                + 0.2 * (tau * 1150.0 * t).sin())
    }

    /// `seconds` of silence carrying [`BURST_S`] bursts of speech-like content, one every
    /// `every_s` seconds -- so `every_s` sets how much of the track is talking and `amplitude`
    /// sets how loud that talking is, independently.
    pub(crate) fn bursts(seconds: f64, every_s: f64, amplitude: f32, rate: u32) -> Vec<f32> {
        let period = (every_s * f64::from(rate)) as usize;
        let burst = (BURST_S * f64::from(rate)) as usize;
        (0..(seconds * f64::from(rate)) as usize)
            .map(|i| {
                if i % period < burst {
                    voiced(i, rate, amplitude)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// How long one burst lasts. Comfortably longer than a 400 ms measurement block, so that
    /// most blocks in a burst are wholly inside it rather than straddling its edge.
    pub(crate) const BURST_S: f64 = 1.5;
}

#[cfg(test)]
mod tests {
    use super::fixtures::{bursts, voiced};
    use super::*;
    use crate::TARGET_RATE;

    #[test]
    fn the_48_khz_coefficients_reproduce_the_standards_table() {
        // BS.1770-4's tabulated 48 kHz coefficients. The only independent check of the
        // bilinear transform available without a reference implementation in the tree, and
        // what makes the absolute LUFS figure trustworthy rather than merely self-consistent.
        let shelf = Biquad::high_shelf(48_000);
        let published_b = [1.53512485958697, -2.69169618940638, 1.19839281085285];
        let published_a = [-1.69065929318241, 0.73248077421585];
        for (got, want) in shelf.b.iter().zip(&published_b) {
            assert!((got - want).abs() < 1e-9, "shelf b: {got} vs {want}");
        }
        for (got, want) in shelf.a.iter().zip(&published_a) {
            assert!((got - want).abs() < 1e-9, "shelf a: {got} vs {want}");
        }

        let rlb = Biquad::rlb(48_000);
        assert_eq!(rlb.b, [1.0, -2.0, 1.0]);
        let published_a = [-1.99004745483398, 0.99007225036621];
        for (got, want) in rlb.a.iter().zip(&published_a) {
            assert!((got - want).abs() < 1e-9, "rlb a: {got} vs {want}");
        }
    }

    #[test]
    fn the_coefficients_depend_on_the_sample_rate() {
        // Cheap, and it is exactly the bug of pasting the 48 kHz table in and running it at
        // the 16 kHz this crate actually mixes at.
        assert_ne!(Biquad::high_shelf(16_000).a, Biquad::high_shelf(48_000).a);
        assert_ne!(Biquad::rlb(16_000).a, Biquad::rlb(48_000).a);
    }

    #[test]
    fn doubling_the_amplitude_raises_the_reading_by_six_db() {
        let quiet = bursts(20.0, 4.0, 0.1, TARGET_RATE);
        let loud: Vec<f32> = quiet.iter().map(|s| s * 2.0).collect();

        let quiet = integrated_lufs(&quiet, TARGET_RATE).unwrap();
        let loud = integrated_lufs(&loud, TARGET_RATE).unwrap();

        assert!(
            (loud - quiet - 6.0206).abs() < 0.01,
            "{quiet} LUFS to {loud} LUFS"
        );
    }

    #[test]
    fn appending_silence_does_not_change_the_reading() {
        // The gating property, stated as a test: how long someone stayed quiet must not change
        // how loud they were when they talked.
        let speech = bursts(20.0, 4.0, 0.1, TARGET_RATE);
        let mut padded = speech.clone();
        padded.extend(std::iter::repeat_n(0.0, 30 * TARGET_RATE as usize));

        let speech = integrated_lufs(&speech, TARGET_RATE).unwrap();
        let padded = integrated_lufs(&padded, TARGET_RATE).unwrap();

        assert!(
            (speech - padded).abs() < 0.1,
            "30 s of silence moved the reading from {speech} to {padded} LUFS"
        );
    }

    #[test]
    fn nothing_measurable_reads_as_none_rather_than_as_an_error() {
        assert_eq!(integrated_lufs(&[0.0; 48_000], TARGET_RATE), None);
        assert_eq!(integrated_lufs(&[], TARGET_RATE), None);

        // 100 ms of full-scale tone: loud, and still shorter than one block.
        let short: Vec<f32> = (0..TARGET_RATE as usize / 10)
            .map(|i| voiced(i, TARGET_RATE, 1.0))
            .collect();
        assert_eq!(integrated_lufs(&short, TARGET_RATE), None);

        // A rate too low to hold a block at all, rather than a panic on a zero-length step.
        assert_eq!(integrated_lufs(&[0.5; 1000], 0), None);
    }
}
