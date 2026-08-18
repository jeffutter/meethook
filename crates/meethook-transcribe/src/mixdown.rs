//! One compressed file a person can open and hear the whole meeting from.
//!
//! A finished session holds its audio as two mono tracks that only make sense played
//! together, uncompressed, at roughly 230 MB per hour each. That is the right shape for
//! everything downstream of it -- the echo canceller wants the local voice alone, diarization
//! wants the far end alone -- and the wrong shape for the one reader who is a person. This
//! module produces the artefact for that reader: both tracks on one timeline, levelled against
//! each other, panned apart far enough to tell apart, in a container any player opens.
//!
//! The levelling is not cosmetic. The two tracks have no reason to arrive matched -- one is a
//! close local microphone at whatever gain CoreAudio hands over, the other is the far end
//! already through a conferencing codec and its own AGC -- so summing them as they arrive lets
//! whichever side is hotter dominate the file and leaves the listener riding the volume knob.
//! Each source is measured and corrected before the sum; the crate's private `loudness` module
//! holds the measurement, including why it is a gated BS.1770 loudness rather than a peak or an
//! RMS, and why it is hand-rolled rather than taken from `ebur128`.
//!
//! The mix and the encode live together rather than in two modules because they are one
//! responsibility with one caller. Split, each half would be a shallow module whose interface
//! is as wide as its body, and neither would be usable without the other.
//!
//! # Why the two halves are shaped the way they are
//!
//! [`mix`] takes samples and offsets, not a session. Alignment is the only interesting thing
//! it does, and expressing it as arithmetic over slices is what makes it testable without a
//! directory on disk -- and what lets the caller hand it the *same* offsets `merge` puts into
//! the transcript, so "the audio agrees with the text" holds by construction rather than by
//! two functions happening to agree.
//!
//! [`write()`] takes an interleaved buffer and a rate, not a mix. Opus is a packet codec with no
//! notion of a file and Ogg is a framing format with no notion of audio, so the RFC 7845
//! mapping between them -- pre-skip, granule positions, header packets -- is this function's
//! whole subject. It is written against `ropus`'s own `examples/encode.rs` rather than against
//! the RFC, with the departures called out at each site.

use std::io::{BufWriter, Write as _};
use std::path::Path;

use meethook_session::write_atomic_with;
use ogg::PacketWriter;
use ogg::writing::PacketWriteEndInfo;
use ropus::{Application, Bitrate, Channels, Encoder, Signal};

use crate::audio::file_name;
use crate::loudness;
use crate::progress::Phase;
use crate::{Error, Result};

/// How far from centre each source sits, on a scale where 1.0 is hard left or hard right.
///
/// 0.3 puts about 4 dB between a source's two channels: enough that the local voice and the
/// far end sit in visibly different places, not so much that either ear is ever doing the
/// listening alone. Hard panning -- what "one track per channel" would give for free -- is
/// genuinely unpleasant across an hour on headphones, and it makes a single-track session
/// arrive in one ear.
///
/// One constant rather than a spread of literals so that the value can be revised from a
/// listening session by editing one line.
///
/// Re-confirmed by ear after levelling landed (TASK-039), which is what retires the open
/// question TASK-032.01 left here. The worry was that levelling would narrow the apparent
/// separation, since part of it had been carried by a level difference that no longer exists;
/// it did not. Swept over 0, 0.3, 0.45, 0.6 and 1.0: 0.6 is too wide, 0 does not separate at
/// all, and 0.3 and 0.45 both read well. 0.3 stands as the narrow end of that band.
pub const PAN_POSITION: f32 = 0.3;

/// The target bitrate for the mixdown, in bits per second.
///
/// 32 kbps sits in the middle of the 24-48 kbps range Opus is normally run at for speech, and
/// at stereo 16 kHz input it costs about 14 MB an hour against the ~690 MB an hour the two
/// source WAVs occupy.
///
/// Kept at TASK-032.01's value when TASK-039 re-opened it. The reason to re-ask was that
/// levelling gives a VBR encoder more signal to spend bits on over a quiet track's turns, so
/// the bitrate at which a far-end voice turns to gravel could have moved; the listener chose
/// to keep 32 kbps rather than reporting a new floor, so this value is confirmed by preference
/// and not by a fresh sweep down from 64.
pub const BITRATE_BPS: u32 = 32_000;

/// The narrowest and widest bitrates Opus itself accepts (RFC 6716 §2.1.1).
///
/// Here so that a caller taking this from a user can refuse a bad value where it was typed and
/// name the range in the refusal, rather than letting the encoder fail partway through a run.
pub const BITRATE_MIN_BPS: u32 = 6_000;
pub const BITRATE_MAX_BPS: u32 = 510_000;

/// The range a pan position may occupy, as a distance from centre.
///
/// The constant-power panning below clamps rather than refusing, which is right for a value
/// computed inside this module and wrong for one a user typed: a clamp turns `--pan 30` into
/// a hard pan and says nothing about it.
pub const PAN_MIN: f32 = 0.0;
pub const PAN_MAX: f32 = 1.0;

/// The two mixdown settings a listener can reasonably disagree about.
///
/// One struct rather than two loose numbers because they travel together through four call
/// layers, and a bare `(u32, f32)` that far from here is a pair of arguments waiting to be
/// swapped. [`Default`] is the pair TASK-032.01 settled by listening, so the values stay
/// written down once, in the two constants above.
///
/// The source rate is deliberately absent. It is not a taste setting: the mix reuses the
/// tracks `transcribe` already holds in memory, and any other rate means reading the session's
/// WAVs a second time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// Target bitrate in bits per second.
    pub bitrate_bps: u32,
    /// How far from centre each source sits. See [`PAN_POSITION`].
    pub pan: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            bitrate_bps: BITRATE_BPS,
            pan: PAN_POSITION,
        }
    }
}

/// Encoder complexity, 0-10.
///
/// 10, the maximum, because the cost is irrelevant here: an hour of meeting encodes in about
/// 30 seconds, beside a Whisper pass over the same hour that takes minutes. There is no
/// budget to save this out of.
const COMPLEXITY: u8 = 10;

/// Packet duration. 20 ms is Opus's default and the one every player is best exercised on.
const FRAME_MS: usize = 20;

/// Ogg granule positions and `OpusHead`'s pre-skip are counted in 48 kHz samples whatever
/// rate the encoder runs at (RFC 7845 §4 and §5.1). This is that rate, and it is deliberately
/// not the same constant as the encoder's input rate.
const GRANULE_RATE: u64 = 48_000;

/// Interleaved stereo, everywhere in this module.
const CHANNELS: usize = 2;

/// Upper bound on one encoded packet. 4000 bytes is what RFC 6716 §3.2's largest legal packet
/// needs; a 20 ms voice frame at 32 kbps is nearer 80.
const MAX_PACKET: usize = 4000;

/// The peak the mix is scaled down to when two loud sources sum past full scale.
///
/// Just under 1.0 rather than exactly it: `encode_float` clips above 1.0, and leaving no
/// margin at all makes the outcome turn on float rounding.
const PEAK_CEILING: f32 = 0.99;

/// The loudness each source is brought to before the two are summed, in LUFS.
///
/// An absolute target rather than merely matching the two tracks to each other, because the
/// measurement is being computed anyway and an absolute one makes `meeting.opus` comparable
/// from session to session instead of only internally balanced. Matching falls out of it for
/// free: two tracks aimed at the same number are aimed at each other.
///
/// A per-track target is legitimate as a whole-mix target here specifically because a meeting
/// is turn-taking. The two tracks rarely carry speech at the same instant, so the mix's gated
/// loudness is about each track's loudness rather than their sum, and the constant-power
/// panning below preserves power on the way. Where they do overlap the mix runs hot by up to
/// 3 LU, which is the correct answer for two people talking over each other.
///
/// -16 LUFS is the podcast and streaming convention rather than EBU R 128's -23, because the
/// reader here is a person on headphones or a laptop speaker rather than a broadcast chain,
/// and -23 arrives too quiet on both. Speech crest factor normally leaves room under
/// this module's peak ceiling at this level; when it does not, the peak scale below pulls the
/// whole mix down uniformly and the balance between the tracks survives it.
///
/// Confirmed by ear (TASK-039), though weakly: swept over -23, -20, -18, -16 and -14 LUFS on a
/// real meeting, on headphones and on a laptop speaker, the listener could not tell the five
/// apart. Read that as "no evidence to move it" rather than as a vindication of -16, and note
/// the reason the sweep was less sensitive than it looks: on that session the mic track was
/// capped by [`MAX_BOOST_DB`] and never reached the target at all, so only the speaker track
/// tracked the sweep. A session where neither track is capped would be a sharper test if this
/// is ever re-opened.
///
/// Public for the same reason [`PAN_POSITION`] is: a listening run has to be able to name the
/// value it is arguing about, and `examples/session-mixdown.rs` sweeps around it.
pub const TARGET_LUFS: f64 = -16.0;

/// The most a single source may be turned up, in dB.
///
/// Gain applied to a quiet-but-not-silent track arrives as hiss and HVAC rumble, so past some
/// point the correction is a worse artefact than the imbalance it fixes. The cap is on the
/// upward direction only: turning a hot track down cannot manufacture noise, so attenuation is
/// deliberately uncapped, and the asymmetry is a decision rather than an oversight.
///
/// The honest consequence: when one track needs more than this and the other does not, the two
/// are left unmatched by the difference. That is the cap declining to amplify a nearly dead
/// track, not a failure to balance.
///
/// 18 dB, raised from the 12 dB this shipped with provisionally, by the listening run in
/// TASK-039. Two things decided it. First, 12 dB was never enough on real material: measured
/// across the sessions under `~/meethook/sessions`, every mic track sat between -32 and
/// -54 LUFS against a speaker track at -20 to -22, and the cap bound on all of them -- the
/// "most tracks want 4-6 dB" reading it was set against turned out to describe the speaker
/// track only. Second, the artefact it guards against was judged directly: sweeping 6, 12, 18
/// and 24 dB, 6 dB is audibly not enough correction, and 18 dB does not bring up hiss or HVAC
/// rumble badly enough to regret it.
///
/// The cap is on the upward direction only, for the reason above, and 18 dB still leaves the
/// deeply quiet end of that corpus unmatched -- a -40 LUFS mic wants 24 dB and gets 18. That is
/// the cap declining to amplify a nearly dead track, which is the intended behaviour.
///
/// Public alongside [`TARGET_LUFS`], and for the same reason.
pub const MAX_BOOST_DB: f64 = 18.0;

/// How each source is brought to a common loudness before the two are summed.
///
/// Deliberately *not* part of [`Settings`]. `Settings` is the pair a listener can reasonably
/// disagree about and that `meethook transcribe` therefore exposes as `--bitrate` and `--pan`;
/// these two are decisions this module makes on the listener's behalf, and the outcome of
/// arguing about them is an edit to [`TARGET_LUFS`] or [`MAX_BOOST_DB`] rather than a new flag.
/// Nothing on the shipping path constructs one of these -- [`mix`] uses [`Default`] and the CLI
/// has no way to say otherwise -- so please do not add a flag for them by symmetry with the
/// other two.
///
/// It exists as a struct at all so that `examples/session-mixdown.rs` can sweep the two values
/// and print the gain each track is about to receive, without restating the formula.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normalization {
    /// The loudness each source is brought to, in LUFS. See [`TARGET_LUFS`].
    pub target_lufs: f64,
    /// The most a single source may be turned up, in dB. See [`MAX_BOOST_DB`].
    pub max_boost_db: f64,
}

impl Default for Normalization {
    fn default() -> Self {
        Normalization {
            target_lufs: TARGET_LUFS,
            max_boost_db: MAX_BOOST_DB,
        }
    }
}

impl Normalization {
    /// The static gain that brings one source to [`Self::target_lufs`], capped at
    /// [`Self::max_boost_db`].
    ///
    /// Unity when there is no speech to measure. That is the whole handling of the empty case:
    /// an imported session has a silent mic track, and the answer for it is "leave it alone"
    /// rather than either an error to propagate or an unbounded boost of nothing.
    ///
    /// The cap is on the upward direction only, so a hot track is still pulled all the way down
    /// to the target however far above it started.
    ///
    /// `rate` is the rate `samples` are already at; nothing here resamples.
    pub fn gain(&self, samples: &[f32], rate: u32) -> f32 {
        match loudness::integrated_lufs(samples, rate) {
            None => 1.0,
            Some(lufs) => {
                let correction_db = (self.target_lufs - lufs).min(self.max_boost_db);
                10.0f64.powf(correction_db / 20.0) as f32
            }
        }
    }
}

/// The logical stream's serial number.
///
/// Any non-zero value identifies a single-stream file, and a fixed one is what makes two runs
/// over the same session produce the same bytes.
const STREAM_SERIAL: u32 = 0x4D45_4554;

/// One mono track going into the mixdown.
pub struct Source<'a> {
    /// Mono samples, already at the mix's sample rate.
    pub samples: &'a [f32],
    /// Seconds from session start to this track's first sample. Negative values are treated
    /// as zero: session start is defined as the earlier of the tracks, so nothing precedes it.
    pub offset_s: f64,
    /// Stereo position, -1.0 hard left through 0.0 centre to 1.0 hard right.
    pub pan: f32,
}

/// Merges mono sources onto one timeline and returns them interleaved as stereo.
///
/// Each source starts `offset_s` into the result and the result runs to the end of whichever
/// source finishes last, so two tracks that were captured at different moments line up the
/// way their offsets say they do. Sources are summed, so an empty one contributes silence and
/// costs only the length it would have occupied -- which is what keeps a session with one
/// usable track producing a normal, playable mix rather than a special case.
///
/// The order of operations is: measure each source, apply its own gain, pan it, sum, then
/// scale the whole result down if the sum would clip. Both gains are static -- the per-source
/// correction and the clip protection alike -- because a compressor that pumps is a worse
/// artefact than a mix that is quiet, and because a constant is a thing a listener can
/// un-learn. The clip scale stays last and stays uniform across the mix, so it cannot undo the
/// balance the per-source gains established.
///
/// A source with no measurable speech in it -- silence, or a track shorter than one 400 ms
/// measurement block -- passes through untouched rather than being boosted on the strength of
/// its own noise floor. See [`Normalization::gain`].
///
/// `rate` turns offsets into sample counts and is the rate the loudness measurement runs at,
/// so it must be the rate the samples are already at; nothing here resamples.
pub fn mix(sources: &[Source<'_>], rate: u32) -> Vec<f32> {
    mix_with(sources, rate, Some(Normalization::default()))
}

/// [`mix`], with the levelling step made answerable.
///
/// `Some(normalization)` is [`mix`]'s behaviour with the two constants replaced. `None` skips
/// the per-source gain entirely, leaving each track at the level it arrived at -- pan, sum, and
/// then the peak ceiling, which still applies, because a mix that clips is not a comparison of
/// anything.
///
/// This exists for one diagnostic: the normalized/unnormalized A/B in
/// `examples/session-mixdown.rs`, which is how the levelling's own value gets confirmed by ear.
/// It is not a configuration point. [`mix`] is the entry point for anything real, and it is the
/// only one the pipeline calls, so the shipping path never sees this `Option` and never has to
/// have an opinion about it.
pub fn mix_with(
    sources: &[Source<'_>],
    rate: u32,
    normalization: Option<Normalization>,
) -> Vec<f32> {
    let starts: Vec<usize> = sources
        .iter()
        .map(|source| (source.offset_s.max(0.0) * f64::from(rate)).round() as usize)
        .collect();

    let frames = sources
        .iter()
        .zip(&starts)
        .map(|(source, start)| start + source.samples.len())
        .max()
        .unwrap_or(0);

    let mut stereo = vec![0.0; frames * CHANNELS];
    for (source, start) in sources.iter().zip(&starts) {
        let gain = normalization.map_or(1.0, |n| n.gain(source.samples, rate));
        let (left, right) = constant_power(source.pan);
        for (index, sample) in source.samples.iter().enumerate() {
            let at = (start + index) * CHANNELS;
            stereo[at] += sample * gain * left;
            stereo[at + 1] += sample * gain * right;
        }
    }

    let peak = stereo.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));
    if peak > PEAK_CEILING {
        let gain = PEAK_CEILING / peak;
        for sample in &mut stereo {
            *sample *= gain;
        }
    }

    stereo
}

/// The left and right gains that place a mono source at `pan`.
///
/// Constant-power rather than linear: the two gains are a sine/cosine pair, so their squares
/// sum to one at every position and a source keeps the same apparent loudness wherever it
/// sits. Linear panning dips ~3 dB in the middle, which is audible as the two voices in a
/// meeting being at different volumes for no reason.
fn constant_power(pan: f32) -> (f32, f32) {
    let theta = (pan.clamp(-1.0, 1.0) + 1.0) * std::f32::consts::FRAC_PI_4;
    (theta.cos(), theta.sin())
}

/// Encodes interleaved stereo as Opus in an Ogg container, atomically.
///
/// `rate` must be one of the five rates Opus accepts -- 8, 12, 16, 24 or 48 kHz -- and is the
/// rate `stereo` is already at; nothing here resamples. An unsupported rate is reported as an
/// error rather than silently reinterpreted.
///
/// An empty `stereo` yields the two header packets and no audio, which is a valid zero-length
/// Ogg Opus stream. That is deliberate: a session with nothing in it should still leave a file
/// a player opens and reports as empty, not a missing file or a failure.
///
/// Atomic, like every other artefact in a session directory: a reader sees the whole file or
/// no file, never a stream truncated mid-page.
pub fn write(path: &Path, stereo: &[f32], rate: u32, bitrate_bps: u32) -> Result<()> {
    let mut encoder = Encoder::builder(rate, Channels::Stereo, Application::Voip)
        .bitrate(Bitrate::Bits(bitrate_bps))
        // Voice, not Auto: this is a meeting, and letting the encoder's music detector have an
        // opinion about it only buys a chance of the wrong one.
        .signal(Signal::Voice)
        // Variable rate. A meeting is mostly one person talking and often nobody, so a
        // constant rate spends the same bits on silence as on speech.
        .vbr(true)
        .complexity(COMPLEXITY)
        .build()
        .map_err(|e| Error::mixdown(path, format!("the encoder rejected {rate} Hz stereo: {e}")))?;

    let frame = rate as usize * FRAME_MS / 1000;
    if frame == 0 {
        return Err(Error::mixdown(
            path,
            format!("{rate} Hz is too low for a {FRAME_MS} ms frame"),
        ));
    }
    let frames = stereo.len() / CHANNELS;

    // Departure from ropus's example, which writes `lookahead()` through unscaled. That value
    // is in the *encoder's* samples (`fs/400` plus the delay compensation, both at `fs`),
    // where `OpusHead`'s pre-skip is defined in 48 kHz samples -- the same 312 either way at
    // 48 kHz, and three times too small at 16 kHz, which would leave a player trimming a third
    // of the priming it should.
    let pre_skip_48k = u64::from(encoder.lookahead()) * GRANULE_RATE / u64::from(rate);
    let pre_skip = u16::try_from(pre_skip_48k).map_err(|_| {
        Error::mixdown(
            path,
            format!("implausible encoder lookahead {pre_skip_48k}"),
        )
    })?;

    // The encoder's output lags its input by `lookahead`, so recovering the last real sample
    // means feeding that many samples of trailing silence past the end of the mix. Without
    // this the file is short by the lookahead and the final granule below would claim more
    // audio than the packets carry.
    let packets = if frames == 0 {
        0
    } else {
        (frames + encoder.lookahead() as usize).div_ceil(frame)
    };
    let frame_granule = frame as u64 * GRANULE_RATE / u64::from(rate);
    // The exact end of the real audio, in the units the container counts in. Written as the
    // final packet's granule so the player end-trims the silence padding the last frame
    // (RFC 7845 §4) and the file's duration is the mix's duration.
    let final_granule = pre_skip_48k + frames as u64 * GRANULE_RATE / u64::from(rate);

    write_atomic_with(path, |file| {
        let mut writer = PacketWriter::new(BufWriter::new(file));

        // Each header gets its own page, as RFC 7845 §3 requires. With no audio to follow, the
        // tags page is the last one in the stream and has to say so, or the file ends without
        // an end-of-stream flag.
        let tags_end = if packets == 0 {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::EndPage
        };
        write_page(
            path,
            &mut writer,
            opus_head(pre_skip, rate),
            PacketWriteEndInfo::EndPage,
            0,
        )?;
        write_page(path, &mut writer, opus_tags(), tags_end, 0)?;

        // Streamed rather than collected. The example buffers every packet into a `Vec` so it
        // can find the last one; the count is known up front here, so the index answers the
        // same question for a long meeting without holding its whole encoded form in memory.
        let mut phase = Phase::start(format!("encoding {}", file_name(path)));
        let mut packet = vec![0; MAX_PACKET];
        let mut pcm = vec![0.0; frame * CHANNELS];
        for index in 0..packets {
            phase.at(index, packets);

            // Clamped rather than assumed in range: the last packet or two exist only to flush
            // the encoder's lookahead, so they start past the end of the mix and are pure
            // padding.
            let from = (index * frame * CHANNELS).min(stereo.len());
            let take = (stereo.len() - from).min(pcm.len());
            pcm[..take].copy_from_slice(&stereo[from..from + take]);
            pcm[take..].fill(0.0);

            let written = encoder.encode_float(&pcm, &mut packet).map_err(|e| {
                Error::mixdown(path, format!("encoding frame {index} of {packets}: {e}"))
            })?;

            let last = index + 1 == packets;
            let (end, granule) = if last {
                (PacketWriteEndInfo::EndStream, final_granule)
            } else {
                (
                    PacketWriteEndInfo::NormalPacket,
                    (index as u64 + 1) * frame_granule,
                )
            };
            write_page(path, &mut writer, packet[..written].to_vec(), end, granule)?;
        }
        phase.done();

        // Explicit, because dropping a `BufWriter` over a `&mut File` discards the error from
        // its final flush -- and that flush is where the last page of a long meeting is.
        writer
            .into_inner()
            .flush()
            .map_err(|e| Error::mixdown(path, e.to_string()))
    })
}

/// One packet, on its own or ending a page, with the container error named after the file.
fn write_page<W: std::io::Write>(
    path: &Path,
    writer: &mut PacketWriter<'_, W>,
    packet: Vec<u8>,
    end: PacketWriteEndInfo,
    granule: u64,
) -> Result<()> {
    writer
        .write_packet(packet, STREAM_SERIAL, end, granule)
        .map_err(|e| Error::mixdown(path, e.to_string()))
}

/// The `OpusHead` identification packet (RFC 7845 §5.1): 19 bytes for stereo under channel
/// mapping family 0.
///
/// The output gain stays at 0 dB on purpose. It would be the obvious place to undo [`mix`]'s
/// clip protection -- and undoing it is exactly what a player would then clip on instead.
fn opus_head(pre_skip: u16, input_rate: u32) -> Vec<u8> {
    let mut head = Vec::with_capacity(19);
    head.extend_from_slice(b"OpusHead");
    head.push(1); // version
    head.push(CHANNELS as u8);
    head.extend_from_slice(&pre_skip.to_le_bytes());
    // Informational: the rate the audio arrived at, which a decoder is free to ignore because
    // Opus always decodes to 48 kHz.
    head.extend_from_slice(&input_rate.to_le_bytes());
    head.extend_from_slice(&0i16.to_le_bytes()); // output gain, Q7.8 dB
    head.push(0); // channel mapping family
    debug_assert_eq!(head.len(), 19);
    head
}

/// The `OpusTags` comment packet (RFC 7845 §5.2): a vendor string and no user comments.
fn opus_tags() -> Vec<u8> {
    let vendor = concat!("meethook ", env!("CARGO_PKG_VERSION")).as_bytes();
    let mut tags = Vec::with_capacity(16 + vendor.len());
    tags.extend_from_slice(b"OpusTags");
    tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    tags.extend_from_slice(vendor);
    tags.extend_from_slice(&0u32.to_le_bytes()); // user comment count
    tags
}

#[cfg(test)]
mod tests {
    use ropus::{DecodeMode, Decoder};
    use tempfile::tempdir;

    use super::*;
    use crate::TARGET_RATE;
    use crate::loudness::fixtures::bursts;

    /// The one channel of an interleaved stereo mix, over `from_s..to_s`.
    ///
    /// Only meaningful for centre-panned sources, where both channels carry every source at the
    /// same gain, which is why the tests that use it pan nothing.
    fn channel(stereo: &[f32], from_s: usize, to_s: usize) -> Vec<f32> {
        let rate = TARGET_RATE as usize;
        stereo[from_s * rate * CHANNELS..to_s * rate * CHANNELS]
            .iter()
            .step_by(CHANNELS)
            .copied()
            .collect()
    }

    /// Mean square of a whole track in dB, with no gating at all -- the naive measure this
    /// module deliberately does not use, kept here so a test can show what it gets wrong.
    fn ungated_db(samples: &[f32]) -> f64 {
        let mean = samples
            .iter()
            .map(|s| f64::from(*s) * f64::from(*s))
            .sum::<f64>()
            / samples.len() as f64;
        10.0 * mean.log10()
    }

    /// A source at the centre, so a test that is not about panning does not have to think
    /// about it.
    fn centred(samples: &[f32], offset_s: f64) -> Source<'_> {
        Source {
            samples,
            offset_s,
            pan: 0.0,
        }
    }

    /// Every 48 kHz sample the file claims, decoded, with the pre-skip already dropped --
    /// i.e. what a player would put on the wire.
    ///
    /// Reads the Ogg back with the same crate that wrote it and the packets back with ropus's
    /// own decoder. That is not an independent check of the container, and it is not meant to
    /// be: what it pins is that the granule arithmetic, the pre-skip and the frame padding
    /// agree with each other, which is where the bugs in this module live.
    fn decode(path: &Path) -> (u16, u8, Vec<f32>) {
        let file = std::fs::File::open(path).unwrap();
        let mut reader = ogg::PacketReader::new(std::io::BufReader::new(file));

        let head = reader.read_packet_expected().unwrap();
        assert_eq!(&head.data[..8], b"OpusHead");
        let channels = head.data[9];
        let pre_skip = u16::from_le_bytes([head.data[10], head.data[11]]);

        let tags = reader.read_packet_expected().unwrap();
        assert_eq!(&tags.data[..8], b"OpusTags");

        let mut decoder = Decoder::new(48_000, Channels::Stereo).unwrap();
        let mut audio = Vec::new();
        let mut last_granule = 0;
        while let Some(packet) = reader.read_packet().unwrap() {
            // 120 ms at 48 kHz, stereo: the largest frame any Opus packet can decode to.
            let mut frame = vec![0.0; 5760 * CHANNELS];
            let decoded = decoder
                .decode_float(&packet.data, &mut frame, DecodeMode::Normal)
                .unwrap();
            audio.extend_from_slice(&frame[..decoded * CHANNELS]);
            last_granule = packet.absgp_page();
        }

        // Pre-skip off the front, and everything past the granule the last page claims off the
        // back: exactly the trimming RFC 7845 §4 asks a player to do. Both bounds are clamped
        // so the headers-only file trims to nothing rather than panicking.
        let skip = (usize::from(pre_skip) * CHANNELS).min(audio.len());
        let end = (last_granule as usize * CHANNELS).clamp(skip, audio.len());
        (pre_skip, channels, audio[skip..end].to_vec())
    }

    #[test]
    fn an_offset_track_starts_that_far_into_the_mix() {
        let mic = vec![0.5; TARGET_RATE as usize];
        let speaker = vec![0.25; TARGET_RATE as usize / 2];

        let stereo = mix(&[centred(&mic, 2.0), centred(&speaker, 0.0)], TARGET_RATE);

        // Three seconds: the mic starts at two and runs for one, which outlasts the speaker's
        // half second from zero.
        assert_eq!(stereo.len(), 3 * TARGET_RATE as usize * CHANNELS);
        // Nothing from the mic before its offset, and the speaker's own contribution has
        // already stopped by then.
        let mic_starts_at = 2 * TARGET_RATE as usize * CHANNELS;
        assert!(
            stereo[mic_starts_at - CHANNELS..mic_starts_at]
                .iter()
                .all(|s| *s == 0.0)
        );
        assert!(
            stereo[mic_starts_at..mic_starts_at + CHANNELS]
                .iter()
                .all(|s| *s > 0.0)
        );
    }

    #[test]
    fn a_panned_source_reaches_both_channels_at_different_levels() {
        let mut mic = vec![0.0; 100];
        mic[0] = 1.0;

        let stereo = mix(
            &[Source {
                samples: &mic,
                offset_s: 0.0,
                pan: -PAN_POSITION,
            }],
            TARGET_RATE,
        );

        let (left, right) = (stereo[0], stereo[1]);
        // Both channels carry it -- this is the assertion that fails if anyone reaches for a
        // hard pan -- and the left one carries more, because that is where the mic was put.
        assert!(right > 0.0, "the far channel is silent: {right}");
        assert!(left > right, "panned the wrong way: {left} vs {right}");
        // Constant power: the two gains' squares sum to one.
        assert!((left * left + right * right - 1.0).abs() < 1e-5);
        // And partially rather than fully apart: under 6 dB between the channels.
        assert!(left / right < 2.0, "panned too far: {left} vs {right}");
    }

    #[test]
    fn two_full_scale_sources_do_not_clip() {
        let mic = vec![1.0; 1000];
        let speaker = vec![1.0; 1000];

        let stereo = mix(
            &[
                Source {
                    samples: &mic,
                    offset_s: 0.0,
                    pan: -PAN_POSITION,
                },
                Source {
                    samples: &speaker,
                    offset_s: 0.0,
                    pan: PAN_POSITION,
                },
            ],
            TARGET_RATE,
        );

        assert!(stereo.iter().all(|s| s.abs() <= 1.0), "the mix clips");
        // Scaled, not squashed: the loudest sample is still at the ceiling.
        let peak = stereo.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));
        assert!((peak - PEAK_CEILING).abs() < 1e-5, "peak {peak}");
    }

    #[test]
    fn the_header_records_the_pre_skip_in_48_khz_samples() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meeting.opus");
        let stereo = vec![0.0; TARGET_RATE as usize * CHANNELS];

        write(&path, &stereo, TARGET_RATE, BITRATE_BPS).unwrap();

        let encoder = Encoder::builder(TARGET_RATE, Channels::Stereo, Application::Voip)
            .build()
            .unwrap();
        let expected = encoder.lookahead() * 48_000 / TARGET_RATE;

        let (pre_skip, channels, _) = decode(&path);
        assert_eq!(channels, 2);
        assert_eq!(u32::from(pre_skip), expected);
        // The scaling is the point: at 16 kHz the encoder's own figure is a third of this.
        assert_ne!(u32::from(pre_skip), encoder.lookahead());
    }

    #[test]
    fn a_mix_round_trips_with_each_source_on_its_own_side() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meeting.opus");

        // Two seconds of two tones far enough apart to tell apart after a lossy codec.
        let seconds = 2;
        let samples = seconds * TARGET_RATE as usize;
        let tone = |hz: f32| -> Vec<f32> {
            (0..samples)
                .map(|i| 0.5 * (std::f32::consts::TAU * hz * i as f32 / TARGET_RATE as f32).sin())
                .collect()
        };
        let mic = tone(300.0);
        let speaker = tone(900.0);

        let stereo = mix(
            &[
                Source {
                    samples: &mic,
                    offset_s: 0.0,
                    pan: -PAN_POSITION,
                },
                Source {
                    samples: &speaker,
                    offset_s: 0.0,
                    pan: PAN_POSITION,
                },
            ],
            TARGET_RATE,
        );
        write(&path, &stereo, TARGET_RATE, BITRATE_BPS).unwrap();

        let (_, channels, decoded) = decode(&path);
        assert_eq!(channels, 2);

        // Duration survives the round trip to within one packet.
        let frames = decoded.len() / CHANNELS;
        let expected = seconds * 48_000;
        assert!(
            frames.abs_diff(expected) <= 960,
            "decoded {frames} frames, expected {expected}"
        );

        // Energy landed on the side each source was panned to. Comparing the two channels'
        // energy rather than either against its source keeps this a claim about the mix and
        // not about how much a 32 kbps encode happens to preserve -- but they are the mirror
        // of each other, so the ratio still fails if the pan is dropped or inverted.
        let energy = |channel: usize| -> f32 {
            decoded
                .iter()
                .skip(channel)
                .step_by(CHANNELS)
                .map(|s| s * s)
                .sum()
        };
        let (left, right) = (energy(0), energy(1));
        assert!(left > 0.0 && right > 0.0, "a channel is silent");
        // Two mirror-image sources, so the two channels should hold about the same energy.
        assert!(
            (left / right - 1.0).abs() < 0.5,
            "channels are lopsided: {left} vs {right}"
        );
        assert!(
            decoded.iter().all(|s| s.abs() <= 1.0),
            "the decoded mix clips"
        );
    }

    #[test]
    fn a_session_with_one_silent_track_still_yields_a_playable_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meeting.opus");

        // What an imported session looks like: a mic track of pure silence, and everything
        // that was actually said on the other one.
        let mic = vec![0.0; TARGET_RATE as usize];
        let speaker: Vec<f32> = (0..TARGET_RATE as usize)
            .map(|i| 0.5 * (std::f32::consts::TAU * 440.0 * i as f32 / TARGET_RATE as f32).sin())
            .collect();

        let stereo = mix(
            &[
                Source {
                    samples: &mic,
                    offset_s: 0.0,
                    pan: -PAN_POSITION,
                },
                Source {
                    samples: &speaker,
                    offset_s: 0.0,
                    pan: PAN_POSITION,
                },
            ],
            TARGET_RATE,
        );
        write(&path, &stereo, TARGET_RATE, BITRATE_BPS).unwrap();

        let (_, channels, decoded) = decode(&path);
        assert_eq!(channels, 2);
        assert!(decoded.len() / CHANNELS > 40_000, "the file is empty");
        assert!(decoded.iter().any(|s| s.abs() > 0.01), "the file is silent");
    }

    #[test]
    fn a_wholly_empty_session_yields_headers_and_no_audio() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meeting.opus");

        let stereo = mix(&[centred(&[], 0.0), centred(&[], 0.0)], TARGET_RATE);
        assert!(stereo.is_empty());

        write(&path, &stereo, TARGET_RATE, BITRATE_BPS).unwrap();

        let (_, channels, decoded) = decode(&path);
        assert_eq!(channels, 2);
        assert!(decoded.is_empty());
    }

    #[test]
    fn equal_speaking_loudness_at_unequal_talk_time_gets_equal_gain() {
        // Two people in the same meeting, speaking at the same level, one of whom holds the
        // floor four times as long. This is the case the gates exist for.
        let quiet_participant = bursts(60.0, 9.0, 0.2, TARGET_RATE); // ~10.5 s of talking
        let talkative_one = bursts(60.0, 2.0, 0.2, TARGET_RATE); // ~45 s of talking

        let quiet_gain = Normalization::default().gain(&quiet_participant, TARGET_RATE);
        let talkative_gain = Normalization::default().gain(&talkative_one, TARGET_RATE);

        let apart = 20.0 * f64::from(quiet_gain / talkative_gain).log10();
        assert!(
            apart.abs() < 0.5,
            "gains {quiet_gain} and {talkative_gain} are {apart} dB apart"
        );
        // And the gains are doing something, rather than agreeing because both are unity.
        assert!(quiet_gain > 1.5, "no correction was applied: {quiet_gain}");

        // The other half of the claim: an ungated mean would have read the same two tracks as
        // 6 dB apart -- 10*log10(45/10.5) -- purely from talk time, and boosted the quieter
        // person's room tone by that much to "fix" it. Asserting this is what makes the test
        // prove the gating is doing the work rather than that two similar signals measure
        // similarly.
        let ungated_apart = ungated_db(&talkative_one) - ungated_db(&quiet_participant);
        assert!(
            (ungated_apart - 6.32).abs() < 0.5,
            "ungated measures them {ungated_apart} dB apart"
        );
    }

    #[test]
    fn a_silent_track_passes_through_at_unity() {
        // An imported session: the mic track is digital silence and everything said is on the
        // other one.
        let mic = vec![0.0; 10 * TARGET_RATE as usize];
        let speaker = bursts(10.0, 2.0, 0.2, TARGET_RATE);

        assert_eq!(Normalization::default().gain(&mic, TARGET_RATE), 1.0);
        // The speaker is corrected on its own merits, not held back or dragged up by an
        // unmeasurable neighbour.
        let alone = Normalization::default().gain(&speaker, TARGET_RATE);
        assert!(alone > 1.0, "the speaker was not corrected: {alone}");

        let both = mix(
            &[
                Source {
                    samples: &mic,
                    offset_s: 0.0,
                    pan: -PAN_POSITION,
                },
                Source {
                    samples: &speaker,
                    offset_s: 0.0,
                    pan: PAN_POSITION,
                },
            ],
            TARGET_RATE,
        );
        let speaker_only = mix(
            &[Source {
                samples: &speaker,
                offset_s: 0.0,
                pan: PAN_POSITION,
            }],
            TARGET_RATE,
        );

        // Sample for sample the same mix: the silent track contributed nothing and changed
        // nothing about what the other track was scaled by.
        assert_eq!(both, speaker_only);
        assert!(both.iter().any(|s| s.abs() > 0.01), "the mix is silent");
    }

    #[test]
    fn normalization_survives_the_peak_ceiling() {
        // The two sources are placed at non-overlapping times so each half of the result can be
        // measured on its own, and centred so that neither half's level is a statement about
        // panning.
        let quiet = {
            let mut samples = bursts(10.0, 2.0, 0.2, TARGET_RATE);
            // A door slamming: 8 ms of decaying 900 Hz at full scale. Far too short to move a
            // gated loudness measurement, and more than enough to drive the sum past the
            // ceiling once this track is turned up -- which is the entire argument against
            // normalizing on peak.
            for i in 0..(0.008 * f64::from(TARGET_RATE)) as usize {
                let t = i as f32 / TARGET_RATE as f32;
                samples[TARGET_RATE as usize + i] =
                    (-t / 0.002).exp() * (std::f32::consts::TAU * 900.0 * t).sin();
            }
            samples
        };
        let loud = bursts(10.0, 2.0, 0.5, TARGET_RATE);

        let apart = crate::loudness::integrated_lufs(&loud, TARGET_RATE).unwrap()
            - crate::loudness::integrated_lufs(&quiet, TARGET_RATE).unwrap();
        assert!(apart > 6.0, "the sources only start {apart} LU apart");

        let stereo = mix(&[centred(&quiet, 0.0), centred(&loud, 10.0)], TARGET_RATE);

        // The ceiling engaged, so this is a mix that had to be scaled down after balancing.
        let peak = stereo.iter().fold(0.0f32, |peak, s| peak.max(s.abs()));
        assert!((peak - PEAK_CEILING).abs() < 1e-5, "peak {peak}");

        // And the balance the per-source gains established is still there afterwards, because
        // the scale that pulled the peak down was uniform across the whole mix.
        let first =
            crate::loudness::integrated_lufs(&channel(&stereo, 0, 10), TARGET_RATE).unwrap();
        let second =
            crate::loudness::integrated_lufs(&channel(&stereo, 10, 20), TARGET_RATE).unwrap();
        assert!(
            (first - second).abs() < 1.5,
            "the two halves ended {first} and {second} LUFS apart"
        );
    }

    #[test]
    fn an_unnormalized_mix_leaves_the_sources_where_they_were() {
        // The mirror of the test above, and the thing the listening A/B in
        // `examples/session-mixdown.rs` rests on: with no normalization the two tracks arrive in
        // the mix as far apart as they were on disk. Same placement as that test -- centred, at
        // non-overlapping times -- so each half can be measured on its own.
        let quiet = bursts(10.0, 2.0, 0.2, TARGET_RATE);
        let loud = bursts(10.0, 2.0, 0.5, TARGET_RATE);

        let sources_apart = crate::loudness::integrated_lufs(&loud, TARGET_RATE).unwrap()
            - crate::loudness::integrated_lufs(&quiet, TARGET_RATE).unwrap();
        assert!(
            sources_apart > 6.0,
            "the sources start {sources_apart} apart"
        );

        let stereo = mix_with(
            &[centred(&quiet, 0.0), centred(&loud, 10.0)],
            TARGET_RATE,
            None,
        );

        let first =
            crate::loudness::integrated_lufs(&channel(&stereo, 0, 10), TARGET_RATE).unwrap();
        let second =
            crate::loudness::integrated_lufs(&channel(&stereo, 10, 20), TARGET_RATE).unwrap();
        let mix_apart = second - first;
        assert!(
            (mix_apart - sources_apart).abs() < 0.5,
            "sources were {sources_apart} LU apart and the mix has them {mix_apart} apart"
        );
        // And the same mix through `mix` does level them, so this is a claim about the argument
        // rather than about a fixture that happens to need no correction.
        let normalized = mix(&[centred(&quiet, 0.0), centred(&loud, 10.0)], TARGET_RATE);
        assert_ne!(normalized, stereo);
    }

    #[test]
    fn a_lower_boost_cap_leaves_a_quiet_track_quieter() {
        // A track far enough under the target that both caps bind, so the two gains differ by
        // exactly the difference between the caps. This is what makes the boost-cap arm of
        // `examples/session-mixdown.rs` a real comparison rather than four identical files.
        let quiet = bursts(10.0, 2.0, 0.01, TARGET_RATE);
        let lufs = crate::loudness::integrated_lufs(&quiet, TARGET_RATE).unwrap();
        assert!(
            TARGET_LUFS - lufs > MAX_BOOST_DB,
            "the fixture only wants {} dB, so neither cap binds",
            TARGET_LUFS - lufs
        );

        let capped = Normalization {
            max_boost_db: 6.0,
            ..Normalization::default()
        }
        .gain(&quiet, TARGET_RATE);
        let default = Normalization::default().gain(&quiet, TARGET_RATE);

        let apart = 20.0 * f64::from(default / capped).log10();
        assert!(
            (apart - (MAX_BOOST_DB - 6.0)).abs() < 0.01,
            "gains {default} and {capped} are {apart} dB apart"
        );
        // The default cap is genuinely a cap here, not the full correction the track wanted.
        assert!(
            f64::from(default) < 10.0f64.powf((TARGET_LUFS - lufs) / 20.0),
            "the cap did not bind: {default}"
        );
    }

    #[test]
    fn a_different_target_moves_the_whole_mix() {
        // The other arm the example sweeps. A track close enough to the target that no cap is
        // involved, so the gain tracks the target one dB for one dB.
        let speech = bursts(10.0, 2.0, 0.2, TARGET_RATE);

        let at_default = Normalization::default().gain(&speech, TARGET_RATE);
        let quieter = Normalization {
            target_lufs: TARGET_LUFS - 6.0,
            ..Normalization::default()
        }
        .gain(&speech, TARGET_RATE);

        let apart = 20.0 * f64::from(at_default / quieter).log10();
        assert!(
            (apart - 6.0).abs() < 0.01,
            "targets moved the gain {apart} dB"
        );
    }

    #[test]
    fn an_unsupported_rate_is_reported_rather_than_reinterpreted() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("meeting.opus");

        // 44.1 kHz is the rate someone reaches for by reflex, and one of the four Opus does
        // not take.
        let error = write(&path, &[0.0; 128], 44_100, BITRATE_BPS).unwrap_err();

        assert!(error.to_string().contains("44100"), "{error}");
        assert!(!path.exists(), "a rejected rate left a file behind");
    }
}
