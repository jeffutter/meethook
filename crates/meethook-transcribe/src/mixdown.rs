//! One compressed file a person can open and hear the whole meeting from.
//!
//! A finished session holds its audio as two mono tracks that only make sense played
//! together, uncompressed, at roughly 230 MB per hour each. That is the right shape for
//! everything downstream of it -- the echo canceller wants the local voice alone, diarization
//! wants the far end alone -- and the wrong shape for the one reader who is a person. This
//! module produces the artefact for that reader: both tracks on one timeline, panned apart
//! far enough to tell apart, in a container any player opens.
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
pub const PAN_POSITION: f32 = 0.3;

/// The target bitrate for the mixdown, in bits per second.
///
/// Provisional, pending a listening session: 32 kbps sits in the middle of the 24-48 kbps
/// range Opus is normally run at for speech, and at stereo 16 kHz input it costs about
/// 14 MB an hour against the ~690 MB an hour the two source WAVs occupy.
pub const BITRATE_BPS: u32 = 32_000;

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
/// The whole result is scaled by one static gain when it would otherwise clip. Static rather
/// than per-window, because a compressor that pumps is a worse artefact than a mix that is
/// quiet, and because a constant is a thing a listener can un-learn.
///
/// `rate` is only used to turn offsets into sample counts; the samples are passed through
/// untouched, so it must be the rate they are already at.
pub fn mix(sources: &[Source<'_>], rate: u32) -> Vec<f32> {
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
        let (left, right) = constant_power(source.pan);
        for (index, sample) in source.samples.iter().enumerate() {
            let at = (start + index) * CHANNELS;
            stereo[at] += sample * left;
            stereo[at + 1] += sample * right;
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
