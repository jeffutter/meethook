//! Renders one session's mixdown at every setting worth arguing about, so the argument can be
//! settled with headphones instead of a table.
//!
//! ```text
//! cargo run --example session-mixdown -- ~/meethook/sessions/20260809-025745 --out /tmp/mixdown
//! ```
//!
//! `transcribe` writes exactly one `meeting.opus`, at one bitrate, from one source, with the
//! two voices at one pan width. Every one of those is a judgement about how a meeting sounds
//! to a person, and nothing in this repository can make it: a test can prove the file decodes
//! and that the channels are not hard-panned, and it cannot hear comb filtering, or notice
//! that 24 kbps turns a quiet participant to gravel. So this writes the whole grid out to a
//! directory, one file per cell, named after its settings.
//!
//! What each arm is for:
//!
//! - **bitrate** (`*-16k-NNkbps-pan30.opus`) -- the headline question. Listen down from 64
//!   until a voice on the far end stops being comfortable, then take the step above it.
//! - **pan** (`cleaned-16k-32kbps-panNN.opus`) -- `pan00` is mono-in-stereo, `pan100` is one
//!   track per ear, `pan30` is what ships. The question is whether the local voice is
//!   *placeable* without either ear ever working alone.
//! - **source** (`raw-...` against `cleaned-...`) -- `transcribe` mixes the echo-cancelled mic
//!   track, since the raw one carries the far end back in through the microphone, delayed.
//!   The risk in that choice is the canceller having chewed something off the local voice, and
//!   this pair is where that would be heard.
//! - **rate** (`*-48k-*`) -- the mix is built from the 16 kHz tracks transcription already
//!   holds in memory. This arm asks whether the band above 8 kHz is worth a second pass over
//!   the originals. Written only when both source files are already at a rate Opus accepts and
//!   agree on it; when they do not, that is said and the arm is skipped rather than resampled
//!   here, because a resampler this file invented would be the thing under test.
//! - **normalization** (`normalized-...` against `unnormalized-...`, and the two `.wav`s) --
//!   whether bringing both tracks to a common loudness is an improvement or a flattening. The
//!   WAV pair is the one to trust: it is the only comparison here where the encoder is not
//!   also varying. Match your listening volume between the two before deciding, because the
//!   unnormalized mix is generally the quieter one and louder wins A/Bs on its own.
//! - **target** (`targetmNNlufs-...`) -- the loudness both tracks are brought to. `m` is the
//!   minus sign, so `targetm23lufs` is -23 LUFS: EBU R128 at one end of the sweep, the podcast
//!   convention of -16 at the other, so the two live conventions bracket whatever wins.
//! - **boost** (`boostNNdb-...`) -- the ceiling on how far a quiet track may be turned *up*.
//!
//! Those last two are only comparisons on a session where their value actually changes
//! something, and each can collapse: a track far enough under the target that every cap in the
//! boost sweep binds hears the same mix at all five targets, and a session where no track is
//! quiet enough for any cap to bind hears the same mix at all four caps. The report below says
//! which arms this session can answer, and an arm with nothing to compare is skipped with a
//! note rather than written out as a row of identical files.
//!
//! - **`mix-16k.wav`** -- the encoder's own input, uncompressed. Every judgement above is
//!   against this, not against memory of the meeting. `mix-16k-unnormalized.wav` is its twin
//!   for the normalization arm.
//!
//! Before anything is encoded the run prints what it measured: each source track's integrated
//! loudness, the correction the shipping normalization asks for, the gain it actually got, and
//! whether the boost cap bound. That report is a few passes of two biquads over each track,
//! which makes pointing this at several sessions a reasonable way to find one where the arms
//! you care about are worth rendering, before committing to a listening run.
//!
//! Both offsets come from `meethook_transcribe`'s own accessors, so every file here sits on
//! the timeline `transcript.md` describes.
//!
//! Like the other diagnostics in this directory, this takes no dependency the library does not
//! already have.

use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavReader, WavSpec};
use meethook_session::{SessionMetadata, SessionPaths};
use meethook_transcribe::mixdown::{self, Normalization, Source};
use meethook_transcribe::{
    TARGET_RATE, integrated_lufs, mic_offset_seconds, read_track_16k_mono, speaker_offset_seconds,
};

/// The bitrates worth hearing, in kbps. Spans the usual speech range and a step past it in
/// both directions, so the chosen value is bracketed rather than merely plausible.
const BITRATES_KBPS: [u32; 5] = [16, 24, 32, 48, 64];

/// Pan positions, as hundredths, for the placement judgement. 0 is centre, 100 is hard.
///
/// Weighted above what ships rather than around it: levelling the two tracks removed the volume
/// difference that was doing part of the work of telling them apart, so the open question is
/// whether 0.3 still separates them, and that is answered by the positions wider than it.
const PANS: [u32; 5] = [0, 30, 45, 60, 100];

/// Loudness targets, in LUFS. -23 is EBU R128 and -16 is the podcast convention; both are in
/// the sweep so the answer is bracketed by the two standards people actually cite.
const TARGETS_LUFS: [f64; 5] = [-23.0, -20.0, -18.0, -16.0, -14.0];

/// Boost caps, in dB, ascending. The first is the one the report tests against: if no track
/// wants more boost than the smallest cap, every file in the arm is the same file.
const BOOST_CAPS_DB: [f64; 4] = [6.0, 12.0, 18.0, 24.0];

/// The bitrate the pan, source, rate, normalization, target and boost arms are all rendered at,
/// so those comparisons vary one thing.
const REFERENCE_KBPS: u32 = 32;

fn main() {
    let mut args = std::env::args_os().skip(1);
    let Some(dir) = args.next().map(PathBuf::from) else {
        usage();
    };
    let out = match (args.next(), args.next()) {
        (Some(flag), Some(path)) if flag == "--out" => PathBuf::from(path),
        (None, None) => dir.join("mixdown-candidates"),
        _ => usage(),
    };

    let paths = SessionPaths::new(&dir);
    let metadata = read_metadata(&paths);
    let (mic_offset, speaker_offset) = (
        mic_offset_seconds(&metadata).unwrap(),
        speaker_offset_seconds(&metadata).unwrap(),
    );

    std::fs::create_dir_all(&out).unwrap();
    println!("{} -> {}", dir.display(), out.display());
    println!("offsets: mic {mic_offset:.3} s, speaker {speaker_offset:.3} s");
    println!(
        "pan {:.0}/100, {} kbps is what ships",
        mixdown::PAN_POSITION * 100.0,
        mixdown::BITRATE_BPS / 1000
    );

    // The cleaned track if `transcribe` has already run over this session, the raw one
    // otherwise -- and say which, because the whole source arm below is about that difference.
    let raw_mic = read_track_16k_mono(&paths.mic_wav()).unwrap();
    let cleaned_mic = match read_track_16k_mono(&paths.mic_cleaned_wav()) {
        Ok(track) => track,
        Err(e) => {
            println!("note: no mic.cleaned.wav ({e}); the `cleaned` arm repeats the raw track");
            raw_mic.clone()
        }
    };
    let speaker = read_track_16k_mono(&paths.speaker_wav()).unwrap_or_default();

    // Everything measurable about the levelling, before a single file exists -- so that a run
    // that turns out not to be worth listening to costs one pass over the tracks rather than a
    // full grid.
    let varies = report(&[
        ("mic", cleaned_mic.as_slice()),
        ("speaker", speaker.as_slice()),
    ]);

    let mixed = |pan: f32, normalization: Option<Normalization>| -> Vec<f32> {
        mixdown::mix_with(
            &sources(&cleaned_mic, mic_offset, &speaker, speaker_offset, pan),
            TARGET_RATE,
            normalization,
        )
    };
    let cleaned = |pan: f32| mixed(pan, Some(Normalization::default()));

    // The encoder's input, so every judgement below has something uncompressed to be a
    // judgement against.
    println!("\n-- reference --");
    let reference = cleaned(mixdown::PAN_POSITION);
    write_wav(&out.join("mix-16k.wav"), &reference, TARGET_RATE);

    // The normalization A/B. Both halves are written under their own names even though the
    // normalized opus is byte-identical to `cleaned-16k-32kbps-pan30.opus` from the bitrate
    // arm: a pair you can name is a pair you can compare, and the encode costs seconds.
    println!("\n-- normalization (match your listening volume between the two) --");
    let flat = mixed(mixdown::PAN_POSITION, None);
    write_wav(&out.join("mix-16k-unnormalized.wav"), &flat, TARGET_RATE);
    encode(
        &out,
        "normalized-16k-32kbps-pan30.opus",
        TARGET_RATE,
        REFERENCE_KBPS,
        &reference,
    );
    encode(
        &out,
        "unnormalized-16k-32kbps-pan30.opus",
        TARGET_RATE,
        REFERENCE_KBPS,
        &flat,
    );

    println!("\n-- bitrate --");
    for kbps in BITRATES_KBPS {
        emit(&out, "cleaned", TARGET_RATE, kbps, 30, &reference);
    }

    println!("\n-- pan --");
    for pan in PANS {
        if pan == 30 {
            continue; // already written by the bitrate arm
        }
        let mix = cleaned(pan as f32 / 100.0);
        emit(&out, "cleaned", TARGET_RATE, REFERENCE_KBPS, pan, &mix);
    }

    println!("\n-- loudness target --");
    if varies.target {
        for target_lufs in TARGETS_LUFS {
            let mix = mixed(
                mixdown::PAN_POSITION,
                Some(Normalization {
                    target_lufs,
                    ..Normalization::default()
                }),
            );
            let name = format!(
                "targetm{:.0}lufs-16k-{REFERENCE_KBPS}kbps-pan30.opus",
                -target_lufs
            );
            encode(&out, &name, TARGET_RATE, REFERENCE_KBPS, &mix);
        }
    } else {
        println!("skipped: see the report above -- every value gives the same mix here");
    }

    println!("\n-- boost cap --");
    if varies.boost {
        for max_boost_db in BOOST_CAPS_DB {
            let mix = mixed(
                mixdown::PAN_POSITION,
                Some(Normalization {
                    max_boost_db,
                    ..Normalization::default()
                }),
            );
            let name = format!("boost{max_boost_db:02.0}db-16k-{REFERENCE_KBPS}kbps-pan30.opus");
            encode(&out, &name, TARGET_RATE, REFERENCE_KBPS, &mix);
        }
    } else {
        println!("skipped: see the report above -- every value gives the same mix here");
    }

    println!("\n-- source --");
    let raw = mixdown::mix(
        &sources(
            &raw_mic,
            mic_offset,
            &speaker,
            speaker_offset,
            mixdown::PAN_POSITION,
        ),
        TARGET_RATE,
    );
    emit(&out, "raw", TARGET_RATE, REFERENCE_KBPS, 30, &raw);

    println!("\n-- rate --");
    match native_rate(&paths) {
        Ok(rate) => {
            let mic = read_native(&paths.mic_wav());
            let speaker = read_native(&paths.speaker_wav());
            let mix = mixdown::mix(
                &sources(
                    &mic,
                    mic_offset,
                    &speaker,
                    speaker_offset,
                    mixdown::PAN_POSITION,
                ),
                rate,
            );
            // The raw mic track, necessarily: `mic.cleaned.wav` only exists at 16 kHz, so this
            // arm answers "is the extra band worth it" with the echo still in. Read it as an
            // upper bound on what a native-rate mix would sound like, not as a candidate.
            for kbps in [REFERENCE_KBPS, 48] {
                emit(&out, "raw", rate, kbps, 30, &mix);
            }
        }
        Err(why) => println!("note: skipping the native-rate arm: {why}"),
    }
}

fn usage() -> ! {
    eprintln!("usage: session-mixdown <session-dir> [--out <dir>]");
    std::process::exit(2);
}

/// The two tracks as [`Source`]s, panned symmetrically about centre.
fn sources<'a>(
    mic: &'a [f32],
    mic_offset: f64,
    speaker: &'a [f32],
    speaker_offset: f64,
    pan: f32,
) -> [Source<'a>; 2] {
    [
        Source {
            samples: mic,
            offset_s: mic_offset,
            pan: -pan,
        },
        Source {
            samples: speaker,
            offset_s: speaker_offset,
            pan,
        },
    ]
}

/// Which of the two levelling sweeps is a comparison on this session rather than a row of
/// identical files.
///
/// Both arms can collapse, and for the same reason: a swept value only produces a different mix
/// when it changes some track's gain, and the boost cap and the loudness target each stop
/// mattering once the other one is doing all the work. A session whose quiet track wants more
/// boost than every cap in the sweep renders one file five times over in the target arm; a
/// session where no track is far enough under the target for any cap to bind renders one file
/// four times over in the boost arm.
struct Varies {
    /// Whether [`TARGETS_LUFS`] contains two values that mix differently.
    target: bool,
    /// Whether [`BOOST_CAPS_DB`] contains two values that mix differently.
    boost: bool,
}

/// What the shipping normalization measures on this session, printed before anything is
/// encoded, and which of the two levelling arms is worth rendering because of it.
///
/// Each track is measured for display and then again inside [`Normalization::gain`] for every
/// swept value, so that every number printed here is one the module will actually apply rather
/// than this file's restatement of how it is derived. A measurement is two biquads and a sum
/// over the track; the whole report costs less than one arm of the grid it decides about, which
/// is what makes running this against several sessions a reasonable way to find one worth
/// listening to.
fn report(tracks: &[(&str, &[f32])]) -> Varies {
    let shipping = Normalization::default();
    println!(
        "\n-- levels (target {:.0} LUFS, boost capped at {:.0} dB) --",
        mixdown::TARGET_LUFS,
        mixdown::MAX_BOOST_DB
    );

    for (name, samples) in tracks {
        let Some(lufs) = integrated_lufs(samples, TARGET_RATE) else {
            println!("{name:<8}  no measurable speech; passes through at unity");
            continue;
        };
        let wanted_db = mixdown::TARGET_LUFS - lufs;
        let applied_db = 20.0 * f64::from(shipping.gain(samples, TARGET_RATE)).log10();
        println!(
            "{name:<8} {lufs:>7.1} LUFS  wants {wanted_db:>+6.1} dB  gets {applied_db:>+6.1} dB  {}",
            if wanted_db > mixdown::MAX_BOOST_DB {
                "cap bound"
            } else {
                "uncapped"
            }
        );
    }

    let targets: Vec<Normalization> = TARGETS_LUFS
        .iter()
        .map(|&target_lufs| Normalization {
            target_lufs,
            ..Normalization::default()
        })
        .collect();
    let caps: Vec<Normalization> = BOOST_CAPS_DB
        .iter()
        .map(|&max_boost_db| Normalization {
            max_boost_db,
            ..Normalization::default()
        })
        .collect();
    let (target, boost) = (distinct(tracks, &targets), distinct(tracks, &caps));
    println!(
        "target arm: {target} of {} values mix differently here",
        targets.len()
    );
    println!(
        "boost arm:  {boost} of {} values mix differently here",
        caps.len()
    );

    Varies {
        target: target > 1,
        boost: boost > 1,
    }
}

/// How many of `sweep` produce a mix unlike the others'.
///
/// The gains are the whole difference between two settings -- everything downstream of them,
/// pan and sum and peak ceiling, is identical -- so counting distinct gain vectors counts
/// distinct files without encoding any. One means the arm has nothing to compare.
fn distinct(tracks: &[(&str, &[f32])], sweep: &[Normalization]) -> usize {
    let mut gains: Vec<Vec<u32>> = sweep
        .iter()
        .map(|normalization| {
            tracks
                .iter()
                .map(|(_, samples)| normalization.gain(samples, TARGET_RATE).to_bits())
                .collect()
        })
        .collect();
    gains.sort();
    gains.dedup();
    gains.len()
}

/// Encodes one cell of the grid, naming the file after the settings that produced it and
/// printing what it cost.
fn emit(out: &Path, source: &str, rate: u32, kbps: u32, pan: u32, mix: &[f32]) {
    let name = format!("{source}-{}k-{kbps}kbps-pan{pan:02}.opus", rate / 1000);
    encode(out, &name, rate, kbps, mix);
}

/// [`emit`] for the arms whose name is not `source`/rate/bitrate/pan, which build their own.
fn encode(out: &Path, name: &str, rate: u32, kbps: u32, mix: &[f32]) {
    let path = out.join(name);
    mixdown::write(&path, mix, rate, kbps * 1000).unwrap();

    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let seconds = mix.len() as f64 / 2.0 / f64::from(rate);
    println!(
        "{name:<40} {:>8.1} MB  {:>6.1} MB/hour",
        bytes as f64 / 1e6,
        bytes as f64 / 1e6 * 3600.0 / seconds.max(1e-9),
    );
}

fn read_metadata(paths: &SessionPaths) -> SessionMetadata {
    SessionMetadata::read(&paths.session_json()).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    })
}

/// The rate both source tracks are already at, when that is a rate Opus takes.
///
/// The mixdown cannot combine two tracks at different rates and nothing here resamples, so
/// this is a precondition rather than a preference.
fn native_rate(paths: &SessionPaths) -> Result<u32, String> {
    let rate = |path: PathBuf| -> Result<u32, String> {
        WavReader::open(&path)
            .map(|reader| reader.spec().sample_rate)
            .map_err(|e| format!("{}: {e}", path.display()))
    };
    let mic = rate(paths.mic_wav())?;
    let speaker = rate(paths.speaker_wav())?;
    if mic != speaker {
        return Err(format!("mic is {mic} Hz and speaker is {speaker} Hz"));
    }
    if ![8_000, 12_000, 16_000, 24_000, 48_000].contains(&mic) {
        return Err(format!("{mic} Hz is not a rate Opus encodes"));
    }
    if mic == TARGET_RATE {
        return Err("the tracks are already at 16 kHz".to_string());
    }
    Ok(mic)
}

/// A mono float track at whatever rate it was recorded at.
fn read_native(path: &Path) -> Vec<f32> {
    WavReader::open(path)
        .unwrap()
        .into_samples::<f32>()
        .map(|s| s.unwrap())
        .collect()
}

/// The interleaved mix, as a stereo WAV, so the encoder can be judged against its own input.
fn write_wav(path: &Path, stereo: &[f32], rate: u32) {
    let spec = WavSpec {
        channels: 2,
        sample_rate: rate,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    // hound directly rather than `meethook_session::wav`: that helper exists to correct the
    // channel mask of a *mono* header, and a stereo one hound writes is already right.
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for sample in stereo {
        writer.write_sample(*sample).unwrap();
    }
    writer.finalize().unwrap();
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    println!("{name:<40} (uncompressed)");
}
