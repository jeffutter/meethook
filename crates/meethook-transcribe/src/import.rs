//! Turning wav files meethook did not record into a session directory it will accept.
//!
//! Everything downstream of `record` -- `transcribe`, `enroll`, `cluster-speaker-track` --
//! works from a session directory, so until now the only way to get a voice into
//! `speakers.json` was to hold a meeting. That makes measuring the thing the enrolled-speaker
//! threshold is calibrated against (one person's stored reference versus that same person in
//! a *different* recording, and versus somebody else) cost a recording sitting per attempt,
//! for a question the audio does not actually need meethook to have captured.
//!
//! [`build_session`] closes that gap. Given any wav files `hound` can decode, it writes a
//! directory that classifies as [`Classification::Valid`](meethook_session::Classification)
//! and that the real `transcribe` and `enroll` code paths then treat like any other
//! recording. Nothing here is a shortcut around those paths: the point is that the reference
//! which comes out the far end was produced by the same diarization, clustering and
//! enrollment a meeting goes through, rather than hand-written into JSON.
//!
//! # The shape of a constructed session
//!
//! **The supplied audio goes on `speaker.wav`, and `mic.wav` gets digital silence.** Both
//! files must exist -- [`crate::transcribe_session`] treats an unreadable `mic.wav` as fatal
//! while an absent `speaker.wav` merely degrades to a mic-only transcript -- but only one of
//! them can carry a measurement, because diarization and embedding are run over the speaker
//! track and nothing else. Audio placed on the mic track could never reach clustering,
//! enrollment or `cluster-speaker-track`.
//!
//! `session.json` records the same `host_ticks` for both tracks, which makes both of
//! `transcribe`'s offsets zero, so transcript timestamps are audio timestamps with nothing
//! added. That is what anybody reading a measurement wants, and it is honest: there was no
//! capture clock here to be offset from.
//!
//! Several sources are concatenated in the order given, separated by [`SPLICE_GAP_S`] of
//! silence. The gap is not decoration -- it is wider than segmentation's own
//! [`MAX_GAP_IN_TURN_S`], without which a splice between two people would be read as one
//! continuous turn spanning both and quietly corrupt the clustering that follows.

use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavReader};
use jiff::{Timestamp, Zoned};
use meethook_session::{
    Paths, SessionId, SessionMetadata, SessionPaths, TrackSync, create_session_dir,
};

use crate::audio::{self, TARGET_RATE};
use crate::levels::LevelSummary;
use crate::segmentation::MAX_GAP_IN_TURN_S;
use crate::{Error, Result};

/// Silence spliced between two consecutive source files, in seconds.
///
/// Written against `MAX_GAP_IN_TURN_S` rather than as a loose number, because the only
/// thing that matters about it is that it is comfortably the larger of the two: a shorter
/// gap lets segmentation join the last turn of one file to the first turn of the next, and
/// two different people merged into one turn produce an embedding of neither.
pub const SPLICE_GAP_S: f64 = MAX_GAP_IN_TURN_S * 2.0;

/// How much digital silence `mic.wav` gets when the caller supplies no local track.
///
/// One second rather than zero: below the `asr` module's minimum, whisper.cpp pads the buffer
/// itself, and a track that exists is the difference between a session `transcribe` reads and
/// one it refuses. A silent mic against a real speaker track makes the echo-cancellation
/// pre-pass find no measurable lag and write `mic.cleaned.wav` as a pass-through, which is
/// the correct and boring outcome.
pub const MIC_SILENCE_S: f64 = 1.0;

/// Apple Silicon's `mach_timebase_info`, matching what the recorder writes.
///
/// The actual value is immaterial here -- both tracks carry the same `host_ticks`, so every
/// derived offset is zero whatever the ratio -- but it must be a sane one: `transcribe`
/// rejects a degenerate timebase, and a session nobody can read is not a useful fixture.
const TIMEBASE_NUMER: u32 = 125;
const TIMEBASE_DENOM: u32 = 3;

/// What one source file contributed, so a run can *show* its conversion instead of leaving
/// the reader to assume one happened.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportedSource {
    pub path: PathBuf,
    /// The rate in the file's own header.
    pub sample_rate: u32,
    pub channels: u16,
    /// Frames in the file, at its own rate.
    pub frames: usize,
    /// Frames it contributed to the track, at [`TARGET_RATE`], excluding any splice gap.
    pub samples: usize,
}

/// A session directory assembled from foreign audio, and everything that went into it.
///
/// The two [`LevelSummary`] values are the reason this is a struct rather than a
/// [`SessionId`]: they answer "is there actually anything on this track" for a few
/// microseconds, before a whisper pass over a dead recording costs minutes and comes back as
/// an empty transcript that looks like a model failure.
#[derive(Debug, Clone, PartialEq)]
pub struct BuiltSession {
    pub id: SessionId,
    pub paths: SessionPaths,
    pub speaker_sources: Vec<ImportedSource>,
    /// Empty when `mic.wav` is the constructed silence rather than supplied audio.
    pub mic_sources: Vec<ImportedSource>,
    pub speaker: LevelSummary,
    pub mic: LevelSummary,
}

/// Builds a session directory under `root` from the given wav files.
///
/// `speaker_sources` is the audio being measured and must not be empty. `mic_sources` is the
/// local half of a two-track recording, for the rare caller that has one; leaving it empty
/// writes [`MIC_SILENCE_S`] of silence instead, which is what every measurement wants.
///
/// Every file this writes lands inside `root`, and it creates exactly one directory:
/// `<root>/sessions/<id>/`. Nothing consults `~/meethook`, and nothing outside `root` is
/// read except the sources themselves, so pointing this at a scratch directory cannot
/// disturb a real set of recordings.
///
/// All of the audio is decoded before the directory is created, so a source that turns out
/// to be unreadable leaves nothing behind at all rather than a half-built session the next
/// `transcribe` would try to work with.
pub fn build_session(
    root: &Paths,
    speaker_sources: &[PathBuf],
    mic_sources: &[PathBuf],
) -> Result<BuiltSession> {
    let (speaker, speaker_summaries) = concatenate(speaker_sources)?;
    let (mic, mic_summaries) = if mic_sources.is_empty() {
        (
            vec![0.0; (f64::from(TARGET_RATE) * MIC_SILENCE_S) as usize],
            Vec::new(),
        )
    } else {
        concatenate(mic_sources)?
    };

    // Creating the directory *is* the collision check, and it is the one place this function
    // writes, which is what makes the "only inside `root`" claim structural rather than
    // careful.
    let (id, paths) = create_session_dir(root, &Zoned::now())?;

    audio::write_track_16k_mono(&paths.speaker_wav(), &speaker)?;
    audio::write_track_16k_mono(&paths.mic_wav(), &mic)?;

    // Equal ticks on both tracks: neither started before the other, because neither was
    // captured. `session.json` is written last, since its presence is what marks the session
    // complete -- an interrupted build leaves an orphan, which every command already skips.
    let sync = TrackSync {
        host_ticks: 0,
        timebase_numer: TIMEBASE_NUMER,
        timebase_denom: TIMEBASE_DENOM,
    };
    SessionMetadata::new(id.clone(), Timestamp::now(), sync, sync).write(&paths.session_json())?;

    Ok(BuiltSession {
        speaker: LevelSummary::measure(&speaker, TARGET_RATE),
        mic: LevelSummary::measure(&mic, TARGET_RATE),
        id,
        paths,
        speaker_sources: speaker_summaries,
        mic_sources: mic_summaries,
    })
}

/// One track's worth of audio, read in argument order with [`SPLICE_GAP_S`] between files.
fn concatenate(sources: &[PathBuf]) -> Result<(Vec<f32>, Vec<ImportedSource>)> {
    if sources.is_empty() {
        return Err(Error::NoAudio {
            detail: "no source files were given".to_string(),
        });
    }

    let gap = vec![0.0f32; (f64::from(TARGET_RATE) * SPLICE_GAP_S) as usize];
    let mut track = Vec::new();
    let mut summaries = Vec::with_capacity(sources.len());

    for (index, path) in sources.iter().enumerate() {
        if index > 0 {
            track.extend_from_slice(&gap);
        }
        let (samples, summary) = read_any_wav_16k_mono(path)?;
        track.extend_from_slice(&samples);
        summaries.push(summary);
    }

    Ok((track, summaries))
}

/// Decodes any wav `hound` understands into 16 kHz mono on the ±1.0 float scale.
///
/// Deliberately *not* [`audio::read_track_16k_mono`], which stays exactly as strict as it is:
/// that function guards files the recorder itself wrote, where a stereo or integer header
/// means something has gone wrong and reinterpreting it would produce noise that looks like a
/// transcription bug. Here a foreign header is the normal case, and the conversion is
/// reported rather than hidden.
fn read_any_wav_16k_mono(path: &Path) -> Result<(Vec<f32>, ImportedSource)> {
    let reader = WavReader::open(path).map_err(|e| Error::wav(path, e))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1);

    if spec.sample_rate == 0 {
        return Err(Error::UnsupportedAudio {
            path: path.to_path_buf(),
            detail: "the header reports a sample rate of 0".to_string(),
        });
    }

    // Integer formats are scaled by their own full scale rather than refused -- the same
    // arithmetic `examples/track-levels.rs` does -- because 16-bit PCM is what most corpora
    // ship, and refusing it would make this tool useless for the audio it exists to import.
    let mut interleaved = Vec::with_capacity(reader.len() as usize);
    match spec.sample_format {
        SampleFormat::Float => {
            for sample in reader.into_samples::<f32>() {
                interleaved.push(sample.map_err(|e| Error::wav(path, e))?);
            }
        }
        SampleFormat::Int => {
            let full_scale = (1i64 << (spec.bits_per_sample.max(1) - 1)) as f32;
            for sample in reader.into_samples::<i32>() {
                interleaved.push(sample.map_err(|e| Error::wav(path, e))? as f32 / full_scale);
            }
        }
    }

    // Averaged rather than "take the first channel": a recording that put one talker on each
    // side, or the speech on the right, would otherwise import as half a conversation or as
    // silence.
    let mono: Vec<f32> = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks_exact(usize::from(channels))
            .map(|frame| frame.iter().sum::<f32>() / f32::from(channels))
            .collect()
    };

    if mono.is_empty() {
        return Err(Error::NoAudio {
            detail: format!("{} holds no audio samples", path.display()),
        });
    }

    let frames = mono.len();
    let samples = audio::resample_to_target(&mono, spec.sample_rate)?;

    Ok((
        samples.clone(),
        ImportedSource {
            path: path.to_path_buf(),
            sample_rate: spec.sample_rate,
            channels,
            frames,
            samples: samples.len(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use hound::{WavSpec, WavWriter};
    use meethook_session::{Classification, discover_sessions};

    use super::*;

    /// Every fixture is generated at run time into a temporary directory. No audio file of
    /// any kind belongs in this repository: third-party corpus audio and real meeting
    /// recordings are both other people speaking.
    fn write_wav(path: &Path, spec: WavSpec, samples: &[f32]) {
        let mut writer = WavWriter::create(path, spec).unwrap();
        for sample in samples {
            match spec.sample_format {
                SampleFormat::Float => writer.write_sample(*sample).unwrap(),
                SampleFormat::Int => {
                    let full_scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
                    writer.write_sample((*sample * full_scale) as i32).unwrap();
                }
            }
        }
        writer.finalize().unwrap();
    }

    fn float_spec(rate: u32, channels: u16) -> WavSpec {
        WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        }
    }

    fn int_spec(rate: u32, channels: u16) -> WavSpec {
        WavSpec {
            channels,
            sample_rate: rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        }
    }

    /// `seconds` of a tone, interleaved across `channels` identical copies.
    fn tone(rate: u32, channels: u16, seconds: f64, hz: f32) -> Vec<f32> {
        let frames = (f64::from(rate) * seconds) as usize;
        (0..frames)
            .flat_map(|i| {
                let sample = (i as f32 / rate as f32 * hz * std::f32::consts::TAU).sin() * 0.5;
                std::iter::repeat_n(sample, usize::from(channels))
            })
            .collect()
    }

    /// Every path under `root`, relative and slash-joined, so a test can state exactly what a
    /// build was allowed to create.
    fn tree(root: &Path) -> BTreeSet<String> {
        fn walk(dir: &Path, root: &Path, found: &mut BTreeSet<String>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                found.insert(
                    path.strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                );
                if path.is_dir() {
                    walk(&path, root, found);
                }
            }
        }
        let mut found = BTreeSet::new();
        walk(root, root, &mut found);
        found
    }

    /// The claim `transcribe` and `enroll` both depend on: what comes out is an ordinary
    /// valid session, whatever the source header said.
    ///
    /// Run over 48 kHz stereo 16-bit PCM -- what a corpus ships -- and over the 16 kHz mono
    /// float the recorder itself writes, because those are the two ends of the conversion and
    /// the pass-through end is the one a refactor can silently break.
    #[test]
    fn a_built_session_is_discovered_as_valid_and_reads_back_through_the_strict_reader() {
        for spec in [int_spec(48_000, 2), float_spec(TARGET_RATE, 1)] {
            let dir = tempfile::tempdir().unwrap();
            let source = dir.path().join("voice.wav");
            write_wav(
                &source,
                spec,
                &tone(spec.sample_rate, spec.channels, 2.0, 440.0),
            );

            let root = tempfile::tempdir().unwrap();
            let paths = Paths::new(root.path());
            let built = build_session(&paths, &[source], &[]).unwrap();

            let discovered = discover_sessions(&paths).unwrap();
            assert_eq!(discovered.len(), 1, "{spec:?}");
            assert_eq!(discovered[0].id, built.id, "{spec:?}");
            assert_eq!(
                discovered[0].classification,
                Classification::Valid,
                "{spec:?}"
            );

            // `session.json` round-trips, and both offsets come out zero.
            let metadata = discovered[0].load_metadata().unwrap();
            assert_eq!(metadata.session_id, built.id, "{spec:?}");
            assert_eq!(metadata.mic.host_ticks, metadata.speaker.host_ticks);
            assert_eq!(crate::mic_offset_seconds(&metadata).unwrap(), 0.0);
            assert_eq!(crate::speaker_offset_seconds(&metadata).unwrap(), 0.0);

            // The strict reader is what `transcribe` opens both tracks with, so a track it
            // refuses is a session nothing can transcribe.
            let speaker = audio::read_track_16k_mono(&built.paths.speaker_wav()).unwrap();
            let mic = audio::read_track_16k_mono(&built.paths.mic_wav()).unwrap();
            assert_eq!(speaker.len(), TARGET_RATE as usize * 2, "{spec:?}");
            assert_eq!(mic.len(), TARGET_RATE as usize, "{spec:?}");
            assert!(mic.iter().all(|s| *s == 0.0), "{spec:?}");
        }
    }

    /// The conversion is reported per source, which is what makes a resample visible rather
    /// than assumed when somebody is reading a measurement back later.
    #[test]
    fn each_source_reports_the_header_it_came_from_and_what_it_contributed() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("voice.wav");
        write_wav(&source, int_spec(48_000, 2), &tone(48_000, 2, 1.0, 440.0));

        let root = tempfile::tempdir().unwrap();
        let built =
            build_session(&Paths::new(root.path()), std::slice::from_ref(&source), &[]).unwrap();

        assert_eq!(
            built.speaker_sources,
            [ImportedSource {
                path: source,
                sample_rate: 48_000,
                channels: 2,
                frames: 48_000,
                samples: TARGET_RATE as usize,
            }]
        );
        assert!(
            built.mic_sources.is_empty(),
            "the mic track was constructed"
        );
    }

    /// Three sources means three clips and two gaps, and the gap has to be the one
    /// segmentation will not join across.
    #[test]
    fn sources_are_concatenated_in_order_with_a_splice_gap_between_them() {
        let dir = tempfile::tempdir().unwrap();
        let sources: Vec<PathBuf> = (0..3u16)
            .map(|i| {
                let path = dir.path().join(format!("voice{i}.wav"));
                write_wav(
                    &path,
                    float_spec(TARGET_RATE, 1),
                    &tone(TARGET_RATE, 1, 1.0, 300.0 + f32::from(i) * 100.0),
                );
                path
            })
            .collect();

        let root = tempfile::tempdir().unwrap();
        let built = build_session(&Paths::new(root.path()), &sources, &[]).unwrap();

        let gap = (f64::from(TARGET_RATE) * SPLICE_GAP_S) as usize;
        // The property that makes the gap worth writing at all, checked at compile time so
        // that narrowing either constant fails the build rather than one test: a splice no
        // wider than segmentation's own tolerance would be read as one continuous turn
        // spanning both sources, and embed as neither speaker.
        const { assert!(SPLICE_GAP_S > MAX_GAP_IN_TURN_S) };

        let speaker = audio::read_track_16k_mono(&built.paths.speaker_wav()).unwrap();
        assert_eq!(speaker.len(), TARGET_RATE as usize * 3 + gap * 2);
        // The gaps land exactly where the arithmetic says, which is the part clustering
        // depends on.
        for splice in 1..3 {
            let start = splice * (TARGET_RATE as usize + gap) - gap;
            assert!(
                speaker[start..start + gap].iter().all(|s| *s == 0.0),
                "splice {splice} should be silent"
            );
        }
    }

    /// Acceptance criterion #5, the failing half. A zero-length source cannot produce a turn,
    /// a cluster or an embedding, so a session built from it would look exactly like a
    /// successful run that found nobody -- the misleading result worth refusing.
    #[test]
    fn a_zero_length_source_fails_by_name_rather_than_building_a_silent_session() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("empty.wav");
        write_wav(&source, float_spec(TARGET_RATE, 1), &[]);

        let root = tempfile::tempdir().unwrap();
        let error = build_session(&Paths::new(root.path()), std::slice::from_ref(&source), &[])
            .unwrap_err();

        let message = error.to_string();
        assert!(message.contains(&source.display().to_string()), "{message}");
        assert!(message.contains("no audio"), "{message}");
        assert!(
            tree(root.path()).is_empty(),
            "a build that failed must leave no session behind"
        );
    }

    /// A header claiming 0 Hz is a corrupt file, not a resampling ratio. Named rather than
    /// left to divide by zero somewhere inside the resampler.
    #[test]
    fn a_source_with_no_sample_rate_is_named_rather_than_resampled() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("broken.wav");
        write_wav(&source, float_spec(1, 1), &[0.1; 64]);
        // hound refuses to *write* a 0 Hz header, so it is patched into the bytes: this is a
        // file somebody else produced, which is the whole category being guarded against.
        // The byte rate goes with it -- hound rejects a fmt chunk whose byte rate is not the
        // sample rate times the block align before it ever reports the rate itself, and the
        // guard under test is the one that fires on a *well-formed* header claiming 0 Hz.
        let mut bytes = std::fs::read(&source).unwrap();
        let fmt = bytes
            .windows(4)
            .position(|w| w == b"fmt ")
            .expect("every wav file has a fmt chunk");
        bytes[fmt + 12..fmt + 16].copy_from_slice(&0u32.to_le_bytes()); // sample rate
        bytes[fmt + 16..fmt + 20].copy_from_slice(&0u32.to_le_bytes()); // byte rate
        std::fs::write(&source, &bytes).unwrap();

        let root = tempfile::tempdir().unwrap();
        let error = build_session(&Paths::new(root.path()), std::slice::from_ref(&source), &[])
            .unwrap_err();

        let message = error.to_string();
        assert!(message.contains(&source.display().to_string()), "{message}");
        assert!(message.contains("sample rate of 0"), "{message}");
    }

    /// Silence is not an error -- a corpus clip with a long lead-in is normal -- but it is
    /// reported, so "the measurement came back empty" can be read as "the input was dead"
    /// before a whisper pass is paid for rather than after.
    #[test]
    fn a_silent_source_builds_and_says_so_in_its_levels() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("silence.wav");
        write_wav(&source, float_spec(TARGET_RATE, 1), &[0.0; 16_000]);

        let root = tempfile::tempdir().unwrap();
        let built = build_session(&Paths::new(root.path()), &[source], &[]).unwrap();

        assert_eq!(built.speaker.peak, 0.0);
        assert_eq!(built.speaker.above_fraction(), 0.0);
        assert_eq!(built.speaker.longest_run, 0);
        assert!((built.speaker.duration_s() - 1.0).abs() < 1e-9);
        assert!(built.speaker.peak_dbfs().is_infinite());
    }

    /// Acceptance criterion #2, made structural: the only thing a build creates anywhere
    /// under the root it was handed is one session directory and the three files in it.
    #[test]
    fn a_build_creates_one_session_directory_under_the_root_and_nothing_else() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("voice.wav");
        write_wav(
            &source,
            float_spec(TARGET_RATE, 1),
            &tone(TARGET_RATE, 1, 0.5, 440.0),
        );

        let root = tempfile::tempdir().unwrap();
        let built = build_session(&Paths::new(root.path()), &[source], &[]).unwrap();

        let id = built.id.as_str();
        assert_eq!(
            tree(root.path()),
            BTreeSet::from([
                "sessions".to_string(),
                format!("sessions/{id}"),
                format!("sessions/{id}/mic.wav"),
                format!("sessions/{id}/session.json"),
                format!("sessions/{id}/speaker.wav"),
            ])
        );
    }

    /// A caller with a genuine two-track recording can supply both, and the mic track is then
    /// audio rather than the constructed silence.
    #[test]
    fn a_supplied_mic_track_replaces_the_constructed_silence() {
        let dir = tempfile::tempdir().unwrap();
        let far = dir.path().join("far.wav");
        let near = dir.path().join("near.wav");
        write_wav(
            &far,
            float_spec(TARGET_RATE, 1),
            &tone(TARGET_RATE, 1, 1.0, 440.0),
        );
        write_wav(
            &near,
            float_spec(TARGET_RATE, 1),
            &tone(TARGET_RATE, 1, 0.5, 220.0),
        );

        let root = tempfile::tempdir().unwrap();
        let built = build_session(
            &Paths::new(root.path()),
            &[far],
            std::slice::from_ref(&near),
        )
        .unwrap();

        assert_eq!(built.mic_sources.len(), 1);
        assert_eq!(built.mic_sources[0].path, near);
        assert_eq!(built.mic.samples, TARGET_RATE as usize / 2);
        assert!(built.mic.peak > 0.4, "{:?}", built.mic);
    }

    #[test]
    fn building_from_no_sources_at_all_says_so_rather_than_writing_an_empty_session() {
        let root = tempfile::tempdir().unwrap();
        let error = build_session(&Paths::new(root.path()), &[], &[]).unwrap_err();

        assert!(error.to_string().contains("no source files"), "{error}");
        assert!(tree(root.path()).is_empty());
    }

    /// Two builds against the same root are two sessions, not one overwriting the other --
    /// which is what a cross-session measurement needs, and it comes from `create_session_dir`
    /// rather than from anything here.
    #[test]
    fn two_builds_against_one_root_produce_two_distinct_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("voice.wav");
        write_wav(
            &source,
            float_spec(TARGET_RATE, 1),
            &tone(TARGET_RATE, 1, 0.5, 440.0),
        );

        let root = tempfile::tempdir().unwrap();
        let paths = Paths::new(root.path());
        let first = build_session(&paths, std::slice::from_ref(&source), &[]).unwrap();
        let second = build_session(&paths, &[source], &[]).unwrap();

        assert_ne!(first.id, second.id);
        assert_eq!(discover_sessions(&paths).unwrap().len(), 2);
    }
}
