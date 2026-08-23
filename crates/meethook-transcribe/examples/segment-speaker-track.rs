//! Prints the speaker turns diarization finds in a recorded session, against the waveform.
//!
//! ```text
//! cargo run --example segment-speaker-track ~/meethook/sessions/20260809-021730
//! cargo run --example segment-speaker-track some-other-recording.wav
//! ```
//!
//! A session directory means its `speaker.wav`; any other path is used as given, which is
//! how a recording made somewhere else gets checked against the same eyes.
//!
//! Unit tests over synthetic logits prove the decoder; they cannot prove the timing. An
//! off-by-one-window error puts every turn ten seconds from where it belongs and passes all
//! of them. So this prints the turns next to a strip of the track's own loudness, at a
//! tenth of a second per column, and the check is whether the speaker rows start and stop
//! where the loudness does:
//!
//! ```text
//!    30.0  |....########.....###|
//!          |    000000000    0000|
//! ```
//!
//! Nothing on the normal `meethook transcribe` path calls this. It exists because "the
//! boundaries land in the right place" is a judgement a person has to make once, and this
//! is how they make it again after a change.

use std::path::PathBuf;

use meethook_transcribe::{LocalTurn, SEGMENTATION_MODEL, TARGET_RATE, open_session};

/// One column of the strip. Fine enough to see a turn edge, coarse enough that a minute of
/// meeting is ten lines.
const COLUMN_S: f64 = 0.1;
const COLUMNS_PER_LINE: usize = 60;

fn main() {
    let mut args = std::env::args_os().skip(1);
    let Some(target) = args.next().map(PathBuf::from) else {
        eprintln!("usage: segment-speaker-track <session-dir | wav-file>");
        std::process::exit(2);
    };

    let track = if target.is_dir() {
        target.join("speaker.wav")
    } else {
        target
    };
    let audio =
        meethook_transcribe::read_track_16k_mono(&track).unwrap_or_else(|e| fail(&format!("{e}")));

    let model = models_dir().join(SEGMENTATION_MODEL.file_name);
    let loaded = open_session(&model).unwrap_or_else(|e| {
        fail(&format!(
            "{e}\nrun `cargo run --example fetch-onnx-models` first"
        ))
    });
    // Only meaningful where a CoreML EP was compiled in: off macOS `accelerated` is always
    // false by construction of the build, so printing this there would name a component the
    // platform never had.
    #[cfg(target_os = "macos")]
    if !loaded.accelerated {
        eprintln!("note: CoreML declined this graph; running on CPU");
    }
    let mut session = loaded.session;

    let started = std::time::Instant::now();
    let turns = meethook_transcribe::segment_speaker_track(&audio, &mut session)
        .unwrap_or_else(|e| fail(&format!("{e}")));
    let elapsed = started.elapsed();

    let seconds = audio.len() as f64 / TARGET_RATE as f64;
    println!(
        "{}: {seconds:.1} s of audio, {} turns, segmented in {:.1} s",
        track.display(),
        turns.len(),
        elapsed.as_secs_f64()
    );
    for turn in &turns {
        println!(
            "  {:>8.2} -> {:>8.2}  ({:>5.2} s)  window {:>3}  local speaker {}",
            turn.start_s,
            turn.end_s,
            turn.end_s - turn.start_s,
            turn.window,
            turn.local_speaker
        );
    }

    println!();
    print_strip(&audio, &turns);
}

/// Prints loudness and turn occupancy on the same time axis.
fn print_strip(audio: &[f32], turns: &[LocalTurn]) {
    let per_column = (COLUMN_S * TARGET_RATE as f64) as usize;
    let columns = audio.len().div_ceil(per_column);

    // Loudness as a peak per column, on a coarse log scale: speech and silence differ by
    // tens of dB, so a linear scale would show one blob and one flat line.
    let loudness: Vec<char> = (0..columns)
        .map(|c| {
            let block = &audio[c * per_column..audio.len().min((c + 1) * per_column)];
            let peak = block.iter().fold(0.0f32, |m, s| m.max(s.abs()));
            match peak {
                p if p < 0.001 => ' ',
                p if p < 0.01 => '.',
                p if p < 0.05 => ':',
                p if p < 0.2 => '#',
                _ => '@',
            }
        })
        .collect();

    // The local speaker index active in each column, or a space. Overlap shows as whichever
    // index is lowest, which is enough for a timing check.
    let mut occupancy = vec![' '; columns];
    for turn in turns {
        let from = (turn.start_s / COLUMN_S) as usize;
        let to = (turn.end_s / COLUMN_S).ceil() as usize;
        for column in occupancy.iter_mut().take(to.min(columns)).skip(from) {
            let digit = char::from_digit(turn.local_speaker as u32, 10).unwrap_or('?');
            if *column == ' ' || *column > digit {
                *column = digit;
            }
        }
    }

    for line in 0..columns.div_ceil(COLUMNS_PER_LINE) {
        let range = line * COLUMNS_PER_LINE..columns.min((line + 1) * COLUMNS_PER_LINE);
        let at_s = range.start as f64 * COLUMN_S;
        println!(
            "{at_s:>8.1}  |{}|",
            loudness[range.clone()].iter().collect::<String>()
        );
        println!(
            "          |{}|",
            occupancy[range].iter().collect::<String>()
        );
    }
}

fn models_dir() -> PathBuf {
    match std::env::var_os("MEETHOOK_ROOT") {
        Some(root) => PathBuf::from(root),
        None => std::env::home_dir()
            .expect("could not determine the home directory; set MEETHOOK_ROOT")
            .join("meethook"),
    }
    .join("models")
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}
