//! Holds the default input device open for N seconds, standing in for a meeting app.
//!
//! ```text
//! cargo run --example mic-hold -- 20
//! ```
//!
//! The activity trigger looks for *another process* capturing input, so checking it has so
//! far meant joining a real call: the one thing an automated run cannot do and a person can
//! only do slowly. This is that other process, on demand and in a known state. Run it in a
//! second terminal against `meethook record` (or `--example mic-activity`) and the whole
//! start/stop matrix becomes two terminals and thirty seconds.
//!
//! The reason it matters most is the hazard the trigger's own pid filter cannot cover:
//! ScreenCaptureKit services system audio out of process, so a helper woken up by *our*
//! capture could register as a foreign process capturing input and pin the predicate true
//! forever. That is only visible while meethook is actually recording, which is exactly the
//! situation this example puts it in without a meeting.
//!
//! What to look for, with `MEETHOOK_ACTIVITY_DEBUG=1 meethook record` running alongside:
//! while recording, the only pids reported at `IsRunningInput=true` should be this process's
//! and meethook's own (shown as excluded). A third one is the hazard happening, and the
//! session it opens will never end.
//!
//! It needs the microphone TCC grant and nothing else. On a machine with *no* audio devices
//! at all -- which a sandbox can produce, and a real Mac cannot -- `AVAudioEngine`'s
//! `inputNode` raises an Objective-C exception that aborts the process before any of this
//! runs. `mic.rs` reaches the same call the same way, so that is a property of the machine
//! rather than of this harness.
//!
//! Deliberately shares no code with
//! `mic.rs`: the only reusable piece there is `InputDevice`, whose `start` is inseparable
//! from a WAV writer, and a harness whose whole job is to discard every buffer should not be
//! writing a file nobody reads.

use std::ptr::NonNull;
use std::time::Duration;

use block2::RcBlock;
use objc2_avf_audio::{AVAudioEngine, AVAudioNodeTapBlock, AVAudioPCMBuffer, AVAudioTime};

/// Matches the recorder's tap size. Nothing here depends on it -- every buffer is dropped --
/// but a wildly different size would make this a less faithful stand-in.
const TAP_BUFFER_FRAMES: u32 = 4096;

fn main() {
    let seconds: u64 = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(20);

    // SAFETY: `inputNode` is a plain property read, always present on macOS; its *format* is
    // what reveals whether a usable device is behind it.
    let engine = unsafe { AVAudioEngine::new() };
    let input = unsafe { engine.inputNode() };
    let format = unsafe { input.outputFormatForBus(0) };

    // SAFETY: plain property reads on a live format object.
    let sample_rate = unsafe { format.sampleRate() };
    let channels = unsafe { format.channelCount() };

    // A zero-ish format is a missing device or a missing microphone grant. Tapping it would
    // "succeed" while holding nothing, which would look to the watcher exactly like a trigger
    // that does not fire -- the one conclusion this harness must never fabricate.
    if sample_rate.is_nan() || sample_rate <= 0.0 || channels == 0 {
        eprintln!(
            "the default input device reported an unusable format ({sample_rate} Hz, \
             {channels} channels); check the microphone permission and \
             System Settings > Sound > Input"
        );
        std::process::exit(1);
    }

    // Every buffer is discarded. The tap exists only because an engine with no tap on its
    // input node does not open the device, and opening the device is the entire point.
    let tap = RcBlock::new(|_buffer: NonNull<AVAudioPCMBuffer>, _when: NonNull<AVAudioTime>| {});

    // SAFETY: the block is retained by the framework and kept alive by `tap`, which outlives
    // the `removeTapOnBus` below.
    unsafe {
        let block: AVAudioNodeTapBlock = (&*tap as *const block2::DynBlock<_>).cast_mut();
        input.installTapOnBus_bufferSize_format_block(0, TAP_BUFFER_FRAMES, Some(&format), block);
    }

    // SAFETY: the graph is a bare input tap, which needs no output connection on macOS.
    unsafe { engine.prepare() };
    if let Err(e) = unsafe { engine.startAndReturnError() } {
        eprintln!(
            "could not start the input engine: {}",
            e.localizedDescription()
        );
        std::process::exit(1);
    }

    println!(
        "pid {} holding the default input device ({sample_rate} Hz, {channels} channel(s)) \
         for {seconds}s",
        std::process::id()
    );
    std::thread::sleep(Duration::from_secs(seconds));

    // Stop before removing the tap, matching the recorder: the reverse order leaves a window
    // in which a callback can fire into a torn-down tap.
    // SAFETY: both are live objects owned by this function.
    unsafe {
        engine.stop();
        input.removeTapOnBus(0);
    }
    println!("released");
}
