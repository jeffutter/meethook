//! Prints microphone-activity edges without recording anything.
//!
//! `meethook record` cannot be used to check the *trigger* on its own: it needs both TCC
//! grants, it writes audio to disk, and a fault in the trigger looks exactly like a fault
//! in capture. This example isolates the trigger. It opens no device and needs no
//! permission, so a mismatch between what the watcher believes and what the meeting app is
//! doing shows up here with nothing else in the frame.
//!
//! ```text
//! MEETHOOK_ACTIVITY_DEBUG=1 cargo run --example mic-activity -- 300
//! ```
//!
//! What to look for, with a call joined and left while it runs:
//!
//! - a `Started` edge when the call opens the microphone, and a `Stopped` edge when it
//!   closes it;
//! - no edge at all across a mute/unmute toggle;
//! - this process's own pid listed and marked excluded in the debug lines.
//!
//! And, with the input device swapped while it runs -- System Settings > Sound > Input,
//! unplugging a USB interface, AirPods going away:
//!
//! - exactly one `InputDeviceChanged` per real swap, including the swap to *no* device at all,
//!   and none at startup or across the repeat `DefaultInputDevice` notifications macOS emits;
//! - the `IsRunningSomewhere listener attached to device N` debug line naming the new device.
//!
//! With an app named in `<root>/exclusions.json` (`$MEETHOOK_ROOT`, else `~/meethook`) opening
//! the microphone instead of a meeting app: no edge at all, and in the debug lines the app
//! marked `<- excluded: user-excluded bundle id` (or `executable`).
//!
//! That edge is what `meethook record` finalizes a session on, so this is where to check that
//! the notification arrives at all and arrives once. What this example cannot show is the other
//! half -- that `AVAudioEngine` really has stopped delivering buffers by then -- because it
//! opens no device; that needs a live session (TASK-011.01).
//!
//! This shows the *listener* path alone. It holds a watcher and never calls
//! `MicActivityWatcher::recheck`, so a release edge lost to a stale read shows up here as a
//! missing `Stopped` rather than as a late one -- which is what makes it the right place to
//! ask how often that race actually fires. The recovery re-check lives in the record loop,
//! where there is a live session for it to protect.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use meethook_record::MicActivityWatcher;

fn main() {
    let seconds: u64 = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(120);

    // The same root the binary resolves: `$MEETHOOK_ROOT`, else `~/meethook`. The watcher
    // reads `<root>/exclusions.json` from it once at start.
    let root = std::env::var_os("MEETHOOK_ROOT")
        .map(PathBuf::from)
        .or_else(|| std::env::home_dir().map(|home| home.join("meethook")))
        .unwrap_or_default();

    let (tx, rx) = mpsc::channel();
    let (_watcher, active) = match MicActivityWatcher::start(&root, move |activity| {
        let _ = tx.send(activity);
    }) {
        Ok(started) => started,
        Err(e) => {
            eprintln!("could not watch microphone activity: {e}");
            std::process::exit(1);
        }
    };

    let start = Instant::now();
    println!("pid {} watching for {seconds}s", std::process::id());
    println!("  +0.0s  already active: {active}");

    let deadline = start + Duration::from_secs(seconds);
    while let Ok(activity) = rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
        println!("  +{:.1}s  {activity:?}", start.elapsed().as_secs_f64());
    }
    println!("done");
}
