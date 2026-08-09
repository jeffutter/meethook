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

use std::sync::mpsc;
use std::time::{Duration, Instant};

use meethook_record::MicActivityWatcher;

fn main() {
    let seconds: u64 = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(120);

    let (tx, rx) = mpsc::channel();
    let (_watcher, active) = match MicActivityWatcher::start(move |activity| {
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
