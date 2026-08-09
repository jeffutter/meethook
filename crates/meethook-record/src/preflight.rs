//! macOS privacy (TCC) preflight.
//!
//! Discovering a permission problem *after* a meeting has already gone unrecorded is the
//! failure mode this module exists to prevent. It runs before a session directory is
//! created, checks both permissions in one pass, and reports every missing one at once --
//! reporting them one at a time would make the user grant a permission, re-run, and get
//! told about the next one.

use std::fmt;
use std::sync::mpsc;
use std::time::Duration;

use block2::RcBlock;
use objc2::runtime::Bool;
use objc2_av_foundation::{AVAuthorizationStatus, AVCaptureDevice, AVMediaTypeAudio};
use objc2_core_graphics::{CGPreflightScreenCaptureAccess, CGRequestScreenCaptureAccess};

use crate::Result;

/// How long to wait for the user to answer the microphone prompt.
///
/// Long, because the answer is a human decision, but finite, because a wedged recorder is
/// worse than one that fails with a message.
const PROMPT_TIMEOUT: Duration = Duration::from_secs(120);

/// Proof that both required permissions were granted.
///
/// Deliberately unconstructable outside this module: [`crate::Recorder::new`] demands one,
/// which is what turns "preflight runs before anything is written to disk" from an ordering
/// convention into a compile-time guarantee.
#[derive(Debug)]
pub struct Authorized(());

/// One or both TCC grants are missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MissingPermissions {
    pub screen_recording: bool,
    pub microphone: bool,
}

impl std::error::Error for MissingPermissions {}

impl fmt::Display for MissingPermissions {
    /// Names the exact System Settings panes, and explains the terminal-inheritance trap.
    ///
    /// A non-bundled CLI inherits TCC grants from the terminal application that launched it,
    /// so the user will be looking for a "meethook" entry that macOS will never create.
    /// Saying so here is the difference between a two-minute fix and a missed meeting.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "meethook record cannot start: missing macOS permissions."
        )?;
        writeln!(f)?;
        if self.screen_recording {
            writeln!(
                f,
                "  - Screen & System Audio Recording (needed to capture the other participants)"
            )?;
            writeln!(
                f,
                "      System Settings > Privacy & Security > Screen & System Audio Recording"
            )?;
        }
        if self.microphone {
            writeln!(f, "  - Microphone (needed to capture your own voice)")?;
            writeln!(f, "      System Settings > Privacy & Security > Microphone")?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "meethook is a command-line tool, so macOS attributes these permissions to the"
        )?;
        writeln!(
            f,
            "terminal application you launched it from -- grant them to that app, not to a"
        )?;
        write!(
            f,
            "\"meethook\" entry. You may need to quit and reopen the terminal afterwards."
        )
    }
}

/// Checks screen-recording and microphone authorization, prompting where macOS has never
/// asked, and returns proof of success.
///
/// A `NotDetermined` status is a missing prompt, not a denial: a tool that refused to record
/// because it never asked would be a bug. Only an actual refusal (or a restriction the user
/// cannot lift) is an error.
pub fn preflight() -> Result<Authorized> {
    let screen_ok = screen_recording_authorized();
    let microphone_ok = microphone_authorized();

    if screen_ok && microphone_ok {
        return Ok(Authorized(()));
    }

    Err(MissingPermissions {
        screen_recording: !screen_ok,
        microphone: !microphone_ok,
    }
    .into())
}

/// `CGPreflightScreenCaptureAccess` never prompts, so a `false` is followed by one explicit
/// request and a re-check.
fn screen_recording_authorized() -> bool {
    if CGPreflightScreenCaptureAccess() {
        return true;
    }
    // Returns the post-prompt answer directly; the re-check covers the case where the
    // system has already recorded a denial and returns without showing anything.
    CGRequestScreenCaptureAccess() || CGPreflightScreenCaptureAccess()
}

fn microphone_authorized() -> bool {
    match microphone_status() {
        AVAuthorizationStatus::Authorized => true,
        AVAuthorizationStatus::NotDetermined => {
            request_microphone_access();
            microphone_status() == AVAuthorizationStatus::Authorized
        }
        // Denied or Restricted. Requesting again would not prompt.
        _ => false,
    }
}

fn microphone_status() -> AVAuthorizationStatus {
    // SAFETY: `AVMediaTypeAudio` is one of the two media types this API accepts; anything
    // else would raise an Objective-C exception.
    unsafe {
        let media_type =
            AVMediaTypeAudio.expect("AVMediaTypeAudio is a non-null framework constant");
        AVCaptureDevice::authorizationStatusForMediaType(media_type)
    }
}

/// Triggers the OS prompt and waits for the answer.
///
/// The completion block fires on an arbitrary dispatch queue, so the answer comes back over
/// a channel rather than by mutating shared state. A timeout means an unanswered prompt
/// degrades to a clear permission error instead of hanging the process forever.
fn request_microphone_access() {
    let (tx, rx) = mpsc::channel::<bool>();
    let handler = RcBlock::new(move |granted: Bool| {
        let _ = tx.send(granted.as_bool());
    });

    // SAFETY: `AVMediaTypeAudio` is an accepted media type, and `handler` outlives the call
    // because `RcBlock` retains it and this function blocks until the block has fired or the
    // wait times out.
    unsafe {
        let media_type =
            AVMediaTypeAudio.expect("AVMediaTypeAudio is a non-null framework constant");
        AVCaptureDevice::requestAccessForMediaType_completionHandler(media_type, &handler);
    }

    // The answer itself is ignored: `microphone_authorized` re-reads the authoritative
    // status afterwards, which also covers a block that never fired.
    let _ = rx.recv_timeout(PROMPT_TIMEOUT);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_message_lists_only_the_missing_permissions() {
        let screen_only = MissingPermissions {
            screen_recording: true,
            microphone: false,
        }
        .to_string();
        assert!(screen_only.contains("Screen & System Audio Recording"));
        assert!(!screen_only.contains("- Microphone"));

        let mic_only = MissingPermissions {
            screen_recording: false,
            microphone: true,
        }
        .to_string();
        assert!(mic_only.contains("Privacy & Security > Microphone"));
        assert!(!mic_only.contains("Screen & System Audio Recording"));

        let both = MissingPermissions {
            screen_recording: true,
            microphone: true,
        }
        .to_string();
        assert!(both.contains("Screen & System Audio Recording"));
        assert!(both.contains("Privacy & Security > Microphone"));
    }

    /// The inheritance note is the part users actually need; a refactor that drops it turns
    /// an actionable error back into a confusing one.
    #[test]
    fn the_message_explains_the_terminal_inheritance_trap() {
        let message = MissingPermissions {
            screen_recording: true,
            microphone: true,
        }
        .to_string();
        assert!(message.contains("terminal application"));
    }
}
