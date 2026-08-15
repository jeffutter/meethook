//! Objective-C exceptions, turned into [`Error`]s at the framework boundary.
//!
//! Apple's audio frameworks report some failures by raising an `NSException` rather than by
//! returning an error, and an exception raised inside a framework call has no Rust frame
//! willing to catch it. It unwinds every frame of this program until `lang_start`'s landing
//! pad sees a non-Rust exception and calls `abort`. Nothing is recoverable past that point and
//! nothing useful is printed about what raised: the process is simply gone, and with it the
//! watcher and whatever session it was recording.
//!
//! This is not a hypothetical. Plugging in a dock mid-meeting moved the default input device
//! while a new session was being opened, `installTapOnBus` was handed a format that no longer
//! described the hardware, and the resulting exception killed a recorder that had been
//! watching for three hours -- reported to the user as nothing more than `fatal runtime error:
//! Rust cannot catch foreign exceptions`.
//!
//! So every framework call that can raise goes through [`catching`], and the failure arrives
//! as an ordinary [`Error::Framework`]. Nothing downstream needs a new branch to handle it:
//! the record loop already retries a session start that failed, and a raise is exactly that.
//!
//! What this cannot cover is a raise on a thread this program does not own -- a
//! ScreenCaptureKit delegate callback on its dispatch queue, a CoreAudio listener block.
//! There is no Rust frame between those and the runtime's uncaught-exception handler, so an
//! exception raised there aborts regardless. Those callbacks are kept to publishing a value
//! for another thread to read, which is the only defence available.

use std::panic::AssertUnwindSafe;

use crate::{Error, Result};

/// Calls an Apple framework API that may raise an Objective-C exception, returning the
/// exception as an [`Error::Framework`] instead of letting it abort the process.
///
/// `api` names the raising call rather than the operation being attempted, because a raise is
/// a framework-level fault and the first question about one is always which framework call
/// produced it.
///
/// Unwind safety is asserted here rather than demanded of the caller, and that is sound by
/// construction: on the raising path every object the closure touched is dropped unused. Each
/// caller either returns the error immediately, or -- in [`crate::mic::MicCapture::stop`] --
/// goes on to finalize a WAV that the framework never held a reference to. Requiring
/// `UnwindSafe` of the closure instead would push an `AssertUnwindSafe` out to every call
/// site, each of which would have to re-derive this same argument.
pub(crate) fn catching<R>(api: &'static str, call: impl FnOnce() -> R) -> Result<R> {
    objc2::exception::catch(AssertUnwindSafe(call)).map_err(|exception| Error::Framework {
        api,
        // `Display` on an exception renders the `reason` an `NSException` carries, which is
        // the half a user can act on ("required condition is false: ..."). `None` is `@throw
        // nil`, which should never happen but must not be unwrapped in an error path.
        message: exception.map_or_else(
            || "a nil exception was raised".to_owned(),
            |exception| exception.to_string(),
        ),
    })
}

/// The boundary, decided with no framework that raises on purpose anywhere near it.
///
/// An unrecognized selector is used as the stand-in for a framework raise because it is a
/// genuine `NSException` travelling the genuine unwind path -- `catch` cannot tell it from the
/// one `installTapOnBus` throws. What these cannot decide is *which* framework calls raise;
/// that is a fact about macOS, and the call sites carry the argument for each.
#[cfg(test)]
mod tests {
    use objc2::msg_send;
    use objc2::rc::Retained;
    use objc2::runtime::NSObject;

    use super::*;

    #[test]
    fn a_call_that_does_not_raise_returns_its_value() {
        assert_eq!(catching("test", || 7).unwrap(), 7);
    }

    #[test]
    fn a_raise_becomes_an_error_naming_the_call_that_raised() {
        let object = NSObject::new();
        let error = catching("AVAudioInputNode.installTapOnBus", || {
            // SAFETY: none required -- sending a selector `NSObject` does not implement is
            // exactly the misuse being provoked, and the raise it produces is the subject.
            let _: Retained<NSObject> = unsafe { msg_send![&*object, copy] };
        })
        .unwrap_err();

        let message = error.to_string();
        assert!(
            message.contains("AVAudioInputNode.installTapOnBus"),
            "the error does not say what raised: {message}"
        );
        assert!(
            message.contains("unrecognized selector"),
            "the error does not carry the exception's own reason: {message}"
        );
    }

    /// The property the whole module exists for: control returns to the caller at all. Before
    /// this, reaching the line after the raise was impossible -- the process had aborted.
    #[test]
    fn a_raise_does_not_end_the_process() {
        let object = NSObject::new();
        let _ = catching("test", || {
            // SAFETY: as above.
            let _: Retained<NSObject> = unsafe { msg_send![&*object, copy] };
        });
        assert_eq!(catching("test", || "still here").unwrap(), "still here");
    }
}
