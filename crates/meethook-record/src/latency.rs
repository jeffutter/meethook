//! What the audio hardware says its own latency is.
//!
//! Diagnostic only. Nothing on the recording path reads this module.
//!
//! It exists because a click test showed the two capture APIs disagree by a constant ~13 ms
//! about when a sound happened, while both tracks' timestamps were separately shown to be
//! linear in their own sample counts to under 0.1 ms over 35 seconds -- so neither is
//! drifting or mis-anchored, and the disagreement had to be a *latency compensation*
//! difference rather than a broken timestamp.
//!
//! What the numbers below established, across takes on built-in, USB and Bluetooth outputs:
//!
//! - A fixed ~16 ms bias originates inside the ScreenCaptureKit audio pipeline. Swapping the
//!   display between 72 Hz and 30 Hz moved it by 0.63 ms against a predicted 19.44 ms, so it
//!   is not clocked off the video frame rate. It is simply a constant, on one Mac.
//! - `SCStream` presentation timestamps do not include output latency at all: swapping the
//!   output device moved the click residual by 423 ms.
//! - CoreAudio cannot supply that missing term either. It reported 186.688 ms for a Bluetooth
//!   output path measured at roughly 426 ms -- an under-report of ~240 ms.
//!
//! So these figures are worth *seeing* and not worth *subtracting*: a correction built from
//! them would fix 16 ms of an error that reached 410 ms, while making the stored offset look
//! authoritative. Acoustic alignment has to be measured from the signals, which the AEC stage
//! does anyway. Note also that `AVAudioIONode.presentationLatency` tracks the OUTPUT device
//! (1.542 ms on USB output, 203.197 ms on Bluetooth, same USB mic) and is therefore not a
//! usable measure of input latency.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2_core_audio::{
    AudioObjectGetPropertyData, AudioObjectID, AudioObjectPropertyAddress,
    kAudioDevicePropertyLatency, kAudioDevicePropertySafetyOffset, kAudioDevicePropertyStreams,
    kAudioHardwarePropertyDefaultInputDevice, kAudioHardwarePropertyDefaultOutputDevice,
    kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyScopeInput, kAudioObjectPropertyScopeOutput, kAudioObjectSystemObject,
    kAudioStreamPropertyLatency,
};

/// One signal path's latency, as the device reports it, in frames.
///
/// The three terms are separate because they are compensated separately and inconsistently
/// across APIs: a timestamp may account for the device latency but not the safety offset,
/// which is exactly the kind of partial compensation that produces a stable few-millisecond
/// bias.
#[derive(Debug, Clone, Copy, Default)]
pub struct PathLatency {
    pub device_frames: u32,
    pub stream_frames: u32,
    pub safety_offset_frames: u32,
}

impl PathLatency {
    pub fn total_frames(&self) -> u32 {
        self.device_frames + self.stream_frames + self.safety_offset_frames
    }

    pub fn total_millis(&self, sample_rate: u32) -> f64 {
        if sample_rate == 0 {
            return 0.0;
        }
        f64::from(self.total_frames()) / f64::from(sample_rate) * 1000.0
    }
}

/// Reads a fixed-size property, returning `None` for any failure.
///
/// Every caller here is a diagnostic that must never take the recorder down, and a device
/// that declines to report its latency is a normal thing rather than an error -- aggregate
/// and virtual devices routinely omit these properties.
///
/// # Safety
///
/// `T` must be the exact type CoreAudio returns for `selector` on `object`.
unsafe fn property<T: Copy + Default>(
    object: AudioObjectID,
    selector: u32,
    scope: u32,
) -> Option<T> {
    let address = AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain,
    };
    let mut value = T::default();
    let mut size = size_of::<T>() as u32;

    // SAFETY: `address`, `size`, and `value` are live locals; the qualifier is null, which
    // the API accepts for every selector used here.
    let status = unsafe {
        AudioObjectGetPropertyData(
            object,
            NonNull::from(&address),
            0,
            std::ptr::null(),
            NonNull::from(&mut size),
            NonNull::from(&mut value).cast::<c_void>(),
        )
    };

    (status == 0).then_some(value)
}

/// The device CoreAudio would use for the given scope, or `None` if there isn't one.
fn default_device(input: bool) -> Option<AudioObjectID> {
    let selector = if input {
        kAudioHardwarePropertyDefaultInputDevice
    } else {
        kAudioHardwarePropertyDefaultOutputDevice
    };
    // SAFETY: both default-device selectors return a single `AudioObjectID`.
    let id: AudioObjectID = unsafe {
        property(
            kAudioObjectSystemObject as AudioObjectID,
            selector,
            kAudioObjectPropertyScopeGlobal,
        )?
    };
    (id != 0).then_some(id)
}

/// The reported latency of the default input or output path.
///
/// The stream term comes from the device's first stream in that scope. Devices with several
/// streams can report different latencies per stream, but the first is the one a default
/// two-channel capture or playback actually runs through, and a diagnostic that surveyed
/// them all would report numbers no caller could act on.
pub fn default_path(input: bool) -> Option<PathLatency> {
    let device = default_device(input)?;
    let scope = if input {
        kAudioObjectPropertyScopeInput
    } else {
        kAudioObjectPropertyScopeOutput
    };

    // SAFETY: both selectors return a single `u32` on a device object.
    let device_frames = unsafe { property(device, kAudioDevicePropertyLatency, scope) };
    let safety_offset_frames = unsafe { property(device, kAudioDevicePropertySafetyOffset, scope) };

    // SAFETY: `kAudioDevicePropertyStreams` returns an array, so the fixed-size read gets
    // the first element -- which is the one wanted -- and reports the truncation via a
    // status this helper turns into `None`. Reading one ID is therefore best-effort.
    let stream_frames = unsafe { property::<AudioObjectID>(device, kAudioDevicePropertyStreams, scope) }
        .and_then(|stream| unsafe {
            property::<u32>(stream, kAudioStreamPropertyLatency, kAudioObjectPropertyScopeGlobal)
        });

    Some(PathLatency {
        device_frames: device_frames.unwrap_or(0),
        stream_frames: stream_frames.unwrap_or(0),
        safety_offset_frames: safety_offset_frames.unwrap_or(0),
    })
}
