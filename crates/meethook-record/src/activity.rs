//! Whether some *other* process is capturing from an input device.
//!
//! This is the trigger the auto start/stop loop runs on: a meeting app opening the
//! microphone starts a session, and the same app releasing it ends one.
//!
//! # Why the predicate is not `kAudioDevicePropertyDeviceIsRunningSomewhere`
//!
//! That property is the obvious candidate, and it is the one the design originally named.
//! It cannot work on its own here. It means "the device is running in **at least one
//! process** on the system", and once this recorder starts its own `AVAudioEngine` input
//! tap, meethook *is* one of those processes. The flag is pinned true for the whole
//! session, so when the call actually ends the property does not change, no listener
//! fires, and the session would never finalize. The earlier empirical validation of this
//! property never hit that, because it only *observed* the device -- it never opened it.
//!
//! The predicate is therefore derived from CoreAudio's per-process audio objects
//! (macOS 14.4+), with meethook's own pid filtered out:
//!
//! ```text
//! active = any process object P where pid(P) != our pid and IsRunningInput(P) == 1
//! ```
//!
//! Excluding ourselves is what makes the signal immune to our own capture: when recording
//! starts, our process object flips to `IsRunningInput = 1`, the predicate is recomputed,
//! and the answer does not change.
//!
//! `IsRunningSomewhere` is still listened to, as a *trigger* rather than as the predicate.
//! It is the fastest signal for the start edge -- it fires before we hold the mic, so the
//! feedback problem does not apply there -- and it is re-attached whenever the default
//! input device changes.
//!
//! # Shape
//!
//! Four listener sites all feed one recomputation, and nothing is polled on a timer:
//!
//! | Object | Property | On notification |
//! |---|---|---|
//! | system | `DefaultInputDevice` | move the device listener to the new device |
//! | current input device | `DeviceIsRunningSomewhere` | -- |
//! | system | `ProcessObjectList` | diff the per-process listeners |
//! | each process object | `IsRunningInput` | -- |
//!
//! Recomputation is cheap and idempotent, and an edge is delivered only when the boolean
//! actually changes. That last property is what keeps a mute toggle from splitting a
//! session: whatever notifications a mute produces, none of them change the answer.

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex, Weak};

use block2::RcBlock;
use dispatch2::{DispatchQueue, DispatchQueueAttr, DispatchRetained};
use objc2_core_audio::{
    AudioObjectAddPropertyListenerBlock, AudioObjectGetPropertyData,
    AudioObjectGetPropertyDataSize, AudioObjectID, AudioObjectPropertyAddress,
    AudioObjectPropertySelector, AudioObjectRemovePropertyListenerBlock,
    kAudioDevicePropertyDeviceIsRunningSomewhere, kAudioHardwarePropertyDefaultInputDevice,
    kAudioHardwarePropertyProcessObjectList, kAudioObjectPropertyElementMain,
    kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject, kAudioProcessPropertyBundleID,
    kAudioProcessPropertyIsRunningInput, kAudioProcessPropertyPID,
};
use objc2_core_foundation::{CFRetained, CFString};

use crate::{Error, Result};

/// The transitions of "some other process is capturing from an input device".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activity {
    Started,
    Stopped,
}

/// A live set of CoreAudio property listeners reporting microphone activity.
///
/// Dropping it removes every listener. Keep it alive for as long as the edges matter.
pub struct MicActivityWatcher {
    state: Arc<Mutex<State>>,
    /// Held here as well as in [`State`] so `drop` can remove listeners *without* the
    /// state lock; see the safety note there.
    queue: DispatchRetained<DispatchQueue>,
}

impl MicActivityWatcher {
    /// Installs every listener and reports the predicate at this instant.
    ///
    /// The returned `bool` is the *level*, not an edge: if a call is already in progress
    /// when the watcher starts, the caller should act on it immediately rather than wait
    /// for a transition that has already happened.
    ///
    /// `on_change` is called from a private serial dispatch queue, once per real change,
    /// while the watcher's own lock is held. It must not block: sending on an unbounded
    /// channel is the intended shape, and anything slower stalls the next notification.
    pub fn start(
        on_change: impl Fn(Activity) + Send + Sync + 'static,
    ) -> Result<(MicActivityWatcher, bool)> {
        // Serial: every listener block is dispatched here, so all recomputation is
        // serialized and the listener bookkeeping needs no ordering beyond the mutex.
        let queue = DispatchQueue::new("com.meethook.activity", DispatchQueueAttr::SERIAL);

        // Cyclic because a listener block must be able to install *more* listeners (the
        // per-process ones, as processes come and go), which needs a handle back to the
        // state it is already inside. `Weak` is what keeps that from being a leak.
        let state = Arc::new_cyclic(|weak: &Weak<Mutex<State>>| {
            Mutex::new(State {
                weak: weak.clone(),
                queue: queue.clone(),
                on_change: Box::new(on_change),
                our_pid: std::process::id() as i32,
                debug: std::env::var_os("MEETHOOK_ACTIVITY_DEBUG").is_some(),
                active: false,
                system: Vec::new(),
                device: None,
                processes: HashMap::new(),
            })
        });

        // Constructed before `install`, so a partial install is unwound by `drop` rather
        // than by a second, near-duplicate cleanup path here.
        let watcher = MicActivityWatcher { state, queue };
        let active = {
            let mut state = watcher.lock();
            state.install()?
        };
        Ok((watcher, active))
    }

    /// The predicate as of the last recomputation: is some other process capturing *now*?
    ///
    /// A level, not an edge -- the same value [`MicActivityWatcher::start`] returns. The
    /// record loop needs it because a failed `Recorder::start` has to be retried while the
    /// call is still up: the level is already true, so no further [`Activity::Started`] can
    /// arrive until this call ends and a different one begins.
    pub fn is_active(&self) -> bool {
        self.lock().active
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        // A panic inside a listener block would otherwise poison the watcher permanently,
        // turning a one-off fault into a recorder that silently never triggers again.
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Drop for MicActivityWatcher {
    fn drop(&mut self) {
        // The listeners are taken under the lock but removed outside it: a block already
        // dispatched on the queue is waiting for that same lock, and holding it across an
        // FFI call that may synchronize with the queue is how this deadlocks.
        let listeners = self.lock().take_listeners();
        for listener in &listeners {
            remove_listener(listener, &self.queue);
        }
    }
}

/// Which listener fired. Carried only so the debug log can say what woke the recomputation
/// and so the handler knows whether any bookkeeping is due first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Trigger {
    /// The system's default input device changed.
    DefaultInputDevice,
    /// The set of processes doing audio changed.
    ProcessList,
    /// The current input device started or stopped running somewhere.
    DeviceRunning,
    /// Some process started or stopped capturing input.
    ProcessInput,
    /// Not a listener: the initial reading taken at `start`.
    Install,
}

impl Trigger {
    /// Names the listener in an error the user has to act on.
    fn what(self) -> &'static str {
        match self {
            Trigger::DefaultInputDevice => "default input device",
            Trigger::ProcessList => "audio process list",
            Trigger::DeviceRunning => "input device activity",
            Trigger::ProcessInput => "per-process input activity",
            Trigger::Install => "activity",
        }
    }
}

/// One installed property listener, kept so it can be removed again.
///
/// `AudioObjectRemovePropertyListenerBlock` matches on object + address + queue + block
/// pointer, so all four have to be the ones used to install it -- which is why the block
/// is owned here rather than dropped after the add.
struct Installed {
    object: AudioObjectID,
    address: AudioObjectPropertyAddress,
    block: RcBlock<dyn Fn(u32, NonNull<AudioObjectPropertyAddress>)>,
}

/// Everything the listener blocks share.
struct State {
    weak: Weak<Mutex<State>>,
    queue: DispatchRetained<DispatchQueue>,
    on_change: Box<dyn Fn(Activity) + Send + Sync>,
    our_pid: i32,
    debug: bool,
    /// The last value delivered to `on_change`.
    active: bool,
    system: Vec<Installed>,
    device: Option<Installed>,
    processes: HashMap<AudioObjectID, Installed>,
}

// SAFETY: the only non-`Send` members are `RcBlock`s and a `DispatchRetained`, both of
// which are pointers to objects whose reference counts are maintained atomically by
// libclosure and libdispatch respectively, and neither of which has thread affinity. All
// access to the state itself -- including creating and dropping those blocks -- happens
// under the `Mutex` this is always wrapped in.
unsafe impl Send for State {}

impl State {
    /// Installs all four listener kinds and takes the first reading.
    fn install(&mut self) -> Result<bool> {
        let system = kAudioObjectSystemObject as AudioObjectID;
        let default_device = self.listen(
            system,
            kAudioHardwarePropertyDefaultInputDevice,
            Trigger::DefaultInputDevice,
        )?;
        self.system.push(default_device);
        let process_list = self.listen(
            system,
            kAudioHardwarePropertyProcessObjectList,
            Trigger::ProcessList,
        )?;
        self.system.push(process_list);

        // A missing input device is not fatal here. The device listener is only a wake-up
        // trigger -- the predicate is computed from process objects and does not read it --
        // and `notified` already treats a device that goes away as a normal state to keep
        // watching from. Launching meethook before the USB mic is plugged in should not end
        // differently from unplugging it afterwards: the `DefaultInputDevice` notification
        // attaches the listener as soon as a device appears.
        //
        // `Error::CoreAudio` stays fatal by contrast: a listener the HAL *refuses* means no
        // trigger at all, and a recorder that silently never fires is this command's worst
        // outcome.
        match self.attach_device_listener() {
            Ok(()) => {}
            Err(e @ Error::NoInputDevice) => {
                eprintln!("Warning: {e}. Watching anyway; a device selected later is picked up.");
            }
            Err(e) => return Err(e),
        }
        self.sync_process_listeners();

        self.active = self.someone_else_is_capturing();
        if self.debug {
            self.log(Trigger::Install, self.active);
        }
        Ok(self.active)
    }

    /// Handles one notification: bookkeeping first, then a recomputation.
    fn notified(&mut self, trigger: Trigger) {
        match trigger {
            // A device that has gone away is a normal state to be watching from -- the
            // user unplugged a USB mic -- so this is not allowed to tear the watcher down.
            Trigger::DefaultInputDevice => {
                if let Err(e) = self.attach_device_listener()
                    && self.debug
                {
                    eprintln!("[activity] could not follow the default input device: {e}");
                }
            }
            Trigger::ProcessList => self.sync_process_listeners(),
            _ => {}
        }

        let active = self.someone_else_is_capturing();
        if self.debug {
            self.log(trigger, active);
        }
        let Some(activity) = edge(self.active, active) else {
            return;
        };
        self.active = active;
        (self.on_change)(activity);
    }

    /// Points the `IsRunningSomewhere` listener at the current default input device.
    ///
    /// A no-op when the device has not actually changed, so the frequent
    /// `DefaultInputDevice` notifications macOS emits do not churn listeners.
    fn attach_device_listener(&mut self) -> Result<()> {
        let device = default_input_device().ok_or(Error::NoInputDevice)?;
        if self.device.as_ref().is_some_and(|d| d.object == device) {
            return Ok(());
        }
        if let Some(previous) = self.device.take() {
            remove_listener(&previous, &self.queue);
        }
        self.device = Some(self.listen(
            device,
            kAudioDevicePropertyDeviceIsRunningSomewhere,
            Trigger::DeviceRunning,
        )?);
        if self.debug {
            eprintln!("[activity] IsRunningSomewhere listener attached to device {device}");
        }
        Ok(())
    }

    /// Adds a listener for every new process object and removes the departed ones.
    fn sync_process_listeners(&mut self) {
        let current = object_list(
            kAudioObjectSystemObject as AudioObjectID,
            kAudioHardwarePropertyProcessObjectList,
        );

        // Linear membership tests: the list is a few dozen entries on a busy machine, and
        // a set would cost more to build than it saves.
        let departed: Vec<AudioObjectID> = self
            .processes
            .keys()
            .copied()
            .filter(|id| !current.contains(id))
            .collect();
        for id in departed {
            if let Some(listener) = self.processes.remove(&id) {
                remove_listener(&listener, &self.queue);
            }
        }

        for id in current {
            if self.processes.contains_key(&id) {
                continue;
            }
            // A process object can disappear between being listed and being listened to.
            // That is a race with a normal outcome, not an error.
            if let Ok(listener) = self.listen(
                id,
                kAudioProcessPropertyIsRunningInput,
                Trigger::ProcessInput,
            ) {
                self.processes.insert(id, listener);
            }
        }
    }

    /// The predicate. See the module docs for why our own pid is excluded.
    ///
    /// A process whose pid cannot be read is treated as not capturing: without the pid we
    /// cannot prove it is not us, and mistaking ourselves for a meeting would mean a
    /// session that never ends.
    fn someone_else_is_capturing(&self) -> bool {
        object_list(
            kAudioObjectSystemObject as AudioObjectID,
            kAudioHardwarePropertyProcessObjectList,
        )
        .into_iter()
        .any(|process| {
            process_pid(process).is_some_and(|pid| pid != self.our_pid)
                && process_is_running_input(process)
        })
    }

    /// Everything needed to confirm or refute the reasoning in the module docs, at the
    /// machine, from one run. Gated behind `MEETHOOK_ACTIVITY_DEBUG`.
    ///
    /// The `IsRunningSomewhere` line is the load-bearing one: it is expected to stay
    /// `true` across a call ending while meethook records, which is exactly why it cannot
    /// be the predicate.
    fn log(&self, trigger: Trigger, active: bool) {
        let running_somewhere = self
            .device
            .as_ref()
            .and_then(|d| device_is_running_somewhere(d.object));
        eprintln!(
            "[activity] {trigger:?}: someone_else_is_capturing={active} \
             IsRunningSomewhere={running_somewhere:?}"
        );
        for process in object_list(
            kAudioObjectSystemObject as AudioObjectID,
            kAudioHardwarePropertyProcessObjectList,
        ) {
            let pid = process_pid(process);
            let ours = pid == Some(self.our_pid);
            let capturing = process_is_running_input(process);
            if !capturing && !ours {
                continue;
            }
            eprintln!(
                "[activity]   pid={} {} IsRunningInput={capturing}{}",
                pid.map_or_else(|| "?".to_owned(), |p| p.to_string()),
                process_bundle_id(process).unwrap_or_else(|| "(no bundle id)".to_owned()),
                if ours { "   <- meethook, excluded" } else { "" },
            );
        }
    }

    /// Installs one listener that funnels back into [`State::notified`].
    fn listen(
        &self,
        object: AudioObjectID,
        selector: AudioObjectPropertySelector,
        trigger: Trigger,
    ) -> Result<Installed> {
        let weak = self.weak.clone();
        let block = RcBlock::new(
            move |_count: u32, _addresses: NonNull<AudioObjectPropertyAddress>| {
                // The addresses are ignored on purpose. Every listener here is registered
                // for exactly one property, and the recomputation reads the world afresh
                // rather than trusting the notification to say what changed.
                let Some(state) = weak.upgrade() else {
                    return;
                };
                let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
                state.notified(trigger);
            },
        );

        let address = address(selector);
        // SAFETY: `address` is a live local that CoreAudio copies; the queue outlives every
        // listener (it is retained by the framework until removal, and separately by the
        // watcher); the block is kept alive by the `Installed` returned below.
        let status = unsafe {
            AudioObjectAddPropertyListenerBlock(
                object,
                NonNull::from(&address),
                Some(&self.queue),
                (&*block as *const block2::DynBlock<_>).cast_mut(),
            )
        };
        if status != 0 {
            return Err(Error::CoreAudio {
                what: trigger.what(),
                status,
            });
        }

        Ok(Installed {
            object,
            address,
            block,
        })
    }

    /// Empties every listener slot, for removal by the caller outside the lock.
    fn take_listeners(&mut self) -> Vec<Installed> {
        let mut listeners = std::mem::take(&mut self.system);
        listeners.extend(self.device.take());
        listeners.extend(std::mem::take(&mut self.processes).into_values());
        listeners
    }
}

/// The transition, if the predicate actually moved.
///
/// Separated out because "no change means no edge" is the property a mute toggle depends
/// on, and it is worth being able to test it without a microphone.
fn edge(previous: bool, current: bool) -> Option<Activity> {
    match (previous, current) {
        (false, true) => Some(Activity::Started),
        (true, false) => Some(Activity::Stopped),
        _ => None,
    }
}

/// Removes one listener.
///
/// Also called from *inside* a listener block -- following a device change, or pruning a
/// process that has gone away -- which is safe because the notification was dispatched to
/// our queue asynchronously: the HAL is not holding anything while our block runs, and it
/// does not synchronize back onto the listener queue to unregister.
fn remove_listener(listener: &Installed, queue: &DispatchQueue) {
    // SAFETY: object, address, queue and block pointer are the same four values the
    // matching `AudioObjectAddPropertyListenerBlock` was given.
    //
    // The status is discarded: the common failure is a process object that no longer
    // exists, which is precisely the case where there is nothing left to do.
    unsafe {
        AudioObjectRemovePropertyListenerBlock(
            listener.object,
            NonNull::from(&listener.address),
            Some(queue),
            (&*listener.block as *const block2::DynBlock<_>).cast_mut(),
        );
    }
}

fn address(selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}

/// Reads a fixed-size global-scope property, returning `None` for any failure.
///
/// Every caller is part of a trigger that must not take the recorder down, and an object
/// that declines to answer is a normal thing rather than an error.
///
/// # Safety
///
/// `T` must be the exact type CoreAudio returns for `selector` on `object`.
unsafe fn property<T: Copy + Default>(
    object: AudioObjectID,
    selector: AudioObjectPropertySelector,
) -> Option<T> {
    let address = address(selector);
    let mut value = T::default();
    let mut size = size_of::<T>() as u32;

    // SAFETY: `address`, `size` and `value` are live locals; the qualifier is null, which
    // the API accepts for every selector used here.
    let status = unsafe {
        AudioObjectGetPropertyData(
            object,
            NonNull::from(&address),
            0,
            ptr::null(),
            NonNull::from(&mut size),
            NonNull::from(&mut value).cast::<c_void>(),
        )
    };

    (status == 0).then_some(value)
}

/// Reads an `AudioObjectID` array property, sized first so nothing is truncated.
fn object_list(object: AudioObjectID, selector: AudioObjectPropertySelector) -> Vec<AudioObjectID> {
    let address = address(selector);
    let mut size: u32 = 0;
    // SAFETY: `address` and `size` are live locals; the qualifier is null.
    let status = unsafe {
        AudioObjectGetPropertyDataSize(
            object,
            NonNull::from(&address),
            0,
            ptr::null(),
            NonNull::from(&mut size),
        )
    };
    if status != 0 {
        return Vec::new();
    }

    let mut ids = vec![0 as AudioObjectID; size as usize / size_of::<AudioObjectID>()];
    let Some(buffer) = NonNull::new(ids.as_mut_ptr()) else {
        return Vec::new();
    };
    // SAFETY: `buffer` points at `size` bytes of owned, correctly typed storage.
    let status = unsafe {
        AudioObjectGetPropertyData(
            object,
            NonNull::from(&address),
            0,
            ptr::null(),
            NonNull::from(&mut size),
            buffer.cast::<c_void>(),
        )
    };
    if status != 0 {
        return Vec::new();
    }

    // The set can shrink between the two calls; trust the second answer.
    ids.truncate(size as usize / size_of::<AudioObjectID>());
    ids
}

fn default_input_device() -> Option<AudioObjectID> {
    // SAFETY: this selector returns a single `AudioObjectID`.
    let id: AudioObjectID = unsafe {
        property(
            kAudioObjectSystemObject as AudioObjectID,
            kAudioHardwarePropertyDefaultInputDevice,
        )?
    };
    (id != 0).then_some(id)
}

fn device_is_running_somewhere(device: AudioObjectID) -> Option<bool> {
    // SAFETY: this selector returns a single `UInt32` used as a boolean.
    unsafe { property::<u32>(device, kAudioDevicePropertyDeviceIsRunningSomewhere) }
        .map(|running| running != 0)
}

fn process_pid(process: AudioObjectID) -> Option<i32> {
    // SAFETY: this selector returns a single `pid_t`, which is `i32` on Darwin.
    unsafe { property::<i32>(process, kAudioProcessPropertyPID) }
}

fn process_is_running_input(process: AudioObjectID) -> bool {
    // SAFETY: this selector returns a single `UInt32` used as a boolean.
    unsafe { property::<u32>(process, kAudioProcessPropertyIsRunningInput) }
        .is_some_and(|running| running != 0)
}

/// The bundle id of a process object, for the debug log only.
fn process_bundle_id(process: AudioObjectID) -> Option<String> {
    // SAFETY: this selector returns a single `CFStringRef`. `Option<NonNull<_>>` is used
    // as the destination because it is pointer-sized and its `Default` is null, which is
    // exactly what a "no value written" outcome should read as.
    let raw: Option<NonNull<CFString>> =
        unsafe { property(process, kAudioProcessPropertyBundleID)? };
    // SAFETY: the property is a "copy" accessor, so the returned string is owned by us.
    let string = unsafe { CFRetained::from_raw(raw?) };
    Some(string.to_string())
}

#[cfg(test)]
mod tests {
    use super::{Activity, edge};

    #[test]
    fn an_unchanged_predicate_emits_nothing() {
        // The property a mute toggle depends on: whatever notifications muting produces,
        // none of them change the answer, so none of them split the session.
        assert_eq!(edge(false, false), None);
        assert_eq!(edge(true, true), None);
    }

    #[test]
    fn a_changed_predicate_emits_the_transition() {
        assert_eq!(edge(false, true), Some(Activity::Started));
        assert_eq!(edge(true, false), Some(Activity::Stopped));
    }
}
