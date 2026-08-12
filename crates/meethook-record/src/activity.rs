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
//! (macOS 14.4+), with our own capture filtered out:
//!
//! ```text
//! active = any process object P where IsRunningInput(P) == 1
//!                                 and pid(P) != our pid
//!                                 and bundle_id(P) is not one of our helpers
//!                                 and executable(P) != our executable
//! ```
//!
//! Excluding ourselves is what makes the signal immune to our own capture: when recording
//! starts, our process object flips to `IsRunningInput = 1`, the predicate is recomputed,
//! and the answer does not change.
//!
//! # Why excluding our own pid is not enough
//!
//! Our capture is not confined to our process. ScreenCaptureKit services system-audio
//! capture out of process, in `com.apple.replayd`, and that helper registers a process
//! object of its own with `IsRunningInput = 1`. Its pid is not ours, so the pid filter
//! does not catch it, and the predicate stays true for as long as we hold the capture --
//! which is the whole session. Observed on hardware (TASK-005.01): `replayd` is absent
//! from the process list before `Recorder::start` and appears immediately after it, and
//! when the Teams call ended, Teams left the list entirely while `replayd` went on
//! reporting `IsRunningInput = true`, so no stop edge was ever emitted.
//!
//! Helpers are therefore excluded by bundle id as well, unconditionally rather than only
//! while recording: a process whose reason for existing is to service screen capture is
//! never itself the signal that a meeting is under way. The cost is that another app
//! capturing system audio through ScreenCaptureKit no longer reads as microphone activity.
//! That is the right answer -- screen recording is not a meeting, and a meeting app holds
//! the input device directly, as Teams does above.
//!
//! `IsRunningSomewhere` is still listened to, as a *trigger* rather than as the predicate.
//! It is the fastest signal for the start edge -- it fires before we hold the mic, so the
//! feedback problem does not apply there -- and it is re-attached whenever the default
//! input device changes.
//!
//! # Why neither of those catches a second meethook
//!
//! A second `meethook record` has a different pid, and being a plain Rust binary it reports
//! no bundle id at all, so both exclusions above miss it and it classifies as somebody
//! else's meeting. Two concurrent instances then hold each other recording for as long as
//! both are alive, and neither ever sees a stop edge. Observed on hardware (TASK-005.03.01),
//! where a meethook left over from an earlier run was still counted as the meeting signal
//! 41 minutes later.
//!
//! The third exclusion is therefore keyed on the *executable* behind the pid, which is the
//! only fact that distinguishes another copy of ourselves from an unbundled meeting app.
//!
//! Its known limit: two meethooks launched from *different* binaries -- `target/debug/meethook`
//! under `cargo run` next to an installed copy -- have different executables and are not
//! caught. That is the development-time shape rather than the user-facing one, and the
//! general answer is a single-instance lock at startup, which would define the whole class
//! away instead of filtering it.
//!
//! # Shape
//!
//! Three listener sites all feed one recomputation:
//!
//! | Object | Property | On notification |
//! |---|---|---|
//! | system | `DefaultInputDevice` | move the device listener to the new device |
//! | current input device | `DeviceIsRunningSomewhere` | -- |
//! | system | `ProcessObjectList` | -- |
//!
//! Recomputation is cheap and idempotent, and an edge is delivered only when the boolean
//! actually changes. That last property is what keeps a mute toggle from splitting a
//! session: whatever notifications a mute produces, none of them change the answer. It is
//! also what makes the re-check below invisible except when it has something to report.
//!
//! # Why the listeners alone are not enough
//!
//! A notification is not trusted to say what changed; it wakes a fresh walk of the process
//! objects. That walk reads a world which can move underneath it, and on hardware it has.
//! In run 1 of the TASK-005.02 matrix the predicate's walk answered `true` while the debug
//! log's walk, microseconds later, no longer contained the capturing process at all -- so
//! the process had already released the microphone. The notification was correct and on
//! time; the read behind it was stale, and the release edge was never emitted.
//!
//! Losing one edge that way is permanent. The machine has settled, so no further
//! notification is coming, and that run kept recording for 1862.8 s until the user pressed
//! Ctrl-C. No read can be made to win that race -- any walk reads a world that can move
//! under it -- so the answer is not a better read but a way back from a lost one.
//!
//! [`MicActivityWatcher::recheck`] is that way back: it recomputes on demand and delivers
//! an edge through `on_change` if the answer has moved. The record loop calls it every
//! couple of seconds *while a session is live*, which turns a lost release edge into a few
//! seconds of extra tail rather than a session that runs until the process is killed.
//!
//! The idle path has no timer at all, which is deliberate. Every start edge observed on
//! hardware so far arrived from a listener, and a call already in progress at launch is
//! covered by the level [`MicActivityWatcher::start`] returns, so polling while idle would
//! cost a wake-up every couple of seconds and buy nothing.
//!
//! # The per-process listener, removed
//!
//! An `IsRunningInput` listener on every process object used to be attached as well, and
//! it never delivered: its trigger appears in none of the four TASK-005.02 hardware runs,
//! across roughly an hour of recording and several hundred notifications, even though each
//! attach succeeded (a refusal is fatal through [`Error::CoreAudio`]). It was cost with no
//! value, and worse, it made the design look as though it had a second chance at every
//! edge when in fact it had exactly one. It is gone; the re-check above is the real second
//! chance, and unlike that listener it is decidable in a test.
//!
//! The system-level `ProcessObjectList` listener is kept -- that one demonstrably fires,
//! including as a meeting app's process object appears -- and after the removal it simply
//! wakes a recomputation instead of doing listener bookkeeping first.

use std::ffi::{OsString, c_void};
use std::os::unix::ffi::OsStringExt;
use std::path::{Path, PathBuf};
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
    /// `on_change` is called once per real change, while the watcher's own lock is held --
    /// from a private serial dispatch queue for a listener, and on the caller's own thread
    /// for [`MicActivityWatcher::recheck`]. It must not block: sending on an unbounded
    /// channel is the intended shape, and anything slower stalls the next notification.
    pub fn start(
        on_change: impl Fn(Activity) + Send + Sync + 'static,
    ) -> Result<(MicActivityWatcher, bool)> {
        // Serial: every listener block is dispatched here, so all recomputation is
        // serialized and the listener bookkeeping needs no ordering beyond the mutex.
        let queue = DispatchQueue::new("com.meethook.activity", DispatchQueueAttr::SERIAL);

        // Cyclic because a listener block must be able to install *more* listeners -- the
        // `DefaultInputDevice` one re-attaches the device listener as the default moves --
        // which needs a handle back to the state it is already inside. `Weak` is what keeps
        // that from being a leak.
        let state = Arc::new_cyclic(|weak: &Weak<Mutex<State>>| {
            Mutex::new(State {
                weak: weak.clone(),
                queue: queue.clone(),
                on_change: Box::new(on_change),
                our_pid: std::process::id() as i32,
                // Resolved once here rather than per predicate walk, and canonicalized so
                // it compares equal to the canonicalized path of another instance however
                // that one was invoked.
                our_exe: std::env::current_exe()
                    .ok()
                    .and_then(|path| std::fs::canonicalize(path).ok()),
                debug: std::env::var_os("MEETHOOK_ACTIVITY_DEBUG").is_some(),
                active: false,
                system: Vec::new(),
                device: None,
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

    /// Recomputes the predicate now, delivering an edge if the answer has moved.
    ///
    /// This is the recovery path for a notification whose recomputation read the world a
    /// moment too early: a walk of the process objects can be stale, and once the machine
    /// has settled no further notification is coming, so without this a lost edge is
    /// permanent. See the module docs for the hardware evidence.
    ///
    /// Any edge is delivered through `on_change` rather than returned, so a caller that is
    /// also the one draining `on_change` sees it as an ordinary edge and needs no second
    /// path for it. `on_change` therefore runs on *this* thread, under the watcher's lock;
    /// the contract that it must not block is what makes that safe.
    ///
    /// The returned `bool` is the *level*, for the one caller that needs it: a failed
    /// `Recorder::start` has to be retried while the call is still up, and since the level
    /// is already true no further [`Activity::Started`] can arrive until this call ends and
    /// a different one begins.
    pub fn recheck(&self) -> bool {
        let mut state = self.lock();
        state.notified(Trigger::Recheck);
        state.active
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

/// What woke a recomputation. Carried only so the debug log can say which one it was, and
/// so the handler knows whether any bookkeeping is due first.
///
/// Not all of these are listeners; the two that are not say so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Trigger {
    /// The system's default input device changed.
    DefaultInputDevice,
    /// The set of processes doing audio changed.
    ProcessList,
    /// The current input device started or stopped running somewhere.
    DeviceRunning,
    /// Not a listener: the initial reading taken at `start`.
    Install,
    /// Not a listener: the safety-net recomputation the record loop asks for while a
    /// session is live. A `Recheck` line in the debug log therefore means exactly one
    /// thing -- the net caught an edge the listeners lost.
    Recheck,
}

impl Trigger {
    /// Names the listener in an error the user has to act on.
    fn what(self) -> &'static str {
        match self {
            Trigger::DefaultInputDevice => "default input device",
            Trigger::ProcessList => "audio process list",
            Trigger::DeviceRunning => "input device activity",
            Trigger::Install | Trigger::Recheck => "activity",
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
    /// Our own executable, canonicalized. `None` when it cannot be resolved, which turns
    /// the second-instance exclusion off rather than making it fire on everything.
    our_exe: Option<PathBuf>,
    debug: bool,
    /// The last value delivered to `on_change`.
    active: bool,
    system: Vec<Installed>,
    device: Option<Installed>,
}

// SAFETY: the only non-`Send` members are `RcBlock`s and a `DispatchRetained`, both of
// which are pointers to objects whose reference counts are maintained atomically by
// libclosure and libdispatch respectively, and neither of which has thread affinity. All
// access to the state itself -- including creating and dropping those blocks -- happens
// under the `Mutex` this is always wrapped in.
unsafe impl Send for State {}

impl State {
    /// Installs all three listeners and takes the first reading.
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

        self.active = self.someone_else_is_capturing();
        if self.debug {
            self.log(Trigger::Install, self.active);
        }
        Ok(self.active)
    }

    /// Handles one notification: bookkeeping first, then a recomputation.
    fn notified(&mut self, trigger: Trigger) {
        // Only the default-device listener has any bookkeeping left; every other trigger,
        // including the re-check, is purely a wake-up for the recomputation below.
        if trigger == Trigger::DefaultInputDevice {
            // A device that has gone away is a normal state to be watching from -- the
            // user unplugged a USB mic -- so this is not allowed to tear the watcher down.
            if let Err(e) = self.attach_device_listener()
                && self.debug
            {
                eprintln!("[activity] could not follow the default input device: {e}");
            }
        }

        let active = self.someone_else_is_capturing();
        let activity = edge(self.active, active);
        // A re-check that changes nothing is the overwhelmingly common case and carries no
        // information; logging every one would bury the notifications that do. A `Recheck`
        // line therefore always means the safety net caught something.
        if self.debug && (trigger != Trigger::Recheck || activity.is_some()) {
            self.log(trigger, active);
        }
        let Some(activity) = activity else {
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

    /// The predicate: is any process other than us and our helpers capturing input?
    fn someone_else_is_capturing(&self) -> bool {
        object_list(
            kAudioObjectSystemObject as AudioObjectID,
            kAudioHardwarePropertyProcessObjectList,
        )
        .into_iter()
        .any(|process| self.bearing_of(process) == Bearing::Activity)
    }

    /// Reads one process object and asks [`bearing`] what it means.
    ///
    /// The reads are ordered cheapest-first and short-circuit: a process that is not
    /// capturing -- almost all of them -- costs one property read, and the bundle id and
    /// executable path are only fetched for one that is.
    fn bearing_of(&self, process: AudioObjectID) -> Bearing {
        if !process_is_running_input(process) {
            return Bearing::Idle;
        }
        let pid = process_pid(process);
        bearing(
            pid,
            process_bundle_id(process).as_deref(),
            pid.and_then(process_executable).as_deref(),
            self.our_pid,
            self.our_exe.as_deref(),
        )
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
            // The verdict comes from the same classifier the predicate uses, so the log
            // cannot describe a rule the recorder is not actually applying.
            let bearing = self.bearing_of(process);
            if bearing == Bearing::Idle && !ours {
                continue;
            }
            eprintln!(
                "[activity]   pid={} {} IsRunningInput={}{}",
                pid.map_or_else(|| "?".to_owned(), |p| p.to_string()),
                process_bundle_id(process).unwrap_or_else(|| "(no bundle id)".to_owned()),
                bearing != Bearing::Idle,
                match bearing {
                    Bearing::Excluded(why) => format!("   <- excluded: {why}"),
                    // Idle and ours: shown anyway, because "meethook is not capturing yet"
                    // is the baseline the later lines are read against.
                    _ if ours => "   <- meethook".to_owned(),
                    _ => String::new(),
                },
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
        listeners
    }
}

/// Bundle ids that capture input only because meethook asked something to.
///
/// `com.apple.replayd` is the ScreenCaptureKit system-audio capture service: starting the
/// speaker track starts it, and it reports `IsRunningInput = 1` under its own pid for as
/// long as we record. See the module docs for the hardware evidence.
const OUR_HELPER_BUNDLE_IDS: &[&str] = &["com.apple.replayd"];

/// What one capturing process means for the predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bearing {
    /// Not capturing input.
    Idle,
    /// Capturing, and it is somebody else: this is the meeting signal.
    Activity,
    /// Capturing, but it is us or on our behalf. Carries the reason, for the debug log.
    Excluded(&'static str),
}

/// The exclusion rule, over facts already read from a capturing process object.
///
/// Pure so that the rule the whole trigger turns on can be tested with no CoreAudio device
/// present -- which the sandbox this is developed in does not have.
///
/// A process whose pid cannot be read is excluded: without the pid we cannot prove it is
/// not us, and mistaking our own capture for a meeting means a session that never ends.
/// That is the failure this rule exists to prevent, so it is the safe direction to fail in.
///
/// An unreadable `executable` fails the *other* way, and the asymmetry is deliberate: by
/// the time it is consulted the pid has already been read and shown not to be ours, so the
/// only thing an unknown path leaves open is which *other* program it is -- while excluding
/// every process whose path we cannot read would silence the trigger for whole classes of
/// meeting app. Missing an exclusion costs a session that overstays; over-excluding costs
/// every session, so the unknown is counted as activity.
fn bearing(
    pid: Option<i32>,
    bundle_id: Option<&str>,
    executable: Option<&Path>,
    our_pid: i32,
    our_exe: Option<&Path>,
) -> Bearing {
    let Some(pid) = pid else {
        return Bearing::Excluded("pid unreadable, so it cannot be shown not to be meethook");
    };
    if pid == our_pid {
        return Bearing::Excluded("meethook");
    }
    if bundle_id.is_some_and(|id| OUR_HELPER_BUNDLE_IDS.contains(&id)) {
        return Bearing::Excluded("captures on meethook's behalf");
    }
    if let (Some(exe), Some(ours)) = (executable, our_exe)
        && exe == ours
    {
        return Bearing::Excluded("another meethook instance");
    }
    Bearing::Activity
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
/// Also called from *inside* a listener block, when the default input device changes and
/// the `IsRunningSomewhere` listener has to move with it. That is safe because the
/// notification was dispatched to our queue asynchronously: the HAL is not holding anything
/// while our block runs, and it does not synchronize back onto the listener queue to
/// unregister.
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

/// The bundle id of a process object, or `None` for one that does not report it.
///
/// Read by the predicate, not only by the log: it is how a helper capturing on our behalf
/// is told apart from a meeting app.
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

/// The executable behind a pid, canonicalized, or `None` for one that cannot be read.
///
/// Not a CoreAudio property: the HAL reports a bundle id, and a plain binary has none,
/// which is exactly why a second meethook is invisible to the bundle-id rule. The path
/// comes from libproc instead, and is canonicalized so a symlinked or `../`-flavoured
/// invocation still compares equal to our own.
///
/// `None` covers every failure alike -- a process that has already exited, one owned by
/// another user, a path that no longer resolves. [`bearing`] treats that as "not known to
/// be us", which is the direction its doc comment explains.
fn process_executable(pid: i32) -> Option<PathBuf> {
    let mut buffer = vec![0u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
    // SAFETY: `buffer` is owned, writable storage of exactly the length passed alongside
    // it. `proc_pidpath` writes at most that many bytes and returns the length written.
    let length = unsafe {
        libc::proc_pidpath(
            pid,
            buffer.as_mut_ptr().cast::<c_void>(),
            buffer.len() as u32,
        )
    };
    if length <= 0 {
        return None;
    }
    buffer.truncate(length as usize);
    std::fs::canonicalize(PathBuf::from(OsString::from_vec(buffer))).ok()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{Activity, Bearing, bearing, edge, process_executable};

    const OUR_PID: i32 = 500;

    /// Our own executable, as `MicActivityWatcher::start` would have canonicalized it.
    fn our_exe() -> &'static Path {
        Path::new("/usr/local/bin/meethook")
    }

    #[test]
    fn a_meeting_app_capturing_is_the_signal() {
        assert_eq!(
            bearing(
                Some(26975),
                Some("com.microsoft.teams2.modulehost"),
                Some(Path::new(
                    "/Applications/Microsoft Teams.app/Contents/MacOS/MSTeams"
                )),
                OUR_PID,
                Some(our_exe()),
            ),
            Bearing::Activity
        );
        // A meeting app that reports no bundle id is still a meeting app.
        assert_eq!(
            bearing(Some(26975), None, None, OUR_PID, Some(our_exe())),
            Bearing::Activity
        );
    }

    #[test]
    fn our_own_capture_is_not_the_signal() {
        assert!(matches!(
            bearing(
                Some(OUR_PID),
                Some("com.meethook"),
                Some(our_exe()),
                OUR_PID,
                Some(our_exe()),
            ),
            Bearing::Excluded(_)
        ));
    }

    #[test]
    fn the_screencapturekit_helper_is_not_the_signal() {
        // The regression this rule exists for: replayd captures input under its own pid
        // for as long as we record the speaker track, so a pid filter alone leaves the
        // predicate pinned true and the session never ends.
        assert!(matches!(
            bearing(
                Some(997),
                Some("com.apple.replayd"),
                Some(Path::new("/usr/libexec/replayd")),
                OUR_PID,
                Some(our_exe()),
            ),
            Bearing::Excluded(_)
        ));
    }

    #[test]
    fn a_second_meethook_instance_is_not_the_signal() {
        // Run 3 of the TASK-005.02 hardware matrix: pid 41809, a meethook left over from
        // the previous run, reported IsRunningInput=true with no bundle id and was counted
        // as the meeting signal, so the live instance never saw a stop edge. Different pid,
        // no bundle id, same executable -- the executable is the only fact that catches it.
        //
        // The reason is asserted exactly, not just matched: `State::log` renders it as
        // `<- excluded: {why}`, so this string is what a hardware run shows against the
        // filtered pid.
        assert_eq!(
            bearing(Some(41809), None, Some(our_exe()), OUR_PID, Some(our_exe())),
            Bearing::Excluded("another meethook instance")
        );
    }

    #[test]
    fn a_meeting_app_with_a_different_executable_is_still_the_signal() {
        // The over-exclusion guard: an unbundled binary looks like a second meethook in
        // every respect except the one that matters.
        assert_eq!(
            bearing(
                Some(26975),
                None,
                Some(Path::new("/opt/homebrew/bin/some-meeting-app")),
                OUR_PID,
                Some(our_exe()),
            ),
            Bearing::Activity
        );
    }

    #[test]
    fn an_unreadable_executable_does_not_exclude() {
        // The opposite direction from an unreadable pid, on purpose: the pid has already
        // been read and is not ours, so the only open question is which other program this
        // is -- and excluding every path we cannot read would silence the trigger for whole
        // classes of meeting app.
        assert_eq!(
            bearing(Some(26975), None, None, OUR_PID, Some(our_exe())),
            Bearing::Activity
        );
        // Same when it is *our* path that could not be resolved: the exclusion turns off
        // rather than firing on everything.
        assert_eq!(
            bearing(Some(26975), None, Some(our_exe()), OUR_PID, None),
            Bearing::Activity
        );
    }

    #[test]
    fn a_process_with_no_readable_pid_is_not_the_signal() {
        // Excluded rather than counted: it cannot be shown not to be us, and counting it
        // would be the never-ending session again.
        assert!(matches!(
            bearing(
                None,
                Some("com.example.mystery"),
                None,
                OUR_PID,
                Some(our_exe())
            ),
            Bearing::Excluded(_)
        ));
    }

    #[test]
    fn a_pid_resolves_to_the_executable_the_exclusion_compares() {
        // The rule above is pure, so it would pass just as well if `proc_pidpath` never
        // returned anything and the whole arm were dead on hardware. This is the other
        // half: the live reader, against the one pid whose executable the test knows.
        let ours = std::fs::canonicalize(std::env::current_exe().unwrap()).unwrap();
        assert_eq!(
            process_executable(std::process::id() as i32),
            Some(ours.clone())
        );
        // And the two halves joined: the reader's output, fed to the rule, excludes.
        assert_eq!(
            bearing(
                Some(41809),
                None,
                process_executable(std::process::id() as i32).as_deref(),
                OUR_PID,
                Some(&ours),
            ),
            Bearing::Excluded("another meethook instance")
        );
    }

    #[test]
    fn an_unreadable_pid_resolves_to_no_executable() {
        // pid 0 is the kernel, which has no path libproc will hand back. Exercises the
        // `length <= 0` arm, which is what feeds the "counted as activity" direction.
        assert_eq!(process_executable(0), None);
    }

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
