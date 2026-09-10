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
//!                                 and neither fact names a user-excluded app
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
//! caught. That used to be dismissed as the development-time shape rather than the user-facing
//! one, which turned out to be wrong: a Nix install changes
//! `/nix/store/<hash>-meethook-<version>/bin/meethook` at every upgrade, so a recorder that
//! survived one has always looked like somebody else's meeting to the next.
//!
//! That class is defined away one level up rather than filtered harder here: `record` takes
//! `<root>/record.lock` before it ever gets this far, so a second instance cannot reach the
//! predicate at all (see [`meethook_session::RecordLock`], and decision-006's reversal).
//!
//! This exclusion stays regardless. Binaries built before that guard exist keep running without
//! ever taking the lock, and the recorder an upgrade left behind is precisely the process the
//! predicate has to survive on its own.
//!
//! # User-excluded apps
//!
//! The exclusions above are fixed: they name what *meethook* is. A fourth class names what
//! the user says is not a meeting, in `<root>/exclusions.json` -- a dictation tool that opens
//! the microphone for local dictation is the reported case, and it looks like a meeting in
//! every respect the fixed rules can see.
//!
//! Entries are exact matches on bundle id or executable path; there are deliberately no
//! wildcards or prefixes. The predicate fails asymmetrically -- an over-exclusion costs
//! *every* session, a missed entry costs one stray one -- so a user entry must match
//! positively and never act as a catch-all: a `com.apple.` prefix would swallow FaceTime.
//!
//! The list is loaded once in [`MicActivityWatcher::start`], beside the other one-time facts,
//! and never read again: the re-check runs under the watcher lock every couple of seconds
//! while a session is live, and a file read there would be disk I/O on the hottest path in
//! the recorder. Changes therefore take effect on the next `meethook record` run. A corrupt
//! file is a hard error at startup, never a fallback to the empty list -- the fallback is
//! exactly the bug the user was fixing, and a `record` that quietly ignored their fix would
//! leave them debugging why nothing changed.
//!
//! # Shape
//!
//! Three listener sites all feed one recomputation:
//!
//! | Object | Property | On notification |
//! |---|---|---|
//! | system | `DefaultInputDevice` | move the device listener, and report a device that moved |
//! | current input device | `DeviceIsRunningSomewhere` | -- |
//! | system | `ProcessObjectList` | -- |
//!
//! The first of those three carries a second report of its own, and it is the only thing this
//! module says that is not about the predicate: [`Activity::InputDeviceChanged`], whenever the
//! default input device genuinely moves. A capture's `AVAudioEngine` tap is bound to the device
//! that was default when it started, so a swap mid-session -- unplugged AirPods, a USB
//! interface dropping, System Settings > Sound > Input -- leaves it attached to a device that
//! is gone and delivering nothing. Reporting it here rather than from a second listener is what
//! keeps "one enum, one channel" true for the record loop; the de-duplication the device
//! listener already does is what keeps it from firing on notifications where nothing moved.
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
//!
//! # Reading the debug log
//!
//! `MEETHOOK_ACTIVITY_DEBUG=1` prints one summary line per recomputation plus one line for each
//! holder that is capturing (and one for our own process, capturing or not):
//!
//! ```text
//! [activity] Install: someone_else_is_capturing=true IsRunningSomewhere=Some(true) default-input="MacBook Pro Microphone"#79 uid=BuiltInMicrophoneDevice
//! [activity]   pid=662 com.apple.CoreSpeech IsRunningInput=true exe=/System/Library/PrivateFrameworks/CoreSpeech.framework/Versions/A/CoreSpeech devices=[] on-default=unknown
//! [activity]   pid=9848 (no bundle id) IsRunningInput=true exe=/path/to/mic-hold devices=["MacBook Pro Microphone"#79 uid=BuiltInMicrophoneDevice] on-default=yes
//! [activity]   pid=9851 (no bundle id) IsRunningInput=false exe=/path/to/mic-activity devices=[] on-default=unknown   <- meethook
//! ```
//!
//! - `exe=` is the executable behind the pid (`proc_pidpath`, canonicalized) -- the only fact
//!   that separates a second meethook or a driver helper from a bundled meeting app. A bare
//!   binary reports an *empty* bundle id rather than none, so this is the only name such a
//!   process has; `exe-unreadable` and `exe=<path> unresolved` are the two ways the path itself
//!   fails, the latter still printing the raw path.
//! - `devices=[...]` lists the device(s) that process holds *input* on, by name, id and UID.
//!   Three states are kept distinct deliberately: a non-empty list, `devices=[]` (the HAL
//!   answered "none") and `devices=? (status=n)` (the HAL refused). Measured: `com.apple.CoreSpeech`
//!   reports `IsRunningInput=true` with `devices=[]`, and only while some other process holds
//!   input -- a holder of *no* device, which the predicate counts today. That is the difference
//!   between a suspect and a phantom, and why a failed read must not print as an empty list.
//! - `on-default=yes|no|unknown` says whether that set contains the current default input device,
//!   which the summary line names again so a pasted log is self-explanatory. `unknown` covers the
//!   empty list, a refused read, and a machine with no default input device at all; an empty set
//!   never reads as "not on the default".
//!
//! `on-default=no` is not an acquittal. Aggregate and virtual devices (BlackHole, Loopback, Wave
//! Link) are their own `AudioObjectID`s, so a holder capturing through an aggregate that
//! *contains* the built-in mic reads `no`; the printed `uid=` is how a human spots that. This
//! module reports the association and does not act on it: requiring the default device would
//! silently stop counting a meeting app that captured anywhere else, and deciding that is the job
//! of the machine that actually has the problem.

use std::collections::HashMap;
use std::ffi::{OsString, c_void};
use std::fmt;
use std::os::unix::ffi::OsStringExt;
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex, Weak};

use block2::RcBlock;
use dispatch2::{DispatchQueue, DispatchQueueAttr, DispatchRetained};
use objc2_core_audio::{
    AudioObjectAddPropertyListenerBlock, AudioObjectGetPropertyData,
    AudioObjectGetPropertyDataSize, AudioObjectID, AudioObjectPropertyAddress,
    AudioObjectPropertyScope, AudioObjectPropertySelector, AudioObjectRemovePropertyListenerBlock,
    kAudioDevicePropertyDeviceIsRunningSomewhere, kAudioDevicePropertyDeviceUID,
    kAudioHardwarePropertyDefaultInputDevice, kAudioHardwarePropertyProcessObjectList,
    kAudioObjectPropertyElementMain, kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal,
    kAudioObjectPropertyScopeInput, kAudioObjectPropertyScopeOutput, kAudioObjectSystemObject,
    kAudioProcessPropertyBundleID, kAudioProcessPropertyDevices,
    kAudioProcessPropertyIsRunningInput, kAudioProcessPropertyPID,
};
use objc2_core_foundation::{CFRetained, CFString};

use meethook_session::{AppExclusions, Paths};

use crate::{Error, Result};

/// What the microphone world did.
///
/// [`Activity::Started`] and [`Activity::Stopped`] are the transitions of "some other process
/// is capturing from an input device" -- the predicate this module computes, and the trigger
/// the auto start/stop loop runs on.
///
/// [`Activity::InputDeviceChanged`] is not a transition of that predicate at all. It reports
/// which *device* the microphone world is pointing at, because an `AVAudioEngine` input tap is
/// bound to whatever device was default when it started: once that device is gone, or is no
/// longer the one in use, the tap delivers no further buffers, and a live recording that is
/// not told would be silently truncated for the rest of the meeting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activity {
    Started,
    Stopped,
    /// The system's default input device moved while the watcher was running.
    ///
    /// Never emitted for the first device seen at [`MicActivityWatcher::start`]: nothing was
    /// bound to anything before that. Emitted when the default moves from one device to
    /// another, and when the last input device disappears entirely -- an unplugged USB
    /// interface, which is the case a live capture most needs to hear about.
    InputDeviceChanged,
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
    ///
    /// "Real change" covers both kinds of [`Activity`]: an edge of the capture predicate, and
    /// the default input device moving. When one notification does both -- unplugging the
    /// device a call was using -- [`Activity::InputDeviceChanged`] is delivered first, because
    /// the device bookkeeping runs before the predicate is recomputed.
    ///
    /// `root` is the meethook data directory: the user's app-exclusion list is read from
    /// `<root>/exclusions.json` here, once, before any listener is installed. An absent file
    /// means no exclusions; a corrupt one is an error naming the path, so a `record` started
    /// with a broken list fails before watching begins rather than ignoring the user's fix.
    pub fn start(
        root: &Path,
        on_change: impl Fn(Activity) + Send + Sync + 'static,
    ) -> Result<(MicActivityWatcher, bool)> {
        // Serial: every listener block is dispatched here, so all recomputation is
        // serialized and the listener bookkeeping needs no ordering beyond the mutex.
        let queue = DispatchQueue::new("com.meethook.activity", DispatchQueueAttr::SERIAL);

        // Resolved once here, outside the state, so a load failure propagates through this
        // `Result` before any listener exists to unwind. It happens once at command start,
        // before the session-start retry loop, so a plain `?` is the whole retry story.
        let exclusions = AppExclusions::read_or_empty(&Paths::new(root))?;

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
                exclusions,
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
    /// Apps the user named in `exclusions.json`, resolved once at `start`. Consulted per
    /// process object by [`bearing`]; never re-read, see the module docs for why.
    exclusions: AppExclusions,
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
            eprintln!(
                "[activity] user exclusions: {} bundle ids, {} executables",
                self.exclusions.bundle_ids.len(),
                self.exclusions.executables.len(),
            );
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

    /// Points the `IsRunningSomewhere` listener at the current default input device, and
    /// reports a device that actually moved through `on_change`.
    ///
    /// A no-op when the device has not actually changed, so the frequent
    /// `DefaultInputDevice` notifications macOS emits neither churn listeners nor split a
    /// session. [`device_changed`] is the rule for which of them is a real move.
    ///
    /// Still `Err(Error::NoInputDevice)` when there is no device to listen to, because
    /// [`State::install`] distinguishes that from a listener the HAL refused. The difference
    /// from before is that the state is updated and the edge emitted on the way out, rather
    /// than returning early and leaving a listener attached to a device that is gone.
    fn attach_device_listener(&mut self) -> Result<()> {
        let previous = self.device.as_ref().map(|installed| installed.object);
        let current = default_input_device();
        let changed = device_changed(previous, current);

        // No device at all is a normal state to keep watching from -- the user unplugged a USB
        // mic -- and the `DefaultInputDevice` listener is on the system object rather than on
        // the device, so it still fires when one appears. The stale listener goes anyway: it is
        // attached to a device that is gone, and leaving it there is what used to make an
        // unplugged microphone the one device change nothing could hear.
        let Some(device) = current else {
            if let Some(previous) = self.device.take() {
                remove_listener(&previous, &self.queue);
            }
            if changed {
                (self.on_change)(Activity::InputDeviceChanged);
            }
            return Err(Error::NoInputDevice);
        };
        if previous == Some(device) {
            return Ok(());
        }
        if let Some(previous) = self.device.take() {
            remove_listener(&previous, &self.queue);
        }

        // Emitted before the new listener is installed, so that a HAL which refuses that
        // listener -- fatal from `install`, logged and retried from `notified` -- still cannot
        // leave a live capture bound to the device that went away without being told.
        if changed {
            (self.on_change)(Activity::InputDeviceChanged);
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
            &self.exclusions,
        )
    }

    /// Everything needed to confirm or refute the reasoning in the module docs, at the
    /// machine, from one run. Gated behind `MEETHOOK_ACTIVITY_DEBUG`.
    ///
    /// The `IsRunningSomewhere` line is the load-bearing one: it is expected to stay
    /// `true` across a call ending while meethook records, which is exactly why it cannot
    /// be the predicate.
    ///
    /// Each holder line then names everything the classifier had to work with: the executable
    /// behind the pid, the device(s) it holds input on, and whether that set includes the
    /// default input device -- whose name the summary line repeats, so a captured log needs no
    /// second tool. Those three are what TASK-066's reproduction was missing: the culprit shape
    /// (a bare binary, a driver helper) reports no bundle id at all, so the old line rendered
    /// it as `pid=NNNN  IsRunningInput=true` -- nothing but a number.
    ///
    /// Nothing here changes the verdict: the bearing comes from the same [`bearing`] the
    /// predicate uses, and this prints rather than decides. In particular `on-default` is
    /// reported and not acted on -- requiring it would silently stop counting a meeting app
    /// that captured on a non-default device, and only the machine that has the problem gets to
    /// make that call.
    ///
    /// Device labels are read once per pass and memoized, including the default device's: this
    /// runs on every non-`Recheck` notification and `record` drives
    /// [`MicActivityWatcher::recheck`] every couple of seconds, so an un-memoized read would
    /// repeat the same cross-process calls several times a second for a device set of one or
    /// two entries.
    fn log(&self, trigger: Trigger, active: bool) {
        let running_somewhere = self
            .device
            .as_ref()
            .and_then(|d| device_is_running_somewhere(d.object));
        let default_device = default_input_device();
        let mut labels: HashMap<AudioObjectID, String> = HashMap::new();
        let default_label = default_device.map(|device| device_label(device, &mut labels));
        eprintln!(
            "[activity] {trigger:?}: someone_else_is_capturing={active} \
             IsRunningSomewhere={running_somewhere:?} default-input={}",
            default_label.unwrap_or_else(|| "none".to_owned()),
        );
        for process in object_list(
            kAudioObjectSystemObject as AudioObjectID,
            kAudioHardwarePropertyProcessObjectList,
        ) {
            let running_input = process_is_running_input(process);
            let pid = process_pid(process);
            let ours = pid == Some(self.our_pid);
            if !running_input && !ours {
                continue;
            }
            // The facts a shown holder is described by and judged on are read once, here.
            // [`bearing_of`] is not reused: it throws the facts away, which is right for the
            // predicate's hot path (a non-capturing process costs one read) and wrong for a log
            // whose whole job is to print them.
            let bundle_id = process_bundle_id(process);
            let exe = pid.map_or(Exe::Unreadable, executable_of);
            // The verdict comes from the same classifier the predicate uses, so the log
            // cannot describe a rule the recorder is not actually applying -- including that
            // a process which is not capturing is Idle whatever else it says about itself.
            let bearing = if running_input {
                bearing(
                    pid,
                    bundle_id.as_deref(),
                    exe.canonical_path(),
                    self.our_pid,
                    self.our_exe.as_deref(),
                    &self.exclusions,
                )
            } else {
                Bearing::Idle
            };
            let (devices, on_default) =
                devices_reported(process_input_devices(process), default_device, &mut labels);
            let marker = match bearing {
                Bearing::Excluded(why) => format!("   <- excluded: {why}"),
                // Idle and ours: shown anyway, because "meethook is not capturing yet"
                // is the baseline the later lines are read against.
                _ if ours => "   <- meethook".to_owned(),
                _ => String::new(),
            };
            eprintln!(
                "{}",
                holder_line(
                    pid,
                    bundle_id.as_deref(),
                    running_input,
                    &exe,
                    &devices,
                    on_default,
                    &marker,
                )
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

/// The executable behind a pid, in the three states [`executable_of`] can land in.
///
/// [`bearing`] sees only the [`Exe::Resolved`] arm (through [`Exe::canonical_path`]) -- the
/// predicate keeps comparing canonicalized paths and nothing else about it changes. The other
/// two arms exist for the debug log, where "which kind of failure is this" is the question.
#[derive(Debug, PartialEq)]
enum Exe {
    /// `proc_pidpath` answered and the path canonicalized.
    Resolved(PathBuf),
    /// `proc_pidpath` answered; the path did not canonicalize. Carries the raw path, because a
    /// name we cannot compare is still a name.
    Unresolved(PathBuf),
    /// `proc_pidpath` refused: an exited process, another user's, or pid 0.
    Unreadable,
}

impl Exe {
    /// The canonicalized path [`bearing`] compares; `None` for both failure arms.
    fn canonical_path(&self) -> Option<&Path> {
        match self {
            Exe::Resolved(path) => Some(path),
            Exe::Unresolved(_) | Exe::Unreadable => None,
        }
    }
}

impl fmt::Display for Exe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Exe::Resolved(path) => write!(f, "exe={}", path.display()),
            // Printed with its limitation rather than dropped: the raw path names the program,
            // which is the thing a captured log is being read for.
            Exe::Unresolved(path) => write!(f, "exe={} unresolved", path.display()),
            Exe::Unreadable => f.write_str("exe-unreadable"),
        }
    }
}

/// What the read of one holder's input devices said, with all three outcomes kept apart.
#[derive(Debug, PartialEq)]
enum Devices {
    /// Non-empty, each entry labelled by [`device_label`].
    Read(Vec<String>),
    /// The HAL answered, and the answer was "no input device": a holder of nothing.
    Empty,
    /// The HAL refused the property; the `OSStatus` is printed with it.
    Unreadable(i32),
}

/// Turns a [`process_input_devices`] answer into what the log shows and what it means.
///
/// Kept together because the two halves read as one sentence -- a holder on no device is
/// `devices=[] on-default=unknown`, never `on-default=no` -- and because the label cache is
/// shared across every holder in one pass.
fn devices_reported(
    held: std::result::Result<Vec<AudioObjectID>, i32>,
    default: Option<AudioObjectID>,
    cache: &mut HashMap<AudioObjectID, String>,
) -> (Devices, OnDefault) {
    match held {
        Err(status) => (Devices::Unreadable(status), OnDefault::Unknown),
        Ok(ids) if ids.is_empty() => (Devices::Empty, OnDefault::Unknown),
        Ok(ids) => {
            let labels = ids.iter().map(|id| device_label(*id, cache)).collect();
            (Devices::Read(labels), shares_default_device(default, &ids))
        }
    }
}

impl fmt::Display for Devices {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Devices::Read(labels) => write!(f, "devices=[{}]", labels.join(", ")),
            Devices::Empty => f.write_str("devices=[]"),
            Devices::Unreadable(status) => write!(f, "devices=? (status={status})"),
        }
    }
}

/// Whether a holder's input-device set contains the current default input device.
///
/// Three values, because "not on the default" and "we could not tell" lead a reader to opposite
/// conclusions, and an empty set must never read as the former.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OnDefault {
    Yes,
    No,
    /// The holder holds no device, the read failed, or the machine has no default input device
    /// at all (an unplugged-USB machine, which [`State::attach_device_listener`] treats as
    /// normal).
    Unknown,
}

impl fmt::Display for OnDefault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            OnDefault::Yes => "yes",
            OnDefault::No => "no",
            OnDefault::Unknown => "unknown",
        })
    }
}

/// Whether the devices a holder input on include the system's default input device.
///
/// Pure over ids for the same reason [`bearing`] is: this decides what a pasted log means, and
/// the sandbox this is developed in has no audio device to ask.
///
/// Compared by exact `AudioObjectID`, which is why `no` is not an acquittal for a virtual or
/// aggregate device: BlackHole, Loopback and Wave Link devices are their own ids, so a holder
/// capturing through an aggregate that *contains* the built-in mic reads `no`. The printed `uid=`
/// is how a human spots that case; reading sub-device lists to decide it automatically is
/// deliberately left to whichever machine actually shows one.
fn shares_default_device(default: Option<AudioObjectID>, held: &[AudioObjectID]) -> OnDefault {
    match (default, held.is_empty()) {
        (_, true) => OnDefault::Unknown,
        (Some(device), false) if held.contains(&device) => OnDefault::Yes,
        (Some(_), false) => OnDefault::No,
        // A machine with no default input device cannot say anything about the set it would
        // have compared to.
        (None, false) => OnDefault::Unknown,
    }
}

/// One holder line of the debug log, assembled apart from the reads it describes.
///
/// Pure so the shapes a human diagnoses from are testable with no CoreAudio device present,
/// including the trailing marker: exclusion reasons are rendered against it, and a new field
/// must not move it.
///
/// Field order keeps the pre-existing prefix bytes (`pid=`, bundle id, `IsRunningInput=`) in
/// place so muscle memory and greps survive; the marker stays last.
fn holder_line(
    pid: Option<i32>,
    bundle_id: Option<&str>,
    running_input: bool,
    exe: &Exe,
    devices: &Devices,
    on_default: OnDefault,
    marker: &str,
) -> String {
    format!(
        "[activity]   pid={} {} IsRunningInput={running_input} {exe} {devices} on-default={on_default}{marker}",
        pid.map_or_else(|| "?".to_owned(), |p| p.to_string()),
        bundle_id.unwrap_or("(no bundle id)"),
    )
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
///
/// A user entry participates under the same rule: it fires only when the fact it keys on is
/// present. A bundle-id entry cannot fire for a process whose bundle id is unreadable --
/// that process still counts as activity unless its executable entry names it -- so an
/// unreadable fact never silently widens the user's list.
fn bearing(
    pid: Option<i32>,
    bundle_id: Option<&str>,
    executable: Option<&Path>,
    our_pid: i32,
    our_exe: Option<&Path>,
    exclusions: &AppExclusions,
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
    if bundle_id.is_some_and(|id| exclusions.contains_bundle_id(id)) {
        return Bearing::Excluded("user-excluded bundle id");
    }
    if let (Some(exe), Some(ours)) = (executable, our_exe)
        && exe == ours
    {
        return Bearing::Excluded("another meethook instance");
    }
    if executable.is_some_and(|exe| exclusions.contains_executable(exe)) {
        return Bearing::Excluded("user-excluded executable");
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

/// Whether moving from `previous` to `current` is a change a live capture has to be told
/// about.
///
/// Pure for the same reason [`bearing`] and [`edge`] are: the sandbox this is developed in has
/// no audio device at all, so the rule the whole reaction turns on has to be decidable without
/// one.
///
/// The first attach is not a change -- there was no engine bound to anything before it, so
/// starting the watcher must not look like a device swap. Losing the last input device *is*
/// one: whatever a live tap is bound to has gone away, which is exactly the truncation this
/// reports.
fn device_changed(previous: Option<AudioObjectID>, current: Option<AudioObjectID>) -> bool {
    match (previous, current) {
        (None, _) => false,
        (Some(previous), Some(current)) => previous != current,
        (Some(_), None) => true,
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
    address_scoped(selector, kAudioObjectPropertyScopeGlobal)
}

fn address_scoped(
    selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope,
) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
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
    // SAFETY: same requirements as [`property_scoped`], at the global scope.
    unsafe { property_scoped(object, selector, kAudioObjectPropertyScopeGlobal) }
}

/// Reads a fixed-size property at an explicit scope.
///
/// The scope matters for more than device properties: `kAudioProcessPropertyDevices` is
/// scope-selected, so reading it globally answers "no devices" even for a process holding the
/// microphone (see [`process_input_devices`]).
///
/// # Safety
///
/// `T` must be the exact type CoreAudio returns for `selector` on `object` *at that scope*.
unsafe fn property_scoped<T: Copy + Default>(
    object: AudioObjectID,
    selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope,
) -> Option<T> {
    let address = address_scoped(selector, scope);
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

/// Reads an `AudioObjectID` array property at the global scope, or empty if it will not answer.
fn object_list(object: AudioObjectID, selector: AudioObjectPropertySelector) -> Vec<AudioObjectID> {
    object_list_scoped(object, selector, kAudioObjectPropertyScopeGlobal).unwrap_or_default()
}

/// Reads an `AudioObjectID` array property at an explicit scope, sized first so nothing is
/// truncated.
///
/// The `OSStatus` is kept rather than swallowed into an empty list because "the object has no
/// such property" and "the object says the list is empty" are different facts, and the debug
/// log has to tell them apart -- see [`process_input_devices`].
fn object_list_scoped(
    object: AudioObjectID,
    selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope,
) -> std::result::Result<Vec<AudioObjectID>, i32> {
    let address = address_scoped(selector, scope);
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
        return Err(status);
    }

    let mut ids = vec![0 as AudioObjectID; size as usize / size_of::<AudioObjectID>()];
    let Some(buffer) = NonNull::new(ids.as_mut_ptr()) else {
        // An allocation that failed at zero-ish size is not a HAL answer; report it as one so
        // the caller shows `?` rather than an empty list.
        return Err(-1);
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
        return Err(status);
    }

    // The set can shrink between the two calls; trust the second answer.
    ids.truncate(size as usize / size_of::<AudioObjectID>());
    Ok(ids)
}

/// The devices a process object holds *input* on.
///
/// The scope is the whole point of this function. Apple documents `kAudioProcessPropertyDevices`
/// as scope-selected (`AudioHardware.h:1958-1961`: "The scope will select the input or output
/// device list"), measured here with `mic-hold` holding the microphone: the global scope answers
/// `[]`, the input scope `[79 'MacBook Pro Microphone']`, the output scope
/// `[72 'MacBook Pro Speakers']`. Reading it globally would print an empty list for every
/// holder forever.
///
/// The three outcomes are kept distinct on purpose. `Ok(vec![])` is a real observed state --
/// `com.apple.CoreSpeech` reports `IsRunningInput = 1` with an empty *input* device list, and
/// only while some other process holds input -- i.e. a holder of no device at all, which the
/// predicate counts today. That is a different fact from an `Err`, where the HAL has no such
/// property or will not answer, and collapsing them would lose exactly the discrimination this
/// output exists to provide.
fn process_input_devices(process: AudioObjectID) -> std::result::Result<Vec<AudioObjectID>, i32> {
    object_list_scoped(
        process,
        kAudioProcessPropertyDevices,
        kAudioObjectPropertyScopeInput,
    )
}

/// Reads a copy-accessor `CFStringRef` property, which `CFRetained` then owns.
///
/// # Safety
///
/// `selector` must return a single `CFStringRef` on `object` at `scope`.
unsafe fn cfstring_property(
    object: AudioObjectID,
    selector: AudioObjectPropertySelector,
    scope: AudioObjectPropertyScope,
) -> Option<CFRetained<CFString>> {
    // `Option<NonNull<_>>` is used as the destination because it is pointer-sized and its
    // `Default` is null, which is exactly what a "no value written" outcome should read as.
    // SAFETY: the qualifier is null and the destination is a live local, as in `property`.
    let raw: Option<NonNull<CFString>> = unsafe { property_scoped(object, selector, scope)? };
    // SAFETY: a copy accessor hands back a +1 reference, which `CFRetained` now owns.
    Some(unsafe { CFRetained::from_raw(raw?) })
}

/// Reads a string device property, falling back across the scopes it plausibly lives at.
///
/// The three candidates are tried in order and the first non-empty answer wins; which one
/// answered is never interesting downstream. On the planning machine the global scope answers
/// for both name and UID, so the fallbacks are cheap insurance against a driver that files them
/// under the device's direction instead -- they cost two extra refused reads only on a device
/// whose label we would otherwise print as `?`. An empty answer counts as no answer for the same
/// reason: a driver returning `""` for a name has told us nothing.
fn device_string(device: AudioObjectID, selector: AudioObjectPropertySelector) -> Option<String> {
    [
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyScopeInput,
        kAudioObjectPropertyScopeOutput,
    ]
    .into_iter()
    .find_map(|scope| {
        // SAFETY: the selectors used at the one call site below return a single `CFStringRef`.
        unsafe { cfstring_property(device, selector, scope) }
            .map(|string| string.to_string())
            .filter(|string| !string.trim().is_empty())
    })
}

/// The device's label as `"NAME"#<id> uid=<UID>`, memoized for the duration of one log pass.
///
/// Each half falls back on its own (`"?#79 uid=BuiltInMicrophoneDevice", `uid=?`) instead of
/// dropping the whole label: the numeric id alone is nearly worthless in a pasted log because it
/// changes across boots, which is precisely why the stable UID rides beside it.
fn device_label(id: AudioObjectID, cache: &mut HashMap<AudioObjectID, String>) -> String {
    cache
        .entry(id)
        .or_insert_with(|| {
            let name = device_string(id, kAudioObjectPropertyName);
            let uid = device_string(id, kAudioDevicePropertyDeviceUID);
            format!(
                "{}#{id} uid={}",
                name.as_deref()
                    .map_or_else(|| "?".to_owned(), |n| format!("\"{n}\"")),
                uid.as_deref().unwrap_or("?"),
            )
        })
        .clone()
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
    // SAFETY: this selector returns a single `CFStringRef` at the global scope.
    let string = unsafe {
        cfstring_property(
            process,
            kAudioProcessPropertyBundleID,
            kAudioObjectPropertyScopeGlobal,
        )?
    };
    // A bare binary reports an *empty* bundle id rather than none -- measured with `afplay`
    // and with `mic-hold`, both of which come back `''` -- so without this the predicate's
    // `is_some_and` checks saw `Some("")` and the log printed a blank where the name should
    // be. Empty is not a fact about the process, so it normalizes to "not reported".
    //
    // The predicate is unaffected: the empty string is in neither `OUR_HELPER_BUNDLE_IDS` nor
    // any plausible `exclusions.json`. One side effect is intended -- a hand-edited `""` entry
    // in that file can no longer exclude *every* bare binary.
    let id = string.to_string();
    (!id.trim().is_empty()).then_some(id)
}

/// The executable behind a pid, canonicalized, or `None` for one that cannot be read.
///
/// This is the [`Exe::Resolved`] view only, because that is all [`bearing`] compares; use
/// [`executable_of`] when the failure kind matters, as the debug log does.
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
    match executable_of(pid) {
        Exe::Resolved(path) => Some(path),
        Exe::Unresolved(_) | Exe::Unreadable => None,
    }
}

/// Reads the executable behind a pid, keeping which kind of failure it was.
///
/// The two failure arms are different things to say in a log: one still names the program.
fn executable_of(pid: i32) -> Exe {
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
        return Exe::Unreadable;
    }
    buffer.truncate(length as usize);
    let raw = PathBuf::from(OsString::from_vec(buffer));
    match std::fs::canonicalize(&raw) {
        Ok(path) => Exe::Resolved(path),
        // Reached by unreadable path components -- a root-owned helper under a `0700`
        // directory -- rather than by a deleted or rebuilt binary: unlinking or renaming a
        // running executable's file gets the process `SIGKILL`ed on macOS, which was tried
        // twice while planning this. So the name libproc already gave us is worth printing
        // even though it cannot be compared to ours.
        Err(_) => Exe::Unresolved(raw),
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use meethook_session::AppExclusions;

    use super::{
        Activity, Bearing, Devices, Exe, OnDefault, bearing, device_changed, edge, holder_line,
        process_executable, shares_default_device,
    };

    const OUR_PID: i32 = 500;

    /// Our own executable, as `MicActivityWatcher::start` would have canonicalized it.
    fn our_exe() -> &'static Path {
        Path::new("/usr/local/bin/meethook")
    }

    /// No user exclusions: the whole pre-`exclusions.json` matrix below runs against this.
    fn no_exclusions() -> &'static AppExclusions {
        static EXCLUSIONS: AppExclusions = AppExclusions {
            bundle_ids: Vec::new(),
            executables: Vec::new(),
        };
        &EXCLUSIONS
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
                no_exclusions(),
            ),
            Bearing::Activity
        );
        // A meeting app that reports no bundle id is still a meeting app.
        assert_eq!(
            bearing(
                Some(26975),
                None,
                None,
                OUR_PID,
                Some(our_exe()),
                no_exclusions(),
            ),
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
                no_exclusions(),
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
                no_exclusions(),
            ),
            Bearing::Excluded(_)
        ));
    }

    #[test]
    fn a_user_excluded_bundle_id_is_not_the_signal() {
        // The reported trigger case: a dictation tool opening the microphone for local
        // dictation, named by the user rather than by a rebuild. The reason is asserted
        // exactly, not just matched: `State::log` renders it as `<- excluded: {why}`, so
        // this string is what a hardware run shows against the filtered pid.
        let exclusions = AppExclusions {
            bundle_ids: vec!["com.example.voiceink".to_owned()],
            executables: Vec::new(),
        };
        assert_eq!(
            bearing(
                Some(777),
                Some("com.example.voiceink"),
                Some(Path::new(
                    "/Applications/VoiceInk.app/Contents/MacOS/VoiceInk"
                )),
                OUR_PID,
                Some(our_exe()),
                &exclusions,
            ),
            Bearing::Excluded("user-excluded bundle id")
        );
    }

    #[test]
    fn a_user_excluded_executable_is_not_the_signal() {
        // The plain-binary case: no bundle id to key on, so the executable entry is the
        // only fact that can name it -- which is also why the file documents listing the
        // real executable inside `.app/Contents/MacOS/`.
        let exclusions = AppExclusions {
            bundle_ids: Vec::new(),
            executables: vec![PathBuf::from("/opt/homebrew/bin/some-dictation-tool")],
        };
        assert_eq!(
            bearing(
                Some(777),
                None,
                Some(Path::new("/opt/homebrew/bin/some-dictation-tool")),
                OUR_PID,
                Some(our_exe()),
                &exclusions,
            ),
            Bearing::Excluded("user-excluded executable")
        );
    }

    #[test]
    fn an_unlisted_app_is_still_the_signal_with_a_populated_exclusion_list() {
        // The over-exclusion guard with the list populated: membership is exact, so an app
        // the list does not name -- even a near-miss spelling of an entry -- is still the
        // meeting signal.
        let exclusions = AppExclusions {
            bundle_ids: vec!["com.example.voiceink".to_owned()],
            executables: Vec::new(),
        };
        assert_eq!(
            bearing(
                Some(26975),
                Some("com.example.voiceink.helper"),
                Some(Path::new("/Applications/Other.app/Contents/MacOS/Other")),
                OUR_PID,
                Some(our_exe()),
                &exclusions,
            ),
            Bearing::Activity
        );
    }

    #[test]
    fn a_user_entry_only_fires_when_its_fact_is_present() {
        // An unreadable fact never silently widens the user's list: a bundle-id entry cannot
        // fire for a process whose bundle id is unreadable (counted as activity unless its
        // executable entry names it), and the pid-unreadable exclusion still outranks it.
        let exclusions = AppExclusions {
            bundle_ids: vec!["com.example.voiceink".to_owned()],
            executables: Vec::new(),
        };
        assert_eq!(
            bearing(Some(777), None, None, OUR_PID, Some(our_exe()), &exclusions,),
            Bearing::Activity
        );
        assert_eq!(
            bearing(
                None,
                Some("com.example.voiceink"),
                None,
                OUR_PID,
                Some(our_exe()),
                &exclusions,
            ),
            Bearing::Excluded("pid unreadable, so it cannot be shown not to be meethook")
        );
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
            bearing(
                Some(41809),
                None,
                Some(our_exe()),
                OUR_PID,
                Some(our_exe()),
                no_exclusions(),
            ),
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
                no_exclusions(),
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
            bearing(
                Some(26975),
                None,
                None,
                OUR_PID,
                Some(our_exe()),
                no_exclusions(),
            ),
            Bearing::Activity
        );
        // Same when it is *our* path that could not be resolved: the exclusion turns off
        // rather than firing on everything.
        assert_eq!(
            bearing(
                Some(26975),
                None,
                Some(our_exe()),
                OUR_PID,
                None,
                no_exclusions(),
            ),
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
                Some(our_exe()),
                no_exclusions()
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
                no_exclusions(),
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

    #[test]
    fn the_first_device_attach_is_not_a_change() {
        // Installing the watcher must not look like a device swap: nothing is recording yet,
        // and a `record` that split a session at startup would be worse than the bug.
        assert!(!device_changed(None, Some(1)));
        // Nor is starting with no input device at all, which is a state the watcher is
        // explicitly allowed to run in.
        assert!(!device_changed(None, None));
    }

    #[test]
    fn moving_to_a_different_device_is_a_change() {
        assert!(device_changed(Some(1), Some(2)));
        // The de-dupe macOS's frequent `DefaultInputDevice` notifications depend on: the same
        // device reported again is not a swap, and must not split a session.
        assert!(!device_changed(Some(1), Some(1)));
    }

    #[test]
    fn losing_the_only_input_device_is_a_change() {
        // Unplugging the USB mic a call is being recorded through. The engine is bound to a
        // device that no longer exists, so this is the case a live capture most needs to hear
        // about -- and the one that used to return early before the state was touched, leaving
        // the microphone track to truncate in silence.
        assert!(device_changed(Some(1), None));
        // Repeat notifications with still no device are not further changes.
        assert!(!device_changed(None, None));
    }

    /// A holder line with everything except the field under test fixed, so the assertions read
    /// as the difference they are about.
    fn line_for(exe: Exe, devices: Devices, on_default: OnDefault) -> String {
        holder_line(Some(41809), None, true, &exe, &devices, on_default, "")
    }

    const MIC: &str = "\"MacBook Pro Microphone\"#79 uid=BuiltInMicrophoneDevice";

    #[test]
    fn an_excluded_holder_line_names_everything_and_leaves_the_marker_last() {
        // Pinned byte-for-byte, marker included: exclusion reasons are rendered as
        // `<- excluded: {why}` and read off the end of the line, so a field inserted later must
        // not push it around. This is the shape a captured log is diagnosed from.
        let line = holder_line(
            Some(997),
            Some("com.apple.replayd"),
            true,
            &Exe::Resolved(PathBuf::from("/usr/libexec/replayd")),
            &Devices::Read(vec![MIC.to_owned()]),
            OnDefault::Yes,
            "   <- excluded: captures on meethook's behalf",
        );
        let body = "[activity]   pid=997 com.apple.replayd IsRunningInput=true \
                    exe=/usr/libexec/replayd devices=[\"MacBook Pro Microphone\"#79 \
                    uid=BuiltInMicrophoneDevice] on-default=yes";
        assert_eq!(
            line,
            format!("{body}   <- excluded: captures on meethook's behalf")
        );
    }

    #[test]
    fn a_bare_binary_holder_is_named_by_its_executable() {
        // The shape TASK-066 could not diagnose: a plain binary reports no bundle id, so the
        // old line was `pid=NNNN  IsRunningInput=true` -- a number and nothing else. `(no bundle
        // id)` is still said, but it is never the whole answer now.
        assert_eq!(
            line_for(
                Exe::Resolved(PathBuf::from("/tmp/mic-hold")),
                Devices::Read(vec![MIC.to_owned()]),
                OnDefault::Yes,
            ),
            "[activity]   pid=41809 (no bundle id) IsRunningInput=true exe=/tmp/mic-hold \
             devices=[\"MacBook Pro Microphone\"#79 uid=BuiltInMicrophoneDevice] on-default=yes"
        );
    }

    #[test]
    fn an_unreadable_executable_says_so_instead_of_blanking_the_name() {
        assert!(
            line_for(Exe::Unreadable, Devices::Empty, OnDefault::Unknown)
                .contains(" exe-unreadable devices=[]")
        );
    }

    #[test]
    fn an_unresolved_executable_keeps_the_raw_path() {
        // Reached by unreadable path components, not by a rebuilt binary (see `executable_of`).
        // The name is worth printing even though it cannot be compared to ours, so it prints
        // with its limitation stated rather than being replaced by nothing.
        assert!(
            line_for(
                Exe::Unresolved(PathBuf::from("/usr/libexec/private/helper")),
                Devices::Empty,
                OnDefault::Unknown,
            )
            .contains(" exe=/usr/libexec/private/helper unresolved")
        );
    }

    #[test]
    fn an_empty_device_list_is_not_a_failed_read() {
        // Both mean "we cannot say on-default", but they are opposite claims about the process:
        // one holds no device at all (the observed `com.apple.CoreSpeech` shape), the other
        // answered nothing at all. Collapsing them is what made the old output unusable.
        let empty = line_for(
            Exe::Resolved(PathBuf::from("/tmp/mic-hold")),
            Devices::Empty,
            OnDefault::Unknown,
        );
        let failed = line_for(
            Exe::Resolved(PathBuf::from("/tmp/mic-hold")),
            Devices::Unreadable(-4),
            OnDefault::Unknown,
        );
        assert!(empty.ends_with("devices=[] on-default=unknown"), "{empty}");
        assert!(
            failed.ends_with("devices=? (status=-4) on-default=unknown"),
            "{failed}"
        );
    }

    #[test]
    fn the_default_device_flag_has_three_answers() {
        assert_eq!(shares_default_device(Some(79), &[79]), OnDefault::Yes);
        // A virtual or aggregate device is its own id, so this reads `no` even when it contains
        // the built-in mic -- see the module docs on why that is reported, not resolved.
        assert_eq!(shares_default_device(Some(79), &[120]), OnDefault::No);
        // Holding input on nothing is not the same as holding it somewhere else.
        assert_eq!(shares_default_device(Some(79), &[]), OnDefault::Unknown);
        // Nor is anything else comparable on a machine whose last input device went away.
        assert_eq!(shares_default_device(None, &[79]), OnDefault::Unknown);
    }

    #[test]
    fn the_meethook_marker_line_keeps_its_shape() {
        // "meethook is not capturing yet" is the baseline every later line is read against, so
        // it keeps the same field order and the same trailing marker as the pre-existing line.
        assert_eq!(
            holder_line(
                Some(500),
                None,
                false,
                &Exe::Resolved(PathBuf::from("/usr/local/bin/meethook")),
                &Devices::Empty,
                OnDefault::Unknown,
                "   <- meethook",
            ),
            "[activity]   pid=500 (no bundle id) IsRunningInput=false \
             exe=/usr/local/bin/meethook devices=[] on-default=unknown   <- meethook"
        );
    }
}
