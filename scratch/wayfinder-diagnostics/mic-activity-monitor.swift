import CoreAudio
import Foundation

// Throwaway diagnostic for TASK-001.06 (empirically validate mic mute-vs-deactivate
// device behavior). Watches the default input device's
// kAudioDevicePropertyDeviceIsRunningSomewhere flag and prints every change with a
// timestamp, so you can join a solo call, toggle that app's mute button, and see
// whether the OS-level device actually stops running or only the app silences it.
//
// Usage:
//   swift mic-activity-monitor.swift
//
// Leave it running in a terminal, join a solo Zoom/Meet/Teams test call in another
// window, toggle mute a few times, then end the call and watch what happens when it's
// fully over. Ctrl+C to stop.

func getDefaultInputDevice() -> AudioDeviceID? {
    var deviceID = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultInputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID)
    guard status == noErr else {
        print("Failed to get default input device, OSStatus \(status)")
        return nil
    }
    return deviceID
}

func getDeviceName(_ deviceID: AudioDeviceID) -> String {
    var name: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioObjectPropertyName,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    let status = AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &name)
    if status == noErr, let name {
        return name.takeRetainedValue() as String
    }
    return "<unknown>"
}

func getRunningSomewhere(_ deviceID: AudioDeviceID) -> Bool {
    var value: UInt32 = 0
    var size = UInt32(MemoryLayout<UInt32>.size)
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyDeviceIsRunningSomewhere,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    let status = AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &value)
    if status != noErr {
        print("Failed to read IsRunningSomewhere, OSStatus \(status)")
    }
    return value != 0
}

guard let deviceID = getDefaultInputDevice() else {
    exit(1)
}

let name = getDeviceName(deviceID)
let formatter = DateFormatter()
formatter.dateFormat = "HH:mm:ss.SSS"

print("Monitoring default input device: \(name) (id \(deviceID))")
print("Initial kAudioDevicePropertyDeviceIsRunningSomewhere = \(getRunningSomewhere(deviceID))")
print("Press Ctrl+C to stop. Join your solo test call, toggle mute, and watch below.\n")

var address = AudioObjectPropertyAddress(
    mSelector: kAudioDevicePropertyDeviceIsRunningSomewhere,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain
)

let listenerQueue = DispatchQueue(label: "mic-activity-monitor")
let status = AudioObjectAddPropertyListenerBlock(deviceID, &address, listenerQueue) { _, _ in
    let running = getRunningSomewhere(deviceID)
    let ts = formatter.string(from: Date())
    print("[\(ts)] IsRunningSomewhere changed -> \(running)")
}

if status != noErr {
    print("Failed to add property listener, OSStatus \(status)")
    exit(1)
}

RunLoop.main.run()
