import AVFoundation
import CoreAudio
import Foundation

func defaultInputDeviceName() -> String {
    var deviceID = AudioDeviceID(0)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultInputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID) == noErr else {
        return "<unknown device>"
    }

    var name: Unmanaged<CFString>?
    var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    var nameAddress = AudioObjectPropertyAddress(
        mSelector: kAudioObjectPropertyName,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    guard AudioObjectGetPropertyData(deviceID, &nameAddress, 0, nil, &nameSize, &name) == noErr, let name else {
        return "<unknown device>"
    }
    return name.takeRetainedValue() as String
}

// Throwaway diagnostic for TASK-001.04 (empirically test acoustic speaker-bleed severity).
// Records the default mic input while `say` speaks a test phrase through the speakers,
// so you can listen back and judge whether the mic picked up the spoken audio.
//
// Usage:
//   swift speaker-bleed-test.swift <output.wav> ["phrase to speak"]
//
// Run once with headphones in, once with speakers/no headphones, and compare.

let args = CommandLine.arguments
guard args.count >= 2 else {
    print("Usage: swift speaker-bleed-test.swift <output.wav> [\"phrase to speak\"]")
    exit(1)
}
let outputPath = args[1]
let phrase = args.count >= 3
    ? args[2]
    : "The quick brown fox jumps over the lazy dog. This is a speaker bleed test. " +
      "Testing one, two, three, four, five. If you can hear this clearly in the " +
      "microphone recording, speaker bleed is present."

let outputURL = URL(fileURLWithPath: outputPath)

let engine = AVAudioEngine()
let input = engine.inputNode
let inputFormat = input.outputFormat(forBus: 0)

print("Recording from default input device: \(defaultInputDeviceName())")
print("Input device format: \(inputFormat)")

guard let file = try? AVAudioFile(forWriting: outputURL, settings: inputFormat.settings) else {
    print("Failed to create output file at \(outputPath)")
    exit(1)
}

input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
    try? file.write(from: buffer)
}

do {
    try engine.start()
} catch {
    print("Failed to start audio engine: \(error)")
    print("If this is a permissions error, grant microphone access to your terminal app")
    print("in System Settings > Privacy & Security > Microphone, then try again.")
    exit(1)
}

print("Recording started. Speaking test phrase in 1 second...")
Thread.sleep(forTimeInterval: 1.0)

let say = Process()
say.executableURL = URL(fileURLWithPath: "/usr/bin/say")
say.arguments = [phrase]
do {
    try say.run()
    say.waitUntilExit()
} catch {
    print("Failed to run /usr/bin/say: \(error)")
}

print("Speech finished. Recording 1 more second, then stopping...")
Thread.sleep(forTimeInterval: 1.0)

input.removeTap(onBus: 0)
engine.stop()

print("Done. Wrote \(outputPath)")
print("Play it back with: afplay \(outputPath)")
