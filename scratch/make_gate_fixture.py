"""Builds the speech-in-silence fixture the speech gate is verified against.

The gate in `meethook-transcribe`'s `gate` module has to be checked against real
speech surrounded by real silence, at a known offset, with text the code has
never been told about. A tone is not speech -- Silero is trained on speech and
correctly declines a sine -- and this repository ships no speech fixture, so the
clip is cut out of a session that is already on disk.

    python3 make_gate_fixture.py <session-dir> <out-dir> [start_s] [clip_s] [offset_s] [total_s]

Writes two 48 kHz mono float32 wavs into <out-dir>:

    speech-at-<offset>.wav   total_s of digital silence with clip_s of speech at offset_s
    silence.wav              total_s of digital silence, nothing else

48 kHz float32 because that is what the recorder writes and what
`read_track_16k_mono` accepts; meethook's own resampler then takes it to 16 kHz,
so the fixture exercises the same read path a real session does.

Defaults cut 1720.88..1730.88 s of `20260810-093047`'s speaker track -- a turn
the baseline transcript reads as "Like, let's solve the immediate problem..." --
and place it at 45.0 s of a 120 s track. The speaker track's session offset is
zero for that session, so transcript time and `speaker.wav` time are the same
number.
"""

import struct
import sys
from pathlib import Path

import numpy as np

RATE = 48_000


def read_f32_mono(path, rate=RATE):
    """Reads a mono float32 wav, whatever fmt tag it carries.

    `wave` refuses the WAVE_FORMAT_EXTENSIBLE header the recorder writes, so the
    chunks are walked directly. Only the fields that must be true are checked.
    """
    with open(path, "rb") as f:
        assert f.read(4) == b"RIFF", path
        f.read(4)
        assert f.read(4) == b"WAVE", path
        while True:
            header = f.read(8)
            if len(header) < 8:
                raise SystemExit(f"{path}: no data chunk")
            chunk_id, size = struct.unpack("<4sI", header)
            body_at = f.tell()
            if chunk_id == b"fmt ":
                body = f.read(size)
                _, channels, sample_rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
                assert channels == 1, f"{path}: {channels} channels"
                assert bits == 32, f"{path}: {bits}-bit"
                assert sample_rate == rate, f"{path}: {sample_rate} Hz"
            elif chunk_id == b"data":
                return np.fromfile(f, dtype="<f4", count=size // 4)
            f.seek(body_at + size)


def write_f32_mono(path, samples, rate=RATE):
    """Writes WAVE_FORMAT_IEEE_FLOAT, which is what hound reads as `Float`."""
    data = np.asarray(samples, dtype="<f4").tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, rate, rate * 4, 4, 32))
        f.write(b"data" + struct.pack("<I", len(data)))
        f.write(data)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    session = Path(sys.argv[1])
    out = Path(sys.argv[2])
    start_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1720.88
    clip_s = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
    offset_s = float(sys.argv[5]) if len(sys.argv) > 5 else 45.0
    total_s = float(sys.argv[6]) if len(sys.argv) > 6 else 120.0

    out.mkdir(parents=True, exist_ok=True)
    speaker = read_f32_mono(session / "speaker.wav")
    clip = speaker[int(start_s * RATE) : int((start_s + clip_s) * RATE)]
    if len(clip) < int(clip_s * RATE):
        raise SystemExit(f"the session is too short to cut {start_s}..{start_s + clip_s} s")

    peak = float(np.max(np.abs(clip)))
    rms = float(np.sqrt(np.mean(clip.astype(np.float64) ** 2)))
    print(f"clip {start_s}..{start_s + clip_s} s  peak {peak:.4f}  rms {rms:.5f}")
    if peak < 0.01:
        raise SystemExit("that stretch of the session is nearly silent; pick another offset")

    track = np.zeros(int(total_s * RATE), dtype="<f4")
    track[int(offset_s * RATE) : int(offset_s * RATE) + len(clip)] = clip
    speech_path = out / f"speech-at-{offset_s:g}.wav"
    write_f32_mono(speech_path, track)
    silence_path = out / "silence.wav"
    write_f32_mono(silence_path, np.zeros(int(total_s * RATE), dtype="<f4"))

    print(f"wrote {speech_path}  ({total_s:g} s, speech {offset_s:g}..{offset_s + clip_s:g} s)")
    print(f"wrote {silence_path}  ({total_s:g} s, digital silence)")


if __name__ == "__main__":
    main()
