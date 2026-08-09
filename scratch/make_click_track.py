"""Generate a click track for the TASK-003 AC #6 sync check.

Each click is a 3 ms burst of white noise, not an ideal impulse: broadband
content survives laptop-speaker rolloff and room absorption far better, and
cross-correlation only needs the envelope edge to be sharp, which a fast
attack gives regardless of spectrum.

Written as 16-bit PCM so `afplay` and every other tool accepts it — this file
is only ever played, never analyzed, so the recorder's float32 rule is
irrelevant here.

Usage: python3 make_click_track.py <out.wav> [duration_s] [interval_s]
"""

import struct
import sys

import numpy as np

RATE = 48000
CLICK_MS = 3.0
AMPLITUDE = 0.5  # leaves headroom; the mic picks this up fine at normal volume


def main(out_path, duration_s=30.0, interval_s=3.0):
    n = int(RATE * duration_s)
    track = np.zeros(n, dtype=np.float64)

    click_n = int(RATE * CLICK_MS / 1000.0)
    rng = np.random.default_rng(0)  # fixed seed: every generated track is identical
    burst = rng.uniform(-1.0, 1.0, click_n)
    # Fast attack, quick decay. The attack edge is what the envelope
    # correlation locks onto, so it must not be smoothed.
    burst *= np.concatenate([np.ones(1), np.linspace(1.0, 0.0, click_n - 1) ** 2])
    burst *= AMPLITUDE

    # First click one second in, so it is clear of any playback fade-in.
    for start in range(RATE, n - click_n, int(RATE * interval_s)):
        track[start : start + click_n] = burst

    pcm = np.clip(track, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    data = pcm.tobytes()

    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, RATE, RATE * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(data))

    with open(out_path, "wb") as f:
        f.write(header + data)

    clicks = len(range(RATE, n - click_n, int(RATE * interval_s)))
    print(f"wrote {out_path}: {duration_s:g}s, {clicks} clicks every {interval_s:g}s, {RATE} Hz mono")


if __name__ == "__main__":
    main(
        sys.argv[1],
        float(sys.argv[2]) if len(sys.argv) > 2 else 30.0,
        float(sys.argv[3]) if len(sys.argv) > 3 else 3.0,
    )
