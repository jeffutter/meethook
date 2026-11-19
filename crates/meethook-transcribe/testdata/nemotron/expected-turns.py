#!/usr/bin/env python3
"""Independent expectation for nemotron_turns::slot_turns (TASK-088.07).

Reads nemotron-probs.bin and prints the decoded turns as Rust tuples
(start_frame, end_frame, slot), sorted by start then slot. Written separately from the Rust
logic, as a state machine over whole turns rather than runs: enter above 0.5, stay while >= 0.25,
merge gaps <= 100 frames, drop turns < 50 frames.

    python3 expected-turns.py
"""
import os
import struct

here = os.path.dirname(os.path.abspath(__file__))
raw = open(os.path.join(here, "nemotron-probs.bin"), "rb").read()
frames, speakers = struct.unpack("<II", raw[:8])
vals = struct.unpack(f"<{frames * speakers}f", raw[8:])

out = []
for slot in range(speakers):
    col = [vals[f * speakers + slot] for f in range(frames)]
    turns = []  # [start, end) in frames
    cur = None
    f = 0
    while f < frames:
        p = col[f]
        if cur is None:
            if p > 0.5:
                cur = [f, None]
        elif p < 0.25:
            cur[1] = f
            turns.append(cur)
            cur = None
        f += 1
    if cur is not None:
        cur[1] = frames
        turns.append(cur)
    merged = []
    for t in turns:
        if merged and t[0] - merged[-1][1] <= 100:
            merged[-1][1] = t[1]
        else:
            merged.append(t)
    for s, e in merged:
        if e - s >= 50:
            out.append((s, e, slot))

out.sort(key=lambda t: (t[0], t[2]))
print(f"// {frames} frames, {len(out)} turns")
for s, e, slot in out:
    print(f"({s}, {e}, {slot}),")
