"""Per-click alignment of a meethook session recorded against click.wav.

Replaces the global envelope correlation in sync_check.py, which fails on this
signal: ten 3 ms clicks are ~30 ms of shared content in a 35 s take, so a
whole-track correlation is dominated by mic speech against speaker silence and
returns noise.

Instead: locate each click in the speaker track (where it is clean and known),
then search a short window of the mic track around the expected arrival for the
matching transient. Each click yields an independent residual, so the spread
across clicks measures drift directly and the per-click SNR says whether the mic
heard the click at all.

Usage: python3 click_align.py <session-dir>
"""

import json
import sys
from pathlib import Path

import numpy as np

from sync_check import envelope, read_float_wav

# How far after the speaker click to look for it in the mic. Output buffering
# plus a couple of metres of flight plus input buffering lands well inside this.
SEARCH_LO_MS = -50.0
# Wide enough for a Bluetooth output path, whose over-the-air delay CoreAudio does
# not report and which can reach 200 ms. The click train spaces bursts 3 s apart,
# so a window this size still cannot reach the next click.
SEARCH_HI_MS = 600.0


def find_onsets(x, rate, min_gap_s=1.0, thresh_frac=0.25):
    """Onset sample indices in a clean click track: threshold crossings of the
    envelope, thinned so each burst reports once."""
    env = envelope(x, rate, win_ms=1.0)
    thresh = env.max() * thresh_frac
    above = np.flatnonzero(env > thresh)
    if len(above) == 0:
        return np.array([], dtype=int)
    # Keep the first index of each run separated by at least min_gap_s.
    keep = [above[0]]
    for i in above[1:]:
        if i - keep[-1] > min_gap_s * rate:
            keep.append(i)
    return np.array(keep, dtype=int)


def main(session_dir):
    d = Path(session_dir)
    meta = json.loads((d / "session.json").read_text())
    mic, mic_rate = read_float_wav(d / "mic.wav")
    spk, spk_rate = read_float_wav(d / "speaker.wav")

    tick_ns = meta["mic"]["timebase_numer"] / meta["mic"]["timebase_denom"]
    stored_s = (meta["mic"]["host_ticks"] - meta["speaker"]["host_ticks"]) * tick_ns / 1e9
    print(f"session:        {meta['session_id']}")
    print(f"rates:          mic {mic_rate} Hz, speaker {spk_rate} Hz")
    print(f"stored offset:  mic starts {stored_s * 1000:+.3f} ms after speaker\n")

    # Work in seconds rather than by slicing one array to match the other, so the
    # two tracks never have to share a sample rate. A Bluetooth headset switches
    # the default input to a 16 kHz HFP mic, which is exactly the case where a
    # resample would introduce timing error into the thing being measured.
    #
    # Common timeline t=0 at the mic's first sample:
    #   speaker sample i -> i / spk_rate - stored_s
    #   mic sample     j -> j / mic_rate
    onsets = find_onsets(spk, spk_rate)
    print(f"clicks found in speaker track: {len(onsets)}")
    if len(onsets) < 2:
        print("Not enough clicks to measure. Was click.wav actually played?")
        return

    e_mic = envelope(mic, mic_rate, win_ms=1.0)

    print(f"\n{'click':>6}  {'at':>8}  {'residual':>10}  {'peak/floor':>11}  verdict")
    results = []
    for k, o in enumerate(onsets):
        # Where this click should land in the mic track if the stored offset is right.
        expect_s = o / spk_rate - stored_s
        j0 = int(round((expect_s + SEARCH_LO_MS / 1000) * mic_rate))
        j1 = int(round((expect_s + SEARCH_HI_MS / 1000) * mic_rate))
        if j0 < 0 or j1 > len(e_mic):
            continue
        seg = e_mic[j0:j1]
        # Noise floor from the 200 ms before the window, so a quiet room and a hot
        # room are judged on the same scale.
        floor_lo = max(0, j0 - int(0.2 * mic_rate))
        floor = np.median(e_mic[floor_lo:j0]) if j0 > floor_lo else seg.min()
        peak_i = int(np.argmax(seg))
        peak = seg[peak_i]
        found_s = (j0 + peak_i) / mic_rate
        residual_ms = (found_s - expect_s) * 1000.0

        # A ratio needs a floor to divide by. A Bluetooth HFP mic gates to exact
        # digital zero between sounds, which makes every ratio infinite and every
        # click look detected -- the failure that made a whole take read as valid
        # when the mic had heard nothing at all.
        if floor <= 0:
            verdict, ok = "NO NOISE FLOOR (gated mic; ratio meaningless)", False
        # A peak sitting on the window edge was not found, it was clipped: the real
        # transient is outside the search range, or there is no transient at all.
        elif peak_i <= 1 or peak_i >= len(seg) - 2:
            verdict, ok = "AT WINDOW EDGE (not a detection)", False
        elif peak / floor <= 3.0:
            verdict, ok = "NOT HEARD (below 3x floor)", False
        else:
            verdict, ok = "ok", True

        snr = peak / floor if floor > 0 else float("inf")
        results.append((k, o / spk_rate, residual_ms, snr, ok))
        print(f"{k:>6}  {o/spk_rate:>7.2f}s  {residual_ms:>9.2f}ms  {snr:>10.1f}x  {verdict}")

    heard = [r for r in results if r[4]]
    print(f"\nspeaker click peak amplitude: {np.abs(spk).max():.4f}")
    print(f"mic overall peak / rms:       {np.abs(mic).max():.5f} / {np.sqrt((mic**2).mean()):.5f}")

    if len(heard) < 2:
        print(f"\nOnly {len(heard)} of {len(results)} clicks rose above the mic noise floor.")
        print("The mic did not hear the speakers. Likely causes, in order:")
        print("  1. Output volume too low, or output routed somewhere the mic cannot hear.")
        print("  2. macOS mic mode set to Voice Isolation, which suppresses speaker echo.")
        print("  3. Headphones plugged in.")
        return

    res = np.array([r[2] for r in heard])
    t = np.array([r[1] for r in heard])

    # Drop anything far from the bulk before fitting. A single stray detection —
    # a keystroke, a notification sound after the click track ended — will
    # otherwise dominate a least-squares slope and manufacture drift that is not
    # there. MAD rather than standard deviation, since std is itself skewed by
    # the outlier it is meant to catch.
    med = np.median(res)
    mad = np.median(np.abs(res - med))
    keep = np.abs(res - med) <= max(4 * mad, 2.0)
    dropped = [(r[0], r[1], r[2]) for r, k in zip(heard, keep) if not k]
    for idx, at, r in dropped:
        print(f"\nexcluded click {idx} at {at:.2f}s ({r:+.2f} ms): {abs(r-med):.1f} ms from the median"
              f" — a stray detection, not part of the click train")

    res_f, t_f = res[keep], t[keep]
    print(f"\nresidual over {len(res_f)} clicks:")
    print(f"  median        {np.median(res_f):+.2f} ms")
    print(f"  min / max     {res_f.min():+.2f} / {res_f.max():+.2f} ms")
    print(f"  spread        {res_f.max() - res_f.min():.2f} ms")

    if np.median(res_f) < -1.0:
        print("\n  NEGATIVE residual: the mic appears to hear the click BEFORE the speaker")
        print("  emitted it, which is physically impossible. The stored timestamps are off")
        print("  by at least this much, and by more once the true acoustic delay (which must")
        print("  be positive) is added back.")

    # Drift: Theil-Sen, the median of all pairwise slopes. Robust to the odd bad
    # detection in a way least squares is not.
    if len(res_f) >= 3:
        slopes = [(res_f[j] - res_f[i]) / (t_f[j] - t_f[i])
                  for i in range(len(t_f)) for j in range(i + 1, len(t_f))
                  if t_f[j] != t_f[i]]
        slope = np.median(slopes) * 1000  # ms per 1000 s
        span = t_f.max() - t_f.min()
        print(f"\ndrift (Theil-Sen over {len(res_f)} clicks spanning {span:.1f}s):")
        print(f"  slope         {slope:+.1f} ms per 1000 s")
        # Jitter floor: the envelope smoothing window bounds how finely any
        # single click can be placed, so a slope below it is not measurable.
        floor = 1.0 / span * 1000
        if abs(slope) < floor:
            print(f"  -> no measurable drift (below the {floor:.1f} ms/1000s resolution of this take)")
        else:
            print(f"  -> possible drift; confirm over a ~10 min recording before believing it")


if __name__ == "__main__":
    main(sys.argv[1])
