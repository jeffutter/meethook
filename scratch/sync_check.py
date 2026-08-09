"""Measure the residual mic-vs-speaker offset in a meethook session.

Applies the session.json host-tick offset to align the two tracks, then
cross-correlates them to find whatever lag is left over. That leftover is what
AC #6 asks about: it is not pure clock error, it also contains output
buffering, speaker-to-mic acoustic flight time, and input buffering.

Usage: python3 sync_check.py <session-dir>
"""

import json
import sys
import wave

import numpy as np


def read_float_wav(path):
    """Minimal RIFF walker. Python's `wave` module refuses WAVE_FORMAT_EXTENSIBLE
    with the IEEE-float subformat, which is what hound writes — the files are
    valid (afinfo/afplay read them), the stdlib is just narrow."""
    import struct

    data = path.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", f"{path}: not RIFF/WAVE"
    pos, fmt, payload = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", body, 0)  # tag, ch, rate, bps, align, bits
        elif cid == b"data":
            payload = body
            # Trust the header's declared length, but report a mismatch: a
            # truncated file is exactly what AC #9 is about.
            if len(body) != size:
                print(f"WARNING {path.name}: data chunk declares {size} bytes, file holds {len(body)}")
        pos += 8 + size + (size & 1)
    assert fmt and payload is not None, f"{path}: missing fmt or data chunk"
    _, channels, rate, _, _, bits = fmt
    assert channels == 1, f"{path}: expected mono, got {channels} ch"
    assert bits == 32, f"{path}: expected 32-bit, got {bits}"
    return np.frombuffer(payload, dtype="<f4").astype(np.float64), rate


def envelope(x, rate, win_ms=2.0):
    """Rectified, smoothed amplitude envelope — robust to the mic and speaker
    signals differing in spectrum, which raw-sample correlation is not.

    Boxcar smoothing via a cumulative sum: `np.convolve` here would be an
    O(n*win) direct convolution over ~1e6 samples."""
    win = max(1, int(rate * win_ms / 1000.0))
    c = np.cumsum(np.concatenate([[0.0], np.abs(x)]))
    smoothed = (c[win:] - c[:-win]) / win
    # Re-centre so the envelope stays sample-aligned with the input.
    pad = len(x) - len(smoothed)
    return np.concatenate([np.zeros(pad // 2), smoothed, np.zeros(pad - pad // 2)])


def lag_by_correlation(a, b, max_lag):
    """Lag (in samples) of `a` relative to `b`, searched over +/- max_lag, plus
    a 0..1 normalized correlation at that lag.

    FFT-based on purpose: `np.correlate(mode="full")` is a direct O(n^2)
    convolution, which on a 20-minute 48 kHz track is ~1e12 operations."""
    n = len(a)
    size = 1 << int(np.ceil(np.log2(2 * n - 1)))
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(b, size)), size)
    # corr[k] is the correlation at lag +k; negative lags live at the tail.
    cap = min(max_lag, n - 1)
    window = np.concatenate([corr[size - cap :], corr[: cap + 1]])
    best = int(np.argmax(window)) - cap

    norm = np.sqrt((a**2).sum() * (b**2).sum())
    return best, (window.max() / norm if norm else 0.0)


def main(session_dir):
    from pathlib import Path

    d = Path(session_dir)
    meta = json.loads((d / "session.json").read_text())
    mic, mic_rate = read_float_wav(d / "mic.wav")
    spk, spk_rate = read_float_wav(d / "speaker.wav")

    numer = meta["mic"]["timebase_numer"]
    denom = meta["mic"]["timebase_denom"]
    tick_ns = numer / denom
    delta_ticks = meta["mic"]["host_ticks"] - meta["speaker"]["host_ticks"]
    stored_offset_ms = delta_ticks * tick_ns / 1e6

    print(f"session:            {meta['session_id']}")
    print(f"mic:                {len(mic)/mic_rate:.3f} s @ {mic_rate} Hz")
    print(f"speaker:            {len(spk)/spk_rate:.3f} s @ {spk_rate} Hz")
    print(f"mic peak / rms:     {np.abs(mic).max():.5f} / {np.sqrt((mic**2).mean()):.5f}")
    print(f"speaker peak / rms: {np.abs(spk).max():.5f} / {np.sqrt((spk**2).mean()):.5f}")
    print()
    print(f"stored offset:      mic starts {stored_offset_ms:+.3f} ms after speaker")

    if np.abs(spk).max() < 1e-4:
        print("\nspeaker track is silent — no shared event to align against.")
        return
    if np.abs(mic).max() < 1e-4:
        print("\nmic track is silent — no shared event to align against.")
        return

    assert mic_rate == spk_rate, "rates differ; this script does not resample"
    rate = mic_rate

    # Align to a common timeline: drop the leading part of the speaker track
    # that predates the mic's first sample.
    lead = int(round(stored_offset_ms / 1000.0 * rate))
    spk_al = spk[lead:] if lead > 0 else spk
    mic_al = mic[-lead:] if lead < 0 else mic
    n = min(len(spk_al), len(mic_al))
    spk_al, mic_al = spk_al[:n], mic_al[:n]

    e_spk = envelope(spk_al, rate)
    e_mic = envelope(mic_al, rate)
    e_spk -= e_spk.mean()
    e_mic -= e_mic.mean()

    # Search +/- 500 ms of residual lag.
    max_lag = int(0.5 * rate)
    best, peak_corr = lag_by_correlation(e_mic, e_spk, max_lag)
    residual_ms = best / rate * 1000.0

    print(f"residual lag:       mic lags speaker by {residual_ms:+.2f} ms (after applying stored offset)")
    print(f"correlation:        {peak_corr:.4f}  (below ~0.1 means the tracks share no real content;")
    print("                    treat the lag above as meaningless in that case)")

    # Drift: measure the residual independently in the first and last third. If
    # the two devices run on independent clocks, these diverge — and no single
    # first-sample offset can align the tracks, which is a design problem rather
    # than an implementation bug.
    third = n // 3
    if third > rate:  # need at least a second per segment to be meaningful
        print("\ndrift check (these two should agree to well under a millisecond):")
        lags = {}
        for name, sl in (("first third", slice(0, third)), ("last third", slice(2 * third, n))):
            lag, corr = lag_by_correlation(e_mic[sl], e_spk[sl], max_lag)
            lags[name] = lag / rate * 1000.0
            print(f"  {name:<12}      {lags[name]:+.2f} ms   (corr {corr:.4f})")
        spread = lags["last third"] - lags["first third"]
        elapsed_s = (n - third) / rate
        print(f"  spread            {spread:+.2f} ms over {elapsed_s:.1f} s"
              f"  ({spread / elapsed_s * 1000:+.1f} ms per 1000 s if it is real drift)")


if __name__ == "__main__":
    main(sys.argv[1])
