"""What CoreAudio reports for the default input and output paths.

Standalone on purpose: the same numbers are printed by the recorder's
MEETHOOK_TIMING_DEBUG output, but reading them here needs no recording session
and no audio permission, so they can be checked against a click measurement
after the fact.
"""
import ctypes, ctypes.util

ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))

def fourcc(s): return int.from_bytes(s.encode(), "big")

class Addr(ctypes.Structure):
    _fields_ = [("sel", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("elem", ctypes.c_uint32)]

GLOBAL, INPUT, OUTPUT = fourcc("glob"), fourcc("inpt"), fourcc("outp")
SYSTEM = 1

def get_u32(obj, sel, scope):
    a = Addr(sel, scope, 0)
    val, size = ctypes.c_uint32(0), ctypes.c_uint32(4)
    st = ca.AudioObjectGetPropertyData(ctypes.c_uint32(obj), ctypes.byref(a),
                                       0, None, ctypes.byref(size), ctypes.byref(val))
    return val.value if st == 0 else None

def get_f64(obj, sel, scope):
    a = Addr(sel, scope, 0)
    val, size = ctypes.c_double(0), ctypes.c_uint32(8)
    st = ca.AudioObjectGetPropertyData(ctypes.c_uint32(obj), ctypes.byref(a),
                                       0, None, ctypes.byref(size), ctypes.byref(val))
    return val.value if st == 0 else None

def first_stream(dev, scope):
    a = Addr(fourcc("stm#"), scope, 0)
    val, size = ctypes.c_uint32(0), ctypes.c_uint32(4)
    ca.AudioObjectGetPropertyData(ctypes.c_uint32(dev), ctypes.byref(a),
                                  0, None, ctypes.byref(size), ctypes.byref(val))
    return val.value or None

for label, sel, scope in [("input ", "dIn ", INPUT), ("output", "dOut", OUTPUT)]:
    dev = get_u32(SYSTEM, fourcc(sel), GLOBAL)
    if not dev:
        print(f"{label}: no default device"); continue
    rate = get_f64(dev, fourcc("nsrt"), GLOBAL) or 48000.0
    d = get_u32(dev, fourcc("ltnc"), scope) or 0
    s = get_u32(dev, fourcc("saft"), scope) or 0
    st = first_stream(dev, scope)
    sl = (get_u32(st, fourcc("ltnc"), GLOBAL) or 0) if st else 0
    total = d + s + sl
    print(f"{label} id={dev} rate={rate:.0f}  device={d} stream={sl} safety={s} frames"
          f"  total={total} frames = {total / rate * 1000:.3f} ms")
