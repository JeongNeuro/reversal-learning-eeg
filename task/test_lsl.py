"""
test_lsl.py
A minimal test of LSL marker output alone, without PsychoPy.

Purpose
----
Separate whether the symptom "PsychoPy dies when Connect is pressed in EmotivPRO" is a
problem with LSL itself or with its interaction with PsychoPy.

  - this script survives  -> LSL is fine. Look at the PsychoPy side.
  - this script dies too  -> a pylsl / liblsl / firewall problem.

Running it
----
At a command prompt:
    "C:\\Program Files\\PsychoPy\\python.exe" test_lsl.py

(Use python.exe, not pythonw.exe. The console output is the point.)

How to use
------
1. Start Record in EmotivPRO.
2. Run this script.
3. In EmotivPRO > Settings > Lab Streaming Layer > Inlet, select
   'PRL_Markers_TEST' and press Connect.
4. Watch whether the marker numbers keep climbing in the console and whether markers appear in EmotivPRO.
5. Ctrl+C to quit.
"""

import sys
import time

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import pylsl
except ImportError:
    print("[FAIL] pylsl not installed.")
    print("       PsychoPy > Tools > Plugin/packages manager > Open PIP terminal")
    print("       then type:  install pylsl")
    raise SystemExit(1)

STREAM_NAME = "PRL_Markers_TEST"

print("=" * 60)
print(" LSL marker test")
print("=" * 60)
print(f"pylsl version   : {getattr(pylsl, '__version__', 'unknown')}")
try:
    print(f"liblsl version  : {pylsl.library_version()}")
except Exception as e:
    print(f"liblsl version  : (query failed: {e})")
print()

# --- create the outlet (EmotivPRO 3-channel form) ---
info = pylsl.StreamInfo(
    name=STREAM_NAME,
    type="Markers",
    channel_count=3,
    nominal_srate=pylsl.IRREGULAR_RATE,
    channel_format=pylsl.cf_double64,
    source_id="prl_lsl_test",
)
chns = info.desc().append_child("channels")
for label in ("MarkerTime", "MarkerValue", "CurrentTime"):
    chns.append_child("channel").append_child_value("label", label)

outlet = pylsl.StreamOutlet(info)   # info stays referenced below

# monotonic <-> epoch offset
offs = []
for _ in range(25):
    l0 = pylsl.local_clock()
    e = time.time()
    l1 = pylsl.local_clock()
    offs.append((l0 + l1) / 2.0 - e)
offs.sort()
lsl_offset = offs[len(offs) // 2]

print(f"Outlet created  : '{STREAM_NAME}' (3 channels, double64)")
print(f"LSL-epoch offset: {lsl_offset:.6f} s")
print()
print("Now in EmotivPRO:")
print("  1. Start Record")
print("  2. Settings > Lab Streaming Layer > Inlet")
print(f"  3. Select '{STREAM_NAME}' and click Connect")
print()
print("Pushing one marker per second. Ctrl+C to stop.")
print("-" * 60)

n = 0
seen_consumer = False
try:
    while True:
        n += 1
        now = time.time()
        marker_time = now                 # a test, so event time = current time
        value = float(100 + (n % 5))      # cycles 100-104
        outlet.push_sample(
            [marker_time, value, time.time()],
            timestamp=marker_time + lsl_offset,
            pushthrough=True)

        has = outlet.have_consumers()
        if has and not seen_consumer:
            print(">>> CONSUMER CONNECTED (EmotivPRO subscribed)")
            seen_consumer = True
        elif not has and seen_consumer:
            print(">>> consumer disconnected")
            seen_consumer = False

        print(f"  #{n:3d}  value={int(value)}  consumers={'yes' if has else 'no'}")
        time.sleep(1.0)

except KeyboardInterrupt:
    print("-" * 60)
    print(f"Stopped. {n} markers pushed.")
    print("If this script survived the Connect click, LSL itself is fine.")
