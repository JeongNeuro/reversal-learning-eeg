# -*- coding: utf-8 -*-
"""clock_offset.py - per-session clock offset drift (preregistration 25)

The task program measures the offset between the LSL clock and the system clock at the start
and at the end of a session and writes the difference to meta.json as epoch_offset_drift_ms.
Preregistration 25 settled that a session whose absolute value exceeds 20 ms is excluded from
the feedback-locked analysis (W2).

The median and range quoted in the 'Recording' part of manuscript 2.2 come from here.

Input   data/raw/*_meta.json
Output  outputs/clock_offset.csv and a summary on standard output
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import RAW, OUT                                  # noqa: E402

import csv                                                  # noqa: E402
import glob                                                 # noqa: E402
import io                                                   # noqa: E402
import json                                                 # noqa: E402

import numpy as np                                          # noqa: E402

LIMIT_MS = 20.0        # preregistration 25

rows = []
for p in sorted(glob.glob(os.path.join(str(RAW), '**', '*_meta.json'),
                          recursive=True)):
    d = json.load(io.open(p, encoding='utf-8'))
    rows.append(dict(
        file=os.path.basename(p),
        pid=str(d.get('note') or '').zfill(3),
        timestamp=d.get('timestamp'),
        offset_start=d.get('epoch_offset_start'),
        offset_end=d.get('epoch_offset_end'),
        drift_ms=d.get('epoch_offset_drift_ms'),
        pilot=d.get('pilot'),
    ))

if not rows:
    raise SystemExit(f'no meta.json found: {RAW}')

print('%-6s %-18s %14s' % ('id', 'recorded at', 'drift (ms)'))
for r in rows:
    v = r['drift_ms']
    print('%-6s %-18s %14s' % (r['pid'], r['timestamp'] or '',
                               '%+.4f' % v if v is not None else 'none'))

v = np.array([r['drift_ms'] for r in rows if r['drift_ms'] is not None],
             dtype=float)
over = int((np.abs(v) > LIMIT_MS).sum())
print()
print('%d sessions (%d participants)' % (len(v), len({r['pid'] for r in rows})))
print('median %+.4f ms - range %+.4f to %+.4f ms' % (np.median(v), v.min(), v.max()))
print('largest absolute value %.4f ms' % np.abs(v).max())
print('%d sessions with |drift| > %.0f ms - excluded under preregistration 25' % (over, LIMIT_MS))
print()
print('as quoted in manuscript 2.2: median %+.2f ms - range %+.2f to %+.2f ms'
      % (np.median(v), v.min(), v.max()))

dst = os.path.join(str(OUT), 'clock_offset.csv')
with io.open(dst, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print('\nwritten ->', dst)
