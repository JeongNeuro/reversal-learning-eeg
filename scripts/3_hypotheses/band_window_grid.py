# -*- coding: utf-8 -*-
"""band_window_grid.py - feedback-locked unrewarded - rewarded, band x time-window grid

This replaces the place where manuscript 3.5 wrote "there was no difference in five
combinations of time window and band, including frontal alpha (n = 10, minimum P = .49)".

That sentence makes a threshold judgement from a P value in the body text, which does not fit
the reporting scheme of preregistration 34 (estimates and intervals, no threshold judgement).
The same data are used to build a band x time-window grid, with an estimate and a participant bootstrap interval in every cell.

The baseline is the stimulus-locked -300 to -100 ms of preregistration 30. The U and R of
tfr.json were built with a feedback-locked baseline and are therefore not used; the registered
baseline is rebuilt by subtracting stimbasedb_* from rawdb_* (both take the log per channel
before averaging, so they line up directly).

Input   data/tfr.json   (produced by tfr.py)
Output  outputs/band_window_grid.csv  outputs/band_window_grid.txt
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import DATA, OUT                                 # noqa: E402

import csv                                                  # noqa: E402
import io                                                   # noqa: E402
import json                                                 # noqa: E402

import numpy as np                                          # noqa: E402

SEED = 1
NB = 10000

# (name, low frequency, high frequency)
BANDS = [('theta 4-8 Hz', 4.0, 8.0),
         ('alpha 8-13 Hz (reg. 18.4)', 8.0, 13.0),
         ('alpha 7-14 Hz (manuscript definition)', 7.0, 14.0),
         ('beta 13-30 Hz', 13.0, 30.0)]
# (name, start s, end s)
WINS = [('200-500 ms (reg. 18.5)', 0.200, 0.500),
        ('300–600 ms', 0.300, 0.600),
        ('0–500 ms', 0.000, 0.500),
        ('500–1000 ms', 0.500, 1.000)]

L = []


def w(t=''):
    L.append(t)
    print(t)


def boot(x, nb=NB, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    o = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)]
    return np.percentile(o, [2.5, 97.5])


TF = json.load(io.open(os.path.join(str(DATA), 'tfr.json'), encoding='utf-8'))
pids = sorted(TF)

rows = []
w('=' * 86)
w(' feedback-locked frontal theta/alpha/beta - unrewarded minus rewarded (dB)')
w('=' * 86)
w('  baseline  stimulus-locked -300 to -100 ms (preregistration 30)')
w('  sample    %d participants: %s' % (len(pids), ' '.join(pids)))
w('  interval  participant bootstrap, %d resamples, seed %d' % (NB, SEED))
w('  sign      positive = larger when unrewarded (the direction the preregistration predicts)')
w()

for bname, flo, fhi in BANDS:
    w('  [%s]' % bname)
    w('    %-30s %10s %22s %14s' % ('time window', 'diff (dB)', '95% CI', 'same direction'))
    for wname, tlo, thi in WINS:
        d = []
        for p in pids:
            v = TF[p]
            t = np.asarray(v['t'], float)
            f = np.asarray(v['f'], float)
            fm = (f >= flo) & (f <= fhi)
            tm = (t >= tlo) & (t <= thi)
            if not fm.any() or not tm.any():
                continue
            out = {}
            for c in ('U', 'R'):
                raw = np.asarray(v[f'rawdb_{c}'], float)        # (f, t)
                base = np.asarray(v[f'stimbasedb_{c}'], float)  # (f,)
                out[c] = (raw - base[:, None])[np.ix_(fm, tm)].mean()
            d.append(out['U'] - out['R'])
        d = np.asarray(d, float)
        lo, hi = boot(d)
        same = int((d > 0).sum())
        w('    %-26s %+10.3f   [%+.3f, %+.3f] %6d/%d'
          % (wname, d.mean(), lo, hi, same, len(d)))
        rows.append(dict(band=bname, window=wname, n=len(d),
                         diff_dB=round(float(d.mean()), 4),
                         ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                         same_direction=same))
    w()

a = np.array([abs(r['diff_dB']) for r in rows])
cross = sum(1 for r in rows if r['ci_lo'] < 0 < r['ci_hi'])
w('  %d cells - %d cells whose interval includes 0' % (len(rows), cross))
w('  absolute difference  median %.3f dB - maximum %.3f dB' % (np.median(a), a.max()))
w()
w('  Note. No cell is called significant or non-significant (preregistration 34). An interval')
w('        including 0 means the size and direction of the difference cannot be pinned down in')
w('        this sample; it does not mean there is no difference.')

dst = os.path.join(str(OUT), 'band_window_grid.csv')
with io.open(dst, 'w', encoding='utf-8-sig', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
io.open(os.path.join(str(OUT), 'band_window_grid.txt'), 'w',
        encoding='utf-8').write('\n'.join(L))
print('\nwritten ->', dst)
