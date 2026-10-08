# -*- coding: utf-8 -*-
"""band_window_grid_prereg.py - the band x time-window grid, with the **preregistered wavelet**

band_window_grid.py reads tfr.json, and that file was built with 20 log-spaced bins and
n_cycles = 5, which differs from preregistration 29.4 (3-30 Hz in **1 Hz steps**,
n_cycles = f / 2). Printed in the same section beside the preregistered-wavelet values, the
two read as different results.

This script reads the trial-level grid that w2_prereg_trials.py produced with the registered
wavelet and builds the same table. The baseline is the stimulus-locked -300 to -100 ms of
registration 30, and epoch rejection is the per-cluster judgement of registration 28.

**Check** - the theta 4-8 Hz x 200-500 ms cell is by definition the same quantity as the
frontal 2-level contrast of w2_prereg_fit.py.

Input   outputs/W2_prereg_grid_trials.csv   (produced by w2_prereg_trials.py)
Output  outputs/band_window_grid_prereg.csv  .txt
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import OUT                                       # noqa: E402

import csv                                                  # noqa: E402
import io                                                   # noqa: E402

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

SEED = 1
NB = 10000
MINEP = 10        # epochs that must survive in each condition (registration 28)

SRC = sys.argv[1] if len(sys.argv) > 1 else str(
    OUT / 'W2_prereg_grid_trials.csv')
if not os.path.exists(SRC):
    raise SystemExit(f'no grid data: {SRC}\n'
                     f'  run w2_prereg_trials.py first.')

L = []


def w(t=''):
    L.append(t)
    print(t)


def boot(x, nb=NB, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    o = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)]
    return np.percentile(o, [2.5, 97.5])


T = pd.read_csv(SRC, encoding='utf-8')
T['pid'] = T['pid'].astype(str).str.zfill(3)
T['cond'] = np.where(T['rewarded'] == 1, 'R', 'U')

w('=' * 86)
w(' feedback-locked frontal - unrewarded minus rewarded (dB) - **preregistered wavelet**')
w('=' * 86)
w('  wavelet    3-30 Hz in 1 Hz steps, n_cycles = f / 2 (registration 29.4)')
w('  baseline   stimulus-locked -300 to -100 ms (registration 30)')
w('  rejection  channels contributing to the frontal cluster, 150 uV (registration 28)')
w('  included   participants with at least %d epochs in each condition' % MINEP)
w('  interval   participant bootstrap, %d resamples, seed %d' % (NB, SEED))
w('  sign       positive = larger when unrewarded (the direction the registration predicts)')
w()

rows = []
bands = list(dict.fromkeys(T['band']))
wins = list(dict.fromkeys(T['window']))
for b in bands:
    w('  [%s]' % b)
    w('    %-16s %10s %22s %10s %6s' %
      ('time window', 'diff (dB)', '95% CI', 'same direction', 'n'))
    for wn in wins:
        d = T[(T['band'] == b) & (T['window'] == wn)]
        cnt = d.pivot_table(index='pid', columns='cond', values='dB',
                            aggfunc='size').fillna(0)
        if not {'R', 'U'} <= set(cnt.columns):
            continue
        ok = [p for p in cnt.index
              if cnt.loc[p, 'R'] >= MINEP and cnt.loc[p, 'U'] >= MINEP]
        if len(ok) < 4:
            w('    %-16s below the floor (n = %d)' % (wn, len(ok)))
            continue
        m = d[d['pid'].isin(ok)].pivot_table(index='pid', columns='cond',
                                             values='dB', aggfunc='mean')
        diff = (m['U'] - m['R']).values
        lo, hi = boot(diff)
        same = int((diff > 0).sum())
        w('    %-16s %+10.3f   [%+.3f, %+.3f] %6d/%d %6d'
          % (wn, diff.mean(), lo, hi, same, len(diff), len(ok)))
        rows.append(dict(band=b, window=wn, n=len(ok),
                         diff_dB=round(float(diff.mean()), 4),
                         ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                         same_direction=same))
    w()

if rows:
    a = np.array([abs(r['diff_dB']) for r in rows])
    cross = sum(1 for r in rows if r['ci_lo'] < 0 < r['ci_hi'])
    w('  %d cells - %d cells whose interval includes 0' % (len(rows), cross))
    w('  absolute difference  median %.3f dB - maximum %.3f dB' % (np.median(a), a.max()))
    w()
    w('  Note. No cell is called significant or non-significant (registration 34). An interval')
    w('        including 0 means the size and direction of the difference cannot be pinned down')
    w('        in this sample; it does not mean there is no difference.')

    dst = str(OUT / 'band_window_grid_prereg.csv')
    with io.open(dst, 'w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    io.open(str(OUT / 'band_window_grid_prereg.txt'), 'w',
            encoding='utf-8').write('\n'.join(L))
    print('\nwritten ->', dst)
