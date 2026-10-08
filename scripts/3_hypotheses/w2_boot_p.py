# -*- coding: utf-8 -*-
"""w2_boot_p.py - bootstrap P for the W2 3-level contrasts (the same method as the interval)

The manuscript reports the 3-level contrasts with a **participant cluster bootstrap interval**
(unrewarded-correct − rewarded-correct = -0.59 [-1.05, -0.07]). The Wald test of the mixed
model, however, gives P = .054 for the same estimate, and its Wald interval [-1.19, +0.01]
includes 0. The interval and the P disagree because they are different methods.

Here **the P is produced by the same method as the interval.** From the bootstrap distribution,
  P = 2 × min( Pr(θ* ≤ 0), Pr(θ* ≥ 0) )
gives the two-sided P (truncated at 1). When the interval excludes 0, this P is below .05.

The third contrast the manuscript reports (unrewarded-incorrect − unrewarded-correct) is
produced as well. It was missing from the earlier P-value table.

Input   outputs/W2_prereg_trials.csv
Output  outputs/w2_boot_p.csv  .txt
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import OUT                                       # noqa: E402

import csv                                                  # noqa: E402
import io                                                   # noqa: E402
import warnings                                             # noqa: E402

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
import statsmodels.formula.api as smf                       # noqa: E402

warnings.filterwarnings('ignore')

SEED = 1
NB = 10000
MINEP = 10
CL = [('front', 'frontal (registered primary)'), ('po4', 'parieto-occipital'),
      ('temp', 'temporo-parietal'), ('glob', 'global')]
FORM = 'theta_dB ~ C(cond3, Treatment("R_cor"))'

LOG = []


def w(t=''):
    LOG.append(t)
    print(t, flush=True)


def fit3(frame):
    m = smf.mixedlm(FORM, frame, groups=frame['pid']).fit(reml=True,
                                                          method='lbfgs')
    return {k.split('T.')[-1].rstrip(']'): float(m.params[k])
            for k in m.params.index if k.startswith('C(cond3')}, m


def boot_p(draws):
    """Two-sided P from the bootstrap distribution - the same method as the interval."""
    d = np.asarray(draws, float)
    n = len(d)
    lo = (np.sum(d <= 0) + 1) / (n + 1)
    hi = (np.sum(d >= 0) + 1) / (n + 1)
    return min(1.0, 2 * min(lo, hi))


T = pd.read_csv(os.path.join(str(OUT), 'W2_prereg_trials.csv'),
                encoding='utf-8')
T['pid'] = T['pid'].astype(str).str.zfill(3)

w('=' * 94)
w(' W2 3-level contrasts - P by the same method as the bootstrap interval')
w('=' * 94)
w('  participant cluster bootstrap, %d resamples, seed %d, at least %d epochs per cell'
  % (NB, SEED, MINEP))
w('  P = 2 x min(Pr(theta* <= 0), Pr(theta* >= 0))')
w()

rows = []
for cl, nm in CL:
    d = T[T['cluster'] == cl]
    cnt = d.pivot_table(index='pid', columns='cond3', values='theta_dB',
                        aggfunc='size').fillna(0)
    need = [c for c in ('R_cor', 'U_cor', 'U_inc') if c in cnt.columns]
    if len(need) < 3:
        continue
    ok = [p for p in cnt.index if all(cnt.loc[p, c] >= MINEP for c in need)]
    if len(ok) < 6:
        w('  [%s] below the floor (n = %d)' % (nm, len(ok)))
        continue
    d3 = d[d['pid'].isin(ok)]
    co, _ = fit3(d3)
    rng = np.random.default_rng(SEED)
    idx = {p: d3[d3['pid'] == p] for p in ok}
    B = {'U_cor': [], 'U_inc': []}
    drop = 0
    for _ in range(NB):
        pick = rng.integers(0, len(ok), len(ok))
        r = pd.concat([idx[ok[i]].assign(pid='%s_%d' % (ok[i], j))
                       for j, i in enumerate(pick)], ignore_index=True)
        if r['cond3'].nunique() < 3:
            drop += 1
            continue
        try:
            c, _ = fit3(r)
        except Exception:
            drop += 1
            continue
        if c.get('U_cor') is None or c.get('U_inc') is None:
            drop += 1
            continue
        B['U_cor'].append(c['U_cor'])
        B['U_inc'].append(c['U_inc'])
    uc = np.asarray(B['U_cor'])
    ui = np.asarray(B['U_inc'])
    w('  [%s]  n = %d - valid resamples %d' % (nm, len(ok), len(uc)))
    for lab, est, draws in (
            ('unrewarded-correct − rewarded-correct', co['U_cor'], uc),
            ('unrewarded-incorrect − rewarded-correct', co['U_inc'], ui),
            ('unrewarded-incorrect − unrewarded-correct', co['U_inc'] - co['U_cor'], ui - uc)):
        ci = np.percentile(draws, [2.5, 97.5])
        p = boot_p(draws)
        st = '**' if p < .01 else ('*' if p < .05 else '')
        w('    %-44s %+7.3f dB  [%+.3f, %+.3f]  bootstrap P = %.4f %s'
          % (lab, est, ci[0], ci[1], p, st))
        rows.append(dict(cluster=nm, contrast=lab, n=len(ok),
                         estimate=round(float(est), 4),
                         ci_lo=round(float(ci[0]), 4),
                         ci_hi=round(float(ci[1]), 4),
                         p_bootstrap=round(float(p), 4), star=st))
    if drop:
        w('    %d resamples discarded' % drop)
    w()

w('  Note. When the interval excludes 0 this P is below .05 - they come from the same distribution.')
w('        The Wald P of the mixed model is a different method and does not count participants as the independent unit.')

with io.open(os.path.join(str(OUT), 'w2_boot_p.csv'), 'w',
             encoding='utf-8-sig', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
io.open(os.path.join(str(OUT), 'w2_boot_p.txt'), 'w',
        encoding='utf-8').write('\n'.join(LOG))
print('\nwritten -> outputs/w2_boot_p.csv  .txt')
