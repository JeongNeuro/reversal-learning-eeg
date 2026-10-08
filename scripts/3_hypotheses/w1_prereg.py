# -*- coding: utf-8 -*-
"""w1_prereg.py - the preregistered W1. Aperiodic exponent of the post-reversal and stable windows

w1.py produces only the front and post clusters. The value for the registered primary cluster
(parieto-occipital) was not in the repository and lived only in a working file. The situation
is the same as for W2, so it is moved here.

**Sign convention** - the coefficient is always `post-reversal window (post) - stable window
(stable)`. A different reference window flips the sign. The preregistration predicted that
the exponent just after a reversal would be **lower** (negative).

The model is exactly as registered.
    exponent ~ window + reward density + window duration + epoch count
    participant random intercept + participant-by-pair random intercept

Only pairs that have both windows are used. Including a pair with only one window mixes the
window effect with between-participant differences.

Input   outputs/W1_prereg_windows.csv  (participant x pair x window x cluster)
Output  outputs/w1_prereg.csv  .txt
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

SRC = os.path.join(str(OUT), 'W1_prereg_windows.csv')
BASE = 'stable'                   # reference window - the coefficient is post minus stable
CL = [('po4', 'parieto-occipital (registered primary)'), ('front', 'frontal'),
      ('temp', 'temporo-parietal'), ('glob', 'global')]
COV = ['reward_density', 'window_duration_s', 'n_epochs']

LOG = []


def w(t=''):
    LOG.append(t)
    print(t)


if not os.path.exists(SRC):
    raise SystemExit('no window data: %s' % SRC)

D = pd.read_csv(SRC, encoding='utf-8-sig')
D['pid'] = D['pid'].astype(str).str.zfill(3)

w('=' * 92)
w(' W1 - aperiodic exponent, post-reversal window minus stable window (preregistered)')
w('=' * 92)
w('  model  exponent ~ window + %s' % ' + '.join(COV))
w('         participant random intercept + participant-by-pair random intercept')
w('  sign   positive = the post-reversal window is higher. The preregistered prediction was negative (lower).')
w('  reference window = %s - only pairs with both windows are used' % BASE)
w()

rows = []
for cl, nm in CL:
    s = D[D['cluster'] == cl].dropna(subset=['exponent']).copy()
    if s.empty:
        continue
    full = s.groupby(['pid', 'pair'])['window'].nunique() >= 2
    keep = set(full[full].index)
    s = s[[(a, b) in keep for a, b in zip(s['pid'], s['pair'])]].copy()
    if s['pid'].nunique() < 4:
        w('  %-38s below the floor (%d participants)' % (nm, s['pid'].nunique()))
        continue
    s['ppair'] = s['pid'] + '_' + s['pair'].astype(str)
    have = [c for c in COV if c in s.columns and s[c].notna().any()]
    f = 'exponent ~ C(window, Treatment("%s"))' % BASE
    if have:
        f += ' + ' + ' + '.join(have)
    m = smf.mixedlm(f, s, groups=s['pid'], re_formula='1',
                    vc_formula={'ppair': '0+C(ppair)'}).fit(reml=True,
                                                            method='lbfgs')
    k = [q for q in m.params.index if q.startswith('C(window')][0]
    est, se, p = float(m.params[k]), float(m.bse[k]), float(m.pvalues[k])
    lo, hi = est - 1.96 * se, est + 1.96 * se
    st = '**' if p < .01 else ('*' if p < .05 else '')
    w('  %-38s %+.4f  [%+.4f, %+.4f]  Wald P = %.4f %-2s  (%d pairs, %d participants)'
      % (nm, est, lo, hi, p, st, len(keep), s['pid'].nunique()))
    rows.append(dict(cluster=nm, n_participants=int(s['pid'].nunique()),
                     n_pairs=len(keep), estimate=round(est, 4),
                     ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                     p_wald=round(p, 4), star=st,
                     covariates=' '.join(have)))
w()
w('  Note. The preregistration predicted that the aperiodic exponent would fall just after a')
w('        reversal. The observed sign is positive in all four clusters, the opposite of the prediction, and the interval includes 0.')

with io.open(os.path.join(str(OUT), 'w1_prereg.csv'), 'w',
             encoding='utf-8-sig', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
io.open(os.path.join(str(OUT), 'w1_prereg.txt'), 'w',
        encoding='utf-8').write('\n'.join(LOG))
print('\nwritten -> outputs/w1_prereg.csv  .txt')
