# -*- coding: utf-8 -*-
"""topo_cluster_perm.py - cluster-based permutation correction for the topographies of Figure 5a

The topographies of Figure 5a are the paired t of per-channel relative band power. Putting an
asterisk on them requires correcting for multiple comparisons over 14 electrodes. The
correction used here is cluster-based permutation (Maris & Oostenveld 2007).

Procedure
  1. Build the participant x channel matrix of differences (task - rest), leaving gaps as they are.
  2. Compute the paired t per channel and select the channels where |t| exceeds that channel's
     critical value (two-sided .05, df = n-1).
  3. Group adjacent selected channels into clusters and measure the cluster mass = sum|t|.
  4. Flip the sign of each participant's differences at random (the exact null distribution of
     a paired design), repeat the same procedure 10,000 times, and take the distribution of the maximum cluster mass.
  5. The corrected P of an observed cluster = the proportion of that distribution at or above the observed mass.

Sign flipping is done **per participant** and applied identically to all of that participant's
channels. The samples differ by channel (6-10 participants), so this is what preserves the spatial structure.

Adjacency is set by the distance between electrode coordinates (ADJ_R). With a 14-channel
layout and no midline, each electrode has few neighbours, which is conservative in that clusters do not grow easily.

Input   data/psd_all.json
Output  outputs/topo_cluster_perm.csv  .txt
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import DATA, OUT                                 # noqa: E402

import csv                                                  # noqa: E402
import io                                                   # noqa: E402
import json                                                 # noqa: E402
import warnings                                             # noqa: E402

import numpy as np                                          # noqa: E402
from scipy import stats                                     # noqa: E402

warnings.filterwarnings('ignore')

SEED = 1
NPERM = 10000
ADJ_R = 0.75                      # distance counted as adjacent
MIN_N = 3                         # minimum participants for a channel to yield a t

CH = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
      'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
POS = {'AF3': (-.30, .66), 'AF4': (.30, .66), 'F7': (-.68, .40),
       'F3': (-.36, .42), 'F4': (.36, .42), 'F8': (.68, .40),
       'FC5': (-.55, .16), 'FC6': (.55, .16), 'T7': (-.80, .00),
       'T8': (.80, .00), 'P7': (-.60, -.48), 'P8': (.60, -.48),
       'O1': (-.27, -.76), 'O2': (.27, -.76)}
BANDS = [('Delta', 2, 4), ('Theta', 4, 8), ('Alpha', 8, 13),
         ('Beta', 13, 30), ('Gamma', 30, 45)]

LOG = []


def w(t=''):
    LOG.append(t)
    print(t)


PS = json.load(io.open(os.path.join(str(DATA), 'psd_all.json'),
                       encoding='utf-8'))


def rel(pid, cond, lo, hi):
    d = PS.get(pid)
    if not d or d.get(cond) is None:
        return None
    f = np.asarray(d['f'], float)
    A = np.asarray(d[cond], float)
    den = (f >= 1) & (f <= 45)
    bb = (f >= lo) & (f < hi)
    out = {}
    for i, c in enumerate(d['ch']):
        if i < A.shape[0] and not np.all(np.isnan(A[i])):
            out[c] = A[i][bb].sum() / A[i][den].sum()
    return out


# adjacency list
NB = {c: [d for d in CH if d != c
          and np.hypot(POS[c][0] - POS[d][0], POS[c][1] - POS[d][1]) < ADJ_R]
      for c in CH}


def chan_t(D):
    """participant x channel difference matrix -> per-channel t and critical value."""
    t = np.full(len(CH), np.nan)
    crit = np.full(len(CH), np.inf)
    for j in range(len(CH)):
        v = D[:, j]
        v = v[np.isfinite(v)]
        if len(v) < MIN_N or v.std(ddof=1) == 0:
            continue
        t[j] = v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))
        crit[j] = stats.t.ppf(.975, len(v) - 1)
    return t, crit


def clusters(t, crit):
    """Group the channels with |t| > critical value, joining neighbours of the same sign."""
    sel = np.isfinite(t) & (np.abs(t) > crit)
    seen, out = set(), []
    for j in range(len(CH)):
        if not sel[j] or j in seen:
            continue
        stack, comp = [j], []
        seen.add(j)
        while stack:
            k = stack.pop()
            comp.append(k)
            for nb in NB[CH[k]]:
                m = CH.index(nb)
                if sel[m] and m not in seen and np.sign(t[m]) == np.sign(t[k]):
                    seen.add(m)
                    stack.append(m)
        out.append((comp, float(np.abs(t[comp]).sum())))
    return out


w('=' * 92)
w(' Figure 5a topographies - cluster-based permutation correction')
w('=' * 92)
w('  %d permutations - seed %d - participant sign flipping - adjacency radius %.2f' %
  (NPERM, SEED, ADJ_R))
w('  neighbours: ' + '  '.join('%s %d' % (c, len(NB[c])) for c in CH[:7]))
w('              ' + '  '.join('%s %d' % (c, len(NB[c])) for c in CH[7:]))
w()

rows = []
for nm, lo, hi in BANDS:
    ids = []
    for p in sorted(PS):
        a = rel(p, 'rest1', lo, hi)
        b = rel(p, 'task', lo, hi)
        if a and b:
            ids.append(p)
    D = np.full((len(ids), len(CH)), np.nan)
    for i, p in enumerate(ids):
        a = rel(p, 'rest1', lo, hi)
        b = rel(p, 'task', lo, hi)
        for j, c in enumerate(CH):
            if c in a and c in b:
                D[i, j] = b[c] - a[c]
    t, crit = chan_t(D)
    obs = clusters(t, crit)
    n_ch = [int(np.isfinite(D[:, j]).sum()) for j in range(len(CH))]

    rng = np.random.default_rng(SEED)
    null = np.zeros(NPERM)
    for b_ in range(NPERM):
        sg = rng.choice([-1.0, 1.0], size=(len(ids), 1))
        tp, cp = chan_t(D * sg)
        cl = clusters(tp, cp)
        null[b_] = max((m for _, m in cl), default=0.0)

    w('  [%s]  %d participants - n per channel %d-%d - critical |t| %.2f-%.2f'
      % (nm, len(ids), min(n_ch), max(n_ch),
         np.nanmin(crit[np.isfinite(t)]), np.nanmax(crit[np.isfinite(t)])))
    if not obs:
        w('    no channel clears the threshold - no cluster')
        rows.append(dict(band=nm, cluster='', channels='', mass=None,
                         p_corrected=None, survives='no'))
    for comp, mass in sorted(obs, key=lambda q: -q[1]):
        p_ = (np.sum(null >= mass) + 1) / (NPERM + 1)
        chs = ' '.join(CH[k] for k in sorted(comp, key=lambda k: -abs(t[k])))
        w('    cluster %-38s mass %6.2f  corrected P = %.4f %s'
          % (chs, mass, p_, '**' if p_ < .01 else ('*' if p_ < .05 else '')))
        rows.append(dict(band=nm, cluster=chs, channels=len(comp),
                         mass=round(mass, 3), p_corrected=round(float(p_), 4),
                         survives=('yes' if p_ < .05 else 'no')))
    w()

surv = [r for r in rows if r['survives'] == 'yes']
w('  surviving clusters: %d' % len(surv))
for r in surv:
    w('    %s - %s (corrected P = %.4f)' % (r['band'], r['cluster'],
                                      r['p_corrected']))
if not surv:
    w('    none - no asterisk is placed on the topographies')

with io.open(os.path.join(str(OUT), 'topo_cluster_perm.csv'), 'w',
             encoding='utf-8-sig', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
io.open(os.path.join(str(OUT), 'topo_cluster_perm.txt'), 'w',
        encoding='utf-8').write('\n'.join(LOG))
print('\nwritten -> outputs/topo_cluster_perm.csv  .txt')
