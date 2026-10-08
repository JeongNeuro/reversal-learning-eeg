# -*- coding: utf-8 -*-
"""robustness.py - the robustness values of Appendix 1

No script in the repository produced the values Appendix 1 reports. The situation is the
same as for the W1 window data, so it is moved here.

Three things are produced.

  1. results under alternative samples
       main-text sample      007, 008 and 009 excluded (n = 9)
       009 restored          007 and 008 excluded (n = 10) - the value Appendix 1 reports
       007 only excluded     both win-stay exclusions restored (n = 11)
       no criteria applied   n = 12 - also reported in Appendix 1
  2. leave-one-participant-out reanalysis (on the main-text sample of 9)
  3. an age-confound check - run only when data/demographics.csv is present

**Careful with the sample wording** - the sample with "no exclusion criteria applied" is the
12. The 10 is the one where only 009 of the two win-stay exclusions is restored, and 007 is
still out on the reaction-time criterion. The two samples lead to different conclusions (the
switch-bias correlation is .612 against .112), so a robustness statement that does not say
which sample it means is not accurate.

Age, sex, disability type and medication are participant data and are not in the repository.
Section 3 runs only when data/demographics.csv is present and is skipped otherwise
(the same convention as subtype_validate.py).

Cluster exponents are obtained by averaging the channel spectra first and fitting once (the
same rule as spectro.py). The interval follows the repository-wide convention: participant
resampling, 10,000 resamples, seed 1, and resamples with fewer than 4 distinct participants
are discarded.

Output  outputs/robustness.csv  .txt
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import DATA, OUT                                  # noqa: E402

import csv                                                   # noqa: E402
import io                                                    # noqa: E402
import json                                                  # noqa: E402
import warnings                                              # noqa: E402

import numpy as np                                           # noqa: E402
from scipy import stats                                      # noqa: E402

warnings.filterwarnings('ignore')
from fooof import FOOOF                                      # noqa: E402

import cohort                                                # noqa: E402

SEED = 1
NB = 10000
GUARD = 4                       # discard a resample with fewer distinct participants than this
R2_MIN = 0.85

CL = {'front': ['AF3', 'AF4', 'F3', 'F4'],
      'po4': ['O1', 'O2', 'P7', 'P8'],
      'occ': ['O1', 'O2']}

LOG = []


def w(t=''):
    LOG.append(t)
    print(t, flush=True)


def fit(f, p):
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.05,
               peak_threshold=2.0, aperiodic_mode='fixed', verbose=False)
    try:
        fm.fit(f, p, [2.0, 40.0])
    except Exception:
        return None
    if not np.isfinite(fm.r_squared_) or fm.r_squared_ < R2_MIN:
        return None
    return float(fm.aperiodic_params_[-1])


PS = json.load(io.open(os.path.join(str(DATA), 'psd_all.json'),
                       encoding='utf-8'))
B = json.load(io.open(os.path.join(str(DATA), 'behav_metrics.json'),
                      encoding='utf-8'))

_CACHE = {}


def exponent(pid, cl):
    """Mean cluster aperiodic exponent over the two resting runs."""
    k = (pid, cl)
    if k in _CACHE:
        return _CACHE[k]
    v = PS.get(pid)
    out = None
    if v:
        f = np.asarray(v['f'], float)
        ix = [v['ch'].index(c) for c in CL[cl] if c in v['ch']]
        if ix:
            o = [fit(f, np.asarray(v[r], float)[ix].mean(0))
                 for r in ('rest1', 'rest2') if v.get(r)]
            o = [q for q in o if q is not None]
            if o:
                out = float(np.mean(o))
    _CACHE[k] = out
    return out


def boot_rho(x, y, seed=SEED, nb=NB, guard=GUARD):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = []
    for _ in range(nb):
        i = rng.integers(0, len(x), len(x))
        if len(set(i.tolist())) < guard:
            continue
        v = stats.spearmanr(x[i], y[i]).statistic
        if np.isfinite(v):
            o.append(v)
    return np.percentile(o, [2.5, 97.5])


def pair(ids, cl, beh):
    xs, ys, used = [], [], []
    for p in ids:
        e = exponent(p, cl)
        if e is None or p not in B or B[p].get(beh) is None:
            continue
        xs.append(e)
        ys.append(B[p][beh])
        used.append(p)
    return xs, ys, used


ROWS = []


def report(tag, ids, cl, beh, nm):
    xs, ys, used = pair(ids, cl, beh)
    if len(xs) < 4:
        w('  %-44s insufficient data (n = %d)' % (nm, len(xs)))
        return
    r = float(stats.spearmanr(xs, ys).statistic)
    lo, hi = boot_rho(xs, ys)
    w('  %-34s n=%2d  rho %+.3f  [%+.3f, %+.3f]' % (nm, len(xs), r, lo, hi))
    ROWS.append(dict(block=tag, label=nm, cluster=cl, behaviour=beh,
                     n=len(xs), rho=round(r, 4),
                     ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                     participants=' '.join(used)))


ALL12 = sorted(B)                                   # the 12 who completed the task
SAMPLES = [
    ('main-text sample (007, 008, 009 excluded)',
     [p for p in ALL12 if p not in ('007', '008', '009')]),
    ('009 restored (007, 008 excluded)',
     [p for p in ALL12 if p not in ('007', '008')]),
    ('007 only excluded (both win-stay exclusions restored)',
     [p for p in ALL12 if p != '007']),
    ('no criteria applied (all 12)', ALL12),
]

w('=' * 92)
w(' Appendix 1 - robustness')
w('=' * 92)
w('  The correlation of the frontal aperiodic exponent with two behavioural indices, under varying samples.')
w('  The interval is participant resampling, %d resamples, seed %d, discarding resamples with fewer than %d distinct participants'
  % (NB, SEED, GUARD))
w()

w('-' * 92)
w(' 1. results under alternative samples')
w('-' * 92)
for nm, ids in SAMPLES:
    w('  [%s]' % nm)
    report('sample', ids, 'front', 'sbi', 'frontal exponent × switch-bias index')
    report('sample', ids, 'front', 'lose_shift', 'frontal exponent × lose-shift')
    w()

w('  Note. The sample with "no exclusion criteria applied" is the 12. The 10 is the one where')
w('        only 009 of the two win-stay exclusions is restored; 007 is still out on the')
w('        reaction-time criterion. The switch-bias correlation differs greatly in size between')
w('        the two samples, so which sample is meant has to be stated.')
w()

w('-' * 92)
w(' 2. leave-one-participant-out reanalysis - on the main-text sample of 9')
w('-' * 92)
base = [p for p in ALL12 if p not in ('007', '008', '009')]
for beh, nm in (('lose_shift', 'lose-shift'), ('sbi', 'switch-bias index')):
    vals = {}
    for drop in base:
        xs, ys, _ = pair([p for p in base if p != drop], 'front', beh)
        if len(xs) >= 4:
            vals[drop] = float(stats.spearmanr(xs, ys).statistic)
    if not vals:
        continue
    w('  [frontal exponent × %s]  range %.3f to %.3f'
      % (nm, min(vals.values()), max(vals.values())))
    w('    ' + ' · '.join('%s %.3f' % (k, v) for k, v in sorted(vals.items())))
    for k, v in sorted(vals.items()):
        ROWS.append(dict(block='leave-one-out', label='%s excluded' % k,
                         cluster='front', behaviour=beh, n=len(base) - 1,
                         rho=round(v, 4), ci_lo='', ci_hi='',
                         participants=' '.join(p for p in base if p != k)))
    w()

w('-' * 92)
w(' 3. age-confound check')
w('-' * 92)
DEMO = cohort.demographics(str(DATA), verbose=False) or {}
if not DEMO:
    w('  Skipped - age is participant data and is not in the repository.')
    w('  To run it, put id and age columns in data/%s.' % cohort.DEMO_FILE)
    w('  The values in Appendix 1 of the manuscript are frontal rho = -.301 [-.75, +.29] (n = 15)')
    w('  and occipital O1/O2 rho = -.592 [-.88, -.12] (n = 14).')
else:
    for cl, nm in (('front', 'frontal'), ('occ', 'occipital O1/O2')):
        xs, ys, used = [], [], []
        for p in sorted(PS):
            e = exponent(p, cl)
            a = (DEMO.get(p) or {}).get('age')
            if e is None or a in (None, ''):
                continue
            xs.append(e)
            ys.append(float(a))
            used.append(p)
        if len(xs) < 4:
            w('  %-18s insufficient data (n = %d)' % (nm, len(xs)))
            continue
        r = float(stats.spearmanr(xs, ys).statistic)
        lo, hi = boot_rho(xs, ys)
        w('  %-18s × age  n=%2d  rho %+.3f  [%+.3f, %+.3f]'
          % (nm, len(xs), r, lo, hi))
        ROWS.append(dict(block='age', label='%s × age' % nm, cluster=cl,
                         behaviour='age', n=len(xs), rho=round(r, 4),
                         ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4),
                         participants=' '.join(used)))
w()

with io.open(os.path.join(str(OUT), 'robustness.csv'), 'w',
             encoding='utf-8-sig', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(ROWS[0]))
    wr.writeheader()
    wr.writerows(ROWS)
io.open(os.path.join(str(OUT), 'robustness.txt'), 'w',
        encoding='utf-8').write('\n'.join(LOG))
print('\nwritten -> outputs/robustness.csv  .txt')
