# -*- coding: utf-8 -*-
"""w1_prereg_windows.py - builds the per-window aperiodic exponent of W1 from the raw data

No script in the repository produced the outputs/W1_prereg_windows.csv that w1_prereg.py
reads. The situation is the same as for Figures 2 and 5, so it is rebuilt here. If the values
live only in a working file, the Code availability statement is not true.

reg. 29.2  stable window = the 8 trials before a reversal; post-reversal window = 8 trials
           from the reversal trial on. The time bounds of a window run **from 500 ms before
           the first trial's stimulus to 200 ms after the last trial's feedback**. Cutting
           stimulus to stimulus would put the last trial's feedback outside the window.
reg. 26/27 average re-reference, 0.5 Hz FIR high-pass, no offline notch
reg. 28    an epoch is rejected when the p2p of a **channel contributing to the cluster**
           exceeds the threshold. The number of surviving epochs therefore differs by cluster
           even for the same window. A channel is bad when its standard deviation exceeds
           three times the median or falls below 0.1 times it (the same rule as w2_prereg_trials.py).
reg. 18.6  all four clusters
reg. 24    2 s non-overlapping epochs, Welch with a Hann window
reg. 21    FOOOF 2-40 Hz, fixed, width [1,8], at most 6 peaks, minimum height 0.05,
           threshold 2.0, R2 >= .85

Cluster exponents are obtained by **averaging the channel spectra first and fitting once**
(the same rule as spectro.py), not by averaging per-channel exponents.

**Limit of reproduction** - this script produces the structure of the W1_prereg_windows.csv
the manuscript used. The window bounds, reward density and good-channel counts match that
file **exactly**, and the epoch count per window matches for 98.7%. What the surviving
records do not determine is where the 2 s tiles were aligned inside a window. On a window
of about 20 s a shift of only 0.3 s moves the per-window exponent by 0.05 to 0.3, so the

per-window values do not agree to four decimals. The conclusion for all four clusters is
unchanged by the alignment - every one is positive with an interval containing 0 (registered
primary cluster +0.012 [-0.050, +0.074], 11 participants, 46 pairs). The +0.033

[-0.033, +0.099] the manuscript reports is the value produced from W1_prereg_windows.csv.
The default output is W1_windows_rebuild.csv, so the file the manuscript used is never overwritten silently. Pass --out to overwrite it.

Output  outputs/W1_windows_rebuild.csv - participant x pair x window x cluster
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import RAW, OUT                                   # noqa: E402

import argparse                                              # noqa: E402
import csv                                                   # noqa: E402
import glob                                                  # noqa: E402
import io                                                    # noqa: E402
import warnings                                              # noqa: E402

import numpy as np                                           # noqa: E402
import mne                                                   # noqa: E402
from scipy.signal import welch                               # noqa: E402

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')
from fooof import FOOOF                                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--amp', type=float, default=150.0)          # the registered primary specification
ap.add_argument('--win', type=int, default=8)                # registration 29.2
# Minimum epochs per window AND per cluster, applied after the cluster-wise amplitude
# rejection of registration 28. The default of 5 epochs is 10 s. Note that the 60 s /
# 30-epoch minimum of registration 28.7 is the criterion for a RESTING RUN and does not
# apply here: a W1 window spans 8 trials, which is about 23 s, so no window could ever
# reach it. The longest window in this dataset is 37.8 s (18 epochs).
ap.add_argument('--min-epochs', type=int, default=5)
ap.add_argument('--out', default=str(OUT / 'W1_windows_rebuild.csv'))
A = ap.parse_args()

RAW = str(RAW)
CH = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
      'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
CLUST = [('po4', ['O1', 'O2', 'P7', 'P8']),        # the registered primary cluster
         ('front', ['AF3', 'AF4', 'F3', 'F4']),
         ('temp', ['T7', 'T8', 'P7', 'P8']),
         ('glob', None)]                            # None = every good channel
EPOCH_S = 2.0
PRE, POST = 0.5, 0.2         # window bounds - before the first stimulus, after the last feedback (s)
FRANGE = [2, 40]
R2_MIN = 0.85


def find(pid, pat):
    return sorted(set(p for p in glob.glob('%s/**/%s' % (RAW, pat),
                                           recursive=True)
                      if os.path.basename(p).startswith(pid)))


def i_(x):
    try:
        return int(float(x))
    except Exception:
        return 0


def fit_spec(f, p):
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.05,
               peak_threshold=2.0, aperiodic_mode='fixed', verbose=False)
    try:
        fm.fit(f, p, list(FRANGE))
    except Exception:
        return None
    if not np.isfinite(fm.r_squared_) or fm.r_squared_ < R2_MIN:
        return None
    return fm


rows = []
for pid in ['%03d' % k for k in range(1, 17)]:
    edfs = [p for p in find(pid, '*.edf') if not p.endswith('.md.edf')]
    mks = find(pid, '*_intervalMarker.csv')
    behp = glob.glob('%s/**/*_%s_behav.csv' % (RAW, pid), recursive=True)
    if not (edfs and mks and behp):
        continue
    B = list(csv.DictReader(io.open(behp[0], encoding='utf-8-sig')))
    M = [(i_(r['marker_value']), float(r['latency']))
         for r in csv.DictReader(io.open(mks[0], encoding='utf-8-sig'))]
    stim = sorted(t for v, t in M if v == 31)
    fball = sorted(t for v, t in M if v in (33, 34))       # unrewarded and rewarded
    if len(stim) < 30 or not fball:
        continue
    n = min(len(stim), len(B))
    rev = [i for i in range(n) if B[i].get('is_reversal_trial') == '1']
    if not rev:
        continue

    raw = mne.io.read_raw_edf(edfs[0], preload=True)
    have = [c for c in CH if c in raw.ch_names]
    raw.pick(have)
    raw.filter(0.5, None, method='fir', fir_window='hamming',
               phase='zero', verbose=False)          # registration 27
    fs = raw.info['sfreq']
    npts = int(EPOCH_S * fs)
    X = raw.get_data() * 1e6
    tmax = raw.times[-1]

    # bad channels - standard deviation above three times the median or below 0.1 times it
    sd = X.std(axis=1)
    msd = np.median(sd)
    good = [have[k] for k in range(len(have))
            if not (sd[k] > 3 * msd or sd[k] < 0.1 * msd)]
    if len(good) < 8:
        continue
    X = X[[have.index(c) for c in good]]
    X = X - X.mean(axis=0, keepdims=True)            # average re-reference, registration 26

    npair = 0
    for r_ in rev:
        a0, a1 = r_ - A.win, r_                      # trial range of the stable window
        b0, b1 = r_, r_ + A.win                      # trial range of the post-reversal window
        if a0 < 0 or b1 >= n:
            continue                                 # the window runs past the recording
        npair += 1
        for wname, (t0i, t1i) in (('stable', (a0, a1)), ('post', (b0, b1))):
            nx = [t for t in fball if t >= stim[t1i - 1]]
            if not nx:
                continue
            t0 = stim[t0i] - PRE                     # 500 ms before the first stimulus
            t1 = nx[0] + POST                        # 200 ms after the last feedback
            if t0 < 0 or t1 > tmax:
                continue
            i0, i1 = int(round(t0 * fs)), int(round(t1 * fs))
            if i1 > X.shape[1]:
                continue
            k = (i1 - i0) // npts                    # 2 s non-overlapping tiles
            if k < A.min_epochs:
                continue
            ep = np.stack([X[:, i0 + j * npts:i0 + (j + 1) * npts]
                           for j in range(k)])       # (ep, ch, t)
            pp = ep.max(2) - ep.min(2)               # (ep, ch)
            # reward density = the proportion of trials in the window scheduled to reward the correct option
            rd = float(np.mean([float(B[j]['rewarded_if_correct'])
                                for j in range(t0i, t1i)]))
            dur = round(t1 - t0, 3)
            for cl, want in CLUST:
                ix = ([good.index(c) for c in want if c in good]
                      if want else list(range(len(good))))
                if not ix:
                    continue
                # registration 28 - rejection is judged on that cluster's contributing channels only
                ok = [j for j in range(k) if pp[j][ix].max() <= A.amp]
                if len(ok) < A.min_epochs:
                    continue
                f, P = welch(ep[np.asarray(ok)][:, ix], fs=fs, nperseg=npts,
                             window='hann', axis=-1)
                fm = fit_spec(f, P.mean(0).mean(0))
                if fm is None:
                    continue
                rows.append(dict(
                    pid=pid, pair=npair, window=wname, cluster=cl,
                    exponent=round(float(fm.aperiodic_params_[-1]), 6),
                    reward_density=rd, window_duration_s=dur,
                    n_epochs=len(ok), n_epochs_total=k,
                    r2=round(float(fm.r_squared_), 4),
                    fit_error=round(float(fm.error_), 4), n_chan=len(ix)))
    print('  %s: %d reversals - %d pairs used - %d rows'
          % (pid, len(rev), npair, sum(1 for q in rows if q['pid'] == pid)),
          flush=True)

if not rows:
    raise SystemExit('no window could be built')

with io.open(A.out, 'w', encoding='utf-8-sig', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
print('\n%d rows - %d participants -> %s'
      % (len(rows), len(set(q['pid'] for q in rows)), A.out))
