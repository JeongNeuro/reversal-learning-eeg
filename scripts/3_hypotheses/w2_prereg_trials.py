# -*- coding: utf-8 -*-
"""W2 stage 1 - produces the trial-level theta of W2 under the preregistered specification.

reg. 29.5  epochs are stimulus-locked, stimulus -1000 to +5200 ms
           (feedback = stimulus + reaction time + 200 ms, so under feedback locking the
            baseline can fall outside the epoch. Hence the lock to the stimulus.)
reg. 30    baseline = stimulus-locked -300 to -100 ms of the same trial, dB change
reg. 29.4  Morlet, 3-30 Hz in **1 Hz steps**, n_cycles = frequency / 2
reg. 18.5  theta = 4-8 Hz, 200-500 ms after that trial's feedback marker
reg. 28    an epoch is rejected when the p2p of a channel contributing to the cluster exceeds
           the threshold within 500 ms before the stimulus to 800 ms after that trial's
           feedback. A recording that does not fill the window is excluded. A participant
           contributes to a condition mean only with at least 10 surviving epochs in it.
reg. 26/27 average re-reference, 0.5 Hz FIR high-pass, no offline notch
reg. 18.6  all four clusters      reg. 29.3  the 3-level secondary contrast = rewarded-correct / unrewarded-correct / unrewarded-incorrect
reg. 25    sessions with |clock drift| > 20 ms are excluded from W2

Output  outputs/W2_prereg_trials.csv - participant x trial x cluster
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import RAW, OUT

import argparse, csv, glob, json, warnings
import numpy as np
import mne
warnings.filterwarnings('ignore'); mne.set_log_level('ERROR')

ap = argparse.ArgumentParser()
ap.add_argument('--amp', type=float, default=150.0)                  # the registered primary specification
ap.add_argument('--reference', choices=['average', 'original'], default='average')
ap.add_argument('--out', default=str(OUT / 'W2_prereg_trials.csv'))
ap.add_argument('--grid-out', default=str(OUT / 'W2_prereg_grid_trials.csv'),
                help='band x time-window grid for the frontal cluster (long format)')
A = ap.parse_args()

# band x time-window grid - produced with the registered wavelet. The windows are feedback-locked.
GRID_BANDS = [('theta 4-8 Hz', 4.0, 8.0),
              ('alpha 8-13 Hz', 8.0, 13.0),
              ('alpha 7-14 Hz', 7.0, 14.0),
              ('beta 13-30 Hz', 13.0, 30.0)]
GRID_WINS = [('200–500 ms', 0.200, 0.500), ('300–600 ms', 0.300, 0.600),
             ('0–500 ms', 0.000, 0.500), ('500–1000 ms', 0.500, 1.000)]

RAW = str(RAW)
CH = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
CLUST = {'front':['AF3','AF4','F3','F4'],      # registration 18.6 - the W2 primary cluster
         'po4':  ['O1','O2','P7','P8'],
         'temp': ['T7','T8','P7','P8'],
         'glob':  None}
FR = np.arange(3.0, 31.0, 1.0)                 # registration 29.4 - 1 Hz steps
TH = (FR >= 4) & (FR <= 8)                     # theta
PRE, POST = 1.0, 5.2                           # epoch bounds of registration 29.5 (s)
BL = (-0.300, -0.100)                          # baseline of registration 30 (stimulus-locked)
WIN = (0.200, 0.500)                           # post-feedback window of registration 18.5
REJ = (-0.500, 0.800)                          # rejection window of registration 28

def find(pid, pat):
    return sorted(set(p for p in glob.glob(f'{RAW}/**/{pat}', recursive=True)
                      if os.path.basename(p).startswith(pid)))

def i_(x):
    try: return int(float(x))
    except Exception: return 0

def morlet(x, fs, freqs):
    """x: (ch, t) -> (ch, f, t) power. n_cycles = f / 2 (registration 29.4)."""
    n = x.shape[-1]
    X = np.fft.fft(x, axis=-1); w = np.fft.fftfreq(n, 1.0 / fs)
    out = np.empty((x.shape[0], len(freqs), n))
    for k, f0 in enumerate(freqs):
        nc = f0 / 2.0
        sd = nc / (2 * np.pi * f0)              # = 1/(4pi), independent of frequency
        W = np.exp(-2 * (np.pi * sd) ** 2 * (w - f0) ** 2) * np.sqrt(2 * np.pi * sd)
        out[:, k, :] = np.abs(np.fft.ifft(X * W, axis=-1)) ** 2
    return out

rows = []
grid = []
drift = {}
for pid in [f'{k:03d}' for k in range(1, 17)]:
    edfs = [p for p in find(pid, '*.edf') if not p.endswith('.md.edf')]
    mks = find(pid, '*_intervalMarker.csv')
    beh = glob.glob(f'{RAW}/**/*_{pid}_behav.csv', recursive=True)
    mts = glob.glob(f'{RAW}/**/*{pid}*meta*.json', recursive=True) or \
          [p for p in glob.glob(f'{RAW}/**/*meta*.json', recursive=True)
           if f'_{pid}_' in os.path.basename(p) or os.path.basename(p).startswith(pid)]
    if not (edfs and mks and beh):
        continue
    B = list(csv.DictReader(open(beh[0], encoding='utf-8-sig')))
    M = [(i_(r['marker_value']), float(r['latency']))
         for r in csv.DictReader(open(mks[0], encoding='utf-8-sig'))]
    stim = sorted(t for v, t in M if v == 31)
    fb33 = sorted(t for v, t in M if v == 33)      # unrewarded
    fb34 = sorted(t for v, t in M if v == 34)      # rewarded
    fball = sorted(fb33 + fb34)
    if len(stim) < 30:
        continue
    if mts:
        try: drift[pid] = float(json.load(open(mts[0], encoding='utf-8'))
                               .get('epoch_offset_drift_ms'))
        except Exception: pass

    raw = mne.io.read_raw_edf(edfs[0], preload=True)
    have = [c for c in CH if c in raw.ch_names]
    raw.pick(have)
    raw.filter(0.5, None, method='fir', fir_window='hamming', verbose=False)
    fs = raw.info['sfreq']
    X = raw.get_data() * 1e6
    sd = X.std(axis=1); msd = np.median(sd)
    good = [have[k] for k in range(len(have)) if not (sd[k] > 3*msd or sd[k] < 0.1*msd)]
    if len(good) < 2:
        continue
    X = X[[have.index(c) for c in good]]
    if A.reference == 'average':
        X = X - X.mean(axis=0, keepdims=True)
    tmax = raw.times[-1]

    nkept = 0
    for j in range(min(len(stim), len(B))):
        t0 = stim[j]
        if i_(B[j]['timeout']):
            continue                                   # registration 35 - timeouts are excluded
        nx = [t for t in fball if t >= t0]
        if not nx:
            continue
        tfb = nx[0]                                    # that trial's feedback marker
        rew = 1 if (fb34 and min(abs(tfb - t) for t in fb34) < 1e-6) else 0
        cor = i_(B[j]['correct'])
        a = t0 - PRE; b = t0 + POST
        if a < 0 or b > tmax:
            continue                                   # the recording does not fill the window
        i0 = int(round(a * fs)); i1 = i0 + int(round((PRE + POST) * fs))
        if i1 > X.shape[1]:
            continue
        seg = X[:, i0:i1]
        tt = np.arange(seg.shape[1]) / fs - PRE        # time relative to the stimulus
        # rejection window - 500 ms before the stimulus to 800 ms after feedback
        rm = (tt >= REJ[0]) & (tt <= (tfb - t0) + REJ[1])
        bm = (tt >= BL[0]) & (tt <= BL[1])
        wm = (tt >= (tfb - t0) + WIN[0]) & (tt <= (tfb - t0) + WIN[1])
        if bm.sum() < 3 or wm.sum() < 3:
            continue
        P = morlet(seg, fs, FR)                        # (ch, f, t)
        base = P[:, :, bm].mean(axis=2, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            dB = 10.0 * np.log10(P / base)
        for cl, chans in CLUST.items():
            ix = ([good.index(c) for c in chans if c in good]
                  if chans else list(range(len(good))))
            if not ix:
                continue
            if (seg[np.ix_(ix, np.where(rm)[0])].max(1)
                    - seg[np.ix_(ix, np.where(rm)[0])].min(1)).max() > A.amp:
                continue                               # amplitude rejection, registration 28
            v = dB[np.ix_(ix, np.where(TH)[0], np.where(wm)[0])]
            if not np.isfinite(v).all():
                continue
            # theta dB is computed per channel and then averaged over the cluster (registration 19)
            rows.append(dict(pid=pid, trial=j, cluster=cl,
                             theta_dB=round(float(v.mean(axis=(1, 2)).mean()), 6),
                             rewarded=rew, correct=cor,
                             cond3=('R_cor' if rew else ('U_cor' if cor else 'U_inc')),
                             rt=float(B[j]['rt'] or 0), n_chan=len(ix)))
            nkept += 1
            if cl != 'front':
                continue
            # The grid is produced for the frontal cluster only. Being the same dB array and the
            # same baseline, the theta x 200-500 ms cell equals the theta_dB above.
            fbo = tfb - t0
            for bname, flo, fhi in GRID_BANDS:
                fm = (FR >= flo) & (FR <= fhi)
                if not fm.any():
                    continue
                for wname, wlo, whi in GRID_WINS:
                    gm = (tt >= fbo + wlo) & (tt <= fbo + whi)
                    if gm.sum() < 3:
                        continue
                    gv = dB[np.ix_(ix, np.where(fm)[0], np.where(gm)[0])]
                    if not np.isfinite(gv).all():
                        continue
                    grid.append(dict(pid=pid, trial=j, band=bname,
                                     window=wname, rewarded=rew, correct=cor,
                                     dB=round(float(gv.mean(axis=(1, 2)).mean()), 6)))
    print(f'  {pid}: {len(good)}/{len(have)} channels - {nkept} trial x cluster cells -'
          f' drift {drift.get(pid, float("nan")):+.2f} ms')

with open(A.out, 'w', newline='', encoding='utf-8') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
if grid:
    with open(A.grid_out, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(grid[0]))
        wr.writeheader(); wr.writerows(grid)
    print(f'grid {len(grid)} rows -> {A.grid_out}')
print(f'\n{len(rows)} rows written -> {A.out}')
bad = [p for p, d in drift.items() if abs(d) > 20]
print('registration 25 - sessions excluded from W2 for |drift| > 20 ms: %s' % (' '.join(bad) if bad else 'none'))
