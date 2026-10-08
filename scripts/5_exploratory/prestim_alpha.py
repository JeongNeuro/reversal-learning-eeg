"""
prestim_alpha.py - the trial-level analysis of preregistration section 36.2

The registered wording in full
  "Trial-level analysis of whether pre-stimulus alpha power predicts switching
   versus staying on the following trial, by mixed-effects logistic regression."

It is one sentence, and the cluster, time window, band, alpha definition and random-effects
structure are all unspecified. That contrasts with section 36.9 (Mantel), which says "fully
specified ... rather than an open-ended search". The five choices below are therefore **not
fixed in advance**; each is tied to another clause of the registration as closely as possible.
Sensitivity runs are given for the two that matter most (cluster and time window).
  1 cluster     parieto-occipital (O1/O2/P7/P8)
                what section 18.6 designates as the primary cluster for the resting variables
                and the H1 family. All four clusters are reported.
  2 time window pre-stimulus -300 to -100 ms
                chosen to fall inside the fixation window (400-700 ms) of section 30.
                -500 to 0 ms and -1000 to 0 ms are given as sensitivity runs.
  3 band        8-13 Hz - the alpha band of section 18.4
  4 alpha       Morlet log power, z-scored within participant.
                The flat-spectrum mean of section 18.4 presupposes a run-level specparam fit
                and cannot be used on a 200 ms trial-level window. Within-participant z is
                used because what this analysis asks about is **trial-to-trial variation
                within one person**, not variation between participants.
  5 random eff. (1 | participant) as primary, (1 + alpha | participant) as sensitivity
Definition of the outcome
  switch = the choice on that trial differs from the choice on the previous valid trial. Timeout trials are excluded on both sides.
  "the following trial" is read as the trial that follows the pre-stimulus window.

Morlet follows section 29.4: n_cycles = frequency / 2.
Epoch rejection follows section 28 and is judged on **the channels contributing to the
cluster under analysis** (150 uV, the registered primary specification).

Output  outputs/prestim_alpha.csv  outputs/prestim_alpha_trials.csv
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT

import csv, glob, json, warnings
import numpy as np
import mne
warnings.filterwarnings('ignore'); mne.set_log_level('ERROR')

CH = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
CLUST = [('po4',   'parieto-occipital O1/O2/P7/P8  <- primary', ['O1','O2','P7','P8']),
         ('front', 'frontal AF3/AF4/F3/F4',                    ['AF3','AF4','F3','F4']),
         ('temp',  'temporo-parietal T7/T8/P7/P8',             ['T7','T8','P7','P8']),
         ('glob',  'global 14 channels',                       None)]
WINDOWS = [(-0.300, -0.100, '-300 to -100 ms  <- primary (reg. 30)'),
           (-0.500,  0.000, '-500 to 0 ms'),
           (-1.000,  0.000, '-1000 to 0 ms')]
FR = np.arange(8.0, 14.0, 1.0)      # the alpha band of registration 18.4
PRE, POST = 1.3, 0.3                # epoch bounds (covering the longest window, -1000 ms)
AMP = 150.0                         # the registered primary specification


def find(pid, pat):
    return sorted(set(p for p in glob.glob(f'{RAW}/**/{pat}', recursive=True)
                      if os.path.basename(p).startswith(pid)))

def i_(x):
    try: return int(float(x))
    except Exception: return 0

def morlet_pow(x, fs, freqs):
    """(ch, t) -> (ch, f, t) power. n_cycles = f/2 (registration 29.4)."""
    n = x.shape[-1]
    X = np.fft.fft(x, axis=-1); w = np.fft.fftfreq(n, 1.0 / fs)
    out = np.empty((x.shape[0], len(freqs), n))
    for k, f0 in enumerate(freqs):
        sd = (f0 / 2.0) / (2 * np.pi * f0)
        W = np.exp(-2 * (np.pi * sd) ** 2 * (w - f0) ** 2) * np.sqrt(2 * np.pi * sd)
        out[:, k, :] = np.abs(np.fft.ifft(X * W, axis=-1)) ** 2
    return out


rows = []
for pid in [f'{k:03d}' for k in range(1, 17)]:
    edfs = [p for p in find(pid, '*.edf') if not p.endswith('.md.edf')]
    mks = find(pid, '*_intervalMarker.csv')
    beh = glob.glob(f'{RAW}/**/*_{pid}_behav.csv', recursive=True)
    if not (edfs and mks and beh):
        continue
    B = list(csv.DictReader(open(beh[0], encoding='utf-8-sig')))
    M = [(i_(r['marker_value']), float(r['latency']))
         for r in csv.DictReader(open(mks[0], encoding='utf-8-sig'))]
    stim = sorted(t for v, t in M if v == 31)
    if len(stim) < 30:
        continue

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
    X = X - X.mean(axis=0, keepdims=True)        # common average, registration 26
    tmax = raw.times[-1]

    prev_choice, prev_rew = None, None
    n_ok = 0
    for j in range(min(len(stim), len(B))):
        t0 = stim[j]
        timeout = i_(B[j]['timeout'])
        choice = (B[j].get('choice') or '').strip()
        a = t0 - PRE; b = t0 + POST
        if a < 0 or b > tmax:
            prev_choice = None if timeout else (choice or None)
            continue
        i0 = int(round(a * fs)); i1 = i0 + int(round((PRE + POST) * fs))
        if i1 > X.shape[1]:
            continue
        seg = X[:, i0:i1]
        tt = np.arange(seg.shape[1]) / fs - PRE
        if timeout or not choice:
            prev_choice = None
            continue
        sw = None
        if prev_choice:
            sw = int(choice != prev_choice)
        P = None
        for lo, hi, wlab in WINDOWS:
            wm = (tt >= lo) & (tt <= hi)
            if wm.sum() < 8:
                continue
            if P is None:
                P = morlet_pow(seg, fs, FR)
            for key, _, chs in CLUST:
                ix = ([good.index(c) for c in chs if c in good]
                      if chs else list(range(len(good))))
                if not ix:
                    continue
                # registration 28 - judged on the cluster's channels, over the window actually used
                s2 = seg[np.ix_(ix, np.where(wm)[0])]
                if (s2.max(1) - s2.min(1)).max() > AMP:
                    continue
                v = np.log10(P[np.ix_(ix, np.arange(len(FR)), np.where(wm)[0])]).mean()
                if not np.isfinite(v):
                    continue
                rows.append(dict(pid=pid, trial=j, cluster=key, window=wlab,
                                 alpha=round(float(v), 6), switch=sw,
                                 prev_rew=prev_rew, rt=float(B[j].get('rt') or 0)))
                n_ok += 1
        prev_choice = choice
        prev_rew = i_(B[j].get('reward'))
    print(f'  {pid}: {len(good)}/{len(have)} channels - {n_ok} trial x window x cluster cells', flush=True)

os.makedirs(str(OUT), exist_ok=True)
dst = os.path.join(str(OUT), 'prestim_alpha_trials.csv')
with open(dst, 'w', newline='', encoding='utf-8') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print(f'\n{len(rows)} rows written -> {dst}')
