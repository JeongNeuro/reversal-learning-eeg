"""
feasibility_prereg.py - produces F3 and F4 exactly under the preregistered rule

feasibility.py reads spectro.json, and that file is the result of judging epoch rejection
across **every** surviving channel. Registration 28 differs.

  "An epoch is rejected if the peak-to-peak amplitude **in any channel
   contributing to the cluster under analysis** exceeds a threshold."

Judging rejection per cluster changes the yield greatly (under common average, 150 uV and the
60 s criterion, parieto-occipital gives 6 participants under all-channel judgement against 14
per cluster). This script recomputes from the raw data under the registered rule.

The specification is the registered primary one.
  common-average re-reference, 150 uV, 2-40 Hz, fixed, peak_width_limits [1,8],
  max_n_peaks 6 · min_peak_height 0.1 · peak_threshold 2.0 ·
  R2 >= .90, fit error <= .10, first 10 s discarded, 2 s non-overlapping, mean of linear power

The minimum data requirement is given under both the 60 s of registration 28.7 and the manuscript's post hoc 30 s.
Alpha follows the definition of registration 18.4 (the mean of the flat log10 spectrum over 8-13 Hz).

Output  outputs/feasibility_prereg.csv
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT

import csv, glob, json, warnings
import numpy as np
import mne
from scipy.signal import get_window
from fooof import FOOOF

warnings.filterwarnings('ignore'); mne.set_log_level('ERROR')

CH = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
CLUST = [('po4',   'parieto-occipital O1/O2/P7/P8  <- primary', ['O1','O2','P7','P8']),
         ('temp',  'temporo-parietal T7/T8/P7/P8',             ['T7','T8','P7','P8']),
         ('front', 'frontal AF3/AF4/F3/F4',                    ['AF3','AF4','F3','F4']),
         ('glob',  'global 14 channels',                       None)]
AMP, LEADIN, EPL = 150.0, 10.0, 2.0
MINEP = (('60 s (reg. 28.7)', 30), ('30 s (post hoc)', 15))
FLOOR = 8


def find(pid, pat):
    return sorted(set(p for p in glob.glob(f'{RAW}/**/{pat}', recursive=True)
                      if os.path.basename(p).startswith(pid)))

def i_(x):
    try: return int(float(x))
    except Exception: return 0

def psd_of(X, idx, fs, nfft):
    win = get_window('hann', nfft)
    scale = 1.0 / (fs * (win ** 2).sum())
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    keep = []
    for a in range(0, X.shape[1] - nfft + 1, nfft):
        ep = X[:, a:a + nfft]
        if (ep[idx].max(1) - ep[idx].min(1)).max() > AMP:
            continue
        sp = np.fft.rfft(ep * win, axis=-1)
        pw = (np.abs(sp) ** 2) * scale
        pw[:, 1:-1] *= 2.0
        keep.append(pw)
    if not keep:
        return freqs, None, 0
    return freqs, np.mean(keep, axis=0), len(keep)

def fit(f, p):
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1,
               peak_threshold=2.0, aperiodic_mode='fixed', verbose=False)
    try: fm.fit(f, p, [2, 40])
    except Exception: return None
    if not np.isfinite(fm.r_squared_) or fm.r_squared_ < 0.90: return None
    if not np.isfinite(fm.error_) or fm.error_ > 0.10: return None
    flat = fm.power_spectrum - fm._ap_fit
    m = (fm.freqs >= 8) & (fm.freqs <= 13)
    return float(fm.aperiodic_params_[-1]), float(np.mean(flat[m]))


# -- computation --------------------------------------------------
V = {}          # (pid, run, cl) -> (n_epoch, exponent, alpha_flat)
runs_of = {}
print('recomputing from the raw data under the registered rule')
for pid in [f'{k:03d}' for k in range(1, 17)]:
    edfs = [p for p in find(pid, '*.edf') if not p.endswith('.md.edf')]
    mks = find(pid, '*_intervalMarker.csv')
    if not (edfs and mks):
        continue
    M = [(i_(r['marker_value']), float(r['latency']))
         for r in csv.DictReader(open(mks[0], encoding='utf-8-sig'))]
    runs = []
    for s, e in ((10, 11), (12, 13)):
        a = [t for v, t in M if v == s]; b = [t for v, t in M if v == e]
        if a and b:
            runs.append((a[0], b[0]))
    if not runs:
        continue
    runs_of[pid] = len(runs)
    raw = mne.io.read_raw_edf(edfs[0], preload=True)
    have = [c for c in CH if c in raw.ch_names]
    raw.pick(have)
    raw.filter(0.5, None, method='fir', fir_window='hamming', verbose=False)
    fs = raw.info['sfreq']; nfft = int(EPL * fs)
    X0 = raw.get_data() * 1e6
    sd = X0.std(axis=1); msd = np.median(sd)
    good = [have[k] for k in range(len(have)) if not (sd[k] > 3*msd or sd[k] < 0.1*msd)]
    if len(good) < 2:
        continue
    gi = [have.index(c) for c in good]
    for ri, (t0, t1) in enumerate(runs, 1):
        a = int((t0 + LEADIN) * fs); b = int(min(t1, raw.times[-1]) * fs)
        if b - a < nfft:
            continue
        Z = X0[gi, a:b]
        Z = Z - Z.mean(axis=0, keepdims=True)           # common average, registration 26
        for key, _, chs in CLUST:
            ix = ([good.index(c) for c in chs if c in good]
                  if chs else list(range(len(good))))
            if not ix:
                continue
            f, P, ne = psd_of(Z, ix, fs, nfft)
            if P is None:
                continue
            r = fit(f, P[ix].mean(0))
            V[(pid, ri, key)] = (ne, None, None) if r is None else (ne, r[0], r[1])
    print(f'  {pid}: {len(runs)} runs - {len(good)}/{len(have)} channels', flush=True)

completed = sorted(p for p, n in runs_of.items() if n >= 2)      # registration 13
L, rows = [], []
def P_(t=''): L.append(t); print(t)

P_()
P_('=' * 78)
P_(' F3 and F4 - the preregistered rule (per-cluster epoch rejection)')
P_('=' * 78)
P_(f'  completed sessions {len(completed)}: {" ".join(completed)}')
P_('  specification  common average, 150 uV, 2-40 Hz, fixed, R2 >= .90, fit error <= .10')
P_()

P_('-' * 78)
P_(' F3  EEG yield - at least 80% of completed sessions (prespecified)')
P_('-' * 78)
P_(f"  {'criterion':<20}" + ''.join(f'{n:>44}' for _, n, _ in CLUST))
for lab, ep in MINEP:
    cells = []
    for key, _, _ in CLUST:
        ok = [p for p in completed
              if any((V.get((p, r, key)) or (0, None, None))[0] >= ep
                     and (V.get((p, r, key)) or (0, None, None))[1] is not None
                     for r in (1, 2))]
        cells.append(f'{len(ok)}/{len(completed)} {100*len(ok)/len(completed):.0f}%')
        rows.append(dict(item=f'F3 {key} - {lab}',
                         value=f'{len(ok)}/{len(completed)} = {100*len(ok)/len(completed):.1f}%',
                         criterion='>= 80%',
                         verdict='met' if len(ok)/len(completed) >= .80 else 'not met'))
    P_(f'  {lab:<20}' + ''.join(f'{c:>44}' for c in cells))
P_()

P_('-' * 78)
P_(' F4  run-to-run reliability - no threshold (prespecified)')
P_('-' * 78)
def icc21(a, b):
    Y = np.column_stack([a, b]); n, k = Y.shape
    gm = Y.mean(); rm = Y.mean(1); cm = Y.mean(0)
    msr = k*((rm-gm)**2).sum()/(n-1); msc = n*((cm-gm)**2).sum()/(k-1)
    mse = ((Y-rm[:, None]-cm[None, :]+gm)**2).sum()/((n-1)*(k-1))
    return (msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n)
def boot_icc(a, b, nb=10000, seed=1):   # seed and count unified (2026-09-27)
    rng = np.random.default_rng(seed); a = np.asarray(a); b = np.asarray(b); o = []
    for _ in range(nb):
        i = rng.integers(0, len(a), len(a))
        try:
            v = icc21(a[i], b[i])
            if np.isfinite(v): o.append(v)
        except Exception: pass
    return (np.percentile(o, 2.5), np.percentile(o, 97.5)) if len(o) > 100 else (np.nan, np.nan)

for lab, ep in MINEP:
    P_(f'  [{lab}]')
    P_(f"    {'cluster':<44}{'index':<16}{'ICC(2,1)':>10}{'95% CI':>22}{'n':>5}")
    for key, nm, _ in CLUST:
        for j, kn in ((1, 'exponent'), (2, 'periodic alpha')):
            A, B = [], []
            for p in completed:
                v = []
                for r in (1, 2):
                    t = V.get((p, r, key))
                    if t and t[0] >= ep and t[j] is not None:
                        v.append(t[j])
                if len(v) >= 2:
                    A.append(v[0]); B.append(v[1])
            if len(A) < 4:
                P_(f'    {nm:<22}{kn:<12}{"—":>10}{"(n < 4)":>22}{len(A):>5}')
                continue
            val = icc21(np.array(A), np.array(B)); lo, hi = boot_icc(A, B)
            flag = '' if len(A) >= FLOOR else '  below the floor'
            P_(f'    {nm:<44}{kn:<16}{val:>+10.3f}   [{lo:+.2f}, {hi:+.2f}]{len(A):>5}{flag}')
            rows.append(dict(item=f'F4 ICC {key} {kn} - {lab}',
                             value=f'{val:+.3f} [{lo:+.2f}, {hi:+.2f}]', criterion='no threshold',
                             verdict=f'n = {len(A)}' + (' - below the floor' if len(A) < FLOOR else '')))
    P_()

P_('  Note. Alpha follows the definition of registration 18.4 (the mean of the flat log10 spectrum over 8-13 Hz')
P_('        after removing the aperiodic component). It is a different quantity from the peak amplitude (7-14 Hz) the manuscript used in the correlations.')

os.makedirs(str(OUT), exist_ok=True)
dst = os.path.join(str(OUT), 'feasibility_prereg.csv')
with open(dst, 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=['item', 'value', 'criterion', 'verdict'])
    w.writeheader(); w.writerows(rows)
open(os.path.join(str(OUT), 'feasibility_prereg.txt'), 'w',
     encoding='utf-8').write('\n'.join(L))
print(f'\nwritten -> {dst}')
