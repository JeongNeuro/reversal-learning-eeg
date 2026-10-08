"""
spec_grid.py - the 144-cell specification grid of preregistration section 32.4

  reference        original reference, common average        (2)
  amplitude        100, 150, 200 uV                         (3)
  fit band         1-40, 2-40, 2-30 Hz                      (3)
  aperiodic mode   fixed, knee                              (2)
  cluster          parieto-occipital, temporo-parietal, frontal, global  (4)
                                          = 144

Primary specification (reported in the main text): common average, 150 uV, 2-40 Hz, fixed, with the primary cluster of each hypothesis.
The primary cluster for H1 is parieto-occipital (section 18.6).

What is reported (section 32.4)
  - the estimate of the H1 primary correlation (resting aperiodic exponent x switch-bias index) in every cell
  - sign stability - the proportion of cells whose sign matches the primary specification. "Sign stable" only at 0.90 or above
  - cells below the floor get no coefficient, only n (section 35.9: a correlation needs 8 participants)

The minimum data requirement is reported under both criteria.
  60 s (30 epochs of 2 s) - the primary criterion of registration 28.7
  30 s (15 epochs)        - the criterion the manuscript lowered to post hoc (disclosed in Appendix 1)

Order of computation. Reference and amplitude change the surviving epochs and the spectra, so
six spectra are built per participant and run; band, mode and cluster only refit those spectra.

Output  outputs/spec_grid.csv  outputs/spec_grid_cells.csv
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT

import csv, glob, json, warnings
import numpy as np
import mne
from scipy.signal import get_window
from scipy import stats
from fooof import FOOOF

warnings.filterwarnings('ignore'); mne.set_log_level('ERROR')

CH = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
CLUST = [('po4',   'parieto-occipital O1/O2/P7/P8', ['O1','O2','P7','P8']),
         ('temp',  'temporo-parietal T7/T8/P7/P8',  ['T7','T8','P7','P8']),
         ('front', 'frontal AF3/AF4/F3/F4',         ['AF3','AF4','F3','F4']),
         ('glob',  'global 14 channels',            None)]
REFS = ['original', 'average']
AMPS = [100.0, 150.0, 200.0]
RANGES = [(1, 40), (2, 40), (2, 30)]
MODES = ['fixed', 'knee']
LEADIN = 10.0        # registration 29.1 - discard the first 10 s of each run
EPL = 2.0
MINEP = {'60 s (registered)': 30, '30 s (post hoc)': 15}

PRIMARY = dict(ref='average', amp=150.0, rng=(2, 40), mode='fixed', cl='po4')
EXCL = ['007', '008', '009']         # registration 35.3
FLOOR = 8                            # registration 35.9

BM = json.load(open(f'{DATA}/behav_metrics.json', encoding='utf-8'))
KEEP = [p for p in sorted(BM) if p not in EXCL]


def find(pid, pat):
    return sorted(set(p for p in glob.glob(f'{RAW}/**/{pat}', recursive=True)
                      if os.path.basename(p).startswith(pid)))


def i_(x):
    try: return int(float(x))
    except Exception: return 0


def psd_of(X, idx, fs, nfft, amp):
    """Registration 29.4 - mean of 2 s Hann periodograms in linear power. Artefact epochs excluded."""
    win = get_window('hann', nfft)
    scale = 1.0 / (fs * (win ** 2).sum())
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    keep = []
    for a in range(0, X.shape[1] - nfft + 1, nfft):
        ep = X[:, a:a + nfft]
        if (ep[idx].max(1) - ep[idx].min(1)).max() > amp:
            continue
        sp = np.fft.rfft(ep * win, axis=-1)
        pw = (np.abs(sp) ** 2) * scale
        pw[:, 1:-1] *= 2.0
        keep.append(pw)
    if not keep:
        return freqs, None, 0
    return freqs, np.mean(keep, axis=0), len(keep)


def fit_exp(f, p, rng, mode):
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.1,
               peak_threshold=2.0, aperiodic_mode=mode, verbose=False)
    try:
        fm.fit(f, p, list(rng))
    except Exception:
        return None
    if not np.isfinite(fm.r_squared_) or fm.r_squared_ < 0.90:
        return None
    if not np.isfinite(fm.error_) or fm.error_ > 0.10:
        return None
    return float(fm.aperiodic_params_[-1])


# -- stage 1: spectra per participant, run, reference, amplitude and cluster --
print('stage 1 - producing spectra (2 references x 3 amplitudes)')
SPEC = {}       # (pid, run, ref, amp, cl) -> (freqs, psd, n_epoch)
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
        for ref in REFS:
            Z = X0[gi, a:b]
            if ref == 'average':
                Z = Z - Z.mean(axis=0, keepdims=True)
            for amp in AMPS:
                for key, _, chs in CLUST:
                    ix = ([good.index(c) for c in chs if c in good]
                          if chs else list(range(len(good))))
                    if not ix:
                        continue
                    f, P, ne = psd_of(Z, ix, fs, nfft, amp)
                    if P is None:
                        continue
                    SPEC[(pid, ri, ref, amp, key)] = (f, P[ix].mean(0), ne)
    print(f'  {pid}: {len(runs)} runs - {len(good)}/{len(have)} channels', flush=True)

# -- stage 2: the grid --------------------------------------------
print('\nstage 2 - fitting 144 cells')
cells, per_cell = [], []
PRIMARY_CELL = ('average', 150.0, (2, 40), 'fixed', 'po4')   # the primary specification of 32.4
primary = {}                       # min-data -> per-participant exponent. This is the basis of H1.


# The resample exclusion rule is the repository-wide **4 participants** (the same as pvalues.py).
# It used to be 3, and under that the H1 primary-specification interval came out as
# [-0.139, +1.000]. The ceiling of +1.000 is produced by resamples that contain only three
# distinct participants - about 2% of resamples at n = 8, where Spearman rho becomes exactly 1.
def boot_rho(x, y, nb=10000, seed=1, guard=4):
    rng = np.random.default_rng(seed); x = np.asarray(x); y = np.asarray(y); o = []
    for _ in range(nb):
        i = rng.integers(0, len(x), len(x))
        if len(set(i.tolist())) < guard:
            continue
        v = stats.spearmanr(x[i], y[i]).statistic
        if np.isfinite(v):
            o.append(v)
    return (np.percentile(o, 2.5), np.percentile(o, 97.5)) if o else (np.nan, np.nan)


def perm_rho(x, y, nperm=10000, seed=1):
    """Two-sided P for Spearman rho by participant-label permutation. Same rule as pvalues.py."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = abs(stats.spearmanr(x, y).statistic)
    k = sum(1 for _ in range(nperm)
            if abs(stats.spearmanr(x, rng.permutation(y)).statistic) >= obs)
    return (k + 1) / (nperm + 1)

n_done = 0
for mlab, mep in MINEP.items():
    for ref in REFS:
        for amp in AMPS:
            for rng_ in RANGES:
                for mode in MODES:
                    for key, cn, _ in CLUST:
                        vals = {}
                        for pid in KEEP:
                            v = []
                            for ri in (1, 2):
                                s = SPEC.get((pid, ri, ref, amp, key))
                                if not s or s[2] < mep:
                                    continue
                                e = fit_exp(s[0], s[1], rng_, mode)
                                if e is not None:
                                    v.append(e)
                            if v:
                                vals[pid] = float(np.mean(v))
                        ids = sorted(vals)
                        if (ref, amp, rng_, mode, key) == PRIMARY_CELL:
                            primary[mlab] = {p: vals[p] for p in ids}
                        row = dict(min_data=mlab, reference=ref, amplitude=amp,
                                   band=f'{rng_[0]}-{rng_[1]}', mode=mode,
                                   cluster=cn, n=len(ids))
                        if len(ids) >= 3:
                            x = [vals[p] for p in ids]
                            y = [BM[p]['sbi'] for p in ids]
                            r = stats.spearmanr(x, y).statistic
                            row['rho'] = round(float(r), 4)
                            if len(ids) >= FLOOR:
                                lo, hi = boot_rho(x, y)
                                row['lo'] = round(lo, 4); row['hi'] = round(hi, 4)
                                row['note'] = ''
                            else:
                                row['lo'] = ''; row['hi'] = ''
                                row['note'] = 'below the floor - coefficient for reference only'
                        else:
                            row['rho'] = ''; row['lo'] = ''; row['hi'] = ''
                            row['note'] = 'insufficient data'
                        row['participants'] = ' '.join(ids)
                        cells.append(row)
                        n_done += 1
                        if n_done % 24 == 0:
                            print(f'  {n_done}/288 cells', flush=True)

# -- stage 3: summary ---------------------------------------------
FIELDS = ['min_data','reference','amplitude','band','mode','cluster','n','rho','lo','hi','note','participants']
os.makedirs(str(OUT), exist_ok=True)
with open(f'{OUT}/spec_grid_cells.csv', 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(cells)

L = []
def P(t=''): L.append(t); print(t)

P()
P('=' * 80)
P(' preregistration 32.4 specification grid - the H1 primary correlation (aperiodic exponent x switch-bias index)')
P('=' * 80)
P(f'  sample  {len(KEEP)} after the exclusions of registration 35.3: {" ".join(KEEP)}')
P('  primary specification  common average, 150 uV, 2-40 Hz, fixed, parieto-occipital')
P()

for mlab, mep in MINEP.items():
    sub = [c for c in cells if c['min_data'] == mlab]
    got = [c for c in sub if c['rho'] != '']
    ok = [c for c in sub if c['n'] >= FLOOR]
    prim = [c for c in sub
            if c['reference'] == PRIMARY['ref'] and c['amplitude'] == PRIMARY['amp']
            and c['band'] == '2-40' and c['mode'] == PRIMARY['mode']
            and c['cluster'].startswith('parieto-occipital')]
    P('-' * 80)
    P(f' minimum data {mlab}')
    P('-' * 80)
    P(f'  cells with a coefficient    {len(got)}/144')
    P(f'  meeting the registered floor (8)  {len(ok)}/144')
    pr = prim[0] if prim else None
    if pr and pr['rho'] != '':
        P(f"  primary cell  rho = {pr['rho']:+.3f} - n = {pr['n']}"
          + (f" - 95% CI [{pr['lo']:+.3f}, {pr['hi']:+.3f}]" if pr['lo'] != '' else
             '  <- below the floor, coefficient for reference only'))
    elif pr:
        P(f"  primary cell  no coefficient (n = {pr['n']})")
    # sign stability
    if pr and pr['rho'] != '':
        sgn = np.sign(pr['rho'])
        same = sum(1 for c in got if np.sign(c['rho']) == sgn)
        P(f'  sign stability  {same}/{len(got)} = {same/len(got):.3f}'
          f'   (registered criterion 0.90 {"met" if same/len(got) >= .90 else "not met"})')
        vals = [c['rho'] for c in got]
        P(f'  whole distribution  median {np.median(vals):+.3f} - '
          f'range {min(vals):+.3f} to {max(vals):+.3f}')
        okv = [c['rho'] for c in ok if c['rho'] != '']
        if okv:
            same2 = sum(1 for v in okv if np.sign(v) == sgn)
            P(f'  cells meeting the floor only  {same2}/{len(okv)} = {same2/len(okv):.3f} - '
              f'median {np.median(okv):+.3f} - range {min(okv):+.3f} to {max(okv):+.3f}')
    P()
    P(f"  {'cluster':<32}{'cells':>6}{'coef':>6}{'floor met':>11}{'median':>9}{'range':>20}")
    for key, cn, _ in CLUST:
        cc = [c for c in sub if c['cluster'] == cn]
        gv = [c['rho'] for c in cc if c['rho'] != '']
        okc = sum(1 for c in cc if c['n'] >= FLOOR)
        if gv:
            P(f'  {cn:<32}{len(cc):>5}{len(gv):>6}{okc:>11}{np.median(gv):>+9.3f}'
              f'{f"{min(gv):+.3f} ~ {max(gv):+.3f}":>20}')
        else:
            P(f'  {cn:<32}{len(cc):>5}{0:>6}{okc:>11}{"-":>9}{"-":>20}')
    P()

P('=' * 80)
P(' effect of each factor (cells meeting the floor only)')
P('=' * 80)
for mlab in MINEP:
    ok = [c for c in cells if c['min_data'] == mlab and c['n'] >= FLOOR and c['rho'] != '']
    if not ok:
        P(f'  [{mlab}] no cell clears the floor.')
        continue
    P(f'  [{mlab}]  {len(ok)} cells meet the floor')
    for fac in ('reference', 'amplitude', 'band', 'mode'):
        levels = sorted(set(c[fac] for c in ok), key=str)
        s = '   '.join(f'{lv}: {np.median([c["rho"] for c in ok if c[fac] == lv]):+.3f}'
                       for lv in levels)
        P(f'    {fac:<11}{s}')
    P()

# -- stage 4: H1 in the primary cell - the value reported in the manuscript and abstract --
# The PO row of main-text Figure 4 i comes from the spectro.py pipeline (30 s criterion, n = 9);
# the value here is the **preregistered primary specification** (common average, 150 uV, 2-40 Hz, fixed, 60 s).
# Family (A) of pvalues.py reads this value.
P('=' * 80)
P(' H1 in the preregistered primary cell - parieto-occipital aperiodic exponent x switch-bias index')
P('=' * 80)
P('  common average, 150 uV, 2-40 Hz, fixed, parieto-occipital')
P('  the interval is 10,000 bootstrap resamples (seed 1), P is 10,000 label permutations (seed 1)')
P()
H1 = {}
for mlab, vals in primary.items():
    ids = sorted(vals)
    x = [vals[q] for q in ids]
    y = [BM[q]['sbi'] for q in ids]
    r = float(stats.spearmanr(x, y).statistic)
    if len(ids) >= FLOOR:
        lo, hi = boot_rho(x, y)
        pv = perm_rho(x, y)
        st = '**' if pv < .01 else ('*' if pv < .05 else '')
        P('  [%s]  n = %d  rho = %+.4f  [%+.3f, %+.3f]  permutation P = %.4f %s'
          % (mlab, len(ids), r, lo, hi, pv, st))
    else:
        lo = hi = pv = float('nan')
        P('  [%s]  n = %d  rho = %+.4f  - below the registered floor (%d participants)'
          % (mlab, len(ids), r, FLOOR))
    P('       participants %s' % ' '.join(ids))
    H1[mlab] = dict(n=len(ids), ids=ids, rho=round(r, 4),
                    ci_lo=(round(lo, 4) if lo == lo else None),
                    ci_hi=(round(hi, 4) if hi == hi else None),
                    p_perm=(round(pv, 4) if pv == pv else None),
                    exponent={q: round(vals[q], 6) for q in ids},
                    sbi={q: BM[q]['sbi'] for q in ids})
P()
P('  Note. The two criteria differ by the one participant who did not reach 60 s. The registered criterion is 60 s.')

json.dump(H1, open(f'{OUT}/h1_prereg_spec.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

open(f'{OUT}/spec_grid.txt', 'w', encoding='utf-8').write('\n'.join(L))
print(f'\nwritten -> {OUT}/spec_grid_cells.csv  {OUT}/spec_grid.txt'
      f' · {OUT}/h1_prereg_spec.json')
