# -*- coding: utf-8 -*-
"""A4 and A5 - recomputed under the preregistered definitions, with the aggregation order matched to the manuscript.

The manuscript (cluster_value of spectro.py) **fits each run and then averages the values**.
The same order is used here, so that the only things that change are the alpha definition and the cluster.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, OUT

import json, warnings, csv
import numpy as np
from scipy import stats
from fooof import FOOOF
warnings.filterwarnings('ignore')

D = str(DATA)
PA = json.load(open(os.path.join(D, 'psd_all.json'), encoding='utf-8'))
BM = json.load(open(os.path.join(D, 'behav_metrics.json'), encoding='utf-8'))

CLUS = {
    'Frontal (AF3/AF4/F3/F4)':          ['AF3','AF4','F3','F4'],
    'Parieto-occipital (O1/O2/P7/P8)':  ['O1','O2','P7','P8'],
    'Temporo-parietal (T7/T8/P7/P8)':   ['T7','T8','P7','P8'],
    'Global (14 channels)':             ['AF3','F7','F3','FC5','T7','P7','O1',
                                   'O2','P8','T8','FC6','F4','F8','AF4'],
}
S9 = ['001','002','003','004','005','006','012','014','016']

def fit_one(f, P):
    fm = FOOOF(peak_width_limits=[1,8], max_n_peaks=6, min_peak_height=0.05,
               peak_threshold=2.0, verbose=False)
    try:
        fm.fit(f, P, [2, 40])
    except Exception:
        return None
    if fm.r_squared_ < 0.85:
        return None
    pk = [q for q in fm.peak_params_ if 7 <= q[0] < 14]
    flat = fm.power_spectrum - fm._ap_fit
    m = (fm.freqs >= 8) & (fm.freqs <= 13)
    return dict(exponent=float(fm.aperiodic_params_[-1]),
                alpha_peak=(float(max(pk, key=lambda q: q[1])[1]) if len(pk) else None),
                alpha_flat=float(flat[m].mean()))

def cluster_value(pid, chs):
    """Fit each run, then average the values (the same order as cluster_value of spectro.py)."""
    d = PA.get(pid)
    if not d:
        return None
    f = np.array(d['f']); have = d['ch']
    ii = [have.index(c) for c in chs if c in have]
    if not ii:
        return None
    acc = []
    for k in ('rest1', 'rest2'):
        if k not in d:
            continue
        r = fit_one(f, np.array(d[k])[ii].mean(0))
        if r:
            acc.append(r)
    if not acc:
        return None
    out = {}
    for key in ('exponent', 'alpha_peak', 'alpha_flat'):
        v = [a[key] for a in acc if a[key] is not None]
        out[key] = float(np.mean(v)) if v else None
    return out

FIT = {c: {p: cluster_value(p, chs) for p in S9} for c, chs in CLUS.items()}
rho = lambda x, y: float(stats.spearmanr(x, y).statistic)
BEH = [('lose_shift','lose-shift'), ('sbi','switch-bias index'), ('n_persev','perseverative errors'),
       ('accuracy','accuracy'), ('trials_to_recover','trials to recover')]

PAPER = {'lose_shift': .803, 'sbi': .717, 'n_persev': -.723}
PAPER_A = {'accuracy': .767, 'trials_to_recover': -.850}

rows = []
print('== check: does the frontal cluster reproduce the manuscript values ==')
ok = [p for p in S9 if FIT['Frontal (AF3/AF4/F3/F4)'][p]]
E = [FIT['Frontal (AF3/AF4/F3/F4)'][p]['exponent'] for p in ok]
A = [FIT['Frontal (AF3/AF4/F3/F4)'][p]['alpha_peak'] for p in ok]
for k, l in BEH:
    y = [BM[p][k] for p in ok]
    tgt = PAPER.get(k); tgta = PAPER_A.get(k)
    s1 = f"  exponent x {l:22s} {rho(E,y):+.3f}" + (f"  manuscript {tgt:+.3f}" if tgt else "")
    s2 = f"   alpha x {l:22s} {rho(A,y):+.3f}" + (f"  manuscript {tgta:+.3f}" if tgta else "")
    print(s1 + s2)

print('\n' + '='*96)
print(' A5 - aperiodic exponent by cluster x behaviour')
print('='*96)
print(f"{'cluster':36s}{'n':>4}" + ''.join(f'{l:>26}' for _, l in BEH))
for cname in CLUS:
    ok = [p for p in S9 if FIT[cname][p]]
    E = [FIT[cname][p]['exponent'] for p in ok]
    line = f"{cname:26s}{len(ok):>4}"
    for k, l in BEH:
        r = rho(E, [BM[p][k] for p in ok]); line += f"{r:>+14.3f}"
        rows.append(dict(section='A5 aperiodic exponent', cluster=cname, alpha_definition='-',
                         n=len(ok), behaviour=l, rho=round(r,4)))
    print(line)

print('\n' + '='*96)
print(' A4 - by definition of the alpha index')
print('='*96)
for defn, key in [('peak amplitude (a change from manuscript 18.4)', 'alpha_peak'),
                  ('mean of the flat spectrum over 8-13 Hz (preregistration 18.4)', 'alpha_flat')]:
    print(f"\n [{defn}]")
    print(f"{'cluster':36s}{'n':>4}" + ''.join(f'{l:>26}' for _, l in BEH))
    for cname in CLUS:
        ok = [p for p in S9 if FIT[cname][p] and FIT[cname][p][key] is not None]
        if len(ok) < 4:
            print(f"{cname:36s}{len(ok):>4}   sample too small"); continue
        A = [FIT[cname][p][key] for p in ok]
        line = f"{cname:26s}{len(ok):>4}"
        for k, l in BEH:
            r = rho(A, [BM[p][k] for p in ok]); line += f"{r:>+14.3f}"
            rows.append(dict(section='A4 alpha', cluster=cname, alpha_definition=defn,
                             n=len(ok), behaviour=l, rho=round(r,4)))
        print(line)

print('\n== missing data ==')
for cname in CLUS:
    nofit = [p for p in S9 if not FIT[cname][p]]
    nopk = [p for p in S9 if FIT[cname][p] and FIT[cname][p]['alpha_peak'] is None]
    print(f"  {cname:36s} fit failed {nofit or 'none'} - no peak detected {nopk or 'none'}")

out = f'{OUT}/prereg_A4A5.csv'
with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['section','cluster','alpha_definition','n','behaviour','rho'])
    w.writeheader(); w.writerows(rows)
print(f'\nwritten: {out} ({len(rows)} rows)')
