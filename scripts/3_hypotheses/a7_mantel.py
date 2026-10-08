# -*- coding: utf-8 -*-
"""A7 - behaviour <-> EEG correspondence (preregistration 36.9, Mantel).

The procedure is exactly as in section 36.9.
  1) z-score the variables within a block across participants
  2) build a participant x participant matrix of Euclidean distances
  3) Spearman between the upper-triangle elements of the two matrices
  4) reference distribution from 10,000 permutations of the participant labels
  5) computed only when both blocks have complete data for n >= 8

Blocks
  behaviour  perseverative errors, regressive errors, lose-shift, win-stay, switch rate, median reaction time
  model      kappa, eta_unrew - the two parameters that cleared the recovery criterion (.60) of section 32.3.1
             (kappa +.924 and eta_unrew +.666 pass; eta_rew +.253 and beta +.410 do not)
             the model was not identified in the real data, so the values are not interpreted (Appendix 2)
  EEG        aperiodic exponent, periodic alpha power (the definition of 18.4: the mean of the
             flat spectrum over 8-13 Hz), for each of the four clusters of 18.6

Exclusions apply all four criteria of preregistration 35.3 (cohort.prereg_keep).
007 (RT<150ms 24.2%), 008 (win-stay .595) and 009 (win-stay .538) drop out.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, OUT

import json, csv, glob, warnings
import numpy as np
from scipy import stats
from fooof import FOOOF
warnings.filterwarnings('ignore')

D = str(DATA)
RAWD = os.path.join(D, 'raw')
PA = json.load(open(os.path.join(D, 'psd_all.json'), encoding='utf-8'))
BM = json.load(open(os.path.join(D, 'behav_metrics.json'), encoding='utf-8'))

CLUS = {
    'Frontal (AF3/AF4/F3/F4)':          ['AF3','AF4','F3','F4'],
    'Parieto-occipital (O1/O2/P7/P8)':  ['O1','O2','P7','P8'],
    'Temporo-parietal (T7/T8/P7/P8)':   ['T7','T8','P7','P8'],
    'Global (14 channels)':             ['AF3','F7','F3','FC5','T7','P7','O1',
                               'O2','P8','T8','FC6','F4','F8','AF4'],
}

def median_rt(pid):
    for p in glob.glob(os.path.join(RAWD, '*_behav.csv')):
        if os.path.basename(p).split('_')[-2] != pid:
            continue
        v = []
        for r in csv.DictReader(open(p, encoding='utf-8-sig')):
            if not r['choice']:
                continue
            try:
                v.append(float(r['rt']))
            except (TypeError, ValueError):
                pass
        return float(np.median(v)) if v else None
    return None

def eeg_vals(pid, chs):
    """Fit each run then average (the same order as the manuscript and spectro.py)."""
    d = PA.get(pid)
    if not d:
        return None
    f = np.array(d['f']); have = d['ch']
    ii = [have.index(c) for c in chs if c in have]
    if not ii:
        return None
    exps, alphas = [], []
    for k in ('rest1', 'rest2'):
        if k not in d:
            continue
        fm = FOOOF(peak_width_limits=[1,8], max_n_peaks=6, min_peak_height=0.05,
                   peak_threshold=2.0, verbose=False)
        try:
            fm.fit(f, np.array(d[k])[ii].mean(0), [2, 40])
        except Exception:
            continue
        if fm.r_squared_ < 0.85:
            continue
        flat = fm.power_spectrum - fm._ap_fit
        m = (fm.freqs >= 8) & (fm.freqs <= 13)
        exps.append(float(fm.aperiodic_params_[-1]))
        alphas.append(float(flat[m].mean()))
    if not exps:
        return None
    return [float(np.mean(exps)), float(np.mean(alphas))]

BEH_KEYS = ['n_persev', 'n_regress', 'lose_shift', 'win_stay', 'switch_rate']

def dist(Z):
    n = len(Z)
    return np.array([[np.linalg.norm(Z[i] - Z[j]) for j in range(n)] for i in range(n)])

def mantel(A, B, n_perm=10000, seed=0):
    n = len(A); iu = np.triu_indices(n, 1)
    r = stats.spearmanr(A[iu], B[iu]).statistic
    rng = np.random.default_rng(seed); c = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        if stats.spearmanr(A[iu], B[np.ix_(p, p)][iu]).statistic >= r:
            c += 1
    return float(r), (c + 1) / (n_perm + 1)

# Candidate participants - all four criteria of preregistration 35.3
# An earlier version applied win-stay only, which left 007 in although it fails the reaction-time criterion.
import cohort
cand = [p for p in cohort.prereg_keep(str(DATA)) if p in BM]
excl = [p for p in sorted(BM) if p not in cand]
print(f"candidates with behavioural data: {len(cand)}: {cand}\n")

print('=' * 84)
print(' A7 - behaviour <-> EEG Mantel (preregistration 36.9)')
print('=' * 84)
rows = []
for cname, chs in CLUS.items():
    use, Bh, Eg = [], [], []
    for p in cand:
        e = eeg_vals(p, chs)
        rt = median_rt(p)
        if e is None or rt is None:
            continue
        use.append(p)
        Bh.append([BM[p][k] for k in BEH_KEYS] + [rt])
        Eg.append(e)
    n = len(use)
    if n < 8:
        print(f"\n[{cname}]  n = {n} - below the n >= 8 of section 36.9, not computed")
        rows.append(dict(cluster=cname, n=n, rho='', P='', note='n<8, not computed'))
        continue
    Z = lambda X: (np.array(X) - np.array(X).mean(0)) / np.array(X).std(0, ddof=1)
    A = dist(Z(Bh)); B = dist(Z(Eg))
    r, p = mantel(A, B)
    print(f"\n[{cname}]  n = {n}  ({', '.join(use)})")
    print(f"   Mantel rho = {r:+.3f} - permutation P = {p:.4f}  (10,000)")
    rows.append(dict(cluster=cname, n=n, rho=round(r,4), P=round(p,4), note=''))

# -- model block - the remaining two comparisons of section 36.9 ---
print('\n' + '=' * 84)
print(' model block - behaviour <-> model and model <-> EEG')
print('=' * 84)
print("  Section 36.9 requires the model block to be restricted to parameters that cleared recovery.")
print("  In the recovery check kappa (+.924) and eta_unrew (+.666) exceed the .60 of section 32.3.1")
print("  while eta_rew (+.253) and beta (+.410) do not. The block is therefore kappa and eta_unrew.")
print()
print("  Warning. The real-data estimates of these two parameters are not trustworthy. The learning")
print("  rate converged to a boundary and the inverse temperature diverged, so the model was not")
print("  identified (Appendix 2). The values below are produced so that the registered")
print("  computation is not skipped; they are not for interpretation.")
print()
try:
    RL = json.load(open(os.path.join(D, 'rlfit.json'), encoding='utf-8'))
except FileNotFoundError:
    RL = {}
MOD_KEYS = ['kappa', 'eu']          # the parameters that cleared recovery
Zf = lambda X: (np.array(X) - np.array(X).mean(0)) / np.array(X).std(0, ddof=1)
if not RL:
    print("  rlfit.json is missing, so this cannot be computed.")
    rows.append(dict(cluster='behaviour <-> model', n=0, rho='', P='', note='rlfit.json missing'))
else:
    use_m, Bh_m, Md = [], [], []
    for p in cand:
        f_ = (RL.get(p) or {}).get('full') or {}
        rt = median_rt(p)
        if rt is None or any(f_.get(k) is None for k in MOD_KEYS):
            continue
        use_m.append(p)
        Bh_m.append([BM[p][k] for k in BEH_KEYS] + [rt])
        Md.append([float(f_[k]) for k in MOD_KEYS])
    if len(use_m) < 8:
        print(f"  behaviour <-> model  n = {len(use_m)} - below the n >= 8 of section 36.9, not computed")
        rows.append(dict(cluster='behaviour <-> model', n=len(use_m), rho='', P='',
                         note='n<8, not computed'))
    else:
        r_, p_ = mantel(dist(Zf(Bh_m)), dist(Zf(Md)))
        print(f"  behaviour <-> model  n = {len(use_m)}  Mantel rho = {r_:+.3f} - permutation P = {p_:.4f}")
        rows.append(dict(cluster='behaviour <-> model', n=len(use_m), rho=round(r_, 4),
                         P=round(p_, 4), note='model not identified - not interpretable'))
    for cname, chs in CLUS.items():
        use2, Md2, Eg2 = [], [], []
        for p in cand:
            f_ = (RL.get(p) or {}).get('full') or {}
            e = eeg_vals(p, chs)
            if e is None or any(f_.get(k) is None for k in MOD_KEYS):
                continue
            use2.append(p); Md2.append([float(f_[k]) for k in MOD_KEYS]); Eg2.append(e)
        if len(use2) < 8:
            print(f"  model <-> EEG - {cname}  n = {len(use2)} - below n >= 8, not computed")
            rows.append(dict(cluster=f'model <-> EEG - {cname}', n=len(use2), rho='',
                             P='', note='n<8, not computed'))
            continue
        r_, p_ = mantel(dist(Zf(Md2)), dist(Zf(Eg2)))
        print(f"  model <-> EEG - {cname}  n = {len(use2)}  "
              f"Mantel rho = {r_:+.3f} - permutation P = {p_:.4f}")
        rows.append(dict(cluster=f'model <-> EEG - {cname}', n=len(use2),
                         rho=round(r_, 4), P=round(p_, 4), note='model not identified - not interpretable'))

print()
print("  Caution from section 36.9. The permutation P describes where the observed value falls in")
print("  the permutation distribution; it is not a decision threshold. No comparison is called")
print("  significant or non-significant.")
print("  Participants generate many pairwise distances, but the number of independent units is")
print("  still the number of participants.")

out = f'{OUT}/A7_mantel.csv'
with open(out, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['cluster','n','rho','P','note'])
    w.writeheader(); w.writerows(rows)
print(f"\nwritten: {out}")
