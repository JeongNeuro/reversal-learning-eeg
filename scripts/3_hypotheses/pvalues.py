# -*- coding: utf-8 -*-
"""pvalues.py - test method and P value for every result reported with an interval

Following preregistration section 34, the manuscript reported estimates and intervals
only. Under that scheme no multiple-comparison correction was needed, because no
threshold decision was being made. Once P values and asterisks are printed alongside,
**dozens of uncorrected threshold decisions appear.** This script also counts them.

The test follows the structure of the data.

  correlation       participant-label permutation (10,000, seed 1). With n = 7-10 the
                    asymptotic approximation is not used
  partial           y is permuted with the covariate held fixed (20,000, seed 1)
  difference of two Williams' test for dependent correlations (Steiger 1980)
  paired            Wilcoxon signed-rank (exact) and the paired t, both reported
  mixed model       the model's Wald test

Output  outputs/pvalues.csv  outputs/pvalues.txt  outputs/pvalues.json
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
import cohort                                               # noqa: E402

SEED = 1
NPERM = 10000
NPERM_P = 20000
ALPHA_STAR = [(.01, '**'), (.05, '*')]

ROWS = []
LOG = []


def w(t=''):
    LOG.append(t)
    print(t)


def star(p):
    if p is None or not np.isfinite(p):
        return ''
    for thr, s in ALPHA_STAR:
        if p < thr:
            return s
    return ''


def add(group, label, n, est, ci, method, p, note=''):
    # Store five decimals. Storing four and then rounding again in the table
    # produces double rounding (the floor of Figure 5c Posterior is -0.08147,
    # hence -0.081, not -0.082 by way of -0.0815).
    ROWS.append(dict(group=group, label=label, n=n,
                     estimate=(None if est is None else round(float(est), 5)),
                     ci_lo=(None if ci is None else round(float(ci[0]), 5)),
                     ci_hi=(None if ci is None else round(float(ci[1]), 5)),
                     method=method,
                     p=(None if p is None else round(float(p), 4)),
                     star=star(p), note=note))


def boot_mean(x, nb=NPERM, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    return np.percentile([x[rng.integers(0, len(x), len(x))].mean()
                          for _ in range(nb)], [2.5, 97.5])


def boot_rho(x, y, nb=NPERM, seed=SEED, guard=4):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = []
    for _ in range(nb):
        k = rng.integers(0, len(x), len(x))
        if len(set(k.tolist())) < guard:
            continue
        v = stats.spearmanr(x[k], y[k]).statistic
        if np.isfinite(v):
            o.append(v)
    return np.percentile(o, [2.5, 97.5])


def perm_rho(x, y, nperm=NPERM, seed=SEED):
    """Two-sided P for Spearman rho by participant-label permutation."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = abs(stats.spearmanr(x, y).statistic)
    k = sum(1 for _ in range(nperm)
            if abs(stats.spearmanr(x, rng.permutation(y)).statistic) >= obs)
    return (k + 1) / (nperm + 1)


def pcorr(x, y, c):
    """Partial correlation after rank transform - same definition as panel j of fig4_full.py."""
    rx, ry, rc = stats.rankdata(x), stats.rankdata(y), stats.rankdata(c)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxc = np.corrcoef(rx, rc)[0, 1]
    ryc = np.corrcoef(ry, rc)[0, 1]
    return (rxy - rxc * ryc) / np.sqrt((1 - rxc ** 2) * (1 - ryc ** 2))


def perm_pcorr(x, y, c, nperm=NPERM_P, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = abs(pcorr(x, y, c))
    k = sum(1 for _ in range(nperm)
            if abs(pcorr(x, rng.permutation(y), c)) >= obs)
    return (k + 1) / (nperm + 1)


def williams(rjk, rjh, rkh, n):
    """Difference of two dependent correlations sharing variable j - Williams' t (Steiger 1980)."""
    if n < 5:
        return np.nan, np.nan
    R = (1 - rjk ** 2 - rjh ** 2 - rkh ** 2) + 2 * rjk * rjh * rkh
    rbar = (rjk + rjh) / 2
    den = (2 * (n - 1) / (n - 3) * R
           + rbar ** 2 * (1 - rkh) ** 3)
    if den <= 0:
        return np.nan, np.nan
    t = (rjk - rjh) * np.sqrt((n - 1) * (1 + rkh) / den)
    return t, 2 * stats.t.sf(abs(t), n - 3)


def t_ci(d):
    """t interval of the paired difference. This is the interval the manuscript reports for W2 2-level."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 2:
        return (np.nan, np.nan)
    se = d.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se
    return (d.mean() - h, d.mean() + h)


def paired(a, b):
    """Paired comparison - (mean difference, bootstrap interval, Wilcoxon P, paired t P, n, t interval).

    Two intervals are produced. Figure 5c and section 3.4 report the bootstrap
    interval; W2 2-level is reported with the t interval in the main text and in
    Appendix 3. Each call site picks one so that no result appears with two
    different intervals in the same manuscript.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    ci = boot_mean(d)
    try:
        pw = stats.wilcoxon(a, b).pvalue
    except Exception:
        pw = np.nan
    pt = stats.ttest_rel(b, a).pvalue if len(d) > 1 else np.nan
    return d.mean(), ci, pw, pt, len(d), t_ci(d)


# ---------- data ----------
beh, eeg = cohort.load(str(DATA), verbose=True)
IDS = [d['id'] for d in eeg]
B = {d['id']: d for d in eeg}
E = np.array([B[p]['front_exp'] for p in IDS], float)
A = np.array([B[p]['front_alpha'] for p in IDS], float)

w('=' * 96)
w(' Test method and P value by result')
w('=' * 96)
w('  correlation n = %d (%s)' % (len(IDS), ' '.join(IDS)))
w('  permutation %d  partial permutation %d  seed %d' % (NPERM, NPERM_P, SEED))
w()

# -- Figure 4 d-h ------------------------------------------------
w('[Figure 4 d-h] resting-state index × behavioural index - Spearman, label permutation')
PAN = [('d  exponent × lose-shift', E, [B[p]['lose_shift'] * 100 for p in IDS]),
       ('e  exponent × switch bias', E, [B[p]['sbi'] for p in IDS]),
       ('f  exponent × perseverative errors', E, [B[p]['n_persev'] for p in IDS]),
       ('g  alpha × accuracy', A, [B[p]['accuracy'] * 100 for p in IDS]),
       ('h  alpha × trials to recover', A, [B[p]['trials_to_recover'] for p in IDS])]
for nm, x, y in PAN:
    y = np.asarray(y, float)
    r = stats.spearmanr(x, y).statistic
    ci = boot_rho(x, y)
    p = perm_rho(x, y)
    add('Figure 4 d-h', nm, len(IDS), r, ci, 'permutation (Spearman)', p)
    w('  %-36s rho %+.3f  [%+.3f, %+.3f]  P = %.4f %s'
      % (nm, r, ci[0], ci[1], p, star(p)))
w()

# -- Figure 4 i --------------------------------------------------
w('[Figure 4 i] aperiodic exponent by cluster × switch bias - Spearman, label permutation')
PS = json.load(io.open(os.path.join(str(DATA), 'psd_all.json'), encoding='utf-8'))
from fooof import FOOOF                                      # noqa: E402
CL = [('PO parieto-occipital', ['O1', 'O2', 'P7', 'P8']),
      ('Fr frontal', ['AF3', 'AF4', 'F3', 'F4']),
      ('Gl global', ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                     'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']),
      ('TP temporo-parietal', ['T7', 'T8', 'P7', 'P8'])]


def fit_exp(f, p):
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.05,
               peak_threshold=2.0, aperiodic_mode='fixed', verbose=False)
    try:
        fm.fit(f, p, [2.0, 40.0])
    except Exception:
        return None
    if not np.isfinite(fm.r_squared_) or fm.r_squared_ < 0.85:
        return None
    return float(fm.aperiodic_params_[-1])


def cluster_exp(pid, want):
    v = PS.get(pid)
    if not v:
        return None
    f = np.asarray(v['f'], float)
    ix = [v['ch'].index(c) for c in want if c in v['ch']]
    if not ix:
        return None
    out = [fit_exp(f, np.asarray(v[r], float)[ix].mean(0))
           for r in ('rest1', 'rest2') if v.get(r)]
    out = [q for q in out if q is not None]
    return float(np.mean(out)) if out else None


SBI = {p: B[p]['sbi'] for p in IDS}
for nm, want in CL:
    xs, ys = [], []
    for p in IDS:
        v = cluster_exp(p, want)
        if v is not None:
            xs.append(v)
            ys.append(SBI[p])
    if len(xs) < 4:
        continue
    r = stats.spearmanr(xs, ys).statistic
    ci = boot_rho(xs, ys)
    p_ = perm_rho(xs, ys)
    add('Figure 4 i', nm, len(xs), r, ci, 'permutation (Spearman)', p_)
    w('  %-22s n=%2d  rho %+.3f  [%+.3f, %+.3f]  P = %.4f %s'
      % (nm, len(xs), r, ci[0], ci[1], p_, star(p_)))
w()

# -- Figure 4 j --------------------------------------------------
w('[Figure 4 j] simple correlation, partial correlation, and their difference')
LS = np.array([B[p]['lose_shift'] for p in IDS], float)
AC = np.array([B[p]['accuracy'] for p in IDS], float)
JOBS = [('exp × lose-shift', E, LS, A), ('α × lose-shift', A, LS, E),
        ('exp × accuracy', E, AC, A), ('α × accuracy', A, AC, E)]
for nm, x, y, c in JOBS:
    r = stats.spearmanr(x, y).statistic
    ci = boot_rho(x, y)
    p_ = perm_rho(x, y)
    add('Figure 4 j', nm + ' (simple)', len(IDS), r, ci, 'permutation (Spearman)', p_)
    w('  %-18s simple   rho %+.3f  [%+.3f, %+.3f]  P = %.4f %s'
      % (nm, r, ci[0], ci[1], p_, star(p_)))
    pr = pcorr(x, y, c)
    pp = perm_pcorr(x, y, c)
    rng = np.random.default_rng(SEED)
    bo = []
    for _ in range(NPERM):
        k = rng.integers(0, len(x), len(x))
        if len(set(k.tolist())) < 4:
            continue
        try:
            v = pcorr(x[k], y[k], c[k])
        except Exception:
            continue
        if np.isfinite(v):
            bo.append(v)
    pci = np.percentile(bo, [2.5, 97.5])
    add('Figure 4 j', nm + ' (partial)', len(IDS), pr, pci, 'permutation (partial)', pp)
    w('  %-18s partial  rho %+.3f  [%+.3f, %+.3f]  P = %.4f %s'
      % ('', pr, pci[0], pci[1], pp, star(pp)))
for lab, ybeh in (('lose-shift', LS), ('accuracy', AC)):
    rjk = stats.spearmanr(E, ybeh).statistic      # exponent × behaviour
    rjh = stats.spearmanr(A, ybeh).statistic      # alpha × behaviour
    rkh = stats.spearmanr(E, A).statistic         # exponent × alpha
    t_, p_ = williams(rjk, rjh, rkh, len(IDS))
    rng = np.random.default_rng(SEED)
    bo = []
    for _ in range(NPERM):
        k = rng.integers(0, len(IDS), len(IDS))
        if len(set(k.tolist())) < 4:
            continue
        v = (stats.spearmanr(E[k], ybeh[k]).statistic
             - stats.spearmanr(A[k], ybeh[k]).statistic)
        if np.isfinite(v):
            bo.append(v)
    ci = np.percentile(bo, [2.5, 97.5])
    add('Figure 4 j', 'Δρ %s (exponent − alpha)' % lab, len(IDS),
        rjk - rjh, ci, 'Williams dependent-correlation test', p_,
        'the two correlations share the behavioural index')
    w('  Δρ %-13s %+.3f  [%+.3f, %+.3f]  t = %+.3f  P = %.4f %s'
      % (lab, rjk - rjh, ci[0], ci[1], t_, p_, star(p_)))
w()


# ---------- part 2 - Figure 5c, section 3.4, W1, W2 ----------
import pandas as pd                                          # noqa: E402
import statsmodels.formula.api as smf                        # noqa: E402

CH14 = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
        'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

# -- Figure 5 c - alpha PLV by region -----------------------------
w('[Figure 5 c] alpha phase synchronisation by region - paired comparison (task - rest)')
CONN = json.load(io.open(os.path.join(str(DATA), 'conn.json'), encoding='utf-8'))
IDXC = {c: i for i, c in enumerate(CH14)}
FRO = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8']
CEN = ['FC5', 'FC6', 'T7', 'T8']
POSTR = ['P7', 'P8', 'O1', 'O2']


def prs(a, b=None):
    if b is None:
        return [(IDXC[x], IDXC[y]) for k, x in enumerate(a) for y in a[k + 1:]]
    return [(min(IDXC[x], IDXC[y]), max(IDXC[x], IDXC[y]))
            for x in a for y in b]


REGS = [('Frontal', prs(FRO)), ('Central', prs(CEN)),
        ('Posterior', prs(POSTR)), ('Front-Post', prs(FRO, POSTR)),
        ('Global', prs(CH14))]


def alignc(pid, M):
    ch = CONN[pid]['ch']
    out = np.full((14, 14), np.nan)
    idx = {c: i for i, c in enumerate(ch)}
    for i, a in enumerate(CH14):
        for j, b in enumerate(CH14):
            if a in idx and b in idx:
                out[i, j] = M[idx[a]][idx[b]]
    return out


def plvm(pid, cond):
    ms = [alignc(pid, CONN[pid][cond][b]) for b in ('8-10', '10-12')
          if CONN[pid][cond].get(b)]
    return np.nanmean(ms, axis=0) if ms else None


cids = [p for p in sorted(CONN)
        if plvm(p, 'rest') is not None and plvm(p, 'task') is not None]
MRc = {p: plvm(p, 'rest') for p in cids}
MTc = {p: plvm(p, 'task') for p in cids}
for nm, pr in REGS:
    r = np.array([np.nanmean([MRc[p][i, j] for i, j in pr]) for p in cids])
    t = np.array([np.nanmean([MTc[p][i, j] for i, j in pr]) for p in cids])
    d, ci, pw, pt, n, _tci = paired(r, t)
    add('Figure 5 c', nm, n, d, ci, 'Wilcoxon signed-rank', pw,
        'paired t P = %.4f' % pt)
    w('  %-12s n=%2d  difference %+.3f  [%+.3f, %+.3f]  Wilcoxon P = %.4f %s  (t P = %.4f)'
      % (nm, n, d, ci[0], ci[1], pw, star(pw), pt))
w()

# -- 3.4 change in band power and periodic alpha ------------------
w('[3.4] rest -> task change - paired comparison')
BP = json.load(io.open(os.path.join(str(DATA), 'bandpow.json'), encoding='utf-8'))
bids = [p for p in sorted(BP) if BP[p].get('rest') and BP[p].get('task')]
for bd in ('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'):
    r = np.array([BP[p]['rest'][bd] for p in bids], float)
    t = np.array([BP[p]['task'][bd] for p in bids], float)
    d, ci, pw, pt, n, _tci = paired(r, t)
    add('3.4 band relative power', 'frontal %s' % bd, n, d, ci, 'Wilcoxon signed-rank',
        pw, 'paired t P = %.4f' % pt)
    w('  frontal %-6s n=%2d  difference %+.4f  [%+.4f, %+.4f]  Wilcoxon P = %.4f %s  (t P = %.4f)'
      % (bd, n, d, ci[0], ci[1], pw, star(pw), pt))

FL = json.load(io.open(os.path.join(str(DATA), 'flatspec.json'), encoding='utf-8'))
fids = sorted(FL)
ff = np.asarray(FL[fids[0]]['f'], float)
am = (ff >= 8) & (ff < 13)   # same half-open interval as the pipeline
r = np.array([np.mean(np.asarray(FL[p]['rest']['flat'], float)[am])
              for p in fids])
t = np.array([np.mean(np.asarray(FL[p]['task']['flat'], float)[am])
              for p in fids])
d, ci, pw, pt, n, _tci = paired(r, t)
add('3.4 periodic alpha', 'frontal periodic alpha (sec 18.4)', n, d, ci,
    'Wilcoxon signed-rank', pw, 'paired t P = %.4f' % pt)
w('  frontal periodic alpha  n=%2d  %.3f -> %.3f  difference %+.3f  [%+.3f, %+.3f]  '
  'Wilcoxon P = %.4f %s  (t P = %.4f)'
  % (n, r.mean(), t.mean(), d, ci[0], ci[1], pw, star(pw), pt))
w()

# ── W1 ──────────────────────────────────────────────────────────
w('[W1] post-reversal window vs stable window - preregistered mixed model')
# The per-window exponent of the registered primary cluster (po4) is in
# outputs/W1_prereg_windows.csv. data/w1.json holds only front and post, so the
# registered primary analysis cannot be produced from it.
w1c = os.path.join(str(OUT), 'W1_prereg_windows.csv')
if os.path.exists(w1c):
    D1 = pd.read_csv(w1c, encoding='utf-8-sig')
    D1['pid'] = D1['pid'].astype(str).str.zfill(3)
    for cl, nm in (('po4', 'po4 (registered primary)'), ('front', 'front'),
                   ('temp', 'temp'), ('glob', 'glob')):
        s1 = D1[(D1['cluster'] == cl)].dropna(subset=['exponent'])
        if s1.empty:
            continue
        full = (s1.groupby(['pid', 'pair'])['window'].nunique() >= 2)
        keep = set(full[full].index)
        s1 = s1[[(a, b) in keep for a, b in zip(s1['pid'], s1['pair'])]]
        if s1['pid'].nunique() < 4:
            continue
        base = 'stable' if 'stable' in set(s1['window']) else sorted(s1['window'].unique())[0]
        s1 = s1.copy(); s1['ppair'] = s1['pid'] + '_' + s1['pair'].astype(str)
        _cv = [c for c in ('reward_density', 'window_duration_s', 'n_epochs')
               if c in s1.columns and s1[c].notna().any()]
        _f = 'exponent ~ C(window, Treatment("%s"))' % base
        if _cv:
            _f += ' + ' + ' + '.join(_cv)
        m = smf.mixedlm(_f, s1, groups=s1['pid'], re_formula='1',
                        vc_formula={'ppair': '0+C(ppair)'}).fit(reml=True, method='lbfgs')
        k = [q for q in m.params.index if q.startswith('C(window')][0]
        est, p_, se = float(m.params[k]), float(m.pvalues[k]), float(m.bse[k])
        ci = (est - 1.96 * se, est + 1.96 * se)
        add('W1', nm, int(s1['pid'].nunique()), est, ci, 'mixed model Wald', p_,
            '%d pairs' % len(keep))
        w('  %-26s n=%2d pairs %2d  %+.4f  [%+.4f, %+.4f]  Wald P = %.4f %s'
          % (nm, s1['pid'].nunique(), len(keep), est, ci[0], ci[1], p_,
             star(p_)))
else:
    w('  outputs/W1_prereg_windows.csv is missing')
w()

w('[W2] feedback-locked theta - registered specification')
tp = os.path.join(str(OUT), 'W2_prereg_trials.csv')
if os.path.exists(tp):
    T2 = pd.read_csv(tp, encoding='utf-8')
    T2['pid'] = T2['pid'].astype(str).str.zfill(3)
    T2['val'] = np.where(T2['rewarded'] == 1, 'R', 'U')
    for cl, nm in (('front', 'frontal (registered primary)'), ('po4', 'parieto-occipital'),
                   ('temp', 'temporo-parietal'), ('glob', 'global')):
        d2 = T2[T2['cluster'] == cl]
        cnt = d2.pivot_table(index='pid', columns='val', values='theta_dB',
                             aggfunc='size').fillna(0)
        if not {'R', 'U'} <= set(cnt.columns):
            continue
        ok = [p for p in cnt.index
              if cnt.loc[p, 'R'] >= 10 and cnt.loc[p, 'U'] >= 10]
        if len(ok) < 4:
            continue
        m2 = d2[d2['pid'].isin(ok)].pivot_table(
            index='pid', columns='val', values='theta_dB', aggfunc='mean')
        d, _bci, pw, pt, n, ci = paired(m2['R'].values, m2['U'].values)
        # The interval here is the **t interval** - the value reported in main text 3.5
        # and Appendix 3, and the same method as the 'test' column of the table (paired t).
        add('W2 2-level', nm, n, d, ci, 'paired t', pt,
            'Wilcoxon P = %.4f  t interval (df = %d)' % (pw, n - 1))
        w('  2-level %-26s n=%2d  %+.3f dB  [%+.3f, %+.3f]  t P = %.4f %s  '
          '(Wilcoxon P = %.4f)'
          % (nm, n, d, ci[0], ci[1], pt, star(pt), pw))
        # 3-level - registered section 29.3
        c3 = d2.pivot_table(index='pid', columns='cond3', values='theta_dB',
                            aggfunc='size').fillna(0)
        need = [c for c in ('R_cor', 'U_cor', 'U_inc') if c in c3.columns]
        if len(need) < 3:
            continue
        ok3 = [p for p in c3.index if all(c3.loc[p, c] >= 10 for c in need)]
        if len(ok3) < 6:
            continue
        # For the 3-level contrasts the manuscript reports the **bootstrap interval**.
        # The mixed model's Wald P is a different method and disagrees with that
        # interval, so where w2_boot_p.py has produced a P by the same method as the
        # interval, that P is used.
        BP3 = {}
        _bp = os.path.join(str(OUT), 'w2_boot_p.csv')
        if os.path.exists(_bp):
            for _r in csv.DictReader(io.open(_bp, encoding='utf-8-sig')):
                BP3[(_r['cluster'], _r['contrast'])] = _r
        CONTRASTS = [('U_cor', 'unrewarded-correct − rewarded-correct'),
                     ('U_inc', 'unrewarded-incorrect − rewarded-correct'),
                     ('diff', 'unrewarded-incorrect − unrewarded-correct')]
        for lab3, cname in CONTRASTS:
            rec = BP3.get((nm, cname))
            if rec is None:
                continue
            est = float(rec['estimate'])
            ci = (float(rec['ci_lo']), float(rec['ci_hi']))
            p_ = float(rec['p_bootstrap'])
            add('W2 3-level', '%s  %s' % (nm, cname), int(rec['n']), est, ci,
                'participant cluster bootstrap', p_, 'same method as the interval')
            w('  3-level %-26s %-44s n=%2d  %+.3f dB  [%+.3f, %+.3f]  '
              'bootstrap P = %.4f %s'
              % (nm, cname, int(rec['n']), est, ci[0], ci[1], p_, star(p_)))
        if not BP3:
            w('  3-level %-26s w2_boot_p.csv is missing - run it first' % nm)
else:
    w('  outputs/W2_prereg_trials.csv is missing')
w()

# -- H1, the preregistered primary specification ------------------
# The H1 reported in the manuscript and the abstract is the **primary specification**
# cell of section 32.4 - common average, 150 uV, 2-40 Hz, fixed, parieto-occipital,
# minimum 60 s of data. spec_grid.py produces it. The PO row of main-text Figure 4 i
# (.53, n = 9) comes from the spectro.py pipeline (30 s), so its sample and
# preprocessing differ. Both test the same hypothesis, so neither is corrected.
H1J = os.path.join(str(OUT), 'h1_prereg_spec.json')
w('[H1 preregistered primary spec] common average, 150 uV, 2-40 Hz, fixed, parieto-occipital')
if os.path.exists(H1J):
    _h1 = json.load(io.open(H1J, encoding='utf-8'))
    for _k, _lab in (('60 s (registered)', 'parieto-occipital 60 s (registered primary spec)'),
                     ('30 s (post hoc)', 'parieto-occipital 30 s (post hoc criterion)')):
        _r = _h1.get(_k)
        if not _r or _r.get('p_perm') is None:
            continue
        add('H1 primary spec', _lab, _r['n'], _r['rho'],
            (_r['ci_lo'], _r['ci_hi']), 'permutation (Spearman)', _r['p_perm'])
        w('  %-46s n=%2d  rho %+.3f  [%+.3f, %+.3f]  P = %.4f %s'
          % (_lab, _r['n'], _r['rho'], _r['ci_lo'], _r['ci_hi'],
             _r['p_perm'], star(_r['p_perm'])))
    w('  Note. The two criteria differ by the one participant who did not reach 60 s.')
else:
    w('  outputs/h1_prereg_spec.json is missing - run spec_grid.py first')
w()

# -- preregistered primary hypotheses -----------------------------
w('[preregistered primary hypotheses] summary')
w('  H1  parieto-occipital aperiodic exponent × switch bias - the 60 s row of H1 primary spec above')
w('      (the PO row of Figure 4 i is the main-text-specification version of the same hypothesis)')
w('  W1  aperiodic exponent in the post-reversal window - the po4 row of W1 above')
w('  W2  feedback-locked theta, unrewarded − rewarded - the frontal row of W2 2-level above')
w()

# -- disagreements ------------------------------------------------
w('=' * 96)
w(' Results where the interval and the P value disagree')
w('=' * 96)
mism = []
for r_ in ROWS:
    if r_['p'] is None or r_['ci_lo'] is None:
        continue
    excl = not (r_['ci_lo'] < 0 < r_['ci_hi'])
    sig = r_['p'] < .05
    if excl != sig:
        mism.append(r_)
        w('  %-18s %-46s  interval [%+.3f, %+.3f] %s  P = %.4f %s'
          % (r_['group'], r_['label'], r_['ci_lo'], r_['ci_hi'],
             ('excludes 0' if excl else 'includes 0'), r_['p'],
             ('significant' if sig else 'not significant')))
if not mism:
    w('  None - interval and P value point the same way for every result')
w()
n_test = sum(1 for r_ in ROWS if r_['p'] is not None)
n_sig = sum(1 for r_ in ROWS if r_['p'] is not None and r_['p'] < .05)
w('  %d tests  %d results with P < .05' % (n_test, n_sig))
w('  That is %d uncorrected threshold decisions. Under Bonferroni the threshold would be %.5f,' % (n_test, .05 / n_test))
w('  and %d results would clear it.'
  % sum(1 for r_ in ROWS if r_['p'] is not None and r_['p'] < .05 / n_test))


# ---------- part 3 - families, duplicate flags, BH FDR ----------
# The four families are the ones agreed for the manuscript. A result that does not
# sit cleanly in any of them is marked 'ambiguous' and enters no family's FDR.
#
#   (A) preregistered primary   not corrected (three tests fixed in advance)
#   (B) task-related change     5 PLV regions + 5 band powers + periodic alpha
#   (C) EEG-behaviour           Figure 4 d-h, 4j, and the unregistered clusters of 4i
#   (D) W1 and W2
#
# A row that counts the same quantity twice (exp x lose-shift appears in both 4d and
# 4j) is counted once in the FDR family. Leaving the duplicate in inflates the family
# size and makes q optimistic.
# For H1 the registered primary spec (60 s) is the primary test. The post hoc 30 s
# criterion and the main-text specification (the PO row of Figure 4 i) are other
# versions of the same hypothesis, so rather than moving them to an exploratory
# family they are reported here, also uncorrected. Moving them to (C) would grow
# that family from 16 to 17 and raise the q of the other correlations with it.
PRIMARY = {('H1 primary spec', 'parieto-occipital 60 s (registered primary spec)'),
           ('H1 primary spec', 'parieto-occipital 30 s (post hoc criterion)'),
           ('Figure 4 i', 'PO parieto-occipital'),
           ('W1', 'po4 (registered primary)'),
           ('W2 2-level', 'frontal (registered primary)')}
DUP_OF = {('Figure 4 j', 'exp × lose-shift (simple)'):
          ('Figure 4 d-h', 'd  exponent × lose-shift'),
          ('Figure 4 j', 'α × accuracy (simple)'):
          ('Figure 4 d-h', 'g  alpha × accuracy')}

def family_of(r):
    key = (r['group'], r['label'])
    if key in PRIMARY:
        return 'A preregistered primary'
    if r['group'] in ('Figure 5 c', '3.4 band relative power', '3.4 periodic alpha'):
        return 'B task-related change'
    if r['group'] in ('Figure 4 d-h', 'Figure 4 j', 'Figure 4 i'):
        return 'C EEG-behaviour correlation'
    if r['group'] == 'H1 primary spec':
        return 'A preregistered primary'
    if r['group'].startswith('W1') or r['group'].startswith('W2'):
        return 'D W1 and W2'
    return 'ambiguous'


def bh(ps):
    """Benjamini-Hochberg q, returned in input order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    q = [None] * m
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        prev = min(prev, ps[i] * m / k)
        q[i] = prev
    return q


for r in ROWS:
    r['family'] = family_of(r)
    key = (r['group'], r['label'])
    r['duplicate_of'] = ('%s / %s' % DUP_OF[key]) if key in DUP_OF else ''
    if r['group'] == 'Figure 4 i' and r['label'] != 'PO parieto-occipital':
        r['note'] = (r['note'] + '  the PO bar of panel i is held out into (A)').strip(' ')
    if key == ('Figure 4 i', 'PO parieto-occipital'):
        r['note'] = (r['note'] + '  main-text-specification version of H1 (the registered '
                     'primary spec is the 60 s row of H1 primary spec)').strip(' ')

for fam in sorted({r['family'] for r in ROWS}):
    sel = [r for r in ROWS if r['family'] == fam
           and r['p'] is not None and not r['duplicate_of']]
    if fam.startswith('A ') or fam == 'ambiguous' or not sel:
        for r in sel:
            r['q'] = None
        continue
    for r, q in zip(sel, bh([r['p'] for r in sel])):
        r['q'] = round(float(q), 4)
for r in ROWS:
    r.setdefault('q', None)
    if r['duplicate_of']:
        tgt = tuple(r['duplicate_of'].split(' / '))
        src = [x for x in ROWS if (x['group'], x['label']) == tgt]
        if src:
            r['q'] = src[0].get('q')

w('=' * 96)
w(' Benjamini-Hochberg FDR within each family')
w('=' * 96)
w('  (A) is the three preregistered tests and is not corrected.')
w('  Duplicate rows (the same quantity counted twice) are removed from the family size and take the q of the original row.')
w()
for fam in sorted({r['family'] for r in ROWS}):
    sel = [r for r in ROWS if r['family'] == fam and r['p'] is not None]
    uniq = [r for r in sel if not r['duplicate_of']]
    if not sel:
        continue
    w('  [%s]  %d tests (%d duplicates excluded)'
      % (fam, len(uniq), len(sel) - len(uniq)))
    for r in sorted(sel, key=lambda x: x['p']):
        qq = '     —' if r['q'] is None else '%.4f' % r['q']
        dup = ('  <- duplicate of: %s' % r['duplicate_of']) if r['duplicate_of'] else ''
        w('    %-58s P = %.4f %-2s  q = %-8s%s'
          % ('%s  %s' % (r['group'], r['label']), r['p'], star(r['p']),
             qq, dup))
    ns = sum(1 for r in uniq if r['p'] < .05)
    nq = sum(1 for r in uniq if r['q'] is not None and r['q'] < .05)
    if fam.startswith('A '):
        w('    -> %d results with P < .05 (uncorrected)' % ns)
    else:
        w('    -> P < .05 in %d  q < .05 in %d' % (ns, nq))
    w()

# Bonferroni, for reference. The duplicate rows are already excluded from the BH family
# sizes, so the same de-duplicated count is used here. Counting all rows instead would
# contradict the family sizes; both counts are printed so the figure is traceable.
_nt = sum(1 for r in ROWS if r['p'] is not None)
_nd = sum(1 for r in ROWS if r['p'] is not None and not r['duplicate_of'])
_clr = lambda n, sel: sum(1 for r in sel if r['p'] is not None and r['p'] < .05 / n)
_uniq = [r for r in ROWS if not r['duplicate_of']]
w('  For reference - Bonferroni. %d rows carry a P value; %d of them duplicate another row,'
  % (_nt, _nt - _nd))
w('  leaving %d distinct tests. Threshold .05/%d = %.5f, which leaves %d results.'
  % (_nd, _nd, .05 / _nd, _clr(_nd, _uniq)))
w('  Counting all %d rows gives .05/%d = %.5f and leaves %d.'
  % (_nt, _nt, .05 / _nt, _clr(_nt, ROWS)))
w('  The smallest attainable two-sided Wilcoxon P is .00195 at n = 10 and larger at smaller n,')
w('  so it exceeds either threshold and the Bonferroni comparison carries no information.')

with io.open(os.path.join(str(OUT), 'pvalues.csv'), 'w', encoding='utf-8-sig',
             newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=list(ROWS[0]))
    wr.writeheader()
    wr.writerows(ROWS)
json.dump(dict(rows=ROWS, n_tests=_nt, n_tests_distinct=_nd, seed=SEED, nperm=NPERM,
               mismatches=mism),
          io.open(os.path.join(str(OUT), 'pvalues.json'), 'w',
                  encoding='utf-8'), ensure_ascii=False, indent=1)
io.open(os.path.join(str(OUT), 'pvalues.txt'), 'w',
        encoding='utf-8').write('\n'.join(LOG))
print('\nwritten -> outputs/pvalues.csv  .json  .txt  (%d rows)' % len(ROWS))
