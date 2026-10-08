"""
prestim_alpha_fit.py - the mixed-effects logistic regression of section 36.2

Registered wording: "by mixed-effects logistic regression."

Primary model
  switch ~ alpha_z + (1 + alpha_z | participant)
  statsmodels BinomialBayesMixedGLM (participant random intercept + alpha random slope)
  Both fit_vb() and fit_map() are reported.

Primary sample
  9 participants after the exclusions of registration 35.3. Section 35.3 is "PARTICIPANT-LEVEL
  EXCLUSION FOR NON-ENGAGEMENT" and is not a rule that applies to Tier 2 only. The outcome
  variable here is switching itself, so switches by a participant who was not reading the
  feedback (win-stay < .60), or who responded under 150 ms often, are not task-driven.

Sensitivity
  (a) the same model on the 12 who completed the task
  (b) the mean of per-participant logistic coefficients (participant bootstrap interval)
  (c) GEE (exchangeable correlation, participant clusters, cluster-robust standard errors)

Caution. fit_vb of BinomialBayesMixedGLM is mean-field variational Bayes and tends to
underestimate the posterior variance, so the interval can come out narrower than it should
be; fit_map and GEE are read alongside. A random slope on 9 participants also leaves the
slope variance poorly identified, so that estimate is reported too.

alpha_z is z-scored within participant. What this analysis asks about is trial-to-trial
variation within one person, not variation between participants.

Output  outputs/prestim_alpha.csv  outputs/prestim_alpha.txt
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, OUT

import csv, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
warnings.filterwarnings('ignore')

SRC = os.path.join(str(OUT), 'prestim_alpha_trials.csv')
CLN = {'po4': 'parieto-occipital  <- primary', 'front': 'frontal',
       'temp': 'temporo-parietal', 'glob': 'global'}
PRIMARY_W = '-300 to -100 ms  <- primary (reg. 30)'
EXCL = ['007', '008', '009']          # registration 35.3

D = pd.read_csv(SRC, encoding='utf-8')
D = D[D['switch'].notna()].copy()
D['pid'] = D['pid'].astype(str).str.zfill(3)
D['switch'] = D['switch'].astype(int)
D['alpha_z'] = D.groupby(['pid', 'cluster', 'window'])['alpha'].transform(
    lambda s: (s - s.mean()) / (s.std(ddof=1) if s.std(ddof=1) > 0 else 1.0))
D9 = D[~D['pid'].isin(EXCL)].copy()

L = []
def P(t=''): L.append(t); print(t)

P('# 36.2 - does pre-stimulus alpha predict switching on that trial')
P()
P('  outcome    switch = choice differs from the previous valid trial (1), stay (0)')
P('  predictor  pre-stimulus alpha 8-13 Hz log power, z-scored within participant')
P('  primary model  switch ~ alpha_z + (1 + alpha_z | participant)   BinomialBayesMixedGLM')
P('  primary sample  9 after the exclusions of registration 35.3 (007 RT<150ms 24.2%, 008 .595, 009 .538)')
P()
P('  Caution  the registered wording specifies neither cluster, time window, band nor alpha definition.')
P('           The choices below were not fixed in advance (see the header of this script).')
P()

rows = []


def glmm(d, method='vb'):
    """switch ~ alpha_z + (1 + alpha_z | pid). Returns coefficient, SD and slope variance."""
    if d['pid'].nunique() < 3 or d['switch'].nunique() < 2:
        return None
    rnd = {'a': '0 + C(pid)', 'b': '0 + C(pid):alpha_z'}
    try:
        m = BinomialBayesMixedGLM.from_formula('switch ~ alpha_z', rnd, d)
        r = m.fit_vb(verbose=False) if method == 'vb' else m.fit_map()
    except Exception:
        return None
    try:
        i = list(r.model.exog_names).index('alpha_z')
    except ValueError:
        return None
    b = float(r.fe_mean[i])
    if method == 'vb':
        sd = float(r.fe_sd[i])
    else:
        # cov_params() of fit_map can come back as a DataFrame, so it is read positionally
        cp = np.asarray(r.cov_params())
        sd = float(np.sqrt(abs(cp[i, i])))
    # the variance components are stored as log standard deviations
    vc = {n: float(np.exp(v)) for n, v in zip(r.model.vcp_names, r.vcp_mean)}
    return b, sd, vc


def per_sub(d, nb=10000, seed=1):
    b = []
    for p, g in d.groupby('pid'):
        if g['switch'].nunique() < 2 or len(g) < 20:
            continue
        try:
            m = sm.Logit(g['switch'].values,
                         sm.add_constant(g['alpha_z'].values)).fit(disp=0)
            v = float(m.params[1])
            if np.isfinite(v) and abs(v) < 10:
                b.append(v)
        except Exception:
            pass
    if len(b) < 4:
        return None
    b = np.array(b); rng = np.random.default_rng(seed)
    o = [b[rng.integers(0, len(b), len(b))].mean() for _ in range(nb)]
    return b.mean(), np.percentile(o, 2.5), np.percentile(o, 97.5), len(b), b


def gee(d):
    try:
        g = smf.gee('switch ~ alpha_z', 'pid', data=d,
                    family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        c, se = float(g.params['alpha_z']), float(g.bse['alpha_z'])
        return c, c - 1.96*se, c + 1.96*se
    except Exception:
        return None


# -- primary model, primary sample (n = 9) - 4 clusters x 3 windows --
for wlab in [PRIMARY_W] + [w for w in D['window'].unique() if w != PRIMARY_W]:
    P('=' * 84)
    P(f' [primary sample n=9 - mixed-effects logistic]  window  {wlab}')
    P('=' * 84)
    P(f"  {'cluster':<32}{'trials':>8}{'n':>4}{'log odds':>10}{'95% CI':>22}{'odds ratio':>12}")
    for cl in ('po4', 'front', 'temp', 'glob'):
        d = D9[(D9['cluster'] == cl) & (D9['window'] == wlab)]
        if d.empty:
            continue
        r = glmm(d, 'vb')
        if r is None:
            P(f'  {CLN[cl]:<32}{len(d):>8}{d["pid"].nunique():>4}   fit failed')
            continue
        b, sd, vc = r
        lo, hi = b - 1.96*sd, b + 1.96*sd
        P(f'  {CLN[cl]:<32}{len(d):>8}{d["pid"].nunique():>4}{b:>+10.3f}'
          f'   [{lo:+.3f}, {hi:+.3f}]{np.exp(b):>9.3f}')
        rows.append(dict(sample='n=9', window=wlab, cluster=CLN[cl], trials=len(d),
                         n=d['pid'].nunique(), log_odds=round(b, 4),
                         lo=round(lo, 4), hi=round(hi, 4),
                         odds_ratio=round(float(np.exp(b)), 4), model='GLMM (vb)'))
    P()

# -- sensitivity of the primary cell ------------------------------
P('=' * 84)
P(' sensitivity of the primary cell (parieto-occipital, -300 to -100 ms)')
P('=' * 84)
d9 = D9[(D9['cluster'] == 'po4') & (D9['window'] == PRIMARY_W)]
d12 = D[(D['cluster'] == 'po4') & (D['window'] == PRIMARY_W)]

for lab, dd in (('n=9 (primary)', d9), ('n=12', d12)):
    P(f'  [{lab}]  trials {len(dd)} - participants {dd["pid"].nunique()}')
    for meth, mn in (('vb', 'GLMM variational Bayes'), ('map', 'GLMM MAP')):
        r = glmm(dd, meth)
        if r is None:
            P(f'    {mn:<26} fit failed'); continue
        b, sd, vc = r
        P(f'    {mn:<26}{b:>+8.3f}  [{b-1.96*sd:+.3f}, {b+1.96*sd:+.3f}]'
          f'  - odds ratio {np.exp(b):.3f}')
        if meth == 'vb':
            P(f'      variance-component SD  intercept {vc.get("a", float("nan")):.3f} - '
              f'alpha slope {vc.get("b", float("nan")):.3f}')
        rows.append(dict(sample=lab, window=PRIMARY_W, cluster='parieto-occipital  <- primary',
                         trials=len(dd), n=dd['pid'].nunique(), log_odds=round(b, 4),
                         lo=round(b-1.96*sd, 4), hi=round(b+1.96*sd, 4),
                         odds_ratio=round(float(np.exp(b)), 4), model=mn))
    r = per_sub(dd)
    if r:
        m, lo, hi, n, bb = r
        P(f'    {"mean of per-participant coefficients":<26}{m:>+8.3f}  [{lo:+.3f}, {hi:+.3f}]'
          f'  - same direction in {int((bb > 0).sum())}/{len(bb)}')
        P('      individual coefficients: ' + '  '.join(f'{v:+.2f}' for v in bb))
        rows.append(dict(sample=lab, window=PRIMARY_W, cluster='parieto-occipital  <- primary',
                         trials=len(dd), n=n, log_odds=round(m, 4), lo=round(lo, 4),
                         hi=round(hi, 4), odds_ratio=round(float(np.exp(m)), 4),
                         model='per-participant mean'))
    g = gee(dd)
    if g:
        c, lo, hi = g
        P(f'    {"GEE cluster robust":<26}{c:>+8.3f}  [{lo:+.3f}, {hi:+.3f}]'
          f'  - odds ratio {np.exp(c):.3f}')
        rows.append(dict(sample=lab, window=PRIMARY_W, cluster='parieto-occipital  <- primary',
                         trials=len(dd), n=dd['pid'].nunique(), log_odds=round(c, 4),
                         lo=round(lo, 4), hi=round(hi, 4),
                         odds_ratio=round(float(np.exp(c)), 4), model='GEE'))
    P()

# -- previous reward (a covariate not in the registration) --------
P('=' * 84)
P(' contrast - previous reward (a covariate not in the registration)')
P('=' * 84)
dd = d9[d9['prev_rew'].notna()]
if len(dd) > 100:
    try:
        g2 = smf.gee('switch ~ alpha_z + prev_rew', 'pid', data=dd,
                     family=sm.families.Binomial(),
                     cov_struct=sm.cov_struct.Exchangeable()).fit()
        for k, nm in (('alpha_z', 'pre-stimulus alpha (z)'), ('prev_rew', 'previous reward')):
            c, se = float(g2.params[k]), float(g2.bse[k])
            P(f'  {nm:<24}{c:>+8.3f}  [{c-1.96*se:+.3f}, {c+1.96*se:+.3f}]'
              f'  - odds ratio {np.exp(c):.3f}')
            rows.append(dict(sample='n=9', window=PRIMARY_W, cluster='parieto-occipital  <- primary',
                             trials=len(dd), n=dd['pid'].nunique(), log_odds=round(c, 4),
                             lo=round(c-1.96*se, 4), hi=round(c+1.96*se, 4),
                             odds_ratio=round(float(np.exp(c)), 4), model=f'GEE+prev reward - {nm}'))
        P()
        P('  Included to show that the design can detect a trial-level predictor. It is a covariate')
        P('  not in the registration, so it is read only as part of the result.')
    except Exception as e:
        P(f'  fit failed: {e}')
P()
P('  Caution 1. fit_vb of BinomialBayesMixedGLM is mean-field variational Bayes and tends to')
P('             underestimate the posterior variance. The fit_map and GEE intervals must be read alongside.')
P('  Caution 2. A random slope on 9 participants leaves the slope variance poorly identified.')
P('  Registration 34 - the values above are not called significant or non-significant.')

dst = os.path.join(str(OUT), 'prestim_alpha.csv')
with open(dst, 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=['sample','window','cluster','trials','n',
                                       'log_odds','lo','hi','odds_ratio','model'])
    w.writeheader(); w.writerows(rows)
open(os.path.join(str(OUT), 'prestim_alpha.txt'), 'w', encoding='utf-8').write('\n'.join(L))
print(f'\nwritten -> {dst}')
