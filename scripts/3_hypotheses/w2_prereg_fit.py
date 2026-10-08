# -*- coding: utf-8 -*-
"""W2 stage 2 - the W2 model of registration 32.1.

  primary      theta_dB ~ valence + (1 | participant)          on participant condition means
  sensitivity  theta_dB ~ valence + (1 + valence | participant) at trial level
  secondary    3 levels (rewarded-correct / unrewarded-correct / unrewarded-incorrect), at least 10 epochs per cell
  reference level = rewarded (registration 31). Prediction: theta is larger when unrewarded, so the coefficient > 0
  at least 10 epochs per condition (reg. 28), a floor of 6 participants per cluster (reg. 35.9), all four clusters
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import RAW, OUT

import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

SRC = sys.argv[1] if len(sys.argv) > 1 else str(OUT / 'W2_prereg_trials.csv')
DST = sys.argv[2] if len(sys.argv) > 2 else str(OUT / 'W2_prereg.txt')
SPEC = sys.argv[3] if len(sys.argv) > 3 else 'average re-reference, 150 uV (the registered primary specification)'
MINEP = 10
SEED = 1          # the repository-wide seed (manuscript 2.4)
NB = 10000        # participant cluster bootstrap for the 3-level contrasts
CL = [('front', 'frontal AF3/AF4/F3/F4  <- registered primary (W2)'),
      ('po4', 'parieto-occipital O1/O2/P7/P8'), ('temp', 'temporo-parietal T7/T8/P7/P8'),
      ('glob', 'global')]

L = []
def w(t=''): L.append(t); print(t)

D = pd.read_csv(SRC, encoding='utf-8')
D['pid'] = D['pid'].astype(str).str.zfill(3)

w('# W2 feedback-locked theta (preregistration 32.1)')
w()
w('specification  %s' % SPEC)
w('epoch  stimulus-locked -1000 to +5200 ms, baseline -300 to -100 ms before stimulus (reg. 29.5, 30)')
w('theta  4-8 Hz, 200-500 ms after feedback, Morlet 3-30 Hz in 1 Hz steps, n_cycles = f/2')
w('sign   reference = rewarded. The prediction is larger theta when unrewarded, so the valence coefficient > 0')
w()

def boot(x, nb=10000, seed=1):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    o = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)]
    return np.percentile(o, [2.5, 97.5])

for cl, nm in CL:
    d = D[D['cluster'] == cl].copy()
    w('=' * 74); w(' %s' % nm); w('=' * 74)
    if d.empty:
        w('  no observations'); w(); continue

    # epochs per condition - the floor of 10 in registration 28
    cnt = d.pivot_table(index='pid', columns='rewarded', values='theta_dB',
                        aggfunc='size').fillna(0).astype(int)
    cnt.columns = ['unrewarded' if c == 0 else 'rewarded' for c in cnt.columns]
    for c in ('rewarded', 'unrewarded'):
        if c not in cnt: cnt[c] = 0
    ok = cnt[(cnt['rewarded'] >= MINEP) & (cnt['unrewarded'] >= MINEP)].index.tolist()
    w('  surviving epochs per condition (floor %d)' % MINEP)
    w('    %-6s%10s%12s%10s' % ('id', 'rewarded', 'unrewarded', 'included'))
    for p in cnt.index:
        w('    %-6s%10d%12d%10s' % (p, cnt.loc[p, 'rewarded'], cnt.loc[p, 'unrewarded'],
                                      'yes' if p in ok else '-'))
    w('  participants included: %d' % len(ok))
    if len(ok) < 6:
        w('  -> below the floor of registration 35.9 (n = %d < 6). The model is not fitted.' % len(ok))
        w(); continue

    d = d[d['pid'].isin(ok)].copy()
    d['valence'] = (d['rewarded'] == 0).astype(float)      # reference = rewarded

    # primary analysis - participant condition means
    pm = (d.groupby(['pid', 'valence'])['theta_dB'].mean().unstack()
          .rename(columns={0.0: 'rew', 1.0: 'unrew'}).dropna())
    dif = (pm['unrew'] - pm['rew']).values
    lo, hi = boot(dif)
    w()
    w('  [primary] participant condition means - theta_dB ~ valence + (1 | participant)')
    w('     Note. With balanced data of two observations per participant (rewarded and')
    w('           unrewarded), the valence fixed effect of this model is algebraically the mean')
    w('           paired difference and its standard error is SD(difference)/sqrt(n). The variance')
    w('           component sits on the boundary and makes the numerical optimisation singular, so it is computed analytically.')
    from scipy import stats as _st
    n = len(dif); b = float(dif.mean()); se = float(dif.std(ddof=1) / np.sqrt(n))
    tc = float(_st.t.ppf(0.975, n - 1))
    w('     valence[unrewarded] %+.4f dB - SE %.4f - 95%% CI [%+.4f, %+.4f]  (t based, df = %d)'
      % (b, se, b - tc*se, b + tc*se, n - 1))
    w('     bootstrap 95%% CI [%+.4f, %+.4f]' % (lo, hi))
    w('     larger when unrewarded in %d/%d participants - median %+.4f dB'
      % (int((dif > 0).sum()), len(dif), np.median(dif)))

    # sensitivity - trial level, random slope
    try:
        m2 = smf.mixedlm('theta_dB ~ valence', d, groups='pid',
                         re_formula='1 + valence').fit(reml=True, method='lbfgs')
        b2, se2 = m2.params['valence'], m2.bse['valence']
        w()
        w('  [sensitivity] trial level - (1 + valence | participant)   %d trials' % len(d))
        w('     valence[unrewarded] %+.4f dB - SE %.4f - Wald 95%% CI [%+.4f, %+.4f]'
          % (b2, se2, b2 - 1.96*se2, b2 + 1.96*se2))
    except Exception as e:
        w('  [sensitivity] fit failed: %s' % e)

    # secondary, 3 levels - registration 29.3
    c3 = d.pivot_table(index='pid', columns='cond3', values='theta_dB',
                       aggfunc='size').fillna(0).astype(int)
    need = [c for c in ('R_cor', 'U_cor', 'U_inc') if c in c3.columns]
    ok3 = ([p for p in c3.index if all(c3.loc[p, c] >= MINEP for c in need)]
           if len(need) == 3 else [])
    w()
    w('  [secondary] 3 levels (rewarded-correct, unrewarded-correct, unrewarded-incorrect) - registration 29.3')
    w('     %d participants with at least %d epochs in every cell' % (len(ok3), MINEP))
    if len(need) == 3:
        w('     %-6s%10s%10s%10s' % ('id', 'R_cor', 'U_cor', 'U_inc'))
        for p in c3.index:
            w('     %-6s%10d%10d%10d' % (p, c3.loc[p, 'R_cor'],
                                         c3.loc[p, 'U_cor'], c3.loc[p, 'U_inc']))
    if len(ok3) >= 6:
        d3 = d[d['pid'].isin(ok3)]

        def fit3(frame):
            """The registered model. bfgs does not converge for frontal, so lbfgs is fixed here."""
            m = smf.mixedlm('theta_dB ~ C(cond3, Treatment("R_cor"))', frame,
                            groups=frame['pid']).fit(reml=True, method='lbfgs')
            return {k.split('T.')[-1].rstrip(']'): m.params[k]
                    for k in m.params.index if k.startswith('C(cond3')}, m

        co, m3 = fit3(d3)
        for lab in ('U_cor', 'U_inc'):
            k = [x for x in m3.params.index if x.endswith(f'T.{lab}]')][0]
            w('     %-10s %+.4f dB (SE %.4f)' % (lab, m3.params[k], m3.bse[k]))

        # The Wald SE does not count participants as the independent unit. The interval the
        # manuscript reports is the participant cluster bootstrap. The random-intercept
        # variance sits at 0, so this model effectively reduces to trial-level OLS.
        rng = np.random.default_rng(SEED)
        idx = {p: d3[d3['pid'] == p] for p in ok3}
        B = {'U_cor': [], 'U_inc': []}
        drop = 0
        for _ in range(NB):
            pick = rng.integers(0, len(ok3), len(ok3))
            r = pd.concat([idx[ok3[i]].assign(pid=f'{ok3[i]}_{j}')
                           for j, i in enumerate(pick)], ignore_index=True)
            if r['cond3'].nunique() < 3:
                drop += 1; continue
            try:
                c, _ = fit3(r)
            except Exception:
                drop += 1; continue
            if c.get('U_cor') is None or c.get('U_inc') is None:
                drop += 1; continue
            B['U_cor'].append(c['U_cor']); B['U_inc'].append(c['U_inc'])
        w('     bootstrap 95%% CI (participant clusters, %d resamples, seed %d)' % (NB, SEED))
        for lab, nm in (('U_cor', 'unrewarded-correct − rewarded-correct'),
                        ('U_inc', 'unrewarded-incorrect − rewarded-correct')):
            v = np.asarray(B[lab]); lo, hi = np.percentile(v, [2.5, 97.5])
            w('       %-44s %+7.3f dB  [%+.3f, %+.3f]' % (nm, co[lab], lo, hi))
        v = np.asarray(B['U_inc']) - np.asarray(B['U_cor'])
        lo, hi = np.percentile(v, [2.5, 97.5])
        w('       %-44s %+7.3f dB  [%+.3f, %+.3f]'
          % ('unrewarded-incorrect − unrewarded-correct', co['U_inc'] - co['U_cor'], lo, hi))
        up = int((d3[d3['cond3'] == 'U_inc'].groupby('pid')['theta_dB'].mean()
                  > d3[d3['cond3'] == 'U_cor'].groupby('pid')['theta_dB'].mean()).sum())
        w('       larger for unrewarded-incorrect in %d/%d participants' % (up, len(ok3)))
        if drop:
            w('       %d resamples discarded' % drop)
    else:
        w('     -> below the floor. Not fitted; only the cell counts are reported.')
        if len(ok3) > 0:
            g = d[d['pid'].isin(ok3)].groupby('cond3')['theta_dB'].mean()
            w('     For reference - condition means of the included participants: %s'
              % ' · '.join('%s %+.3f' % (k, v) for k, v in g.items()))
    w()

open(DST, 'w', encoding='utf-8').write('\n'.join(L))
print('written ->', DST)
