"""
w2_theta.py - the preregistered within-participant hypothesis W2

Compares frontal-cluster theta (4-8 Hz) 200-500 ms after feedback between unrewarded and
rewarded trials. The baseline is the stimulus-locked -300 to -100 ms fixed by the preregistration.

Note on the condition labels. In tfr.json, U is unrewarded (marker 33) and R is rewarded
(marker 34). For all 12 participants the count of marker 34 equals the number of reward==1
trials in the behavioural record and also matches total_rewards in meta.json.

Changing the baseline changes the result, so four are reported together. The window just
before feedback (-600 to -200 ms) contains the response and the choice highlight, so a
baseline difference between conditions mixes straight into the dB. The manuscript reports the
registered (stimulus-locked) baseline.
Input   tfr.json  (produced by tfr.py)
Output  W2_feedback_theta.csv  W2_baseline_check.csv  (both in outputs/)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, OUT

import csv, json, warnings
import numpy as np
from scipy import stats
warnings.filterwarnings('ignore')

DATA = str(DATA); OUT = str(OUT)
TH_LO, TH_HI = 4.0, 8.0          # theta
W_LO, W_HI = 0.20, 0.50          # post-feedback window

T = json.load(open(f'{DATA}/tfr.json', encoding='utf-8'))
pids = sorted(T)
if not pids:
    raise SystemExit('tfr.json is empty. Run scripts/tfr.py first.')
if 'U' not in T[pids[0]]:
    raise SystemExit('tfr.json is in the old format (C/E). Re-run scripts/tfr.py.')

f = np.array(T[pids[0]]['f']); t = np.array(T[pids[0]]['t'])
FM = (f >= TH_LO) & (f <= TH_HI)
TM = (t >= W_LO) & (t <= W_HI)


def boot_ci(d, n=10000, seed=1):   # seed unified (2026-09-27)
    rng = np.random.default_rng(seed)
    b = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def diff(kind):
    """Condition difference (unrewarded - rewarded) per baseline type, in dB.

    dB takes the log per channel before averaging, so the baseline has to be subtracted in the same space.
    """
    out, use = [], []
    for p in pids:
        r = T[p]
        if kind == 'stim' and 'stimbasedb_U' not in r:
            continue
        rU = np.array(r['rawdb_U']); rR = np.array(r['rawdb_R'])
        if kind == 'feedback':
            bU = np.array(r['basedb_U'])[:, None]; bR = np.array(r['basedb_R'])[:, None]
        elif kind == 'stim':
            bU = np.array(r['stimbasedb_U'])[:, None]; bR = np.array(r['stimbasedb_R'])[:, None]
        elif kind == 'common':
            m = (np.array(r['basedb_U']) + np.array(r['basedb_R'])) / 2
            bU = bR = m[:, None]
        else:                                    # no baseline
            bU = bR = 0.0
        out.append(float((rU - bU)[np.ix_(FM, TM)].mean()
                         - (rR - bR)[np.ix_(FM, TM)].mean()))
        use.append(p)
    return np.array(out), use


# -- per-participant values (registered baseline) -----------------
rows = []
for p in pids:
    r = T[p]
    if 'stimbasedb_U' not in r:
        continue
    bU = np.array(r['stimbasedb_U'])[:, None]; bR = np.array(r['stimbasedb_R'])[:, None]
    u = float((np.array(r['rawdb_U']) - bU)[np.ix_(FM, TM)].mean())
    w = float((np.array(r['rawdb_R']) - bR)[np.ix_(FM, TM)].mean())
    rows.append(dict(id=p, unrewarded_dB=round(u, 4), rewarded_dB=round(w, 4),
                     diff_dB=round(u - w, 4),
                     n_unrewarded=r['n_U'], n_rewarded=r['n_R']))

print('=' * 74)
print(' W2 - feedback-locked frontal theta, unrewarded vs rewarded')
print('=' * 74)
print(f'  frontal cluster - {TH_LO:.0f}-{TH_HI:.0f} Hz - '
      f'{W_LO*1000:.0f}-{W_HI*1000:.0f} ms - baseline stimulus-locked -300 to -100 ms')
print(f"\n{'id':>5}{'unrewarded':>13}{'rewarded':>11}{'difference':>12}{'n unrew/rew':>14}")
for r in rows:
    print(f"{r['id']:>5}{r['unrewarded_dB']:>11.3f}{r['rewarded_dB']:>11.3f}"
          f"{r['diff_dB']:>+10.3f}{r['n_unrewarded']:>7}/{r['n_rewarded']:<5}")

LAB = [('stim', 'stimulus-locked -300 to -100 ms (preregistered)'),
       ('feedback', 'feedback-locked -600 to -200 ms'),
       ('common', 'condition-common baseline (mean of the two)'),
       ('none', 'no baseline (raw power)')]
print('\n' + '=' * 74)
print(' results by baseline')
print('=' * 74)
print(f"{'baseline':48s}{'n':>4}{'diff':>9}{'95% CI':>20}{'same dir.':>11}{'P':>9}")
out_rows = []
for kind, lab in LAB:
    d, use = diff(kind)
    if len(d) < 3:
        continue
    lo, hi = boot_ci(d)
    P = float(stats.wilcoxon(d).pvalue)
    print(f"{lab:48s}{len(d):>4}{d.mean():>+9.3f}"
          f"   [{lo:>+.2f}, {hi:>+.2f}]{int((d>0).sum()):>7}/{len(d)}{P:>9.4f}")
    out_rows.append(dict(baseline=lab, n=len(d), diff_dB=round(float(d.mean()), 4),
                         CI_lo=round(lo, 4), CI_hi=round(hi, 4),
                         same_direction=f'{int((d>0).sum())}/{len(d)}', P=round(P, 4)))

# -- sensitivity to a trial-count floor ---------------------------
# The trial counts per condition differ greatly between participants (unrewarded 23-65,
# rewarded 11-58), so this checks that the smaller side is not driving the result. It is the value of manuscript 3.4.
print('\n' + '=' * 74)
print(' sensitivity to a trial-count floor (preregistered baseline)')
print('=' * 74)
print(f"{'floor':48s}{'n':>4}{'diff':>9}{'95% CI':>20}{'same dir.':>11}{'P':>9}")
_d_all, _use_all = diff('stim')
for _thr in (0, 15, 20):
    _keep = [i for i, _p in enumerate(_use_all)
             if min(T[_p]['n_U'], T[_p]['n_R']) >= _thr]
    _d = _d_all[_keep]
    if len(_d) < 3:
        continue
    _lo, _hi = boot_ci(_d)
    _P = float(stats.wilcoxon(_d).pvalue)
    _lab = 'no floor' if _thr == 0 else f'at least {_thr} in both conditions'
    print(f"{_lab:48s}{len(_d):>4}{_d.mean():>+9.3f}"
          f"   [{_lo:>+.2f}, {_hi:>+.2f}]{int((_d>0).sum()):>7}/{len(_d)}{_P:>9.4f}")
    out_rows.append(dict(baseline=f'stimulus-locked - {_lab}', n=len(_d),
                         diff_dB=round(float(_d.mean()), 4),
                         CI_lo=round(_lo, 4), CI_hi=round(_hi, 4),
                         same_direction=f'{int((_d>0).sum())}/{len(_d)}', P=round(_P, 4)))

# -- is the baseline itself different between conditions ----------
print('\n' + '=' * 74)
print(' condition difference in the baseline itself (theta)')
print('=' * 74)
for key, lab in [('basedb', 'feedback-locked'), ('stimbasedb', 'stimulus-locked')]:
    v = [np.array(T[p][f'{key}_U'])[FM].mean() - np.array(T[p][f'{key}_R'])[FM].mean()
         for p in pids if f'{key}_U' in T[p]]
    v = np.array(v)
    lo, hi = boot_ci(v)
    print(f"  {lab:12s} {v.mean():+.3f} dB · 95% CI [{lo:+.3f}, {hi:+.3f}] · "
          f"{int((v>0).sum())}/{len(v)} · P = {stats.wilcoxon(v).pvalue:.4f}")

os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/W2_feedback_theta.csv', 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
with open(f'{OUT}/W2_baseline_check.csv', 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
print(f'\nwritten: {OUT}/W2_feedback_theta.csv  {OUT}/W2_baseline_check.csv')
