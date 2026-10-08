"""
feasibility.py - the feasibility outcomes F1-F7 of preregistration section 34.3, plus the remaining reporting items

Registration section 34.3 settled that these seven are "reported irrespective of the outcome of any other analysis".
The manuscript reports some of them scattered through the text but not as one F-numbered set.

  F1 recruitment      at least 10 completed sessions within the window      descriptive
  F2 practice pass    at least 70% of those who started practice pass stage 1   descriptive
  F3 EEG yield        at least 80% of completed sessions give a parieto-occipital value under the primary spec   prespecified
  F4 run-to-run reliability  reported without a threshold                  prespecified
  F5 retention        fewer than 20% withdraw after consenting            descriptive
  F6 task engagement  median timeout rate below 10%                       prespecified
  F7 model fit        at least 80% better explained by M3 than by random choice   prespecified

Definition of a completed session (registration section 13) - a session in which consent was given and both resting runs were recorded.
Completing the task is not required. Resting state comes before practice, so it yields usable
resting data even for participants who did not pass practice.

Remaining reporting items, produced alongside
  18.4  number of channels with a fitted alpha peak
  28    epoch rejection rate (per participant and per run)
  18.2  number of participants for whom SBI is undefined
  36.7  per-participant rate of choosing stimulus A, and the group distribution
  36.1  change from run 1 to run 2
  34.3  per-channel data yield (a methodological outcome reported without a pass/fail criterion)

Output  outputs/feasibility.csv
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT

import csv, glob, json, math, warnings
import numpy as np
from scipy import stats
warnings.filterwarnings("ignore")

CH = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
EP60 = 30          # 60 s = 30 epochs of 2 s (the primary criterion of registration 28.7)
EP30 = 15          # 30 s = 15 epochs (the criterion the manuscript lowered to post hoc)

SP = json.load(open(f'{DATA}/spectro.json', encoding='utf-8'))
BM = json.load(open(f'{DATA}/behav_metrics.json', encoding='utf-8'))

rows = []
def rec(fid, item, value, crit='', verdict=''):
    rows.append({'id': fid, 'item': item, 'value': value, 'criterion': crit, 'verdict': verdict})

def head(t):
    print('\n' + '=' * 78); print(' ' + t); print('=' * 78)

def kept(r):
    return r.get('n_kept')

def usable(pid, cl, ep):
    """Is there at least one run that clears the epoch floor and yields a value for that cluster."""
    for r in SP.get(pid, []):
        k = kept(r)
        if k is None or k < ep:
            continue
        c = r.get(cl)
        if c and c.get('exponent') is not None:
            return True
    return False

# -- sample definition ---------------------------------------------
all_pid = sorted(SP)
completed = [p for p in all_pid if len(SP[p]) >= 2]     # registration 13
withdrew = [p for p in all_pid if len(SP[p]) < 2]
task = sorted(BM)
no_task = [p for p in completed if p not in BM]

head('sample')
print(f'  participants with a resting recording  {len(all_pid)}  {" ".join(all_pid)}')
print(f'  completed sessions (reg. 13, both runs) {len(completed)}  {" ".join(completed)}')
print(f'  participants without both runs          {len(withdrew)}  {" ".join(withdrew) or "none"}')
print(f'  participants who performed the task     {len(task)}  {" ".join(task)}')
print(f'  participants who did not pass practice  {len(no_task)}  {" ".join(no_task) or "none"}')

# ── F1 ─────────────────────────────────────────────────────────────
head('F1  recruitment - at least 10 completed sessions (descriptive)')
v = len(completed)
print(f'  completed sessions {v}')
rec('F1', 'completed sessions', v, '>= 10', 'met' if v >= 10 else 'not met')

# ── F2 ─────────────────────────────────────────────────────────────
head('F2  practice pass - at least 70% of those who started practice (descriptive)')
den = len(completed); num = len(task)
print(f'  {num}/{den} = {100*num/den:.1f}%')
print('  Note. The denominator is the completed sessions. Practice comes after resting state, so')
print('        every completed session started practice.')
rec('F2', 'practice pass rate', f'{num}/{den} = {100*num/den:.1f}%', '>= 70%',
    'met' if num/den >= .70 else 'not met')

# ── F3 ─────────────────────────────────────────────────────────────
head('F3  EEG yield - at least 80% of completed sessions give a parieto-occipital value (prespecified)')
print(f"  {'criterion':<22}{'parieto-occ.':>14}{'frontal':>10}{'occ. O1/O2':>14}")
for ep, lab in ((EP60, '60 s (registered primary)'), (EP30, '30 s (post hoc)')):
    a = sum(usable(p, 'po4', ep) for p in completed)
    b = sum(usable(p, 'front', ep) for p in completed)
    c = sum(usable(p, 'post', ep) for p in completed)
    print(f'  {lab:<22}{a:>4}/{len(completed)} {100*a/len(completed):>5.1f}%'
          f'{b:>4}/{len(completed)}{c:>8}/{len(completed)}')
    rec('F3', f'parieto-occipital yield - {lab}', f'{a}/{len(completed)} = {100*a/len(completed):.1f}%',
        '>= 80%', 'met' if a/len(completed) >= .80 else 'not met')
print()
print('  Note. The primary criterion of registration 28.7 is 60 s per run (30 epochs of 2 s). The')
print('        manuscript lowered it to 30 s and disclosed that in Appendix 1. Both are reported here.')

# ── F4 ─────────────────────────────────────────────────────────────
head('F4  run-to-run reliability - reported without a threshold (prespecified)')
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

print(f"  {'cluster':<18}{'index':<14}{'ICC(2,1)':>10}{'95% CI':>22}{'n':>5}")
for cl, cn in (('po4', 'parieto-occipital'), ('front', 'frontal'), ('post', 'occipital O1/O2')):
    for key, kn in (('exponent', 'exponent'), ('alpha_amp', 'alpha peak')):
        A, B = [], []
        for p in completed:
            v = [r[cl][key] for r in SP[p]
                 if not r.get('dropped') and r.get(cl) and r[cl].get(key) is not None]
            if len(v) >= 2:
                A.append(v[0]); B.append(v[1])
        if len(A) < 4:
            print(f'  {cn:<12}{kn:<10}{"—":>10}{"(n < 4)":>22}{len(A):>5}')
            continue
        val = icc21(np.array(A), np.array(B)); lo, hi = boot_icc(A, B)
        flag = '' if len(A) >= 8 else '  below the floor'
        print(f'  {cn:<12}{kn:<10}{val:>+10.3f}   [{lo:+.2f}, {hi:+.2f}]{len(A):>5}{flag}')
        rec('F4', f'ICC {cn} {kn}', f'{val:+.3f} [{lo:+.2f}, {hi:+.2f}]', 'no threshold',
            f'n = {len(A)}' + (' - below the floor' if len(A) < 8 else ''))
print()
print('  Note. Registration section 34 settled that this index carries no pass/fail threshold,')
print('        because at this sample size the confidence interval is wide enough that any single')
print('        cut-off would be arbitrary. The temporo-parietal and global clusters need the output of prereg_clusters.py.')

# ── F5 ─────────────────────────────────────────────────────────────
head('F5  retention - fewer than 20% withdraw after consenting (descriptive)')
v = len(withdrew) / len(all_pid)
print(f'  {len(withdrew)}/{len(all_pid)} = {100*v:.1f}%'
      + (f'  ({" ".join(withdrew)})' if withdrew else ''))
rec('F5', 'withdrawal rate', f'{len(withdrew)}/{len(all_pid)} = {100*v:.1f}%', '< 20%',
    'met' if v < .20 else 'not met')

# ── F6 ─────────────────────────────────────────────────────────────
head('F6  task engagement - median timeout rate below 10% (prespecified)')
to = {p: BM[p]['n_timeout'] / BM[p]['n_trials'] for p in task}
med = float(np.median(list(to.values())))
print(f'  median {100*med:.1f}% - range {100*min(to.values()):.1f}-{100*max(to.values()):.1f}%')
print('  per participant: ' + '  '.join(f'{p} {100*v:.1f}%' for p, v in sorted(to.items())))
rec('F6', 'median timeout rate', f'{100*med:.1f}%', '< 10%', 'met' if med < .10 else 'not met')

# ── F7 ─────────────────────────────────────────────────────────────
head('F7  model fit - at least 80% better explained by M3 than by random choice (prespecified)')
try:
    RL = json.load(open(f'{DATA}/rlfit.json', encoding='utf-8'))
except FileNotFoundError:
    RL = {}
if not RL:
    print('  rlfit.json is missing, so this cannot be produced.')
    rec('F7', 'model fit', 'not computable', '>= 80%', 'rlfit.json missing')
else:
    ok_ll = ok_bic = n = 0
    print(f"  {'id':<6}{'M3 nll':>10}{'random nll':>12}{'difference':>12}{'BIC diff':>10}")
    for p in sorted(RL):
        f = RL[p].get('full') or {}
        nll = f.get('nll'); nt = f.get('n')
        if nll is None or not nt: continue
        rnd = nt * math.log(2)
        bic_full = f.get('bic', 2*nll + 4*math.log(nt))
        bic_rnd = 2*rnd
        n += 1
        ok_ll += nll < rnd
        ok_bic += bic_full < bic_rnd
        print(f'  {p:<6}{nll:>10.1f}{rnd:>12.1f}{rnd-nll:>+10.1f}{bic_rnd-bic_full:>+10.1f}')
    print()
    print(f'  by log-likelihood  {ok_ll}/{n} = {100*ok_ll/n:.1f}%')
    print(f'  by BIC             {ok_bic}/{n} = {100*ok_bic/n:.1f}%')
    rec('F7', 'participants better than random (log-likelihood)', f'{ok_ll}/{n} = {100*ok_ll/n:.1f}%',
        '>= 80%', 'met' if ok_ll/n >= .80 else 'not met')
    rec('F7', 'participants better than random (BIC)', f'{ok_bic}/{n} = {100*ok_bic/n:.1f}%',
        '>= 80%', 'met' if ok_bic/n >= .80 else 'not met')
    print()
    print('  Note. The criterion set in registration 32.3 is the PSIS-LOO elpd difference. The')
    print('        rlmodel.py of this repository estimates per participant by maximum likelihood')
    print('        rather than hierarchically and in a Bayesian way, so elpd cannot be produced.')
    print('        The log-likelihood difference and the BIC difference are given instead; the')
    print('        former favours the model with more parameters, so BIC should be read alongside.')

# ---------- remaining reporting items ----------
head('18.4  number of channels with a fitted alpha peak')
try:
    PK = json.load(open(f'{DATA}/chan_peak.json', encoding='utf-8'))
except FileNotFoundError:
    PK = {}
if PK:
    tot = fit = 0
    per = {}
    for p in sorted(PK):
        a = sum(1 for c in PK[p] if (PK[p][c] or {}).get('pk') is not None)
        b = len(PK[p]); per[p] = (a, b); tot += b; fit += a
    print(f'  overall {fit}/{tot} channel-participant cells = {100*fit/tot:.1f}%')
    print('  per participant: ' + '  '.join(f'{p} {a}/{b}' for p, (a, b) in sorted(per.items())))
    print()
    print('  Note. Registration 18.4 asked for a descriptive report of the number of channels with')
    print('        a fitted peak whose centre frequency is 8-13 Hz. The search band in this')
    print('        repository is the manuscript specification 7-14 Hz, so the count above is on that band.')
    rec('18.4', 'rate of channels with a fitted peak', f'{fit}/{tot} = {100*fit/tot:.1f}%', '-', 'descriptive')
else:
    print('  chan_peak.json is missing.')

head('28  epoch rejection rate')
rej = []
print(f"  {'id':<6}{'run':>5}{'total':>7}{'kept':>7}{'rejected':>10}")
for p in sorted(SP):
    for r in SP[p]:
        t, k = r.get('n_total'), kept(r)
        if not t or k is None: continue
        v = 1 - k/t; rej.append(v)
        print(f'  {p:<6}{r.get("run", "?"):>5}{t:>7}{k:>7}{100*v:>8.1f}%')
if rej:
    print(f'\n  overall median {100*np.median(rej):.1f}% - '
          f'range {100*min(rej):.1f}-{100*max(rej):.1f}%')
    rec('28', 'median epoch rejection rate', f'{100*np.median(rej):.1f}%', '-', 'descriptive')

head('18.2  number of participants for whom SBI is undefined')
undef = [p for p in task if BM[p]['n_persev'] == 0 and BM[p]['n_regress'] == 0]
small = [p for p in task if 0 < BM[p]['n_persev'] + BM[p]['n_regress'] < 3]
print(f'  perseverative = regressive = 0 (undefined)  {len(undef)}  {" ".join(undef) or "none"}')
print(f'  perseverative + regressive < 3 (flagged)    {len(small)}  {" ".join(small) or "none"}')
rec('18.2', 'participants with SBI undefined', len(undef), '-', 'descriptive')

head('36.7  per-participant rate of choosing stimulus A')
prop = {}
for path in sorted(glob.glob(f'{RAW}/**/*_behav.csv', recursive=True)):
    pid = os.path.basename(path).split('_')[-2]
    if pid in prop: continue
    R = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    ch = [r.get('choice') for r in R if r.get('timeout') in ('0', 'False', 'false', '')]
    ch = [c for c in ch if c]
    if ch: prop[pid] = sum(1 for c in ch if c == 'A') / len(ch)
if prop:
    v = np.array(list(prop.values()))
    print('  ' + '  '.join(f'{p} {100*x:.0f}%' for p, x in sorted(prop.items())))
    print(f'\n  mean {100*v.mean():.1f}% - median {100*np.median(v):.1f}% - '
          f'range {100*v.min():.0f}-{100*v.max():.0f}%')
    print('  Note. Registration 7.1 - the correct option in the first segment is fixed to stimulus A')
    print('        for every participant, so this rate is read descriptively only.')
    rec('36.7', 'mean rate of choosing stimulus A', f'{100*v.mean():.1f}%', '-', 'descriptive')

head('36.1  change from run 1 to run 2')
print(f"  {'cluster':<18}{'index':<14}{'run 1':>9}{'run 2':>9}{'difference':>12}{'95% CI':>20}{'n':>4}")
def boot_mean(x, nb=10000, seed=1):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    o = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)]
    return np.percentile(o, [2.5, 97.5])
for cl, cn in (('po4', 'parieto-occipital'), ('front', 'frontal'), ('post', 'occipital O1/O2')):
    for key, kn in (('exponent', 'exponent'), ('alpha_amp', 'alpha peak')):
        A, B = [], []
        for p in completed:
            v = [r[cl][key] for r in SP[p]
                 if not r.get('dropped') and r.get(cl) and r[cl].get(key) is not None]
            if len(v) >= 2:
                A.append(v[0]); B.append(v[1])
        if len(A) < 4:
            print(f'  {cn:<12}{kn:<10}{"—":>9}{"":>9}{"":>10}{"(n < 4)":>20}{len(A):>4}')
            continue
        d = np.array(B) - np.array(A); lo, hi = boot_mean(d)
        print(f'  {cn:<18}{kn:<14}{np.mean(A):>9.3f}{np.mean(B):>9.3f}'
              f'{d.mean():>+10.3f}   [{lo:+.3f}, {hi:+.3f}]{len(A):>4}')
        rec('36.1', f'{cn} {kn} run 2 - run 1', f'{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]',
            '—', f'n = {len(A)}')
print()
print('  Note. Registration 36.1 recorded in advance that this change is not statistically')
print('        distinguishable from measurement instability (the "RELIABILITY VERSUS ADAPTATION"')
print('        section of 34). The gap between the two runs was participant- and experimenter-paced and so is not fixed.')

head('34.3  per-channel data yield (no pass/fail criterion)')
bad = {c: 0 for c in CH}; tot_run = 0
for p in sorted(SP):
    for r in SP[p]:
        tot_run += 1
        for c in (r.get('bad') or []):
            if c in bad: bad[c] += 1
print(f'  rate flagged bad, over {tot_run} runs')
for c in CH:
    print(f'    {c:<5}{100*bad[c]/tot_run:>6.1f}%')
rec('34.3', 'most frequently bad channel', max(bad, key=bad.get),
    '—', f'{100*max(bad.values())/tot_run:.1f}%')

# -- write ---------------------------------------------------------
os.makedirs(str(OUT), exist_ok=True)
dst = os.path.join(str(OUT), 'feasibility.csv')
with open(dst, 'w', newline='', encoding='utf-8') as fh:
    wr = csv.DictWriter(fh, fieldnames=['id', 'item', 'value', 'criterion', 'verdict'])
    wr.writeheader(); wr.writerows(rows)
print(f'\nwritten -> {dst}')
