"""
behav_metrics.py - produces every behavioural index of the manuscript from the trial-level CSV.

This rebuilds the values of the workbook's behaviour sheet from the raw data. The output
matches the descriptives and correlations of manuscript section 3.2 to three decimals.

Two production rules decide the result. They are not stated in the manuscript's methods, so
they are recorded here.

  1. Timeout trials are removed first, and adjacent pairs are formed from the remaining valid
     choices only. Leaving the timeout trials in and merely skipping those pairs gives a
     lose-shift mean of .511 and a maximum of .846, against the manuscript's .508 and .792.
     Accuracy also needs the timeouts out of the denominator to reach the manuscript's .597.
     There are 45 timeout trials in all, 23 of them in one participant (006), so this choice really does move the numbers.

  2. switch-bias index = (regressive errors - perseverative errors) / (regressive + perseverative)
     Positive means regression-dominant, negative perseveration-dominant.

The perseverative/regressive classification itself is done not here but by the PsychoPy task
script, and recorded in the error_type column as correct / acquisition / perseverative /
regressive. Publishing the definition means publishing the task script alongside.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW

import csv, glob, json, warnings
import numpy as np
from scipy import stats
warnings.filterwarnings("ignore")

RAW = str(RAW)


def _int(v, d=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return d


def load():
    """Participant number -> list of trials. The second-to-last segment of the filename is the participant number."""
    out = {}
    for p in sorted(glob.glob(f'{RAW}/**/*_behav.csv', recursive=True)):
        pid = os.path.basename(p).split('_')[-2]
        out.setdefault(pid, list(csv.DictReader(open(p, encoding='utf-8-sig'))))
    return out


def metrics(B):
    et = [r['error_type'] for r in B]
    n_persev = et.count('perseverative')
    n_regress = et.count('regressive')

    valid = [r for r in B if r['choice']]            # rule 1 - compacted after removing timeouts
    ls_n = ls_d = ws_n = ws_d = sw_n = sw_d = 0
    for i in range(1, len(valid)):
        stay = valid[i]['choice'] == valid[i - 1]['choice']
        sw_d += 1
        if not stay:
            sw_n += 1
        if _int(valid[i - 1]['reward']) == 0:
            ls_d += 1
            if not stay:
                ls_n += 1
        else:
            ws_d += 1
            if stay:
                ws_n += 1

    run_lens, s = [], 0
    ch = [r['choice'] for r in B]
    for i in range(1, len(ch)):
        if ch[i] == ch[i - 1]:
            continue
        run_lens.append((s, i - 1)); s = i
    run_lens.append((s, len(ch) - 1))
    lens = [b - a + 1 for a, b in run_lens]
    rev = [i for i, r in enumerate(B) if _int(r['is_reversal_trial'])]
    a, b = run_lens[int(np.argmax(lens))]

    seg = {}
    for i, r in enumerate(B):
        seg.setdefault(int(r['segment']), []).append(i)
    recov = []
    for k in sorted(seg)[1:]:
        idx = seg[k]; tgt = B[idx[0]]['target']; run = 0; hit = None
        for j, i in enumerate(idx):
            if B[i]['choice'] == tgt:
                run += 1
                if run >= 3:
                    hit = j - 2; break
            else:
                run = 0
        recov.append(hit if hit is not None else len(idx))

    # The four non-engagement criteria of preregistration 35.3 are produced here as well.
    # An earlier version could only produce win-stay, so each script applied the exclusions
    # for itself, and 007 - which fails the reaction-time criterion - stayed in some analyses.
    n_to = sum(_int(r['timeout']) for r in B)
    n_fast = 0
    for r in B:
        if _int(r['timeout']):
            continue
        try:
            if float(r['rt']) < 0.150:
                n_fast += 1
        except (TypeError, ValueError):
            pass
    n_valid = len(B) - n_to
    _ws = ws_n / ws_d if ws_d else float('nan')
    _rs = []
    if _ws == _ws and _ws < 0.60:
        _rs.append(f'win-stay {_ws:.3f} < .60')
    if len(B) and n_fast / len(B) > 0.20:
        _rs.append(f'RT<150ms {100 * n_fast / len(B):.1f}% > 20%')
    if len(B) and n_to / len(B) > 0.20:
        _rs.append(f'timeouts {100 * n_to / len(B):.1f}% > 20%')
    if n_valid < 96:
        _rs.append(f'valid trials {n_valid} < 96')

    # Reversal manipulation check (main text 3.1) - the rate of choosing the new correct option
    # on trials 6-15 after a reversal. If the manipulation worked it exceeds 0.5. Timeouts are dropped.
    pr = []
    for r0 in rev:
        w = [q for q in B[r0 + 5:r0 + 15] if not _int(q['timeout'])]
        if w:
            pr.append(sum(1.0 for q in w if _int(q['correct'])) / len(w))
    post_rev = float(np.mean(pr)) if pr else float('nan')

    tot = n_persev + n_regress
    return dict(
        post_rev_accuracy=post_rev,
        n_trials=len(B),
        n_timeout=n_to,
        n_fast=n_fast,
        frac_fast=n_fast / len(B) if len(B) else float('nan'),
        n_valid=n_valid,
        excl_prereg=1 if _rs else 0,
        excl_reason=' · '.join(_rs),
        n_persev=n_persev,
        n_regress=n_regress,
        sbi=(n_regress - n_persev) / tot if tot else float('nan'),   # rule 2
        lose_shift=ls_n / ls_d if ls_d else float('nan'),
        win_stay=ws_n / ws_d if ws_d else float('nan'),
        switch_rate=sw_n / sw_d if sw_d else float('nan'),
        accuracy=float(np.mean([_int(r['correct']) for r in B
                                if _int(r['timeout']) == 0])),
        trials_to_recover=float(np.mean(recov)) if recov else float('nan'),
        longest_run=int(max(lens)),
        rev_crossed=sum(1 for r in rev if a < r <= b),
        n_reversals_seen=len(rev),
    )


if __name__ == '__main__':
    B = load()
    if not B:
        raise SystemExit(f'no behavioural CSV found: {RAW}/**/*_behav.csv')
    M = {pid: metrics(rows) for pid, rows in B.items()}
    pids = sorted(M)

    hdr = ('id', 'trials', 'timeout', 'persev', 'regress', 'SBI', 'lose-shift',
           'win-stay', 'accuracy', 'switch', 'recover', 'longest run', 'missed rev', 'reversals')
    key = ('n_trials', 'n_timeout', 'n_persev', 'n_regress', 'sbi', 'lose_shift',
           'win_stay', 'accuracy', 'switch_rate', 'trials_to_recover',
           'longest_run', 'rev_crossed', 'n_reversals_seen')
    print('=' * 100)
    print(f' behavioural indices (n = {len(pids)})')
    print('=' * 100)
    print(f"{hdr[0]:>5}" + ''.join(f'{h:>11}' for h in hdr[1:]))
    for p in pids:
        d = M[p]
        cells = []
        for k in key:
            v = d[k]
            cells.append(f'{v:>11.3f}' if isinstance(v, float) else f'{v:>11}')
        print(f'{p:>5}' + ''.join(cells))

    print('\ndescriptives')
    for k, lab in [('n_persev', 'perseverative errors'), ('n_regress', 'regressive errors'),
                   ('lose_shift', 'lose-shift'), ('win_stay', 'win-stay'),
                   ('accuracy', 'accuracy'), ('sbi', 'switch-bias index'),
                   ('switch_rate', 'switch rate'), ('trials_to_recover', 'trials to recover')]:
        v = np.array([M[p][k] for p in pids], float)
        print(f'  {lab:>24}: mean {v.mean():7.3f} (SD {v.std(ddof=1):6.3f}) - '
              f'range {v.min():.3f}-{v.max():.3f}')

    pv = [M[p]['n_persev'] for p in pids]
    rg = [M[p]['n_regress'] for p in pids]
    sb = [M[p]['sbi'] for p in pids]
    sw = [M[p]['switch_rate'] for p in pids]
    r = stats.spearmanr(pv, rg)
    print(f"\n  perseverative x regressive : rho = {r.statistic:+.3f}, P = {r.pvalue:.4f}")
    print(f"  SBI x switch rate          : r = {stats.pearsonr(sb, sw)[0]:+.3f}")
    print(f"  perseveration-dominant {sum(1 for x in sb if x < 0)} - "
          f"regression-dominant {sum(1 for x in sb if x >= 0)}")

    # Reversal manipulation check - the value reported in main text 3.1
    pr = np.array([M[p]['post_rev_accuracy'] for p in pids
                   if M[p]['post_rev_accuracy'] == M[p]['post_rev_accuracy']])
    if len(pr):
        _lo = min(pids, key=lambda p: M[p]['post_rev_accuracy'])
        print(chr(10) + '  reversal manipulation check - rate of choosing the new correct option on trials 6-15 after a reversal')
        print(f"    n={len(pr)}  mean {pr.mean():.3f} (SD {pr.std(ddof=1):.3f}) - "
              f"range {pr.min():.3f} to {pr.max():.3f}")
        print(f"    above 0.5 in {(pr > 0.5).sum()}/{len(pr)} - "
              f"lowest participant {_lo} ({M[_lo]['post_rev_accuracy']:.3f})")

    short = [p for p in pids if M[p]['n_reversals_seen'] < 5]
    if short:
        print('\n  Caution - participants who did not experience all five reversals:')
        for p in short:
            print(f"    {p}: {M[p]['n_trials']} trials - {M[p]['n_reversals_seen']} reversals")

    json.dump(M, open(f'{DATA}/behav_metrics.json', 'w'), ensure_ascii=False, indent=1)
    print(f'\nwritten: {DATA}/behav_metrics.json')
