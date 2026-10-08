"""
rbq.py - RBQ-2A subscale scoring and its association with the behavioural indices (preregistration 18.7 and 36.8)

Preregistration 36.8 moved this family from the confirmatory plan to exploratory and in exchange
required that **all four associations be reported in full, under both scorings**.

  "The four associations between the insistence-on-sameness subscale and SBI,
   perseverative count, regressive count and kappa are computed and reported in
   full, in the same estimation format and for both published subscale scorings."

The item composition fixed by preregistration 18.7 (the 11-17 and 19 of the researcher's earlier
plan match no published solution and are not used)

  primary      Barrett et al. (2018) insistence on sameness - items 1, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19
  sensitivity  Barrett et al. (2015) insistence on sameness - items 12, 13, 14, 15, 16, 17, 19
  secondary    20-item total (including item 20, following 2015)
  exploratory  repetitive sensory and motor behaviour 2, 3, 4, 5, 6, 10 (2018); repetitive motor behaviour 1-7, 10, 11 (2015)

kappa was withdrawn because the reinforcement-learning model was not identified (Appendix 2).
Of the four indices the registration fixed, the kappa slot is left empty and the lose-shift the manuscript uses instead is reported alongside.

Input   data/questionnaire.csv  (optional) - columns: id, i01 ... i20
        It is participant data and is not in the repository. Without it this script does nothing and exits.
Output  outputs/RBQ2A_associations.csv
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, OUT

import csv, math, warnings
import numpy as np
from scipy import stats
warnings.filterwarnings("ignore")

import cohort

QFILE = 'questionnaire.csv'
QPATH = os.path.join(str(DATA), QFILE)

# preregistration 18.7
SCORING = [
    ('IS_2018', 'insistence on sameness - Barrett 2018 (primary)',
     [1, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
    ('IS_2015', 'insistence on sameness - Barrett 2015 (sensitivity)',
     [12, 13, 14, 15, 16, 17, 19]),
    ('TOTAL', '20-item total (secondary)', list(range(1, 21))),
    ('RSMB_2018', 'repetitive sensory and motor behaviour - 2018 (exploratory)', [2, 3, 4, 5, 6, 10]),
    ('RMB_2015', 'repetitive motor behaviour - 2015 (exploratory)', [1, 2, 3, 4, 5, 6, 7, 10, 11]),
]

# The co-primary indices of preregistration 18.2. kappa is withdrawn in Appendix 2.
BEH = [('sbi', 'switch-bias index (SBI)'),
       ('n_persev', 'perseverative error count'),
       ('n_regress', 'regressive error count'),
       ('kappa', 'kappa - the reinforcement-learning stickiness parameter'),
       ('lose_shift', 'lose-shift (the index the manuscript uses in place of kappa)')]

MISS_MAX = 0.20      # preregistration - no score is produced if more than 20% of contributing items are missing
FLOOR = 8            # preregistration 35.9 - a coefficient is produced only at 8 participants or more


def boot_rho(x, y, nb=10000, seed=1):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float); o = []
    for _ in range(nb):
        i = rng.integers(0, len(x), len(x))
        if len(set(i.tolist())) < 3:
            continue
        v = stats.spearmanr(x[i], y[i]).statistic
        if np.isfinite(v):
            o.append(v)
    return (np.percentile(o, 2.5), np.percentile(o, 97.5)) if o else (np.nan, np.nan)


def cronbach(M):
    """M: (participants x items). Rows with any missing value are dropped."""
    M = np.asarray(M, float)
    M = M[~np.isnan(M).any(axis=1)]
    k = M.shape[1]
    if M.shape[0] < 3 or k < 2:
        return float('nan'), M.shape[0]
    tot = M.sum(axis=1)
    if tot.var(ddof=1) == 0:
        return float('nan'), M.shape[0]
    a = k / (k - 1) * (1 - M.var(axis=0, ddof=1).sum() / tot.var(ddof=1))
    return float(a), M.shape[0]


def main():
    if not os.path.exists(QPATH):
        print(f"{QFILE} not found - skipping the RBQ-2A analysis.")
        print(f"To run it, put the following columns in data/{QFILE}:")
        print("  id, i01, i02, ... , i20   (responses 1-3, missing values blank)")
        return

    Q = {}
    with open(QPATH, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            pid = str(r.get('id') or '').strip().zfill(3)
            if not pid:
                continue
            v = []
            for i in range(1, 21):
                s = (r.get(f'i{i:02d}') or '').strip()
                try:
                    v.append(float(s))
                except ValueError:
                    v.append(np.nan)
            Q[pid] = np.array(v, float)
    print(f"[rbq] read item responses for {len(Q)} participants from {QFILE}.")

    B = cohort.by_id(str(DATA))
    pids = sorted(p for p in Q if p in B)
    print(f"[rbq] {len(pids)} participants with both questionnaire and behavioural indices: {' '.join(pids)}")
    print()
    print("Response scale, note. Preregistration 18.7 states that the published instrument scores")
    print("items 1-6 and 13-19 on four points and 7-12 and 20 on three, and that scoring collapses")
    print("the top two categories of the four-point items. This study administered every item on")
    print("three points (manuscript 2.4), so the scores below are not on the published scale.")
    print()

    rows = []
    for key, lab, items in SCORING:
        idx = [i - 1 for i in items]
        score, keep = {}, []
        for p in pids:
            v = Q[p][idx]
            nmiss = int(np.isnan(v).sum())
            if nmiss > MISS_MAX * len(idx):
                continue                      # the preregistered missing-data rule
            score[p] = float(np.nansum(v)) if nmiss else float(v.sum())
            keep.append(p)
        a, na = cronbach([Q[p][idx] for p in keep])
        a_all, na_all = cronbach([Q[p][idx] for p in sorted(Q)])
        print("=" * 78)
        print(f" {lab}   {len(items)} items - n = {len(keep)}")
        print("=" * 78)
        print(f"  score  mean {np.mean(list(score.values())):.2f} "
              f"(SD {np.std(list(score.values()), ddof=1):.2f}) · "
              f"range {min(score.values()):.0f}-{max(score.values()):.0f} - "
              f"possible range {len(items)}-{3*len(items)}")
        # The alpha the manuscript quotes comes from everyone who answered the questionnaire, so both are reported
        print(f"  internal consistency  Cronbach alpha = {a:.3f} (analysis sample n={na}) - "
              f"{a_all:.3f} (all questionnaire respondents n={na_all})")
        print()
        print(f"  {'behavioural index':<52}{'n':>4}{'rho':>9}{'95% CI':>22}")
        for bk, blab in BEH:
            xs, ys = [], []
            for p in keep:
                bv = B[p].get(bk)
                if bv is None or (isinstance(bv, float) and math.isnan(bv)):
                    continue
                xs.append(score[p]); ys.append(float(bv))
            if len(xs) < 3:
                print(f"  {blab:<52}{len(xs):>4}     -     "
                      f"{'(no data - see Appendix 2)' if bk == 'kappa' else '(insufficient data)'}")
                rows.append(dict(scoring=key, behav=bk, n=len(xs),
                                 rho='', lo='', hi='', note='no data'))
                continue
            r = stats.spearmanr(xs, ys).statistic
            if len(xs) < FLOOR:
                print(f"  {blab:<52}{len(xs):>4}{r:>+9.3f}"
                      f"{'  below the floor - coefficient only':>38}")
                rows.append(dict(scoring=key, behav=bk, n=len(xs), rho=round(r, 4),
                                 lo='', hi='', note='below the registered floor of 8'))
            else:
                lo, hi = boot_rho(xs, ys)
                print(f"  {blab:<52}{len(xs):>4}{r:>+9.3f}"
                      f"      [{lo:+.3f}, {hi:+.3f}]")
                rows.append(dict(scoring=key, behav=bk, n=len(xs), rho=round(r, 4),
                                 lo=round(lo, 4), hi=round(hi, 4), note=''))
        print()

    os.makedirs(str(OUT), exist_ok=True)
    dst = os.path.join(str(OUT), 'RBQ2A_associations.csv')
    with open(dst, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=['scoring', 'behav', 'n', 'rho',
                                            'lo', 'hi', 'note'])
        wr.writeheader(); wr.writerows(rows)
    print(f"written -> {dst}")


if __name__ == '__main__':
    main()
