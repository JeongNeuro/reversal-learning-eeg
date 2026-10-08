# -*- coding: utf-8 -*-
"""make_schedule.py - writes the trial schedule of seed 0 to a machine-readable file.

Publishing this is a commitment made in section 38-2 of the preregistration.

    "The reconstructed seed-0 schedule as a machine-readable file: per-trial
     segment index, correct stimulus, reward availability, and left/right
     position for all 120 trials, together with the script that regenerates it."

Every participant experienced the **same schedule** generated from the same seed (0), so this
one file fully specifies the stimulus order that every participant received.

    python task/make_schedule.py

The schedule generation rule is not restated here. build_trials of prl_task.py is called
directly - if the rule were written in two places the two would eventually disagree.
prl_task.py imports PsychoPy at the top, though, so it cannot be imported whole. Only the
parts needed for schedule generation (CFG, build_schedule, build_trials) are read out of it.
It runs without PsychoPy.
"""
import ast
import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.join(HERE, 'prl_task.py')
NEEDED = ('CFG', 'build_schedule', 'build_trials')


def load_from_task():
    """Pull just the names needed for schedule generation out of prl_task.py."""
    tree = ast.parse(open(TASK, encoding='utf-8').read())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in NEEDED:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in NEEDED:
                    keep.append(node)
    got = set()
    for n in keep:
        got.add(n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id)
    missing = set(NEEDED) - got
    if missing:
        raise SystemExit(f'could not find {sorted(missing)} in prl_task.py. '
                         'If the task script has changed, check this script too.')
    ns = {'__name__': 'prl_task_schedule'}
    exec(compile(ast.Module(body=keep, type_ignores=[]), TASK, 'exec'), ns)
    return ns


def main():
    ns = load_from_task()
    cfg = ns['CFG']
    rng = random.Random(0)                      # the value the real sessions used
    trials, rev_at = ns['build_trials'](cfg, rng)

    out = os.path.join(HERE, 'seed0_schedule.csv')
    cols = ['trial', 'block', 'segment', 'trial_in_seg', 'target',
            'is_reversal_trial', 'rewarded_if_correct', 'left_stim']
    with open(out, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in trials:
            w.writerow({c: t[c] for c in cols})

    segs = [sum(1 for t in trials if t['segment'] == s)
            for s in sorted({t['segment'] for t in trials})]
    n_rew = sum(t['rewarded_if_correct'] for t in trials)
    n_left_a = sum(1 for t in trials if t['left_stim'] == 'A')

    print(f'wrote {len(trials)} trials to {out}.\n')
    print(f'  segment lengths   {"  ".join(map(str, segs))}  (sum {sum(segs)})')
    print(f'  reversal trials   {"  ".join(map(str, rev_at))}')
    print(f'  first correct     {trials[0]["target"]}')
    print(f'  rewards scheduled {n_rew} of {len(trials)} trials = {n_rew/len(trials):.3f}')
    print(f'                    (the setting is p_reward = {cfg["p_reward"]:.2f}; the rate '
          'realised in this fixed random sequence differs from it)')
    print(f'  A on the left     {n_left_a} times')

    verify(trials)


def verify(trials):
    """If the raw data are present, check against the schedule actually presented."""
    sys.path.insert(0, os.path.dirname(HERE))
    try:
        from paths import RAW
    except Exception:
        return
    import glob
    files = sorted(glob.glob(os.path.join(str(RAW), '**', '*_behav.csv'),
                             recursive=True))
    if not files:
        print('\n  (no raw data, so the check is skipped.)')
        return

    cols = ['segment', 'trial_in_seg', 'target', 'is_reversal_trial',
            'rewarded_if_correct', 'left_stim']
    print(f'\n  checking against {len(files)} real sessions')
    bad = 0
    for p in files:
        rows = list(csv.DictReader(open(p, encoding='utf-8-sig')))
        pid = os.path.basename(p).split('_')[-2]
        n = min(len(rows), len(trials))
        diff = [i for i in range(n)
                if any(str(rows[i][c]) != str(trials[i][c]) for c in cols)]
        mark = 'match' if not diff else f'*** {len(diff)} trials differ ***'
        if diff:
            bad += 1
        print(f'    {pid}  {n:>3} trials  {mark}')
    print('\n  ' + ('all sessions match - every participant experienced the same schedule.'
                    if not bad else f'{bad} sessions disagree. This needs checking.'))


if __name__ == '__main__':
    main()
