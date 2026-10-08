"""cohort.py - the sample and index composition shared by every script.

The sample rule lives in one place only. An earlier version had each script read the
analysis sheet of the internal workbook to build the sample and the behavioural indices, and
that file is participant data and is not published. Anyone who had only the repository could
therefore not run the analysis to the end. Here only the pipeline outputs are used.

  spectro.json        resting-state spectra (produced by spectro.py)
  behav_metrics.json  behavioural indices (produced by behav_metrics.py)

Three rules for the EEG sample
  1. both resting runs pass the quality criteria - 007 drops out with only run 1 surviving
  2. win-stay >= .60 of preregistration 35.3 - 008 (.595) and 009 (.538) drop out
  3. both the aperiodic exponent and the alpha peak amplitude of the frontal cluster are produced

These three rules reproduce the manuscript's 9 participants (001-006, 012, 014, 016) and the
main-text values .803 / .717 / -.723 / .767 / -.850 exactly.

The behavioural sample is every participant who completed the task (12) and nothing is excluded.

The demographic information (age, sex, disability type, psychotropic medication) is participant
data that the pipeline does not produce, so it is not in the repository. demographics() returns
values only when the optional input data/demographics.csv is present and returns None
otherwise. An analysis that uses it (the confound check, for instance) skips and says so.
"""
import csv
import json
import os

import numpy as np

BEH_KEYS = ('lose_shift', 'accuracy', 'sbi', 'n_persev', 'n_regress',
            'switch_rate', 'win_stay', 'trials_to_recover', 'n_trials',
            'n_timeout', 'n_fast', 'frac_fast', 'n_valid',
            'excl_prereg', 'excl_reason')

DEMO_FILE = 'demographics.csv'
DEMO_KEYS = ('age', 'sex', 'disability', 'psychotropic_med')


def _front(spectro, pid, key):
    """Run mean of the frontal cluster values. key is 'exponent' or 'alpha'."""
    v = []
    for r in spectro.get(pid, []):
        if r.get('dropped') or not r.get('front'):
            continue
        f = r['front']
        if key == 'exponent':
            v.append(f['exponent'])
        elif f.get('alpha_amp') is not None:
            v.append(f['alpha_amp'])
    return float(np.mean(v)) if v else None


def n_live(spectro, pid):
    """Number of resting runs that passed the quality criteria."""
    return sum(1 for r in spectro.get(pid, []) if not r.get('dropped'))


def load(data_dir, verbose=True):
    """Returns (behav, eeg).

    behav - every participant who completed the task. Each item is a dict holding id and the
            behavioural indices. Where EEG values exist, front_exp and front_alpha are filled too (None otherwise).
    eeg   - the subset passing the three rules above. This is the sample for the correlations and scatter plots.
    """
    spectro = json.load(open(f'{data_dir}/spectro.json', encoding='utf-8'))
    bm = json.load(open(f'{data_dir}/behav_metrics.json', encoding='utf-8'))
    behav = []
    for pid in sorted(bm):
        b = bm[pid]
        d = dict(id=pid,
                 front_exp=_front(spectro, pid, 'exponent'),
                 front_alpha=_front(spectro, pid, 'alpha'),
                 n_rest_runs=n_live(spectro, pid))
        d.update({k: b.get(k) for k in BEH_KEYS})
        ws = b.get('win_stay')
        d['excl_winstay'] = 0 if (ws is not None and ws >= .60) else 1
        # The EEG values of a participant with only one resting run (007) are not used.
        # This is what "did not meet the EEG quality criteria" in manuscript 3.1 refers to; in
        # the earlier version the workbook's front_exp column was empty and gave the same result.
        if d['n_rest_runs'] < 2:
            d['front_exp'] = None
            d['front_alpha'] = None
        behav.append(d)
    eeg = [d for d in behav
           if d['n_rest_runs'] >= 2 and not d['excl_winstay']
           and d['front_exp'] is not None and d['front_alpha'] is not None]
    if verbose:
        print(f'[cohort] behaviour {len(behav)} - EEG {len(eeg)}: '
              + ' '.join(d['id'] for d in eeg))
    return behav, eeg


def by_id(data_dir, verbose=False):
    """Returns the behavioural sample as a dict keyed by participant number."""
    behav, _ = load(data_dir, verbose=verbose)
    return {d['id']: d for d in behav}


def prereg_keep(data_dir, verbose=True):
    """The participants passing all four criteria of preregistration 35.3.

    win-stay < .60, RT<150 ms > 20%, timeouts > 20%, valid trials < 96.
    All four are produced by behav_metrics.py. An earlier version could use win-stay only, so
    the exclusions differed between scripts and 007 - which fails the reaction-time criterion -
    stayed in some analyses. Exclusion happens in this one function only.
    """
    behav, _ = load(data_dir, verbose=False)
    keep = [d['id'] for d in behav if not d.get('excl_prereg')]
    if verbose:
        out = [d for d in behav if d.get('excl_prereg')]
        print(f'[cohort] passing preregistration 35.3: {len(keep)}: ' + ' '.join(keep))
        for d in out:
            print(f"         excluded {d['id']} - {d.get('excl_reason')}")
    return keep


def demographics(data_dir, verbose=True):
    """Reads the optional input data/demographics.csv. None when absent.

    Columns: id, age, sex, disability, psychotropic_med
    This file is participant data and is not included in the repository.
    """
    path = os.path.join(str(data_dir), DEMO_FILE)
    if not os.path.exists(path):
        if verbose:
            print(f'[cohort] {DEMO_FILE} not found - skipping the parts that need '
                  f'the demographic information.')
        return None
    out = {}
    with open(path, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            pid = str(r.get('id') or '').strip().zfill(3)
            if pid:
                out[pid] = {k: (r.get(k) or None) for k in DEMO_KEYS}
    if verbose:
        print(f'[cohort] read demographic information for {len(out)} participants from {DEMO_FILE}.')
    return out
