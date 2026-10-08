# -*- coding: utf-8 -*-
"""fig3_errors.py - Figure 3. Distribution of error types and individual differences

Reproduces the layout of Figure 3 as published. No code survives for the original, so it is
rebuilt from the repository (the same approach as Figures 2, 4, 5 and S2).

  a  the trial record per participant. One row is one participant and one cell is one trial.
     Red perseverative errors, blue regressive errors, grey correct and acquisition errors, white timeouts.
     The vertical lines are the reversals, and participants are ordered by switch-bias index.
  b  error composition per participant (perseverative on the left, regressive on the right)

The per-trial classification is taken straight from the error_type column of the raw CSV. That
column was written by the task program, and the perseverative and regressive counts of
behav_metrics.py count the same column. So the cell counts of a and the bar lengths of b agree by definition.

The colours were taken from the published figure - perseverative #D42A2A, regressive #1668D9, correct #E8E8E8.

Input   data/raw/*_behav.csv  data/raw/*_meta.json  data/behav_metrics.json
Output  outputs/Fig3_errors.png  outputs/fig3_errors.json
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT                            # noqa: E402

import csv                                                  # noqa: E402
import glob                                                 # noqa: E402
import io                                                   # noqa: E402
import json                                                 # noqa: E402

import numpy as np                                          # noqa: E402
import matplotlib                                           # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                             # noqa: E402

# Final width fixed at 173 mm; text uses only the three steps of natstyle.
# The internal coordinate ratio is left as it was, and line widths are converted with pt_per_px.
import natstyle as _N                      # noqa: E402

DPI = _N.FIG_DPI
W_PX, H_PX = 3868, 1405          # internal coordinate ratio
PX = _N.pt_per_px(W_PX)
FS_TICK = _N.PT_TICK
FS_AXIS = _N.PT_AXIS
FS_LET = _N.PT_PANEL
SPINE = 4 * PX

RED, BLUE, GREY = '#D42A2A', '#1668D9', '#E8E8E8'
INK, LINE = '#1A1A1A', '#7A7A7A'
N_TRIAL = 120

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

BM = json.load(io.open(os.path.join(str(DATA), 'behav_metrics.json'),
                       encoding='utf-8'))


def trials(pid):
    """error_type per trial. None for a timeout."""
    hit = sorted(glob.glob(os.path.join(str(RAW), '**', f'*_{pid}_behav.csv'),
                           recursive=True))
    if not hit:
        return None
    rows = list(csv.DictReader(io.open(hit[0], encoding='utf-8-sig')))
    out = []
    for r in rows:
        if str(r.get('timeout', '0')).strip() in ('1', '1.0', 'True'):
            out.append(None)
        else:
            out.append(r.get('error_type') or 'correct')
    return out


def reversals():
    """The reversal points - the value meta.json recorded. Every participant had the same schedule."""
    for p in sorted(glob.glob(os.path.join(str(RAW), '**', '*_meta.json'),
                              recursive=True)):
        d = json.load(io.open(p, encoding='utf-8'))
        if d.get('reversal_trials'):
            return list(d['reversal_trials'])
    return []


pids = [p for p in sorted(BM) if trials(p)]
# switch-bias index descending - regression-dominant at the top, perseveration-dominant at the bottom
pids.sort(key=lambda p: -BM[p]['sbi'])
REV = reversals()
ET = {p: trials(p) for p in pids}

fig, axes = plt.subplots(1, 2, figsize=_N.figsize(W_PX, H_PX), dpi=DPI,
                         gridspec_kw=dict(width_ratios=[1.30, 1.0]))
fig.subplots_adjust(left=.068, right=.985, top=.905, bottom=.140, wspace=.235)

# -- a  trial record ----------------------------------------------
axa = axes[0]
COLOR = {'perseverative': RED, 'regressive': BLUE,
         'correct': GREY, 'acquisition': GREY}
for i, p in enumerate(pids):
    for t, e in enumerate(ET[p]):
        if e is None:
            continue                     # timeout - left as a white cell
        axa.add_patch(plt.Rectangle((t, i - .40), 1.0, .80,
                                    color=COLOR.get(e, GREY), lw=0, zorder=2))
for r in REV:
    axa.axvline(r, color=LINE, lw=SPINE * 1.1, zorder=4)
axa.set_xlim(0, N_TRIAL)
axa.set_ylim(len(pids) - .5, -.5)
axa.set_yticks(range(len(pids)))
axa.set_yticklabels(pids, fontsize=FS_TICK)
axa.set_xticks([0, 20, 40, 60, 80, 100, 120])
axa.set_xlabel('Trial', fontsize=FS_AXIS, color=INK, labelpad=FS_AXIS * .55)
for s in ('top', 'right', 'left'):
    axa.spines[s].set_visible(False)
axa.spines['bottom'].set_color(INK)
axa.spines['bottom'].set_linewidth(SPINE)
axa.tick_params(colors=INK, labelsize=FS_TICK, width=SPINE,
                length=FS_TICK * .40, pad=FS_TICK * .35)
axa.tick_params(axis='y', length=0)
_ha = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (RED, BLUE, GREY)]
axa.legend(_ha, ['perseverative', 'regressive', 'correct'],
           loc='lower center', bbox_to_anchor=(.62, 1.005), frameon=False,
           fontsize=FS_TICK, ncol=3, handlelength=1.0, handleheight=1.0,
           columnspacing=1.6, handletextpad=.7, borderpad=0)
axa.text(-.105, 1.085, 'a', transform=axa.transAxes, fontsize=FS_LET,
         fontweight='bold', color=INK, ha='left', va='top')

# -- b  error composition -----------------------------------------
axb = axes[1]
pv = np.array([BM[p]['n_persev'] for p in pids], float)
rg = np.array([BM[p]['n_regress'] for p in pids], float)
axb.barh(range(len(pids)), -pv, height=.66, color=RED, lw=0, zorder=3)
axb.barh(range(len(pids)), rg, height=.66, color=BLUE, lw=0, zorder=3)
off = max(pv.max(), rg.max()) * .035
for i in range(len(pids)):
    axb.text(-pv[i] - off, i, '%d' % pv[i], ha='right', va='center',
             fontsize=FS_TICK, color=RED, zorder=5)
    axb.text(rg[i] + off, i, '%d' % rg[i], ha='left', va='center',
             fontsize=FS_TICK, color=BLUE, zorder=5)
axb.axvline(0, color=INK, lw=SPINE, zorder=4)
axb.set_ylim(len(pids) - .5, -.5)
axb.set_yticks(range(len(pids)))
axb.set_yticklabels(pids, fontsize=FS_TICK)
lim = max(pv.max(), rg.max()) * 1.18
axb.set_xlim(-lim, lim)
axb.set_xticks([-60, -40, -20, 0, 20, 40])
axb.set_xticklabels(['60', '40', '20', '0', '20', '40'], fontsize=FS_TICK)
axb.set_xlabel('Number of errors', fontsize=FS_AXIS, color=INK,
               labelpad=FS_TICK * .55)
for s in ('top', 'right', 'left'):
    axb.spines[s].set_visible(False)
axb.spines['bottom'].set_color(INK)
axb.spines['bottom'].set_linewidth(SPINE)
axb.tick_params(colors=INK, labelsize=FS_TICK, width=SPINE,
                length=FS_TICK * .40, pad=FS_TICK * .35)
axb.tick_params(axis='y', length=0)
_hb = [plt.Rectangle((0, 0), 1, 1, color=RED), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
axb.legend(_hb, ['Perseverative', 'Regressive'], loc='upper left',
           bbox_to_anchor=(.02, 1.02), frameon=False, fontsize=FS_TICK,
           handlelength=1.0, handleheight=1.0, labelspacing=.40, borderpad=0)
axb.text(-.135, 1.085, 'b', transform=axb.transAxes, fontsize=FS_LET,
         fontweight='bold', color=INK, ha='left', va='top')

dst = os.path.join(str(OUT), 'Fig3_errors.png')
fig.savefig(dst, dpi=DPI, facecolor='white')
plt.close(fig)

rep = dict(order=pids, reversals=REV,
           errors={p: dict(perseverative=int(BM[p]['n_persev']),
                           regressive=int(BM[p]['n_regress']),
                           sbi=round(float(BM[p]['sbi']), 4),
                           n_timeout=int(sum(1 for e in ET[p] if e is None)))
                   for p in pids})
json.dump(rep, io.open(os.path.join(str(OUT), 'fig3_errors.json'), 'w',
                       encoding='utf-8'), ensure_ascii=False, indent=1)
print('[fig3] %d participants - reversals %s' % (len(pids), REV))
for p in pids:
    print('  %-4s perseverative %2d - regressive %2d - SBI %+.3f'
          % (p, BM[p]['n_persev'], BM[p]['n_regress'], BM[p]['sbi']))
print('written ->', dst)
