# -*- coding: utf-8 -*-
"""fig1_task.py - Figure 1. Task structure

Reproduces the layout of Figure 1 as published. No code survives for the original, so it is
rebuilt from the repository (the same approach as Figures 2, 3, 4, 5 and S2).

  a  the resting measurement and the flow of one trial - four screens laid over one another
  b  the reversal schedule over 120 trials. A shaded span is where that stimulus is rewarded,
     and the number is the rate actually rewarded when the correct option was chosen (incorrect is 0%)

The reward rates are not hard-coded. The schedule the task program generated from a fixed
seed is in the rewarded_if_correct column of the raw CSV, and the mean is taken per span.
The schedule is the same for every participant, so one participant's file is enough.

The reversal points are the reversal_trials of meta.json (zero-based indices). In the figure
the span boundaries and axis ticks are written as the following trial number (+1): the rule
changes from the trial after index 19, so the boundary is 20.

**The coordinates are the pixel values measured from the published figure.** The whole figure
sits on one axes drawn in the same pixel coordinates as the manuscript. That keeps circles
from becoming ellipses and keeps the spacing of the stacked screens as published.

Input   data/raw/*_behav.csv  data/raw/*_meta.json
Output  outputs/Fig1_task.png  outputs/fig1_task.json
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import RAW, OUT                                  # noqa: E402

import csv                                                  # noqa: E402
import glob                                                 # noqa: E402
import io                                                   # noqa: E402
import json                                                 # noqa: E402

import numpy as np                                          # noqa: E402
import matplotlib                                           # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                             # noqa: E402
from matplotlib.patches import Rectangle, Circle            # noqa: E402

# Final width fixed at 173 mm; text uses only the three steps of natstyle.
# W and H serve only as internal coordinate ratios; line widths are converted with pt_per_px.
import natstyle as _N                      # noqa: E402

DPI = _N.FIG_DPI
W, H = 4093, 1205                 # internal coordinate ratio
PX = _N.pt_per_px(W)
FS = _N.PT_TICK                   # ticks and annotations
FS_AXIS = _N.PT_AXIS              # axis titles
FS_LET = _N.PT_PANEL              # panel letters
LW = 5 * PX                       # 5 px border

# colours taken from the manuscript
BLUE_BAR, ORANGE_BAR = '#377CDC', '#E29335'     # the bars of b
BLUE_STIM, ORANGE_STIM = '#1668D9', '#E08214'   # the stimuli of a
EMPTY, SCREEN, EDGE = '#F2F2F2', '#F4F4F4', '#7A7A7A'
INK, GREEN, RED = '#231F20', '#1B7837', '#D42A2A'

# a - the four screens (manuscript pixels)
SX, SY, SW, SH = 195, 156, 457, 310
DX, DY = 329, 226                 # gap between screens
CIRC_X, SQ_X, MID_Y = 125, 330, 155     # stimulus positions within a screen
CIRC_D, SQ_S = 60, 57
LABELS = ['Rest', 'Choice', 'Selection', 'Feedback']

# b - the schedule (manuscript pixels)
# Empty space was left between panels a and b, so the bars were widened to the left.
BX0, BX1 = 2150, 4040             # horizontal range holding trials 0 to 120
ROW_Y = {'A': 378, 'B': 634}      # top y of each row
ROW_H = 144

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def schedule():
    """Reward rate per span and the reversal points - read from the raw data."""
    beh = sorted(glob.glob(os.path.join(str(RAW), '**', '*_behav.csv'),
                           recursive=True))
    met = sorted(glob.glob(os.path.join(str(RAW), '**', '*_meta.json'),
                           recursive=True))
    if not beh or not met:
        raise SystemExit(f'no raw data found: {RAW}')
    rows = list(csv.DictReader(io.open(beh[0], encoding='utf-8-sig')))
    rev = None
    for p in met:
        d = json.load(io.open(p, encoding='utf-8'))
        if d.get('reversal_trials'):
            rev = list(d['reversal_trials'])
            break
    bounds = [0] + rev + [len(rows)]
    segs = []
    for k in range(len(bounds) - 1):
        a, b = bounds[k], bounds[k + 1]
        seg = rows[a:b]
        segs.append(dict(start=a, end=b, target=seg[0]['target'], n=len(seg),
                         pct=100 * float(np.mean(
                             [int(r['rewarded_if_correct']) for r in seg]))))
    return segs, rev, len(rows)


SEG, REV, N_TRIAL = schedule()
EDGES = [0] + [r + 1 for r in REV] + [N_TRIAL]      # the boundaries drawn in the figure

fig = plt.figure(figsize=_N.figsize(W, H), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)                 # pixel coordinates - 0 at the top
ax.set_aspect('equal')
ax.axis('off')

# ── a ────────────────────────────────────────────────────────────
for k, lab in enumerate(LABELS):
    x, y = SX + DX * k, SY + DY * k
    ax.add_patch(Rectangle((x, y), SW, SH, fc=SCREEN, ec=EDGE, lw=LW,
                           zorder=3 + 2 * k))
    z = 4 + 2 * k
    cy = y + MID_Y
    if k == 0:                                    # fixation
        c = x + SW / 2
        ax.plot([c - 26, c + 26], [cy, cy], color=INK, lw=LW * 1.6, zorder=z,
                solid_capstyle='butt')
        ax.plot([c, c], [cy - 26, cy + 26], color=INK, lw=LW * 1.6, zorder=z,
                solid_capstyle='butt')
    elif k in (1, 2):                             # the two options
        ax.add_patch(Circle((x + CIRC_X, cy), CIRC_D / 2, fc=BLUE_STIM,
                            ec='none', zorder=z))
        ax.add_patch(Rectangle((x + SQ_X - SQ_S / 2, cy - SQ_S / 2), SQ_S,
                               SQ_S, fc=ORANGE_STIM, ec='none', zorder=z))
        if k == 2:                                # choice highlight
            ax.add_patch(Rectangle((x + CIRC_X - 62, cy - 62), 124, 124,
                                   fc='none', ec=INK, lw=LW * 1.9, zorder=z))
    else:                                         # feedback
        # The O and X of the manuscript are larger than the body text (glyph height about 58 px)
        ax.text(x + CIRC_X, cy, 'O', ha='center', va='center',
                fontsize=FS * 58 / 36, color=GREEN, zorder=z)
        ax.text(x + SQ_X, cy, 'X', ha='center', va='center',
                fontsize=FS * 58 / 36, color=RED, zorder=z)
    ax.text(x + SW + 34, cy, lab, ha='left', va='center', fontsize=FS,
            color=INK, zorder=z)
ax.text(78, 58, 'a', fontsize=FS_LET, fontweight='bold', color=INK,
        ha='left', va='top', zorder=20)

# ── b ────────────────────────────────────────────────────────────
def tx(t):
    return BX0 + (BX1 - BX0) * t / N_TRIAL


for y in ROW_Y.values():          # background of the empty spans
    ax.add_patch(Rectangle((BX0, y), BX1 - BX0, ROW_H, fc=EMPTY, ec='none',
                           zorder=2))
for k, s in enumerate(SEG):
    y = ROW_Y[s['target']]
    col = BLUE_BAR if s['target'] == 'A' else ORANGE_BAR
    x0, x1 = tx(EDGES[k]), tx(EDGES[k + 1])
    ax.add_patch(Rectangle((x0, y), x1 - x0, ROW_H, fc=col, ec='none',
                           zorder=3))
    ax.text((x0 + x1) / 2, y + ROW_H / 2, '%d%%' % round(s['pct']),
            ha='center', va='center', fontsize=FS, color='white',
            fontweight='bold', zorder=5)

TOP = ROW_Y['A'] - 60
for r in REV:                     # reversal markers
    x = tx(r + 1)
    ax.plot([x, x], [TOP, ROW_Y['B'] + ROW_H + 26], color=INK, lw=LW * .8,
            ls=(0, (4, 3)), zorder=6)
    ax.plot(x, TOP - 6, marker='v', ms=8.5, mfc=INK, mec=INK, zorder=6)
# Centred directly above the first reversal triangle (trial 20).
ax.text(tx(REV[0] + 1), TOP - 48, 'Rule reversal', ha='center', va='bottom',
        fontsize=FS, color=INK, zorder=6)

ax.add_patch(Circle((BX0 - 96, ROW_Y['A'] + ROW_H / 2), 32, fc=BLUE_STIM,
                    ec=INK, lw=LW * .6, zorder=5))
ax.add_patch(Rectangle((BX0 - 126, ROW_Y['B'] + ROW_H / 2 - 30), 60, 60,
                       fc=ORANGE_STIM, ec=INK, lw=LW * .6, zorder=5))

AXY = ROW_Y['B'] + ROW_H + 26
ax.plot([BX0, BX1], [AXY, AXY], color=INK, lw=LW * .8, zorder=5)
for t in EDGES:
    x = tx(t)
    ax.plot([x, x], [AXY, AXY + 22], color=INK, lw=LW * .8, zorder=5)
    ax.text(x, AXY + 40, str(t), ha='center', va='top', fontsize=FS,
            color=INK, zorder=5)
ax.text((BX0 + BX1) / 2, AXY + 150, 'Trial', ha='center', va='top',
        fontsize=FS_AXIS, color=INK, zorder=5)
ax.text(BX0 - 330, 58, 'b', fontsize=FS_LET, fontweight='bold', color=INK,
        ha='left', va='top', zorder=20)

dst = os.path.join(str(OUT), 'Fig1_task.png')
fig.savefig(dst, dpi=DPI, facecolor='white')
plt.close(fig)

rep = dict(n_trials=N_TRIAL, reversals=REV, drawn_boundaries=EDGES,
           segments=[dict(trials='%d-%d' % (EDGES[k], EDGES[k + 1]),
                          stimulus=('circle' if s['target'] == 'A'
                                    else 'square'),
                          n=s['n'], reward_pct=round(s['pct'], 1),
                          shown='%d%%' % round(s['pct']))
                     for k, s in enumerate(SEG)])
json.dump(rep, io.open(os.path.join(str(OUT), 'fig1_task.json'), 'w',
                       encoding='utf-8'), ensure_ascii=False, indent=1)
print('[fig1] reversal indices %s - figure boundaries %s' % (REV, EDGES))
for k, s in enumerate(SEG):
    print('  %3d-%-3d %s  n=%2d  reward rate %.1f%% -> %d%%'
          % (EDGES[k], EDGES[k + 1], s['target'], s['n'], s['pct'],
             round(s['pct'])))
print('written ->', dst)
