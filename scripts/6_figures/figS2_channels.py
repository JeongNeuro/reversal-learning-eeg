# -*- coding: utf-8 -*-
"""figS2_channels.py - Supplementary Figure S1. Per-channel alpha relative power and aperiodic exponent

Reproduces the layout of the published supplementary figure, which is numbered S1 in the
manuscript. The S2 in this filename is an earlier figure number, kept so that the output
names below do not change. No code survives for the original, so it is rebuilt here.

  a  group mean of per-channel alpha relative power. Red is the two occipital electrodes (O1, O2)
  b  group mean of the per-channel aperiodic exponent. Red is the two temporal electrodes (T7, T8)

The sample is the 15 participants of chan_indices.json (those with a per-channel resting fit).
The O1 .364 and O2 .354, frontal 1.30-1.45 and temporal 0.74-1.13 quoted in main text 3.6 come
from this sample. Reducing it to the 9 participants of preregistration 35.3 changes the values
(O1 .394), so matching the main text requires the 15.

The dimensions were measured in pixels from the published figure (3888 x 1004, tick text 36 px,
axis line 4 px, bar width 77 px, spacing 116 px).

Input   data/chan_indices.json
Output  outputs/FigS2_channels.png  outputs/figS2_channels.json
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import DATA, OUT                                 # noqa: E402

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
W_PX, H_PX = 3888, 1004          # internal coordinate ratio
PX = _N.pt_per_px(W_PX)
FS_TICK = _N.PT_TICK              # ticks
FS_AXIS = _N.PT_AXIS              # axis titles
FS_LET = _N.PT_PANEL              # panel letters
SPINE = 4 * PX
BARW = 77 / 116                   # bar width / spacing
INK, RED = '#1A1A1A', '#E01B1B'

CH = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
      'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
OCC = ('O1', 'O2')                # the red of a - occipital
TEMP = ('T7', 'T8')               # the red of b - temporal

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

CI = json.load(io.open(os.path.join(str(DATA), 'chan_indices.json'),
                       encoding='utf-8'))
ids = sorted(CI)


def chan_mean(key):
    out, n = {}, {}
    for c in CH:
        v = [CI[p][c][key] for p in ids
             if c in CI[p] and CI[p][c] and CI[p][c].get(key) is not None]
        if v:
            out[c] = float(np.mean(v))
            n[c] = len(v)
    return out, n


alpha, n_alpha = chan_mean('alpha_rel')
expo, n_expo = chan_mean('exponent')

fig, axes = plt.subplots(1, 2, figsize=_N.figsize(W_PX, H_PX), dpi=DPI)
fig.subplots_adjust(left=.052, right=.988, top=.915, bottom=.225, wspace=.185)

report = {'n_participants': len(ids), 'ids': ids, 'panels': {}}
for ax, (letter, vals, ns, hot, ylab, ticks) in zip(axes, [
        ('a', alpha, n_alpha, OCC, 'Alpha relative power', [0, .1, .2, .3]),
        ('b', expo, n_expo, TEMP, 'Aperiodic exponent',
         [0, .25, .50, .75, 1.00, 1.25, 1.50])]):
    order = sorted(vals, key=lambda c: -vals[c])
    y = [vals[c] for c in order]
    col = [RED if c in hot else INK for c in order]
    ax.bar(range(len(order)), y, width=BARW, color=col, lw=0, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=FS_TICK)
    ax.set_yticks(ticks)
    ax.set_ylabel(ylab, fontsize=FS_AXIS, color=INK, labelpad=FS_AXIS * .55)
    ax.set_xlim(-.72, len(order) - .28)
    ax.set_ylim(0, max(y) * 1.06)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(SPINE)
    ax.tick_params(colors=INK, labelsize=FS_TICK, width=SPINE,
                   length=FS_TICK * .40, pad=FS_TICK * .35)
    ax.text(-.105, 1.10, letter, transform=ax.transAxes, fontsize=FS_LET,
            fontweight='bold', color=INK, ha='left', va='top')
    report['panels'][letter] = dict(
        ylabel=ylab, highlighted=list(hot),
        values={c: round(vals[c], 4) for c in order},
        n={c: ns[c] for c in order})

dst = os.path.join(str(OUT), 'FigS2_channels.png')
fig.savefig(dst, dpi=DPI, facecolor='white')
plt.close(fig)
json.dump(report, io.open(os.path.join(str(OUT), 'figS2_channels.json'), 'w',
                          encoding='utf-8'), ensure_ascii=False, indent=1)

print(f'[figS2] {len(ids)} participants')
print('  alpha relative power  O1 %.3f - O2 %.3f' % (alpha['O1'], alpha['O2']))
fr = [expo[c] for c in ('AF3', 'AF4', 'F3', 'F4', 'F7', 'F8') if c in expo]
tp = [expo[c] for c in TEMP if c in expo]
print('  aperiodic exponent    frontal %.2f-%.2f - temporal %.2f-%.2f'
      % (min(fr), max(fr), min(tp), max(tp)))
print('written ->', dst)
