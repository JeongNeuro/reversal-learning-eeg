"""figM_flow.py - Figure 2. Session procedure and participant flow

Reproduces the layout of Figure 2 as published (one horizontal row with the exclusion boxes
hung off it by dashed lines). An earlier version was a vertical flowchart with diamonds and
Yes/No, which is structurally different from the published figure and did not reproduce it.

The counts are computed from the data, not hard-coded.

  withdrew          fewer than two resting runs recorded                011
  practice not passed  run 2 recorded but no task data                   010, 013, 015
  EEG quality       fewer than two runs passed the quality criteria      007
  win-stay          win-stay < .60 of preregistration 35.3              008, 009

The flowchart is sequential. A participant dropped at an earlier gate is not counted again at
a later one. 007 also meets the reaction-time criterion of 35.3, but drops out at the earlier
EEG-quality gate and is counted only there (the same account as main text 3.1).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import DATA, OUT                                 # noqa: E402

import json                                                 # noqa: E402

import matplotlib                                           # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                             # noqa: E402
from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch  # noqa: E402

import cohort                                               # noqa: E402

# -- specification (the same pixel size as published Figure 2) ---
# Final width fixed at 173 mm; text uses only the three steps of natstyle.
# The internal coordinate ratio is left as it was, and line widths are converted with pt_per_px.
import natstyle as _N                      # noqa: E402

DPI = _N.FIG_DPI
W_PX, H_PX = 4134, 1014          # internal coordinate ratio
PX = _N.pt_per_px(W_PX)
INK, GREY, FILL = '#1A1A1A', '#666666', '#EFEFEF'
# The dimensions below were measured in pixels from published Figure 2.
#   border 6 px, text 36 px, box 485 x 217 px, ellipse 466 x 217 px
#   exclusion box 470 x 179 px, arrows stand 17 px clear of a box
LW = 6 * PX
FS = _N.PT_TICK
LS = 1.47                    # line spacing - 53 px between baselines in the manuscript
SHRINK = 17 * PX             # gap between an arrow and a box (pt)

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# -- count the participants from the data -------------------------
SP = json.load(open(f'{DATA}/spectro.json', encoding='utf-8'))
BM = json.load(open(f'{DATA}/behav_metrics.json', encoding='utf-8'))
keep = set(cohort.prereg_keep(str(DATA), verbose=False))

n_rec = len(SP)
withdrew = [p for p in SP if len(SP[p]) < 2]
done = [p for p in SP if len(SP[p]) >= 2]
nopract = [p for p in done if p not in BM]
task = [p for p in done if p in BM]
nolive = [p for p in task if sum(1 for r in SP[p] if not r.get('dropped')) < 2]
winstay = [p for p in task if p not in nolive and p not in keep]
final = [p for p in task if p not in nolive and p in keep]
N_WD, N_NP, N_EEG, N_WS, N_FIN = (len(withdrew), len(nopract), len(nolive),
                                  len(winstay), len(final))
print(f'[figM_flow] recruited {n_rec} - withdrew {N_WD} ({" ".join(withdrew)}) - '
      f'practice not passed {N_NP} ({" ".join(nopract)}) - '
      f'EEG quality {N_EEG} ({" ".join(nolive)}) - '
      f'win-stay {N_WS} ({" ".join(winstay)}) - analysed {N_FIN}')

# -- figure -------------------------------------------------------
fig = plt.figure(figsize=_N.figsize(W_PX, H_PX), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

Y = .7342           # the main row
YE = .2658          # the exclusion row
BW, BH = .1173, .214          # box
EW, EH = .1127, .214          # ellipse
XW, XH = .1137, .1765         # exclusion box

NODES = [                     # (x, kind, text)
    (.0774, 'ell', f'Recruited\nn = {n_rec}'),
    (.2389, 'box', 'Consent ·\nEEG headset'),
    (.3999, 'box', 'Resting state\n90 s × 2'),
    (.5608, 'box', 'Practice\n→ main task'),
    (.7219, 'box', '120 trials\n5 reversals'),
    (.9037, 'ell', f'Analysed\nn = {N_FIN}'),
]
DROPS = [                     # (x on the main row, text)
    (.3999, f'Withdrew during\nresting state   n = {N_WD}'),
    (.5608, f'Did not pass\npractice   n = {N_NP}'),
    (.7219, f'Low EEG quality\nn = {N_EEG}'),
    (.9037, f'win-stay < .60\nn = {N_WS}'),
]

for x, kind, txt in NODES:
    if kind == 'ell':
        ax.add_patch(Ellipse((x, Y), EW, EH, fc=FILL, ec=INK, lw=LW, zorder=3))
    else:
        ax.add_patch(Rectangle((x - BW / 2, Y - BH / 2), BW, BH,
                               fc='white', ec=INK, lw=LW, zorder=3))
    ax.text(x, Y, txt, ha='center', va='center', fontsize=FS, color=INK,
            fontweight=('bold' if kind == 'ell' else 'normal'),
            linespacing=LS, zorder=4)

for (x0, k0, _), (x1, k1, _) in zip(NODES[:-1], NODES[1:]):
    a = x0 + (EW if k0 == 'ell' else BW) / 2
    b = x1 - (EW if k1 == 'ell' else BW) / 2
    ax.add_patch(FancyArrowPatch((a, Y), (b, Y), arrowstyle='-|>',
                                 mutation_scale=9, lw=LW, color=INK,
                                 zorder=5, shrinkA=SHRINK, shrinkB=SHRINK))

for x, txt in DROPS:
    ax.add_patch(FancyArrowPatch((x, Y - BH / 2), (x, YE + XH / 2),
                                 arrowstyle='-|>', mutation_scale=9,
                                 lw=LW * .85, color=GREY, zorder=2,
                                 linestyle=(0, (4, 3)),
                                 shrinkA=SHRINK, shrinkB=SHRINK))
    ax.add_patch(Rectangle((x - XW / 2, YE - XH / 2), XW, XH,
                           fc='white', ec=GREY, lw=LW, zorder=3))
    ax.text(x, YE, txt, ha='center', va='center', fontsize=FS, color=INK,
            linespacing=LS, zorder=4)

fig.savefig(f'{OUT}/FigM_flow.png', dpi=DPI, facecolor='white')
fig.savefig(f'{OUT}/FigM_flow.svg', facecolor='white')
plt.close()

json.dump(dict(recruited=n_rec, withdrew=N_WD, no_practice=N_NP,
               eeg_quality=N_EEG, win_stay=N_WS, analysed=N_FIN,
               ids=dict(withdrew=withdrew, no_practice=nopract,
                        eeg_quality=nolive, win_stay=winstay, analysed=final)),
          open(f'{OUT}/FigM_flow.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('flow diagram done')
