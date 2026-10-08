# -*- coding: utf-8 -*-
"""Figure 4 scatter panels d-h - the version with the fit band clipped to the measurable range.

In panels d, e and f of the published figure the bootstrap band ran outside the definition of
the measure: lose-shift went negative (-23%), the perseverative error count went negative
(-11), and switch-bias ran outside +/-1 (-1.32 to +1.31). Every data point is inside the
definition, so clipping the band alone fits the axes to the data and makes the trend clearer.

Two versions are produced.
  ols    least-squares line + bootstrap band (the same estimator as the manuscript, band clipped)
  theil  Theil-Sen line + bootstrap band (the median of the pairwise slopes - being rank
         based it is of the same family as the Spearman rho printed beside it, and is less pulled by extremes)

The seed is fixed. The last digit of a bootstrap interval moves with the seed, so a value
reported in the manuscript has to carry both the value this script produced and the seed.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import cohort                                             # noqa: E402

DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'outputs')
os.makedirs(OUT, exist_ok=True)

SEED = 1
NB_CI = 10000          # rho interval
NB_BAND = 10000        # fit band - matched to the same count as the rho interval
MIN_DISTINCT = 4       # discard a resample with fewer distinct participants than this.
                       # The partial correlation (Figure 4j) is undefined below 4, so the
                       # whole of Figure 4 is unified on this rule.

# -- the row specification of published Figure 4 ------------------
# The earlier version was drawn 5,360 px wide, so fitting it to the manuscript width
# (3,995 px) shrank every element to 75%. The physical size is therefore fixed to the
# manuscript specification: final width 173 mm, and text uses only the three steps of
# natstyle. FIG_W_PX serves only as an internal coordinate ratio; line widths and spacings
# are converted with pt_per_px to keep the earlier appearance. Only the text size changes.
import natstyle as _N                      # noqa: E402

DPI = _N.FIG_DPI
FIG_W_PX, FIG_H_PX = 3995, 950          # internal coordinate ratio
PX = _N.pt_per_px(FIG_W_PX)              # internal pixels -> points
CAP = 0.72                               # digit height / em (Arial)

FS_TICK = _N.PT_TICK     # tick numbers
FS_AXIS = _N.PT_AXIS     # axis titles
FS_TITLE = _N.PT_AXIS    # panel titles
FS_RHO = _N.PT_TICK      # correlation annotation
FS_ID = _N.PT_TICK       # participant number
FS_LETTER = _N.PT_PANEL  # panel letter

BLUE, RED = '#1668D9', '#D42A2A'
BAND = {BLUE: '#E0EBFA', RED: '#F9E3E3'}
INK, GREY = '#1A1A1A', '#6E6E6E'
FILLC = '#1A1A1A'                        # fill for 003 and 004
MARK_LW = 5 * PX                         # marker line width, 5 px
MARK_OUTER = 34 * PX                     # outer diameter, 34 px
MARK_S = (MARK_OUTER - MARK_LW) ** 2     # scatter's s is diameter squared (pt^2)
LINE_LW = 8 * PX                         # fit line
SPINE_LW = 4 * PX
HILITE = ('003', '004')       # the two participants used as examples in the text - filled points
# Which participants are numbered, and whether leader lines are used.
# Numbering all twelve made the numbers more prominent than the data, and a distant number
# read as belonging to the neighbouring point even with a leader line. Only the two the text
# uses as examples are kept, with no leader lines. The other values are in Table 2 and Table S2.
LABEL_IDS = HILITE
LEADER = False

# Touching the global rcParams would make the other figure scripts in the same session
# inherit this list and look for fonts that are not there. It is used inside this figure only.
RC = {'font.family': 'sans-serif',
      'font.sans-serif': ['Arial', 'DejaVu Sans'],
      'pdf.fonttype': 42, 'ps.fonttype': 42}

# panel = (letter, title, x key, y key, y scale, y name, x name, colour, floor, ceiling)
PANELS = [
    ('d', 'Lose-shift',         'front_exp',   'lose_shift',        100,
     'Lose-shift (%)',          'Exponent',   BLUE,  0.0,  100.0),
    ('e', 'Switch bias',        'front_exp',   'sbi',                 1,
     'Switch-bias index',       'Exponent',   BLUE, -1.0,    1.0),
    ('f', 'Perseverative errors', 'front_exp', 'n_persev',            1,
     'Perseverative errors (n)', 'Exponent',  BLUE,  0.0,   None),
    ('g', 'Accuracy',           'front_alpha', 'accuracy',          100,
     'Accuracy (%)',            'Alpha peak',  RED,  0.0,  100.0),
    ('h', 'Trials to recover',  'front_alpha', 'trials_to_recover',   1,
     'Trials to recover (n)',   'Alpha peak',  RED,  0.0,   None),
]


def boot_rho(x, y, nb=NB_CI, seed=SEED):
    """Percentile interval of Spearman rho by participant resampling. Also returns the proportion touching +/-1."""
    rng = np.random.default_rng(seed)
    n = len(x)
    out = []
    for _ in range(nb):
        k = rng.integers(0, n, n)
        if len(set(x[k])) < MIN_DISTINCT:
            continue
        v = stats.spearmanr(x[k], y[k]).statistic
        if np.isfinite(v):
            out.append(v)
    out = np.asarray(out)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return lo, hi, float(np.mean(np.abs(out) >= 1.0))


def _slope(x, y, how):
    if how == 'theil':
        s, b = stats.theilslopes(y, x)[:2]
        return s, b
    return tuple(np.polyfit(x, y, 1))


def fit_band(x, y, how, lo_b, hi_b, nb=NB_BAND, seed=SEED):
    """The fit line and its bootstrap band, clipped to the measurable range [lo_b, hi_b]."""
    rng = np.random.default_rng(seed)
    n = len(x)
    gx = np.linspace(x.min(), x.max(), 200)
    s, b = _slope(x, y, how)
    line = s * gx + b
    draws = []
    for _ in range(nb):
        k = rng.integers(0, n, n)
        if len(set(x[k])) < MIN_DISTINCT:
            continue
        ss, bb = _slope(x[k], y[k], how)
        draws.append(ss * gx + bb)
    draws = np.asarray(draws)
    lo = np.percentile(draws, 2.5, axis=0)
    hi = np.percentile(draws, 97.5, axis=0)
    raw = (lo.min(), hi.max())                 # before clipping - for reporting
    if lo_b is not None:
        lo = np.maximum(lo, lo_b)
        hi = np.maximum(hi, lo_b)
        line = np.maximum(line, lo_b)
    if hi_b is not None:
        hi = np.minimum(hi, hi_b)
        lo = np.minimum(lo, hi_b)
        line = np.minimum(line, hi_b)
    return gx, line, lo, hi, raw


def label_points(ax, xs, ys, ids, col, fs=6.2, marker_pt=6.6,
                 only=None, leader=None):
    """Attach each participant number to its own point.

    only   participants to number (default LABEL_IDS). Pass None for all of them.
    leader whether to use leader lines (default LEADER).

    Even when fewer points are numbered, the placement search still avoids **every** point.

    A number far from its point reads as the neighbouring point's number. So the code
      1. measures the text box for real,
      2. sweeps eight directions from the nearest ring that clears the marker, choosing a
         spot that overlaps no placed number, no other point and no axis boundary,
      3. and attaches a short leader line if nothing fits on the nearest ring.
    The geometry is computed in screen pixels; the placement is given as a point-relative offset in pt.
    """
    fig = ax.figure
    fig.canvas.draw()
    px_per_pt = fig.dpi / 72.0
    mr = marker_pt / 2.0 * px_per_pt + 1.2          # marker radius plus clearance

    # The text box is measured once (it does not depend on position)
    probe = [ax.text(0, 0, p, fontsize=fs, ha='center', va='center')
             for p in ids]
    fig.canvas.draw()
    size = [(t.get_window_extent().width, t.get_window_extent().height)
            for t in probe]
    for t in probe:
        t.remove()

    pts = ax.transData.transform(np.column_stack([xs, ys]))
    x0, y0, x1, y1 = (ax.get_window_extent().x0, ax.get_window_extent().y0,
                      ax.get_window_extent().x1, ax.get_window_extent().y1)
    DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1),
            (.71, .71), (-.71, .71), (-.71, -.71), (.71, -.71)]

    only = LABEL_IDS if only is None else only
    leader = LEADER if leader is None else leader
    want = (range(len(ids)) if only is None
            else [i for i in range(len(ids)) if ids[i] in set(only)])
    order = sorted(want,
                   key=lambda i: -sum(1 for j in range(len(ids)) if j != i and
                                      np.hypot(*(pts[i] - pts[j])) < 46))
    boxes = []
    made = []
    for i in order:
        w, h = size[i]
        cx0, cy0 = pts[i]
        best = None
        for ring, grow in enumerate((1.0, 1.75, 2.6, 3.5, 4.6)):
            for ux, uy in DIRS:
                d = mr + (abs(ux) * w + abs(uy) * h) / 2.0 + 2.0
                tx, ty = cx0 + ux * d * grow, cy0 + uy * d * grow
                gap = max(2.5, h * .30)      # keep numbers from touching each other
                bb = (tx - w / 2 - gap, ty - h / 2 - gap,
                      tx + w / 2 + gap, ty + h / 2 + gap)
                # Better far with a leader line than overlapping - an overlap costs more than a ring
                cost = ring * 2.0
                cost += 25 * sum(1 for b in boxes if
                                 bb[0] < b[2] and b[0] < bb[2] and
                                 bb[1] < b[3] and b[1] < bb[3])
                cost += 25 * sum(1 for j, (qx, qy) in enumerate(pts)
                                 if bb[0] - mr < qx < bb[2] + mr and
                                 bb[1] - mr < qy < bb[3] + mr)
                if not (x0 < bb[0] and bb[2] < x1 and y0 < bb[1] and bb[3] < y1):
                    cost += 60
                if best is None or cost < best[0]:
                    best = (cost, tx, ty, bb, ring)
            if best[0] <= ring * 2.0 + 1e-9:
                break
        cost, tx, ty, bb, ring = best
        off = ((tx - cx0) / px_per_pt, (ty - cy0) / px_per_pt)
        # If a number is closer to someone else's point than to its own, a leader line is mandatory.
        # (A distant number reads as the neighbouring participant's number.)
        mine = np.hypot(tx - cx0, ty - cy0)
        other = min((np.hypot(tx - qx, ty - qy)
                     for j, (qx, qy) in enumerate(pts) if j != i),
                    default=np.inf)
        arrow = (dict(arrowstyle='-', lw=.45, color=col,
                      shrinkA=0, shrinkB=marker_pt / 2 + .8)
                 if (leader and (ring > 0 or other <= mine * 1.25)) else None)
        # A white stroke keeps the fit line and band from running through the text
        made.append(ax.annotate(ids[i], xy=(xs[i], ys[i]), xytext=off,
                                textcoords='offset points', fontsize=fs,
                                color=col, ha='center', va='center',
                                zorder=8, annotation_clip=False,
                                arrowprops=arrow,
                                path_effects=[pe.withStroke(
                                    linewidth=6 * PX,
                                    foreground='white')]))
        boxes.append(bb)
    return made


CORNERS = ((.03, .03, 'left', 'bottom'), (.97, .03, 'right', 'bottom'),
           (.03, .97, 'left', 'top'), (.97, .97, 'right', 'top'))


def rho_conflicts(ax, txt, xs, ys, labels, fs):
    """How many data points and numbers each corner collides with. Used to align the position across a row."""
    fig = ax.figure
    fig.canvas.draw()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    boxes = [t.get_window_extent() for t in labels]
    out = []
    for cx, cy, ha, va in CORNERS:
        t = ax.text(cx, cy, txt, transform=ax.transAxes, fontsize=fs,
                    ha=ha, va=va)
        fig.canvas.draw()
        bb = t.get_window_extent().expanded(1.06, 1.30)
        t.remove()
        n = sum(1 for px, py in pts if bb.contains(px, py))
        n += sum(1 for b in boxes if bb.overlaps(b))
        out.append(n)
    return out


def place_rho(ax, txt, fs, col, corner):
    """With corner None it goes below the title, outside the axes. All five panels use the same spot."""
    if corner is None:
        ax.text(.5, 1.015, txt, transform=ax.transAxes, fontsize=fs,
                color=col, ha='center', va='bottom', zorder=9)
        return
    cx, cy, ha, va = CORNERS[corner]
    ax.text(cx, cy, txt, transform=ax.transAxes, fontsize=fs, color=col,
            ha=ha, va=va, zorder=9,
            bbox=dict(boxstyle='square,pad=.25', fc='white', ec='none',
                      alpha=.75))


def clean(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GREY)
        ax.spines[s].set_linewidth(SPINE_LW)
    ax.tick_params(colors=GREY, labelsize=FS_TICK, width=SPINE_LW,
                   length=FS_TICK * .42, pad=FS_TICK * .40)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(INK)


def draw(how, eeg, report):
    with matplotlib.rc_context(RC):
        return _draw(how, eeg, report)


def _draw(how, eeg, report):
    fig, axes = plt.subplots(1, 5, figsize=(FIG_W_PX / DPI, FIG_H_PX / DPI),
                             dpi=DPI)
    ids = [d['id'] for d in eeg]
    pend = []
    titles = []
    for ax, (letter, title, xk, yk, sc, ylab, xlab, col,
             lo_b, hi_b) in zip(axes, PANELS):
        titles.append((ax, title))
        x = np.array([d[xk] for d in eeg], float)
        y = np.array([d[yk] * sc for d in eeg], float)

        gx, line, blo, bhi, raw = fit_band(x, y, how, lo_b, hi_b)
        ax.fill_between(gx, blo, bhi, color=BAND[col], lw=0, zorder=1)
        ax.plot(gx, line, color=col, lw=LINE_LW, zorder=2,
                solid_capstyle='round')

        fill = [c in HILITE for c in ids]
        ax.scatter(x[[not f for f in fill]], y[[not f for f in fill]],
                   s=MARK_S, facecolors='white', edgecolors=col,
                   lw=MARK_LW, zorder=6)
        ax.scatter(x[fill], y[fill], s=MARK_S, facecolors=FILLC,
                   edgecolors=col, lw=MARK_LW, zorder=6)

        pad_x = (x.max() - x.min()) * .16
        ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
        span = max(y.max(), bhi.max()) - min(y.min(), blo.min())
        ylo = min(y.min(), blo.min()) - span * .17
        yhi = max(y.max(), bhi.max()) + span * .17
        if lo_b is not None:
            ylo = max(ylo, lo_b - span * .06)
        if hi_b is not None:
            yhi = min(yhi, hi_b + span * .06)
        ax.set_ylim(ylo, yhi)

        rho = stats.spearmanr(x, y).statistic
        lo, hi, touch = boot_rho(x, y)
        at_bound = max(abs(lo), abs(hi)) >= .995
        mark = '†' if at_bound else ''

        ax.set_xlabel(xlab, fontsize=FS_AXIS, color=INK,
                      labelpad=FS_AXIS * .55)
        ax.set_ylabel(ylab, fontsize=FS_AXIS, color=INK,
                      labelpad=FS_AXIS * .55)
        clean(ax)
        labs = label_points(ax, x, y, ids, col, fs=FS_ID,
                            marker_pt=MARK_OUTER)
        rtxt = f'ρ = {rho:+.2f}  [{lo:+.2f}, {hi:+.2f}]{mark}'
        pend.append((ax, rtxt,
                     rho_conflicts(ax, rtxt, x, y, labs, FS_RHO)))
        ax.text(-.32, 1.16, letter, transform=ax.transAxes, fontsize=FS_LETTER,
                fontweight='bold', color=INK, ha='left', va='top')

        report.append(dict(panel=letter, fit=how, rho=round(float(rho), 4),
                           ci=[round(float(lo), 4), round(float(hi), 4)],
                           touch_one=round(touch, 4), at_bound=bool(at_bound),
                           band_raw=[round(float(raw[0]), 2),
                                     round(float(raw[1]), 2)],
                           band_clipped=[round(float(blo.min()), 2),
                                         round(float(bhi.max()), 2)],
                           bound=[lo_b, hi_b], n=len(x)))

    # The rho annotation uses the same spot in all five panels - if the position wobbles
    # across a row, a reviewer reads it as a different value written in each panel.
    tot = [sum(c[i] for _, _, c in pend) for i in range(len(CORNERS))]
    corner = int(np.argmin(tot)) if min(tot) == 0 else None
    pad = FS_TITLE * (2.1 if corner is None else .9)
    for ax, title in titles:
        ax.set_title(title, fontsize=FS_TITLE, color=INK, pad=pad)
    for ax, rtxt, _ in pend:
        place_rho(ax, rtxt, FS_RHO, INK, corner)

    fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, .97))
    dst = os.path.join(OUT, f'fig4_scatter_{how}.png')
    fig.savefig(dst, dpi=DPI, facecolor='white')
    plt.close(fig)
    return dst


def main():
    beh, eeg = cohort.load(DATA, verbose=True)
    report = []
    made = [draw(h, eeg, report) for h in ('ols', 'theil')]
    with open(os.path.join(OUT, 'fig4_scatter.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(seed=SEED, nb_ci=NB_CI, nb_band=NB_BAND,
                       ids=[d['id'] for d in eeg], panels=report),
                  f, ensure_ascii=False, indent=1)
    print(f'\nparticipants {len(eeg)} - seed {SEED}')
    print('%-3s %-6s %7s %-20s %-19s %-19s' %
          ('panel', 'fit', 'rho', '95% CI', 'band (before clip)', 'band (after clip)'))
    for r in report:
        print('%-3s %-6s %+7.3f  [%+.3f, %+.3f]%s  %-19s %-19s' %
              (r['panel'], r['fit'], r['rho'], r['ci'][0], r['ci'][1],
               ' †' if r['at_bound'] else '  ',
               '%.1f ~ %.1f' % tuple(r['band_raw']),
               '%.1f ~ %.1f' % tuple(r['band_clipped'])))
    print('\n† = an end of the interval lies on the boundary (±1.00)')
    for r in report:
        if r['at_bound']:
            print('   panel %s (%s) - %.1f%% of resamples give |rho| = 1' % (
                r['panel'], r['fit'], r['touch_one'] * 100))
    for m in made:
        print('\nwritten ->', m)


if __name__ == '__main__':
    main()
