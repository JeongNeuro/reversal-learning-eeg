# -*- coding: utf-8 -*-
"""fig5_full.py - Figure 5. From resting state to task

Reproduces the layout of Figure 5 as published. The original script was not kept, so it
was rebuilt from the repository. All five panels (a-e) are produced in one pass.

  a  the frontal spectrum with the aperiodic component removed (rest/task), plus five
     paired-t topographies of per-channel relative band power
  b  heatmap of the paired t of per-electrode log power (task - rest), the alpha t bars on
     the right, and the electrode-mean t per frequency underneath

Agreed for the manuscript - no significance marking is added (preregistration section 34).
  a  no asterisk on the band names, no star-shaped electrode markers, no band shading
  b  no outline around significant regions

The dimensions were measured in pixels from the published figure (4034 x 2252, the upper block 1238 px).

Input   data/flatspec.json  data/psd_all.json
Output  outputs/Fig5_ab.png  outputs/fig5_full.json
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import DATA, OUT                                 # noqa: E402

import io                                                   # noqa: E402
import json                                                 # noqa: E402
import warnings                                             # noqa: E402

import numpy as np                                          # noqa: E402
import matplotlib                                           # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                             # noqa: E402
from matplotlib.gridspec import GridSpec                    # noqa: E402
import matplotlib.transforms as mtransforms                 # noqa: E402
from scipy import stats                                     # noqa: E402
from scipy.interpolate import Rbf                           # noqa: E402

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import natstyle as N                                        # noqa: E402

from matplotlib.patches import Circle, Polygon, Ellipse       # noqa: E402

INK, GREY = N.INK, N.GREY

# Electrode layout and head outline - the 10-20 coordinates placed on a unit circle
CH = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
      'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
POS = {'AF3': (-.30, .66), 'AF4': (.30, .66), 'F7': (-.68, .40),
       'F3': (-.36, .42), 'F4': (.36, .42), 'F8': (.68, .40),
       'FC5': (-.55, .16), 'FC6': (.55, .16), 'T7': (-.80, .00),
       'T8': (.80, .00), 'P7': (-.60, -.48), 'P8': (.60, -.48),
       'O1': (-.27, -.76), 'O2': (.27, -.76)}


def headline(ax, lw=.6):
    ax.add_patch(Circle((0, 0), 1, fill=False, ec=INK, lw=lw, zorder=8))
    ax.add_patch(Polygon([[-.10, .99], [0, 1.13], [.10, .99]], closed=False,
                         fill=False, ec=INK, lw=lw, zorder=8))
    for sg in (-1, 1):
        ax.add_patch(Ellipse((sg * 1.02, 0), .10, .26, fc='none', ec=INK,
                             lw=lw, zorder=8))

# Final width fixed at 173 mm; text uses only the three steps defined in natstyle.
# W_PX and H_PX serve only as internal coordinate ratios; line widths and spacings are
# converted with pt_per_px to keep the earlier appearance. Only the text size changes.
DPI = N.FIG_DPI
W_PX, H_PX = 4034, 2252          # internal coordinate ratio (the whole of published Figure 5)
PX = N.pt_per_px(W_PX)
FS_TICK = N.PT_TICK
FS_AXIS = N.PT_AXIS
FS_LET = N.PT_PANEL
LW = 5 * PX
SPINE = 4 * PX

BLUE, RED = '#1668D9', '#D42A2A'
TLIM = 5.0                        # colour scale of the heatmap and topographies, +/-5
BND5 = [('Delta', 2, 4), ('Theta', 4, 8), ('Alpha', 8, 13),
        ('Beta', 13, 30), ('Gamma', 30, 45)]
EDGES = [4, 8, 13, 30]            # band edges - vertical guides
ROW = ['AF4', 'AF3', 'F4', 'F3', 'F8', 'F7', 'FC6', 'FC5',
       'T8', 'T7', 'P8', 'P7', 'O2', 'O1']      # top to bottom

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# Cluster-based permutation correction - produced by topo_cluster_perm.py.
# Only electrodes in a surviving cluster are highlighted with a white dot, and the corrected P is written beside the band name.
# The within-family FDR q - produced by pvalues.py. The asterisks follow that value.
QVAL = {}
_qp = os.path.join(str(OUT), 'pvalues.csv')
if os.path.exists(_qp):
    import csv as _csv2
    for _r in _csv2.DictReader(io.open(_qp, encoding='utf-8-sig')):
        if _r.get('q'):
            QVAL[(_r['group'], _r['label'])] = float(_r['q'])

CLUST_P = {}
_cp = os.path.join(str(OUT), 'topo_cluster_perm.csv')
if os.path.exists(_cp):
    import csv as _csv
    for _r in _csv.DictReader(io.open(_cp, encoding='utf-8-sig')):
        if _r.get('survives') == 'yes':
            CLUST_P.setdefault(_r['band'], []).append(
                (_r['cluster'].split(), float(_r['p_corrected'])))

FL = json.load(io.open(os.path.join(str(DATA), 'flatspec.json'), encoding='utf-8'))
PS = json.load(io.open(os.path.join(str(DATA), 'psd_all.json'), encoding='utf-8'))
report = {}


def perchan_rel(pid, cond, lo, hi):
    """Per-channel relative band power - against 1-45 Hz total power (the definition in section 2.4 of the manuscript)."""
    d = PS.get(pid)
    if not d or d.get(cond) is None:
        return None
    f = np.asarray(d['f'], float)
    A = np.asarray(d[cond], float)
    den = (f >= 1) & (f <= 45)
    bb = (f >= lo) & (f < hi)
    out = {}
    for i, c in enumerate(d['ch']):
        if i < A.shape[0] and not np.all(np.isnan(A[i])):
            out[c] = A[i][bb].sum() / A[i][den].sum()
    return out


SEED = 1
NB = 10000


def boot_mean(x, nb=NB, seed=SEED):
    """95%% interval of the mean by participant resampling - the repository-wide convention (seed 1, 10,000)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    o = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)]
    return np.percentile(o, [2.5, 97.5])


def paired_t(a, b):
    """Paired t (b - a). nan if the sample is below 2 or the variance is zero."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    if len(d) < 2 or not np.isfinite(d).all() or d.std(ddof=1) == 0:
        return np.nan
    return float(stats.ttest_rel(b, a).statistic)


def topo(ax, vals, levels, cmap='jet', mark=()):
    """Scalp topography - radial basis function interpolation plus the head outline."""
    xs = np.array([POS[c][0] for c in CH])
    ys = np.array([POS[c][1] for c in CH])
    zs = np.array([vals.get(c, np.nan) for c in CH], float)
    ok = ~np.isnan(zs)
    xs, ys, zs = xs[ok], ys[ok], zs[ok]
    th = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    rx, ry = 1.18 * np.cos(th), 1.18 * np.sin(th)
    # The border is filled with the nearest electrode value, not the overall mean. Filling
    # with the mean lets a one-sided map (alpha, for instance) stain the whole edge with
    # that value and the differences between electrodes stop being visible.
    near = [int(np.argmin((xs - a) ** 2 + (ys - b) ** 2))
            for a, b in zip(rx, ry)]
    rb = Rbf(np.r_[xs, rx], np.r_[ys, ry], np.r_[zs, zs[near]],
             function='multiquadric', smooth=.05)
    g = np.linspace(-1.05, 1.05, 170)
    X, Y = np.meshgrid(g, g)
    Z = rb(X, Y)
    Z[X ** 2 + Y ** 2 > 1] = np.nan
    im = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, zorder=1,
                     extend='both')
    ax.contour(X, Y, Z, levels=levels[::4], colors=[INK], linewidths=.25,
               alpha=.5, zorder=3)
    headline(ax, lw=.6)
    for c in CH:
        if c in mark:                     # electrode belongs to a surviving cluster
            ax.plot(*POS[c], marker='o', ms=3.0, mfc='white', mec=INK,
                    mew=.5, zorder=7)
        else:
            ax.plot(*POS[c], marker='o', ms=1.8, mfc=INK, mec=INK, mew=.2,
                    zorder=6)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.26)
    ax.set_aspect('equal')
    ax.axis('off')
    return im


def clean(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(SPINE)
    ax.tick_params(colors=INK, labelsize=FS_TICK, width=SPINE,
                   length=FS_TICK * .40, pad=FS_TICK * .35)


fig = plt.figure(figsize=N.figsize(W_PX, H_PX), dpi=DPI)
outer = GridSpec(2, 1, figure=fig, height_ratios=[1238, 1014], hspace=.30,
                 left=.052, right=.935, top=.965, bottom=.115)
gs = outer[0].subgridspec(1, 2, width_ratios=[1.02, 1.0], wspace=.30)
gs2 = outer[1].subgridspec(1, 3, width_ratios=[1.30, .80, .78], wspace=.42)

# ── a ────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0])
axT = ax          # the topographies sit on the upper part of the same axes

ids_a = sorted(FL)
f = np.asarray(FL[ids_a[0]]['f'], float)
R = np.array([FL[p]['rest']['flat'] for p in ids_a], float)
T = np.array([FL[p]['task']['flat'] for p in ids_a], float)
for i in range(len(ids_a)):
    ax.plot(f, R[i], color=BLUE, lw=LW * .40, alpha=.35, zorder=2)
    ax.plot(f, T[i], color=RED, lw=LW * .40, alpha=.35, zorder=2)
ax.plot(f, R.mean(0), color=BLUE, lw=LW * 1.5, zorder=5, label=f'Rest  (n = {len(ids_a)})')
ax.plot(f, T.mean(0), color=RED, lw=LW * 1.5, zorder=5, label=f'Task  (n = {len(ids_a)})')
ax.axhline(0, color=INK, lw=SPINE, zorder=3)
for e in EDGES:
    ax.axvline(e, color='#BBBBBB', lw=SPINE, ls=(0, (4, 4)), zorder=1)
ax.set_xlim(2, 40)
# Leave the top 45% free for the topographies
_lo = min(R.min(), T.min())
_hi = max(R.max(), T.max())
ax.set_ylim(_lo - (_hi - _lo) * .10, _hi + (_hi - _lo) * 1.03)
ax.set_yticks([0.0, 0.5, 1.0])   # no ticks where the topographies sit
ax.set_xticks([2, 4, 8, 13, 20, 30, 40])
ax.set_xlabel('Frequency (Hz)', fontsize=FS_AXIS, color=INK, labelpad=FS_AXIS * .45)
ax.set_ylabel('above fit (log)', fontsize=FS_AXIS, color=INK, labelpad=FS_AXIS * .45)
ax.legend(frameon=False, fontsize=FS_AXIS * .92, loc='upper right',
          bbox_to_anchor=(.99, .540), handlelength=1.4, labelspacing=.38,
          borderpad=0, handletextpad=.7)
clean(ax)
ax.text(-.115, 1.045, 'a', transform=ax.transAxes, fontsize=FS_LET,
        fontweight='bold', color=INK, ha='left', va='top')
report['panel_a'] = dict(n=len(ids_a), ids=ids_a)

# topographies - paired t of per-channel relative band power
lv = np.linspace(-TLIM, TLIM, 21)
topo_n, chan_n, topo_report = {}, {}, {}
for k, (nm, lo, hi) in enumerate(BND5):
    Rv, Tv = {}, {}
    for p in sorted(PS):
        a_ = perchan_rel(p, 'rest1', lo, hi)
        b_ = perchan_rel(p, 'task', lo, hi)
        if a_ and b_:
            Rv[p], Tv[p] = a_, b_
    ids = sorted(set(Rv) & set(Tv))
    vals = {}
    for c in CH:
        aa = [Rv[p][c] for p in ids if c in Rv[p] and c in Tv[p]]
        bb = [Tv[p][c] for p in ids if c in Rv[p] and c in Tv[p]]
        if len(aa) >= 3:
            vals[c] = paired_t(aa, bb)
            chan_n.setdefault(nm, {})[c] = len(aa)
    topo_n[nm] = len(ids)
    sub = axT.inset_axes([.068 + k * .176, .560, .180, .400])
    _cl = CLUST_P.get(nm, [])
    _mark = {c for chs, _ in _cl for c in chs}
    im = topo(sub, vals, lv, mark=_mark)
    _t = nm
    if _cl:
        _p = min(q for _, q in _cl)
        _t += '\n%s P = %s' % ('**' if _p < .01 else '*',
                               ('%.3f' % _p).lstrip('0'))
    sub.set_title(_t, fontsize=FS_AXIS, color=INK, pad=FS_AXIS * .22,
                  linespacing=1.25)
    topo_report[nm] = dict(clusters=[dict(channels=chs, p=q) for chs, q in _cl])
cax = axT.inset_axes([.950, .620, .014, .250])
cb = fig.colorbar(im, cax=cax, ticks=[-5, 0, 5])
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=FS_TICK, colors=INK, width=SPINE,
                  length=FS_TICK * .35, pad=FS_TICK * .25)
cb.set_label('t', fontsize=FS_AXIS, color=INK, rotation=0,
             labelpad=FS_AXIS * .55)
report['panel_a_topo_n'] = topo_n
report['panel_a_topo_n_per_channel'] = chan_n
report['panel_a_clusters'] = topo_report

# ── b ────────────────────────────────────────────────────────────
gsb = gs[1].subgridspec(2, 3, height_ratios=[1.0, .155],
                        width_ratios=[1.0, .150, .050],
                        hspace=.62, wspace=.05)
axH = fig.add_subplot(gsb[0, 0])
axB = fig.add_subplot(gsb[0, 1])
axC = fig.add_subplot(gsb[0, 2])
axM = fig.add_subplot(gsb[1, 0])
for a_ in (fig.add_subplot(gsb[1, 1]), fig.add_subplot(gsb[1, 2])):
    a_.axis('off')

ids_b = sorted(p for p in PS if PS[p].get('rest1') and PS[p].get('task'))
fb = np.asarray(PS[ids_b[0]]['f'], float)
m = (fb >= 2) & (fb <= 40)
fb = fb[m]
Tm = np.full((len(ROW), len(fb)), np.nan)
heat_n = {}
for i, c in enumerate(ROW):
    ra, ta = [], []
    for p in ids_b:
        d = PS[p]
        if c not in d['ch']:
            continue
        j = d['ch'].index(c)
        A = np.asarray(d['rest1'], float)[j][m]
        B = np.asarray(d['task'], float)[j][m]
        if np.all(np.isnan(A)) or np.all(np.isnan(B)):
            continue
        ra.append(np.log10(np.maximum(A, 1e-20)))
        ta.append(np.log10(np.maximum(B, 1e-20)))
    heat_n[c] = len(ra)
    if len(ra) >= 3:
        ra, ta = np.array(ra), np.array(ta)
        Tm[i] = [paired_t(ra[:, q], ta[:, q]) for q in range(len(fb))]

axH.imshow(Tm, aspect='auto', cmap='jet', vmin=-TLIM, vmax=TLIM,
           interpolation='bicubic',
           extent=[fb[0], fb[-1], len(ROW) - .5, -.5], zorder=2)
for e in EDGES:
    axH.axvline(e, color='white', lw=SPINE * 1.3, zorder=3)
axH.set_yticks(range(len(ROW)))
axH.set_yticklabels(ROW, fontsize=FS_TICK)
axH.set_xticks([2, 4, 8, 13, 20, 30, 40])
axH.set_xlabel('Frequency (Hz)', fontsize=FS_AXIS, color=INK,
               labelpad=FS_AXIS * .45)
axH.set_ylabel('Electrode', fontsize=FS_AXIS, color=INK, labelpad=FS_AXIS * .45)
clean(axH)
axH.text(-.175, 1.10, 'b', transform=axH.transAxes, fontsize=FS_LET,
         fontweight='bold', color=INK, ha='left', va='top')

am = (fb >= 8) & (fb <= 13)
at = np.nanmean(Tm[:, am], axis=1)
axB.barh(range(len(ROW)), at, height=.66, color='#3A3A3A', lw=0, zorder=3)
axB.set_ylim(len(ROW) - .5, -.5)
axB.set_xlim(min(-.2, np.nanmin(at) * 1.08), max(.2, np.nanmax(at) * 1.08))
axB.set_yticks([])
axB.set_xticks([])
for s in ('top', 'right', 'bottom'):
    axB.spines[s].set_visible(False)
axB.spines['left'].set_color(INK)
axB.spines['left'].set_linewidth(SPINE)
axB.text(.5, -.045, 'α t', transform=axB.transAxes, fontsize=FS_AXIS,
         color=GREY, ha='center', va='top')

sm = plt.cm.ScalarMappable(cmap='jet',
                           norm=plt.Normalize(vmin=-TLIM, vmax=TLIM))
cb2 = fig.colorbar(sm, cax=axC, ticks=[-5, 0, 5])
cb2.outline.set_visible(False)
cb2.ax.tick_params(labelsize=FS_TICK, colors=INK, width=SPINE,
                   length=FS_TICK * .35, pad=FS_TICK * .25)
cb2.set_label('t  (task − rest)', fontsize=FS_AXIS, color=INK,
              labelpad=FS_AXIS * .35)

mt = np.nanmean(Tm, axis=0)
axM.fill_between(fb, 0, np.minimum(mt, 0), color=BLUE, lw=0, zorder=2)
axM.fill_between(fb, 0, np.maximum(mt, 0), color=RED, lw=0, zorder=2)
axM.axhline(0, color=INK, lw=SPINE, zorder=3)
axM.set_xlim(fb[0], fb[-1])
axM.set_xticks([])
axM.set_yticks([])
axM.axis('off')
axM.text(-.018, .5, 'mean t', transform=axM.transAxes, fontsize=FS_AXIS,
         color=GREY, ha='right', va='center')

report['panel_b'] = dict(n=len(ids_b), ids=ids_b, n_per_channel=heat_n,
                         alpha_t={c: (None if not np.isfinite(v) else round(float(v), 3))
                                  for c, v in zip(ROW, at)},
                         t_range=[round(float(np.nanmin(Tm)), 3),
                                  round(float(np.nanmax(Tm)), 3)])

# ── c · d · e ────────────────────────────────────────────────────
import csv                                                     # noqa: E402

CONN = json.load(io.open(os.path.join(str(DATA), 'conn.json'), encoding='utf-8'))
ABANDS = ['8-10', '10-12']          # alpha
FRO = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8']
CEN = ['FC5', 'FC6', 'T7', 'T8']
POSTR = ['P7', 'P8', 'O1', 'O2']
IDX = {c: i for i, c in enumerate(CH)}
ORANGE = '#E08214'


def pairs(a, b=None):
    if b is None:
        return [(IDX[x], IDX[y]) for k, x in enumerate(a) for y in a[k + 1:]]
    return [(min(IDX[x], IDX[y]), max(IDX[x], IDX[y])) for x in a for y in b]


REG = [('Frontal', pairs(FRO)), ('Central', pairs(CEN)),
       ('Posterior', pairs(POSTR)),
       ('Front–Post', pairs(FRO, POSTR)), ('Global', pairs(CH))]


def align(pid, A):
    ch = CONN[pid]['ch']
    out = np.full((14, 14), np.nan)
    idx = {c: i for i, c in enumerate(ch)}
    for i, a in enumerate(CH):
        for j, b in enumerate(CH):
            if a in idx and b in idx:
                out[i, j] = A[idx[a]][idx[b]]
    return out


def plv_mat(pid, cond):
    ms = [align(pid, CONN[pid][cond][b]) for b in ABANDS
          if CONN[pid][cond].get(b)]
    return np.nanmean(ms, axis=0) if ms else None


MR = {p: plv_mat(p, 'rest') for p in sorted(CONN)}
MT = {p: plv_mat(p, 'task') for p in sorted(CONN)}
cids = [p for p in sorted(CONN) if MR[p] is not None and MT[p] is not None]

# -- c  alpha PLV by region (rest vs task) ------------------------
axc = fig.add_subplot(gs2[0])
crep = []
for k, (nm, pr) in enumerate(REG):
    r = np.array([np.nanmean([MR[p][i, j] for i, j in pr]) for p in cids])
    t = np.array([np.nanmean([MT[p][i, j] for i, j in pr]) for p in cids])
    d = t - r
    lo, hi = boot_mean(d)
    x0, x1 = k - .19, k + .19
    for a_, b_ in zip(r, t):
        axc.plot([x0, x1], [a_, b_], color='#BBBBBB', lw=SPINE * .8, zorder=2)
    for xx, vv, col in ((x0, r, BLUE), (x1, t, RED)):
        axc.bar(xx, vv.mean(), width=.36, color=col, lw=0, zorder=3)
        se = vv.std(ddof=1) / np.sqrt(len(vv))
        axc.errorbar(xx, vv.mean(), yerr=1.96 * se, fmt='none', ecolor=INK,
                     elinewidth=SPINE, capsize=FS_TICK * .30,
                     capthick=SPINE, zorder=5)
    # Alternate the height per region so the interval labels do not run together into one line.
    # Since the text grew to 6 pt, one step (0.115) put the lower line of the upper label and
    # the upper line of the lower label at the same height and they overlapped. Two line
    # heights (0.24) keeps the two blocks from intruding on each other.
    # The asterisks follow the within-family FDR q (* q < .05, ** q < .01).
    _q = QVAL.get(('Figure 5 c', nm.replace('–', '-')))
    _st = '' if _q is None else ('**' if _q < .01 else ('*' if _q < .05 else ''))
    axc.text(k, (.96, .72, .96, .72, .96)[k],
             '%+.3f%s\n[%+.3f, %+.3f]' % (d.mean(), _st, lo, hi),
             ha='center', va='bottom', fontsize=FS_TICK, color=INK,
             linespacing=1.30, zorder=7,
             bbox=dict(facecolor='white', edgecolor='none', pad=.8))
    crep.append(dict(region=nm.replace('–', '-'), n=len(cids), q=_q, star=_st,
                     rest=round(float(r.mean()), 4),
                     task=round(float(t.mean()), 4),
                     diff=round(float(d.mean()), 5),
                     # Five decimals are kept - rounding at the fourth decides whether the
                     # posterior floor (-0.08149) reads as -0.081 or as -0.082
                     ci=[round(float(lo), 5), round(float(hi), 5)],
                     decreased=int((d < 0).sum())))
axc.set_xticks(range(len(REG)))
axc.set_xticklabels([n for n, _ in REG], fontsize=FS_TICK)
axc.set_ylabel('Alpha phase synchrony (PLV)', fontsize=FS_AXIS, color=INK,
               labelpad=FS_AXIS * .45)
# The legend is moved outside (above) the bar area and the space above is opened up by the
# same amount, which removes the overlap between the legend and the value above Global.
axc.set_ylim(0, 1.25)
axc.set_yticks([0, .2, .4, .6, .8])
for k in range(1, len(REG)):
    axc.axvline(k - .5, color='#CCCCCC', lw=SPINE, ls=(0, (3, 3)), zorder=1)
_h = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
      plt.Rectangle((0, 0), 1, 1, color=RED)]
axc.legend(_h, ['rest', 'task'], loc='lower right',
           bbox_to_anchor=(1.0, 1.005), frameon=False,
           fontsize=FS_TICK, handlelength=1.0, handleheight=1.0,
           labelspacing=.35, borderpad=0, ncol=2, columnspacing=1.0)
clean(axc)
# The white background of the value text is allowed to cover the region divider, but the
# axis line is drawn on top of it; otherwise the left axis line looks broken.
for _sp in ('left', 'bottom'):
    if _sp in axc.spines:
        axc.spines[_sp].set_zorder(9)
axc.text(-.105, 1.10, 'c', transform=axc.transAxes, fontsize=FS_LET,
         fontweight='bold', color=INK, ha='left', va='top')
report['panel_c'] = crep

# -- d  change in alpha PLV per electrode pair - connectivity matrix --
# The circular connectogram is replaced by the lower triangle of a 14 x 14 matrix.
# Instead of only the pairs that decreased, all 91 pairs are shown. No pairwise test was
# run, so there is no threshold and no asterisk - this is a descriptive panel.
#
# The colormap is turbo. The range is symmetric about 0, but turbo is not a diverging map,
# so 0 does not stand out as a particular colour. The boundary between increase and
# decrease is read off the 0 tick of the colour bar.
axd = fig.add_subplot(gs2[1])
dR = np.nanmean([MR[p] for p in cids], axis=0)
dT = np.nanmean([MT[p] for p in cids], axis=0)
D = dT - dR

# The electrode order follows the region definitions of 5c exactly. The three regions cover
# all 14 electrodes, so there is nothing left to group separately. Front-Post and Global are
# groupings of between-region and all pairs and are not marked on the axis.
GRP = [('Frontal', FRO), ('Central', CEN), ('Posterior', POSTR)]
ORDER = [c for _, g in GRP for c in g]
OI = [IDX[c] for c in ORDER]
# The empty AF3 row and O2 column are dropped - being the lower triangle, they hold no cell
# at all. The vertical runs AF4-O2 and the horizontal AF3-O1, and the cell count stays 91.
n = len(ORDER)
ROWL, COLL = ORDER[1:], ORDER[:-1]
nn = n - 1
M = np.full((nn, nn), np.nan)
for r in range(nn):
    for c in range(nn):
        if r >= c:                      # in the reduced matrix the diagonal is used too
            a_, b_ = OI[r + 1], OI[c]
            M[r, c] = D[min(a_, b_), max(a_, b_)]
vmax = float(np.nanmax(np.abs(M)))
im_d = axd.imshow(np.ma.masked_invalid(M), cmap='turbo',
                  vmin=-vmax, vmax=vmax, interpolation='nearest',
                  origin='upper')
# The square matrix is pushed to the top of its slot, leaving room below for the region brackets and names.
axd.set_anchor('N')
axd.set_xticks(range(nn)); axd.set_yticks(range(nn))
axd.set_xticklabels(COLL, rotation=90, fontsize=FS_TICK)
axd.set_yticklabels(ROWL, fontsize=FS_TICK)
axd.tick_params(length=FS_TICK * .30, width=SPINE, pad=FS_TICK * .25,
                colors=INK)
for sp in axd.spines.values():
    sp.set_visible(False)
# No region boundary lines are drawn. On a turbo background a black line competes with the
# cell colours and made the panel harder to read. The regions are shown only by the brackets
# below the axis.
# Region names and brackets outside the axis - on the horizontal axis only. The matrix is
# symmetric, so writing them on both sides overlaps the tick labels and only adds clutter.
_bt = mtransforms.blended_transform_factory(axd.transData, axd.transAxes)
st = 0
for nm_, g in GRP:
    en = min(st + len(g), nn)        # shortened by the O2 that is dropped from the horizontal axis
    axd.plot([st - .42, en - .58], [-.235, -.235], color=INK,
             lw=SPINE * 1.1, transform=_bt, clip_on=False, zorder=6)
    axd.text((st + en - 1) / 2, -.275, nm_, ha='center', va='top',
             fontsize=FS_TICK, color=INK, transform=_bt, clip_on=False)
    st = en
cb = fig.colorbar(im_d, ax=axd, fraction=.034, pad=.02, shrink=.70,
                  ticks=[-.10, 0, .10])
cb.ax.set_yticklabels(['−.10', '0', '+.10'])   # as printed in the manuscript - leading zero omitted
cb.ax.tick_params(labelsize=FS_TICK, length=FS_TICK * .30, width=SPINE,
                  pad=FS_TICK * .25, color=INK)
cb.outline.set_linewidth(SPINE); cb.outline.set_edgecolor(INK)
# The title sits horizontally above the colour bar - a vertical label intruded on panel e.
cb.ax.set_title('ΔPLV' + chr(10) + '(task − rest)', fontsize=FS_TICK,
                color=INK, pad=FS_TICK * .55, linespacing=1.2)
axd.text(-.30, 1.10, 'd', transform=axd.transAxes, fontsize=FS_LET,
         fontweight='bold', color=INK, ha='left', va='top')

# The 91 pair values are written out separately, for checking against the caption
_grp_of = {c: nm_ for nm_, g in GRP for c in g}
_rows = []
for r in range(nn):
    for c in range(nn):
        if r >= c:
            a_, b_ = COLL[c], ROWL[r]
            i_, j_ = min(OI[c], OI[r + 1]), max(OI[c], OI[r + 1])
            _rows.append(dict(ch1=a_, ch2=b_, group1=_grp_of[a_],
                              group2=_grp_of[b_],
                              rest=round(float(dR[i_, j_]), 5),
                              task=round(float(dT[i_, j_]), 5),
                              delta=round(float(M[r, c]), 5)))
with io.open(os.path.join(str(OUT), 'fig5d_matrix.csv'), 'w',
             encoding='utf-8-sig', newline='') as _fh:
    _w = csv.DictWriter(_fh, fieldnames=list(_rows[0]))
    _w.writeheader(); _w.writerows(_rows)
_dec = sum(1 for q in _rows if q['delta'] < 0)
report['panel_d'] = dict(n=len(cids), total_pairs=len(_rows),
                         decreased_pairs=_dec,
                         increased_pairs=len(_rows) - _dec,
                         vmax=round(vmax, 5),
                         order=ORDER, rows=ROWL, cols=COLL,
                         groups={nm_: list(g) for nm_, g in GRP})
print('[fig5d] %d pairs - decreased %d - increased %d - max |dPLV| %.4f'
      % (len(_rows), _dec, len(_rows) - _dec, vmax))

# -- e  run-to-run intraclass correlation -------------------------
axe = fig.add_subplot(gs2[2])
ICC = {}
with io.open(os.path.join(str(OUT), 'feasibility.csv'),
             encoding='utf-8-sig') as fh:
    for row in csv.reader(fh):
        if len(row) > 4 and row[1].startswith('ICC'):
            v = row[2].replace('[', ' ').replace(']', ' ').replace(',', ' ')
            a, b, c2 = [float(x) for x in v.split()]
            ICC[row[1]] = (a, b, c2, int(row[4].split('=')[1]))
EROWS = [('Frontal', 'exp', 'ICC frontal exponent', BLUE),
         ('Frontal', 'α', 'ICC frontal alpha peak', RED),
         ('Occipital (O1·O2)', 'exp', 'ICC occipital O1/O2 exponent', BLUE),
         ('Occipital (O1·O2)', 'α', 'ICC occipital O1/O2 alpha peak', RED)]
for lo_, hi_, lab, col2 in ((.90, 1.00, 'excellent', '#EDEDED'),
                            (.75, .90, 'good', '#F6F6F6'),
                            (.50, .75, 'moderate', '#EDEDED')):
    axe.axhspan(lo_, hi_, color=col2, lw=0, zorder=1)
    axe.text(3.60, (lo_ + hi_) / 2, lab, fontsize=FS_TICK * .86,
             color=GREY, ha='left', va='center', zorder=4)
erep = []
for k, (grp_, kind, key, col) in enumerate(EROWS):
    if key not in ICC:
        continue
    v, lo_, hi_, n_ = ICC[key]
    axe.bar(k, v, width=.64, color=col, lw=0, zorder=3)
    axe.errorbar(k, v, yerr=[[v - lo_], [hi_ - v]], fmt='none', ecolor=INK,
                 elinewidth=SPINE, capsize=FS_TICK * .30, capthick=SPINE,
                 zorder=5)
    axe.text(k, .04, ('%.2f' % v).replace('0.', '.'), ha='center',
             va='bottom', fontsize=FS_TICK, color='white', zorder=6)
    erep.append(dict(cluster=grp_, metric=kind, icc=v, ci=[lo_, hi_], n=n_))
axe.set_xlim(-.64, 3.66)
axe.set_ylim(0, 1.20)
axe.set_yticks([0, .25, .50, .75, 1.00])
axe.set_xticks(range(4))
axe.set_xticklabels([k for _, k, _, _ in EROWS], fontsize=FS_TICK)
axe.set_ylabel('ICC', fontsize=FS_AXIS, color=INK, labelpad=FS_AXIS * .45)
# Grouped bars - the same approach as Figure 4j.
#   - flush against the axis line (upper corner at y = 0)
#   - broken between the two groups
#   - the label centred under each group
# The tick labels (exp, alpha) sit on top of this band, so the band reads as one block below the axis.
for x0, x1, lab in ((-.60, 1.46, 'Frontal'),
                    (1.54, 3.62, 'Occipital (O1·O2)')):
    axe.add_patch(plt.Rectangle((x0, -.215), x1 - x0, .215, color='#E8E8E8',
                                lw=0, zorder=1, clip_on=False))
    axe.text((x0 + x1) / 2, -.172, lab, ha='center', va='center',
             fontsize=FS_TICK, color=INK, zorder=4, clip_on=False)
_he = [plt.Rectangle((0, 0), 1, 1, color=BLUE),
       plt.Rectangle((0, 0), 1, 1, color=RED)]
axe.legend(_he, ['exponent', 'alpha peak'], loc='upper left',
           bbox_to_anchor=(.01, 1.04), frameon=False, fontsize=FS_TICK,
           handlelength=1.0, handleheight=1.0, labelspacing=.35, borderpad=0)
clean(axe)
axe.text(-.22, 1.10, 'e', transform=axe.transAxes, fontsize=FS_LET,
         fontweight='bold', color=INK, ha='left', va='top')
report['panel_e'] = erep

dst = os.path.join(str(OUT), 'Fig5_full.png')
fig.savefig(dst, dpi=DPI, facecolor='white')
plt.close(fig)
json.dump(report, io.open(os.path.join(str(OUT), 'fig5_full.json'), 'w',
                          encoding='utf-8'), ensure_ascii=False, indent=1)
print(f"[fig5] a curves n={report['panel_a']['n']} - topographies {topo_n} - "
      f"b n={report['panel_b']['n']}")
print('t range %.2f to %.2f' % tuple(report['panel_b']['t_range']))
print('written ->', dst)
