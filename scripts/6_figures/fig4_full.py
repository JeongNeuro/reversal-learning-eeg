# -*- coding: utf-8 -*-
"""fig4_full.py - builds the whole of published Figure 4 (a-j) from the repository.

No code survives for Figure 4 as published (this machine was searched end to end).
Panels d-h are already produced to specification by fig4_scatter.py; this script produces
a-c, i and j to the same specification and combines everything into one sheet. The
appearance follows the published figure - this is not a new design.

Panels
  a  the frontal spectrum (a thin line per participant plus a thick mean) with the aperiodic fit
  b  the spectrum with the aperiodic component removed. The shading is the alpha peak search window (7-14 Hz)
  c  per-participant frontal spectra with their aperiodic fits (3 x 3)
  d-h  scatter plots of resting index x behavioural index (fig4_scatter.py)
  i  correlation of aperiodic exponent by cluster with the switch-bias index
  j  simple and partial correlations, and the difference drho between the two EEG indices

Note on panel i - the cluster exponent is the value obtained by **averaging the channel
spectra first and fitting once** (the same rule as spectro.py). Fitting per channel and
averaging the exponents gives a different value. spectro.json holds only front and po4, so
all four clusters are refitted from psd_all.json under the same rule. front and po4 matching
spectro.json confirms that the rule is the same.

Input   data/psd_all.json  data/spectro.json  data/behav_metrics.json
Output  outputs/fig4_full.png  outputs/fig4_full.json
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))
from paths import DATA, OUT                                 # noqa: E402

import importlib.util                                       # noqa: E402
import io                                                   # noqa: E402
import json                                                 # noqa: E402
import warnings                                             # noqa: E402

import numpy as np                                          # noqa: E402
import matplotlib                                           # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                             # noqa: E402
from matplotlib.gridspec import GridSpec                    # noqa: E402
from scipy import stats                                     # noqa: E402
from fooof import FOOOF                                     # noqa: E402

warnings.filterwarnings('ignore')

# The specification and helpers of fig4_scatter are used as they are
_spec = importlib.util.spec_from_file_location(
    'fig4_scatter', os.path.join(os.path.dirname(__file__), 'fig4_scatter.py'))
F4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F4)

DPI, PX = F4.DPI, F4.PX
BLUE, RED, INK, GREY = F4.BLUE, F4.RED, F4.INK, F4.GREY
FILLC, HILITE = F4.FILLC, F4.HILITE
FS_TICK, FS_AXIS, FS_TITLE = F4.FS_TICK, F4.FS_AXIS, F4.FS_TITLE
FS_ID, FS_LETTER, FS_RHO = F4.FS_ID, F4.FS_LETTER, F4.FS_RHO
MARK_S, MARK_LW, MARK_OUTER = F4.MARK_S, F4.MARK_LW, F4.MARK_OUTER
LINE_LW, SPINE_LW = F4.LINE_LW, F4.SPINE_LW
SEED, NB = F4.SEED, F4.NB_CI

FIG_W_PX, FIG_H_PX = 3995, 2865          # internal coordinate ratio (d-h 950 + 1,915 above)

FRONT = ['AF3', 'AF4', 'F3', 'F4']
PO4 = ['O1', 'O2', 'P7', 'P8']
TEMP = ['T7', 'T8', 'P7', 'P8']
GLOB = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
        'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
CLUSTERS = [('PO', PO4, True), ('Fr', FRONT, False),
            ('Gl', GLOB, False), ('TP', TEMP, False)]   # True = the preregistered primary
R2MIN, FRANGE = 0.85, (2.0, 40.0)
ALPHA_LO, ALPHA_HI = 7.0, 14.0           # the manuscript's alpha peak search window


def fit(f, p):
    """Same setting as fit_spec in spectro.py."""
    fm = FOOOF(peak_width_limits=[1, 8], max_n_peaks=6, min_peak_height=0.05,
               peak_threshold=2.0, aperiodic_mode='fixed', verbose=False)
    try:
        fm.fit(f, p, list(FRANGE))
    except Exception:
        return None
    if not np.isfinite(fm.r_squared_) or fm.r_squared_ < R2MIN:
        return None
    return fm


def alpha_peak(fm):
    pk = [q for q in fm.peak_params_ if ALPHA_LO <= q[0] < ALPHA_HI]
    if not pk:
        return None, None
    b = max(pk, key=lambda q: q[1])
    return float(b[1]), float(b[0])


def clean(ax, spines=('left', 'bottom')):
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(s in spines)
    for s in spines:
        ax.spines[s].set_color(GREY)
        ax.spines[s].set_linewidth(SPINE_LW)
    ax.tick_params(colors=GREY, labelsize=FS_TICK, width=SPINE_LW,
                   length=FS_TICK * .42, pad=FS_TICK * .40)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(INK)


def letter(ax, ch, dx=-.30, dy=1.16):
    ax.text(dx, dy, ch, transform=ax.transAxes, fontsize=FS_LETTER,
            fontweight='bold', color=INK, ha='left', va='top')


def boot_mean(x, nb=NB, seed=SEED):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    o = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(nb)]
    return np.percentile(o, [2.5, 97.5])


def boot_stat(fn, n, nb=NB, seed=SEED, guard=4):
    rng = np.random.default_rng(seed)
    o = []
    for _ in range(nb):
        ii = rng.integers(0, n, n)
        if len(set(ii.tolist())) < guard:
            continue
        try:
            v = fn(ii)
        except Exception:
            continue
        if np.isfinite(v):
            o.append(v)
    return np.percentile(o, [2.5, 97.5])


# -- data ---------------------------------------------------------
PS = json.load(io.open(os.path.join(str(DATA), 'psd_all.json'), encoding='utf-8'))
SP = json.load(io.open(os.path.join(str(DATA), 'spectro.json'), encoding='utf-8'))
import cohort                                              # noqa: E402
beh, eeg = cohort.load(str(DATA), verbose=True)
IDS = [d['id'] for d in eeg]
BEH = {d['id']: d for d in eeg}

# The within-family FDR q - produced by pvalues.py. The asterisks follow q
# (* q < .05, ** q < .01). Only the three preregistered tests use the uncorrected P.
QVAL, PVAL = {}, {}
_qp = os.path.join(str(OUT), 'pvalues.csv')
if os.path.exists(_qp):
    import csv as _csv
    for _r in _csv.DictReader(io.open(_qp, encoding='utf-8-sig')):
        if _r.get('q'):
            QVAL[(_r['group'], _r['label'])] = float(_r['q'])
        if _r.get('p'):
            PVAL[(_r['group'], _r['label'])] = float(_r['p'])


def qstar(group, label, primary=False):
    """The asterisk string. The preregistered primary hypotheses are judged on the uncorrected P."""
    v = PVAL.get((group, label)) if primary else QVAL.get((group, label))
    if v is None:
        return ''
    return '**' if v < .01 else ('*' if v < .05 else '')


report = {'ids': IDS}


def cluster_exp(pid, want):
    """Average the channel spectra, then fit once - averaged over runs."""
    v = PS.get(pid)
    if not v:
        return None
    f = np.asarray(v['f'], float)
    ch = v['ch']
    ix = [ch.index(c) for c in want if c in ch]
    if not ix:
        return None
    out = []
    for run in ('rest1', 'rest2'):
        P = np.asarray(v.get(run) or [], float)
        if P.size == 0:
            continue
        fm = fit(f, P[ix].mean(0))
        if fm is not None:
            out.append(float(fm.aperiodic_params_[-1]))
    return float(np.mean(out)) if out else None


def pipeline_alpha(pid):
    """Amplitude and centre frequency of the frontal alpha peak - the values in spectro.json.

    This keeps the numbers of panel c coming from one source. Runs without a peak are
    left out of the average (012 has no 7-14 Hz peak in run 2, so only run 1 is used).
    """
    amps, cfs = [], []
    for r in SP.get(pid, []):
        d = r.get('front') or {}
        if d.get('alpha_amp') is not None and d.get('alpha_cf') is not None:
            amps.append(float(d['alpha_amp']))
            cfs.append(float(d['alpha_cf']))
    if not amps:
        return None, None
    return float(np.mean(amps)), float(np.mean(cfs))


def frontal_curve(pid):
    """The frontal mean spectrum and its fit.

    Exponent and alpha are obtained by **fitting each run and then averaging**, because
    that is what spectro.py does and it is the only way the numbers of panel c match the
    front_exp and front_alpha of d-h. Averaging the two runs first and fitting once differs.
    """
    v = PS.get(pid)
    if not v:
        return None
    f = np.asarray(v['f'], float)
    ch = v['ch']
    ix = [ch.index(c) for c in FRONT if c in ch]
    if not ix:
        return None
    curves, ex, off, amps, cfs, flats, ffit = [], [], [], [], [], [], None
    for r in ('rest1', 'rest2'):
        P = np.asarray(v.get(r) or [], float)
        if P.size == 0:
            continue
        pw = P[ix].mean(0)
        fm = fit(f, pw)
        if fm is None:
            continue
        curves.append(pw)
        ex.append(float(fm.aperiodic_params_[-1]))
        off.append(float(fm.aperiodic_params_[0]))
        a, c = alpha_peak(fm)
        if a is not None:
            amps.append(a)
            cfs.append(c)
        flats.append(fm.power_spectrum - fm._ap_fit)
        ffit = fm.freqs
    if not curves:
        return None
    m = (f >= FRANGE[0]) & (f <= FRANGE[1])
    e, o = float(np.mean(ex)), float(np.mean(off))
    return dict(f=f[m], logp=np.log10(np.mean(curves, axis=0)[m]),
                ap=o - e * np.log10(ffit), ffit=ffit,
                flat=np.mean(flats, axis=0), exp=e,
                amp=(float(np.mean(amps)) if amps else None),
                cf=(float(np.mean(cfs)) if cfs else None),
                n_runs=len(curves))


CURVE = {p: frontal_curve(p) for p in IDS}
CURVE = {p: v for p, v in CURVE.items() if v}
print(f'[fig4_full] frontal curves n={len(CURVE)}')

# -- figure -------------------------------------------------------
with matplotlib.rc_context(F4.RC):
    fig = plt.figure(figsize=F4._N.figsize(FIG_W_PX, FIG_H_PX), dpi=DPI)
    # The upper block (a-c, i, j) and the lower scatter row are laid out separately. In one
    # grid the titles of d-h ride up into the tick labels of panel c.
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1915, 950],
                     hspace=.30, left=.055, right=.985, top=.945, bottom=.055)
    gs = outer[0].subgridspec(9, 24, hspace=1.9, wspace=2.6)
    gs_s = outer[1].subgridspec(1, 24, wspace=2.6)

    # -- a  frontal spectrum --------------------------------------
    ax = fig.add_subplot(gs[0:4, 0:5])
    curves = list(CURVE.values())
    for c in curves:
        ax.semilogx(c['f'], c['logp'], color=BLUE, lw=LINE_LW * .35, alpha=.35)
    mf = curves[0]['f']
    mean_logp = np.mean([c['logp'] for c in curves], axis=0)
    ax.semilogx(mf, mean_logp, color=BLUE, lw=LINE_LW * 1.25)
    mean_ap = np.mean([c['ap'] for c in curves], axis=0)
    ax.semilogx(curves[0]['ffit'], mean_ap, color='k', lw=LINE_LW * .8, ls=(0, (4, 3)))
    chi = float(np.mean([BEH[p]['front_exp'] for p in CURVE]))
    ax.text(.03, .05, f'χ = {chi:.2f}', transform=ax.transAxes,
            fontsize=FS_RHO, color=INK, ha='left', va='bottom')
    ax.set_xticks([2, 5, 10, 20, 40])
    ax.set_xticklabels(['2', '5', '10', '20', '40'])
    ax.set_xlabel('Frequency (Hz)', fontsize=FS_AXIS, color=INK,
                  labelpad=FS_AXIS * .5)
    ax.set_ylabel('log$_{10}$ power', fontsize=FS_AXIS, color=INK,
                  labelpad=FS_AXIS * .5)
    clean(ax)
    letter(ax, 'a', dx=-.26)
    report['chi'] = round(chi, 4)

    # -- b  aperiodic-removed spectrum ----------------------------
    ax = fig.add_subplot(gs[0:4, 6:11])
    ff = curves[0]['ffit']
    keep = ff <= 30.0
    for c in curves:
        ax.semilogx(ff[keep], c['flat'][keep], color=RED,
                    lw=LINE_LW * .35, alpha=.35)
    ax.semilogx(ff[keep], np.mean([c['flat'] for c in curves], axis=0)[keep],
                color=RED, lw=LINE_LW * 1.25)
    ax.axhline(0, color=INK, lw=SPINE_LW * .9)
    ax.axvspan(ALPHA_LO, ALPHA_HI, color=RED, alpha=.10, lw=0)
    ax.set_xticks([2, 5, 10, 20, 30])
    ax.set_xticklabels(['2', '5', '10', '20', '30'])
    ax.set_xlabel('Frequency (Hz)', fontsize=FS_AXIS, color=INK,
                  labelpad=FS_AXIS * .5)
    ax.set_ylabel('above fit (log)', fontsize=FS_AXIS, color=INK,
                  labelpad=FS_AXIS * .5)
    clean(ax)
    letter(ax, 'b', dx=-.26)

    # -- c  per-participant spectra, 3 x 3 ------------------------
    order = sorted(CURVE, key=lambda p: -BEH[p]['front_exp'])[:9]
    # The 3 x 3 grid is laid out separately. Since the text grew to 6-7 pt, the spacing of
    # the outer grid let the upper row's ticks collide with the lower row's titles.
    gs_c = gs[0:9, 13:24].subgridspec(3, 3, hspace=.95, wspace=.46)
    for k, pid in enumerate(order):
        r, cc = divmod(k, 3)
        ax = fig.add_subplot(gs_c[r, cc])
        c = CURVE[pid]
        ax.semilogx(c['f'], c['logp'], color=BLUE, lw=LINE_LW * .9)
        ax.semilogx(c['ffit'], c['ap'], color='k', lw=LINE_LW * .6,
                    ls=(0, (4, 3)))
        ax.axvspan(8, 13, color='0.5', alpha=.16, lw=0)
        ax.set_xticks([2, 10, 40])
        ax.set_xticklabels(['2', '10', '40'])
        ax.set_title(f'{pid}      exp {BEH[pid]["front_exp"]:.2f}',
                     fontsize=FS_TICK, color=INK, pad=FS_TICK * .7)
        amp, cf = pipeline_alpha(pid)
        if amp is not None and cf is not None:
            # Two lines - one line overruns the panel width at 6 pt.
            ax.text(.03, .04, f'α {amp:.2f}' + chr(10) + f'@ {cf:.1f} Hz',
                    transform=ax.transAxes, fontsize=FS_TICK,
                    color=INK, ha='left', va='bottom', linespacing=1.15,
                    zorder=6,
                    bbox=dict(facecolor='white', edgecolor='none', pad=.8))
        if cc == 0:
            ax.set_ylabel('log$_{10}$ power', fontsize=FS_AXIS * .9, color=INK,
                          labelpad=FS_AXIS * .45)
        clean(ax)
        if k == 0:
            # Open a gap between the block title and the first row of panel titles
            # (002 exp 1.90 ...). At 1.26 the two looked joined.
            letter(ax, 'c', dx=-.34, dy=1.52)
            ax.text(1.75, 1.52, 'Individual spectra with fitted aperiodic line',
                    transform=ax.transAxes, fontsize=FS_TITLE, color=INK,
                    ha='center', va='top')

    # -- i  correlation by cluster --------------------------------
    ax = fig.add_subplot(gs[5:9, 0:3])
    sbi = {p: BEH[p]['sbi'] for p in IDS}
    bars = []
    for nm, want, is_pre in CLUSTERS:
        xs, ys = [], []
        for p in IDS:
            v = cluster_exp(p, want)
            if v is not None:
                xs.append(v)
                ys.append(sbi[p])
        rho = stats.spearmanr(xs, ys).statistic if len(xs) >= 4 else np.nan
        bars.append((nm, rho, len(xs), is_pre))
    for k, (nm, rho, n, is_pre) in enumerate(bars):
        ax.bar(k, rho, width=.62, color=(INK if is_pre else '#9a9a9a'),
               lw=0, zorder=3)
        _lab = {'PO': 'PO parieto-occipital', 'Fr': 'Fr frontal', 'Gl': 'Gl global',
                'TP': 'TP temporo-parietal'}[nm]
        _st = qstar('Figure 4 i', _lab, primary=is_pre)
        ax.text(k, rho + .06, f'{rho:+.2f}'.replace('+0.', '.') + _st,
                ha='center', va='bottom', fontsize=FS_TICK,
                color=(INK if is_pre else GREY))
        # The participant count moves to a second line below the ticks. Since the text grew
        # to 6 pt, stacking two lines above a bar overlapped the neighbouring bar's value.

    ax.axhline(0, color=INK, lw=SPINE_LW)
    ax.axvline(.5, color=INK, lw=SPINE_LW, ls=(0, (4, 3)))
    ax.text(0, -.92, 'pre', ha='center', va='bottom', fontsize=FS_TICK,
            color=GREY)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] + chr(10) + 'n=%d' % b[2] for b in bars],
                       linespacing=1.25)
    ax.set_ylim(-1.0, 1.25)
    ax.set_yticks([-1, -.5, 0, .5, 1])
    ax.set_ylabel('ρ switch bias', fontsize=FS_AXIS, color=INK,
                  labelpad=FS_AXIS * .5)
    clean(ax)
    letter(ax, 'i', dx=-.52)
    report['panel_i'] = [dict(cluster=b[0], rho=round(float(b[1]), 4), n=b[2])
                         for b in bars]

    # -- j  simple and partial correlations -----------------------
    ax = fig.add_subplot(gs[5:9, 4:11])
    _e = np.array([BEH[p]['front_exp'] for p in IDS], float)
    _a = np.array([BEH[p]['front_alpha'] for p in IDS], float)
    _ls = np.array([BEH[p]['lose_shift'] for p in IDS], float)
    _ac = np.array([BEH[p]['accuracy'] for p in IDS], float)

    def sp(x, y):
        return stats.spearmanr(x, y).statistic

    def pt(x, y, c):
        rx, ry, rc = stats.rankdata(x), stats.rankdata(y), stats.rankdata(c)
        rxy = np.corrcoef(rx, ry)[0, 1]
        rxc = np.corrcoef(rx, rc)[0, 1]
        ryc = np.corrcoef(ry, rc)[0, 1]
        return (rxy - rxc * ryc) / np.sqrt((1 - rxc ** 2) * (1 - ryc ** 2))

    COLS = [('exp', 'lose-shift', BLUE, lambda i: sp(_e[i], _ls[i]),
             lambda i: pt(_e[i], _ls[i], _a[i])),
            ('α', 'lose-shift', RED, lambda i: sp(_a[i], _ls[i]),
             lambda i: pt(_a[i], _ls[i], _e[i])),
            ('exp', 'accuracy', BLUE, lambda i: sp(_e[i], _ac[i]),
             lambda i: pt(_e[i], _ac[i], _a[i])),
            ('α', 'accuracy', RED, lambda i: sp(_a[i], _ac[i]),
             lambda i: pt(_a[i], _ac[i], _e[i]))]
    ALL = np.arange(len(IDS))
    jrep = []
    for k, (lab, beh_lab, col, fs, fp) in enumerate(COLS):
        for m, (fn, alpha_) in enumerate(((fs, .35), (fp, 1.0))):
            v = float(fn(ALL))
            lo, hi = boot_stat(fn, len(IDS))
            x = k + (-.19 if m == 0 else .19)
            ax.bar(x, v, width=.34, color=col, alpha=alpha_, lw=0, zorder=3)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt='none',
                        ecolor=INK, elinewidth=SPINE_LW * 1.1,
                        capsize=FS_TICK * .34, capthick=SPINE_LW * 1.1,
                        zorder=5)
            _key = '%s × %s (%s)' % (lab, beh_lab,
                                      'simple' if m == 0 else 'partial')
            _st = qstar('Figure 4 j', _key)
            if _st:
                ax.text(x, max(v, hi) + .04, _st, ha='center', va='bottom',
                        fontsize=FS_TICK, color=INK, zorder=7)
            jrep.append(dict(pair=f'{lab} × {beh_lab}',
                             kind=('simple' if m == 0 else 'partial'),
                             rho=round(v, 4), ci=[round(lo, 4), round(hi, 4)],
                             q=QVAL.get(('Figure 4 j', _key)), star=_st))
    # drho = aperiodic exponent - alpha (the manuscript's sign). It is +0.47 for lose-shift.
    for a0, b0, xm in ((0, 1, .5), (2, 3, 2.5)):
        d = float(COLS[a0][3](ALL) - COLS[b0][3](ALL))
        lo, hi = boot_stat(
            lambda i, a=a0, b=b0: COLS[a][3](i) - COLS[b][3](i), len(IDS))
        # The bracket and its label stay inside the axes. They used to be drawn above the
        # axes and ran into the 'Frequency (Hz)' of panel b directly above.
        ax.plot([a0, a0, b0, b0], [1.06, 1.13, 1.13, 1.06], color=GREY,
                lw=SPINE_LW)
        ax.text(xm, 1.17, f'Δρ = {d:+.2f}' + chr(10) +
                f'[{lo:+.2f}, {hi:+.2f}]',
                ha='center', va='bottom', fontsize=FS_TICK, color=GREY)
        jrep.append(dict(pair=f'Δρ {COLS[a0][1]}', kind='delta',
                         rho=round(d, 4), ci=[round(lo, 4), round(hi, 4)]))
    ax.axhline(0, color=INK, lw=SPINE_LW)
    ax.axvline(1.5, color=GREY, lw=SPINE_LW, ls=(0, (3, 3)))
    ax.set_xticks(range(4))
    ax.set_xticklabels([c[0] for c in COLS])
    ax.set_ylim(-1.15, 1.62)
    ax.set_yticks([-1, 0, 1])
    ax.set_ylabel('Spearman ρ', fontsize=FS_AXIS, color=INK,
                  labelpad=FS_AXIS * .5)
    # The grey group bars sit flush against the axis line - their lower edge is the ylim floor.
    # They used to be at -1.13, which left them 0.02 clear of the axis.
    for x0, x1, lab in ((-.45, 1.45, 'lose-shift'), (1.55, 3.45, 'accuracy')):
        ax.add_patch(plt.Rectangle((x0, -1.15), x1 - x0, .19, color='0.90',
                                   lw=0, zorder=1, clip_on=False))
        ax.text((x0 + x1) / 2, -1.055, lab, ha='center', va='center',
                fontsize=FS_TICK, color=INK, zorder=4)
    h = [plt.Rectangle((0, 0), 1, 1, color='0.78'),
         plt.Rectangle((0, 0), 1, 1, color=INK)]
    ax.legend(h, ['simple ρ', 'partial ρ'], loc='lower left',
              bbox_to_anchor=(.02, .05), frameon=False, fontsize=FS_TICK,
              handlelength=1.0, handleheight=1.0, labelspacing=.45,
              borderpad=0)
    clean(ax)
    letter(ax, 'j', dx=-.20)
    report['panel_j'] = jrep

    # -- d-h  scatter plots ---------------------------------------
    srep = []
    for k, (lt, title, xk, yk, sc, ylab, xlab, col,
            lo_b, hi_b) in enumerate(F4.PANELS):
        ax = fig.add_subplot(gs_s[0, k * 5:(k * 5) + 4])
        x = np.array([BEH[p][xk] for p in IDS], float)
        y = np.array([BEH[p][yk] * sc for p in IDS], float)
        gx, line, blo, bhi, raw = F4.fit_band(x, y, 'ols', lo_b, hi_b)
        ax.fill_between(gx, blo, bhi, color=F4.BAND[col], lw=0, zorder=1)
        ax.plot(gx, line, color=col, lw=LINE_LW, zorder=2,
                solid_capstyle='round')
        fillm = [c in HILITE for c in IDS]
        ax.scatter(x[[not f for f in fillm]], y[[not f for f in fillm]],
                   s=MARK_S, facecolors='white', edgecolors=col,
                   lw=MARK_LW, zorder=6)
        ax.scatter(x[fillm], y[fillm], s=MARK_S, facecolors=FILLC,
                   edgecolors=col, lw=MARK_LW, zorder=6)
        pad = (x.max() - x.min()) * .16
        ax.set_xlim(x.min() - pad, x.max() + pad)
        span = max(y.max(), bhi.max()) - min(y.min(), blo.min())
        ylo = min(y.min(), blo.min()) - span * .17
        yhi = max(y.max(), bhi.max()) + span * .17
        if lo_b is not None:
            ylo = max(ylo, lo_b - span * .06)
        if hi_b is not None:
            yhi = min(yhi, hi_b + span * .06)
        ax.set_ylim(ylo, yhi)
        rho = stats.spearmanr(x, y).statistic
        lo, hi, touch = F4.boot_rho(x, y)
        mark = '†' if max(abs(lo), abs(hi)) >= .995 else ''
        ax.set_title(title, fontsize=FS_TITLE, color=INK, pad=FS_TITLE * 2.1)
        ax.set_xlabel(xlab, fontsize=FS_AXIS, color=INK, labelpad=FS_AXIS * .55)
        ax.set_ylabel(ylab, fontsize=FS_AXIS, color=INK, labelpad=FS_AXIS * .55)
        clean(ax)
        F4.label_points(ax, x, y, IDS, col, fs=FS_ID, marker_pt=MARK_OUTER)
        _st = qstar('Figure 4 d-h', {'d': 'd  exponent × lose-shift',
                                   'e': 'e  exponent × switch bias',
                                   'f': 'f  exponent × perseverative errors',
                                   'g': 'g  alpha × accuracy',
                                   'h': 'h  alpha × trials to recover'}[lt])
        ax.text(.5, 1.015,
                f'ρ = {rho:+.2f}{_st}  [{lo:+.2f}, {hi:+.2f}]{mark}',
                transform=ax.transAxes, fontsize=FS_RHO, color=INK,
                ha='center', va='bottom', zorder=9)
        letter(ax, lt, dx=-.32)
        srep.append(dict(panel=lt, star=_st, rho=round(float(rho), 4),
                         ci=[round(float(lo), 4), round(float(hi), 4)],
                         touch_one=round(touch, 4)))
    report['panels_dh'] = srep

    dst = os.path.join(str(OUT), 'fig4_full.png')
    fig.savefig(dst, dpi=DPI, facecolor='white')
    plt.close(fig)

json.dump(report, io.open(os.path.join(str(OUT), 'fig4_full.json'), 'w',
                          encoding='utf-8'), ensure_ascii=False, indent=1)
print('\npanel i')
for b in report['panel_i']:
    print('  %-4s %+7.4f  n = %d' % (b['cluster'], b['rho'], b['n']))
print('\nwritten ->', dst)
