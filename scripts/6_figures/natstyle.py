"""
natstyle.py - shared figure settings to the Nature specification

Specification
  - font       Helvetica family (Liberation Sans)
  - panel letter  lower-case bold, top left
  - line width    axes 0.5 pt, data 0.75 pt
  - black      #231f20 (rather than pure black)
  - ticks      outward, 2 pt long
  - grid       not used
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from paths import DATA, RAW, OUT, XLSX

import matplotlib
matplotlib.use("Agg")
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np

# -- figure specification ----------------------------------------
# Every figure is saved at a **final width of 173 mm = 6.8 in**, and at that size only the
# three steps below are used. Nothing smaller than 6 pt is used.
#
#   PT_PANEL 9 pt bold   panel letters (a, b, ...)
#   PT_AXIS  7 pt        axis titles and panel titles
#   PT_TICK  6 pt        ticks, legends, values above bars, annotations
#
# Figures used to be 9.7-10.3 in wide with text around 6.5 pt, so reducing them to 173 mm for
# the manuscript shrank the text to about 4.3 pt. The reduction ratio differed per figure, so
# the sizes also differed from one another. Fixing the width makes the pt written in a script
# the pt that is printed.
FIG_W_IN = 6.8                   # 173 mm
FIG_DPI = 600                    # 6.8 in x 600 = 4080 px
PT_PANEL = 9.0
PT_AXIS = 7.0
PT_TICK = 6.0


def pt_per_px(w_px, w_in=FIG_W_IN):
    """How many final pt one px of the figure's internal coordinates is.

    Each script writes its coordinates and line widths in units of 'manuscript pixels'.
    Fixing the width at 6.8 in and converting through this value keeps the line widths and
    spacings exactly as they were. Only the text size changes.
    """
    return 72.0 * w_in / float(w_px)


def figsize(w_px, h_px, w_in=FIG_W_IN):
    """Keep the internal coordinate ratio and set the width to 6.8 in."""
    return (w_in, w_in * float(h_px) / float(w_px))


FONT = "Liberation Sans"
INK  = "#231f20"          # Nature black
GREY = "#6d6e71"
LGREY= "#bcbec0"
FAINT= "#e6e7e8"

# Nature-family colours (low saturation, print-safe)
BLUE   = "#3b6ea5"      # behaviour - regression
RED    = "#c1272d"      # behaviour - perseveration
GREEN  = "#1b7837"      # EEG - aperiodic slope
PURPLE = "#762a83"      # EEG - alpha
ORANGE = "#e08214"
TEAL   = "#01847f"

def apply():
    rcParams.update({
        "font.family": FONT, "font.sans-serif": [FONT],
        "font.size": PT_AXIS,
        "mathtext.fontset": "custom", "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
        "mathtext.default": "regular",
        "text.color": INK,
        "axes.linewidth": .5, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "axes.labelsize": PT_AXIS,
        "axes.titlesize": PT_AXIS, "axes.titlepad": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
        "xtick.major.width": .5, "ytick.major.width": .5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": PT_TICK, "ytick.labelsize": PT_TICK,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "lines.linewidth": .75, "lines.markersize": 3,
        "legend.fontsize": PT_TICK, "legend.frameon": False,
        "legend.handlelength": 1.1, "legend.handletextpad": .5,
        "legend.borderpad": .2, "legend.labelspacing": .3,
        "figure.dpi": 200, "savefig.dpi": FIG_DPI,
        "savefig.bbox": "tight", "savefig.pad_inches": .02,
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })

def panel(ax, letter, x=-0.22, y=1.02, size=PT_PANEL):
    """Lower-case bold panel label"""
    fn = ax.text2D if hasattr(ax, 'text2D') else ax.text
    fn(x, y, letter, transform=ax.transAxes, fontsize=size,
       fontweight='bold', va='bottom', ha='left', color=INK)

def despine(ax, keep=('left','bottom')):
    for s in ('top','right','left','bottom'):
        ax.spines[s].set_visible(s in keep)

def cbar(fig, im, ax, ticks=None, label='', pos='right', frac=.035, pad=.02):
    cb = fig.colorbar(im, ax=ax, fraction=frac, pad=pad, ticks=ticks,
                      orientation='vertical' if pos=='right' else 'horizontal')
    cb.ax.tick_params(labelsize=PT_TICK, length=1.5, width=.4, pad=1, color=INK)
    cb.outline.set_linewidth(.4); cb.outline.set_edgecolor(INK)
    if label: cb.set_label(label, fontsize=PT_TICK, labelpad=1.5, color=INK)
    return cb
