"""lee_plot — house plotting defaults (zz.plan.FIGURE): 12.4 x 6.4 in at 170 dpi, NAVY/TEAL on white."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

W_IN, H_IN, DPI = 12.4, 6.4, 170
NAVY, TEAL, GREY = "#1e3a5f", "#2a9d8f", "#8a94a6"
AXIS, RED = "#1a1a1a", "#ff0000"          # axes near-black, meeting points red
FILL = "#d8efec"                           # a pale teal for shaded bands
FS_TITLE, FS_LABEL, FS_LEGEND, FS_TICK = 13.5, 12.5, 11.5, 11
CMAP = LinearSegmentedColormap.from_list("house", ["#ffffff", TEAL, NAVY])


def _dress(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelsize=FS_TICK)
    ax.grid(True, color=GREY, alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    return ax


def figure():
    fig, ax = plt.subplots(figsize=(W_IN, H_IN), facecolor="white")
    return fig, _dress(ax)


def panels(n=2):
    fig, axes = plt.subplots(1, n, figsize=(W_IN, H_IN), facecolor="white")
    for ax in axes: _dress(ax)
    return fig, axes


def label(ax, x=None, y=None):
    if x: ax.set_xlabel(x, color=AXIS, fontsize=FS_LABEL)
    if y: ax.set_ylabel(y, color=AXIS, fontsize=FS_LABEL)


def title(ax, text):
    ax.set_title(text, color="black", fontsize=FS_LABEL)


def legend(ax, **kw):
    ax.legend(frameon=False, fontsize=FS_LEGEND, labelcolor="black", **kw)


def save(fig, name):
    fig.tight_layout(); fig.savefig(name, dpi=DPI, facecolor="white")
