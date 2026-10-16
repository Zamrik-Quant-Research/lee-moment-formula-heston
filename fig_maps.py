"""fig_maps — Lee's wing slopes beta_R and beta_L at T = 1 over correlation rho and vol-of-vol xi."""
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm

import lee_plot as P

R = np.load(Path(__file__).with_name("results.npz"))
rg, xg, BR, BL = R["rho_grid"], R["xi_grid"], R["map_R"], R["map_L"]
r0, x0 = R["eq_point"]
bR, bL = R["beta_RL"]

lo, hi = min(BR.min(), BL.min()), max(BR.max(), BL.max())
levels = np.geomspace(lo, hi, 25)
lines = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
norm = LogNorm(vmin=lo, vmax=hi)

fig, axes = P.panels(2)
fig.set_layout_engine("constrained")      # panels + colour bar fill the width, like the other figures
for ax, M, name in ((axes[0], BR, r"call wing $\beta_R$"), (axes[1], BL, r"put wing $\beta_L$")):
    cf = ax.contourf(rg, xg, M, levels=levels, cmap=P.CMAP, norm=norm)
    cs = ax.contour(rg, xg, M, levels=lines, colors=P.NAVY, linewidths=1.3)
    ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=10.5, colors=P.NAVY)
    ax.plot([r0], [x0], "o", ms=9, color=P.RED, mec="white", mew=1.4, zorder=5)
    ax.grid(False)
    P.title(ax, name + r" at $T = 1$")
    P.label(ax, x=r"correlation $\rho$", y=r"vol-of-vol $\xi$")
box = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2.0)
axes[0].annotate(f"{bR:.4f}", xy=(r0 + 0.07, x0 - 0.075), color=P.RED, fontsize=12, bbox=box)
axes[1].annotate(f"{bL:.4f}", xy=(r0 + 0.07, x0 - 0.075), color=P.RED, fontsize=12, bbox=box)
cb = fig.colorbar(cf, ax=list(axes), shrink=0.92, pad=0.02, ticks=lines)
cb.ax.set_yticklabels([f"{v:g}" for v in lines])
cb.ax.tick_params(labelsize=P.FS_TICK, colors=P.AXIS)
fig.savefig("fig_maps.png", dpi=P.DPI, facecolor="white")
