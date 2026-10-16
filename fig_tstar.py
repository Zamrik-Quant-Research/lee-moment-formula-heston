"""fig_tstar — the explosion time T*(u) on both sides of the strip, and the finite-moment strip against maturity."""
from pathlib import Path

import numpy as np

import lee_plot as P

R = np.load(Path(__file__).with_name("results.npz"))
uR, tR, uL, tL = R["uR"], R["tR"], R["uL"], R["tL"]
um_inf, up_inf = R["u_inf"]
um1, up1 = R["ucrit_T1"]
cm, cp = R["c_pm"]
Tg, upg, umg = R["T_grid"], R["up_grid"], R["um_grid"]

fig, (ax1, ax2) = P.panels(2)

# left: T*(u), infinite on the band [u-inf, u+inf]
ax1.axvspan(um_inf, up_inf, color=P.FILL, lw=0, zorder=0)
for u, t in ((uR, tR), (uL, tL)):
    ax1.plot(u, np.where(np.isfinite(t), t, np.nan), color=P.NAVY, lw=2.6)
ax1.plot(R["ode_u"], R["ode_t"], "s", ms=7, mfc="white", mec=P.NAVY, mew=1.6, ls="none", zorder=4,
         label="integrated Riccati equation")
ax1.plot([], [], color=P.NAVY, lw=2.6, label=r"closed form $T^*(u)$")
ax1.axhline(1.0, color=P.GREY, lw=1.2, ls="--")
ax1.plot([um1, up1], [1.0, 1.0], "o", ms=8, color=P.RED, mec="white", mew=1.2, zorder=5)
ax1.annotate(r"$u_-^*(1)$", xy=(um1 - 0.6, 1.25), ha="right", color=P.AXIS, fontsize=12)
ax1.annotate(r"$u_+^*(1)$", xy=(up1 + 0.6, 1.25), color=P.AXIS, fontsize=12)
ax1.annotate("no explosion\nat any maturity", xy=(0.5 * (um_inf + up_inf), 0.12), ha="center", color=P.TEAL, fontsize=12)
ax1.set_yscale("log"); ax1.set_xlim(-13, 40); ax1.set_ylim(0.05, 60)
P.title(ax1, r"explosion time of $\mathbb{E}[S_T^{\,u}]$")
P.label(ax1, x=r"moment order $u$", y=r"explosion time $T^*(u)$")
P.legend(ax1, loc="upper right")

# right: the strip of finite moments u-*(T) < u < u+*(T)
ax2.fill_between(Tg, umg, upg, color=P.FILL, lw=0, zorder=0)
ax2.plot(Tg, upg, color=P.NAVY, lw=2.6, label=r"critical moments $u_\pm^*(T)$")
ax2.plot(Tg, umg, color=P.NAVY, lw=2.6)
ks = Tg <= 3.0
ax2.plot(Tg[ks], cp / Tg[ks], color=P.TEAL, lw=1.8, ls=":", label=r"short maturity, $c_\pm / T$")
ax2.plot(Tg[ks], cm / Tg[ks], color=P.TEAL, lw=1.8, ls=":")
for v in (um_inf, up_inf):
    ax2.axhline(v, color=P.GREY, lw=1.4, ls="--")
ax2.plot([], [], color=P.GREY, lw=1.4, ls="--", label=r"long maturity, roots $u_\pm^\infty$")
ax2.plot([1.0, 1.0], [um1, up1], "o", ms=8, color=P.RED, mec="white", mew=1.2, zorder=5)
ax2.annotate("finite moments", xy=(25.0, 4.0), color=P.TEAL, fontsize=12)
ax2.set_xscale("log"); ax2.set_xlim(Tg[0], Tg[-1]); ax2.set_ylim(-12, 40)
P.title(ax2, "the finite-moment strip against maturity")
P.label(ax2, x=r"maturity $T$", y=r"moment order $u$")
P.legend(ax2, loc="upper right")
P.save(fig, "fig_tstar.png")
