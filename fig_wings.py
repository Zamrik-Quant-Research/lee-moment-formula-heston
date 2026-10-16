"""fig_wings — the Heston smile's implied total variance on both wings against Lee's asymptotes, and how slowly
V(x)/|x| reaches its slope."""
from pathlib import Path

import numpy as np

import lee_plot as P

R = np.load(Path(__file__).with_name("results.npz"))
xc, Vc, xp, Vp = R["x_calls"], R["V_calls"], R["x_puts"], R["V_puts"]
bR, bL = R["beta_RL"]
kc, kp = xc <= 8.0, xp >= -8.0
xc, Vc, xp, Vp = xc[kc], Vc[kc], xp[kp], Vp[kp]

fig, (ax1, ax2) = P.panels(2)

# left: V(x) on both wings with the lines beta_R x and beta_L |x|
ax1.plot(xp, Vp, color=P.NAVY, lw=2.6, marker="o", ms=4, label=r"Heston $V(x) = \sigma_{\mathrm{imp}}^2(x)\,T$")
ax1.plot(xc, Vc, color=P.NAVY, lw=2.6, marker="o", ms=4)
g = np.linspace(0, 8, 50)
ax1.plot(-g, bL * g, color=P.TEAL, lw=2.0, ls="--", label=r"Lee asymptotes $\beta_L |x|$ and $\beta_R x$")
ax1.plot(g, bR * g, color=P.TEAL, lw=2.0, ls="--")
ax1.set_xlim(-8.3, 8.3); ax1.set_ylim(0, 0.85)
P.title(ax1, r"implied total variance, $T = 1$")
P.label(ax1, x=r"log-moneyness $x = \log(K/F)$", y=r"implied total variance $V(x)$")
P.legend(ax1, loc="upper center")

# right: (V(x)/|x|) / beta on each wing
rc, rp = Vc / xc / bR, Vp / np.abs(xp) / bL
ax2.axhline(1.0, color=P.GREY, lw=1.4, ls="--")
ax2.plot(xc, rc, color=P.NAVY, lw=2.6, marker="o", ms=4, label=r"call wing, $V(x)/(\beta_R\, x)$")
ax2.plot(np.abs(xp), rp, color=P.TEAL, lw=2.6, marker="o", ms=4, label=r"put wing, $V(x)/(\beta_L |x|)$")
ax2.plot([8.0, 8.0], [rc[-1], rp[-1]], "o", ms=8, color=P.RED, mec="white", mew=1.2, zorder=5)
ax2.annotate(f"{rc[-1]:.3f}", xy=(7.35, rc[-1] - 0.17), color=P.NAVY, fontsize=12)
ax2.annotate(f"{rp[-1]:.3f}", xy=(7.35, rp[-1] + 0.08), color=P.TEAL, fontsize=12)
ax2.annotate("Lee's slope", xy=(0.4, 0.85), color=P.GREY, fontsize=12)
ax2.set_xlim(0, 8.3); ax2.set_ylim(0.7, 3.3)
P.title(ax2, "each wing against its own slope")
P.label(ax2, x=r"distance from the money $|x|$", y=r"$V(x)/|x|$ over $\beta$")
P.legend(ax2, loc="upper right")
P.save(fig, "fig_wings.png")
