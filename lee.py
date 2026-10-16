"""Every number and array in the note on Lee's moment formula in the Heston model, computed once.

Heston, zero rates, F = S_0 = 1, X = log S:  dX = -v/2 dt + sqrt(v) dW1,  dv = kappa (theta - v) dt + xi sqrt(v) dW2,
d<W1, W2> = rho dt.  E[S_T^u] = exp(A(T) + B(T) v0) with
    dB/dtau = xi^2/2 B^2 + (rho xi u - kappa) B + (u^2 - u)/2,  B(0) = 0,   dA/dtau = kappa theta B.
b = rho xi u - kappa, Delta = b^2 - xi^2 u (u - 1).

Writes results.npz (the arrays the three figures read), numbers.json (every number the prose states) and
CHECKS.txt (each check, its tolerance and PASS/FAIL), all beside this script.
Run: nice -n 19 python3 lee.py      (the mpmath prices use a pool of at most 32 workers)
"""
import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erfcx

warnings.simplefilter("error", RuntimeWarning)          # a RuntimeWarning is a bug here, never noise

HERE = Path(__file__).resolve().parent
T0 = time.time()
OUT = {"runtime_s": {}}
CHECKS = []
WORKERS = 32
MP_DPS = 34                                              # digits of the high-precision reference

EQ = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=-0.7)       # the equity-like set of the note
POS = dict(v0=0.04, kappa=1.5, theta=0.04, xi=0.5, rho=0.7)       # the positive-correlation example
LOGB = dict(v0=0.04, kappa=0.1, theta=0.04, xi=1.0, rho=0.9)      # kappa < rho xi: the logarithmic branch
T1 = 1.0


def check(name, value, target, tol, ok=None, note=""):
    """Record one check: |value - target| <= tol unless an explicit verdict is given."""
    if ok is None:
        ok = bool(abs(value - target) <= tol)
    CHECKS.append((name, float(value), float(target), float(tol), "PASS" if ok else "FAIL", note))
    return ok


def stamp(key):
    OUT["runtime_s"][key] = round(time.time() - T0, 1)
    print(f"  [{key}] {OUT['runtime_s'][key]:.0f}s", flush=True)


# ── 1. the explosion time ──────────────────────────────────────────────────────────────────────────────────
def b_delta(u, p):
    b = p["rho"] * p["xi"] * u - p["kappa"]
    return b, b * b - p["xi"] ** 2 * u * (u - 1.0)


def branch(u, p):
    b, D = b_delta(u, p)
    if D >= 0:
        return "infinite" if b <= 0 else "log"
    return "arctan"


def tstar(u, p):
    """T*(u), u > 1 or u < 0.  Delta >= 0, b <= 0: inf.  Delta >= 0, b > 0: (1/s) log((b+s)/(b-s)), s = sqrt(Delta),
    written 2 artanh(s/b)/s.  Delta < 0: (2/s)(pi/2 - arctan(b/s)), s = sqrt(-Delta), written 2 atan2(s, b)/s."""
    b, D = b_delta(u, p)
    if D >= 0:
        if b <= 0:
            return np.inf
        s = np.sqrt(D)
        return 2.0 * np.arctanh(s / b) / s if s > 0 else 2.0 / b
    s = np.sqrt(-D)
    return 2.0 * np.arctan2(s, b) / s


def tstar_textbook(u, p):
    """The same three branches exactly as printed in the note (log and pi/2 - arctan), for the identity check."""
    b, D = b_delta(u, p)
    if D >= 0:
        if b <= 0:
            return np.inf
        s = np.sqrt(D)
        return np.log((b + s) / (b - s)) / s
    s = np.sqrt(-D)
    return 2.0 / s * (np.pi / 2 - np.arctan(b / s))


def tstar_ode(u, p, tmax=2000.0):
    """Blow-up time of the Riccati equation, integrated directly: with B = tan(phi),
    dphi/dtau = a sin^2 phi + b sin phi cos phi + c cos^2 phi, phi(0) = 0, and B explodes when phi reaches pi/2.
    Returns (T*, phi at the end); T* = inf when phi settles below pi/2 by tmax."""
    a, c = 0.5 * p["xi"] ** 2, 0.5 * (u * u - u)
    b = p["rho"] * p["xi"] * u - p["kappa"]

    def rhs(t, y):
        s, co = np.sin(y[0]), np.cos(y[0])
        return [a * s * s + b * s * co + c * co * co]

    ev = lambda t, y: y[0] - np.pi / 2
    ev.terminal, ev.direction = True, 1
    r = solve_ivp(rhs, [0.0, tmax], [0.0], method="DOP853", events=ev, rtol=1e-13, atol=1e-15)
    t = float(r.t_events[0][0]) if r.t_events[0].size else np.inf
    return t, float(r.y[0, -1])


def u_inf(p):
    """The two roots of Delta(u) = xi^2 (rho^2 - 1) u^2 + (xi^2 - 2 rho xi kappa) u + kappa^2 = 0 (minus, plus)."""
    c2 = p["xi"] ** 2 * (p["rho"] ** 2 - 1.0)
    c1 = p["xi"] ** 2 - 2.0 * p["rho"] * p["xi"] * p["kappa"]
    c0 = p["kappa"] ** 2
    q = -0.5 * (c1 + np.copysign(np.sqrt(c1 * c1 - 4.0 * c2 * c0), c1))
    r = sorted([q / c2, c0 / q])
    return r[0], r[1]


def ucrit(T, p, side):
    """u+*(T) (side +1) or u-*(T) (side -1): the root of T*(u) = T beyond the band where T* is infinite.
    Needs kappa > rho xi, so that T* is infinite exactly on [u-inf, 0) and (1, u+inf]."""
    um_, up_ = u_inf(p)
    f = lambda u: tstar(u, p) - T
    if side > 0:
        lo = up_ + 1e-9 * max(1.0, abs(up_))
        hi = 2.0 * lo
        while tstar(hi, p) > T:
            hi *= 2.0
    else:
        hi = um_ - 1e-9 * max(1.0, abs(um_))
        lo = 2.0 * hi
        while tstar(lo, p) > T:
            lo *= 2.0
    assert np.isfinite(f(lo)) and np.isfinite(f(hi)) and f(lo) * f(hi) < 0
    return float(brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=500))


def ucrit_ode(T, p, side, u_guess):
    """The same critical moment found on the integrated Riccati equation (no closed form inside)."""
    um_, up_ = u_inf(p)
    edge = up_ if side > 0 else um_
    gap = u_guess - edge
    lo, hi = edge + 0.5 * gap, edge + 2.0 * gap
    tmax = 4.0 * T + 10.0

    def g(u):
        t, _ = tstar_ode(u, p, tmax=tmax)
        return (t if np.isfinite(t) else tmax) - T

    return float(brentq(g, min(lo, hi), max(lo, hi), xtol=1e-13, rtol=1e-14, maxiter=200))


def beta(r):
    """Lee's map r -> 2 - 4 (sqrt(r^2 + r) - r), evaluated as 2 (sqrt(r + 1) - sqrt(r))^2 = 2 / (sqrt(r+1) + sqrt(r))^2."""
    return 2.0 / (np.sqrt(r + 1.0) + np.sqrt(r)) ** 2


def beta_lee(r):
    return 2.0 - 4.0 * (np.sqrt(r * r + r) - r)


def beta_inv(bt):
    """psi(beta) = (2 - beta)^2 / (8 beta) = 1/(2 beta) + beta/8 - 1/2, the inverse of Lee's map on (0, 2]."""
    return (2.0 - bt) ** 2 / (8.0 * bt)


def slopes(T, p):
    up, um = ucrit(T, p, +1), ucrit(T, p, -1)
    return up, um, beta(up - 1.0), beta(-um)


# ── 2. the moment generating function, little-trap form, and the complex Riccati equation ──────────────────
def logmgf(z, T, p, lib=np):
    """log E[S_T^z] for complex z in the moment strip: C + D v0 in the little-trap form (principal branches)."""
    v0, k, th, xi, rho = p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"]
    b = rho * xi * z - k
    d = lib.sqrt(b * b - xi * xi * (z * z - z))
    g = (-b - d) / (-b + d)
    e = lib.exp(-d * T)
    C = k * th / xi ** 2 * ((-b - d) * T - 2 * lib.log((1 - g * e) / (1 - g)))
    D = (-b - d) / xi ** 2 * (1 - e) / (1 - g * e)
    return C + D * v0


def logmgf_ode(z, T, p):
    """log E[S_T^z] by integrating dB = xi^2/2 B^2 + (rho xi z - kappa) B + (z^2 - z)/2, dA = kappa theta B."""
    v0, k, th, xi, rho = p["v0"], p["kappa"], p["theta"], p["xi"], p["rho"]
    z = complex(z)

    def rhs(t, y):
        B = y[0]
        return [0.5 * xi * xi * B * B + (rho * xi * z - k) * B + 0.5 * (z * z - z), k * th * B]

    r = solve_ivp(rhs, [0.0, T], [0j, 0j], method="DOP853", rtol=1e-13, atol=1e-15)
    return complex(r.y[1, -1] + r.y[0, -1] * v0)


def lm_real(a, T, p):
    return float(logmgf(complex(a + 1.0), T, p).real)


def G(a, x, T, p):
    """log of the damped integrand at v = 0: log E[S^(a+1)] - a x - log(a (a + 1)) (Lord-Kahl's choice of a)."""
    return lm_real(a, T, p) - a * x - np.log(a * (a + 1.0))


def alpha_opt(x, T, p, up, um):
    """The damping minimising G inside (0, u+* - 1) for a call (x >= 0) or (u-* - 1, -1) for a put (x < 0)."""
    if x >= 0:
        lo, hi = 1e-7, up - 1.0 - 1e-7 * up
    else:
        lo, hi = um - 1.0 + 1e-7 * abs(um), -1.0 - 1e-7
    r = minimize_scalar(lambda a: G(a, x, T, p), bounds=(lo, hi), method="bounded",
                        options=dict(xatol=1e-9, maxiter=500))
    return float(r.x)


GX, GW = np.polynomial.legendre.leggauss(32)


def _panels(L, h):
    n = int(np.ceil(L / h))
    left = np.arange(n) * h
    v = (left[:, None] + (GX[None, :] + 1.0) * h / 2.0).ravel()
    w = np.tile(GW * h / 2.0, n)
    return v, w, n


def log_price(x, a, T, p, refine=1, tol=1e-22):
    """log of the OTM option at log-strike x (call for a > 0, put for a < -1) by the damped Fourier integral
        price = e^{-a x}/pi int_0^inf Re[ e^{-i v x} E[S^{a+1+iv}] / ((a + iv)(a + 1 + iv)) ] dv,
    composite 32-point Gauss-Legendre, the integrand scaled to 1 at v = 0. Returns (log price, info)."""
    lm0 = lm_real(a, T, p)
    eps = 1e-4
    s2 = (lm_real(a + eps, T, p) - 2 * lm0 + lm_real(a - eps, T, p)) / eps ** 2       # the tilted variance of X
    h = min(0.25, 0.25 / np.sqrt(s2)) / refine
    L = 1.0
    while True:
        env = abs(np.exp(logmgf(a + 1 + 1j * L, T, p) - lm0)) * abs(a * (a + 1)) / L ** 2
        if env < tol and L > 6.0 / np.sqrt(s2):
            break
        L *= 1.25
    v, w, n = _panels(L, h)
    z = a + 1 + 1j * v
    f = (np.exp(-1j * v * x + logmgf(z, T, p) - lm0) * (a * (a + 1)) / ((a + 1j * v) * (a + 1 + 1j * v))).real
    I = float(np.sum(w * f))
    Iabs = float(np.sum(w * np.abs(f)))
    lp = G(a, x, T, p) + np.log(I) - np.log(np.pi)
    return lp, dict(alpha=a, L=float(L), panels=n, h=float(h), I_scaled=I, cancel=Iabs / I)


def _mp_logmgf(z, T, p):
    v0, k, th, xi, rho = (mp.mpf(repr(float(p[q]))) for q in ("v0", "kappa", "theta", "xi", "rho"))
    return logmgf(z, T, dict(v0=v0, kappa=k, theta=th, xi=xi, rho=rho), lib=mp)


def log_price_mp(args):
    """The same integral at MP_DPS digits with mpmath's tanh-sinh rule on panels of width w; strings out."""
    x, a, T, p, dps, w = args
    mp.mp.dps = dps
    a, x, T = mp.mpf(repr(float(a))), mp.mpf(repr(float(x))), mp.mpf(repr(float(T)))
    lm0 = mp.re(_mp_logmgf(a + 1, T, p))
    tol = mp.mpf(10) ** (-(dps + 4))
    L = mp.mpf(8)
    while abs(mp.exp(_mp_logmgf(a + 1 + 1j * L, T, p) - lm0)) * abs(a * (a + 1)) / L ** 2 > tol:
        L *= mp.mpf("1.25")

    def f(v):
        z = a + 1 + 1j * v
        return mp.re(mp.exp(-1j * v * x + _mp_logmgf(z, T, p) - lm0) * (a * (a + 1)) / ((a + 1j * v) * (a + 1 + 1j * v)))

    n = int(mp.ceil(L / w))
    I = mp.quad(f, [mp.mpf(i) * w for i in range(n + 1)])
    lp = lm0 - a * x - mp.log(a * (a + 1)) + mp.log(I) - mp.log(mp.pi)
    return mp.nstr(lp, dps + 2)


# ── 3. implied total variance from a log price: Mills ratios, no cancellation of the exponential scale ────
SQ_HALF_PI = np.sqrt(np.pi / 2.0)


def mills(z):
    """R(z) = (1 - N(z)) / phi(z) = sqrt(pi/2) erfcx(z / sqrt 2)."""
    return SQ_HALF_PI * erfcx(z / np.sqrt(2.0))


def log_black_otm(x, s):
    """log of the normalized OTM Black price at log-strike x, total volatility s (F = 1):
    call(k) = phi(a) [R(a) - R(b)], a = k/s - s/2, b = k/s + s/2, k = |x|;  put(x) = e^x call(|x|)."""
    k = abs(x)
    a, b = k / s - s / 2.0, k / s + s / 2.0
    return -0.5 * a * a - 0.5 * np.log(2 * np.pi) + np.log(mills(a) - mills(b)) + (x if x < 0 else 0.0)


def implied_V(x, lp):
    s = brentq(lambda s: log_black_otm(x, s) - lp, 1e-4, 6.0, xtol=1e-16, rtol=1e-15, maxiter=300)
    return s * s


def implied_V_mp(x, lp_str, s0, dps=50):
    """The same inversion at 50 digits on N(.) directly (no Mills ratio), from the mpmath price."""
    mp.mp.dps = dps
    x, lp = mp.mpf(repr(float(x))), mp.mpf(lp_str)
    k = abs(x)

    def lb(s):
        d1, d2 = -k / s + s / 2, -k / s - s / 2
        c = mp.ncdf(d1) - mp.exp(k) * mp.ncdf(d2)
        return mp.log(c) + (x if x < 0 else 0)

    s = mp.findroot(lambda s: lb(s) - lp, mp.mpf(repr(float(s0))))
    return float(s * s)


# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════
def main():
    OUT["params"] = dict(equity=EQ, positive=POS, log_branch=LOGB, T=T1, mp_dps=MP_DPS, workers=WORKERS)

    # ── A. the explosion time, closed form against the integrated Riccati equation ─────────────────────────
    print("explosion times ...", flush=True)
    rows = []
    grid = {"equity": (EQ, [1.5, 2, 5, 11, 12, 15, 21.1453729210, 30, 50, 200,
                            -0.5, -1, -1.5, -1.6, -2, -3, -4, -5, -5.3422048572, -8, -12, -50]),
            "log_branch": (LOGB, [1.5, 2.5, 4, 5, 10, -0.5, -2, -10])}
    for name, (p, us) in grid.items():
        for u in us:
            tc, tt = tstar(u, p), tstar_textbook(u, p)
            to, phi_end = tstar_ode(u, p)
            br = branch(u, p)
            rec = dict(set=name, u=u, branch=br, b=b_delta(u, p)[0], delta=b_delta(u, p)[1], closed=tc, ode=to)
            if np.isfinite(tc):
                rec["rel"] = abs(tc - to) / tc
                check(f"T*({u:g}) [{name}, {br} branch]: closed form vs integrated Riccati (rel)", rec["rel"], 0, 1e-10)
                check(f"T*({u:g}) [{name}]: atan2/artanh form vs the printed log/arctan form (rel)",
                      abs(tc - tt) / tc, 0, 1e-12)
            else:
                b, D = b_delta(u, p)
                Beq = (-b - np.sqrt(D)) / p["xi"] ** 2                 # smaller root of a B^2 + b B + c, a = xi^2/2
                rec["phi_end"], rec["phi_eq"] = phi_end, float(np.arctan(Beq))
                check(f"T*({u:g}) [{name}] infinite: the integrated solution settles at the smaller root, never explodes",
                      abs(phi_end - np.arctan(Beq)), 0, 1e-8, ok=bool(not np.isfinite(to) and abs(phi_end - np.arctan(Beq)) < 1e-8))
            rows.append(rec)
    OUT["tstar_rows"] = rows
    for name in grid:
        r = [x["rel"] for x in rows if x["set"] == name and "rel" in x]
        OUT[f"tstar_max_rel_{name}"] = max(r)
    OUT["tstar_max_rel"] = max(OUT["tstar_max_rel_equity"], OUT["tstar_max_rel_log_branch"])
    OUT["tstar_n_finite"] = sum("rel" in x for x in rows)
    OUT["tstar_n_infinite"] = sum("rel" not in x for x in rows)
    OUT["tstar_branches_seen"] = sorted({x["branch"] for x in rows})
    check("all three branches exercised (arctan, log, infinite)", len(OUT["tstar_branches_seen"]), 3, 0)
    OUT["tstar_m2"], OUT["tstar_m4"] = tstar(-2.0, EQ), tstar(-4.0, EQ)
    check("T*(-2) = 4.922772 (prototype, 6 decimals)", OUT["tstar_m2"], 4.922772, 5e-7)
    check("T*(-4) = 1.446423 (prototype, 6 decimals)", OUT["tstar_m4"], 1.446423, 5e-7)

    # the logarithmic branch needs kappa < rho xi: scan both sets over u in (1, 60] and [-60, 0)
    us = np.concatenate([np.linspace(1.0005, 60, 4000), -np.linspace(0.0005, 60, 4000)])
    for name, p in (("equity", EQ), ("positive", POS), ("log_branch", LOGB)):
        n_log = sum(branch(u, p) == "log" for u in us)
        OUT[f"n_log_branch_{name}"] = int(n_log)
    check("no logarithmic branch on a u scan when kappa > rho xi (equity and positive sets)",
          OUT["n_log_branch_equity"] + OUT["n_log_branch_positive"], 0, 0)
    check("logarithmic branch present when kappa < rho xi (log-branch set)", OUT["n_log_branch_log_branch"], 1, 0,
          ok=OUT["n_log_branch_log_branch"] > 0)
    stamp("tstar")

    # ── B. the band of infinite explosion time and the critical moments at T = 1 ───────────────────────────
    print("critical moments ...", flush=True)
    um_inf, up_inf = u_inf(EQ)
    c2 = EQ["xi"] ** 2 * (EQ["rho"] ** 2 - 1); c1 = EQ["xi"] ** 2 - 2 * EQ["rho"] * EQ["xi"] * EQ["kappa"]; c0 = EQ["kappa"] ** 2
    nr = np.sort(np.roots([c2, c1, c0]).real)
    OUT["u_inf"] = dict(plus=up_inf, minus=um_inf)
    check("u+inf: stable root formula vs numpy.roots (rel)", abs(up_inf - nr[1]) / up_inf, 0, 1e-13)
    check("u-inf: stable root formula vs numpy.roots (rel)", abs(um_inf - nr[0]) / abs(um_inf), 0, 1e-13)
    check("Delta(u+inf) = 0", b_delta(up_inf, EQ)[1], 0, 1e-12)
    check("Delta(u-inf) = 0", b_delta(um_inf, EQ)[1], 0, 1e-12)
    check("b < 0 at both roots (so T* is infinite on the band)", max(b_delta(up_inf, EQ)[0], b_delta(um_inf, EQ)[0]), 0, 0,
          ok=bool(b_delta(up_inf, EQ)[0] < 0 and b_delta(um_inf, EQ)[0] < 0))

    up, um, bR, bL = slopes(T1, EQ)
    upo, umo = ucrit_ode(T1, EQ, +1, up), ucrit_ode(T1, EQ, -1, um)
    OUT["T1"] = dict(u_plus=up, u_minus=um, p_tilde=up - 1, q_tilde=-um, beta_R=bR, beta_L=bL,
                     u_plus_ode=upo, u_minus_ode=umo, ratio_L_R=bL / bR,
                     sigma_R_slope=bR / T1, sigma_L_slope=bL / T1)
    OUT["T1"]["rel_ode_plus"], OUT["T1"]["rel_ode_minus"] = abs(up - upo) / up, abs(um - umo) / abs(um)
    check("u+*(1): closed-form root vs root on the integrated Riccati equation (rel)", abs(up - upo) / up, 0, 1e-10)
    check("u-*(1): closed-form root vs root on the integrated Riccati equation (rel)", abs(um - umo) / abs(um), 0, 1e-10)
    check("u+*(1) = 21.1453729210 (prototype)", up, 21.1453729210, 1e-9)
    check("u-*(1) = -5.3422048572 (prototype)", um, -5.3422048572, 1e-9)
    check("beta_R: 2 (sqrt(r+1) - sqrt r)^2 vs Lee's 2 - 4 (sqrt(r^2 + r) - r)", bR, beta_lee(up - 1), 1e-14)
    check("beta_L: 2 (sqrt(r+1) - sqrt r)^2 vs Lee's 2 - 4 (sqrt(r^2 + r) - r)", bL, beta_lee(-um), 1e-14)
    check("psi(beta_R) = p~ (inverse map)", beta_inv(bR), up - 1, 1e-10)
    check("psi(beta_L) = q~ (inverse map)", beta_inv(bL), -um, 1e-10)
    check("beta_R = 0.024222 (prototype)", bR, 0.024222, 5e-7)
    check("beta_L = 0.085741 (prototype)", bL, 0.085741, 5e-7)
    # the moment explodes as u -> u+*(1) and as u -> u-*(1): log E[S^u] against 1/(distance)
    lm_up = [lm_real(up - 1 - d, T1, EQ) for d in (1e-2, 1e-3, 1e-4)]
    lm_um = [lm_real(um - 1 + d, T1, EQ) for d in (1e-2, 1e-3, 1e-4)]
    OUT["logM_near_up"], OUT["logM_near_um"] = lm_up, lm_um
    check("log E[S^u] grows tenfold per decade towards u+*(1) (pole of B)", lm_up[2] / lm_up[1], 10, 0.3)
    check("log E[S^u] grows tenfold per decade towards u-*(1) (pole of B)", lm_um[2] / lm_um[1], 10, 0.3)

    # monotonicity of the critical moments in maturity
    Tg = np.logspace(-2, 3, 301)
    upg = np.array([ucrit(t, EQ, +1) for t in Tg])
    umg = np.array([ucrit(t, EQ, -1) for t in Tg])
    check("u+*(T) strictly decreasing on 301 maturities in [0.01, 1000]", float(np.max(np.diff(upg))), 0, 0,
          ok=bool(np.all(np.diff(upg) < 0)))
    check("u-*(T) strictly increasing on the same maturities", float(np.min(np.diff(umg))), 0, 0,
          ok=bool(np.all(np.diff(umg) > 0)))
    check("u+*(T) > u+inf and u-*(T) < u-inf on the grid", 0, 0, 0, ok=bool(np.all(upg > up_inf) and np.all(umg < um_inf)))
    # T* monotone on each side
    ug_r = np.linspace(up_inf + 1e-3, 60, 2000); ug_l = np.linspace(um_inf - 1e-3, -60, 2000)
    tr = np.array([tstar(u, EQ) for u in ug_r]); tl = np.array([tstar(u, EQ) for u in ug_l])
    check("T*(u) strictly decreasing in |u| on both sides beyond the band", 0, 0, 0,
          ok=bool(np.all(np.diff(tr) < 0) and np.all(np.diff(tl) < 0)))
    stamp("critical")

    # ── C. short maturity ───────────────────────────────────────────────────────────────────────────────────
    print("short and long maturity ...", flush=True)
    rho, xi = EQ["rho"], EQ["xi"]
    w = xi * np.sqrt(1 - rho * rho)
    c_plus = 2 * np.arccos(rho) / w
    c_minus = -2 * (np.pi - np.arccos(rho)) / w
    sR_lim, sL_lim = w / (4 * np.arccos(rho)), w / (4 * (np.pi - np.arccos(rho)))
    ratio_lim = np.arccos(rho) / (np.pi - np.arccos(rho))
    Ts = [1.0, 0.1, 0.01, 0.001]
    short = []
    for t in Ts:
        a_, b_, bR_, bL_ = slopes(t, EQ)
        short.append(dict(T=t, u_plus=a_, u_minus=b_, upT=a_ * t, umT=b_ * t, gap_p=a_ * t - c_plus, gap_m=b_ * t - c_minus,
                          beta_R=bR_, beta_L=bL_, bR_over_T=bR_ / t, bL_over_T=bL_ / t, ratio=bL_ / bR_))
    for i in range(1, len(short)):
        for k_ in ("gap_p", "gap_m"):
            short[i][f"order_{k_}"] = float(np.log10(short[i - 1][k_] / short[i][k_]))
        for k_, lim in (("bR_over_T", sR_lim), ("bL_over_T", sL_lim), ("ratio", ratio_lim)):
            short[i][f"order_{k_}"] = float(np.log10((short[i - 1][k_] - lim) / (short[i][k_] - lim)))
    OUT["short"] = dict(rows=short, c_plus=c_plus, c_minus=c_minus, slopeR_lim=sR_lim, slopeL_lim=sL_lim,
                        ratio_lim=ratio_lim, arccos_rho=float(np.arccos(rho)))
    s3 = short[-1]
    check("u+*(T) T -> 2 arccos(rho)/(xi sqrt(1-rho^2)) (T = 0.001, rel)", abs(s3["gap_p"]) / c_plus, 0, 1e-3)
    check("u-*(T) T -> -2 (pi - arccos rho)/(xi sqrt(1-rho^2)) (T = 0.001, rel)", abs(s3["gap_m"]) / abs(c_minus), 0, 1e-3)
    check("observed order of u+*(T) T at T = 0.001 is one", s3["order_gap_p"], 1.0, 0.05)
    check("observed order of u-*(T) T at T = 0.001 is one", s3["order_gap_m"], 1.0, 0.05)
    check("beta_R(T)/T -> xi sqrt(1-rho^2)/(4 arccos rho) (T = 0.001, rel)", abs(s3["bR_over_T"] - sR_lim) / sR_lim, 0, 2e-3)
    check("beta_L(T)/T -> xi sqrt(1-rho^2)/(4 (pi - arccos rho)) (T = 0.001, rel)", abs(s3["bL_over_T"] - sL_lim) / sL_lim, 0, 2e-3)
    check("beta_L/beta_R -> arccos rho/(pi - arccos rho) (T = 0.001, rel)", abs(s3["ratio"] - ratio_lim) / ratio_lim, 0, 2e-3)
    check("observed order of beta_L/beta_R at T = 0.001 is one", s3["order_ratio"], 1.0, 0.05)
    check("pi/2 - arctan(rho/sqrt(1-rho^2)) = arccos(rho)", np.pi / 2 - np.arctan(rho / np.sqrt(1 - rho * rho)), np.arccos(rho), 1e-15)
    upo3 = ucrit_ode(0.001, EQ, +1, s3["u_plus"]); umo3 = ucrit_ode(0.001, EQ, -1, s3["u_minus"])
    OUT["short"]["ode_T0.001"] = dict(u_plus=upo3, u_minus=umo3, rel_plus=abs(upo3 - s3["u_plus"]) / s3["u_plus"],
                                      rel_minus=abs(umo3 - s3["u_minus"]) / abs(s3["u_minus"]))
    check("u+*(0.001): closed-form root vs integrated Riccati root (rel)", abs(upo3 - s3["u_plus"]) / s3["u_plus"], 0, 1e-9)
    check("u-*(0.001): closed-form root vs integrated Riccati root (rel)", abs(umo3 - s3["u_minus"]) / abs(s3["u_minus"]), 0, 1e-9)
    check("beta(r) 2r -> 1 (r = 1e4)", beta(1e4) * 2e4, 1.0, 1e-4)

    # ── D. long maturity ────────────────────────────────────────────────────────────────────────────────────
    dD = lambda u: 2 * c2 * u + c1                      # Delta'(u)
    bp, bm = b_delta(up_inf, EQ)[0], b_delta(um_inf, EQ)[0]
    Tl = [1.0, 10.0, 100.0, 1000.0]
    longr = []
    for t in Tl:
        a_, b_, bR_, bL_ = slopes(t, EQ)
        pred_p = 4 * np.pi ** 2 / (abs(dD(up_inf)) * t ** 2)
        pred_m = 4 * np.pi ** 2 / (abs(dD(um_inf)) * t ** 2)
        longr.append(dict(T=t, u_plus=a_, u_minus=b_, gap_p=a_ - up_inf, gap_m=b_ - um_inf, pred_p=pred_p, pred_m=-pred_m,
                          ratio_p=(a_ - up_inf) / pred_p, ratio_m=(b_ - um_inf) / (-pred_m), beta_R=bR_, beta_L=bL_))
    for i in range(1, len(longr)):
        for k_ in ("gap_p", "gap_m"):
            longr[i][f"order_{k_}"] = float(np.log10(longr[i - 1][k_] / longr[i][k_]))
    bR_inf, bL_inf = beta(up_inf - 1), beta(-um_inf)
    OUT["long"] = dict(rows=longr, u_plus_inf=up_inf, u_minus_inf=um_inf, dDelta_plus=dD(up_inf), dDelta_minus=dD(um_inf),
                       b_plus=bp, b_minus=bm, beta_R_inf=bR_inf, beta_L_inf=bL_inf)
    l3 = longr[-1]
    check("u+*(1000) -> u+inf (abs)", l3["gap_p"], 0, 1e-4)
    check("u-*(1000) -> u-inf (abs)", l3["gap_m"], 0, 1e-4)
    check("observed order of u+*(T) - u+inf at T = 1000 is two", l3["order_gap_p"], 2.0, 0.02)
    check("observed order of u-*(T) - u-inf at T = 1000 is two", l3["order_gap_m"], 2.0, 0.02)
    # the rate: T*(u) = 2 pi/s - 2/|b| + O(s^2) near the band, so u - u_inf = 4 pi^2/(|Delta'| (T + 2/|b|)^2)(1 + O(T^-2))
    for r_ in longr:
        t = r_["T"]
        r_["pred2_p"] = 4 * np.pi ** 2 / (abs(dD(up_inf)) * (t + 2 / abs(bp)) ** 2)
        r_["pred2_m"] = -4 * np.pi ** 2 / (abs(dD(um_inf)) * (t + 2 / abs(bm)) ** 2)
        r_["ratio2_p"], r_["ratio2_m"] = r_["gap_p"] / r_["pred2_p"], r_["gap_m"] / r_["pred2_m"]
        r_["shift_p"] = (1 - r_["ratio_p"]) * t * abs(bp) / 4          # -> 1 if the O(1/T) term is -4/(|b| T)
        r_["shift_m"] = (1 - r_["ratio_m"]) * t * abs(bm) / 4
    l2 = longr[-2]
    check("u+*(T) - u+inf = 4 pi^2/(|Delta'(u+inf)| (T + 2/|b(u+inf)|)^2) at T = 1000 (rel)", abs(l3["ratio2_p"] - 1), 0, 1e-5)
    check("u-*(T) - u-inf = -4 pi^2/(|Delta'(u-inf)| (T + 2/|b(u-inf)|)^2) at T = 1000 (rel)", abs(l3["ratio2_m"] - 1), 0, 1e-5)
    check("relative error of the shifted rate falls as T^-2 (+ side, 100 -> 1000)",
          np.log10(abs(l2["ratio2_p"] - 1) / abs(l3["ratio2_p"] - 1)), 2.0, 0.1)
    check("relative error of the shifted rate falls as T^-2 (- side, 100 -> 1000)",
          np.log10(abs(l2["ratio2_m"] - 1) / abs(l3["ratio2_m"] - 1)), 2.0, 0.1)
    check("unshifted rate: relative error -4/(|b| T) (+ side, T = 1000)", l3["shift_p"], 1.0, 1e-2)
    check("unshifted rate: relative error -4/(|b| T) (- side, T = 1000)", l3["shift_m"], 1.0, 1e-2)
    check("|Delta'(u+inf)| = |Delta'(u-inf)| = sqrt of the quadratic's discriminant",
          abs(abs(dD(up_inf)) - np.sqrt(c1 * c1 - 4 * c2 * c0)) + abs(abs(dD(um_inf)) - np.sqrt(c1 * c1 - 4 * c2 * c0)), 0, 1e-13)

    # the long-maturity slopes in closed form: beta_R(inf) = 2 sqrt((1+rho)/(1-rho)) (sqrt(q^2+1) - q),
    # beta_L(inf) = 2 sqrt((1-rho)/(1+rho)) (sqrt(q^2+1) - q), q = (2 kappa - rho xi)/(xi sqrt(1 - rho^2))
    def beta_inf_cf(p):
        r, k, x = p["rho"], p["kappa"], p["xi"]
        q = (2 * k - r * x) / (x * np.sqrt(1 - r * r))
        e = 1.0 / (np.sqrt(q * q + 1) + q)
        return 2 * np.sqrt((1 + r) / (1 - r)) * e, 2 * np.sqrt((1 - r) / (1 + r)) * e

    rng = np.random.default_rng(20251215)
    worst_cf, n_cf = 0.0, 0
    sets = [EQ, POS]
    while len(sets) < 22:
        cand = dict(v0=0.04, theta=0.04, kappa=float(rng.uniform(0.2, 5.0)), xi=float(rng.uniform(0.1, 1.5)),
                    rho=float(rng.uniform(-0.95, 0.95)))
        if cand["kappa"] > cand["rho"] * cand["xi"]:
            sets.append(cand)
    for p_ in sets:
        m_, p_plus = u_inf(p_)
        cfR, cfL = beta_inf_cf(p_)
        worst_cf = max(worst_cf, abs(cfR / beta(p_plus - 1) - 1), abs(cfL / beta(-m_) - 1),
                       abs((cfL / cfR) / ((1 - p_["rho"]) / (1 + p_["rho"])) - 1))
        n_cf += 1
    cfR, cfL = beta_inf_cf(EQ)
    OUT["long"]["beta_inf_closed"] = dict(beta_R=cfR, beta_L=cfL, q=(2 * EQ["kappa"] - EQ["rho"] * EQ["xi"]) / w,
                                          ratio=(1 - EQ["rho"]) / (1 + EQ["rho"]), worst_rel=worst_cf, n_sets=n_cf)
    check(f"beta(inf) closed form vs Lee's map at the roots of Delta, and the ratio (1-rho)/(1+rho), {n_cf} sets (max rel)",
          worst_cf, 0, 1e-12)
    check("beta_R(inf) closed form vs beta_R(1000) from the root of T* = 1000 (rel)", abs(l3["beta_R"] / cfR - 1), 0, 1e-4)
    check("beta_L(inf) closed form vs beta_L(1000) from the root of T* = 1000 (rel)", abs(l3["beta_L"] / cfL - 1), 0, 1e-4)
    cfRp, cfLp = beta_inf_cf(POS)
    OUT["positive"] = dict(beta_inf_closed=dict(beta_R=cfRp, beta_L=cfLp, ratio=(1 - POS["rho"]) / (1 + POS["rho"])))
    check("u+inf = 11.703874716832555 (prototype)", up_inf, 11.703874716832555, 1e-12)
    check("u-inf = -1.5077962854600067 (prototype)", um_inf, -1.5077962854600067, 1e-12)
    upo100 = ucrit_ode(100.0, EQ, +1, longr[2]["u_plus"]); umo100 = ucrit_ode(100.0, EQ, -1, longr[2]["u_minus"])
    OUT["long"]["ode_T100"] = dict(u_plus=upo100, u_minus=umo100, abs_plus=abs(upo100 - longr[2]["u_plus"]),
                                   abs_minus=abs(umo100 - longr[2]["u_minus"]))
    check("u+*(100): closed-form root vs integrated Riccati root (abs)", abs(upo100 - longr[2]["u_plus"]), 0, 1e-8)
    check("u-*(100): closed-form root vs integrated Riccati root (abs)", abs(umo100 - longr[2]["u_minus"]), 0, 1e-8)
    stamp("limits")

    # ── E. slopes across maturity ───────────────────────────────────────────────────────────────────────────
    TG = [0.25, 0.5, 1.0, 2.0, 5.0]
    mat = []
    for t in TG:
        a_, b_, bR_, bL_ = slopes(t, EQ)
        mat.append(dict(T=t, u_plus=a_, u_minus=b_, beta_R=bR_, beta_L=bL_, ratio=bL_ / bR_,
                        sR=bR_ / t, sL=bL_ / t))
    OUT["maturity"] = dict(rows=mat, beta_R_inf=bR_inf, beta_L_inf=bL_inf, short_sR=sR_lim, short_sL=sL_lim,
                           short_ratio=ratio_lim, long_ratio=bL_inf / bR_inf)
    check("beta_R(T) and beta_L(T) increase with T on the grid", 0, 0, 0,
          ok=bool(np.all(np.diff([m["beta_R"] for m in mat]) > 0) and np.all(np.diff([m["beta_L"] for m in mat]) > 0)))
    check("beta_R(T) < beta_R(inf) and beta_L(T) < beta_L(inf) on the grid", 0, 0, 0,
          ok=bool(all(m["beta_R"] < bR_inf and m["beta_L"] < bL_inf for m in mat)))
    bRg = np.array([beta(u - 1) for u in upg]); bLg = np.array([beta(-u) for u in umg])

    # ── F. the moment generating function: little trap vs the complex Riccati equation ─────────────────────
    print("moment generating function ...", flush=True)
    check("E[S_T] = 1 (martingale)", abs(np.exp(logmgf(1.0 + 0j, T1, EQ)) - 1), 0, 1e-14)
    check("E[S_T^0] = 1", abs(np.exp(logmgf(1e-300 + 0j, T1, EQ)) - 1), 0, 1e-14)
    mgf_err = 0.0

    # ── G. prices ───────────────────────────────────────────────────────────────────────────────────────────
    print("prices ...", flush=True)
    xs = np.round(np.arange(0.25, 8.25 + 1e-9, 0.25), 10)
    xw = np.concatenate([-xs[::-1], xs])                          # -8.25 ... 8.25, both wings
    jobs = [("eq", float(x), EQ, up, um) for x in xw]
    ups, ums, bRp, bLp = slopes(T1, POS)
    xpos = [-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0]
    jobs += [("pos", float(x), POS, ups, ums) for x in xpos]
    recs = []
    t_dbl = time.time()
    for tag, x, p, u_p, u_m in jobs:
        a = alpha_opt(x, T1, p, u_p, u_m)
        lp, info = log_price(x, a, T1, p)
        lp2, _ = log_price(x, a, T1, p, refine=2)
        recs.append(dict(tag=tag, x=x, lp=lp, lp_refined=lp2, **info))
        # the characteristic function at the damped line, checked against the complex Riccati equation
        if tag == "eq" and abs(x) in (0.25, 1.0, 4.0, 8.0):
            for v in (0.0, 0.5, 2.0, 10.0, 40.0):
                z = a + 1 + 1j * v
                e1, e2 = logmgf(z, T1, p), logmgf_ode(z, T1, p)
                mgf_err = max(mgf_err, abs(np.exp(e1 - e2) - 1))
    OUT["time_double_all_s"] = round(time.time() - t_dbl, 2)          # includes the refined second pass and 40 ODE checks
    OUT["mgf_vs_riccati_max_rel"] = mgf_err
    check("little-trap E[S^z] vs complex Riccati on the damped lines (40 points, max rel)", mgf_err, 0, 1e-9)
    gl = max(abs(r["lp"] - r["lp_refined"]) for r in recs)
    check("Gauss-Legendre: halving the panel width changes no log price by more than 1e-12", gl, 0, 1e-12)
    OUT["gl_refine_max"] = gl
    OUT["cancel_max"] = max(r["cancel"] for r in recs)
    check("no cancellation: int |f| / int f below 3 at every strike", OUT["cancel_max"], 0, 3)

    # the same prices at MP_DPS digits
    t_mp = time.time()
    with Pool(min(WORKERS, len(recs))) as pool:
        mps = pool.map(log_price_mp, [(r["x"], r["alpha"], T1, EQ if r["tag"] == "eq" else POS, MP_DPS, 2.0) for r in recs])
    OUT["time_mp_pool_s"] = round(time.time() - t_mp, 1)
    worst = 0.0
    for r, s in zip(recs, mps):
        r["lp_mp"] = s
        r["rel_gap"] = abs(np.expm1(r["lp"] - float(mp.mpf(s))))
        worst = max(worst, r["rel_gap"])
    OUT["price_worst_rel_gap"] = worst
    OUT["price_n"] = len(recs)
    check(f"double vs mpmath at {MP_DPS} digits: worst relative price gap over {len(recs)} strikes", worst, 0, 1e-10)
    r8 = next(r for r in recs if r["tag"] == "eq" and r["x"] == 8.0)
    t_ = time.time(); a8 = alpha_opt(8.0, T1, EQ, up, um); log_price(8.0, a8, T1, EQ); OUT["time_one_double_x8_s"] = round(time.time() - t_, 3)
    t_ = time.time(); log_price_mp((8.0, r8["alpha"], T1, EQ, MP_DPS, 2.0)); OUT["time_one_mp_x8_s"] = round(time.time() - t_, 1)
    t_ = time.time(); [slopes(T1, EQ) for _ in range(200)]; OUT["time_slopes_us"] = round((time.time() - t_) / 200 * 1e6, 1)
    # the reference checked against itself: other digits, other panels, hardest strike each wing
    hard = [r for r in recs if r["tag"] == "eq" and abs(r["x"]) == 8.0]
    with Pool(len(hard)) as pool:
        mp2 = pool.map(log_price_mp, [(r["x"], r["alpha"], T1, EQ, 44, 1.0) for r in hard])
    mp.mp.dps = 50
    selfgap = max(abs(mp.mpf(a) - mp.mpf(r["lp_mp"])) for a, r in zip(mp2, hard))
    OUT["mp_self_gap"] = float(selfgap)
    check("mpmath reference: 34 digits, panels 2 vs 44 digits, panels 1 at x = +-8 (abs, log price)", float(selfgap), 0, 1e-28)

    # a different damping gives the same price (the formula, not the integrator)
    alt_gap = 0.0
    for r in recs:
        if r["tag"] != "eq" or abs(r["x"]) not in (0.25, 1.0, 2.0, 4.0, 8.0):
            continue
        x, a = r["x"], r["alpha"]
        g0 = G(a, x, T1, EQ)
        far = 1e-7 if x >= 0 else -1.0 - 1e-7
        a2 = brentq(lambda b: G(b, x, T1, EQ) - g0 - np.log(100.0), far, a) if x >= 0 else \
            brentq(lambda b: G(b, x, T1, EQ) - g0 - np.log(100.0), a, far)
        lp_alt, _ = log_price(x, a2, T1, EQ)
        r["alpha_alt"], r["lp_alt"] = a2, lp_alt
        alt_gap = max(alt_gap, abs(np.expm1(lp_alt - r["lp"])))
    OUT["alpha_alt_max_rel"] = alt_gap
    check("a second damping (integrand 100 times larger at v = 0) gives the same prices (max rel)", alt_gap, 0, 1e-10)

    # put-call parity across the two strips at x = +-0.25: P - C = e^x - 1
    par = 0.0
    for x in (-0.25, 0.25):
        ac = alpha_opt(abs(x) if x < 0 else x, T1, EQ, up, um)       # any a in (0, u+* - 1)
        ap = alpha_opt(-abs(x), T1, EQ, up, um)                       # any a in (u-* - 1, -1)
        C = np.exp(log_price(x, ac, T1, EQ)[0]); Pt = np.exp(log_price(x, ap, T1, EQ)[0])
        par = max(par, abs(Pt - C - np.expm1(x)))
    OUT["parity_max_abs"] = par
    check("put-call parity P - C = e^x - 1 at x = +-0.25 (abs)", par, 0, 1e-13)

    # the undamped evaluation: a = -1/2 lies in (-1, 0), where the integral returns C - 1, not C
    # C - 1 = e^{x/2}/pi int Re[...] at a = -1/2; C = 1 + that, formed in double
    und = {}
    for x in (1.0, 4.0, 8.0):
        a = -0.5
        v, w_, n = _panels(400.0, 0.02)
        z = a + 1 + 1j * v
        f = (np.exp(-1j * v * x + logmgf(z, T1, EQ)) / ((a + 1j * v) * (a + 1 + 1j * v))).real
        cm1 = np.exp(-a * x) / np.pi * float(np.sum(w_ * f))
        ct = float(np.exp(next(r["lp"] for r in recs if r["tag"] == "eq" and r["x"] == x)))
        und[str(x)] = dict(C_minus_1=cm1, C_undamped=1.0 + cm1, C_true=ct, rel_err=abs(1.0 + cm1 - ct) / ct)
    OUT["undamped"] = und
    stamp("prices")

    # ── H. implied total variance ───────────────────────────────────────────────────────────────────────────
    print("implied variance ...", flush=True)
    vgap = 0.0
    rt = 0.0
    for r in recs:
        V = implied_V(r["x"], r["lp"])
        r["V"] = V
        rt = max(rt, abs(log_black_otm(r["x"], np.sqrt(V)) - r["lp"]))
        Vm = implied_V_mp(r["x"], r["lp_mp"], np.sqrt(V))
        r["V_mp"] = Vm
        r["V_rel_gap"] = abs(V - Vm) / Vm
        vgap = max(vgap, r["V_rel_gap"])
        r["V_over_x"] = V / abs(r["x"])
    OUT["V_worst_rel_gap"], OUT["black_roundtrip_max"] = vgap, rt
    check("implied variance: Mills-ratio inversion of the double price vs 50-digit inversion of the mpmath price (max rel)",
          vgap, 0, 1e-10)
    check("Black round trip: log price at the implied volatility (max abs)", rt, 0, 1e-12)
    eq = [r for r in recs if r["tag"] == "eq"]
    calls = sorted([r for r in eq if r["x"] > 0], key=lambda r: r["x"])
    puts = sorted([r for r in eq if r["x"] < 0], key=lambda r: -r["x"])
    check("V(x)/x above beta_R at every call strike", min(r["V_over_x"] for r in calls) - bR, 0, 0,
          ok=bool(all(r["V_over_x"] > bR for r in calls)))
    check("V(x)/|x| above beta_L at every put strike", min(r["V_over_x"] for r in puts) - bL, 0, 0,
          ok=bool(all(r["V_over_x"] > bL for r in puts)))
    check("V(x)/|x| decreasing in |x| on both wings", 0, 0, 0,
          ok=bool(np.all(np.diff([r["V_over_x"] for r in calls]) < 0) and np.all(np.diff([r["V_over_x"] for r in puts]) < 0)))
    check("V(x) < 2|x| at every strike (Lee's bound)", 0, 0, 0, ok=bool(all(r["V"] < 2 * abs(r["x"]) for r in recs)))

    # the convergence table: x = 0.25, 0.5, 1, 2, 4, 8
    XT = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    byx = {(r["tag"], r["x"]): r for r in recs}
    conv = []
    for x in XT:
        c, q = byx[("eq", x)], byx[("eq", -x)]
        conv.append(dict(x=x, call=c["V_over_x"], call_gap=c["V_over_x"] - bR, call_rel=c["V_over_x"] / bR - 1,
                         put=q["V_over_x"], put_gap=q["V_over_x"] - bL, put_rel=q["V_over_x"] / bL - 1,
                         call_price=float(np.exp(c["lp"])), put_price=float(np.exp(q["lp"])),
                         call_V=c["V"], put_V=q["V"], call_sigma=float(np.sqrt(c["V"] / T1)), put_sigma=float(np.sqrt(q["V"] / T1)),
                         call_alpha=c["alpha"], put_alpha=q["alpha"]))
    for i in range(1, len(conv)):
        conv[i]["call_order"] = float(np.log2(conv[i - 1]["call_gap"] / conv[i]["call_gap"]))
        conv[i]["put_order"] = float(np.log2(conv[i - 1]["put_gap"] / conv[i]["put_gap"]))
    OUT["converge"] = conv
    # local slopes dV/dx by central differences on the 0.25 grid
    loc = {}
    for x in (1.0, 2.0, 4.0, 8.0):
        for sgn, key in ((1, "call"), (-1, "put")):
            d = (byx[("eq", sgn * (x + 0.25))]["V"] - byx[("eq", sgn * (x - 0.25))]["V"]) / 0.5
            loc[f"{key}_{x:g}"] = d
    OUT["local_slope"] = loc
    OUT["price_range"] = dict(min=float(min(np.exp(r["lp"]) for r in eq)), max=float(max(np.exp(r["lp"]) for r in eq)),
                              alpha_call=[float(min(r["alpha"] for r in calls)), float(max(r["alpha"] for r in calls))],
                              alpha_put=[float(min(r["alpha"] for r in puts)), float(max(r["alpha"] for r in puts))])
    # positive correlation example
    pos = {f"{r['x']:g}": dict(V_over_x=r["V_over_x"], V=r["V"], price=float(np.exp(r["lp"]))) for r in recs if r["tag"] == "pos"}
    rho_p = POS["rho"]; wp = POS["xi"] * np.sqrt(1 - rho_p ** 2)
    OUT["positive"].update(u_plus=ups, u_minus=ums, beta_R=bRp, beta_L=bLp, ratio_R_L=bRp / bLp, smile=pos,
                           short_ratio_L_R=np.arccos(rho_p) / (np.pi - np.arccos(rho_p)),
                           c_plus=2 * np.arccos(rho_p) / wp, c_minus=-2 * (np.pi - np.arccos(rho_p)) / wp,
                           u_inf=u_inf(POS), beta_R_inf=beta(u_inf(POS)[1] - 1), beta_L_inf=beta(-u_inf(POS)[0]))
    check("positive correlation: beta_R > beta_L at T = 1", bRp - bLp, 0, 0, ok=bool(bRp > bLp))
    check("positive correlation: V/|x| larger on the call side at |x| = 1, 2, 4, 8", 0, 0, 0,
          ok=bool(all(pos[f"{x:g}"]["V_over_x"] > pos[f"{-x:g}"]["V_over_x"] for x in (1.0, 2.0, 4.0, 8.0))))
    a_, b_, bR3, bL3 = slopes(0.001, POS)
    OUT["positive"]["short_T0.001"] = dict(ratio_L_R=bL3 / bR3)
    check("positive correlation: beta_L/beta_R -> arccos rho/(pi - arccos rho) (T = 0.001, rel)",
          abs(bL3 / bR3 - OUT["positive"]["short_ratio_L_R"]) / OUT["positive"]["short_ratio_L_R"], 0, 2e-3)
    stamp("smile")

    # ── I. the maps over (rho, xi) at T = 1 ────────────────────────────────────────────────────────────────
    print("maps ...", flush=True)
    rg = np.linspace(-0.9, 0.9, 73)
    xg = np.linspace(0.1, 1.0, 46)
    BR = np.zeros((xg.size, rg.size)); BL = np.zeros_like(BR)
    for j, xi_ in enumerate(xg):
        for i, r_ in enumerate(rg):
            p = dict(EQ, xi=float(xi_), rho=float(r_))
            _, _, BR[j, i], BL[j, i] = slopes(T1, p)
    i7, j5 = int(np.argmin(abs(rg + 0.7))), int(np.argmin(abs(xg - 0.5)))
    check("map at (rho, xi) = (-0.7, 0.5) equals the equity set", abs(BR[j5, i7] - bR) + abs(BL[j5, i7] - bL), 0, 1e-12)
    check("maps inside (0, 2)", 0, 0, 0, ok=bool(np.all((BR > 0) & (BR < 2) & (BL > 0) & (BL < 2))))
    check("beta_R increases with rho and beta_L decreases with rho at every xi", 0, 0, 0,
          ok=bool(np.all(np.diff(BR, axis=1) > 0) and np.all(np.diff(BL, axis=1) < 0)))
    check("both slopes increase with xi at every rho", 0, 0, 0,
          ok=bool(np.all(np.diff(BR, axis=0) > 0) and np.all(np.diff(BL, axis=0) > 0)))
    OUT["maps"] = dict(beta_R_min=float(BR.min()), beta_R_max=float(BR.max()), beta_L_min=float(BL.min()), beta_L_max=float(BL.max()),
                       at_xi01=dict(beta_R_rho_m09=float(BR[0, 0]), beta_L_rho_m09=float(BL[0, 0])),
                       at_xi1=dict(beta_R_rho_m09=float(BR[-1, 0]), beta_L_rho_m09=float(BL[-1, 0]),
                                   beta_R_rho_09=float(BR[-1, -1]), beta_L_rho_09=float(BL[-1, -1])),
                       at_rho0=dict(beta_R_xi1=float(BR[-1, 36]), beta_L_xi1=float(BL[-1, 36]),
                                    beta_R_xi01=float(BR[0, 36]), beta_L_xi01=float(BL[0, 36])),
                       rho0=float(rg[36]))
    stamp("maps")

    # ── derived numbers the prose quotes ───────────────────────────────────────────────────────────────────
    OUT["mills_cancel_max"] = float(max((abs(r["x"]) + r["V"] / 2) / r["V"] for r in recs))     # ~ b/(b - a) in R(a) - R(b)
    OUT["min_V_over_x"] = float(min(r["V_over_x"] for r in recs))
    OUT["strike_e8"] = float(np.exp(8.0))
    OUT["local_slope_rel"] = {k: v / (bR if k.startswith("call") else bL) - 1 for k, v in loc.items()}
    l2, l3 = longr[-2], longr[-1]
    OUT["long"]["ratio2_order_p"] = float(np.log10(abs(l2["ratio2_p"] - 1) / abs(l3["ratio2_p"] - 1)))
    OUT["long"]["ratio2_order_m"] = float(np.log10(abs(l2["ratio2_m"] - 1) / abs(l3["ratio2_m"] - 1)))
    OUT["long"]["beta1000_rel_R"] = abs(l3["beta_R"] / cfR - 1)
    OUT["long"]["beta1000_rel_L"] = abs(l3["beta_L"] / cfL - 1)
    m5 = mat[-1]
    OUT["maturity"]["short_of_limit_T5"] = dict(R=1 - m5["beta_R"] / bR_inf, L=1 - m5["beta_L"] / bL_inf,
                                                um_gap=m5["u_minus"] - um_inf, up_gap=m5["u_plus"] - up_inf)
    # relative sensitivity of Lee's map, |d log psi^{-1}(r)/dr| = 1/sqrt(r (r + 1)), at the T = 5 exponents
    pr5, qr5 = m5["u_plus"] - 1, -m5["u_minus"]
    OUT["maturity"]["sens_T5"] = dict(call=1 / np.sqrt(pr5 * (pr5 + 1)), put=1 / np.sqrt(qr5 * (qr5 + 1)),
                                      call_pred=(m5["u_plus"] - up_inf) / np.sqrt(pr5 * (pr5 + 1)),
                                      put_pred=(um_inf - m5["u_minus"]) / np.sqrt(qr5 * (qr5 + 1)))
    check("d log psi^{-1}/dr = -1/sqrt(r(r+1)) (finite difference at r = 2, rel)",
          (np.log(beta(2 + 1e-6)) - np.log(beta(2 - 1e-6))) / 2e-6 * np.sqrt(6), -1.0, 1e-8)
    OUT["ex_equity"] = dict(sig_lee_call_8=float(np.sqrt(bR * 8 / T1)), sig_lee_put_8=float(np.sqrt(bL * 8 / T1)),
                            sig_lee_call_1=float(np.sqrt(bR / T1)), sig_lee_put_1=float(np.sqrt(bL / T1)))
    OUT["positive"]["one_minus_um_equity"] = 1 - um

    # ── figure arrays ───────────────────────────────────────────────────────────────────────────────────────
    uR = np.linspace(1.0 + 1e-6, 40.0, 1200); uL = np.linspace(-12.0, -1e-6, 600)
    tR = np.array([tstar(u, EQ) for u in uR]); tL = np.array([tstar(u, EQ) for u in uL])
    pts = [x for x in rows if x["set"] == "equity" and np.isfinite(x["closed"])]
    np.savez_compressed(
        HERE / "results.npz",
        uR=uR, tR=tR, uL=uL, tL=tL,
        ode_u=np.array([x["u"] for x in pts]), ode_t=np.array([x["ode"] for x in pts]),
        u_inf=np.array([um_inf, up_inf]), ucrit_T1=np.array([um, up]),
        T_grid=Tg, up_grid=upg, um_grid=umg, bR_grid=bRg, bL_grid=bLg, c_pm=np.array([c_minus, c_plus]),
        x_calls=np.array([r["x"] for r in calls]), V_calls=np.array([r["V"] for r in calls]),
        x_puts=np.array([r["x"] for r in puts]), V_puts=np.array([r["V"] for r in puts]),
        beta_RL=np.array([bR, bL]),
        rho_grid=rg, xi_grid=xg, map_R=BR, map_L=BL, eq_point=np.array([EQ["rho"], EQ["xi"]]))

    for r in recs:                                                    # keep numbers.json readable
        r.pop("lp_refined", None)
    OUT["prices"] = recs
    OUT["runtime_s"]["total"] = round(time.time() - T0, 1)
    (HERE / "numbers.json").write_text(json.dumps(OUT, indent=1, default=float))
    n_pass = sum(c[4] == "PASS" for c in CHECKS)
    lines = [f"CHECKS for the note on Lee's moment formula in the Heston model — {len(CHECKS)} checks, "
             f"{n_pass} PASS, {len(CHECKS) - n_pass} FAIL", ""]
    for name, v, tgt, tol, verdict, note in CHECKS:
        lines.append(f"{verdict}  {name}: value {v:.10g}, target {tgt:.10g}, tolerance {tol:.3g}" + (f"  ({note})" if note else ""))
    (HERE / "CHECKS.txt").write_text("\n".join(lines) + "\n")
    print(lines[0])
    for l in lines[2:]:
        if l.startswith("FAIL"):
            print(l)
    print(f"wrote results.npz, numbers.json, CHECKS.txt in {OUT['runtime_s']['total']:.0f}s")


if __name__ == "__main__":
    main()
