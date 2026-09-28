#!/usr/bin/env python3
"""Regenerate the reference numbers embedded in `src/density.rs`, independently of the crate.

    target/venv/bin/python tools/density_reference.py

Needs numpy, scipy and mpmath. Nothing here calls the crate or reuses its discretisation (the
exponential lattice of threshold integration); every number comes from a closed form printed in a
paper, or from a general-purpose SciPy routine applied to the equation itself. Section 7 integrates
Richardson's own first-order equations backward from threshold, as he does, and so shares the
formulation with the crate, but not one step of its arithmetic.

1. SIEGERT: Brunel (2000) eq. (21), `1/nu = t_rp + tau sqrt(pi) int_{(Vr-mu)/s}^{(th-mu)/s}
   e^{u^2}(1 + erf u) du`, by `scipy.integrate.quad` of `erfcx(-u)` at relative tolerance 1e-13.
   Brunel's sigma is sqrt(2) times Richardson's. Also the rate of a neuron 30 sigma below
   threshold (E0 -68 mV, sigma 0.6 mV), which the crate's lattice reaches only to first order.
2. BRUNEL'S DENSITY, his eq. (19) with the Heaviside argument read as (Vr - mu)/sigma (his own
   eq. (36) writes the lower limit that way), by `quad` of exp(u^2 - y^2), and its slope in V.
   Also, in `mpmath` at 60 digits, the density at y_theta = 30, where nu0 is below the smallest
   double and the integral of e^{u^2} above the largest, but their product is an ordinary number.
3. THE EIF RATE, by a route unrelated to threshold integration: k(x) = int_{x_lb}^x exp(phi(x) -
   phi(y)) dy obeys k' = 1 + phi'(x) k in the dimensionless voltage x = (V - E)/sigma, with
   phi = x^2/2 - (D_T/sigma)^2 exp((E + sigma x - V_T)/D_T); then 1/r0 = tau int_{x_re}^{x_th} k dx.
   Integrated FORWARD from V_lb with DOP853 below the reset and Radau (stiff near threshold) above.
   Also the resting potential at which the sigma = 2 and sigma = 6 mV rate curves of Richardson's
   Fig. 2A cross, by `brentq` on the difference of two such rates.
4. THE LIF RESPONSE, two closed forms, each checked against the other and against the w -> 0
   limit of item 1 before anything is printed:
   (a) Klett and Lindner, arXiv:2503.07434 (2025), eq. (10) -- the current susceptibility with a
       refractory period, in parabolic-cylinder functions, attributed there to Lindner and
       Schimansky-Geier, PRL 86:2934 (2001), and Vilela and Lindner, PRE 80:031909 (2009), eq. (26)
       without one. Their transform is int e^{+iwt}, Richardson's modulation e^{+iwt}, so
       Richardson's r_E is the complex conjugate.
   (b) Brunel (2000) Appendix A.3, eq. (46), with phi_2 of eq. (44) (confluent hypergeometric M)
       and the particular solution of eq. (42): the network loop X(lambda) = [H(phi~_th - phi~_r)
       + W~_th - W~_r] / [phi~_th - phi~_r e^{-lambda t_rp}], for which (46) reads
       1 = e^{-lambda D} X. With (G, H) = (1, 0) it gives r_E = -nu0 X / sigma_B; with (0, 1),
       Richardson's r_sigma^2 = 2 nu0 X / sigma_B^2.
   And (printed as 4b) the zero-frequency conductance response, the derivative of the rate in x
   for the leak g0(1 + x): tau/(1 + x), sigma^2/(1 + x), E unchanged. `python ... captions` then
   locates (printed as 4c), on (b), the peaks of |r_E| and |r_sigma^2| and the phase zeros that
   the Fig. 1 caption describes (a grid, refined by `minimize_scalar` and `brentq`).
5. BRUNEL'S TABLE 1: the rates, by brentq on nu = siegert(mu(nu), sigma(nu)) with his eq. (20);
   and the global-oscillation frequencies, as the roots of eq. (46): |1 - e^{-lambda D} X| is
   mapped on a grid over growth rates -600..400 /s and frequencies 5..1000 Hz, every local minimum
   is refined by mpmath.findroot, and the roots found are printed, largest real part first. Also
   1 - e^{-lambda D} X itself at one lambda, for a cell driven at nu = 55.8 Hz (not the
   self-consistent rate), with G and H built from that cell's own Siegert rate.
6. THE FAST HOPF LINE at g = 8, D = 1.5 ms, for Fig. 7's network (C_E 1000, J 0.1 mV) and Fig. 2A's
   (C_E 4000, J 0.2 mV): bisection on nu_ext/nu_thr for the drive at which the fast root of eq. (46)
   has zero real part, following the root.
7. THE EIF RESPONSE, which has no closed form, by a route that shares the equations and nothing
   else with the crate: Richardson's first-order system (17)-(24), dP/dV = [(E - V + psi)P -
   tau(J + F)]/sigma^2 and dJ/dV = -lambda P, integrated backward from threshold as he does, but
   as ODEs by `solve_ivp`'s Radau at relative tolerance 1e-10, with dP0/dV carried as a state of
   its own for the variance drive, instead of on the first-order exponential lattice (A3)-(A15). The route is
   checked first on the LIF against eq. (46), item 4's (b), and its stationary rate against item
   3; then it prints the EIF's five responses at six frequencies for Fig. 2's two cases, and
   (`... captions`, printed as 7b) the peaks, phase zeros and phase the Fig. 2 and Fig. 3
   captions describe.

    target/venv/bin/python tools/density_reference.py table1   # section 5 alone, about 25 s
    target/venv/bin/python tools/density_reference.py hopf     # section 6 alone, about 3 s
    target/venv/bin/python tools/density_reference.py extras   # 1b-5b, the additions to 1-5
    target/venv/bin/python tools/density_reference.py eif      # section 7's table
    target/venv/bin/python tools/density_reference.py captions # the caption features, 4 and 7

The whole script takes about twenty minutes on 18 cores: section 4 at 10 kHz, where eq. (46) needs
about 600 digits, and the caption search, several hundred stiff ODE solves of about a second each.
Every reference value it prints is embedded, with its section, in the tests of `src/density.rs`.
"""
import warnings

import numpy as np
import mpmath as mp
from scipy import integrate, optimize, special

# Section 7's Radau builds its Jacobian by finite differences, and for the one state no equation
# depends on (the running integral of p0) it grows the difference step without bound until numpy
# warns of an overflow. That column is zero whatever the step, so the warning says nothing about
# the result; it is silenced for that module only.
warnings.filterwarnings("ignore", "overflow encountered in multiply", RuntimeWarning, r"scipy\.integrate\._ivp")

mp.mp.dps = 30
mV = 1e-3


def siegert(tau, tref, vth, vr, mu, sig_b):
    val, _ = integrate.quad(lambda u: special.erfcx(-u), (vr - mu) / sig_b, (vth - mu) / sig_b,
                            epsabs=0, epsrel=1e-13, limit=400)
    return 1.0 / (tref + tau * np.sqrt(np.pi) * val)


def brunel_p0(tau, tref, vth, vr, mu, sig_b, v):
    nu = siegert(tau, tref, vth, vr, mu, sig_b)
    y, yr, yt = (v - mu) / sig_b, (vr - mu) / sig_b, (vth - mu) / sig_b
    lo = min(max(y, yr), yt)
    val, _ = integrate.quad(lambda u: np.exp(u * u - y * y), lo, yt, epsabs=0, epsrel=1e-13, limit=400)
    p = 2 * nu * tau / sig_b * val
    slope = 2 * nu * tau / sig_b ** 2 * (-2 * y * val - (1.0 if y > yr else 0.0))
    return p, slope


def eif_rate(tau, e0, s, vt, dT, vth, vre, vlb):
    a = dT / s

    def dphi(x):
        return x - a * np.exp((e0 + s * x - vt) / dT)

    xlb, xre, xth = (vlb - e0) / s, (vre - e0) / s, (vth - e0) / s
    s1 = integrate.solve_ivp(lambda x, y: [1.0 + dphi(x) * y[0]], (xlb, xre), [0.0], method="DOP853",
                             rtol=1e-13, atol=1e-16)
    s2 = integrate.solve_ivp(lambda x, y: [1.0 + dphi(x) * y[0], y[0]], (xre, xth), [s1.y[0, -1], 0.0],
                             method="Radau", jac=lambda x, y: [[dphi(x), 0.0], [1.0, 0.0]], rtol=1e-12,
                             atol=1e-18)
    assert s1.status == 0 and s2.status == 0
    return 1.0 / (tau * s2.y[1, -1])


def klett_lindner_rE(tau, tref, vth, vre, e0, s, omega):
    """Richardson's r_E (Hz/V) from Klett and Lindner eq. (10); `s` is Richardson's sigma."""
    w = mp.mpf(omega) * tau
    iw = 1j * w
    zt, zr = mp.mpf(e0 - vth) / s, mp.mpf(e0 - vre) / s
    r0 = siegert(tau, tref, vth, vre, e0, np.sqrt(2) * s)
    e = mp.exp((zr ** 2 - zt ** 2) / 4)
    num = mp.pcfd(iw - 1, zt) - e * mp.pcfd(iw - 1, zr)
    den = mp.pcfd(iw, zt) - mp.exp(iw * mp.mpf(tref) / tau) * e * mp.pcfd(iw, zr)
    chi = (1j * r0 * w / s) / (iw - 1) * num / den  # per unit of the mean in time units of tau
    return complex(chi).conjugate()


def phi2(y, lt):
    """Brunel's eq. (44) and its y-derivative; lt = lambda tau."""
    c1 = mp.sqrt(mp.pi) / mp.gamma((1 + lt) / 2)
    c2 = mp.sqrt(mp.pi) / mp.gamma(lt / 2)
    a1, a2, x = (1 - lt) / 2, 1 - lt / 2, -y * y
    f = c1 * mp.hyp1f1(a1, 0.5, x) + c2 * 2 * y * mp.hyp1f1(a2, 1.5, x)
    df = (c1 * (a1 / 0.5) * mp.hyp1f1(a1 + 1, 1.5, x) * (-2 * y)
          + c2 * (2 * mp.hyp1f1(a2, 1.5, x) + 2 * y * (a2 / 1.5) * mp.hyp1f1(a2 + 1, 2.5, x) * (-2 * y)))
    return f, df


def brunel_loop(yth, yr, lt, ltrp, G, H):
    """X(lambda) of Brunel's eq. (46): 1 = e^{-lambda D} X at an eigenvalue.

    Below the mean, phi_2 ~ |y|^{-lambda tau} e^{-y^2} is the small difference of two terms of order
    |y|^{lambda tau - 1}: evaluating it at y = -10 cancels about 44 digits, so the working precision
    grows with y^2 (and with |lambda tau|, which the Gamma prefactors scale by)."""
    extra = int(max(float(yth) ** 2, float(yr) ** 2) / 2.0 + 0.4 * abs(complex(lt)))
    with mp.workdps(40 + extra):
        return _brunel_loop(mp.mpf(yth), mp.mpf(yr), mp.mpmathify(lt), mp.mpmathify(ltrp), G, H)


def _brunel_loop(yth, yr, lt, ltrp, G, H):
    a = G / (1 + lt)
    b = H / (2 * (2 + lt))

    def tilde(y):
        f, df = phi2(y, lt)
        qp = -a + 2 * b * y          # Q^p (at theta) or its jump (at V_r), eq. (42)
        dqp = 2 * a * y + b * (4 - 4 * y * y)
        g = mp.exp(y * y)            # 1/Wr up to the constant 2 sqrt(pi)/Gamma(lt/2), eq. (45)
        return f * g, (qp * df - dqp * f) * g

    pt, wt = tilde(yth)
    pr, wr = tilde(yr)
    return (H * (pt - pr) + wt - wr) / (pt - pr * mp.exp(-ltrp))


def brunel_responses(tau, tref, vth, vr, mu, sig_b, omega):
    """(r_E in Hz/V, Richardson's r_sigma^2 in Hz/V^2) from Brunel's eq. (46)."""
    nu0 = siegert(tau, tref, vth, vr, mu, sig_b)
    yth, yr = (vth - mu) / sig_b, (vr - mu) / sig_b
    lt, ltrp = 1j * mp.mpf(omega) * tau, 1j * mp.mpf(omega) * tref
    xg = brunel_loop(yth, yr, lt, ltrp, 1, 0)
    xh = brunel_loop(yth, yr, lt, ltrp, 0, 1)
    return complex(-nu0 * xg / sig_b), complex(2 * nu0 * xh / sig_b ** 2)


def brunel_p0_mp(tau, trp, th, vr, mu, sb, v, dps=60):
    """Brunel's eq. (19) and eq. (21) in `mpmath`, for where a double cannot hold nu0 or the integral."""
    with mp.workdps(dps):
        th, vr, mu, sb, v = (mp.mpf(x) for x in (th, vr, mu, sb, v))
        yt, yr, y = (th - mu) / sb, (vr - mu) / sb, (v - mu) / sb
        nu = 1 / (trp + tau * mp.sqrt(mp.pi) * mp.quad(lambda u: mp.exp(u * u) * (1 + mp.erf(u)), [yr, 0, yt]))
        lo = min(max(y, yr), yt)
        return nu, 2 * nu * tau / sb * mp.quad(lambda u: mp.exp(u * u - y * y), [lo, yt])


def conductance_derivative(tau, tref, vth, vr, e, s):
    """d r0/dx for the leak g0(1 + x): tau/(1 + x), sigma^2/(1 + x); `s` is Richardson's sigma."""
    x = 1e-6

    def r(k):
        return siegert(tau / (1 + k), tref, vth, vr, e, np.sqrt(2) * s / np.sqrt(1 + k))

    return (r(x) - r(-x)) / (2 * x)


def ode_response(model, e0, s, vth, vre, vlb, tref, lam, which, tau=20e-3, rtol=1e-10):
    """Section 7: (r0, r_alpha(lambda)) from Richardson's first-order system as ODEs, Radau.

    `model` is None for the LIF or (V_T, D_T) for the EIF; `which` is E, var, g, VT or DT. The
    unknown rate is divided out as in Richardson's (16): p0, and the pairs (pr, jr) and (pa, ja),
    are integrated backward from V_th in x = V_th - V, the drive per unit rate.

    The variance drive is (1/tau) dP0/dV, and the slope d0 = dp0/dV is carried as a state of its
    own, d0' = a' p0 + a d0 (the derivative of p0' = a p0 - c j0), which jumps by c where j0 does.
    Computed instead as a p0 - c j0 it is the difference of two nearly equal numbers near the
    EIF's threshold, whose rounding stalls every stiff solver tried (Radau, BDF, LSODA): the same
    cancellation the module documents for the lattice. The integration stops at V_lb or at
    E - 30 sigma, whichever is higher: below that the density is less than e^{-450} of its peak, and
    a stiff solver run on into subnormal numbers slows to a crawl (it does so on Fig. 1's case i,
    sigma = 1 mV, below about -80 mV)."""
    s2 = s * s
    c = tau / s2
    if model is None:
        vt, dT = 0.0, 1.0
        psi = lambda v: 0.0
        dpsi = lambda v: 0.0
    else:
        vt, dT = model
        psi = lambda v: dT * np.exp((v - vt) / dT)
        dpsi = lambda v: np.exp((v - vt) / dT)
    lr, li = lam.real, lam.imag
    reset = np.exp(-lam * tref)

    def drive(v, p0, d0):
        """F_alpha per unit rate (P0 = r p0), from definitions (13) and (33)."""
        return {"E": -p0 / tau, "var": d0 / tau, "g": (v - e0) * p0 / tau, "VT": psi(v) * p0 / (dT * tau),
                "DT": -psi(v) / (dT * tau) * (1 - (v - vt) / dT) * p0}[which]

    def rhs(x, y, j0):
        v = vth - x
        a = (e0 - v + psi(v)) / s2
        p0, _, prr, pri, jrr, jri, par, pai, jar, jai, d0 = y
        f = drive(v, p0, d0)
        return [-(a * p0 - c * j0), p0,
                -(a * prr - c * jrr), -(a * pri - c * jri),
                lr * prr - li * pri, lr * pri + li * prr,
                -(a * par - c * (jar + f)), -(a * pai - c * jai),
                lr * par - li * pai, lr * pai + li * par,
                -((dpsi(v) - 1.0) / s2 * p0 + a * d0)]

    y0 = np.zeros(11)
    y0[4], y0[10] = 1.0, -c
    kw = dict(method="Radau", rtol=rtol, atol=1e-30)
    above = integrate.solve_ivp(lambda x, y: rhs(x, y, 1.0), (0.0, vth - vre), y0, **kw)
    y1 = above.y[:, -1].copy()
    y1[4] -= reset.real
    y1[5] -= reset.imag
    y1[10] += c
    below = integrate.solve_ivp(lambda x, y: rhs(x, y, 0.0), (vth - vre, vth - max(vlb, e0 - 30 * s)), y1, **kw)
    assert above.status == 0 and below.status == 0
    y = below.y[:, -1]
    free = 1.0 / y[1]
    r0 = free / (1.0 + tref * free)
    return r0, -r0 * complex(y[8], y[9]) / complex(y[4], y[5])


EIF_CASES = {"i": (-45e-3, 2e-3), "ii": (-60e-3, 6e-3)}


def eif_point(args):
    """One EIF response of Richardson's Fig. 2 (V_T -53, D_T 3, V_th 0, V_re -60, V_lb -100 mV)."""
    case, which, f = args
    e0, s = EIF_CASES[case]
    return ode_response((-53e-3, 3e-3), e0, s, 0.0, -60e-3, -100e-3, 0.0, 2j * np.pi * f, which)[1]


def eif_point_loose(args):
    """`eif_point` at relative tolerance 1e-8, to bound the error of the 1e-10 table."""
    case, which, f = args
    e0, s = EIF_CASES[case]
    return ode_response((-53e-3, 3e-3), e0, s, 0.0, -60e-3, -100e-3, 0.0, 2j * np.pi * f, which, rtol=1e-8)[1]


def lif_point(args):
    """One LIF response of Richardson's Fig. 1 from Brunel's eq. (46)."""
    case, which, f = args
    e0, s = {"i": (-45e-3, 1e-3), "ii": (-60e-3, 5e-3)}[case]
    e, v = brunel_responses(20e-3, 0.0, -50e-3, -60e-3, e0, np.sqrt(2) * s, 2 * np.pi * f)
    return e if which == "E" else v


def peaks_and_zeros(point, case, which, grid, pool):
    """Strict local maxima of |r| on `grid`, refined; and sign changes of Im r, refined."""
    vals = list(pool.map(point, [(case, which, f) for f in grid]))
    amp = [abs(z) for z in vals]
    peaks = []
    for k in range(1, len(grid) - 1):
        if amp[k] > amp[k - 1] and amp[k] > amp[k + 1]:
            res = optimize.minimize_scalar(lambda f: -abs(point((case, which, f))), bounds=(grid[k - 1], grid[k + 1]),
                                           method="bounded", options={"xatol": 1e-7})
            peaks.append(res.x)
    zeros = [optimize.brentq(lambda f: point((case, which, f)).imag, grid[k], grid[k + 1], xtol=1e-10)
             for k in range(len(grid) - 1) if (vals[k].imag > 0) != (vals[k + 1].imag > 0)]
    return peaks, zeros


def captions():
    """The features the Fig. 1-3 captions describe: from eq. (46) for the LIF, section 7 for the EIF."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as pool:
        print("# 4c. Fig. 1 from eq. (46): peaks of |r| and zeros of Im r")
        for case, which, grid in [("i", "E", np.arange(20.0, 150.5, 1.0)), ("i", "var", np.arange(20.0, 150.5, 1.0)),
                                  ("ii", "var", np.arange(0.5, 200.25, 0.5))]:
            p, z = peaks_and_zeros(lif_point, case, which, grid, pool)
            print(f"lif case {case} {which}: peaks {[float(x) for x in p]!r} zeros {[float(x) for x in z]!r}")
        print("# 7b. Figs. 2 and 3 from section 7: peaks of |r| and zeros of Im r")
        for case, which, grid in [("i", "E", np.arange(20.0, 151.0, 2.0)), ("i", "VT", np.arange(20.0, 151.0, 2.0)),
                                  ("ii", "var", np.arange(0.5, 200.25, 0.5)), ("ii", "VT", np.arange(2.0, 401.0, 2.0))]:
            p, z = peaks_and_zeros(eif_point, case, which, grid, pool)
            print(f"eif case {case} {which}: peaks {[float(x) for x in p]!r} zeros {[float(x) for x in z]!r}")
            if (case, which) == ("ii", "VT"):
                ratio = abs(eif_point(("ii", "VT", 100.0))) / abs(eif_point(("ii", "VT", p[0])))
                print(f"eif case ii |r_VT(100 Hz)| / |r_VT| at its peak = {float(ratio)!r}")
        low = eif_point(("i", "var", 0.01))
        ten = eif_point(("ii", "var", 10.0))
        print(f"eif case i r_var(0.01 Hz) = {complex(low)!r}; case ii arg r_var(10 Hz) = {float(np.degrees(np.angle(ten)))!r} deg")


def eif_table():
    """Section 7: the route checked on the LIF, then the EIF's responses."""
    from concurrent.futures import ProcessPoolExecutor
    print("# 7. EIF response by Radau on Richardson's first-order system")
    for f in [46.0, 1000.0]:
        for case, (e0, s) in [("i", (-45e-3, 1e-3)), ("ii", (-60e-3, 5e-3))]:
            for which in ["E", "var"]:
                got = ode_response(None, e0, s, -50e-3, -60e-3, -100e-3, 0.0, 2j * np.pi * f, which)[1]
                want = lif_point((case, which, f))
                print(f"    route on the LIF, case {case} {which} at {f} Hz: {abs(got - want) / abs(want):.1e} from eq. (46)")
                assert abs(got - want) < 1e-10 * abs(want)
    sb = 5e-3
    for which, want in zip(["E", "var"], brunel_responses(20e-3, 2e-3, 20e-3, 10e-3, 15e-3, sb, 2 * np.pi * 46.0)):
        got = ode_response(None, 15e-3, sb / np.sqrt(2), 20e-3, 10e-3, -40e-3, 2e-3, 2j * np.pi * 46.0, which)[1]
        print(f"    route on Brunel's refractory LIF, {which} at 46 Hz: {abs(got - want) / abs(want):.1e} from eq. (46)")
        assert abs(got - want) < 1e-10 * abs(want)
    for case, (e0, s) in EIF_CASES.items():
        r0 = ode_response((-53e-3, 3e-3), e0, s, 0.0, -60e-3, -100e-3, 0.0, 2j * np.pi, "E")[0]
        ref = eif_rate(20e-3, e0, s, -53e-3, 3e-3, 0.0, -60e-3, -100e-3)
        print(f"    route's EIF rate, case {case}: {float(r0)!r}, {abs(r0 / ref - 1):.1e} from section 3")
        assert abs(r0 / ref - 1) < 1e-10
    freqs = [1.0, 10.0, 44.0, 100.0, 1000.0, 10000.0]
    jobs = [(case, which, f) for case in EIF_CASES for which in ["E", "var", "g", "VT", "DT"] for f in freqs]
    with ProcessPoolExecutor() as pool:
        vals = list(pool.map(eif_point, jobs))
        loose = list(pool.map(eif_point_loose, jobs))
    worst = max(abs(z - w) / abs(z) for z, w in zip(vals, loose))
    print(f"    the same table at rtol 1e-8 differs from it by at most {worst:.1e}")
    assert worst < 1e-6
    for (case, which, f), z in zip(jobs, vals):
        print(f"eif case {case} {which} f={f} Hz: {float(z.real)!r} {float(z.imag)!r}")


def extras(tau):
    """The one-line additions to sections 1-5."""
    print("# 1b.", "E0 -68 mV, sigma_R 0.6 mV: r0 =",
          repr(float(siegert(tau, 0.0, -50 * mV, -60 * mV, -68 * mV, np.sqrt(2) * 0.6 * mV))))
    nu, p = brunel_p0_mp(tau, 2e-3, 20 * mV, 10 * mV, 17 * mV, 0.1 * mV, 16 * mV)
    print(f"# 2b. mu 17 mV, sigma_B 0.1 mV (y_theta = 30): nu0 = {mp.nstr(nu, 10)} Hz, "
          f"P0(16 mV) = {mp.nstr(p, 10)} /V")

    def gap(e0):
        return (eif_rate(tau, e0, 2 * mV, -53 * mV, 3 * mV, 0.0, -60 * mV, -100 * mV)
                - eif_rate(tau, e0, 6 * mV, -53 * mV, 3 * mV, 0.0, -60 * mV, -100 * mV))

    print("# 3b. the sigma 2 and 6 mV EIF rate curves cross at E0 =",
          repr(optimize.brentq(gap, -55 * mV, -40 * mV, xtol=1e-12)), "V")
    for name, vth, vr, e, sg, tref in [("fig1 case i", -50 * mV, -60 * mV, -45 * mV, 1 * mV, 0.0),
                                        ("fig1 case ii", -50 * mV, -60 * mV, -60 * mV, 5 * mV, 0.0),
                                        ("brunel mu15 sB5 t_rp2", 20 * mV, 10 * mV, 15 * mV, 5 * mV / np.sqrt(2), 2e-3)]:
        print(f"# 4b. {name}: dr0/dx for the leak g0(1 + x) = {float(conductance_derivative(tau, tref, vth, vr, e, sg))!r} Hz")
    characteristic_at_one_point(tau)


def characteristic_at_one_point(tau):
    """Section 5b: 1 - e^{-lambda D} X at lambda = 30 + 2 pi i 150 /s, Table 1's point B network with
    every input firing at 55.8 Hz, and the cell's own Siegert rate in G and H."""
    ce, gam, j, th, vr, trp, dly = 1000, 0.25, 0.1 * mV, 20 * mV, 10 * mV, 2e-3, 1.5e-3
    g, ratio, nu = 6.0, 4.0, 55.8
    nthr = th / (ce * j * tau)
    mu = ce * j * tau * (ratio * nthr + nu * (1 - g * gam))
    sb = np.sqrt(ce * j * j * tau * (ratio * nthr + nu * (1 + g * g * gam)))
    nu0 = siegert(tau, trp, th, vr, mu, sb)
    big_g = ce * j * tau * nu0 * (g * gam - 1) / sb
    big_h = ce * j * j * tau * nu0 * (1 + g * g * gam) / sb ** 2
    lam = mp.mpc(30.0, 2 * np.pi * 150.0)
    x = brunel_loop((th - mu) / sb, (vr - mu) / sb, lam * tau, lam * trp, big_g, big_h)
    char = complex(1 - mp.exp(-lam * dly) * x)
    print(f"# 5b. cell at nu = 55.8 Hz (its own rate {float(nu0)!r} Hz): 1 - e^(-lambda D) X = {char.real!r} {char.imag!r}")


def check_phi2():
    """phi_2 solves (1/2)phi'' + y phi' + (1 - lambda tau) phi = 0 and decays as y -> -inf."""
    lt = mp.mpc(0.3, 2.0)
    for y in [-2.0, -0.5, 0.7, 1.9]:
        y = mp.mpf(y)
        d2 = mp.diff(lambda t: phi2(t, lt)[1], y)
        f, df = phi2(y, lt)
        res = 0.5 * d2 + y * df + (1 - lt) * f
        assert abs(res) < 1e-20 * max(1, abs(f)), (y, res)
        assert abs(df - mp.diff(lambda t: phi2(t, lt)[0], y)) < 1e-20 * max(1, abs(df))
    assert abs(phi2(mp.mpf(-6), lt)[0]) < 1e-12


def main():
    import sys
    tau = 20e-3
    check_phi2()
    if sys.argv[1:] == ["table1"]:
        return table1(tau)
    if sys.argv[1:] == ["hopf"]:
        return hopf(tau)
    if sys.argv[1:] == ["extras"]:
        return extras(tau)
    if sys.argv[1:] == ["eif"]:
        return eif_table()
    if sys.argv[1:] == ["captions"]:
        return captions()
    print("# 1. Siegert (Brunel's sigma_B = sqrt(2) Richardson's), V_th -50, V_re -60 mV, tau 20 ms")
    for e0, s, tref in [(-45, 1, 0.0), (-60, 5, 0.0), (-52, 2, 0.0), (-40, 3, 2e-3), (-65, 4, 5e-3)]:
        r = siegert(tau, tref, -50 * mV, -60 * mV, e0 * mV, np.sqrt(2) * s * mV)
        print(f"E0={e0} mV sigma_R={s} mV t_ref={tref}: r0 = {r!r}")

    print("# 2. Brunel eq. (19): mu 15 mV, sigma_B 5 mV, theta 20, V_r 10 mV, tau 20 ms, t_rp 2 ms")
    args = (tau, 2e-3, 20 * mV, 10 * mV, 15 * mV, 5 * mV)
    print(f"nu0 = {siegert(*args)!r}")
    for v in [-5.0, 5.0, 10.0, 12.5, 17.5, 19.75]:
        p, d = brunel_p0(*args, v * mV)
        print(f"V = {v} mV: P0 = {p!r} /V, dP0/dV = {d!r} /V^2")

    print("# 3. EIF rate: V_T -53, D_T 3, V_th 0, V_re -60, V_lb -100 mV, tau 20 ms")
    for e0, s in [(-45, 2), (-60, 6), (-50, 4)]:
        r = eif_rate(tau, e0 * mV, s * mV, -53 * mV, 3 * mV, 0.0, -60 * mV, -100 * mV)
        print(f"eif E0={e0} mV sigma={s} mV: r0 = {r!r}")
    print("lif through the same route, E0 -45 sigma 1 V_lb -100: r0 =",
          repr(1.0 / (tau * integrate.quad(lambda x: integrate.quad(
              lambda y: np.exp(x * x / 2 - y * y / 2), (-100 + 45) / 1.0, x, epsabs=0, epsrel=1e-13)[0],
              (-60 + 45) / 1.0, (-50 + 45) / 1.0, epsabs=0, epsrel=1e-12)[0])))

    print("# 4. LIF response (Richardson's convention: modulation alpha1 e^{+iwt})")
    cases = [("fig1 case i", -50 * mV, -60 * mV, -45 * mV, 1 * mV, 0.0),
             ("fig1 case ii", -50 * mV, -60 * mV, -60 * mV, 5 * mV, 0.0),
             ("brunel mu15 sB5 t_rp2", 20 * mV, 10 * mV, 15 * mV, 5 * mV / np.sqrt(2), 2e-3)]
    for name, vth, vr, e, sg, tref in cases:
        def r(ee, ss):
            return siegert(tau, tref, vth, vr, ee, np.sqrt(2) * ss)

        h = 1e-7
        de = (r(e + h, sg) - r(e - h, sg)) / (2 * h)
        s2 = sg * sg
        dv = (r(e, np.sqrt(s2 + 1e-12)) - r(e, np.sqrt(s2 - 1e-12))) / 2e-12
        low_e, low_v = brunel_responses(tau, tref, vth, vr, e, np.sqrt(2) * sg, 1e-9)
        low_k = klett_lindner_rE(tau, tref, vth, vr, e, sg, 1e-9)
        assert abs(low_e / de - 1) < 1e-7 and abs(low_k / de - 1) < 1e-7, (low_e, low_k, de)
        assert abs(low_v / dv - 1) < 1e-5, (low_v, dv)
        print(f"{name}: dr0/dE = {de!r} Hz/V, dr0/dsigma^2 = {dv!r} Hz/V^2")
        for f in [1.0, 10.0, 46.0, 100.0, 1000.0, 10000.0]:
            w = 2 * np.pi * f
            be, bv = brunel_responses(tau, tref, vth, vr, e, np.sqrt(2) * sg, w)
            ke = klett_lindner_rE(tau, tref, vth, vr, e, sg, w)
            assert abs(be - ke) < 1e-9 * abs(ke), (f, be, ke)
            print(f"{name} f={f} Hz: r_E = {be.real!r} {be.imag!r}  r_var = {bv.real!r} {bv.imag!r}")

    table1(tau)
    hopf(tau)
    extras(tau)
    eif_table()
    captions()


def network(tau, g, ratio, ce=1000, j=0.1 * mV):
    """Brunel's model A, C_I = C_E/4, D = 1.5 ms: (nu0, char(lambda)) with eq. (20) and eq. (46)."""
    gam, th, vr, trp, dly = 0.25, 20 * mV, 10 * mV, 2e-3, 1.5e-3
    nthr = th / (ce * j * tau)

    def f(nu):
        mu = ce * j * tau * (ratio * nthr + nu * (1 - g * gam))
        sb = np.sqrt(ce * j * j * tau * (ratio * nthr + nu * (1 + g * g * gam)))
        return siegert(tau, trp, th, vr, mu, sb) - nu

    nu0 = optimize.brentq(f, 1e-6, 499.0, xtol=1e-13)
    mu = ce * j * tau * (ratio * nthr + nu0 * (1 - g * gam))
    sb = np.sqrt(ce * j * j * tau * (ratio * nthr + nu0 * (1 + g * g * gam)))
    G = ce * j * tau * nu0 * (g * gam - 1) / sb
    H = ce * j * j * tau * nu0 * (1 + g * g * gam) / sb ** 2
    yth, yr = (th - mu) / sb, (vr - mu) / sb
    return nu0, lambda lam: 1 - mp.exp(-lam * dly) * brunel_loop(yth, yr, lam * tau, lam * trp, G, H)


def hopf(tau):
    """The fast Hopf line at g = 8, D = 1.5 ms: the nu_ext/nu_thr at which the fast root's real part
    is zero, by bisection, following the root. For Fig. 7's network (C_E 1000, J 0.1 mV) and for
    Fig. 2A's (C_E 4000, J 0.2 mV), whose frequency on the line is Fig. 3's full curve."""
    print("# 6. The fast Hopf line at g = 8 (D 1.5 ms), from eq. (46)")
    for ce, j, f0 in [(1000, 0.1 * mV, 150.0), (4000, 0.2 * mV, 130.0)]:
        g, lo, hi = 8.0, 2.0, 3.5
        guess = mp.mpc(0.0, 2 * np.pi * f0)
        for ratio in [lo, hi]:
            _, char = network(tau, g, ratio, ce, j)
            z = mp.findroot(char, guess, tol=1e-24, maxsteps=40)
            print(f"    C_E {ce} ratio {ratio}: fast root {float(z.real)!r} + i 2pi {float(z.imag) / (2 * np.pi)!r} Hz")
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            _, char = network(tau, g, mid, ce, j)
            z = mp.findroot(char, guess, tol=1e-24, maxsteps=40)
            guess = z
            if z.real > 0:
                hi = mid
            else:
                lo = mid
        print(f"C_E {ce}, J {j}: Hopf at nu_ext/nu_thr = {0.5 * (lo + hi)!r}, "
              f"frequency {float(guess.imag) / (2 * np.pi)!r} Hz")


def table1(tau):
    print("# 5. Brunel Table 1: C_E 1000, gamma 0.25, J 0.1 mV, tau 20 ms, theta 20, V_r 10, t_rp 2 ms, D 1.5 ms")
    ce, gam, j, th, vr, trp, dly = 1000, 0.25, 0.1 * mV, 20 * mV, 10 * mV, 2e-3, 1.5e-3
    nthr = th / (ce * j * tau)
    for g, ratio in [(6.0, 4.0), (5.0, 2.0), (4.5, 0.9)]:
        def f(nu):
            mu = ce * j * tau * (ratio * nthr + nu * (1 - g * gam))
            sb = np.sqrt(ce * j * j * tau * (ratio * nthr + nu * (1 + g * g * gam)))
            return siegert(tau, trp, th, vr, mu, sb) - nu
        nu0 = optimize.brentq(f, 1e-6, 499.0, xtol=1e-13)
        mu = ce * j * tau * (ratio * nthr + nu0 * (1 - g * gam))
        sb = np.sqrt(ce * j * j * tau * (ratio * nthr + nu0 * (1 + g * g * gam)))
        G = ce * j * tau * nu0 * (g * gam - 1) / sb
        H = ce * j * j * tau * nu0 * (1 + g * g * gam) / sb ** 2
        yth, yr = (th - mu) / sb, (vr - mu) / sb

        def char(lam):
            return 1 - mp.exp(-lam * dly) * brunel_loop(yth, yr, lam * tau, lam * trp, G, H)

        # Map |char| on a grid of lambda = re + i 2 pi f, take its local minima, and refine each
        # with findroot; a refined root that leaves the window is dropped rather than chased.
        res = np.linspace(-600.0, 400.0, 21)
        fs = np.linspace(5.0, 1000.0, 200)
        grid = np.array([[float(abs(char(mp.mpc(r, 2 * np.pi * f)))) for f in fs] for r in res])
        roots = []
        for a_ in range(1, len(res) - 1):
            for b_ in range(1, len(fs) - 1):
                if grid[a_, b_] <= grid[a_ - 1:a_ + 2, b_ - 1:b_ + 2].min():
                    try:
                        z = mp.findroot(char, mp.mpc(res[a_], 2 * np.pi * fs[b_]), tol=1e-24, maxsteps=40)
                    except (ValueError, ZeroDivisionError):
                        continue
                    if 0 < z.imag < 2 * np.pi * 1100 and abs(z.real) < 800 and \
                            not any(abs(z - q) < 1e-6 * abs(z) for q in roots):
                        roots.append(z)
        roots.sort(key=lambda z: -float(z.real))
        print(f"g={g} nu_ext/nu_thr={ratio}: nu0 = {nu0!r} Hz; mu = {mu!r}, sigma_B = {sb!r}")
        for z in roots:
            print(f"    eigenvalue {float(z.real)!r} + i 2pi {float(z.imag) / (2 * np.pi)!r} Hz")


if __name__ == "__main__":
    main()
