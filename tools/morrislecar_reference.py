#!/usr/bin/env python3
"""Every reference number `src/morrislecar.rs`'s tests hold, computed OUTSIDE the crate.

    python3 tools/morrislecar_reference.py          # needs NumPy and SciPy; not a crate dependency

Morris and Lecar, *Voltage oscillations in the barnacle giant muscle fiber*, Biophysical Journal
35:193-213 (1981). The equations are written here again from the paper, independently of the Rust
module, in the paper's units: mV, ms, uA/cm^2, mmho/cm^2, uF/cm^2, lambda-bar in ms^-1. Where the
page and its own figures disagree this script uses the reading the module documents (leak
g_L(V - V_L); V2 = +15 for the Fig. 9 set; V_L = -50 for Fig. 5a; the calcium term of Eq. 7 entered
inward), and says so at the section.

Tools: SciPy's DOP853 at rtol = atol = 1e-12, sampled on the same time grid the Rust tests use;
`brentq` roots; `numpy.linalg.eigvals` of the analytic Jacobian. Measured on NumPy 1.26.4 and
SciPy 1.13.1. Each section prints the values under the name of the Rust test that holds them.
The paper's own integrator (MLAB's Gear-Nordsieck, p. 195) is not available, and no code by the
authors is known, so a general-purpose solver on the same equations is the reference.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

np.set_printoptions(precision=17)


def minf(v, v1, v2):
    """Eq. 2: 1/2 {1 + tanh[(V - V1)/V2]}."""
    return 0.5 * (1 + np.tanh((v - v1) / v2))


def dminf(v, v1, v2):
    return 0.5 / v2 / np.cosh((v - v1) / v2) ** 2


def lam(v, v1, v2, lb):
    """Eq. 2: lambda-bar cosh[(V - V1)/(2 V2)]."""
    return lb * np.cosh((v - v1) / (2 * v2))


def dlam(v, v1, v2, lb):
    return lb * np.sinh((v - v1) / (2 * v2)) / (2 * v2)


def R(v, rho):
    """Eq. 7 as printed, V{1 - rho e^(V/12.5)}/[1 - e^(V/12.5)]; its limit at V = 0."""
    if v == 0:
        return -12.5 * (1 - rho)
    e = np.exp(v / 12.5)
    return v * (1 - rho * e) / (1 - e)


def dR(v, rho, h=1e-6):
    return (R(v + h, rho) - R(v - h, rho)) / (2 * h)


# Fig. 9 caption, p. 208, with V2 = +15 (printed -15).
FIG9 = dict(c=20.0, gl=2.0, gca=4.0, gk=8.0, vl=-50.0, vca=100.0, vk=-70.0,
            v1=10.0, v2=15.0, lm=0.1, v3=-1.0, v4=14.5, ln=1 / 15)
# Fig. 11 caption, p. 210.
FIG11 = dict(FIG9, v1=-1.0, v2=15.0, v3=10.0, v4=14.5)
# Fig. 6 caption, p. 205; g_Ca, g_K per line.
FIG6 = dict(c=20.0, gl=2.0, vl=-50.0, vca=100.0, vk=-70.0, v1=0.0, v2=15.0, lm=1.0, v3=10.0, v4=10.0, ln=0.1)
# Fig. 2b caption, p. 197: the all-K system of Eq. 3.
ALLK = dict(c=20.0, gl=3.0, vl=-50.0, g=8.0, vi=-70.0, vh=-1.0, s=14.5, lb=1 / 15)
# Fig. 3b caption, p. 198: the all-Ca system of Eqs. 3 and 7.
ALLCA = dict(c=20.0, gl=2.0, vl=-35.0, g=40.0, vh=10.0, s=15.0, lb=0.1)


def iss(p, v):
    """The steady current of Eqs. 1 and 9: g_L(V - V_L) + g_Ca M_inf (V - V_Ca) + g_K N_inf (V - V_K)."""
    return (p['gl'] * (v - p['vl']) + p['gca'] * minf(v, p['v1'], p['v2']) * (v - p['vca'])
            + p['gk'] * minf(v, p['v3'], p['v4']) * (v - p['vk']))


def reduced(p, i):
    """Eq. 9."""
    def f(t, y):
        v, n = y
        dv = (i - p['gl'] * (v - p['vl']) - p['gca'] * minf(v, p['v1'], p['v2']) * (v - p['vca'])
              - p['gk'] * n * (v - p['vk'])) / p['c']
        return [dv, lam(v, p['v3'], p['v4'], p['ln']) * (minf(v, p['v3'], p['v4']) - n)]
    return f


def full(p, i):
    """Eq. 1 with the leak g_L(V - V_L)."""
    def f(t, y):
        v, m, n = y
        dv = (i - p['gl'] * (v - p['vl']) - p['gca'] * m * (v - p['vca']) - p['gk'] * n * (v - p['vk'])) / p['c']
        return [dv, lam(v, p['v1'], p['v2'], p['lm']) * (minf(v, p['v1'], p['v2']) - m),
                lam(v, p['v3'], p['v4'], p['ln']) * (minf(v, p['v3'], p['v4']) - n)]
    return f


def single(p, i, drive):
    """Eq. 3 with the driving force `drive(V)`; the Eq. 7 drive is entered inward."""
    def f(t, y):
        v, mu = y
        return [(i - p['gl'] * (v - p['vl']) - p['g'] * mu * drive(v)) / p['c'],
                lam(v, p['vh'], p['s'], p['lb']) * (minf(v, p['vh'], p['s']) - mu)]
    return f


def single_iss(p, v, drive):
    return p['gl'] * (v - p['vl']) + p['g'] * minf(v, p['vh'], p['s']) * drive(v)


def single_jac(p, v, mu, drive, ddrive):
    return np.array([[(-p['gl'] - p['g'] * mu * ddrive(v)) / p['c'], -p['g'] * drive(v) / p['c']],
                     [dlam(v, p['vh'], p['s'], p['lb']) * (minf(v, p['vh'], p['s']) - mu)
                      + lam(v, p['vh'], p['s'], p['lb']) * dminf(v, p['vh'], p['s']), -lam(v, p['vh'], p['s'], p['lb'])]])


def jac9(p, v, n):
    """The Jacobian of Eq. 9 at (V, N)."""
    m, dm = minf(v, p['v1'], p['v2']), dminf(v, p['v1'], p['v2'])
    ninf, dn = minf(v, p['v3'], p['v4']), dminf(v, p['v3'], p['v4'])
    l, dl = lam(v, p['v3'], p['v4'], p['ln']), dlam(v, p['v3'], p['v4'], p['ln'])
    return np.array([[(-p['gl'] - p['gca'] * m - p['gca'] * dm * (v - p['vca']) - p['gk'] * n) / p['c'],
                      -p['gk'] * (v - p['vk']) / p['c']],
                     [dl * (ninf - n) + l * dn, -l]])


def jac1(p, v):
    """The Jacobian of Eq. 1 at the equilibrium on the curve at V."""
    m, n = minf(v, p['v1'], p['v2']), minf(v, p['v3'], p['v4'])
    c = p['c']
    return np.array([
        [(-p['gl'] - p['gca'] * m - p['gk'] * n) / c, -p['gca'] * (v - p['vca']) / c, -p['gk'] * (v - p['vk']) / c],
        [lam(v, p['v1'], p['v2'], p['lm']) * dminf(v, p['v1'], p['v2']), -lam(v, p['v1'], p['v2'], p['lm']), 0.0],
        [lam(v, p['v3'], p['v4'], p['ln']) * dminf(v, p['v3'], p['v4']), 0.0, -lam(v, p['v3'], p['v4'], p['ln'])]])


def zeros(f, lo, hi, n):
    """Every sign change of f on an even grid, by brentq."""
    vs = np.linspace(lo, hi, n)
    g = np.array([f(v) for v in vs])
    return [brentq(f, vs[k], vs[k + 1], xtol=1e-15, rtol=8.9e-16)
            for k in range(n - 1) if (g[k] < 0) != (g[k + 1] < 0)]


def equilibria(p, i, lo=-200.0, hi=300.0, n=50001):
    return zeros(lambda v: iss(p, v) - i, lo, hi, n)


def run(f, y0, T, dt=0.01):
    t = np.arange(0, round(T / dt) + 1) * dt
    s = solve_ivp(f, (0, t[-1]), y0, t_eval=t, method='DOP853', rtol=1e-12, atol=1e-12)
    return s.t, s.y


def peaks(t, v):
    """V[k-1] < V[k] >= V[k+1], as the Rust tests find them."""
    return [(t[k], v[k]) for k in range(1, len(v) - 1) if v[k] > v[k - 1] and v[k] >= v[k + 1]]


def period(t, v):
    """Mean interval between upward crossings of the mid-level over the second half, as the tests."""
    half = len(v) // 2
    level = 0.5 * (v[half:].min() + v[half:].max())
    ups = [t[k] + (level - v[k]) / (v[k + 1] - v[k]) * (t[k + 1] - t[k])
           for k in range(len(v) - 1) if v[k] < level <= v[k + 1]]
    late = np.diff([u for u in ups if u > t[half]])
    return late.mean(), np.abs(late - late.mean()).max()


def section(name):
    print(f"\n=== {name}")


def main():
    lin_k = lambda v: v - ALLK['vi']
    section("figure_2b_the_all_k_responses: V(200) and first peak, all-K, from V = -50, N = N_inf(-50)")
    for i in [25.0, 100.0, 400.0]:
        t, y = run(single(ALLK, i, lin_k), [-50.0, minf(-50.0, -1.0, 14.5)], 200.0)
        pk = peaks(t, y[0])
        print(i, "V(200)", repr(y[0][-1]), "first peak", repr(pk[0][0]), repr(pk[0][1]))
    section("rates_in_per_second_cannot_draw_figure_2b: lambda-bar_N = 1/15000 ms^-1, I = 400")
    p = dict(ALLK, lb=1 / 15000)
    t, y = run(single(p, 400.0, lin_k), [-50.0, minf(-50.0, -1.0, 14.5)], 200.0)
    print("max V", repr(y[0].max()), "V(200)", repr(y[0][-1]), "N(0)", repr(y[1][0]), "N(200)", repr(y[1][-1]))
    section("the_all_k_point_is_unique_and_a_focus_above_two_microamps")
    def disc(v):
        j = single_jac(ALLK, v, minf(v, -1.0, 14.5), lin_k, lambda v: 1.0)
        return np.trace(j) ** 2 - 4 * np.linalg.det(j)
    for v in zeros(disc, -70.0, 40.0, 11001):
        print("node/focus boundary at V", repr(v), "I", repr(single_iss(ALLK, v, lin_k)))
    for i in [0.0, 25.0, 100.0, 400.0]:
        v = zeros(lambda v: single_iss(ALLK, v, lin_k) - i, -80.0, 60.0, 14001)[0]
        print(i, "equilibrium", repr(v), "eigenvalues", np.linalg.eigvals(single_jac(ALLK, v, minf(v, -1.0, 14.5), lin_k, lambda v: 1.0)))
    t, y = run(single(ALLK, 400.0, lin_k), [-50.0, minf(-50.0, -1.0, 14.5)], 200.0)
    k = 701 + int(np.argmin(y[0][701:3000]))
    print("I = 400 dip", repr(y[0][k]), "at", t[k], "second peak", peaks(t, y[0])[1])

    section("eq_1_as_printed_rests_near_minus_7_mv: the leak read as the constant g_L V_L")
    for i in [0.0, 25.0, 100.0, 400.0]:
        print(i, [repr(v) for v in zeros(lambda v: ALLK['gl'] * ALLK['vl'] + ALLK['g'] * minf(v, -1.0, 14.5) * (v + 70.0) - i, -100.0, 100.0, 20001)])
    printed9 = lambda v: (FIG9['gl'] * FIG9['vl'] + FIG9['gca'] * minf(v, 10.0, 15.0) * (v - FIG9['vca'])
                          + FIG9['gk'] * minf(v, -1.0, 14.5) * (v - FIG9['vk']))
    print("Fig. 9 set, I = 0", [repr(v) for v in zeros(printed9, -100.0, 150.0, 25001)])

    rin = lambda v: R(v, 0.0)
    section("figures_3b_and_5b_the_all_ca_plateau: Eq. 7 drive entered inward")
    for i in [0.0, 100.0, 15.0, 25.0, 50.0]:
        for v in zeros(lambda v: single_iss(ALLCA, v, rin) - i, -60.0, 80.0, 14001):
            j = single_jac(ALLCA, v, minf(v, 10.0, 15.0), rin, lambda v: dR(v, 0.0))
            print(i, repr(v), repr(minf(v, 10.0, 15.0)), np.linalg.eigvals(j))
    t, y = run(single(ALLCA, 15.0, rin), [-35.0, minf(-35.0, 10.0, 15.0)], 200.0)
    k = int(np.argmax(y[0] >= 0.0))
    print("I = 15 crosses 0 mV at", repr(t[k - 1] + (0.0 - y[0][k - 1]) / (y[0][k] - y[0][k - 1]) * 0.01))
    section("figure_5c_accumulation_takes_the_plateau_away: (g*_Ca, [Ca]_i) at I = 0")
    for g, cai in [(40.0, 0.001), (30.0, 0.1), (25.0, 0.5), (20.0, 1.0), (10.0, 10.0)]:
        p = dict(ALLCA, g=g)
        print(g, cai, [repr(v) for v in zeros(lambda v: single_iss(p, v, lambda v: R(v, cai / 100)), -60.0, 80.0, 14001)])

    section("figure_9_equilibria_and_eigenvalues")
    for i in [0.0, 100.0, 200.0, 300.0, 400.0, 475.0, 500.0, 550.0]:
        for v in equilibria(FIG9, i):
            n = minf(v, -1.0, 14.5)
            print(i, repr(v), repr(n), np.linalg.eigvals(jac9(FIG9, v, n)))
    section("figure_8_hopf_currents: trace of Eq. 9's Jacobian on the equilibrium curve")
    for v in zeros(lambda v: np.trace(jac9(FIG9, v, minf(v, -1.0, 14.5))), -60.0, 60.0, 12001):
        print("V", repr(v), "I", repr(iss(FIG9, v)), "omega", repr(np.sqrt(np.linalg.det(jac9(FIG9, v, minf(v, -1.0, 14.5))))))
    section("figure_8_root_locus: trace^2 = 4 det on the curve")
    for v in zeros(lambda v: (lambda j: np.trace(j) ** 2 - 4 * np.linalg.det(j))(jac9(FIG9, v, minf(v, -1.0, 14.5))), -80.0, 80.0, 16001):
        print("V", repr(v), "I", repr(iss(FIG9, v)))
    section("figure_9_as_printed_cannot_oscillate: V2 = -15, I = 0..550 in steps of 1")
    p = dict(FIG9, v2=-15.0)
    worst, complex_seen = -np.inf, False
    for i in np.arange(0.0, 551.0, 1.0):
        for v in equilibria(p, i, n=5001):
            e = np.linalg.eigvals(jac9(p, v, minf(v, -1.0, 14.5)))
            worst, complex_seen = max(worst, e.real.max()), complex_seen or bool(np.any(e.imag != 0))
    print("largest real part", repr(worst), "any complex", complex_seen)
    section("eq_10_holds_only_between_its_limits: V-dot at (V_max, N = 0), I = 400")
    vmax = (FIG9['gl'] * FIG9['vl'] + FIG9['gca'] * FIG9['vca'] + 400.0) / (FIG9['gl'] + FIG9['gca'])
    print(repr(reduced(FIG9, 400.0)(0, [vmax, 0.0])[0]))
    section("figure_9_oscillates_on_less_than_the_rising_limb: zeros of G - gbar on the curve")
    limb = lambda v: (FIG9['gca'] * dminf(v, 10.0, 15.0) * (FIG9['vca'] - v)
                      - (FIG9['gl'] + FIG9['gk'] * minf(v, -1.0, 14.5) + FIG9['gca'] * minf(v, 10.0, 15.0)))
    for v in zeros(limb, -60.0, 60.0, 12001):
        print("V", repr(v), "I", repr(iss(FIG9, v)))

    section("figure_10_periods: Eq. 9 over 3 s from V = -50, N = N_inf(-50), I = 300")
    for name, p in [("control", FIG9), ("10a", dict(FIG9, gca=8.0)), ("10c", dict(FIG9, ln=1 / 30)), ("10d", dict(FIG9, gl=1.0))]:
        t, y = run(reduced(p, 300.0), [-50.0, minf(-50.0, p['v3'], p['v4'])], 3000.0)
        print(name, "period, spread", period(t, y[0]))
    for name, p in [("control", FIG9), ("10b", dict(FIG9, v3=12.0))]:
        t, y = run(reduced(p, 300.0), [-50.0, minf(-50.0, p['v3'], p['v4'])], 200.0)
        half = len(t) // 2
        print(name, "over the figure's 200 ms: period, spread", period(t, y[0]),
              "swing of the second half", repr(y[0][half:].max() - y[0][half:].min()))
    p = dict(FIG9, v3=12.0)
    v = equilibria(p, 300.0)[0]
    print("10b equilibrium", repr(v), np.linalg.eigvals(jac9(p, v, minf(v, 12.0, 14.5))))
    section("figure_10b_is_a_decaying_transient: max |V - V_s| over the 200 ms before each end, 0.05 ms grid")
    s = solve_ivp(reduced(p, 300.0), (0, 60000.0), [-50.0, minf(-50.0, 12.0, 14.5)], method='DOP853',
                  rtol=1e-12, atol=1e-12, dense_output=True)
    for end in [1000, 3000, 10000, 30000, 60000]:
        tt = np.arange(round((end - 200) / 0.05), round(end / 0.05) + 1) * 0.05
        print(end, repr(np.abs(s.sol(tt)[0] - v).max()))
    section("figure_10b_is_a_decaying_transient: Eq. 9's Hopf currents with V3 = 12, and how near I = 300 is")
    tr = lambda q, v: np.trace(jac9(q, v, minf(v, q['v3'], q['v4'])))
    for w in zeros(lambda v: tr(p, v), -60.0, 60.0, 12001):
        print("V", repr(w), "I", repr(iss(p, w)), "det", repr(np.linalg.det(jac9(p, w, minf(w, 12.0, 14.5)))))
    upper = lambda v3: iss(dict(p, v3=v3), zeros(lambda v: tr(dict(p, v3=v3), v), -60.0, 60.0, 12001)[-1])
    print("V3 at which the upper Hopf current is 300", repr(brentq(lambda v3: upper(v3) - 300.0, 11.9, 12.1, xtol=1e-14)))
    for ln in [0.066, 0.0666]:
        q = dict(p, ln=ln)
        w = equilibria(q, 300.0)[0]
        print("lambda_N", ln, np.linalg.eigvals(jac9(q, w, minf(w, 12.0, 14.5))))

    section("figure_11_fires_at_35_not_the_printed_50: Eq. 9 over 4 s")
    for i in [50.0, 35.0]:
        t, y = run(reduced(FIG11, i), [-50.0, minf(-50.0, 10.0, 14.5)], 4000.0)
        half = len(t) // 2
        print(i, "period, spread", period(t, y[0]), "second half min, max", repr(y[0][half:].min()), repr(y[0][half:].max()))
    t, y = run(reduced(FIG11, 35.0), [-50.0, minf(-50.0, 10.0, 14.5)], 1000.0)
    print("I = 35 spike times", [round(a, 2) for a, _ in peaks(t, y[0])])
    diss = lambda v: (iss(FIG11, v + 1e-6) - iss(FIG11, v - 1e-6)) / 2e-6
    for v in zeros(diss, -60.0, 40.0, 10001):
        print("fold V", repr(v), "I", repr(iss(FIG11, v)))

    section("figure_6_labels_are_swapped: Eq. 1 at I = 50 from V = -50, gates at steady state")
    for gca, gk in [(6.0, 12.0), (4.0, 8.0)]:
        p = dict(FIG6, gca=gca, gk=gk)
        t, y = run(full(p, 50.0), [-50.0, minf(-50.0, 0.0, 15.0), minf(-50.0, 10.0, 10.0)], 220.0)
        print(gca, gk, "peaks", [(round(a, 2), repr(b)) for a, b in peaks(t, y[0])])
    section("figure_6_labels_are_swapped: the first peak of each set at I = 30, 40, ..., 120, over 220 ms")
    for i in np.arange(30.0, 121.0, 10.0):
        row = []
        for gca, gk in [(6.0, 12.0), (4.0, 8.0)]:
            p = dict(FIG6, gca=gca, gk=gk)
            t, y = run(full(p, i), [-50.0, minf(-50.0, 0.0, 15.0), minf(-50.0, 10.0, 10.0)], 220.0)
            pk = peaks(t, y[0])
            row.append((round(pk[0][0], 2), repr(pk[0][1])) if pk else None)
        print(i, "(6, 12)", row[0], "(4, 8)", row[1])
    section("figure_6_oscillates_around_a_stable_equilibrium")
    for gca, gk in [(6.0, 12.0), (4.0, 8.0)]:
        p = dict(FIG6, gca=gca, gk=gk)
        v = equilibria(p, 50.0)[0]
        print(gca, gk, repr(v), "Eq. 1", np.linalg.eigvals(jac1(p, v)), "Eq. 9", np.linalg.eigvals(jac9(p, v, minf(v, 10.0, 10.0))))
    p = dict(FIG6, gca=6.0, gk=12.0)
    t, y = run(full(p, 50.0), [-50.0, minf(-50.0, 0.0, 15.0), minf(-50.0, 10.0, 10.0)], 1500.0)
    tail = y[0][140000:]
    print("(6, 12) from V = -50: swing over 1.4 to 1.5 s", repr(tail.max() - tail.min()), "from", repr(tail.min()), "to", repr(tail.max()))
    section("figure_6_does_not_outlast_its_stimulus: (6, 12) at I = 50 from V = -50, switched to I = 0 at 200, 201, ..., 230 ms")
    p = dict(FIG6, gca=6.0, gk=12.0)
    for v in equilibria(p, 0.0):
        print("I = 0 equilibrium", repr(v), "Eq. 1", np.linalg.eigvals(jac1(p, v)))
    rest = equilibria(p, 0.0)[0]
    t, y = run(full(p, 50.0), [-50.0, minf(-50.0, 0.0, 15.0), minf(-50.0, 10.0, 10.0)], 230.0)
    settle, late, spikes = 0.0, 0.0, []
    for off in range(200, 231):
        tt, yy = run(full(p, 0.0), list(y[:, round(off / 0.01)]), 300.0)
        far = np.nonzero(np.abs(yy[0] - rest) > 1.0)[0]
        settle = max(settle, tt[far[-1]])
        late = max(late, np.abs(yy[0][10000:] - rest).max())
        spikes += [(off, round(a, 2)) for a, b in peaks(tt, yy[0]) if b > 0.0]
    print("latest sample more than 1 mV from rest", repr(settle), "ms after the switch; largest |V - rest| from 100 ms on",
          repr(late), "; peaks above 0 mV after the switch (switch time, ms after it)", spikes)
    # How much of the RK4 disagreement on `late` is the reference's: the same classical RK4 the
    # module uses, written here with NumPy, at 0.01 ms and at half that, all 31 switches at once.
    def rk4_late(h):
        def g(y, i):
            v, m, n = y
            return np.array([(i - p['gl'] * (v - p['vl']) - p['gca'] * m * (v - p['vca']) - p['gk'] * n * (v - p['vk'])) / p['c'],
                             lam(v, 0.0, 15.0, 1.0) * (minf(v, 0.0, 15.0) - m), lam(v, 10.0, 10.0, 0.1) * (minf(v, 10.0, 10.0) - n)])
        def step(y, i):
            k1 = g(y, i); k2 = g(y + 0.5 * h * k1, i); k3 = g(y + 0.5 * h * k2, i); k4 = g(y + h * k3, i)
            return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        sub = round(0.01 / h)
        y = np.array([-50.0, minf(-50.0, 0.0, 15.0), minf(-50.0, 10.0, 10.0)])
        kept = [y]
        for k in range(23000 * sub):
            y = step(y, 50.0)
            if (k + 1) % sub == 0:
                kept.append(y)
        Y, worst = np.array([kept[off * 100] for off in range(200, 231)]).T, 0.0
        for k in range(30000 * sub):
            Y = step(Y, 0.0)
            if (k + 1) % sub == 0 and (k + 1) // sub >= 10000:
                worst = max(worst, np.abs(Y[0] - rest).max())
        return worst
    a, b = rk4_late(0.01), rk4_late(0.005)
    print("NumPy RK4: largest |V - rest| from 100 ms on", repr(a), "at 0.01 ms,", repr(b), "at 0.005 ms; moved by", repr(abs(a - b)))
    section("the_reduction_changes_the_singular_point: zeros of a2 a1 - a0 of Eq. 1 on the curve")
    def hur(p, v):
        j = jac1(p, v)
        a1 = sum(np.linalg.det(j[np.ix_(k, k)]) for k in [[0, 1], [0, 2], [1, 2]])
        return -np.trace(j) * a1 + np.linalg.det(j)
    for lm in [0.1, 0.2, 0.5, 1.0, 2.0, 10.0]:
        p = dict(FIG9, lm=lm)
        print(lm, [repr(iss(p, v)) for v in zeros(lambda v: hur(p, v), -60.0, 60.0, 12001)])
    grid = np.linspace(-60.0, 60.0, 4801)
    least = lambda lm: min(hur(dict(FIG9, lm=lm), v) for v in grid)
    for lm in [0.1, 0.2, 0.5, 1.0, 2.0]:
        print(lm, "least a2 a1 - a0 on a 0.025 mV grid", repr(least(lm)))
    print("Hopf points appear at lambda_M", repr(brentq(least, 1.0, 2.0, xtol=1e-13)))
    # Eigenvalues cannot rule out a cycle around a stable point (Fig. 6), so Eq. 1 is also RUN, from
    # the start the Fig. 10 tests use, for 3 s: the largest |V - V_s| over the last second.
    for name, p, currents in [("Fig. 9", FIG9, [300.0, 350.0, 400.0, 450.0]), ("10a", dict(FIG9, gca=8.0), [300.0]),
                              ("10b", dict(FIG9, v3=12.0), [300.0]), ("10c", dict(FIG9, ln=1 / 30), [300.0]),
                              ("10d", dict(FIG9, gl=1.0), [300.0])]:
        for i in currents:
            v = equilibria(p, i)
            t, y = run(full(p, i), [-50.0, minf(-50.0, p['v1'], p['v2']), minf(-50.0, p['v3'], p['v4'])], 3000.0)
            print(name, i, "equilibria", [repr(w) for w in v], "Eq. 1 slowest", max(np.linalg.eigvals(jac1(p, v[0])).real),
                  "last second max |V - V_s|", repr(np.abs(y[0][200000:] - v[0]).max()))

    section("figure_12a_is_a_sketch: I = 300")
    for gk, gca in [(20.0, 1.0), (15.0, 2.0), (20.0, 5.0), (5.0, 3.0)]:
        p = dict(FIG9, gk=gk, gca=gca)
        r = equilibria(p, 300.0)
        print(gk, gca, [repr(v) for v in r], [np.linalg.eigvals(jac9(p, v, minf(v, -1.0, 14.5))) for v in r])
    # The integer points at least 0.5 mmho/cm^2 under the lower edge of the drawn damped band.
    under = {9: 3, 10: 4, 11: 5, 12: 5, 13: 6, 14: 6, 15: 6, 16: 7, 17: 7, 18: 7, 19: 8, 20: 8}
    seen = set()
    for gk, top in under.items():
        for gca in range(1, top + 1):
            p = dict(FIG9, gk=float(gk), gca=float(gca))
            r = equilibria(p, 300.0)
            e = np.linalg.eigvals(jac9(p, r[0], minf(r[0], -1.0, 14.5)))
            seen.add((len(r), bool(np.all(e.real < 0)), bool(np.any(e.imag != 0))))
    print(sum(under.values()), "points under the drawn band; (count, stable, complex) seen:", seen)

    section("figure_12b_is_not_everywhere_bistable: I = 0")
    def eqs(gk, gca):
        p = dict(FIG9, gk=gk, gca=gca)
        return p, equilibria(p, 0.0, -120.0, 150.0, 27001)
    for gk, gca in [(13.0, 20.0), (12.8, 19.5), (8.0, 13.0)]:
        p, r = eqs(gk, gca)
        print(gk, gca, [repr(v) for v in r], [np.linalg.eigvals(jac9(p, v, minf(v, -1.0, 14.5))) for v in r])
        up = r[-1]
        for dv in [0.01, -0.01]:
            t, y = run(reduced(p, 0.0), [up + dv, minf(up, -1.0, 14.5)], 3000.0)
            print("   from the focus", dv, "ends at", repr(y[0][-1]), "rest", repr(r[0]))
    for gca in [20.0]:
        print("three points end at g_K", repr(brentq(lambda gk: minimize_scalar(lambda v: iss(dict(FIG9, gk=gk, gca=gca), v),
              bounds=(10.0, 35.0), method='bounded', options={'xatol': 1e-12}).fun, 13.0, 13.5, xtol=1e-14)))
    for gca, (a, b), (c, d) in [(13.0, (7.9, 8.01), (8.01, 8.21)), (19.5, (12.7, 12.8), (12.8, 13.0))]:
        up_trace = lambda gk: (lambda pr: np.trace(jac9(pr[0], pr[1][-1], minf(pr[1][-1], -1.0, 14.5))))(eqs(gk, gca))
        dip = lambda gk: minimize_scalar(lambda v: iss(dict(FIG9, gk=gk, gca=gca), v), bounds=(10.0, 35.0),
                                         method='bounded', options={'xatol': 1e-12}).fun
        print(gca, "upper point unstable from g_K", repr(brentq(up_trace, a, b, xtol=1e-14)),
              "to the fold at", repr(brentq(dip, c, d, xtol=1e-14, rtol=8.9e-16)))

    dip = lambda gk, gca: minimize_scalar(lambda v: iss(dict(FIG9, gk=gk, gca=gca), v), bounds=(0.0, 40.0),
                                          method='bounded', options={'xatol': 1e-12})
    def trace_at_fold(gca):
        gk = brentq(lambda g: dip(g, gca).fun, 0.0, 20.0, xtol=1e-14, rtol=8.9e-16)
        # The minimiser places the fold only to about 1e-7 mV, where I_ss is flat; the zero of the
        # analytic I_ss' beside it places it to the last bits, and the trace there to 1e-15.
        p = dict(FIG9, gk=gk, gca=gca)
        slope = lambda v: (p['gl'] + p['gca'] * (dminf(v, 10.0, 15.0) * (v - p['vca']) + minf(v, 10.0, 15.0))
                           + p['gk'] * (dminf(v, -1.0, 14.5) * (v - p['vk']) + minf(v, -1.0, 14.5)))
        x = dip(gk, gca).x
        v = brentq(slope, x - 0.5, x + 0.5, xtol=1e-15, rtol=8.9e-16)
        return np.trace(jac9(p, v, minf(v, -1.0, 14.5)))
    print("trace at the fold, g_Ca = 5.6 and 5.7:", repr(trace_at_fold(5.6)), repr(trace_at_fold(5.7)))
    print("the sliver begins at g_Ca", repr(brentq(trace_at_fold, 5.5, 5.75, xtol=1e-13)))
    print("with g_K = 0 the three points begin at g_Ca", repr(brentq(lambda gca: dip(0.0, gca).fun, 2.0, 3.0, xtol=1e-14)))
    # The computed edge at the g_Ca where the drawn one was digitised (600 dpi, the g_K tick marks):
    # (3, 0.82), (4.5, 1.77), (6, 2.74), (13, 7.74), (15, 9.42), (16.5, 10.76), (19.5, 13.79).
    for gca in [3.0, 4.5, 6.0, 13.0, 15.0, 16.5, 19.5]:
        print("fold at g_Ca", gca, "g_K", repr(brentq(lambda g: dip(g, gca).fun, 0.0, 20.0, xtol=1e-14, rtol=8.9e-16)))

    section("the_linear_all_ca_system")
    lin = dict(ALLCA)
    for i in [0.0, 15.0, 25.0, 50.0]:
        print(i, [repr(v) for v in zeros(lambda v: single_iss(lin, v, lambda v: v - 100.0) - i, -60.0, 120.0, 18001)])
    ca9 = dict(c=20.0, gl=2.0, vl=-50.0, g=4.0, vh=10.0, s=15.0, lb=0.1)
    print("Fig. 9 gate", [repr(v) for v in zeros(lambda v: single_iss(ca9, v, lambda v: v - 100.0), -80.0, 120.0, 20001)])

    section("eq_8_as_printed_cannot_draw_figure_3d")
    v, rho = 28.0, 0.001 / 100.0
    m = minf(v, 10.0, 15.0)
    print("printed dV/dt", repr((0 - 2 * (v + 35) + 40 * m * v * R(v, rho)) / 20),
          "inward Eq. 3", repr((0 - 2 * (v + 35) - 40 * m * R(v, rho)) / 20))
    print("1/(F x 1 um), mM/ms per uA/cm^2", repr(1e-6 / (96485.33212 * 1e-4) * 1e6 / 1e3))
    def f3d(t, y):
        v, ca = y
        m = minf(v, 10.0, 15.0)
        r = R(v, ca / 100.0)
        return [(0 - 2 * (v + 35) - 40 * m * r) / 20, -1e-4 * 40 * m * r]
    ev = lambda t, y: y[0] + 20.0
    ev.terminal, ev.direction = True, -1
    s = solve_ivp(f3d, (0, 6000.0), [28.0, 0.001], method='DOP853', rtol=1e-12, atol=1e-12, events=ev, dense_output=True)
    print("through -20 mV at", repr(s.t_events[0][0]), "ms, [Ca]_i", repr(s.y_events[0][0][1]),
          "V at 1, 2, 3 s", [repr(s.sol(t)[0]) for t in [1000.0, 2000.0, 3000.0]])


if __name__ == "__main__":
    main()
