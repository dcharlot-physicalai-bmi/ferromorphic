#!/usr/bin/env python3
"""Reference numbers for `src/hindmarshrose.rs`, from SciPy and NumPy run on the same inputs.

    python3 -m venv /tmp/v && /tmp/v/bin/pip install numpy scipy
    /tmp/v/bin/python tools/hindmarshrose_reference.py      # no gate runs it; about half an hour

Hindmarsh and Rose printed no integrator, step size or spike rule, and their numerical code
(credited to D. A. Evans, p. 101) was never published, so there is no authors' implementation to
run. The reference here is SciPy's `solve_ivp` with the DOP853 eighth-order method at
rtol = atol = 1e-12, spikes found as its events on x - 1 = 0 in the upward direction, and the
Fig. 6 and random-burst runs are repeated with Radau (an implicit fifth-order method) at the same
tolerances so the reference's own error is measured rather than assumed. Equilibria are
`numpy.roots` and eigenvalues `numpy.linalg.eigvals`: companion-matrix and Hessenberg QR, algorithms
independent of the module's closed forms. The module's tests quote the numbers this prints.

Run on numpy 1.26.4 and scipy 1.13.1, and again on numpy 2.0.2 and scipy 1.13.1.
"""
import numpy as np
import scipy
from scipy.integrate import solve_ivp

A, B, C, D = 1.0, 3.0, 1.0, 5.0
X1 = -(1 + 5 ** 0.5) / 2
TOL = dict(rtol=1e-12, atol=1e-12)


def field(t, u, I, r, s, x1):
    x, y, z = u
    return [y - A * x**3 + B * x**2 + I - z, C - D * x**2 - y, r * (s * (x - x1) - z)]


def spike(t, u, *args):
    return u[0] - 1.0


spike.direction = 1


def trough(t, u, *args):
    return field(t, u, *args)[0]


trough.direction = 1  # xdot crossing zero upwards: a local minimum of x


def rest(x1):
    return [x1, C - D * x1**2, 0.0]


def run(I, r, s, x1, T, method="DOP853", u0=None, t0=0.0):
    u0 = u0 if u0 is not None else rest(x1)
    sol = solve_ivp(field, (t0, t0 + T), u0, method=method, events=[spike, trough], args=(I, r, s, x1), **TOL)
    return np.array(sol.t_events[0]), sol.y[:, -1], sol


def both(I, r, s, x1, T):
    a, _, _ = run(I, r, s, x1, T, "DOP853")
    b, _, _ = run(I, r, s, x1, T, "Radau")
    n = min(len(a), len(b))
    return a, (len(a), len(b), float(np.max(np.abs(a[:n] - b[:n]))) if n else 0.0)


def starts(sp, gap):
    return [sp[0]] + [sp[i] for i in range(1, len(sp)) if sp[i] - sp[i - 1] > gap]


def sizes(sp, gap):
    out, n = [], 1
    for i in range(1, len(sp)):
        if sp[i] - sp[i - 1] > gap:
            out.append(n)
            n = 1
        else:
            n += 1
    return out + [n]


def pulse(I, W, r, s, after, x1=X1):
    """Spikes during and after a current I held for W from rest, then zero for `after`."""
    a, u, _ = run(I, r, s, x1, W)
    b, _, sol = run(0.0, r, s, x1, after, u0=list(u), t0=W)
    return len(a), len(b), b, sol


print("numpy", np.__version__, "scipy", scipy.__version__)
print("roots x^3+2x^2-1:", repr(sorted(np.roots([1, 2, 0, -1]).real)))
for name, r, s in [("FIG5A", 0.001, 1.0), ("FIG6", 0.001, 4.0), ("RANDOM", 0.005, 4.0)]:
    x = X1
    J = np.array([[-3 * A * x * x + 2 * B * x, 1, -1], [-2 * D * x, -1, 0], [r * s, 0, -r]])
    print(name, "eig at rest:", [repr(v) for v in sorted(np.linalg.eigvals(J).real)])
rts = np.roots([1, 2, 4, -(1 + 0.0 + 4 * -1.6)])
print("rest of (15) with x1 = -1.6, s = 4, I = 0:", repr([v.real for v in rts if abs(v.imag) < 1e-12][0]))
rts = np.roots([1, 2, 1, -(1 + 0.0 + 1 * -1.6)])
print("rest of (15) with x1 = -1.6, s = 1, I = 0:", repr([v.real for v in rts if abs(v.imag) < 1e-12][0]))
for I in [0.4, 2.0, 4.0]:
    for x1 in [X1, -1.6]:
        rts = np.roots([1, 2, 4, -(1 + I + 4 * x1)])
        xr = [v.real for v in rts if abs(v.imag) < 1e-12][0]
        J = np.array([[-3 * xr * xr + 6 * xr, 1, -1], [-10 * xr, -1, 0], [0.004, 0, -0.001]])
        print(f"I={I} x1={x1!r}: e.p. x={xr!r} eig={[complex(v) for v in np.linalg.eigvals(J)]}")

# Fig. 6, with the exact root and with the printed -1.6.
for x1 in [X1, -1.6]:
    sp, chk = both(0.4, 0.001, 4.0, x1, 3000)
    print(f"6a x1={x1!r}: n={len(sp)} spikes={[repr(v) for v in sp]} dop/radau={chk}")
    sp, chk = both(2.0, 0.001, 4.0, x1, 3000)
    print(f"6b x1={x1!r}: sizes={sizes(sp, 50.0)} starts={[repr(v) for v in starts(sp, 50.0)]} dop/radau={chk}")
    sp, chk = both(4.0, 0.001, 4.0, x1, 6000)
    print(f"6c x1={x1!r}: n={len(sp)} first isi={sp[1] - sp[0]!r} last isi={sp[-1] - sp[-2]!r} dop/radau={chk}")

# Fig. 6a's after-hyperpolarization: the lowest trough after the burst, and where x ends.
_, u, sol = run(0.4, 0.001, 4.0, X1, 6000)
tt, xx = sol.t_events[1], sol.y_events[1][:, 0]
k = int(np.argmin(np.where(tt > 150.0, xx, np.inf)))
print(f"6a after the burst: lowest x={xx[k]!r} at t={tt[k]!r}; x at 6000={u[0]!r}")
# ...its recovery at the times the drawn tail is read, and the onset: where x crosses -1.46.
sol = solve_ivp(field, (0.0, 1500.0), rest(X1), method="DOP853", t_eval=[300.0, 400.0, 856.0, 1205.0], args=(0.4, 0.001, 4.0, X1), **TOL)
print("6a x at t = 300, 400, 856, 1205:", [repr(v) for v in sol.y[0]])


def onset(t, u, *args):
    return u[0] + 1.46


onset.direction = 1
sol = solve_ivp(field, (0.0, 100.0), rest(X1), method="DOP853", events=[onset, spike], args=(0.4, 0.001, 4.0, X1), **TOL)
print(f"6a crosses x = -1.46 at t={sol.t_events[0][0]!r}; first spike at {sol.t_events[1][0]!r}")

# Fig. 6c's caption: "after 1000 time units of continuous firing". The twelve spikes after t = 1000.
sp, _, _ = run(4.0, 0.001, 4.0, X1, 1300)
late = sp[sp >= 1000.0][:12]
print(f"6c twelve spikes after 1000: {late[0]!r} .. {late[-1]!r}, intervals {[round(v, 4) for v in np.diff(late)]}, mean {(late[-1] - late[0]) / 11!r}")

# Fig. 3a: the two-variable model is (15) with z held at zero, which r = 0 does exactly.
x_c = (5 ** 0.5 - 1) / 2 + 1e-3
sp, _, _ = run(0.0, 0.0, 0.0, X1, 400, u0=[x_c, C - D * x_c**2, 0.0])
print("limit cycle I=0: last isi", repr(sp[-1] - sp[-2]), "mean of last ten", repr((sp[-1] - sp[-11]) / 10))
lo, hi = 9.0, 10.0
for _ in range(40):
    mid = 0.5 * (lo + hi)
    during, after, _, _ = pulse(1.0, mid, 0.0, 0.0, 300.0)
    lo, hi = (lo, mid) if after > 0 else (mid, hi)
print(f"3a shortest I=1 pulse that switches rest onto the limit cycle: {lo!r} .. {hi!r}; spikes during it {pulse(1.0, hi, 0.0, 0.0, 300.0)[0]}")
# Longer pulses do not all switch: each edge of the module's 0.01 grid, spikes (during, after).
print("3a (W, during, after):", [(W, *pulse(1.0, W, 0.0, 0.0, 300.0)[:2]) for W in [13.80, 13.81, 14.02, 14.03, 16.09, 16.10]])

# Fig. 5a and 5c: spikes during and after a pulse of I = 1, and the lowest x after the burst.
sweep = [(W, *pulse(1.0, W, 0.001, 1.0, 1500.0)[:2]) for W in np.arange(8.0, 24.01, 0.25)]
print("5a (W, during, after):", [(float(w), a, b) for w, a, b in sweep])
print("5a most spikes:", max(a + b for _, a, b in sweep))
for label, s_ in [("5a", 1.0), ("5c", 4.0)]:
    _, _, _, sol = pulse(1.0, 12.0, 0.001, s_, 6000.0)
    tt, xx = sol.t_events[1], sol.y_events[1][:, 0]
    k = int(np.argmin(np.where(tt > (150.0 if label == "5a" else 30.0), xx, np.inf)))
    print(f"{label} W=12: lowest x after the spikes {xx[k]!r} at t={tt[k]!r}; x at the end {sol.y[0, -1]!r}")
print("5c (W, during, after):", [(W, *pulse(1.0, W, 0.001, 4.0, 1500.0)[:2]) for W in [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]])
print("5a eight spikes (W, during, after):", [(W, *pulse(1.0, W, 0.001, 1.0, 1500.0)[:2]) for W in [44.5, 44.75]])

# Fig. 8: rebound spikes after a step of I = -3 held for W, by whole time units.
reb = {}
for W in range(150, 201):
    during, after, b, _ = pulse(-3.0, float(W), 0.001, 4.0, 1500.0)
    reb[W] = (during, after, float(b[0] - W) if len(b) else None)
print("8 (W: during, after, first spike after release):", reb)
print("8 at a step of 141, the Fig. 6b burst duration (during, after):", pulse(-3.0, 141.0, 0.001, 4.0, 1500.0)[:2])

# Random bursts, p. 98.
sp, chk = both(3.25, 0.005, 4.0, X1, 6000)
isi = np.sort(np.diff(sp))
jumps = np.diff(isi)
k = int(np.argmax(np.where(isi[:-1] > 20.0, jumps, 0.0)))
print(f"random: n={len(sp)} first20={[repr(v) for v in sp[:20]]} dop/radau={chk}")
print(f"random: widest hole in the sorted intervals above 20: {isi[k]!r} to {isi[k + 1]!r}; sizes(gap 60)={sizes(sp, 60.0)}")

# --- The runs the review of the first draft asked for: Fig. 6c's place in the continuous firing,
# the rounding x1 = -1.6 in Figs 5c and 8 from both of its starting points, the band structure of
# Fig. 3a on the module's whole grid, and the random-burst split on this run's own troughs.

# Fig. 6c. The drawn interval is 28.2 px, the mean of the panel's twelve spikes, against Fig. 6b's
# period of 642 px on the p. 98 scan: 0.0440 of the period, and 18.7 or 19.8 time units by the
# "100" bar (151 px end to end, 142.5 px between its tick centres). Where does the model get there?
for x1 in [X1, -1.6]:
    sp6b, _, _ = run(2.0, 0.001, 4.0, x1, 3000)
    st = starts(sp6b, 50.0)
    period = (st[5] - st[1]) / 4
    sp, _, _ = run(4.0, 0.001, 4.0, x1, 2500)
    isi = np.diff(sp)
    first = {v: (sp[int(np.argmax(isi >= v))], isi[int(np.argmax(isi >= v))]) for v in [0.0440 * period, 28.2 / 151 * 100, 28.2 / 142.5 * 100]}
    win = next(k for k in range(len(sp) - 11) if (sp[k + 11] - sp[k]) / 11 >= 0.0440 * period)
    k1000 = int(np.argmax(sp >= 1000.0))
    print(f"6c x1={x1!r}: period {period!r}; first interval at least (0.0440 P, 18.7, 19.8) starts at (t, length) {[(repr(a), repr(b)) for a, b in first.values()]}")
    print(f"   twelve spikes first averaging 0.0440 P: {sp[win]!r} to {sp[win + 11]!r}; the twelve from t = 1000 average {(sp[k1000 + 11] - sp[k1000]) / 11 / period!r} of P")


def pulse_from(I, W, r, s, after, x1, u0):
    """As `pulse`, started from `u0` instead of the point (x1, c - d x1^2, 0)."""
    a, u, _ = run(I, r, s, x1, W, u0=list(u0))
    b, _, _ = run(0.0, r, s, x1, after, u0=list(u), t0=W)
    return len(a), len(b), b


# With x1 = -1.6 the model's own rest is not (-1.6, -11.8, 0): both starts are run.
xp = [v.real for v in np.roots([1, 2, 4, -(1 + 4 * -1.6)]) if abs(v.imag) < 1e-12][0]
STARTS = [("exact", X1, rest(X1)), ("-1.6 from its rest", -1.6, [xp, C - D * xp**2, 4 * (xp + 1.6)]), ("-1.6 from (-1.6, -11.8, 0)", -1.6, rest(-1.6))]
for label, x1, u0 in STARTS:
    print(f"5c {label} (W, during, after):", [(float(W), *pulse_from(1.0, float(W), 0.001, 4.0, 1500.0, x1, u0)[:2]) for W in np.arange(10.0, 15.01, 0.25)])
# The drawn pulse of Fig. 5c is 94.5 px between the centres of its edges, against the "20" bar's
# 172 px end to end and 163.5 px between tick centres on the p. 96 scan: 10.99 or 11.56 wide.
for label, x1, u0 in STARTS:
    print(f"5c {label} at the drawn widths (W, during, after):", [(W, *pulse_from(1.0, W, 0.001, 4.0, 1500.0, x1, u0)[:2]) for W in [10.99, 11.56]])
lo, hi = 10.75, 11.0
for _ in range(30):
    mid = 0.5 * (lo + hi)
    d, a, _ = pulse_from(1.0, mid, 0.001, 4.0, 1500.0, -1.6, STARTS[1][2])
    lo, hi = (lo, mid) if d + a >= 2 else (mid, hi)
print(f"5c -1.6 from its rest: the shortest pulse that gives two spikes is {lo!r} .. {hi!r}")
for label, x1, u0 in STARTS:
    edges, prev, lat = [], None, {}
    for W in range(140, 241):
        d, a, b = pulse_from(-3.0, float(W), 0.001, 4.0, 1500.0, x1, u0)
        lat[W] = float(b[0] - W)
        if (d, a) != prev:
            edges.append((W, d, a, lat[W]))
            prev = (d, a)
    nine = [W for W, d, a, _ in edges if a == 9][0]
    last9 = [W for W, d, a, _ in edges if a == 10][0] - 1
    print(f"8 {label}: (W, during, after, latency) where the count changes {edges}")
    soon = next(W for W in range(140, 241) if lat[W] <= 40.7)
    print(f"   latency at 159 {lat[159]!r}, 169 {lat[169]!r}, last nine at {last9}: {lat[last9]!r}; falling at every step: {all(lat[w + 1] < lat[w] for w in range(140, 240))}")
    print(f"   the first step with its first spike within the drawn 40.7: {soon}, {lat[soon]!r} after release, one step before {lat[soon - 1]!r}")

# Fig. 3a on the module's whole grid: (spikes during the pulse, fires in the 300 after) for every
# pulse of I = 1 from 9.80 to 40.00 in hundredths, from A, and the widths where that pair changes.
import multiprocessing


def switch(k):
    during, after, _, _ = pulse(1.0, k / 100, 0.0, 0.0, 300.0)
    return k, during, after > 0


with multiprocessing.get_context("fork").Pool(6) as pool:
    grid = pool.map(switch, range(980, 4001))
bands, prev = [], None
for k, during, fires in grid:
    if (during, fires) != prev:
        bands.append((k, during, fires))
        prev = (during, fires)
print("3a bands (hundredths, spikes during, fires after):", bands)

# The random bursts split at a gap of 60 against the split by this run's own troughs below -1.
sp, _, sol = run(3.25, 0.005, 4.0, X1, 6000)
tt, xx = sol.t_events[1], sol.y_events[1][:, 0]
by_trough, n = [], 0
for i, t in enumerate(sp):
    if i > 0 and np.any((tt > sp[i - 1]) & (tt < t) & (xx < -1.0)):
        by_trough.append(n)
        n = 0
    n += 1
by_trough.append(n)
print("random: the gap-60 split is the trough split:", by_trough == sizes(sp, 60.0))
