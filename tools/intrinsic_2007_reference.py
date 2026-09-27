#!/usr/bin/env python3
"""Reference values for the Triesch 2007 part of `src/intrinsic.rs`, independent of the crate.

    python3 tools/intrinsic_2007_reference.py                       # the quadratures
    python3 tools/intrinsic_2007_reference.py papers/triesch-2007.pdf    # and the figure readings

J. Triesch, *Synergies Between Intrinsic and Synaptic Plasticity Mechanisms*, Neural Computation
19(4):885-909, 2007 (doi:10.1162/neco.2007.19.4.885). The paper is not in this repository; the
figure readings need a local copy (and `pdftocairo` from poppler), and are skipped without one.

What it computes, with SciPy's adaptive quadrature and root finders, never with the crate's code:

  1. E[Omega(y)] under the exponential rate density of mean mu, and the thresholds at which it is
     zero (eq. 3.4) found by `brentq`: theta_cov = mu and theta_BCM = 2 mu.
  2. Eq. B.9, each cluster's contribution, by `quad` in y itself over
     [F^-1(1 - i/N), F^-1(1 - (i-1)/N)] with F^-1(p) = -mu log(1 - p) (eq. B.4), the first
     cluster's interval running to infinity.
  3. Fig. 6's normalised contributions (sum of squares one) under each reading of eq. B.10.
  4. Eq. 3.2's Laplace band, and Fig. 3d's Laplace-Gauss plane: total mass, covariance and fourth
     moments by `dblquad`.
  5. With the PDF: the vector paths of Figs. 1c, 2a, 3c, 3d, 4, 6, 7c and 8, read through each
     panel's tick marks (or its frame, where it has none), and the comparisons the module doc
     quotes.
"""
import math
import os
import re
import subprocess
import sys
import tempfile

import warnings

import numpy as np
import scipy
from scipy import integrate, optimize
from scipy.special import log_ndtr

LN2 = math.log(2.0)

# `quad` warns that roundoff keeps it from 1e-13 relative on a few of the moments below; the values
# it returns agree with the closed forms to a few units in the last place, which the output shows.
warnings.simplefilter("ignore", integrate.IntegrationWarning)


def omega(rule, theta):
    if rule == "hebb":
        return lambda y: y
    if rule == "cov":
        return lambda y: y - theta
    return lambda y: (y - theta) * y


def expected(rule, theta, mu):
    f = omega(rule, theta)
    v, _ = integrate.quad(lambda y: f(y) * math.exp(-y / mu) / mu, 0, np.inf, epsabs=0, epsrel=1e-13)
    return v


def b9(rule, theta, mu, n, i):
    """Eq. B.9 by quadrature in y."""
    f = omega(rule, theta)
    lo = -mu * math.log(i / n)
    hi = np.inf if i == 1 else -mu * math.log((i - 1) / n)
    v, _ = integrate.quad(lambda y: f(y) * math.exp(-y / mu) / mu, lo, hi, epsabs=0, epsrel=1e-13, limit=200)
    return v


def xlogx(x):
    return np.where(x > 0, x * np.log(np.where(x > 0, x, 1.0)), 0.0)


def printed_b8(n):
    i = np.arange(1, n + 1)
    return 1 + np.log(n) - xlogx(i) + xlogx(i - 1)


def printed_b11(n):
    i = np.arange(1, n + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(i > 0, i * np.log(i / n) ** 2, 0.0)
        b = np.where(i - 1 > 0, (i - 1) * np.log(np.where(i > 1, (i - 1) / n, 1.0)) ** 2, 0.0)
    return a - b


def unit(v):
    return v / np.linalg.norm(v)


def quadratures():
    print(f"SciPy {scipy.__version__}, NumPy {np.__version__}")
    print("\n# 1. E[Omega] under the exponential of mean mu, and the balanced thresholds (eq. 3.4)")
    for mu in (0.1, 0.05):
        print(f"mu = {mu}: E[y] = {expected('hebb', 0, mu)!r}, E[y^2] = {expected('bcm', 0, mu)!r}")
        tc = optimize.brentq(lambda t: expected("cov", t, mu), 0, 1, xtol=1e-16, rtol=1e-15)
        tb = optimize.brentq(lambda t: expected("bcm", t, mu), 0, 1, xtol=1e-16, rtol=1e-15)
        print(f"  theta_cov = {tc!r} (mu = {mu}), theta_BCM = {tb!r} (2 mu = {2 * mu})")
        print(f"  median: the exponential's cdf at mu ln 2 = {1 - math.exp(-LN2)!r}")

    print("\n# 2. Eq. B.9 by quad in y, mu = 0.1: (rule, theta, N, i) -> contribution")
    mu = 0.1
    cases = []
    for n, idx in ((50, (1, 2, 7, 19, 50)), (2, (1, 2))):
        for i in idx:
            cases += [("hebb", 0.0, n, i), ("cov", mu, n, i), ("bcm", 2 * mu, n, i)]
    for i in (1, 7, 50):
        cases += [("cov", 0.03, 50, i), ("bcm", 0.07, 50, i), ("cov", mu * LN2, 50, i), ("bcm", mu * LN2, 50, i)]
    for rule, theta, n, i in cases:
        print(f"  ({rule}, {theta!r}, {n}, {i}) -> {b9(rule, theta, mu, n, i)!r}")
    for rule, theta in (("hebb", 0.0), ("cov", mu), ("bcm", 2 * mu)):
        total = sum(b9(rule, theta, mu, 50, i) for i in range(1, 51))
        print(f"  sum over the 50 clusters of {rule}: {total!r}")

    print("\n# 3. Fig. 6, N = 50: normalised contributions under each reading")
    n = 50
    b8 = printed_b8(n)
    readings = {
        "simple Hebb, B.8": unit(b8),
        "covariance by B.9 (theta = mu): B.8 - 1": unit(b8 - 1),
        "B.10 as printed at B.8's scale, mu = 0.1: B.8 - mu/N": unit(b8 - 0.1 / n),
        "B.8 - mu, mu = 0.1": unit(b8 - 0.1),
        "BCM, B.11": unit(printed_b11(n)),
    }
    for name, f in readings.items():
        neg = np.nonzero(f < 0)[0] + 1
        print(f"  {name}: f_1 = {f[0]:.6f}, f_50 = {f[-1]:.6f}, min {f.min():.6f}, "
              f"{len(neg)} negative{(' (i = ' + str(neg[0]) + ' to ' + str(neg[-1]) + ')') if len(neg) else ''}")
    for mu in (0.1, 0.5, 0.99):
        print(f"  B.8 - mu/N negative at mu = {mu}: {int((b8 - mu / n < 0).sum())}; N B.8(N) = {n * b8[-1]:.6f}")

    print("\n# 4. The planes of Fig. 3")
    s3 = math.sqrt(3.0)
    band = lambda u2, u1: math.exp(-math.sqrt(2) * abs(u1)) / (2 * math.sqrt(6))
    gauss = lambda u2, u1: math.exp(-math.sqrt(2) * abs(u1)) / math.sqrt(2) * math.exp(-u2 * u2 / 2) / math.sqrt(2 * math.pi)
    for name, dens, lo, hi in (("band", band, -s3, s3), ("Laplace-Gauss", gauss, -40, 40)):
        m = lambda g: integrate.dblquad(lambda u2, u1: g(u1, u2) * dens(u2, u1), -60, 60, lo, hi, epsabs=1e-13, epsrel=1e-13)[0]
        print(f"  {name}: mass {m(lambda a, b: 1.0)!r}, E[u1^2] {m(lambda a, b: a * a)!r}, "
              f"E[u2^2] {m(lambda a, b: b * b)!r}, E[u1 u2] {m(lambda a, b: a * b)!r}, "
              f"E[u1^4] {m(lambda a, b: a ** 4)!r}, E[u2^4] {m(lambda a, b: b ** 4)!r}")


def svg_page(pdf, page):
    out = os.path.join(tempfile.mkdtemp(), "p.svg")
    subprocess.run(["pdftocairo", "-svg", "-f", str(page), "-l", str(page), pdf, out], check=True)
    s = open(out).read()
    return s[s.index("</defs>"):]


def paths(body):
    return [re.search(r' d="([^"]*)"', a).group(1) for a in re.findall(r"<path([^>]*)>", body) if ' d="' in a]


def vertices(d):
    return np.array([[float(x), float(y)] for x, y in re.findall(r"[ML] ([-\d.]+) ([-\d.]+)", d)])


def calibrate(ticks, values):
    return np.polyfit(np.array(ticks), np.array(values), 1)


def figures(pdf2007):
    print("\n# 5. Figure readings, pdftocairo -svg, axes fitted by least squares to each panel's tick marks")

    # Fig. 6, p. 897 (page 13). Tick marks: x at 0, 10, ..., 50; y at -0.2, 0, ..., 0.8.
    body = svg_page(pdf2007, 13)
    ps = paths(body)
    px = calibrate([157.609375, 186.171875, 214.730469, 243.292969, 271.855469, 300.417969], [0, 10, 20, 30, 40, 50])
    py = calibrate([457.964844, 477.121094, 496.242188, 515.363281, 534.484375, 553.605469], [-0.2, 0, 0.2, 0.4, 0.6, 0.8])
    lines = [p for p in ps if p.startswith("M 160.457031")]
    n = 50
    b8 = printed_b8(n)
    fig6 = {}
    for name, d in zip(("simple Hebb", "covariance", "BCM"), lines):
        v = vertices(d)
        i = np.rint(np.polyval(px, v[:, 0])).astype(int)
        y = np.polyval(py, v[:, 1])
        fig6[name] = (i, y)
        print(f"  Fig. 6 {name}: {len(i)} vertices, clusters without one {sorted(set(range(1, 51)) - set(i))}, "
              f"x within {np.abs(np.polyval(px, v[:, 0]) - i).max():.4f} of an integer")
        print("    drawn: [" + ", ".join(f"{t:.4f}" for t in y) + "]")
    i, y = fig6["simple Hebb"]
    print(f"  simple Hebb within {np.abs(y - unit(b8)[i - 1]).max():.3e} of normalised B.8")
    i, y = fig6["BCM"]
    print(f"  BCM within {np.abs(y - unit(printed_b11(n))[i - 1]).max():.3e} of normalised B.11")
    i, y = fig6["covariance"]
    for label, c in (("B.8 - 1", 1.0), ("B.8 - mu/N, mu = 0.1", 0.1 / n), ("B.8 - mu, mu = 0.1", 0.1)):
        print(f"  covariance against {label}: largest miss {np.abs(y - unit(b8 - c)[i - 1]).max():.4e}")
    fit = optimize.minimize_scalar(lambda c: np.sum((y - unit(b8 - c)[i - 1]) ** 2), bounds=(-1, 2), method="bounded", options={"xatol": 1e-10})
    print(f"  covariance: least-squares c in B.8 - c = {fit.x:.6f}, largest miss {np.abs(y - unit(b8 - fit.x)[i - 1]).max():.4e}, "
          f"{int((y < 0).sum())} drawn vertices below zero")
    # The same with the frame (y = -0.2 at 457.964844, 1.0 at 572.726562) as the calibration.
    pf = calibrate([457.964844, 572.726562], [-0.2, 1.0])
    v = vertices(lines[1])
    yf = np.polyval(pf, v[:, 1])
    fit = optimize.minimize_scalar(lambda c: np.sum((yf - unit(b8 - c)[i - 1]) ** 2), bounds=(-1, 2), method="bounded", options={"xatol": 1e-10})
    print(f"  covariance through the frame: c = {fit.x:.6f}, largest miss {np.abs(yf - unit(b8 - fit.x)[i - 1]).max():.4e}")

    # Fig. 3, p. 893 (page 9): panel c's curve starts "M 166.6875", panel d's "M 320.371094".
    body = svg_page(pdf2007, 9)
    ps = paths(body)
    for panel, head, (x0, x1), (y0, y1) in (
        ("3c", "M 166.6875 380.726562", (93.699219, 197.824219), (380.066406, 445.179688)),
        ("3d", "M 320.371094 383.964844", (247.382812, 351.507812), (380.945312, 446.0625)),
    ):
        v = vertices(next(p for p in ps if p.startswith(head)))
        t = (v[:, 0] - x0) / (x1 - x0) * 1e6
        ang = (v[:, 1] - y0) / (y1 - y0) * 80.0
        o = np.argsort(t, kind="stable")
        t, ang = t[o], ang[o]
        cross = t[np.argmax(np.abs(ang) < 5.0)]
        late = ang[t > 8e5]
        print(f"  Fig. {panel}: {len(t)} vertices from t = {t[0]:.0f} to {t[-1]:.0f}; starts at {ang[0]:.2f} deg "
              f"(first five {', '.join(f'{a:.2f}' for a in ang[:5])}); first |angle| < 5 deg at t = {cross:.0f}; "
              f"mean over the last fifth {late.mean():.2f} deg")

    # Fig. 4, p. 894 (page 10): the covariance and BCM curves against y - 0.1 and 5 (y - 0.2) y.
    body = svg_page(pdf2007, 10)
    ps = paths(body)
    px = calibrate([142.683594, 176.882812, 211.085938, 245.285156, 279.484375, 313.726562], [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    py = calibrate([434.03125, 453.203125, 472.378906, 491.550781, 510.722656, 529.894531, 549.070312, 568.242188],
                   [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    for name, head, f in (
        ("simple Hebb", "M 142.683594 472.378906 L 149.5 476", lambda y: y),
        ("covariance", "M 142.683594 453.203125 L 149.5 457", lambda y: y - 0.1),
        ("BCM, times five", "M 142.683594 472.378906 L 149.5 468", lambda y: 5 * (y - 0.2) * y),
    ):
        v = vertices(next(p for p in ps if p.startswith(head)))
        y, w = np.polyval(px, v[:, 0]), np.polyval(py, v[:, 1])
        print(f"  Fig. 4 {name}: {len(y)} vertices, within {np.abs(w - f(y)).max():.2e} of the formula at mu = 0.1")

    # Fig. 7c, p. 899 (page 15): the histogram's bars, x ticks at 0.02, 0.03, 0.04, y ticks 0..100.
    body = svg_page(pdf2007, 15)
    px = calibrate([105.285156, 127.882812, 150.457031], [0.02, 0.03, 0.04])
    py = calibrate([456.136719, 468.585938, 481.0625, 493.515625, 505.988281, 518.457031], [0, 20, 40, 60, 80, 100])
    bars = [p for p in paths(body) if re.match(r"M [\d.]+ 456\.125 L [\d.]+ 456\.125 L [\d.]+ [\d.]+ L [\d.]+ [\d.]+ Z", p)]
    for d in bars:
        v = vertices(d)
        if v[1, 0] - v[0, 0] > 10:
            continue  # the panel's frame, not a bar
        lo, hi, top = np.polyval(px, v[0, 0]), np.polyval(px, v[1, 0]), np.polyval(py, v[2, 1])
        print(f"  Fig. 7c bar: weights {lo:.5f} to {hi:.5f}, frequency {top:.2f}")

    # Fig. 8, p. 900 (page 16): the dots of the right panel, frame x 290.51 to 370.12, y 78.85 to 138.39.
    body = svg_page(pdf2007, 16)
    dots = []
    for d in paths(body):
        if " C " in d:
            c = [float(t) for t in re.findall(r"[-\d.]+", d)]
            xs, ys = c[0::2], c[1::2]
            dots.append(((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2))
    dots = np.array(dots)
    for name, (x0, x1) in (("left", (76.171875, 155.777344)), ("centre", (183.339844, 262.945312)), ("right", (290.511719, 370.117188))):
        sel = (dots[:, 0] >= x0 - 1) & (dots[:, 0] <= x1 + 1) & (dots[:, 1] >= 77.8) & (dots[:, 1] <= 139.4)
        act = (138.390625 - dots[sel, 1]) / (138.390625 - 78.851562)
        print(f"  Fig. 8 {name}: {sel.sum()} dots, activity {act.min():.4f} to {act.max():.4f}, median {np.median(act):.4f}")

    # Fig. 2a (2007, p. 889, page 5): "We plot the response only to every 10th input" of 5 x 10^4.
    body = svg_page(pdf2007, 5)
    longest = max(paths(body), key=lambda d: len(re.findall(r"[ML] ", d)))
    print(f"  2007 Fig. 2a: the rate path has {len(re.findall(r'[ML] ', longest))} vertices in "
          f"{longest.count('M ')} pieces; every 10th of 5 x 10^4 inputs is 5000")

    # Fig. 1c (2007, p. 888, page 4), the dotted "optimal transfer fct.", against the formula the
    # module doc found 2005's Fig. 1b to miss.
    body = svg_page(pdf2007, 4)
    ps = paths(body)
    px = calibrate([75.566406, 95.675781, 115.785156, 135.894531, 156.0], [-4, -2, 0, 2, 4])
    py = calibrate([388.375, 400.628906, 412.863281, 425.101562, 437.355469, 449.59375], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cands = [vertices(p) for p in ps if len(re.findall(r"[ML] ", p)) >= 500 and 75 < vertices(p)[:, 0].min() < 76]
    for v in cands:
        x, y = np.polyval(px, v[:, 0]), np.polyval(py, v[:, 1])
        formula, shifted = -0.1 * log_ndtr(-x), -0.1 * log_ndtr(-(x - 0.1))
        keep = (x >= -1) & (y < 1)
        if np.abs(shifted - y)[x <= 2.5].max() < 0.01:
            print(f"  2007 Fig. 1c dotted curve: {len(x)} vertices; the formula above it by at least "
                  f"{(formula - y)[keep].min():.5f} from x = -1; shifted right by 0.1, within "
                  f"{np.abs(shifted - y)[x <= 2.5].max():.2e} up to x = 2.5")


if __name__ == "__main__":
    quadratures()
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        figures(sys.argv[1])
