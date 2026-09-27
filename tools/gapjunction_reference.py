#!/usr/bin/env python3
"""Reference numbers for `src/gapjunction.rs`, computed independently of it.

    target/venv/bin/python tools/gapjunction_reference.py      # numpy + scipy; poppler on PATH

This review did not locate code from either paper's authors: ModelDB lists no model for either,
Lewis's publication directory (math.ucdavis.edu/~tjlewis/pubs/) holds PDFs only, and the one program
Chow and Kopell name, XPPAUT, ran their conductance-based models, not the integrate-and-fire model
this module implements. So the reference is this script, written from the papers' equations and NOT
from the Rust: it shares no code or closed form with the module except where both transcribe the
page.

  A. Lewis and Rinzel's eq. (5) integrated by SciPy's DOP853 at rtol = atol = 1e-13, the alpha
     synapse carried as the linear pair x' = -a x, s' = -a s + a^2 x, from the printed initial
     conditions of Figs. 1 and 2.
  B. Their H(psi) of eq. (7) by adaptive quadrature of Z(t) P(t, t + psi T), the delta kick added
     as Z at the partner's spike.
  C. Their section 4 matching conditions at phi = 1/2, both as printed and as derived here, with
     every integral by quadrature; and the synchronous period with inhibition.
  D. Chow and Kopell's eqs. (2.7)-(2.8) and (4.1) integrated by DOP853, the spike current switched
     on at each crossing and the reset -(1 + v_M) applied at t0 + Delta, v_M = eta_+(Delta).
  E. The figures, read from the PDFs: Lewis and Rinzel's are 144 dpi JPEGs (pdfimages), read pixel
     by pixel against their own tick marks; Chow and Kopell's are vector paths (pdftocairo -svg),
     read to the coordinate — Fig. 1's spike peaks and spacings (p. 1647), Fig. 2(b)'s extremes, and
     Fig. 3's curves, fitted by least squares to G(phi, T) of eq. (3.4) summed term by term here
     from the kernels of eq. (2.14), at the printed g = 0.5 and across g from 0.02 to 6.

Every number the Rust tests embed from here is printed below with the label the test cites.
"""
import os
import re
import subprocess
import tempfile

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
PAPERS = os.path.join(os.path.dirname(HERE), "papers")
if not os.path.isdir(PAPERS):  # a working copy under target/wt/<module>/ reads the parent's papers
    PAPERS = os.path.join(HERE, "..", "..", "..", "..", "papers")
LEWIS = os.path.join(PAPERS, "lewis-2003.pdf")
CHOW = os.path.join(PAPERS, "chow-2000.pdf")


# ---------------------------------------------------------------- A. Lewis and Rinzel, eq. (5)
def lr_simulate(I, gc, beta, gs, a, v0, t_end):
    """Spikes of eq. (5): a kick g_c*beta to the partner at each spike, the partner fired at once if
    that carries it to threshold, and no kick back to a cell that has just fired."""
    y = np.array([v0[0], v0[1], 0.0, 0.0, 0.0, 0.0])  # v1, v2, x1, s1, x2, s2; s_j flows INTO j
    t, spikes = 0.0, []

    def f(_t, y):
        v1, v2, x1, s1, x2, s2 = y
        return [-v1 + I - gs * s1 + gc * (v2 - v1), -v2 + I - gs * s2 + gc * (v1 - v2),
                -a * x1, -a * s1 + a * a * x1, -a * x2, -a * s2 + a * a * x2]

    def crossing(j):
        e = lambda _t, y: y[j] - 1.0
        e.terminal, e.direction = True, 1
        return e

    while t < t_end:
        sol = solve_ivp(f, (t, t_end), y, method="DOP853", rtol=1e-13, atol=1e-13,
                        events=[crossing(0), crossing(1)], max_step=0.05)
        y, t = sol.y[:, -1].copy(), sol.t[-1]
        if sol.status != 1:
            break
        j = 0 if len(sol.t_events[0]) else 1
        k = 1 - j
        y[j] = 0.0
        spikes.append((t, j))
        y[k] += gc * beta
        y[2 + 2 * k] += 1.0
        if y[k] >= 1.0:
            y[k] = 0.0
            spikes.append((t, k))
            y[2 + 2 * j] += 1.0
    return spikes


def section_a():
    print("== A. Lewis-Rinzel eq. (5), DOP853 rtol 1e-13, from the printed initial conditions")
    runs = {"FIG1_LOW": (1.1, 0.0, 0.0, 0.2, 3.0, (0.4, 0.0)), "FIG1_HIGH": (1.6, 0.0, 0.0, 0.2, 3.0, (0.4, 0.0)),
            "FIG2_LOW": (1.1, 0.2, 0.2, 0.0, 3.0, (0.59, 0.0)), "FIG2_HIGH": (1.6, 0.2, 0.2, 0.0, 3.0, (0.59, 0.0))}
    for name, (I, gc, beta, gs, a, v0) in runs.items():
        sp = lr_simulate(I, gc, beta, gs, a, v0, 60.0)
        print(f"{name} first eight spikes (time, cell):")
        print("   " + ", ".join(f"({t:.12f}, {c})" for t, c in sp[:8]))
        t1 = [t for t, c in sp if c == 0]
        t2 = [t for t, c in sp if c == 1]
        isi = t1[-1] - t1[-2]
        lag = min(u for u in t2 if u >= t1[-2]) - t1[-2]
        print(f"   by t = 60: period {isi:.12f}, cell 2 lags by {lag / isi:.9f} of it")
    # Both couplings at I = 1.6: the kick captures the partner from t = 24 on, and from then the
    # captured cell's inhibition of the other matters.
    sp = lr_simulate(1.6, 0.2, 0.2, 0.2, 3.0, (0.59, 0.0), 30.0)
    late = [(t, c) for t, c in sp if t > 24.0]
    print("COMBINED (I 1.6, g_c 0.2, beta 0.2, g_s 0.2, alpha 3) spikes after t = 24:")
    print("   " + ", ".join(f"({t:.12f}, {c})" for t, c in late[:8]))


# ---------------------------------------------------------------- B. H(psi) by quadrature
def T_of(I):
    return np.log(I / (I - 1.0))


def s_T(t, a, T):
    q = np.exp(-a * T)
    t = np.mod(t, T)
    return a * a / (1 - q) ** 2 * np.exp(-a * t) * (t * (1 - q) + T * q)


def H_quad(psi, I, gc, beta, gs, a):
    T = T_of(I)
    p = psi % 1.0
    Z = lambda t: np.exp(np.mod(t, T)) / (I * T)
    v = lambda t: I * (1 - np.exp(-np.mod(t, T)))
    f = lambda t: Z(t) * (-gs * s_T(t + p * T, a, T) + gc * (v(t + p * T) - v(t)))
    pts = [(1 - p) * T] if 0 < p < 1 else None
    val = quad(f, 0, T, points=pts, epsabs=1e-15, epsrel=1e-14, limit=400)[0] / T
    if p > 0:
        val += gc * beta * Z((1 - p) * T) / T
    return val


def section_b():
    print("== B. H(psi) = (1/T) int_0^T Z(t) P(t, t + psi T) dt by scipy.integrate.quad")
    for name, prm in {"(1.2, g_c 0.3, beta 0.2, g_s 0.7, alpha 4)": (1.2, 0.3, 0.2, 0.7, 4.0),
                      "(1.6, g_c 1, beta 0.1, g_s -0.4, alpha 0.5)": (1.6, 1.0, 0.1, -0.4, 0.5),
                      "(1.05, g_c 0, beta 0, g_s 1, alpha 1)": (1.05, 0.0, 0.0, 1.0, 1.0)}.items():
        print(name, ", ".join(f"H({psi}) = {H_quad(psi, *prm):.15e}" for psi in (0.0, 0.125, 0.5, 0.875)))


# ---------------------------------------------------------------- C. section 4, beyond weak coupling
def integral(c, lo, hi, a, T):
    return quad(lambda t: np.exp(c * t) * s_T(t, a, T), lo, hi, epsabs=1e-15, epsrel=1e-14, limit=200)[0]


def antiphase_residual(T, I, gc, beta, gs, a, printed):
    """Section 4's four equations at phi = 1/2, u1 = u2 = u: the v- equation gives u, the v+ one
    is returned. `printed` transcribes p. 299 as set; otherwise the derivation in the module doc."""
    phi, mu, k, h = 0.5, 1 + 2 * gc, gc * beta, T / 2
    Sp = np.exp(-T) * integral(1, (1 - phi) * T, T, a, T) + np.exp(-phi * T) * integral(1, 0, phi * T, a, T)
    Sm = np.exp(-mu * T) * integral(mu, (1 - phi) * T, T, a, T) - np.exp(-mu * phi * T) * integral(mu, 0, phi * T, a, T)
    if printed:  # u - b gc - 1 = -u e^{-mu phi T} + gs[..];  u - b gc + 1 = u e^{-phi T} + 2I(1 - e^{-phi T}) + gs[..]
        u = (1 + k + gs * Sm) / (1 + np.exp(-mu * h))
        return u - k + 1 - u * np.exp(-h) - 2 * I * (1 - np.exp(-h)) - gs * Sp, u
    u = (1 - k * np.exp(-mu * h) - gs * Sm) / (1 + np.exp(-mu * h))
    return u + 1 - (u + k) * np.exp(-h) - 2 * I * (1 - np.exp(-h)) + gs * Sp, u


def section_c():
    print("== C. Lewis-Rinzel section 4 at phi = 1/2 (roots on (0.3, 8]) and the synchronous period")
    for name, prm in {"FIG1_LOW": (1.1, 0.0, 0.0, 0.2, 3.0), "FIG2_LOW": (1.1, 0.2, 0.2, 0.0, 3.0)}.items():
        for printed in (False, True):
            Ts = np.linspace(0.3, 8.0, 155)
            r = [antiphase_residual(T, *prm, printed)[0] for T in Ts]
            rts = [brentq(lambda T: antiphase_residual(T, *prm, printed)[0], Ts[i], Ts[i + 1], xtol=1e-15)
                   for i in range(len(Ts) - 1) if np.sign(r[i]) != np.sign(r[i + 1])]
            print(f"{name} {'as printed' if printed else 'derived'}: " +
                  ", ".join(f"T = {T:.12f}, u = {antiphase_residual(T, *prm, printed)[1]:.12f}" for T in rts))
    I, gs, a = 1.6, 0.2, 3.0
    res = lambda T: I * (1 - np.exp(-T)) - gs * np.exp(-T) * integral(1, 0, T, a, T) - 1
    print(f"FIG1_HIGH synchronous period: {brentq(res, 0.5, 3.0, xtol=1e-15):.12f}")


# ---------------------------------------------------------------- D. Chow and Kopell, eq. (4.1)
def ck_simulate(vA, xi, D, g, drives, v0, t_end):
    n = len(drives)
    vM = vA * (np.exp(xi * D) - np.exp(-D)) / (1 + xi)
    v, t = np.array(v0, float), 0.0
    onset = [None] * n
    spikes = []

    def f(tt, v):
        A = np.array([vA * np.exp(xi * (tt - onset[i])) if onset[i] is not None else 0.0 for i in range(n)])
        return np.array(drives) - v - g * (n * v - v.sum()) + A

    while t < t_end:
        for i in range(n):
            if onset[i] is None and v[i] >= 1.0:
                onset[i] = t
                spikes.append((t, i))
        ends = [onset[i] + D for i in range(n) if onset[i] is not None]
        stop = min(ends + [t_end])
        idle = [i for i in range(n) if onset[i] is None]
        evs = []
        for i in idle:
            e = (lambda i: (lambda _t, v: v[i] - 1.0))(i)
            e.terminal, e.direction = True, 1
            evs.append(e)
        sol = solve_ivp(f, (t, stop), v, method="DOP853", rtol=1e-13, atol=1e-13, events=evs or None, max_step=0.01)
        v, t = sol.y[:, -1].copy(), sol.t[-1]
        if sol.status == 1:  # the crossing cell starts its spike here, wherever the root landed
            i = idle[[len(te) > 0 for te in sol.t_events].index(True)]
            onset[i] = t
            spikes.append((t, i))
            continue
        t = stop
        for i in range(n):
            if onset[i] is not None and onset[i] + D == stop:
                v[i] -= 1 + vM
                onset[i] = None
    return spikes


def section_d():
    print("== D. Chow-Kopell eqs. (2.7)-(2.8) and (4.1), DOP853 rtol 1e-13")
    for name, args in {"pair FIG3 spike, I = (1.3, 1.3), v0 = (0.3, 0)": (1.0, 50.0, 0.1, 0.5, [1.3, 1.3], [0.3, 0.0]),
                       "three cells, g = 0.2, I = (1.2, 1.25, 1.3), v0 = (0.5, 0.25, 0)": (1.0, 50.0, 0.1, 0.2, [1.2, 1.25, 1.3], [0.5, 0.25, 0.0])}.items():
        sp = ck_simulate(*args, t_end=6.0)
        print(name + ":")
        print("   " + ", ".join(f"({t:.12f}, {c})" for t, c in sp[:8]))


# ---------------------------------------------------------------- E. the figures
def raster(pdf, index):
    """The index-th image pdfimages extracts, as a grey array (0 black, 255 white)."""
    d = tempfile.mkdtemp()
    subprocess.run(["pdfimages", pdf, os.path.join(d, "p")], check=True)
    name = sorted(x for x in os.listdir(d) if x.startswith(f"p-{index:03d}"))[0]
    data = open(os.path.join(d, name), "rb").read()
    fields, i = [], 0
    while len(fields) < 4:
        while data[i:i + 1].isspace():
            i += 1
        j = i
        while not data[j:j + 1].isspace():
            j += 1
        fields.append(data[i:j])
        i = j
    w, h = int(fields[1]), int(fields[2])
    ch = 3 if fields[0] == b"P6" else 1
    return np.frombuffer(data[i + 1:i + 1 + w * h * ch], dtype=np.uint8).reshape(h, w, ch).mean(axis=2)


def arrows(img, top, bottom):
    """Columns of the vertical arrow shafts drawn between two rows: runs of 8 or more dark pixels."""
    cols = [c for c in range(img.shape[1]) if (img[top:bottom, c] < 120).sum() >= 8]
    out = []
    for c in cols:
        if out and c - out[-1][-1] <= 1:
            out[-1].append(c)
        else:
            out.append([c])
    return [float(np.mean(g)) for g in out]


def dark_groups(img, x, lo, hi):
    """Runs of dark (below 110: black ink, not the grey curves) pixels in column x, rows lo..hi."""
    out = []
    for y in range(lo, hi):
        if img[y, x] < 110:
            if out and y == out[-1][1] + 1:
                out[-1][1] = y
            else:
                out.append([y, y])
    return out


def lowest_curve(img, x, row_half, row_zero):
    """The centre row of the dark run nearest the phi = 0 line, above it and below phi = 1/2: the
    lower branch of a bifurcation diagram, where it is drawn (the dashes leave gaps)."""
    groups = dark_groups(img, x, int(row_half) + 5, int(row_zero) - 4)
    return 0.5 * (groups[-1][0] + groups[-1][1]) if groups else None


def shading_edge(img, x, top, bottom, upward):
    """The first row, scanning column x up from `bottom` or down from `top`, where the grey shading
    (235) meets white (255), the dashed curve between. The figures' labels sit on white boxes, so
    each figure is sampled where the scan meets the curve before any label."""
    col = img[top:bottom, x]
    grey = lambda v: abs(v - 235) < 6
    white = lambda v: v > 250
    ys = range(len(col) - 6, 4, -1) if upward else range(5, len(col) - 5)
    for y in ys:
        if (grey(col[y - 5]) and white(col[y + 5])) or (white(col[y - 5]) and grey(col[y + 5])):
            run = [k for k in range(y - 5, y + 6) if col[k] < 200]
            return top + (float(np.mean(run)) if run else float(y))
    return None


def dash_near_edge(img, x, top, bottom, upward):
    """The dashed curve near column x: the shading edge only says where to look, since the grey
    fill runs a few pixels past the line in the dashes' gaps. Returns (column, row) of the centre
    of the nearest drawn dash within four columns and eight rows of the edge, or None."""
    for c in sorted(range(x - 4, x + 5), key=lambda k: abs(k - x)):
        e = shading_edge(img, c, top, bottom, upward)
        if e is None:
            continue
        near = [g for g in dark_groups(img, c, top, bottom) if abs(0.5 * (g[0] + g[1]) - e) <= 8]
        if near:
            g = min(near, key=lambda g: abs(0.5 * (g[0] + g[1]) - e))
            return c, 0.5 * (g[0] + g[1])
    return None


def stroked_paths(svg):
    out = []
    for m in re.finditer(r"<path ([^>]*)/>", open(svg).read()):
        attrs = m.group(1)
        if "stroke=" not in attrs or 'fill="none"' not in attrs:
            continue
        d = re.search(r' d="([^"]*)"', attrs).group(1)
        subs = [[(float(x), float(y)) for x, y in re.findall(r"[ML] ([-\d.]+) ([-\d.]+)", seg)]
                for seg in re.split(r"(?=M )", d.strip())]
        out.append([s for s in subs if s])
    return out


def section_e():
    print("== E. figures read from the PDFs")
    # Lewis-Rinzel Figs. 4, 7 and 9 share one frame: I = 1 at the left edge, 2 at the right, and
    # the phi = 1, 1/2, 0 lines drawn solid; their tick marks were read off row ~496.
    for fig, idx, x0, x1, rows, grid in (
        ("4", 4, 55, 565, (129, 297, 465), (1.05, 1.1, 1.15, 1.25, 1.3, 1.35, 1.4, 1.45)),
        ("7", 7, 55, 565, (122.5, 290.5, 458), (1.02, 1.05, 1.1, 1.15, 1.2, 1.225, 1.25)),
        ("9", 9, 49, 565, (129, 298.5, 468), (1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5)),
    ):
        img = raster(LEWIS, idx)
        I_of = lambda x: 1 + (x - x0) / (x1 - x0)
        print(f"L&R Fig. {fig} arrows at I = " + ", ".join(f"{I_of(c):.4f}" for c in arrows(img, int(rows[2]) + 8, int(rows[2]) + 40)))
        pts = []
        for I in grid:  # the nearest column within four pixels where a dash is drawn
            x0_ = int(round(x0 + (I - 1) * (x1 - x0)))
            for x in sorted(range(x0_ - 4, x0_ + 5), key=lambda c: abs(c - x0_)):
                y = lowest_curve(img, x, rows[1], rows[2])
                if y is not None:
                    pts.append(f"({I_of(x):.4f}, {(rows[2] - y) / (rows[2] - rows[0]):.3f})")
                    break
        print(f"L&R Fig. {fig} lower dashed branch (I, phi): " + ", ".join(pts))
    # Fig. 5: I from 1 (x 33) to 1.8 (x 321), alpha from 0 (y 331) to 6 (y 104); grey above.
    img = raster(LEWIS, 5)
    pts = []
    for I in (1.3, 1.4, 1.5, 1.7, 1.75):
        x, y = dash_near_edge(img, int(round(33 + (I - 1) / 0.8 * 288)), 108, 329, upward=True)
        pts.append(f"({1 + (x - 33) / 360:.4f}, {(331 - y) / (331 - 104) * 6:.3f})")
    print("L&R Fig. 5 dashed curve (I, alpha): " + ", ".join(pts))
    # Fig. 8: I from 1 (x 55) to 2 (x 344), beta from 0 (y 332) to 0.5 (y 103); grey below.
    img = raster(LEWIS, 8)
    pts = []
    for I in (1.3, 1.4, 1.5, 1.6, 1.8):
        x, y = dash_near_edge(img, int(round(55 + (I - 1) * 289)), 107, 330, upward=True)
        pts.append(f"({1 + (x - 55) / 289:.4f}, {(332 - y) / (332 - 103) * 0.5:.4f})")
    print("L&R Fig. 8 dashed curve (I, beta): " + ", ".join(pts))
    # Fig. 11: alpha from 0 (x 45) to 4 (x 332); rho = 1 at each panel's top row, 0 at its bottom.
    img = raster(LEWIS, 11)
    for I, (top, bot), alphas, up in zip((1.2, 1.25, 1.3), ((90, 237), (284, 431), (478, 625)),
                                         ((0.5, 1.0, 1.4, 1.8, 2.1), (0.5, 1.0, 1.5, 2.0, 2.4), (3.2, 3.5, 3.8)), (True, True, False)):
        pts = []
        for a in alphas:
            found = dash_near_edge(img, int(round(45 + a / 4 * 287)), top + 2, bot - 2, upward=up)
            if found is not None:
                x, y = found
                pts.append(f"({(x - 45) / 287 * 4:.3f}, {(bot - y) / (bot - top):.3f})")
        print(f"L&R Fig. 11 I = {I} (alpha, rho) on the dashed curve: " + ", ".join(pts))
    # Chow-Kopell Fig. 2(b) and Fig. 3: vector paths.
    d = tempfile.mkdtemp()
    for page in (5, 7, 14):
        subprocess.run(["pdftocairo", "-svg", "-f", str(page), "-l", str(page), CHOW, os.path.join(d, f"p{page}.svg")], check=True)
    ps = stroked_paths(os.path.join(d, "p7.svg"))
    for k, g in ((14, 0.5), (15, 5.0)):  # gamma_c: t = 0 at x 205.695, 1 per 27.766; 0 at y 472.469, 0.2 per 42.035
        pts = np.array([p for s in ps[k] for p in s])
        i = int(np.argmin(pts[:, 1]))
        print(f"C&K Fig. 2(b) g = {g}: drawn minimum of gamma_c at t = {(pts[i, 0] - 205.695) / 27.766:.3f}, value {(pts[i, 1] - 472.469) / 42.035 * 0.2:.4f}; drawn maximum {(pts[:, 1].max() - 472.469) / 42.035 * 0.2:.4f}")
    ps = stroked_paths(os.path.join(d, "p14.svg"))
    panels = {"(a) T = 2": ((4, 5), 168.3125, 626.695312, 0.025), "(b) T = 1": ((10, 11), 312.390625, 626.695312, 0.025),
              "(c) T = 0.25": ((1, 2), 168.3125, 513.753906, 0.01), "(d) T = 0.2": ((7, 8), 312.390625, 513.753906, 0.01)}
    for name, (idx, x0, y0, top) in panels.items():
        pts = np.array([p for i in idx for s in ps[i] if len(s) > 2 for p in s])
        phi = (pts[:, 0] - x0) / (244.617188 - 168.3125)
        G = (pts[:, 1] - y0) / 42.351562 * top
        i, j = int(np.argmax(G)), int(np.argmin(G))
        print(f"C&K Fig. 3{name}: drawn max {G[i]:.6f} at phi {phi[i]:.4f}, drawn min {G[j]:.6f} at phi {phi[j]:.4f}; {int((abs(G) >= 0.999 * top).sum())} points clipped at the axis")
    fig3_scales(ps, panels)
    fig1_traces(os.path.join(d, "p5.svg"))


def ck_gamma_c(t, vA, xi, D, g):
    """gamma_c = (eta_+ - eta_-)/2, r = 1 + 2g, with eta_r of eq. (2.14): the spike current
    v_A e^{xi t} on (0, D] and the reset -(1 + v_M) at D, v_M = eta_+(D) (p. 1650). Zero for t <= 0."""
    t = np.asarray(t, float)
    vM = vA * (np.exp(xi * D) - np.exp(-D)) / (1 + xi)

    def eta(r):
        s = np.clip(t, 0.0, D)
        spike = vA * (np.exp(xi * s) - np.exp(-r * s)) / (r + xi)
        at_D = vA * (np.exp(xi * D) - np.exp(-r * D)) / (r + xi)
        tail = (at_D - 1 - vM) * np.exp(-r * np.maximum(t - D, 0.0))
        return np.where(t <= 0, 0.0, np.where(t <= D, spike, tail))

    return 0.5 * (eta(1.0) - eta(1 + 2 * g))


def ck_G(phi, T, g, vA=1.0, xi=50.0, D=0.1, L=400):
    """G(phi, T) of eq. (3.4), gamma_c(phi T) + sum_{l=1}^{L-1} [gamma_c(lT + phi T) - gamma_c(lT - phi T)],
    term by term: at T = 0.2 the last term is below e^{-79}."""
    c = np.asarray(phi, float) * T
    G = ck_gamma_c(c, vA, xi, D, g)
    for l in range(1, L):
        G = G + ck_gamma_c(l * T + c, vA, xi, D, g) - ck_gamma_c(l * T - c, vA, xi, D, g)
    return G


def fig3_scales(ps, panels):
    """The least-squares scale s minimising sum (drawn - s G)^2 over every drawn vertex of each panel
    of Fig. 3, the vertices clipped at the axis left out; and the vertices nearest phi = k/20, which
    the Rust test embeds and fits against the module's own G."""
    periods = {"(a) T = 2": 2.0, "(b) T = 1": 1.0, "(c) T = 0.25": 0.25, "(d) T = 0.2": 0.2}
    data = {}
    for name, (idx, x0, y0, top) in panels.items():
        pts = np.array([p for i in idx for s in ps[i] if len(s) > 2 for p in s])
        phi = (pts[:, 0] - x0) / (244.617188 - 168.3125)
        G = (pts[:, 1] - y0) / 42.351562 * top
        keep = np.abs(G) < 0.999 * top
        data[name] = (phi[keep], G[keep], periods[name])
        thin = []
        for k in range(1, 20):
            i = int(np.argmin(np.abs(phi[keep] - k / 20)))
            thin.append(f"({phi[keep][i]:.6f}, {G[keep][i]:.6f})")
        print(f"C&K Fig. 3{name}: {int(keep.sum())} vertices fitted; nearest phi = k/20: " + ", ".join(thin))
    scale = lambda g: [float((y @ ck_G(p, T, g)) / (ck_G(p, T, g) @ ck_G(p, T, g))) for p, y, T in data.values()]
    for g in (0.5, 0.25):
        print(f"C&K Fig. 3 least-squares scale of the drawing against G at g = {g}: " + ", ".join(f"{s:.4f}" for s in scale(g)))
    grid = np.round(np.concatenate([np.arange(0.02, 1.0001, 0.02), np.arange(1.2, 6.0001, 0.2)]), 2)
    spread = [(g, max(sc) / min(sc)) for g in grid for sc in [scale(g)]]
    best = min(spread, key=lambda x: x[1])
    rest = min(x[1] for x in spread if x[0] != best[0])
    print(f"C&K Fig. 3 per-panel scales agree best at g = {best[0]} (max/min {best[1]:.4f}); at every other g of "
          f"{len(grid)} from 0.02 to 6 they are at least {rest:.4f} apart; "
          + ", ".join(f"g {g}: {r:.4f}" for g, r in spread if g in (0.48, 0.52, 1.0, 5.0)))


def fig1_traces(svg):
    """Chow-Kopell Fig. 1 (p. 1647): each panel's trace, from its single-subpath strokes of more
    than 100 vertices; each drawn spike's highest vertex, and the spacing of those."""
    # Path coordinates run upward: t = 0..5 across x0..x1, v = 0 at y0 and the top tick at y1.
    frames = {"(a)": (176.789062, 259.890625, 592.484375, 675.410156, 4.0),
              "(b)": (309.753906, 392.855469, 592.484375, 675.410156, 1.5),
              "(c)": (176.789062, 259.890625, 468.097656, 551.023438, 40.0),
              "(d)": (309.753906, 392.855469, 468.097656, 551.023438, 8.0)}
    curves = [np.array(p[0]) for p in stroked_paths(svg) if len(p) == 1 and len(p[0]) > 100]
    for name, (x0, x1, y0, y1, top) in frames.items():
        mine = [c for c in curves if c[:, 0].min() >= x0 - 0.5 and c[:, 0].max() <= x1 + 0.5
                and c[:, 1].min() >= y0 - 0.5 and c[:, 1].max() <= y1 + 0.5]
        pts = np.concatenate(mine)
        t = (pts[:, 0] - x0) / (x1 - x0) * 5
        v = (pts[:, 1] - y0) / (y1 - y0) * top
        order = np.argsort(t, kind="stable")
        peaks = []  # vertices above 60% of the highest, grouped where they lie within 0.3 in t
        for k in order:
            if v[k] < 0.6 * v.max():
                continue
            if peaks and t[k] - peaks[-1][2] < 0.3:
                peaks[-1][2] = t[k]
                if v[k] > peaks[-1][1]:
                    peaks[-1][:2] = [t[k], v[k]]
            else:
                peaks.append([t[k], v[k], t[k]])
        print(f"C&K Fig. 1{name}: {len(mine)} strokes; spike peaks drawn at " + ", ".join(f"({p[0]:.4f}, {p[1]:.4f})" for p in peaks)
              + f"; highest {max(p[1] for p in peaks):.4f}; peak-to-peak spacings " + ", ".join(f"{b[0] - a[0]:.4f}" for a, b in zip(peaks, peaks[1:])))


if __name__ == "__main__":
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
