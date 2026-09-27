#!/usr/bin/env python3
"""Regenerate the reference tables in `src/calcium.rs` from Graupner and Brunel's own code.

    python3 tools/calcium_reference.py WORKDIR > reference_block.rs

What it does, so every number in the tests can be traced to its source:

1. Fetches the authors' repository files at commit 56503b0f401536bc0debe996521a5adbd80c6a71
   (github.com/mgraupe/CalciumBasedPlasticityModel, HEAD on 2026-01-16) into WORKDIR, unless a
   checkout is already there: `synapticChange.py`, `parameter_fit_solutions.py` and
   `timeAboveThreshold/timeAboveThreshold.py`. They are run UNMODIFIED; the repository is GPL-3.0,
   so this script fetches them rather than the crate carrying a copy.
2. Prints every parameter set `synapticChange.choseParameterSet` defines, as the code holds it.
3. Recomputes Fig. 2A exactly as `Graupner2012PNAS/Graupner2012PNAS_Fig2.py` does — the six
   Table S1 sets, 60 pairs at 1 Hz, the 2001-point grid of `linspace(-0.1, 0.1, 2001)` (the
   script widens its +-100 ms range only when half the interval is SHORTER than it) — and
   prints sampled points and a digest of the whole grid (count of NaN, `math.fsum` of the finite
   values and of their squares, the extremes and where they are).
4. Recomputes Fig. 3 (`Graupner2012PNAS_Fig3.py`: 'hippocampal slices', 5 Hz, a pair 200 times and
   a pre-spike and post-burst 100 and 30 times, burst interval 11.5 ms) and Fig. 4B
   (`Graupner2012PNAS_Fig4B.py`: 'cortical slices', 75 pairs at 1, 20, 30, 40 and 50 Hz) the same
   way.
5. Prints `spikePairFrequency` and `preSpikePostPair` fractions of time above threshold at chosen
   points, `changeInSynapticStrength`'s intermediate values, `eventBasedIntegration` on a
   deterministic irregular train, and the three points that show `spikePairFrequency`'s wrap of an
   offset longer than one period.
6. Bisects, with the same functions, the frequency above which every offset on the Fig. 4B grid
   potentiates (the caption of Fig. 4B says 29 Hz).
7. Copies the authors' committed simulation output, `numericalSimulation/output/<case>_curve/
   final_camkII_state.dat` (1000 runs of their C++ per point): the transition probabilities of the
   DP, P and D' curves, and the change in strength of the DPD' curve; and the parameters their C++
   read for each curve, from `camkmotifscan.par` beside it.

Needs numpy and scipy (the authors' modules import both). Deterministic: no random numbers.
"""
import io
import math
import os
import sys
import urllib.request
import warnings

COMMIT = "56503b0f401536bc0debe996521a5adbd80c6a71"
RAW = f"https://raw.githubusercontent.com/mgraupe/CalciumBasedPlasticityModel/{COMMIT}/"
FILES = [
    "synapticChange.py",
    "parameter_fit_solutions.py",
    "timeAboveThreshold/__init__.py",
    "timeAboveThreshold/timeAboveThreshold.py",
    "Graupner2012PNAS/numericalSimulation/output/DP_curve/final_camkII_state.dat",
    "Graupner2012PNAS/numericalSimulation/output/P_curve/final_camkII_state.dat",
    "Graupner2012PNAS/numericalSimulation/output/Dprime_curve/final_camkII_state.dat",
    "Graupner2012PNAS/numericalSimulation/output/DPDprime_curve/final_camkII_state.dat",
] + [f"Graupner2012PNAS/numericalSimulation/output/{c}_curve/camkmotifscan.par" for c in ["DP", "DPD", "DPDprime", "P", "D", "Dprime"]]
S1 = ["DP", "DPD", "DPDprime", "P", "D", "Dprime"]


def fetch(work):
    for f in FILES:
        path = os.path.join(work, f)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with urllib.request.urlopen(RAW + f, timeout=120) as r:
                io.open(path, "wb").write(r.read())


def grid(start, end, steps=2000):
    import numpy as np

    return np.linspace(start, end, steps + 1)


def digest(values):
    finite = [v for v in values if math.isfinite(v)]
    nan = len(values) - len(finite)
    imax = max(range(len(values)), key=lambda i: values[i] if math.isfinite(values[i]) else -math.inf)
    imin = min(range(len(values)), key=lambda i: values[i] if math.isfinite(values[i]) else math.inf)
    return nan, math.fsum(finite), math.fsum(v * v for v in finite), values[imax], imax, values[imin], imin


def curve(sc, tat, dts, freq, n, pair=True, burst=None):
    out = []
    for dt in dts:
        if pair:
            ad, ap = tat.spikePairFrequency(dt - sc.D, freq)
        else:
            ad, ap = tat.preSpikePostPair(dt - sc.D, freq, burst)
        sc.changeInSynapticStrength(n / freq, 0.5, ad, ap)
        out.append(float(sc.synChange))
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: calcium_reference.py WORKDIR")
    work = os.path.abspath(sys.argv[1])
    fetch(work)
    sys.path.insert(0, work)
    warnings.simplefilter("ignore")
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        from synapticChange import synapticChange
        from timeAboveThreshold.timeAboveThreshold import timeAboveThreshold

    def make(name):
        with contextlib.redirect_stdout(io.StringIO()):
            sc = synapticChange("genericCase", name)
        return sc, timeAboveThreshold(sc.tauCa, sc.Cpre, sc.Cpost, sc.thetaD, sc.thetaP)

    def r(x):
        x = float(x)
        return "f64::NAN" if math.isnan(x) else repr(x)

    print(f"// Generated by tools/calcium_reference.py from mgraupe/CalciumBasedPlasticityModel@{COMMIT[:7]}.")
    print("// synapticChange.choseParameterSet, in the order DP, DPD, DPDprime, P, D, Dprime, 'hippocampal slices',")
    print("// 'hippocampal cultures', 'cortical slices': tauCa, Cpre, Cpost, thetaD, thetaP, gammaD, gammaP, sigma, tau,")
    print("// rhoStar, D, beta, b.")
    print("const PARAMS: [[f64; 13]; 9] = [")
    for name in S1 + ["hippocampal slices", "hippocampal cultures", "cortical slices"]:
        sc, _ = make(name)
        row = [sc.tauCa, sc.Cpre, sc.Cpost, sc.thetaD, sc.thetaP, sc.gammaD, sc.gammaP, sc.sigma, sc.tau, sc.rhoStar, sc.D, sc.beta, sc.b]
        print(f"    [{', '.join(r(v) for v in row)}],")
    print("];")

    # Fig. 2A.
    dts = grid(-0.1, 0.1)
    samples = sorted(set(list(range(0, 2001, 100)) + list(range(960, 1041, 8))))
    print("const FIG2: [[(usize, f64); %d]; 6] = [" % len(samples))
    digests = []
    for name in S1:
        sc, tat = make(name)
        ys = curve(sc, tat, dts, 1.0, 60)
        digests.append(digest(ys))
        print("    // " + name)
        print("    [" + ", ".join(f"({k}, {r(ys[k])})" for k in samples) + "],")
    print("];")
    print("const FIG2_DIGEST: [(usize, f64, f64, f64, usize, f64, usize); 6] = [")
    for d in digests:
        print(f"    ({d[0]}, {r(d[1])}, {r(d[2])}, {r(d[3])}, {d[4]}, {r(d[5])}, {d[6]}),")
    print("];")

    # Fractions of time above threshold, spike pairs.
    pts = [-0.1, -0.02, -0.005, 0.0, 0.005, 0.02, 0.1]
    print("const PAIR_ALPHAS: [[(f64, f64, f64); 7]; 6] = [")
    for name in S1:
        sc, tat = make(name)
        rows = []
        for dt in pts:
            ad, ap = tat.spikePairFrequency(dt - sc.D, 1.0)
            rows.append(f"({r(dt)}, {r(float(ad))}, {r(float(ap))})")
        print("    [" + ", ".join(rows) + "],")
    print("];")
    print("// spikePairFrequency at 20 Hz, 'cortical slices': (dt, alphaD, alphaP)")
    sc, tat = make("cortical slices")
    print("const PAIR_ALPHAS_20HZ: [(f64, f64, f64); 5] = [")
    for dt in [-0.02, -0.01, 0.0, 0.01, 0.02]:
        ad, ap = tat.spikePairFrequency(dt - sc.D, 20.0)
        print(f"    ({r(dt)}, {r(float(ad))}, {r(float(ap))}),")
    print("];")

    # Pre-spike and post-burst fractions, hippocampal slices at 5 Hz.
    sc, tat = make("hippocampal slices")
    print("const BURST_ALPHAS: [(f64, f64, f64); 9] = [")
    for dt in [-0.08, -0.03, -0.005, 0.0, 0.004, 0.0115, 0.015, 0.03, 0.08]:
        ad, ap = tat.preSpikePostPair(dt - sc.D, 5.0, 0.0115)
        print(f"    ({r(dt)}, {r(float(ad))}, {r(float(ap))}),")
    print("];")

    # Fig. 3.
    dts = grid(-0.1, 0.1)
    samples = list(range(0, 2001, 100))
    print("const FIG3: [[(usize, f64); %d]; 3] = [" % len(samples))
    d3 = []
    for k, n in enumerate([200, 100, 30]):
        ys = curve(sc, tat, dts, 5.0, n, pair=(k == 0), burst=0.0115)
        d3.append(digest(ys))
        print("    [" + ", ".join(f"({i}, {r(ys[i])})" for i in samples) + "],")
    print("];")
    print("const FIG3_DIGEST: [(usize, f64, f64, f64, usize, f64, usize); 3] = [")
    for d in d3:
        print(f"    ({d[0]}, {r(d[1])}, {r(d[2])}, {r(d[3])}, {d[4]}, {r(d[5])}, {d[6]}),")
    print("];")

    # Fig. 4B.
    sc, tat = make("cortical slices")
    freqs = [1.0, 20.0, 30.0, 40.0, 50.0]
    start, end = -0.1, 0.1
    print("// (frequency, the grid's half-width, samples)")
    print("type Fig4bRow = (f64, f64, [(usize, f64); 21]);")
    print("const FIG4B: [Fig4bRow; 5] = [")
    d4 = []
    for f in freqs:
        interval = 1.0 / f
        if interval / 2.0 < abs(start) or interval / 2.0 < end:
            start, end = -interval / 2.0, interval / 2.0
        dts = grid(start, end)
        ys = curve(sc, tat, dts, f, 75)
        d4.append(digest(ys))
        print(f"    ({r(f)}, {r(float(end))}, [" + ", ".join(f"({i}, {r(ys[i])})" for i in range(0, 2001, 100)) + "]),")
    print("];")
    print("const FIG4B_DIGEST: [(usize, f64, f64, f64, usize, f64, usize); 5] = [")
    for d in d4:
        print(f"    ({d[0]}, {r(d[1])}, {r(d[2])}, {r(d[3])}, {d[4]}, {r(d[5])}, {d[6]}),")
    print("];")

    # Fig. 4A: Delta t = +10 and -10 ms against frequency, 75 pairs.
    print("const FIG4A: [(f64, f64, f64); 8] = [")
    for f in [0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]:
        out = []
        for dt in [0.01, -0.01]:
            ad, ap = tat.spikePairFrequency(dt - sc.D, f)
            sc.changeInSynapticStrength(75 / f, 0.5, ad, ap)
            out.append(float(sc.synChange))
        print(f"    ({r(f)}, {r(out[0])}, {r(out[1])}),")
    print("];")

    # changeInSynapticStrength intermediates: (alphaD, alphaP, T) -> rhoBar, sigmaRho^2, tauEff, UP, DOWN, meanUP, meanDOWN, synChange
    print("// changeInSynapticStrength(T, 0.5, alphaD, alphaP) on the DP set and on 'hippocampal cultures'")
    print("const OU: [(usize, f64, f64, f64, [f64; 8]); 6] = [")
    for which, name in [(0, "DP"), (1, "hippocampal cultures")]:
        sc, _ = make(name)
        for ad, ap, T in [(0.02017213031583294, 0.0096775597371333, 60.0), (0.013, 0.021, 60.0), (0.05, 0.001, 10.0)]:
            sc.changeInSynapticStrength(T, 0.5, ad, ap)
            vals = [sc.rhoBar, sc.sigmaRhoSquared, sc.tauEff, sc.UP, sc.DOWN, sc.meanUP, sc.meanDOWN, sc.synChange]
            print(f"    ({which}, {r(ad)}, {r(ap)}, {r(T)}, [{', '.join(r(float(v)) for v in vals)}]),")
    print("];")

    # eventBasedIntegration on a deterministic irregular train (DP amplitudes, pre transients already delayed).
    sc, tat = make("DP")
    events, t = [], 0.0
    for k in range(1, 401):
        t += 0.004 + 0.2 * ((k * 0.6180339887498949) % 1.0)
        kind = k % 3 == 0
        events.append([t, 1 if kind else 0, sc.Cpost if kind else sc.Cpre])
    tl = sorted(events, key=lambda e: e[0])
    tD, tP = tat.eventBasedIntegration(tl)
    print(f"// eventBasedIntegration on 400 events, t_k = t_(k-1) + 0.004 + 0.2*frac(k*0.6180339887498949), post when k % 3 == 0")
    print(f"const EVENT_BASED: (f64, f64, f64) = ({r(tl[-1][0])}, {r(float(tD))}, {r(float(tP))});")

    # The wrap of an offset longer than one period, DP amplitudes at 5 Hz.
    sc, tat = make("DP")
    w = [tat.spikePairFrequency(x, 5.0) for x in (0.23, -0.03, 0.03)]
    print("// spikePairFrequency(x, 5.0) on the DP amplitudes at x = 0.23, -0.03, +0.03")
    print("const WRAP: [(f64, f64); 3] = [" + ", ".join(f"({r(float(a))}, {r(float(b))})" for a, b in w) + "];")

    # The authors' own simulation output, 1000 runs per point: (Delta t in ms, U, D).
    for case, name in [("DP", "DP"), ("P", "P"), ("Dprime", "D_PRIME")]:
        path = os.path.join(work, f"Graupner2012PNAS/numericalSimulation/output/{case}_curve/final_camkII_state.dat")
        rows = [l.split() for l in io.open(path) if l.strip()]
        print(f"// output/{case}_curve/final_camkII_state.dat: (Delta t ms, U = column 7, D = -column 9)")
        print(f"const SIM_{name}: [(f64, f64, f64); {len(rows)}] = [")
        for row in rows:
            print(f"    ({r(row[0])}, {r(row[6])}, {r(-float(row[8]))}),")
        print("];")
    path = os.path.join(work, "Graupner2012PNAS/numericalSimulation/output/DPDprime_curve/final_camkII_state.dat")
    rows = [l.split() for l in io.open(path) if l.strip()]
    print("// output/DPDprime_curve/final_camkII_state.dat: (Delta t ms, change in strength = column 5, its error = column 6)")
    print(f"const SIM_DPD_PRIME_CHANGE: [(f64, f64, f64); {len(rows)}] = [")
    for row in rows:
        print(f"    ({r(row[0])}, {r(row[4])}, {r(row[5])}),")
    print("];")

    # The parameter files the C++ read, in its units (ms, and eq. 1 divided through by 50).
    keys = ["C_pre", "C_post", "tau_pre", "Ct_dephos", "Ct_phos", "dephos", "phos", "sigma", "tau_rho", "epsilon", "delay"]
    print("// output/<case>_curve/camkmotifscan.par, in the order DP, DPD, DPDprime, P, D, Dprime: " + ", ".join(keys))
    print("const PAR_FILES: [[f64; 11]; 6] = [")
    for case in S1:
        path = os.path.join(work, f"Graupner2012PNAS/numericalSimulation/output/{case}_curve/camkmotifscan.par")
        vals = {}
        for line in io.open(path):
            line = line.split("#")[0]
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k.strip()] = float(v.strip())
        print("    [" + ", ".join(r(vals[k]) for k in keys) + "],")
    print("];")

    # Fig. 4B's caption: "potentiation only above 29 Hz for all Delta t". The lowest change over the
    # Fig. 4B script's grid at frequency f, bisected in f between 25 and 30 Hz, 40 halvings.
    sc, tat = make("cortical slices")

    def lowest(f):
        return min(curve(sc, tat, grid(-0.5 / f, 0.5 / f), f, 75))

    lo, hi = 25.0, 30.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if lowest(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    print("// 'cortical slices', 75 pairs: the frequency above which every offset on the Fig. 4B grid potentiates, bisected")
    print(f"const EVERY_OFFSET_POTENTIATES_HZ: (f64, f64) = ({r(lo)}, {r(hi)});")


if __name__ == "__main__":
    main()
