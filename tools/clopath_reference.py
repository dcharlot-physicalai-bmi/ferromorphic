#!/usr/bin/env python3
"""Regenerate the reference numbers in `src/clopath.rs` from the authors' own MATLAB code.

    python3 tools/clopath_reference.py WORKDIR

The code is ModelDB accession 144566, "Voltage-based STDP synapse (Clopath et al. 2010)", mirrored
at github.com/ModelDBRepository/144566. `VoTriCode/aEIF.m` (the neuron), `VoTri.m` (the rule and the
Sjostrom pairing protocol) and `FrequencyDependence.m` (the driver that draws Fig. 2b) each open
with "Code written by Claudia Clopath", as does `RFdevelop.m`, the receptive-field script whose
homeostat (`theta`, over `tau_th`) is transcribed for item 7 below.

No MATLAB or Octave is available where this was written, so the .m files are not RUN. They are
TRANSCRIBED below, one Python line per MATLAB line, keeping MATLAB's 1-based indices by leaving
slot 0 of every array unused. To keep the transcription honest the script first downloads the three
files at the pinned commit and refuses to continue unless every MATLAB statement it transcribes is
present in them verbatim (whitespace ignored) — so an edit upstream, or a slip here, stops the run
rather than silently changing the table.

What it prints, each value with Python's round-tripping repr:

1. `VoTri(rho, +-10, par)` for Fig. 2b's five frequencies with `par = [0.00014 0.00008 7 15 10]`,
   the visual-cortex row of Table 1b, exactly as `FrequencyDependence.m` calls it.
2. Post-pre pairing at periods of 29, 28 and 27 ms, around the "below 35 Hz" of the main text.
3. The same run with three single changes: `b` as Table 1a prints it (0.805), `b` read in the unit
   the code's comment names (0.0805 nA = 80.5 pA), and the filtered voltages read with no delay.
4. Pre-post pairing at 0.1 Hz and 50 Hz with the filtered voltages read 1, 2 and 4 steps late
   instead of the code's 3: which delays Fig. 2b can tell apart.
5. Post-pre pairing at 0.1, 10, 20 and 50 Hz with each of VoTri.m's two train quirks undone on
   its own: every group's last postsynaptic spike (the sixth, for a negative lag) removed; and
   both trains delayed by three samples, so that the loop's start at `t = 4` delivers the first
   postsynaptic spike.
6. The neuron alone under a 1 nA step for 2 s, with the code's `VT_jump = 20` and with the jump
   that Table 1a's unsigned "30.4 mV" would need (80.8): the step of every spike.
7. The homeostat under the same 1 nA step for 20 s, VT_jump = 20: `u - E_L` and its square
   low-passed from zero by an exact exponential filter over 1.2 s (RFdevelop.m's tau_th) and 1 s
   (the paper's), and the ratios mean-square / mean^2 after 5 000 and 20 000 calls; and, beside
   them, RFdevelop.m's own `theta` -- forward Euler, fed the voltage one call late (`u1ss`) -- over
   the same trajectory.

    python3 tools/clopath_reference.py --perturb-exp WORKDIR

reruns every value above twice, with each result of `exp` moved one unit in the last place down and
then up (`math.nextafter`), and prints how many of them change. MATLAB's `exp` is not the C
library's this transcription calls, and the two may differ in the last place; if the table does not
move under a one-ulp change of every `exp`, it does not depend on which library computed it.
"""
import io
import math
import os
import re
import sys
import urllib.request

COMMIT = "8f336a6089fc503cc46c4e8bf5ff7e34a03f3e4d"
BASE = f"https://raw.githubusercontent.com/ModelDBRepository/144566/{COMMIT}/VoTriCode/"

# Every MATLAB statement transcribed below, as it appears in the file.
TRANSCRIBED = {
    "aEIF.m": [
        "th = 20;", "C = 281;", "g_L = 30;", "E_L = -70.6;", "VT_rest = -50.4;", "Delta_T = 2;",
        "tau_w = 144;", "a = 4;", "b = 0.0805;", "w_jump = 400;", "tau_wtail = 40;", "tau_VT = 50;",
        "VT_jump = 20;", "if counter ==2", "u = E_L+15+6.0984;", "w = w+b;", "w_tail = w_jump;",
        "counter = 0;", "V_T = VT_jump+VT_rest;",
        "udot = 1/C*(-g_L*(u-E_L) + g_L*Delta_T*exp((u-V_T)/Delta_T) - w +w_tail+ I);",
        "wdot = 1/tau_w*(a*(u-E_L) - w);", "u= u + udot;", "w = w + wdot;",
        "w_tail = w_tail-w_tail/tau_wtail;", "V_T = VT_rest/tau_VT+(1-1/tau_VT)*V_T;",
        "if counter == 1", "counter = 2;", "u = 29.4+3.462;", "w = w-wdot;",
        "if (u>th && counter ==0)", "u = 29.4;", "counter = 1;",
    ],
    "VoTri.m": [
        "A_m=par(1);", "A_p=par(2);", "tau_p =par(3);", "tetam = -70.6;", "tetap = -45.3;",
        "tau_r = par(4);", "tau_d = par(5);", "n = 5;", "f = 1000/rho;", "l = n*f+abs(Dt)+1;",
        "x(abs(Dt)+1:f:end-1) = 1;", "y(abs(Dt)+1+Dt:f:end) = 1;", "rho_low = 0.1;", "n_rep = 15;",
        "if rho == 0.1", "n_rep = 10;", "x = repmat([x,zeros(1,1000/rho_low-length(x))],1,n_rep);",
        "y = repmat([y,zeros(1,1000/rho_low-length(y))],1,n_rep);", "I_s = y*1000000;",
        "V_T = -50.4;", "u = E_L*ones(1,l);", "w = 0.5;", "counter = 0;", "for t = 4:l",
        "I_tot(t) =  I_s(t)+I_ext(t);",
        "[u(t), wad(t), w_tail(t), counter, V_T] = aEIF(u(t-1), wad(t-1), w_tail(t-1),I_tot(t), counter, V_T);",
        "u_md(t+1) = u(t)/tau_d +(1-(1/tau_d))*u_md(t);", "u_mp(t+1) = u(t)/tau_p +(1-(1/tau_p))*u_mp(t);",
        "r(t+1) = x(t)/tau_r +(1-(1/tau_r))*r(t);", "u_sig = (u(t) > tetap)*(u(t)-tetap);",
        "u_md_sig = ((u_md(t-3)-tetam) > 0)*(u_md(t-3)-tetam);",
        "u_mp_sig = ((u_mp(t-3)-tetam) > 0)*(u_mp(t-3)-tetam);",
        "w=  w- A_m*x(t)*u_md_sig+ A_p*u_sig*r(t)*u_mp_sig;",
    ],
    "RFdevelop.m": [
        "dt = 1;", "tau_th = 1.2*1000.0;", "E_L = -70.6;", "theta = 0.0*ones(n_neur, 1);",
        "u1s = E_L*ones(n_neur, 1);", "u1ss = E_L*ones(n_neur, 1);",
        "theta = (1-dt/tau_th)*theta+dt/tau_th*((u1ss-E_L).^2);", "u1s = u;", "u1ss = u1s;",
    ],
}

# What a probe changes; the defaults are the code's. `ulp` moves every `exp` result by that many
# units in the last place (0: the C library's value). `quirk` undoes one of VoTri.m's two train
# quirks: "no sixth" removes every group's last postsynaptic spike, "first delivered" delays both
# trains by three samples so that the loop, which starts at t = 4, sees the first spike.
P = {"b": 0.0805, "VT_jump": 20.0, "delay": 3, "ulp": 0, "quirk": None}


def mexp(x):
    """MATLAB's exp: an overflow is Inf, not an exception."""
    try:
        e = math.exp(x)
    except OverflowError:
        return math.inf
    for _ in range(abs(P["ulp"])):
        e = math.nextafter(e, math.inf if P["ulp"] > 0 else 0.0)
    return e


def aEIF(u, w, w_tail, I, counter, V_T):
    th = 20.0
    C = 281.0
    g_L = 30.0
    E_L = -70.6
    VT_rest = -50.4
    Delta_T = 2.0
    tau_w = 144.0
    a = 4.0
    b = P["b"]
    w_jump = 400.0
    tau_wtail = 40.0
    tau_VT = 50.0
    VT_jump = P["VT_jump"]
    if counter == 2:
        u = E_L + 15 + 6.0984
        w = w + b
        w_tail = w_jump
        counter = 0
        V_T = VT_jump + VT_rest
    udot = 1 / C * (-g_L * (u - E_L) + g_L * Delta_T * mexp((u - V_T) / Delta_T) - w + w_tail + I)
    wdot = 1 / tau_w * (a * (u - E_L) - w)
    u = u + udot
    w = w + wdot
    w_tail = w_tail - w_tail / tau_wtail
    V_T = VT_rest / tau_VT + (1 - 1 / tau_VT) * V_T
    if counter == 1:
        counter = 2
        u = 29.4 + 3.462
        w = w - wdot
    if u > th and counter == 0:
        u = 29.4
        counter = 1
    return u, w, w_tail, counter, V_T


def trains(f, Dt):
    """`x` and `y` as VoTri.m builds them for a period of `f` ms, 1-based (slot 0 unused)."""
    n = 5
    l = n * f + abs(Dt) + 1
    x = [0] * (l + 1)
    for i in range(abs(Dt) + 1, l - 1 + 1, f):
        x[i] = 1
    y = [0] * (l + 1)
    for i in range(abs(Dt) + 1 + Dt, l + 1, f):
        y[i] = 1
    n_rep = 10 if f == 10000 else 15
    pad = max(0, 10000 - l)
    x, y = [0] + (x[1:] + [0] * pad) * n_rep, [0] + (y[1:] + [0] * pad) * n_rep
    if P["quirk"] == "no sixth":
        block = l + pad
        for rep in range(n_rep):
            last = max(i for i in range(rep * block + 1, (rep + 1) * block + 1) if y[i])
            y[last] = 0
    elif P["quirk"] == "first delivered":
        x, y = [0, 0, 0, 0] + x[1:], [0, 0, 0, 0] + y[1:]
    elif P["quirk"] is not None:
        sys.exit(f"unknown quirk {P['quirk']!r}")
    return x, y


def VoTri(f, Dt, par):
    A_m, A_p, tau_p, tau_r, tau_d = par
    tetam = -70.6
    tetap = -45.3
    x, y = trains(f, Dt)
    E_L = -70.6
    l = len(x) - 1
    I_s = [yy * 1000000 for yy in y]
    V_T = -50.4
    wad = [0.0] * (l + 1)
    w_tail = [0.0] * (l + 1)
    u = [E_L] * (l + 1)
    u_md = [E_L] * (l + 2)
    u_mp = [E_L] * (l + 2)
    r = [0.0] * (l + 2)
    w = 0.5
    counter = 0
    D = P["delay"]
    for t in range(4, l + 1):
        u[t], wad[t], w_tail[t], counter, V_T = aEIF(u[t - 1], wad[t - 1], w_tail[t - 1], I_s[t] + 0.0, counter, V_T)
        u_md[t + 1] = u[t] / tau_d + (1 - (1 / tau_d)) * u_md[t]
        u_mp[t + 1] = u[t] / tau_p + (1 - (1 / tau_p)) * u_mp[t]
        r[t + 1] = x[t] / tau_r + (1 - (1 / tau_r)) * r[t]
        u_sig = (u[t] - tetap) if u[t] > tetap else 0.0
        u_md_sig = (u_md[t - D] - tetam) if (u_md[t - D] - tetam) > 0 else 0.0
        u_mp_sig = (u_mp[t - D] - tetam) if (u_mp[t - D] - tetam) > 0 else 0.0
        w = w - A_m * x[t] * u_md_sig + A_p * u_sig * r[t] * u_mp_sig
    return w


def step_current(I_pa, steps):
    """aEIF.m under a constant current from rest, one call per ms: the calls it spikes on, and the
    potential after every call."""
    u, w, wt, counter, V_T = -70.6, 0.0, 0.0, 0, -50.4
    spikes, trace = [], []
    for t in range(steps):
        u, w, wt, counter, V_T = aEIF(u, w, wt, I_pa, counter, V_T)
        if counter == 1:
            spikes.append(t)
        trace.append(u)
    return spikes, trace


def homeostat(trace, at):
    """The module's two readings of the long-term average over `trace`, and RFdevelop.m's.

    `u - E_L` and its square are low-passed from zero by an exact exponential filter,
    `m <- d + (m - d) e^{-dt/tau}`, over RFdevelop.m's 1.2 s and over the paper's 1 s. Beside them
    runs RFdevelop.m's own `theta` statement, forward Euler fed the previous call's voltage. After
    each call number in `at`: (mean-square / mean^2 at 1.2 s, mean-square at 1.2 s / mean^2 at 1 s,
    theta / mean-square at 1.2 s)."""
    E_L = -70.6
    dt = 1
    tau_th = 1.2 * 1000.0
    code, paper = math.exp(-0.001 / 1.2), math.exp(-0.001 / 1.0)
    mean = mean_sq = mean_1s = 0.0
    theta = 0.0
    u1s = u1ss = E_L
    out = []
    for k, u in enumerate(trace, 1):
        d = u - E_L
        mean = d + (mean - d) * code
        mean_sq = d * d + (mean_sq - d * d) * code
        mean_1s = d + (mean_1s - d) * paper
        theta = (1 - dt / tau_th) * theta + dt / tau_th * ((u1ss - E_L) ** 2)
        u1s = u
        u1ss = u1s
        if k in at:
            out.append((mean_sq / mean ** 2, mean_sq / mean_1s ** 2, theta / mean_sq))
    return out


def check_transcription(work):
    for name, lines in TRANSCRIBED.items():
        path = os.path.join(work, name)
        if not os.path.exists(path):
            with urllib.request.urlopen(BASE + name, timeout=120) as r:
                io.open(path, "wb").write(r.read())
        text = re.sub(r"\s+", "", io.open(path, encoding="latin-1").read())
        missing = [l for l in lines if re.sub(r"\s+", "", l) not in text]
        if missing:
            sys.exit(f"{name}: transcribed statements not in the source: {missing}")


def table():
    """Every value the tests hold, as `(label, value)` lines."""
    par = [0.00014, 0.00008, 7, 15, 10]
    out = [("// Fig. 2b: (period ms, lag ms, final w)", None)]
    for Dt in (10, -10):
        for f in (10000, 100, 50, 25, 20):
            out.append((f"({f}, {Dt}, ", VoTri(f, Dt, par)))
    out.append(("// post-pre around 35 Hz", None))
    for f in (29, 28, 27):
        out.append((f"({f}, -10, ", VoTri(f, -10, par)))
    out.append(("// single changes at 50 Hz pre-post, and at 0.1 Hz pre-post for the delay", None))
    for key, value, f in (("b", 0.805, 20), ("b", 80.5, 20), ("delay", 0, 10000)):
        saved = P[key]
        P[key] = value
        out.append((f"{key} = {value}: ({f}, 10, ", VoTri(f, 10, par)))
        P[key] = saved
    out.append(("// the delay moved: (delay, period ms, final w), pre-post", None))
    for delay in (1, 2, 4):
        for f in (10000, 20):
            P["delay"] = delay
            out.append((f"({delay}, {f}, ", VoTri(f, 10, par)))
    P["delay"] = 3
    out.append(("// one train quirk undone: (quirk, period ms, final w), post-pre", None))
    for quirk in ("no sixth", "first delivered"):
        for f in (10000, 100, 50, 20):
            P["quirk"] = quirk
            out.append((f"({quirk!r}, {f}, ", VoTri(f, -10, par)))
    P["quirk"] = None
    out.append(("// 1 nA step for 2000 ms: the call of every spike", None))
    for jump in (20.0, 80.8):
        P["VT_jump"] = jump
        s, _ = step_current(1000.0, 2000)
        out.append((f"VT_jump = {jump}, {len(s)} spikes: ", s))
    P["VT_jump"] = 20.0
    out.append(("// the homeostat under 1 nA for 20 s, after 5 000 and 20 000 calls:", None))
    out.append(("// (ms/m^2 at 1.2 s, ms at 1.2 s / m^2 at 1 s, RFdevelop theta / ms at 1.2 s)", None))
    _, trace = step_current(1000.0, 20000)
    for k, ratios in zip((5000, 20000), homeostat(trace, (5000, 20000))):
        out.append((f"after {k}: ", ratios))
    return out


def show(label, value):
    if value is None:
        return label
    if isinstance(value, float):
        return f"{label}{value!r}),"
    if isinstance(value, tuple):
        return label + "(" + ", ".join(repr(v) for v in value) + ")"
    return label + "[" + ", ".join(str(v) for v in value) + "]"


def main():
    args = sys.argv[1:]
    perturb = "--perturb-exp" in args
    args = [a for a in args if a != "--perturb-exp"]
    if len(args) != 1:
        sys.exit("usage: clopath_reference.py [--perturb-exp] WORKDIR")
    os.makedirs(args[0], exist_ok=True)
    check_transcription(args[0])
    base = table()
    for label, value in base:
        print(show(label, value))
    if perturb:
        for ulp in (-1, 1):
            P["ulp"] = ulp
            moved = [(label, a, b) for (label, a), (_, b) in zip(base, table()) if a != b]
            P["ulp"] = 0
            values = sum(1 for _, v in base if v is not None)
            print(f"// every exp moved {ulp:+d} ulp: {len(moved)} of {values} values change")
            for label, a, b in moved:
                print(f"//   {show(label, a)} -> {show(label, b)}")


if __name__ == "__main__":
    main()
