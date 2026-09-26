#!/usr/bin/env python3
"""Read the model lines of Figs. 1h and 2b out of the paper's PDF and set them against the module.

    python3 tools/clopath_figures.py papers/clopath-2010.pdf WORKDIR

Figs. 1 and 2 of Clopath et al. (2010) are vector graphics, not bitmaps: their model lines are
stroked paths in the content stream of the PDF's second page (journal p. 345), and their vertices
can be read to a thousandth of a point instead of estimated from pixels. This script

1. parses that page's content stream (`pypdf` only decompresses it), tracking the transformation
   matrix and the stroke colour, and collects every stroked path in page coordinates;
2. takes a panel's two model lines as the one blue and one red single stroke drawn through the same
   abscissae (other panels use the same colours), and refuses unless there is exactly one such pair
   with the expected number of vertices;
3. finds the panel's axes (the nearest black segments spanning the lines' box, below and to the left)
   and the tick marks on them (black strokes under 1 pt long with an end on an axis), reads each
   tick's label from `pdftotext -bbox` (poppler), refuses unless the labels are the expected ones,
   and fits a straight line through each axis's ticks;
4. Fig. 1h: sets both lines against the clamp closed form `100 (1 + n g(u))`, capped at the 250 % the
   axis stops at — the visual-cortex set with 25 presynaptic spikes (blue) and the hippocampal set
   with 100 (red), 100 % being a weight of one;
5. Fig. 2b: refuses unless every vertex sits on 0.1 Hz or on a whole number of milliseconds of
   period — which is what they turn out to be: the lines are drawn through 0.1 Hz and EVERY period
   from 100 ms to 20 ms, not through `FrequencyDependence.m`'s five points — and runs the transcribed
   `VoTri.m` (`tools/clopath_reference.py`, which downloads and checks the code) at each vertex's
   period with the code's three-step delay, and at the five Fig. 2b frequencies with delays of 1, 2,
   4 and 5 steps, and prints figure minus model.

Needs `pypdf` and poppler's `pdftotext`, which is why it lives outside the crate; `src/clopath.rs`
embeds what it prints, with this file named as the provenance.
"""
import json
import multiprocessing
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clopath_reference as ref  # noqa: E402

BLACK = (0.0, 0.0, 0.0)
PAR = [0.00014, 0.00008, 7, 15, 10]
NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
TOKEN = re.compile(r"\((?:\\.|[^\\)])*\)|<<|>>|<[0-9A-Fa-f\s]*>|\[|\]|/[^\s/\[\]()<>{}]+|"
                   r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[A-Za-z'\"*]+")


def stroked_paths(pdf):
    """Every stroked path on page 2 as `(colour, [[(x, y), ...], ...])`, in page coordinates."""
    import pypdf

    data = pypdf.PdfReader(pdf).pages[1].get_contents().get_data().decode("latin-1")
    ctm, stack, stroke, operands = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [], None, []
    path, current, out = [], [], []

    def place(x, y):
        return (ctm[0] * x + ctm[2] * y + ctm[4], ctm[1] * x + ctm[3] * y + ctm[5])

    for tok in TOKEN.findall(data):
        if NUMBER.fullmatch(tok):
            operands.append(float(tok))
            continue
        if tok[0] in "/(<[]" or tok == ">>":
            continue
        if tok == "q":
            stack.append((ctm[:], stroke))
        elif tok == "Q":
            ctm, stroke = stack.pop()
        elif tok == "cm":
            a = operands[-6:]
            ctm = [a[0] * ctm[0] + a[1] * ctm[2], a[0] * ctm[1] + a[1] * ctm[3],
                   a[2] * ctm[0] + a[3] * ctm[2], a[2] * ctm[1] + a[3] * ctm[3],
                   a[4] * ctm[0] + a[5] * ctm[2] + ctm[4], a[4] * ctm[1] + a[5] * ctm[3] + ctm[5]]
        elif tok in ("SCN", "SC", "RG", "K", "G"):
            stroke = tuple(round(v, 3) for v in operands)
        elif tok == "m":
            if current:
                path.append(current)
            current = [place(*operands[-2:])]
        elif tok in ("l", "c", "v", "y"):
            current.append(place(*operands[-2:]))
        elif tok == "re":
            x, y, w, h = operands[-4:]
            if current:
                path.append(current)
            path.append([place(x, y), place(x + w, y), place(x + w, y + h), place(x, y + h)])
            current = []
        elif tok in ("S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n"):
            if current:
                path.append(current)
            if tok in ("S", "s", "B", "B*", "b", "b*"):
                out.append((stroke, path))
            path, current = [], []
        operands = []
    return out


def words(pdf):
    """`pdftotext -bbox` words on page 2 as `(text, x centre, y centre)`, y measured up the page."""
    html = subprocess.run(["pdftotext", "-f", "2", "-l", "2", "-bbox", pdf, "-"], capture_output=True,
                          text=True, check=True).stdout
    height = float(re.search(r'<page width="[\d.]+" height="([\d.]+)"', html).group(1))
    found = re.findall(r'xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)<', html)
    return [(t, (float(a) + float(c)) / 2, height - (float(b) + float(d)) / 2) for a, b, c, d, t in found]


def fit(pairs):
    """Least-squares line through `(coordinate, label)` pairs; the map and its residuals."""
    n = len(pairs)
    mx = sum(c for c, _ in pairs) / n
    my = sum(v for _, v in pairs) / n
    slope = sum((c - mx) * (v - my) for c, v in pairs) / sum((c - mx) ** 2 for c, _ in pairs)
    return (lambda c: my + slope * (c - mx)), [v - (my + slope * (c - mx)) for c, v in pairs]


def panel(paths, text, blue, red, vertices, x_labels, y_labels):
    """A panel's (blue, red) lines and its axis maps from page coordinates to axis units."""
    b_lines = [p[0] for c, p in paths if c == blue and len(p) == 1 and len(p[0]) > 2]
    r_lines = [p[0] for c, p in paths if c == red and len(p) == 1 and len(p[0]) > 2]
    pairs = [(b, r) for b in b_lines for r in r_lines if [x for x, _ in b] == [x for x, _ in r]]
    if len(pairs) != 1 or len(pairs[0][0]) != vertices:
        sys.exit(f"expected one blue-red pair of {vertices} shared abscissae, found {[len(b) for b, _ in pairs]}")
    every = pairs[0][0] + pairs[0][1]
    x0, x1 = min(x for x, _ in every), max(x for x, _ in every)
    y0, y1 = min(y for _, y in every), max(y for _, y in every)
    segments = [(a, b) for c, p in paths if c == BLACK for sub in p for a, b in zip(sub, sub[1:])]
    x_axis = [s for s in segments if s[0][1] == s[1][1] and min(s[0][0], s[1][0]) <= x0 + 0.01
              and max(s[0][0], s[1][0]) >= x1 - 0.01 and s[0][1] < y0]
    y_axis = [s for s in segments if s[0][0] == s[1][0] and min(s[0][1], s[1][1]) <= y0 + 0.01
              and max(s[0][1], s[1][1]) >= y1 - 0.01 and s[0][0] <= x0 + 0.01]
    if not x_axis or not y_axis:
        sys.exit(f"axes: found {len(x_axis)} horizontal and {len(y_axis)} vertical candidates")
    # The panel's own axes are the nearest such lines below and to the left of the model lines.
    ax_y, ax_x = max(s[0][1] for s in x_axis), max(s[0][0] for s in y_axis)
    short = [s for s in segments if abs(s[0][0] - s[1][0]) + abs(s[0][1] - s[1][1]) < 1.0]
    x_ticks = sorted({s[0][0] for s in short if s[0][0] == s[1][0] and abs(min(s[0][1], s[1][1]) - ax_y) < 0.01
                      and x0 - 2 < s[0][0] < x1 + 2})
    y_ticks = sorted({s[0][1] for s in short if s[0][1] == s[1][1] and abs(min(s[0][0], s[1][0]) - ax_x) < 0.01
                      and y0 - 20 < s[0][1] < y1 + 20})

    def label(tick, horizontal):
        if horizontal:
            near = [(abs(x - tick), t) for t, x, y in text if abs(x - tick) < 2.0 and 0 < ax_y - y < 8]
        else:
            near = [(abs(y - tick), t) for t, x, y in text if abs(y - tick) < 2.0 and 0 < ax_x - x < 12]
        return min(near)[1].replace("−", "-") if near else None

    got_x = [label(t, True) for t in x_ticks]
    got_y = [label(t, False) for t in y_ticks]
    if got_x != x_labels or got_y != y_labels:
        sys.exit(f"tick labels: {got_x} {got_y}")
    to_x, x_res = fit([(t, float(v)) for t, v in zip(x_ticks, x_labels)])
    to_y, y_res = fit([(t, float(v)) for t, v in zip(y_ticks, y_labels)])
    print(f"axes at y = {ax_y:.3f} pt and x = {ax_x:.3f} pt")
    print(f"x ticks {[round(t, 3) for t in x_ticks]} pt, labels {x_labels}, residuals {[round(r, 4) for r in x_res]}")
    print(f"y ticks {[round(t, 3) for t in y_ticks]} pt, labels {y_labels}, residuals {[round(r, 4) for r in y_res]}")
    return pairs[0], to_x, to_y


def clamp_percent(u, theta_minus, theta_plus, a_ltd, a_ltp, n):
    """Fig. 1h's closed form: `100 (1 + n g(u))`, `g` the clamp form of eq. (3), u in mV."""
    over_minus = max(u - theta_minus, 0.0)
    over_plus = max(u - theta_plus, 0.0)
    return 100 * (1 + n * (-a_ltd * over_minus + a_ltp * over_plus * over_minus))


def figure_1h(paths, text):
    print("// Fig. 1h")
    (blue, red), mv, pct = panel(paths, text, (0.196, 0.29, 0.592), (0.886, 0.0, 0.102), 426,
                                 ["-80", "-60", "-40", "-20", "0"], ["100", "150", "200", "250"])
    sets = (("visual cortex, 25 spikes", blue, (-70.6, -45.3, 14e-5, 8e-5, 25)),
            ("hippocampal, 100 spikes", red, (-41.0, -38.0, 38e-5, 2e-5, 100)))
    for name, line, par in sets:
        read = [(mv(x), pct(y)) for x, y in line]
        below = [(u, p, p - clamp_percent(u, *par)) for u, p in read if p < 249.5]
        worst = max(below, key=lambda t: abs(t[2]))
        low = min(read, key=lambda t: t[1])
        print(f"{name}: {len(read)} vertices, {read[0][0]:.2f} to {read[-1][0]:.2f} mV; "
              f"{len(below)} below 249.5 %, worst |figure - closed form| {abs(worst[2]):.3f} at {worst[0]:.2f} mV; "
              f"top vertex {max(p for _, p in read):.3f} %")
        print(f"  lowest vertex ({low[0]:.4f} mV, {low[1]:.4f} %), closed form there {clamp_percent(low[0], *par):.4f} %")
        # Every vertex at that lowest ordinate, and the ordinate step down to the first of them; the
        # tests embed these unrounded, so they are printed with the round-tripping repr.
        first = read.index(low)
        flat = [(u, p) for u, p in read if p == low[1]]
        print(f"  lowest ordinate, unrounded: {low[1]!r} %, on {len(flat)} vertices at {[u for u, _ in flat]!r} mV; "
              f"the vertex before it is {read[first - 1][1] - low[1]:.4f} points higher")
        cap = next(i for i, (_, p) in enumerate(read) if p > 249.9)
        if cap:
            print(f"  reaches the cap between vertices at {read[cap - 1][0]:.3f} mV ({read[cap - 1][1]:.3f} %) "
                  f"and {read[cap][0]:.3f} mV ({read[cap][1]:.3f} %); unrounded {read[cap - 1]!r} and {read[cap]!r}")


def model(job):
    delay, period, lag = job
    ref.P["delay"] = delay
    return job, 100 + 100 * (ref.VoTri(period, lag, PAR) - 0.5) / 0.5


def figure_2b(paths, text, work):
    print("\n// Fig. 2b")
    (blue, red), hz, pct = panel(paths, text, (0.141, 0.2, 0.537), (0.882, 0.106, 0.145), 82,
                                 ["0", "20", "40"], ["50", "100", "150"])
    read = {}
    for name, line in (("pre-post", blue), ("post-pre", red)):
        for i, (x, y) in enumerate(line):
            f = hz(x)
            if i == 0:
                if abs(f - 0.1) > 0.02:
                    sys.exit(f"{name}: first vertex at {f} Hz, not 0.1 Hz")
                period = 10000
            else:
                period = round(1000 / f)
                if abs(1000 / f - period) > 0.1:
                    sys.exit(f"{name}: vertex {i} at {1000 / f} ms, not a whole number of milliseconds")
            read[(name, period)] = pct(y)
    periods = sorted({p for _, p in read}, reverse=True)
    print(f"vertices at 0.1 Hz and at every period from {periods[1]} ms to {periods[-1]} ms: {len(periods)} each")

    ref.check_transcription(work)
    lag = {"pre-post": 10, "post-pre": -10}
    five = (10000, 100, 50, 25, 20)
    jobs = [(3, p, lag[n]) for n in lag for p in periods]
    jobs += [(d, p, lag[n]) for d in (1, 2, 4, 5) for n in lag for p in five]
    with multiprocessing.Pool() as pool:
        sim = dict(pool.map(model, jobs))

    print("\n// Fig. 2b read from the PDF at 0.1, 10, 20, 40, 50 Hz, %: pre-post, then post-pre")
    for name in lag:
        print(name, [round(read[(name, p)], 2) for p in five])
    print("\n// figure minus the code's model at those five, points, by read delay")
    for d in (1, 2, 3, 4, 5):
        for name in lag:
            print(f"delay {d} {name}: {[round(read[(name, p)] - sim[(d, p, lag[name])], 2) for p in five]}")
    print("\n// figure minus the code's model (delay 3) at every period, points")
    for name in lag:
        diffs = [(p, read[(name, p)] - sim[(3, p, lag[name])]) for p in periods]
        print(name, json.dumps([[p, round(e, 2)] for p, e in diffs]))
        # From 0.1 Hz down through the periods, how far the line stays within 0.05 points of the code.
        agree = 0
        while agree < len(diffs) and abs(diffs[agree][1]) <= 0.05:
            agree += 1
        if agree:
            print(f"  within 0.05 points for the first {agree} vertices, 0.1 Hz and {diffs[1][0]} ms down to "
                  f"{diffs[agree - 1][0]} ms (max {max(abs(e) for _, e in diffs[:agree]):.3f})")
        print(f"  min, max figure - model over all 82: {min(e for _, e in diffs):.3f}, {max(e for _, e in diffs):.3f}")
    print("\n// unrounded, at the five: figure, and figure minus the delay-3 model")
    for name in lag:
        print(name, [(read[(name, p)], read[(name, p)] - sim[(3, p, lag[name])]) for p in five])


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: clopath_figures.py PDF WORKDIR")
    pdf, work = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)
    paths = stroked_paths(pdf)
    text = words(pdf)
    figure_1h(paths, text)
    figure_2b(paths, text, work)


if __name__ == "__main__":
    main()
