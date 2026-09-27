#!/usr/bin/env python3
"""Rebuild the SOM_PAK reference numbers that `src/som.rs`'s tests hold, from Kohonen's lab's own code.

    python3 tools/som_reference.py [workdir]

SOM_PAK 3.1 (April 7, 1995), the self-organising map package of the SOM Programming Team of the
Helsinki University of Technology, is still served by the lab's old address,
http://www.cis.hut.fi/research/som_pak/som_pak-3.1.tar (Last-Modified 9 Jan 1996, 501 760 bytes).
This script downloads it, refuses unless the tarball's SHA-256 is the one recorded below, builds
it twice, runs the four cases the tests hold, and prints the Rust constants they embed. Nothing
from the package enters the crate: its licence permits scientific use and forbids inclusion in
a commercial application, so it is run beside the crate and only its OUTPUT is recorded.

The two builds, and every change made to the package, since a patched reference is only as good as
the account of its patches:

  both    `getline` and `setprogname` are renamed, because the modern C library declares both and
          the 1995 headers conflict with it. A rename of a helper, no arithmetic touched. Compiled
          with `-O -ffp-contract=off`: without the flag clang fuses `m += a * (x - m)` into one
          fused multiply-add on arm64, which rounds once where the source rounds twice (measured:
          the `double` chain case then ends its first phase `1.1e-16` from the unfused build).
  both    one line added to `som_rout.c`'s training loop, after the winner is found: when the
          environment variable `SOMWIN` is set, it prints the step and the winner's index to
          stderr. A print, no arithmetic touched; the codebooks are the same with it set or not.
          It is how the script finds the step at which the two builds' winners first part.
  float   the package as written, except that codebooks are printed with `%.9g` and the
          quantisation error with `%.9g` instead of `%g` and `%f`: six digits would truncate the
          codebook between the two training phases, and nine are enough to round-trip a float.
  double  every `float` in the source made `double`, `%f` in `sscanf` made `%lf`, and output at
          `%.17g`. This is the build the Rust matches to rounding, because the Rust is `f64`.

Data and initial codebooks are integers `k/1024`, drawn with Python's `random.Random(seed)`,
exactly representable in both `float` and `double` — so both builds start from identical
bits — and gains and radii are binary fractions for the same reason. Samples are taken in file
order, cycling (`vsom` without `-rand`); `-rand` would shuffle the file into a new order with the
package's own generator, which the Rust does not reproduce.
"""
import hashlib, io, os, random, re, shutil, subprocess, sys, tarfile, urllib.request

URL = "http://www.cis.hut.fi/research/som_pak/som_pak-3.1.tar"
SHA256 = "a9421ebbe55217674c3a8a5157a8f8e124379e2689c652b7ca6003d0df889f6c"

# (name, dim, topology, xdim, ydim, neighbourhood, samples, [(rlen, alpha, radius), ...], seed)
CASES = [
    ("chain", 1, "rect", 8, 1, "bubble", 16, [(200, 0.5, 4.0), (800, 0.03125, 1.0)], 1990),
    ("hexgauss", 2, "hexa", 4, 3, "gaussian", 24, [(300, 0.5, 3.0), (600, 0.0625, 1.0)], 1982),
    ("rect3", 3, "rect", 5, 4, "bubble", 40, [(500, 0.5, 5.0), (1500, 0.03125, 1.0)], 1995),
    ("hexbubble", 2, "hexa", 6, 5, "bubble", 50, [(1000, 0.5, 4.0), (3000, 0.03125, 1.0)], 2026),
]


def fetch(work):
    tar = os.path.join(work, "som_pak-3.1.tar")
    if not os.path.exists(tar):
        urllib.request.urlretrieve(URL, tar)
    digest = hashlib.sha256(open(tar, "rb").read()).hexdigest()
    if digest != SHA256:
        sys.exit(f"{tar}: SHA-256 {digest}, expected {SHA256}; refusing to build an unknown package")
    return tar


def build(work, tar, kind):
    dest = os.path.join(work, "build_" + kind)
    shutil.rmtree(dest, ignore_errors=True)
    with tarfile.open(tar) as t:
        t.extractall(work)
    shutil.move(os.path.join(work, "som_pak-3.1"), dest)
    for name in os.listdir(dest):
        if not name.endswith((".c", ".h")):
            continue
        path = os.path.join(dest, name)
        src = io.open(path, encoding="latin-1").read()
        src = re.sub(r"\bgetline\b", "sompak_getline", src)
        src = re.sub(r"\bsetprogname\b", "sompak_setprogname", src)
        if kind == "double":
            src = re.sub(r"\bfloat\b(?!\.h)", "double", src)
            src = src.replace('sscanf(toke, "%f", &ent)', 'sscanf(toke, "%lf", &ent)')
        digits = "%.17g" if kind == "double" else "%.9g"
        if name == "datafile.c":
            src = src.replace('fprintf(fp, "%g ", entry->points[i]);', f'fprintf(fp, "{digits} ", entry->points[i]);')
        if name == "som_rout.c":
            anchor = "      byind = win_info.index / codes->xdim;\n    }\n\n    /* Adapt the units */"
            if src.count(anchor) != 1:
                sys.exit("som_rout.c: the winner line to instrument is not where SOM_PAK 3.1 has it")
            src = src.replace(anchor, anchor.replace("xdim;\n", 'xdim;\n      if (getenv("SOMWIN")) fprintf(stderr, "W %ld %ld\\n", le, win_info.index);\n', 1))
        if name == "qerror.c":
            src = src.replace('is %f per sample', f'is {digits} per sample').replace('"%f\\n"', f'"{digits}\\n"')
        io.open(path, "w", encoding="latin-1").write(src)
    subprocess.run(["make", "-f", "makefile.unix", "CFLAGS=-O -ffp-contract=off", "LDFLAGS=", "vsom", "qerror"],
                   cwd=dest, check=True, capture_output=True)
    return dest


def write(path, header, rows):
    with open(path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(" ".join(repr(v / 1024) for v in r) + "\n")


def run_case(bindir, work, case):
    name, dim, topol, xdim, ydim, neigh, n, phases, seed = case
    rng = random.Random(seed)
    data = [rng.randrange(1024) for _ in range(n * dim)]
    code = [rng.randrange(1024) for _ in range(xdim * ydim * dim)]
    d = os.path.join(work, f"{name}_{os.path.basename(bindir)}")
    os.makedirs(d, exist_ok=True)
    write(os.path.join(d, "data.dat"), str(dim), [data[i:i + dim] for i in range(0, len(data), dim)])
    write(os.path.join(d, "c0.cod"), f"{dim} {topol} {xdim} {ydim} {neigh}", [code[i:i + dim] for i in range(0, len(code), dim)])
    outs, qes, wins = [], [], []
    for p, (rlen, alpha, radius) in enumerate(phases):
        cin, cout = os.path.join(d, f"c{p}.cod"), os.path.join(d, f"c{p + 1}.cod")
        run = subprocess.run([os.path.join(bindir, "vsom"), "-din", os.path.join(d, "data.dat"), "-cin", cin, "-cout", cout,
                              "-rlen", str(rlen), "-alpha", repr(alpha), "-radius", repr(radius), "-v", "0"],
                             check=True, capture_output=True, text=True, env={**os.environ, "SOMWIN": "1"})
        w = [int(l.split()[2]) for l in run.stderr.splitlines() if l.startswith("W ")]
        if len(w) != rlen:
            sys.exit(f"{name}, phase {p + 1}: {len(w)} winners printed for {rlen} steps")
        wins.append(w)
        lines = open(cout).read().split("\n")[1:]
        outs.append([float(t) for l in lines if l.strip() and not l.startswith("#") for t in l.split()])
        q = subprocess.run([os.path.join(bindir, "qerror"), "-din", os.path.join(d, "data.dat"), "-cin", cout, "-v", "0"],
                           check=True, capture_output=True, text=True).stdout.split()[0]
        qes.append(float(q))
    return data, code, outs, qes, wins


def main():
    work = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "target/som_pak")
    os.makedirs(work, exist_ok=True)
    tar = fetch(work)
    bins = {kind: build(work, tar, kind) for kind in ("double", "float")}
    for case in CASES:
        name, dim, topol, xdim, ydim, neigh, n, phases, seed = case
        up = name.upper()
        data, code, dbl, dq, dw = run_case(bins["double"], work, case)
        _, _, flt, fq, fw = run_case(bins["float"], work, case)
        # The first step at which the float build's winner is not the double build's:
        # (phase, step within it, double's winner, float's winner), or None.
        parts = [(p + 1, t, a, b) for p in range(len(dw)) for t, (a, b) in enumerate(zip(dw[p], fw[p])) if a != b]
        print(f"    // {name}: {case[1:]}")
        print(f"    const {up}_DATA: [u16; {len(data)}] = {data};")
        print(f"    const {up}_CODE: [u16; {len(code)}] = {code};")
        for tag, outs in (("D", dbl), ("F", flt)):
            for p, o in enumerate(outs):
                print(f"    const {up}_PHASE{p + 1}_{tag}: [f64; {len(o)}] = [{', '.join(repr(v) for v in o)}];")
        print(f"    const {up}_QE: [f64; 4] = [{', '.join(repr(v) for v in dq + fq)}];")
        first = f"Some({parts[0]})" if parts else "None"
        print(f"    const {up}_FLOAT_PARTS: Option<(usize, usize, usize, usize)> = {first};")


if __name__ == "__main__":
    main()
