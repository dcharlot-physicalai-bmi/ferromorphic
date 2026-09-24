#!/usr/bin/env python3
"""Print the PySpike reference values that `src/synchrony.rs` and `src/distance.rs` test against.

    python3 -m venv /tmp/v && /tmp/v/bin/pip install pyspike==0.9.0 numpy
    /tmp/v/bin/python tools/pyspike_reference.py synchrony   # the block in synchrony's tests
    /tmp/v/bin/python tools/pyspike_reference.py distance    # the block in distance's tests

PySpike (Mulansky and Kreuz, SoftwareX 5:183-189, 2016) is the SPIKE measures' reference
implementation, maintained by their authors. Every train here is strictly increasing and inside
[0, 1], so PySpike's silent clean-up -- sorting, dropping duplicates, dropping spikes up to 1e-6
outside the window -- changes nothing, and the two implementations see the same input.
"""
import random
import sys

import pyspike as spk


def synchrony():
    random.seed(20260924)

    def clean(ts):
        return sorted(set(round(t, 4) for t in ts if 0.0 <= t <= 1.0))

    base = sorted(random.uniform(0.05, 0.95) for _ in range(12))
    volley = []
    for k in range(6):
        tr = [b + 0.003 * k + random.gauss(0, 0.004) for b in base if random.random() > 0.2]
        tr += [random.uniform(0, 1) for _ in range(2)]
        volley.append(clean(tr))
    independent = [clean([random.uniform(0, 1) for _ in range(random.randint(0, 14))]) for _ in range(6)]

    def fmt(ts):
        return "&[" + ", ".join(repr(t) for t in ts) + "]"

    for name, trains in [("VOLLEY", volley), ("INDEPENDENT", independent)]:
        sts = [spk.SpikeTrain(t, edges=(0.0, 1.0)) for t in trains]
        print(f"    const {name}: [&[f64]; 6] = [")
        for t in trains:
            print("        " + fmt(t) + ",")
        print("    ];")
        pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
        sync = [spk.spike_sync(sts[i], sts[j]) for i, j in pairs]
        order = [spk.spike_train_order(sts[i], sts[j]) for i, j in pairs]
        print(f"    const {name}_SYNC: [f64; 15] = [" + ", ".join(repr(float(x)) for x in sync) + "];")
        print(f"    const {name}_ORDER: [f64; 15] = [" + ", ".join(repr(float(x)) for x in order) + "];")
        print(f"    const {name}_SYNC_ALL: f64 = {float(spk.spike_sync(sts))!r};")
        print(f"    const {name}_F: f64 = {float(spk.spike_train_order(sts))!r};")
    sts = [spk.SpikeTrain(t, edges=(0.0, 1.0)) for t in volley]
    d = spk.spike_directionality_values([sts[0], sts[1]])
    print("    const DIRECTIONALITY_0: &[i8] = &[" + ", ".join(str(int(x)) for x in d[0]) + "];")
    print("    const DIRECTIONALITY_1: &[i8] = &[" + ", ".join(str(int(x)) for x in d[1]) + "];")


def distance():
    rng = random.Random(400)
    print("    /// `(a, b, ISI-distance, SPIKE-distance)` from `PySpike` 0.9.0 over `[0, 1]`.")
    print("    const PYSPIKE: [(&[f64], &[f64], f64, f64); 20] = [")
    for _ in range(20):
        a = sorted(set([0.0, 1.0] + [round(rng.random(), 6) for _ in range(rng.randint(1, 12))]))
        b = sorted(set([0.0, 1.0] + [round(rng.random(), 6) for _ in range(rng.randint(1, 12))]))
        sa, sb = spk.SpikeTrain(a, edges=(0.0, 1.0)), spk.SpikeTrain(b, edges=(0.0, 1.0))
        print(f"        (&{a!r}, &{b!r}, {float(spk.isi_distance(sa, sb))!r}, {float(spk.spike_distance(sa, sb))!r}),")
    print("    ];")


if __name__ == "__main__":
    if sys.argv[1:] == ["synchrony"]:
        synchrony()
    elif sys.argv[1:] == ["distance"]:
        distance()
    else:
        sys.exit(__doc__)
