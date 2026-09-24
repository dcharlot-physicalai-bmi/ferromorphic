#!/usr/bin/env python3
"""Regenerate the `REFERENCE` table in `src/glif.rs` from the AllenSDK's own code.

    python3 tools/glif_reference.py WORKDIR > reference_block.rs

What it does, so the numbers in the tests can be traced to their source:

1. Downloads `allensdk/model/glif/glif_neuron.py` and `glif_neuron_methods.py` at commit
   1bdca3ad884c3a5edea8236161424650603e6f29 (the last change to either, 2026-02-20) and runs them
   UNMODIFIED. Their only imports outside numpy are `simplejson` and
   `allensdk.core.json_utilities`, which are stubbed because the run never serialises anything.
2. Downloads five fitted models' `neuron_config.json` from the Allen Cell Types Database, one per
   GLIF level, by well-known-file id.
3. Runs each under 100 ms of nothing, 700 ms at twice its rheobase `th_inf * coeffs.th_inf /
   R_input`, 200 ms of nothing, at its own dt.
4. Runs each again with the methods file's `dynamics_voltage_linear_exact` registered — it is
   defined there and left out of `METHOD_LIBRARY` — in place of `linear_forward_euler`.
5. Prints the Rust table: each config compacted to one line (Python's float repr round-trips, so
   the bits are the served ones), every spike step and interpolated time, the count of cut samples,
   each trace's exact `math.fsum`, and the AllenSDK's interpolated voltage and threshold at the
   first spike.

Needs network access and numpy. The AllenSDK's licence does not permit redistributing its code, so
this script fetches it rather than the repository carrying a copy.
"""
import io
import json
import math
import os
import sys
import urllib.request

COMMIT = "1bdca3ad884c3a5edea8236161424650603e6f29"
SDK = f"https://raw.githubusercontent.com/AllenInstitute/AllenSDK/{COMMIT}/allensdk/model/glif/"
# (neuronal model id, well-known-file id of its neuron_config.json), GLIF levels 1 to 5.
MODELS = [
    (573430216, 573445037),
    (555405421, 587919086),
    (482525253, 497623063),
    (484633260, 566444698),
    (566367589, 566374084),
]


def fetch(url, path):
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=120) as r:
            io.open(path, "wb").write(r.read())
    return path


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: glif_reference.py WORKDIR")
    work = os.path.abspath(sys.argv[1])
    pkg = os.path.join(work, "stubs", "allensdk", "model", "glif")
    os.makedirs(pkg, exist_ok=True)
    os.makedirs(os.path.join(work, "stubs", "allensdk", "core"), exist_ok=True)
    for d in ["allensdk", "allensdk/model", "allensdk/model/glif", "allensdk/core"]:
        io.open(os.path.join(work, "stubs", d, "__init__.py"), "a").close()
    io.open(os.path.join(work, "stubs", "allensdk", "core", "json_utilities.py"), "w").write(
        "def json_handler(o):\n    return str(o)\n"
    )
    io.open(os.path.join(work, "stubs", "simplejson.py"), "w").write("from json import *\n")
    for f in ["glif_neuron.py", "glif_neuron_methods.py"]:
        fetch(SDK + f, os.path.join(pkg, f))
    sys.path.insert(0, os.path.join(work, "stubs"))
    import numpy as np
    from allensdk.model.glif.glif_neuron import GlifNeuron

    methods = sys.modules[GlifNeuron.configure_method(None, lambda: None, {}).__class__.__module__]

    def run(cfg, exact):
        cfg = dict(cfg)
        if exact:
            methods.METHOD_LIBRARY["voltage_dynamics_method"]["linear_exact"] = methods.dynamics_voltage_linear_exact
            cfg["voltage_dynamics_method"] = {"name": "linear_exact", "params": {}}
        rheobase = cfg["th_inf"] * cfg["coeffs"]["th_inf"] / cfg["R_input"]
        stim = np.zeros(20000)
        stim[2000:16000] = 2.0 * rheobase
        return GlifNeuron.from_dict(cfg).run(stim)

    def fsum(xs):
        return repr(math.fsum(float(x) for x in xs if not math.isnan(x)))

    def wrap(items, per):
        return ["                " + ", ".join(items[i : i + per]) + "," for i in range(0, len(items), per)]

    out = ["    const REFERENCE: [Reference; 5] = ["]
    for level, (mid, wkf) in enumerate(MODELS, 1):
        path = fetch(f"http://api.brain-map.org/api/v2/well_known_file_download/{wkf}", os.path.join(work, f"{mid}.json"))
        cfg = json.load(io.open(path))
        text = json.dumps(cfg, separators=(",", ":"))
        assert "\\" not in text and '"#' not in text
        r, e = run(cfg, False), run(cfg, True)
        out += [
            "        Reference {",
            f"            id: {mid},",
            f"            level: {level},",
            f'            config: r#"{text}"#,',
            "            steps: &[",
            *wrap([str(int(s)) for s in r["spike_time_steps"]], 12),
            "            ],",
            "            times: &[",
            *wrap([repr(float(t)) for t in r["interpolated_spike_times"]], 5),
            "            ],",
            f"            nan: {int(np.isnan(r['voltage']).sum())},",
            f"            sum_v: {fsum(r['voltage'])},",
            f"            sum_th: {fsum(r['threshold'])},",
            "            sum_asc: &["
            + (", ".join(fsum(r["AScurrents"][:, j]) for j in range(r["AScurrents"].shape[1])) if level >= 3 else "")
            + "],",
            f"            first_allen_v: {float(r['interpolated_spike_voltage'][0])!r},",
            f"            first_allen_th: {float(r['interpolated_spike_threshold'][0])!r},",
            "            exact_steps: &[",
            *wrap([str(int(s)) for s in e["spike_time_steps"]], 12),
            "            ],",
            f"            exact_sum_v: {fsum(e['voltage'])},",
            "        },",
        ]
    out.append("    ];")
    print("\n".join(out))


if __name__ == "__main__":
    main()
