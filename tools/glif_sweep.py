#!/usr/bin/env python3
"""Run EVERY GLIF model the Allen Cell Types Database serves through the AllenSDK, for comparison
with `examples/glif_allen.rs`.

    python3 tools/glif_sweep.py fetch   CACHE      # list and download every neuron_config.json
    python3 tools/glif_sweep.py sdk     CACHE      # run each through the AllenSDK -> CACHE/allensdk.tsv
    cargo run --release --example glif_allen -- CACHE/configs > CACHE/ferromorphic.tsv
    python3 tools/glif_sweep.py compare CACHE      # CACHE/allensdk.tsv against CACHE/ferromorphic.tsv

The stimulus is the tests' at a third of the length: 50 ms of nothing, 200 ms at twice the model's
rheobase `th_inf * coeffs.th_inf / R_input`, 50 ms of nothing, at the model's own dt. Each output
line is `id  level  spikes  first,steps,...  nan  sum_v  sum_abs_v`, or `id  level  ERROR  message`.

`compare` requires the same spike steps and cut-sample count for every model, and each voltage trace's
sum within 1e-12 of its sum of magnitudes, and prints every model that differs. A model the AllenSDK
cannot run and this crate refuses counts as agreement only when both refuse.

The AllenSDK's GLIF code is fetched at the commit `tools/glif_reference.py` pins and run unmodified,
with `simplejson` and `allensdk.core.json_utilities` stubbed.
"""
import concurrent.futures
import io
import json
import math
import os
import sys
import urllib.request

API = "http://api.brain-map.org/api/v2/data/query.json?criteria="
TEMPLATES = {395310469: 1, 395310479: 2, 395310475: 3, 471355161: 4, 395310498: 5}
COMMIT = "1bdca3ad884c3a5edea8236161424650603e6f29"
SDK = f"https://raw.githubusercontent.com/AllenInstitute/AllenSDK/{COMMIT}/allensdk/model/glif/"


def get(url, attempts=4):
    for k in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception:
            if k == attempts - 1:
                raise


def fetch(cache):
    os.makedirs(os.path.join(cache, "configs"), exist_ok=True)
    rows = []
    for template, level in TEMPLATES.items():
        start = 0
        while True:
            q = (
                f"model::NeuronalModel,rma::criteria,[neuronal_model_template_id$eq{template}],"
                f"rma::include,well_known_files(well_known_file_type),"
                f"rma::options[num_rows$eq500][start_row$eq{start}][order$eq'id']"
            )
            d = json.loads(get(API + q))
            for m in d["msg"]:
                files = [w for w in m["well_known_files"] if w["well_known_file_type"]["name"] == "NeuronalModelParameters"]
                rows.append((m["id"], level, files[0]["id"] if files else None))
            start += len(d["msg"])
            if start >= d["total_rows"] or not d["msg"]:
                break
        print(f"template {template} (level {level}): {sum(1 for r in rows if r[1] == level)} models", file=sys.stderr)
    io.open(os.path.join(cache, "models.tsv"), "w").write("".join(f"{i}\t{l}\t{w}\n" for i, l, w in rows))

    def one(row):
        mid, _, wkf = row
        path = os.path.join(cache, "configs", f"{mid}.json")
        if wkf is None or os.path.exists(path):
            return
        io.open(path + ".part", "wb").write(get(f"http://api.brain-map.org/api/v2/well_known_file_download/{wkf}"))
        os.replace(path + ".part", path)

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        list(pool.map(one, rows))
    missing = [r for r in rows if r[2] is None]
    print(f"{len(rows)} models, {len(missing)} without a neuron_config", file=sys.stderr)


def stubs(cache):
    pkg = os.path.join(cache, "stubs", "allensdk", "model", "glif")
    os.makedirs(pkg, exist_ok=True)
    os.makedirs(os.path.join(cache, "stubs", "allensdk", "core"), exist_ok=True)
    for d in ["allensdk", "allensdk/model", "allensdk/model/glif", "allensdk/core"]:
        io.open(os.path.join(cache, "stubs", d, "__init__.py"), "a").close()
    io.open(os.path.join(cache, "stubs", "allensdk", "core", "json_utilities.py"), "w").write("def json_handler(o):\n    return str(o)\n")
    io.open(os.path.join(cache, "stubs", "simplejson.py"), "w").write("from json import *\n")
    for f in ["glif_neuron.py", "glif_neuron_methods.py"]:
        path = os.path.join(pkg, f)
        if not os.path.exists(path):
            io.open(path, "wb").write(get(SDK + f))
    return os.path.join(cache, "stubs")


def run_one(args):
    stub_dir, path, level = args
    sys.path.insert(0, stub_dir)
    import logging

    import numpy as np
    from allensdk.model.glif.glif_neuron import GlifNeuron

    logging.disable(logging.CRITICAL)
    mid = os.path.basename(path)[:-5]
    try:
        cfg = json.load(io.open(path))
        rheobase = cfg["th_inf"] * cfg["coeffs"]["th_inf"] / cfg["R_input"]
        stim = np.zeros(6000)
        stim[1000:5000] = 2.0 * rheobase
        r = GlifNeuron.from_dict(cfg).run(stim)
        v = r["voltage"]
        steps = ",".join(str(int(s)) for s in r["spike_time_steps"])
        total = math.fsum(float(x) for x in v if not math.isnan(x))
        scale = math.fsum(abs(float(x)) for x in v if not math.isnan(x))
        return f"{mid}\t{level}\t{len(r['spike_time_steps'])}\t{steps or '-'}\t{int(np.isnan(v).sum())}\t{total!r}\t{scale!r}"
    except Exception as e:
        return f"{mid}\t{level}\tERROR\t{type(e).__name__}: {str(e)[:80]}"


def sdk(cache):
    stub_dir = stubs(cache)
    levels = {l.split("\t")[0]: l.split("\t")[1] for l in io.open(os.path.join(cache, "models.tsv")).read().splitlines()}
    paths = sorted(os.path.join(cache, "configs", f) for f in os.listdir(os.path.join(cache, "configs")) if f.endswith(".json"))
    with concurrent.futures.ProcessPoolExecutor() as pool:
        lines = list(pool.map(run_one, [(stub_dir, p, levels.get(os.path.basename(p)[:-5], "?")) for p in paths], chunksize=16))
    io.open(os.path.join(cache, "allensdk.tsv"), "w").write("\n".join(lines) + "\n")
    print(f"{len(lines)} models run, {sum('ERROR' in l for l in lines)} refused by the AllenSDK", file=sys.stderr)


def compare(cache):
    def table(name):
        return {l.split("\t")[0]: l.split("\t") for l in io.open(os.path.join(cache, name)).read().splitlines() if l}

    ours, theirs = table("ferromorphic.tsv"), table("allensdk.tsv")
    same, differ = 0, []
    for mid, t in sorted(theirs.items()):
        o = ours.get(mid)
        if o is None:
            differ.append((mid, "missing from ferromorphic.tsv"))
        elif t[2] == "ERROR" or o[2] == "ERROR":
            if t[2] == o[2] == "ERROR":
                same += 1
            else:
                differ.append((mid, f"allensdk {t[2:]} / ferromorphic {o[2:]}"))
        elif t[1] != o[1]:
            differ.append((mid, f"the database files it at level {t[1]}, this crate reads level {o[1]}"))
        elif t[2:5] != o[2:5]:
            differ.append((mid, f"spikes or cut differ: {t[2]} vs {o[2]}"))
        else:
            a, b = float(t[5]), float(o[5])
            if abs(a - b) <= 1e-12 * float(t[6]):
                same += 1
            else:
                differ.append((mid, f"sum_v {a!r} vs {b!r}"))
    for mid, why in differ:
        print(mid, why)
    print(f"{same} of {len(theirs)} models agree; {len(differ)} differ")
    return 0 if not differ else 1


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("fetch", "sdk", "compare"):
        sys.exit(__doc__)
    sys.exit({"fetch": fetch, "sdk": sdk, "compare": compare}[sys.argv[1]](os.path.abspath(sys.argv[2])) or 0)
