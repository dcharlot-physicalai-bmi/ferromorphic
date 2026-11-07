#!/usr/bin/env python3
"""Official NIR HDF5 files against this crate's text graph.

The `nir` package is the oracle. It is not a dependency of the crate. Install the
pin into a separate environment and point this script at it:

    python3 -m venv /tmp/fm-oracle
    /tmp/fm-oracle/bin/pip install 'numpy==1.26.4' 'h5py==3.11.0' \
        'git+https://github.com/neuromorphs/NIR.git@563c58b45873429e3b1b6f71763ec334ceb981d5'
    /tmp/fm-oracle/bin/python tools/nir_hdf5_oracle.py

The pin is commit 563c58b45873429e3b1b6f71763ec334ceb981d5, SPDX BSD-3-Clause,
copyright 2023 NIR Team. PyPI nir 1.0.8 is tag v1.0.8, four commits behind this
pin, and is not what this script imports.

Official paper fixtures are downloaded from that commit, hashed, read with
`nir.read`, and emitted as `ferromorphic-nir 1` text. `examples/nir_exchange`
parses that text and prints the canonical form. This script checks that the
canonical form re-parses to the same parameter bits.

`nir.read(..., type_check=True)` raises on `lif_rockpool.nir` because the file's
output shape is `[1, 1, 1]` and the preceding LIF output is `[1]`. The graph is
then read with `type_check=False` and the output shape is kept as stored. The
script does not rewrite that shape.

`Flatten.end_dim == -1` is emitted as the token `-1`. The Rust reader stores the
last axis. That expansion is the one documented on `Graph::from_text`.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

PIN = "563c58b45873429e3b1b6f71763ec334ceb981d5"
RAW = f"https://raw.githubusercontent.com/neuromorphs/NIR/{PIN}/"

FIXTURES = {
    "paper/01_lif/lif_norse.nir":
        "0dd9143ef624892d4a6461f474d93653a337b3490a62e23ec9ec5d18fde9b7b6",
    "paper/01_lif/lif_rockpool.nir":
        "f493a2cd00e20be6305acd7808faa1d4558993c91f3af43454bf637371e543d6",
    "paper/01_lif/debug_spike_representation/two_lif_neurons.nir":
        "806c7c1cfae72b5be92ce15a670f37014b82a5d4fba8a077efce46d193c21a82",
    "paper/02_cnn/cnn_sinabs.nir":
        "e2fa55bda7aab5a772485e1b690358bcb825b303eca7dc426e3973937fcb5bcb",
    "paper/03_rnn/braille_noDelay_bias_zero.nir":
        "f1aab3ce74024e7feac508a03b04c58d483ee3a01e293dba4b53d0b519b8650e",
}

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "target" / "verification" / "nir-fixtures"
CANON = ROOT / "tools" / "fixtures" / "nir"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(rel: str) -> Path:
    dest = CACHE / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(RAW + rel, dest)
    data = dest.read_bytes()
    got = sha256(data)
    want = FIXTURES[rel]
    if got != want:
        print(f"FAIL hash {rel} got {got} want {want}")
        sys.exit(1)
    return dest


def fmt_float(value) -> str:
    import math

    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite parameter {value!r}")
    return repr(number)


def fmt_floats(array) -> str:
    import numpy as np

    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    return "[" + ",".join(fmt_float(x) for x in flat) + "]"


def fmt_ints(values) -> str:
    return "[" + ",".join(str(int(v)) for v in values) + "]"


def shape_of(node, attr: str):
    import numpy as np

    array = np.asarray(getattr(node, attr))
    return [int(x) for x in array.shape]


def input_shape(node):
    return [int(x) for x in node.input_type["input"]]


def output_shape(node):
    return [int(x) for x in node.output_type["output"]]


def emit_node(name: str, node) -> str:
    kind = type(node).__name__
    meta = getattr(node, "metadata", None) or {}
    if meta:
        raise ValueError(f"{name} carries metadata {meta!r}, which this text format does not store")
    if kind == "Input":
        body = f"shape={fmt_ints(input_shape(node))}"
    elif kind == "Output":
        body = f"shape={fmt_ints(output_shape(node))}"
    elif kind == "Affine":
        weight = node.weight
        rows, cols = int(weight.shape[0]), int(weight.shape[1])
        body = f"rows={rows} cols={cols} weight={fmt_floats(weight)} bias={fmt_floats(node.bias)}"
    elif kind == "Linear":
        weight = node.weight
        rows, cols = int(weight.shape[0]), int(weight.shape[1])
        body = f"rows={rows} cols={cols} weight={fmt_floats(weight)}"
    elif kind == "LIF":
        body = (
            f"shape={fmt_ints(shape_of(node, 'tau'))} tau={fmt_floats(node.tau)} "
            f"r={fmt_floats(node.r)} v_leak={fmt_floats(node.v_leak)} "
            f"v_threshold={fmt_floats(node.v_threshold)} v_reset={fmt_floats(node.v_reset)}"
        )
    elif kind == "CubaLIF":
        body = (
            f"shape={fmt_ints(shape_of(node, 'tau_mem'))} tau_syn={fmt_floats(node.tau_syn)} "
            f"tau_mem={fmt_floats(node.tau_mem)} r={fmt_floats(node.r)} "
            f"v_leak={fmt_floats(node.v_leak)} v_threshold={fmt_floats(node.v_threshold)} "
            f"v_reset={fmt_floats(node.v_reset)} w_in={fmt_floats(node.w_in)}"
        )
    elif kind == "CubaLI":
        body = (
            f"shape={fmt_ints(shape_of(node, 'tau_mem'))} tau_syn={fmt_floats(node.tau_syn)} "
            f"tau_mem={fmt_floats(node.tau_mem)} r={fmt_floats(node.r)} "
            f"v_leak={fmt_floats(node.v_leak)} w_in={fmt_floats(node.w_in)}"
        )
    elif kind == "IF":
        body = (
            f"shape={fmt_ints(shape_of(node, 'r'))} r={fmt_floats(node.r)} "
            f"v_threshold={fmt_floats(node.v_threshold)} v_reset={fmt_floats(node.v_reset)}"
        )
    elif kind == "Conv2d":
        channels_hw = input_shape(node)
        if len(channels_hw) != 3:
            raise ValueError(f"{name} Conv2d input shape {channels_hw}")
        weight = node.weight
        body = (
            f"in_channels={int(weight.shape[1]) * int(node.groups)} "
            f"out_channels={int(weight.shape[0])} "
            f"size={fmt_ints(channels_hw[1:])} kernel={fmt_ints(weight.shape[2:])} "
            f"stride={fmt_ints(node.stride)} padding={fmt_ints(node.padding)} "
            f"dilation={fmt_ints(node.dilation)} groups={int(node.groups)} "
            f"weight={fmt_floats(weight)} bias={fmt_floats(node.bias)}"
        )
    elif kind == "SumPool2d":
        channels_hw = input_shape(node)
        body = (
            f"channels={channels_hw[0]} size={fmt_ints(channels_hw[1:])} "
            f"kernel={fmt_ints(node.kernel_size)} stride={fmt_ints(node.stride)} "
            f"padding={fmt_ints(node.padding)}"
        )
    elif kind == "Flatten":
        end = int(node.end_dim)
        end_token = "-1" if end == -1 else str(end)
        body = (
            f"size={fmt_ints(input_shape(node))} start_dim={int(node.start_dim)} "
            f"end_dim={end_token}"
        )
    else:
        raise ValueError(f"unsupported NIR node {kind} named {name}")
    if any(c.isspace() or c in "=#" for c in name):
        raise ValueError(f"node name {name!r} is not writable")
    return f"node {name} {kind} {body}"


def emit(graph) -> str:
    lines = ["ferromorphic-nir 1"]
    for name, node in graph.nodes.items():
        lines.append(emit_node(name, node))
    for src, dst in graph.edges:
        lines.append(f"edge {src} {dst}")
    return "\n".join(lines) + "\n"


def canonical(text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write(text)
        path = handle.name
    try:
        proc = subprocess.run(
            ["cargo", "run", "--offline", "--quiet", "--example", "nir_exchange", "--", path],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        print("FAIL nir_exchange")
        sys.exit(1)
    return proc.stdout


def parse_canonical(text: str):
    """Return {name: {key: raw}} and edges from canonical text."""
    nodes = {}
    edges = []
    lines = text.splitlines()
    if not lines or lines[0] != "ferromorphic-nir 1":
        raise ValueError("missing magic")
    for line in lines[1:]:
        if not line:
            continue
        parts = line.split()
        if parts[0] == "edge":
            edges.append((parts[1], parts[2]))
            continue
        if parts[0] != "node":
            raise ValueError(line)
        name, kind = parts[1], parts[2]
        fields = {}
        for item in parts[3:]:
            key, value = item.split("=", 1)
            fields[key] = value
        nodes[name] = (kind, fields)
    return nodes, edges


def bits_of_list(raw: str):
    import numpy as np

    inner = raw[1:-1]
    if inner == "":
        return np.zeros(0, dtype=np.uint64)
    values = np.array([float(part) for part in inner.split(",")], dtype=np.float64)
    return values.view(np.uint64)


def check_bits(rel: str, graph, text: str) -> None:
    import numpy as np

    nodes, edges = parse_canonical(text)
    if [(a, b) for a, b in graph.edges] != edges:
        print(f"FAIL {rel} edges")
        sys.exit(1)
    for name, node in graph.nodes.items():
        kind, fields = nodes[name]
        if kind != type(node).__name__:
            print(f"FAIL {rel} {name} kind {kind}")
            sys.exit(1)
        for attr in (
            "weight", "bias", "tau", "r", "v_leak", "v_threshold", "v_reset",
            "tau_syn", "tau_mem", "w_in",
        ):
            if attr not in fields or not hasattr(node, attr):
                continue
            got = bits_of_list(fields[attr])
            want = np.asarray(getattr(node, attr), dtype=np.float64).reshape(-1).view(np.uint64)
            if got.shape != want.shape or not np.array_equal(got, want):
                print(f"FAIL {rel} {name}.{attr} bits")
                sys.exit(1)


def main() -> int:
    try:
        import nir
    except ImportError:
        print(f"UNAVAILABLE nir package (pin {PIN})")
        return 2
    version = getattr(nir, "__version__", "unknown")
    print(f"nir {version} pin {PIN}")
    if PIN[:7] not in str(version) and not str(version).startswith("1.0.9"):
        print(f"FAIL installed nir {version} is not the pin")
        return 1
    CANON.mkdir(parents=True, exist_ok=True)
    for rel in FIXTURES:
        path = fetch(rel)
        strict_error = None
        try:
            graph = nir.read(str(path), type_check=True)
        except Exception as exc:  # noqa: BLE001 - the file's own type error is the record
            strict_error = f"{type(exc).__name__}: {exc}"
            graph = nir.read(str(path), type_check=False)
        text = canonical(emit(graph))
        check_bits(rel, graph, text)
        out = CANON / (Path(rel).name + ".txt")
        out.write_text(text)
        note = "strict-read" if strict_error is None else "type-check-refused"
        print(f"PASS {rel} {note} nodes={len(graph.nodes)} bytes={len(text)}")
        if strict_error:
            print(f"  type_check: {strict_error}")
    print("PASS official hdf5 exchange")
    try:
        import nirtorch
        import torch
    except ImportError as exc:
        print(f"UNAVAILABLE nirtorch ({exc})")
        return 2
    norse = nir.read(str(CACHE / "paper/01_lif/lif_norse.nir"))
    try:
        nirtorch.nir_to_torch(norse, {})
        print("FAIL default nirtorch map accepted a LIF")
        return 1
    except ValueError as exc:
        if "LIF" not in str(exc):
            print(f"FAIL unexpected nirtorch error: {exc}")
            return 1
        print(f"PASS nirtorch default map rejects LIF: {exc}")
    import numpy as np

    affine = nir.NIRGraph(
        nodes={
            "in": nir.Input(np.array([2])),
            "a": nir.Affine(
                weight=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
                bias=np.array([0.5, -0.5]),
            ),
            "out": nir.Output(np.array([2])),
        },
        edges=[("in", "a"), ("a", "out")],
    )
    module = nirtorch.nir_to_torch(affine, {})
    got = module(torch.tensor([[1.0, 1.0]]))[0].detach().reshape(-1).tolist()
    if [round(v, 6) for v in got] != [3.5, 6.5]:
        print(f"FAIL affine bias {got}")
        return 1
    print(f"PASS nirtorch affine bias {got} version {getattr(nirtorch, '__version__', 'unknown')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
