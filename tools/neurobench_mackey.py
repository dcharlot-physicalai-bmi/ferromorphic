#!/usr/bin/env python3
"""Held-out Mackey–Glass run against the pinned NeuroBench harness.

Data: https://huggingface.co/datasets/NeuroBench/mackey_glass file data.tar.gz
License: CC-BY-4.0 (dataset card). sha256 is checked below. No account.

Harness: NeuroBench commit e521c28f5798cb40bb9423758bf066dad53fe070, tag 2.3.0,
Apache-2.0. The echo-state example and its hyperparameter table are fetched from
that commit into target/ and imported. They are not copied into this repository.

The published comment in examples/mackey_glass/esn_benchmark.py is the average
sMAPE for repeat=1, seed_id = repeat_id = 0, tau 17..30, start_offset 0. This
script runs that one repeat. It is not the repeat=30 average, and it is not a
score for any other NeuroBench task.

Rust `examples/smape_pair` recomputes sMAPE on the same prediction and target
bytes the harness scored.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

NB_PIN = "e521c28f5798cb40bb9423758bf066dad53fe070"
DATA_URL = "https://huggingface.co/datasets/NeuroBench/mackey_glass/resolve/main/data.tar.gz"
DATA_SHA256 = "5e7c2b62a5135b744e7b49f103f58a2d74c70fe2dae67890cd5f93a6110fdf7c"
DATA_LICENSE = "CC-BY-4.0"
# Comment in esn_benchmark.py at the pin, for repeat=1.
PUBLISHED_REPEAT1 = 40.803375080880265

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "target" / "verification" / "mackey"
RAW = f"https://raw.githubusercontent.com/NeuroBench/neurobench/{NB_PIN}/examples/mackey_glass/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)


def prepare() -> Path:
    archive = WORK / "data.tar.gz"
    fetch(DATA_URL, archive)
    got = sha256(archive)
    if got != DATA_SHA256:
        print(f"FAIL data.tar.gz sha256 {got}")
        sys.exit(1)
    data_dir = WORK / "data"
    if not (data_dir / "mg_17.npy").exists():
        with tarfile.open(archive) as tar:
            tar.extractall(WORK)
    example = WORK / "example"
    example.mkdir(parents=True, exist_ok=True)
    for name in (
        "echo_state_network.py",
        "model_data/echo_state_network_hyperparameters.csv",
    ):
        dest = example / name
        fetch(RAW + name, dest)
    return data_dir


def write_f64(path: Path, array) -> None:
    import numpy as np

    values = np.asarray(array, dtype=np.float64).reshape(-1)
    path.write_bytes(struct.pack("<" + "d" * values.size, *values.tolist()))


def rust_smape(pred: Path, label: Path) -> float:
    proc = subprocess.run(
        [
            "cargo", "run", "--offline", "--quiet", "--example", "smape_pair", "--",
            str(pred), str(label),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        print("FAIL smape_pair")
        sys.exit(1)
    return float(proc.stdout.strip())


def main() -> int:
    try:
        import neurobench
        import torch
        from torch.utils.data import DataLoader, Subset
    except ImportError as exc:
        print(f"UNAVAILABLE {exc}")
        return 2
    print(f"neurobench {getattr(neurobench, '__version__', 'unknown')} pin {NB_PIN}")
    print(f"torch {torch.__version__}")
    if getattr(neurobench, "__version__", "") != "2.3.0":
        print("FAIL neurobench version is not 2.3.0")
        return 1

    data_dir = prepare()
    sys.path.insert(0, str(WORK / "example"))
    import pandas as pd
    from echo_state_network import EchoStateNetwork
    from neurobench.benchmarks import Benchmark
    from neurobench.datasets import MackeyGlass
    from neurobench.metrics.static import ConnectionSparsity, Footprint
    from neurobench.metrics.workload import ActivationSparsity, SMAPE, SynapticOperations
    from neurobench.models import TorchModel

    params = pd.read_csv(WORK / "example" / "model_data" / "echo_state_network_hyperparameters.csv")
    scores = []
    rust_scores = []
    rows = []
    out_dir = WORK / "heldout"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tau in range(17, 31):
        path = data_dir / f"mg_{tau}.npy"
        mg = MackeyGlass(str(path), start_offset=0.0, bin_window=1, download=False)
        train_set = Subset(mg, mg.ind_train)
        test_set = Subset(mg, mg.ind_test)
        row = params.index[params["tau"] == tau].tolist()[0]
        esn = EchoStateNetwork(
            in_channels=1,
            reservoir_size=int(params["reservoir_size"][row]),
            input_scale=torch.tensor(
                [params["scale_bias"][row], params["scale_input"][row]], dtype=torch.float64
            ),
            connect_prob=float(params["connect_prob"][row]),
            spectral_radius=float(params["spectral_radius"][row]),
            leakage=float(params["leakage"][row]),
            ridge_param=float(params["ridge_param"][row]),
            seed_id=0,
        )
        esn.train()
        train_data, train_labels = train_set[:]
        warmup_pts = round(0.6 * mg.pts_per_lyaptime)
        esn.fit(train_data, train_labels[warmup_pts:], warmup_pts)
        fitted = {key: value.detach().clone() for key, value in esn.state_dict().items()}
        reservoir = esn.reservoir.detach().clone()
        loader = DataLoader(test_set, batch_size=mg.testtime_pts, shuffle=False)
        model = TorchModel(esn)
        benchmark = Benchmark(
            model,
            loader,
            [],
            [],
            [[Footprint, ConnectionSparsity], [SMAPE, ActivationSparsity, SynapticOperations]],
        )
        results = benchmark.run(quiet=True)
        harness = float(results["SMAPE"])
        esn.load_state_dict(fitted)
        esn.reservoir = reservoir.clone()
        esn.prior_prediction = None
        batch = next(iter(loader))
        with torch.no_grad():
            preds = model(batch[0])
        direct = float(SMAPE()(model, preds, batch))
        if abs(direct - harness) > 1e-8:
            print(f"FAIL tau {tau} replayed predictions differ from the harness: {direct} vs {harness}")
            return 1
        pred_path = out_dir / f"tau{tau}_pred.f64"
        label_path = out_dir / f"tau{tau}_label.f64"
        write_f64(pred_path, preds.detach().cpu().numpy())
        write_f64(label_path, batch[1].detach().cpu().numpy())
        rust = rust_smape(pred_path, label_path)
        scores.append(harness)
        rust_scores.append(rust)
        rows.append({"tau": tau, "harness_smape": harness, "rust_smape": rust, "abs_diff": abs(harness - rust)})
        print(f"tau {tau} harness {harness:.8f} rust {rust:.8f}")

    harness_mean = sum(scores) / len(scores)
    rust_mean = sum(rust_scores) / len(rust_scores)
    report = {
        "task": "mackey_glass",
        "held_out": True,
        "repeat": 1,
        "seed_id": 0,
        "start_offset": 0.0,
        "tau": list(range(17, 31)),
        "neurobench_pin": NB_PIN,
        "neurobench_version": neurobench.__version__,
        "torch": torch.__version__,
        "data_url": DATA_URL,
        "data_sha256": DATA_SHA256,
        "data_license": DATA_LICENSE,
        "harness_smape_mean": harness_mean,
        "rust_smape_mean": rust_mean,
        "published_repeat1_comment": PUBLISHED_REPEAT1,
        "rows": rows,
    }
    (WORK / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"harness mean {harness_mean:.12f}")
    print(f"rust mean    {rust_mean:.12f}")
    print(f"comment      {PUBLISHED_REPEAT1:.12f}")
    worst = max(row["abs_diff"] for row in rows)
    if worst > 1e-6:
        print(f"FAIL rust and harness differ by {worst}")
        return 1
    print("PASS mackey-glass repeat=1 held-out sMAPE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
