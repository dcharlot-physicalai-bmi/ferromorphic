#!/usr/bin/env python3
"""All six official Primate Reaching sessions, upstream 2D ANN protocol.

Zenodo 583331, CC-BY-4.0. NeuroBench e521c28 example benchmark_2d_ann.py:
train_ratio 0.5, bin_width 0.004, biological_delay 0, num_steps 1,
remove_segments_inactive False, test indices, batch 256.
Checkpoints are that commit's model_data/2D_ANN_Weight state dicts.
Rust r2_columns is a metric replay, recorded separately from model inference.
"""
from __future__ import annotations

import gc
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

NB_PIN = "e521c28f5798cb40bb9423758bf066dad53fe070"
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "target" / "verification" / "primate"
RAW = f"https://raw.githubusercontent.com/NeuroBench/neurobench/{NB_PIN}/examples/primate_reaching/"
FILES = {
    "indy_20160622_01": "c33d5fff31320d709d23fe445561fb6e",
    "indy_20160630_01": "197413a5339630ea926cbd22b8b43338",
    "indy_20170131_02": "2790b1c869564afaa7772dbf9e42d784",
    "loco_20170210_03": "4cae63b58c4cb9c8abd44929216c703b",
    "loco_20170215_02": "739b70762d838f3a1f358733c426bb02",
    "loco_20170301_05": "47342da09f9c950050c9213c3df38ea3",
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    import torch
    from torch.utils.data import DataLoader, Subset
    import neurobench
    import numpy as np

    example = WORK / "example"
    example.mkdir(parents=True, exist_ok=True)
    ann = example / "ANN.py"
    if not ann.exists():
        urllib.request.urlretrieve(RAW + "ANN.py", ann)
    sys.path.insert(0, str(example))
    from ANN import ANNModel2D
    from neurobench.benchmarks import Benchmark
    from neurobench.datasets import PrimateReaching
    from neurobench.metrics.static import ConnectionSparsity, Footprint
    from neurobench.metrics.workload import ActivationSparsity, R2, SynapticOperations
    from neurobench.models import TorchModel
    rows = []
    out_dir = WORK / "heldout"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, expect in FILES.items():
        mat = WORK / "data" / f"{name}.mat"
        got = md5(mat)
        if got != expect:
            print(f"FAIL {name} md5 {got}")
            return 1
        weight = example / f"{name}_model_state_dict.pth"
        if not weight.exists():
            urllib.request.urlretrieve(
                RAW + f"model_data/2D_ANN_Weight/{name}_model_state_dict.pth", weight
            )
        dataset = PrimateReaching(
            file_path=str(WORK / "data"),
            filename=name,
            num_steps=1,
            train_ratio=0.5,
            bin_width=0.004,
            biological_delay=0,
            remove_segments_inactive=False,
            download=False,
        )
        loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=256, shuffle=False)
        net = ANNModel2D(
            input_dim=dataset.input_feature_size,
            layer1=32,
            layer2=48,
            output_dim=2,
            bin_window=0.2,
            drop_rate=0.5,
        )
        net.load_state_dict(torch.load(weight, map_location="cpu"))
        model = TorchModel(net)
        benchmark = Benchmark(
            model,
            loader,
            [],
            [],
            [[Footprint, ConnectionSparsity], [R2, ActivationSparsity, SynapticOperations]],
        )
        results = benchmark.run(device=torch.device("cpu"), quiet=True)
        net._buffers["data_buffer"] = torch.zeros(1, net.input_dim)
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                preds.append(model(batch[0]).detach().cpu().reshape(-1, 2).numpy())
                labels.append(batch[1].detach().cpu().reshape(-1, 2).numpy())
        pred = np.concatenate(preds).astype(np.float64)
        label = np.concatenate(labels).astype(np.float64)
        pred_path = out_dir / f"{name}_pred.f64"
        label_path = out_dir / f"{name}_label.f64"
        pred_path.write_bytes(pred.reshape(-1).tobytes())
        label_path.write_bytes(label.reshape(-1).tobytes())
        row = {
            "file": name + ".mat",
            "md5": got,
            "bytes": mat.stat().st_size,
            "checkpoint_sha256": sha256(weight),
            "input_feature_size": int(dataset.input_feature_size),
            "n_test": int(label.shape[0]),
            "harness_r2": float(results["R2"]),
            "license": "CC-BY-4.0",
        }
        rows.append(row)
        print(f"{name} R2 {row['harness_r2']:.8f} n {row['n_test']}", flush=True)
        del dataset, loader, net, model, benchmark, preds, labels, pred, label
        gc.collect()
    mean = sum(row["harness_r2"] for row in rows) / len(rows)
    report = {
        "task": "primate_reaching",
        "protocol": "benchmark_2d_ann.py at e521c28",
        "neurobench": neurobench.__version__,
        "torch": torch.__version__,
        "train_ratio": 0.5,
        "bin_width_s": 0.004,
        "biological_delay": 0,
        "num_steps": 1,
        "held_out": True,
        "aggregate_mean_r2": mean,
        "sessions": rows,
        "note": "mean of six harness R2 values, the example's sum/len aggregation",
    }
    (WORK / "six_session_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"MEAN {mean:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
