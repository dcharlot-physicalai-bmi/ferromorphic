#!/usr/bin/env python3
"""One held-out Primate Reaching file against the pinned NeuroBench harness.

File: indy_20170131_02.mat from Zenodo record 583331, license CC-BY-4.0,
md5 2790b1c869564afaa7772dbf9e42d784. The other five official files are not
downloaded. The checkpoint is the 2D ANN state dict shipped in the NeuroBench
example at commit e521c28f5798cb40bb9423758bf066dad53fe070 (Apache-2.0).

This is one session of one model. It is not the six-file average printed in
benchmark_2d_ann.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

NB_PIN = "e521c28f5798cb40bb9423758bf066dad53fe070"
FILENAME = "indy_20170131_02"
MD5 = "2790b1c869564afaa7772dbf9e42d784"
ZENODO = "https://zenodo.org/records/583331/files/indy_20170131_02.mat?download=1"
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "target" / "verification" / "primate"
RAW = f"https://raw.githubusercontent.com/NeuroBench/neurobench/{NB_PIN}/examples/primate_reaching/"


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)


def main() -> int:
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
        import neurobench
    except ImportError as exc:
        print(f"UNAVAILABLE {exc}")
        return 2
    mat = WORK / "data" / f"{FILENAME}.mat"
    if not mat.exists():
        staged = Path("/tmp/primate") / f"{FILENAME}.mat"
        if staged.exists() and md5(staged) == MD5:
            mat.parent.mkdir(parents=True, exist_ok=True)
            os.link(staged, mat) if False else mat.write_bytes(staged.read_bytes())
        else:
            fetch(ZENODO, mat)
    got = md5(mat)
    if got != MD5:
        print(f"FAIL md5 {got}")
        return 1
    example = WORK / "example"
    fetch(RAW + "ANN.py", example / "ANN.py")
    fetch(
        RAW + f"model_data/2D_ANN_Weight/{FILENAME}_model_state_dict.pth",
        example / f"{FILENAME}_model_state_dict.pth",
    )
    sys.path.insert(0, str(example))
    from ANN import ANNModel2D
    from neurobench.benchmarks import Benchmark
    from neurobench.datasets import PrimateReaching
    from neurobench.metrics.static import ConnectionSparsity, Footprint
    from neurobench.metrics.workload import ActivationSparsity, R2, SynapticOperations
    from neurobench.models import TorchModel

    dataset = PrimateReaching(
        file_path=str(WORK / "data"),
        filename=FILENAME,
        num_steps=1,
        train_ratio=0.5,
        bin_width=0.004,
        biological_delay=0,
        remove_segments_inactive=False,
        download=False,
    )
    loader = DataLoader(Subset(dataset, dataset.ind_test), batch_size=256, shuffle=False)
    device = torch.device("cpu")
    net = ANNModel2D(
        input_dim=dataset.input_feature_size,
        layer1=32,
        layer2=48,
        output_dim=2,
        bin_window=0.2,
        drop_rate=0.5,
    )
    net.load_state_dict(torch.load(example / f"{FILENAME}_model_state_dict.pth", map_location=device))
    net.eval()
    model = TorchModel(net)
    benchmark = Benchmark(
        model,
        loader,
        [],
        [],
        [[Footprint, ConnectionSparsity], [R2, ActivationSparsity, SynapticOperations]],
    )
    results = benchmark.run(device=device, quiet=True)
    # The network keeps a spike buffer across batches. The harness started from the
    # zero buffer created in ANNModel2D. Restore that buffer before replaying.
    net._buffers["data_buffer"] = torch.zeros(1, net.input_dim)
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch[0])
            preds.append(out.detach().cpu().reshape(-1, 2).numpy())
            labels.append(batch[1].detach().cpu().reshape(-1, 2).numpy())
    import numpy as np

    pred = np.concatenate(preds).astype(np.float64)
    label = np.concatenate(labels).astype(np.float64)
    out_dir = WORK / "heldout"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{FILENAME}_pred.f64"
    label_path = out_dir / f"{FILENAME}_label.f64"
    pred_path.write_bytes(struct.pack("<" + "d" * pred.size, *pred.reshape(-1).tolist()))
    label_path.write_bytes(struct.pack("<" + "d" * label.size, *label.reshape(-1).tolist()))
    proc = subprocess.run(
        [
            "cargo", "run", "--offline", "--quiet", "--example", "r2_columns", "--",
            str(pred_path), str(label_path),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        print("FAIL r2_columns")
        return 1
    rust = float(proc.stdout.strip())
    harness = float(results["R2"])
    report = {
        "task": "primate_reaching",
        "file": FILENAME + ".mat",
        "md5": MD5,
        "license": "CC-BY-4.0",
        "zenodo": "10.5281/zenodo.583331",
        "model": "2D ANN state dict from the NeuroBench example",
        "neurobench_pin": NB_PIN,
        "neurobench_version": neurobench.__version__,
        "torch": torch.__version__,
        "split": "ind_test",
        "train_ratio": 0.5,
        "bin_width_s": 0.004,
        "biological_delay": 0,
        "harness_r2": harness,
        "rust_r2": rust,
        "abs_diff": abs(harness - rust),
        "n": int(label.shape[0]),
        "held_out": True,
        "sessions_run": [FILENAME],
        "sessions_not_run": [
            "indy_20160622_01",
            "indy_20160630_01",
            "loco_20170210_03",
            "loco_20170215_02",
            "loco_20170301_05",
        ],
    }
    (WORK / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if abs(harness - rust) > 1e-6:
        print("FAIL rust and harness R2 differ")
        return 1
    print("PASS primate one-file held-out R2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
