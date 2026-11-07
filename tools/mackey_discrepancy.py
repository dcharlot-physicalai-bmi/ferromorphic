#!/usr/bin/env python3
"""Controlled Mackey-Glass sMAPE checks. Not a retune.

Protocol is NeuroBench commit e521c28, example esn_benchmark.py, repeat=1:
seed_id 0, start_offset 0, tau 17..30, hyperparameters sha256
1bece9503c92b16c68ad9bb260cea55d3f8ec24af6172dfcb85ba5153c848c52,
data.tar.gz sha256 5e7c2b62a5135b744e7b49f103f58a2d74c70fe2dae67890cd5f93a6110fdf7c.

The 40.803375080880265 comment was added in ea4207f3d072 (2023-10-09) while
poetry.lock pinned torch 2.0.1, labeled repeat=1 and single_series=False.
The eigenvalue line has been `abs(w[0])` since that date. This script runs
that line and, separately, a scale by the maximum modulus. The second arm is
a hypothesis test. It is not the example.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "target" / "verification" / "mackey" / "example"
DATA = ROOT / "target" / "verification" / "mackey" / "data"
sys.path.insert(0, str(EXAMPLE))
from echo_state_network import EchoStateNetwork  # noqa: E402

COMMENT = 40.803375080880265
HP_SHA = "1bece9503c92b16c68ad9bb260cea55d3f8ec24af6172dfcb85ba5153c848c52"


def series(tau: int):
    import numpy as np

    raw = np.load(DATA / f"mg_{tau}.npy")
    soln = torch.tensor(raw[:1501], dtype=torch.float64).unsqueeze(-1)
    train_x = soln[:750].unsqueeze(1)  # (750, 1, 1), bin_window 1
    train_y = soln[1:751]
    test_x = soln[750:1500].unsqueeze(1)
    test_y = soln[751:1501]
    return train_x, train_y, test_x, test_y


def smape(pred: torch.Tensor, truth: torch.Tensor) -> float:
    value = 200 * torch.mean(torch.abs(pred - truth) / (torch.abs(pred) + torch.abs(truth)))
    return float(torch.nan_to_num(value, nan=200.0))


def run(scale: str) -> dict:
    import hashlib
    import pandas as pd

    hp = EXAMPLE / "model_data" / "echo_state_network_hyperparameters.csv"
    digest = hashlib.sha256(hp.read_bytes()).hexdigest()
    if digest != HP_SHA:
        raise SystemExit(f"hyperparameter sha {digest}")
    params = pd.read_csv(hp)
    real = torch.linalg.eig
    rows = []

    def wrapped(matrix):
        values, vectors = real(matrix)
        modulus = torch.abs(values)
        wrapped.stats = {
            "first_modulus": float(modulus[0]),
            "max_modulus": float(modulus.max()),
            "argmax": int(torch.argmax(modulus)),
        }
        if scale == "max_abs":
            index = wrapped.stats["argmax"]
            values = torch.cat((values[index : index + 1], values[:index], values[index + 1 :]))
        return values, vectors

    wrapped.stats = {}
    torch.linalg.eig = wrapped
    try:
        for tau in range(17, 31):
            row = int(params.index[params["tau"] == tau].tolist()[0])
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
            seen = dict(wrapped.stats)
            train_x, train_y, test_x, test_y = series(tau)
            esn.train()
            esn.fit(train_x, train_y[round(0.6 * 75) :], round(0.6 * 75))
            esn.prior_prediction = None
            with torch.no_grad():
                pred = esn(test_x)
            score = smape(pred.reshape(-1), test_y.reshape(-1))
            rows.append({"tau": tau, "smape": score, **seen})
            print(
                f"{scale} tau {tau} smape {score:.8f} argmax {seen.get('argmax')} "
                f"first {seen.get('first_modulus')} max {seen.get('max_modulus')}",
                flush=True,
            )
    finally:
        torch.linalg.eig = real
    mean = sum(row["smape"] for row in rows) / len(rows)
    return {
        "scale": scale,
        "torch": torch.__version__,
        "mean_smape": mean,
        "comment_repeat1": COMMENT,
        "mean_minus_comment": mean - COMMENT,
        "rows": rows,
    }


def main() -> int:
    arms = sys.argv[1:] or ["first", "max_abs"]
    out = [run(scale) for scale in arms]
    dest = Path(os.environ.get("MACKEY_DISCREPANCY_OUT", "/tmp/mackey_discrepancy.json"))
    dest.write_text(json.dumps(out, indent=2) + "\n")
    for item in out:
        print(f"MEAN {item['scale']} torch {item['torch']} {item['mean_smape']:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
