#!/usr/bin/env python3
"""Export the pinned tau-30 reservoir and the official autonomous trajectory.

NeuroBench e521c28, torch 2.7.1, seed 0, the hyperparameter row for tau 30.
The files are the inputs of examples/closed_loop_sweep. They are not a score.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MACKEY = ROOT / "target" / "verification" / "mackey"
OUT = ROOT / "target" / "verification" / "closed_loop" / "tau30"


def main() -> int:
    sys.path.insert(0, str(MACKEY / "example"))
    import pandas as pd
    from echo_state_network import EchoStateNetwork
    from neurobench.datasets import MackeyGlass

    params = pd.read_csv(MACKEY / "example" / "model_data" / "echo_state_network_hyperparameters.csv")
    row = int(params.index[params["tau"] == 30].tolist()[0])
    mg = MackeyGlass(str(MACKEY / "data" / "mg_30.npy"), start_offset=0.0, bin_window=1, download=False)
    series = mg.mackeyglass_soln.detach().cpu().numpy().reshape(-1)
    train = mg.ind_train.detach().cpu().numpy()
    test = mg.ind_test.detach().cpu().numpy()
    train_u = series[train]
    train_y = series[train + 1]
    test_u = series[test]
    test_y = series[test + 1]
    warmup = round(0.6 * mg.pts_per_lyaptime)
    esn = EchoStateNetwork(
        in_channels=1,
        reservoir_size=int(params["reservoir_size"][row]),
        input_scale=torch.tensor([params["scale_bias"][row], params["scale_input"][row]], dtype=torch.float64),
        connect_prob=float(params["connect_prob"][row]),
        spectral_radius=float(params["spectral_radius"][row]),
        leakage=float(params["leakage"][row]),
        ridge_param=float(params["ridge_param"][row]),
        seed_id=0,
    )
    esn.train()
    esn.fit(torch.tensor(train_u, dtype=torch.float64).view(-1, 1), torch.tensor(train_y[warmup:], dtype=torch.float64).view(-1, 1), warmup)
    esn.mode = "autonomous"
    esn.prior_prediction = None
    batch = torch.tensor(test_u, dtype=torch.float64).view(-1, 1, 1)
    with torch.no_grad():
        pred = esn(batch).detach().cpu().numpy().reshape(-1)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "w_in.npy", esn.Win.weight.detach().cpu().numpy())
    np.save(OUT / "w.npy", esn.W.weight.detach().cpu().numpy())
    np.save(OUT / "train_u.npy", np.ascontiguousarray(train_u))
    np.save(OUT / "train_y.npy", np.ascontiguousarray(train_y))
    np.save(OUT / "test_u.npy", np.ascontiguousarray(test_u))
    np.save(OUT / "test_y.npy", np.ascontiguousarray(test_y))
    np.save(OUT / "reference.npy", np.ascontiguousarray(pred))
    (OUT / "meta.txt").write_text(
        f"leakage {float(params['leakage'][row])}\nwarmup {warmup}\nridge {float(params['ridge_param'][row])}\n"
    )
    print(f"wrote {OUT} pred {pred.shape} train {train_u.shape} warmup {warmup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
