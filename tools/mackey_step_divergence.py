#!/usr/bin/env python3
"""Stepwise Mackey autonomous comparison. Not a search for the 40.803 comment.

Fits tau 17 and tau 30 the way tools/export_heldout_tables.py does, then records
every autonomous step from torch and from a NumPy dot product of the same
float64 weights. A later Rust replay of the torch inputs is a separate command.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "target" / "verification" / "mackey" / "example"))

from echo_state_network import EchoStateNetwork  # noqa: E402
from export_heldout_tables import (  # noqa: E402
    DATA_SHA,
    HP_SHA,
    MACKEY,
    echo_table,
    file_sha,
    series,
    write,
)
from neurobench.datasets import MackeyGlass  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402


def numpy_step(state, drive, win, w, wout, leak):
    aug = np.array([1.0, drive], dtype=np.float64)
    pre = w @ state + win @ aug
    nxt = (1.0 - leak) * state + leak * np.tanh(pre)
    feat = np.concatenate([aug, nxt])
    return float(wout @ feat), nxt


def torch_autonomous(esn, batch):
    esn.mode = "autonomous"
    esn.prior_prediction = None
    inputs, outputs, states = [], [], []
    state0 = esn.reservoir.detach().cpu().numpy().reshape(-1).copy()
    with torch.no_grad():
        for sample in batch:
            used = sample.detach().cpu().numpy().reshape(-1)[0]
            if esn.prior_prediction is not None:
                used = float(esn.prior_prediction.detach().cpu().numpy().reshape(-1)[0])
            inputs.append(used)
            pred = esn.single_forward(sample if esn.prior_prediction is None else esn.prior_prediction)
            esn.prior_prediction = pred
            outputs.append(float(pred.detach().cpu().numpy().reshape(-1)[0]))
            states.append(esn.reservoir.detach().cpu().numpy().reshape(-1).copy())
    return state0, np.array(inputs), np.array(outputs), states


def numpy_autonomous(state, drives_first, win, w, wout, leak, n):
    x = state.copy()
    drive = drives_first
    inputs, outputs, states = [], [], []
    for _ in range(n):
        inputs.append(drive)
        y, x = numpy_step(x, drive, win, w, wout, leak)
        outputs.append(y)
        states.append(x.copy())
        drive = y
    return np.array(inputs), np.array(outputs), states


def first_gap(a, b, tol):
    for i, (x, y) in enumerate(zip(a, b)):
        if abs(x - y) > tol:
            return i, abs(x - y)
    return None, float(np.max(np.abs(a - b))) if len(a) else 0.0


def main() -> int:
    import pandas as pd

    hp = MACKEY / "example" / "model_data" / "echo_state_network_hyperparameters.csv"
    archive = MACKEY / "data.tar.gz"
    if file_sha(hp) != HP_SHA or file_sha(archive) != DATA_SHA:
        raise SystemExit("mackey pins do not match")
    params = pd.read_csv(hp)
    out = ROOT / "target" / "verification" / "heldout_tables" / "divergence"
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for tau in (17, 30):
        mg = MackeyGlass(str(MACKEY / "data" / f"mg_{tau}.npy"), start_offset=0.0, bin_window=1, download=False)
        train_set = Subset(mg, mg.ind_train)
        test_set = Subset(mg, mg.ind_test)
        row = int(params.index[params["tau"] == tau].tolist()[0])
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
        train_data, train_labels = train_set[:]
        warmup = round(0.6 * mg.pts_per_lyaptime)
        esn.fit(train_data, train_labels[warmup:], warmup)
        state = esn.reservoir.detach().clone()
        loader = DataLoader(test_set, batch_size=mg.testtime_pts, shuffle=False)
        batch = next(iter(loader))[0]
        esn.reservoir = state.clone()
        state0, t_in, t_out, t_states = torch_autonomous(esn, batch)
        win = esn.Win.weight.detach().cpu().numpy().astype(np.float64)
        w = esn.W.weight.detach().cpu().numpy().astype(np.float64)
        wout = esn.Wout.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
        leak = float(params["leakage"][row])
        n_in, n_out, n_states = numpy_autonomous(state0, float(t_in[0]), win, w, wout, leak, len(t_out))
        x = state0.copy()
        same_out = []
        same_state = []
        for drive in t_in:
            y, x = numpy_step(x, float(drive), win, w, wout, leak)
            same_out.append(y)
            same_state.append(np.max(np.abs(x - t_states[len(same_out) - 1])))
        same_out = np.array(same_out)
        split_closed, closed_gap = first_gap(t_out, n_out, 1e-8)
        split_same, same_gap = first_gap(t_out, same_out, 1e-12)
        state_gap = float(np.max(same_state))
        prefix = out / f"tau{tau}"
        weights = echo_table(
            int(win.shape[0]), int(win.shape[1]), 1, leak, True, True,
            win, w, wout.reshape(1, -1), state0,
        )
        write(prefix.with_suffix(".echo"), weights)
        write(Path(str(prefix) + ".replay-in"), series("f64", "mackey-replay-input", t_in))
        write(Path(str(prefix) + ".replay-out"), series("f64", "mackey-replay-output", t_out))
        row_out = {
            "tau": tau,
            "units": int(win.shape[0]),
            "n": int(len(t_out)),
            "torch_vs_numpy_closed_split": split_closed,
            "torch_vs_numpy_closed_gap": closed_gap,
            "torch_vs_numpy_same_input_split_1e-12": split_same,
            "torch_vs_numpy_same_input_max": float(np.max(np.abs(t_out - same_out))),
            "torch_vs_numpy_same_input_state_max": state_gap,
            "torch_output_max": float(np.max(np.abs(t_out))),
            "torch_state_max": float(max(np.max(np.abs(s)) for s in t_states)),
            "leakage": leak,
        }
        report.append(row_out)
        print(json.dumps(row_out), flush=True)
        del n_in, n_states
    (out / "numpy_vs_torch.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
