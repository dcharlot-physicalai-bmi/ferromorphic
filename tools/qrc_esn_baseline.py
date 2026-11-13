#!/usr/bin/env python3
"""External check of one classical ESN setting from arXiv:2610.14454.

Owners: Phi Hung Nguyen and Daniel Soh, University of Arizona.
Paper: https://arxiv.org/abs/2610.14454
This script does not implement their quantum reservoir. It checks the echo-state
baseline they publish at m=4, setting (spectral radius 0.7, leak 1, input scale
0.02, bias off), five-step Mackey-Glass, ridge 1e-10. Their Table 19 mean is
over 10,000 draws whose generator is not stated here.

The delay history is also unstated. x(t)=1 for every t<=0 is the equilibrium of
their ODE, so the series they quote (training-target std 0.2331) cannot be that
history. The run below uses a constant delay history of 0.9 and x(0)=1, which
makes the training-target std 0.2330. That history is an assumption, not a line
in the paper. NumPy seeds 0..39 are likewise local. This is not the NeuroBench
Mackey protocol and it is not a reproduction of their 10,000-draw table.
"""
from __future__ import annotations

import json

import numpy as np


def mackey(history_value: float = 0.9, x0: float = 1.0, discard: int = 1000, n_keep: int = 3000, delay: int = 17) -> np.ndarray:
    n = discard + n_keep
    x = np.empty(n + 1)
    x[0] = x0
    hist = np.full(delay, history_value)
    for i in range(n):
        delayed = hist[0]
        hist[:-1] = hist[1:]
        hist[-1] = x[i]
        dx = 0.2 * delayed / (1.0 + delayed**10) - 0.1 * x[i]
        x[i + 1] = x[i] + dx
    return x[discard + 1 : discard + 1 + n_keep]


def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    return rmse / float(np.std(target))


def one_draw(series: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m, rho, leak, scale = 4, 0.7, 1.0, 0.02
    weights = rng.normal(size=(m, m))
    weights *= rho / np.max(np.abs(np.linalg.eigvals(weights)))
    win = rng.uniform(-scale, scale, size=m)
    state = np.zeros(m)
    feats = np.empty((series.size, m))
    for t, sample in enumerate(series):
        feats[t] = state
        pre = weights @ state + win * sample
        state = (1.0 - leak) * state + leak * np.tanh(pre)
    horizon, wash, train_n, test_n = 5, 100, 2000, 895

    def pack(start: int, count: int) -> tuple[np.ndarray, np.ndarray]:
        block = np.concatenate([feats[start : start + count], np.ones((count, 1))], axis=1)
        target = series[start + horizon : start + horizon + count]
        return block, target

    train_x, train_y = pack(wash, train_n)
    test_x, test_y = pack(wash + train_n, test_n)
    gram = train_x.T @ train_x + 1e-10 * np.eye(m + 1)
    readout = np.linalg.solve(gram, train_x.T @ train_y)
    return nrmse(train_x @ readout, train_y), nrmse(test_x @ readout, test_y)


def main() -> None:
    series = mackey()
    train_target = series[105:2105]
    train, held = [], []
    for seed in range(40):
        a, b = one_draw(series, seed)
        train.append(a)
        held.append(b)
    train_a = np.asarray(train)
    held_a = np.asarray(held)
    print(json.dumps({
        "paper": "arXiv:2610.14454",
        "assumption_delay_history": 0.9,
        "x0": 1.0,
        "train_target_std": float(np.std(train_target)),
        "paper_train_target_std": 0.2331,
        "draws": 40,
        "seeds": "numpy default_rng 0..39",
        "setting": [0.7, 1.0, 0.02, 0],
        "train_nrmse_mean": float(train_a.mean()),
        "held_nrmse_mean": float(held_a.mean()),
        "held_nrmse_std": float(held_a.std(ddof=1)),
        "paper_table19_m4_train": 0.2568,
        "paper_table19_m4_held_mean": 0.2516,
        "paper_table19_m4_held_std": 0.0418,
    }, indent=2))


if __name__ == "__main__":
    main()
