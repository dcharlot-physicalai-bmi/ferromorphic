#!/usr/bin/env python3
"""Export pinned Mackey–Glass and Primate Reaching weights for Rust inference.

The files are ferromorphic-table 1: a text header, then a little-endian
row-major payload. Nothing in them is executable. Fit uses the training split
only. The test split is written as a drive series and as the upstream
prediction, and is not used to change a weight.

Mackey–Glass follows tools/neurobench_mackey.py at NeuroBench e521c28:
MackeyGlass, seed_id 0, the hyperparameter row, abs(w[0]) inside
EchoStateNetwork, post-fit reservoir, autonomous forecast, and a separate
single_step forecast from the same state. The 40.803 comment is not a target.

Primate Reaching follows tools/neurobench_primate_all.py: the six official
mats, the 2D ANN checkpoints at that pin, eval mode, one initial zero row,
bin window 0.2 / 0.004 = 50.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

NB_PIN = "e521c28f5798cb40bb9423758bf066dad53fe070"
HP_SHA = "1bece9503c92b16c68ad9bb260cea55d3f8ec24af6172dfcb85ba5153c848c52"
DATA_SHA = "5e7c2b62a5135b744e7b49f103f58a2d74c70fe2dae67890cd5f93a6110fdf7c"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "target" / "verification" / "heldout_tables"
MACKEY = ROOT / "target" / "verification" / "mackey"
PRIMATE = ROOT / "target" / "verification" / "primate"
RAW = f"https://raw.githubusercontent.com/NeuroBench/neurobench/{NB_PIN}/examples/primate_reaching/"
FILES = {
    "indy_20160622_01": "c33d5fff31320d709d23fe445561fb6e",
    "indy_20160630_01": "197413a5339630ea926cbd22b8b43338",
    "indy_20170131_02": "2790b1c869564afaa7772dbf9e42d784",
    "loco_20170210_03": "4cae63b58c4cb9c8abd44929216c703b",
    "loco_20170215_02": "739b70762d838f3a1f358733c426bb02",
    "loco_20170301_05": "47342da09f9c950050c9213c3df38ea3",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finish(header: str, payload: bytes) -> bytes:
    text = f"{header}sha256 {sha256(payload)}\n\n"
    return text.encode("ascii") + payload


def f64_payload(arrays) -> bytes:
    import numpy as np

    parts = []
    for array in arrays:
        values = np.ascontiguousarray(array, dtype="<f8")
        if not np.isfinite(values).all():
            raise SystemExit("non-finite f64 weight")
        parts.append(values.tobytes(order="C"))
    return b"".join(parts)


def f32_payload(arrays) -> bytes:
    import numpy as np

    parts = []
    for array in arrays:
        values = np.ascontiguousarray(array, dtype="<f4")
        if not np.isfinite(values).all():
            raise SystemExit("non-finite f32 weight")
        parts.append(values.tobytes(order="C"))
    return b"".join(parts)


def echo_table(units, in_width, n_out, leakage, include_bias, include_input, w_in, w, w_out, state) -> bytes:
    payload = f64_payload([w_in, w, w_out, state])
    feature = (in_width + units) if include_input else units
    values = units * in_width + units * units + n_out * feature + units
    if len(payload) != values * 8:
        raise SystemExit(f"echo payload {len(payload)} != {values * 8}")
    bits = int.from_bytes(struct.pack("<d", float(leakage)), "little")
    header = (
        "ferromorphic-table 1\n"
        "kind echo\n"
        "endian le\n"
        "dtype f64\n"
        "order row-major\n"
        f"units {units}\n"
        f"in_width {in_width}\n"
        f"n_out {n_out}\n"
        f"leakage_bits {bits:016x}\n"
        f"include_bias {int(include_bias)}\n"
        f"include_input {int(include_input)}\n"
        f"values {values}\n"
    )
    return finish(header, payload)


def ann_table(tensors, eps: float) -> bytes:
    order = [
        "fc1.weight", "fc1.bias",
        "batchnorm1.weight", "batchnorm1.bias", "batchnorm1.running_mean", "batchnorm1.running_var",
        "fc2.weight", "fc2.bias",
        "batchnorm2.weight", "batchnorm2.bias", "batchnorm2.running_mean", "batchnorm2.running_var",
        "fc3.weight", "fc3.bias",
    ]
    arrays = [tensors[name].detach().cpu().numpy() for name in order]
    payload = f32_payload(arrays)
    input_dim = arrays[0].shape[1]
    hidden1 = arrays[0].shape[0]
    hidden2 = arrays[6].shape[0]
    output_dim = arrays[12].shape[0]
    window = 50
    values = sum(array.size for array in arrays)
    if len(payload) != values * 4:
        raise SystemExit("ann payload length")
    eps_bits = int.from_bytes(struct.pack("<f", float(eps)), "little")
    header = (
        "ferromorphic-table 1\n"
        "kind ann\n"
        "endian le\n"
        "dtype f32\n"
        "order row-major\n"
        f"input_dim {input_dim}\n"
        f"hidden1 {hidden1}\n"
        f"hidden2 {hidden2}\n"
        f"output_dim {output_dim}\n"
        f"window {window}\n"
        f"eps_bits {eps_bits:08x}\n"
        f"values {values}\n"
    )
    return finish(header, payload)


def series(kind_dtype: str, role: str, array) -> bytes:
    import numpy as np

    if kind_dtype == "f64":
        values = np.ascontiguousarray(array, dtype="<f8").reshape(-1)
        payload = values.tobytes(order="C")
        width = 8
        dtype = "f64"
    else:
        values = np.ascontiguousarray(array, dtype="<f4").reshape(-1)
        payload = values.tobytes(order="C")
        width = 4
        dtype = "f32"
    if not np.isfinite(values).all():
        raise SystemExit(f"non-finite series {role}")
    header = (
        "ferromorphic-table 1\n"
        "kind series\n"
        "endian le\n"
        f"dtype {dtype}\n"
        "order row-major\n"
        f"role {role}\n"
        f"len {values.size}\n"
    )
    blob = finish(header, payload)
    if len(payload) != values.size * width:
        raise SystemExit("series length")
    return blob


def write(path: Path, blob: bytes) -> dict:
    path.write_bytes(blob)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(blob), "bytes": len(blob)}


def export_mackey() -> list:
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Subset

    sys.path.insert(0, str(MACKEY / "example"))
    from echo_state_network import EchoStateNetwork
    from neurobench.datasets import MackeyGlass

    hp = MACKEY / "example" / "model_data" / "echo_state_network_hyperparameters.csv"
    if file_sha(hp) != HP_SHA:
        raise SystemExit(f"hyperparameter sha {file_sha(hp)}")
    archive = MACKEY / "data.tar.gz"
    if file_sha(archive) != DATA_SHA:
        raise SystemExit(f"data sha {file_sha(archive)}")
    params = pd.read_csv(hp)
    rows = []
    for tau in range(17, 31):
        mg = MackeyGlass(str(MACKEY / "data" / f"mg_{tau}.npy"), start_offset=0.0, bin_window=1, download=False)
        train_set = Subset(mg, mg.ind_train)
        test_set = Subset(mg, mg.ind_test)
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
        esn.train()
        train_data, train_labels = train_set[:]
        warmup = round(0.6 * mg.pts_per_lyaptime)
        esn.fit(train_data, train_labels[warmup:], warmup)
        state = esn.reservoir.detach().clone()
        loader = DataLoader(test_set, batch_size=mg.testtime_pts, shuffle=False)
        batch = next(iter(loader))
        esn.mode = "autonomous"
        esn.prior_prediction = None
        with torch.no_grad():
            auto = esn(batch[0]).detach().cpu().numpy().reshape(-1)
        esn.reservoir = state.clone()
        esn.mode = "single_step"
        esn.prior_prediction = None
        with torch.no_grad():
            teacher = esn(batch[0]).detach().cpu().numpy().reshape(-1)
        drive = batch[0].detach().cpu().numpy().reshape(-1)
        label = batch[1].detach().cpu().numpy().reshape(-1)
        win = esn.Win.weight.detach().cpu().numpy()
        wrec = esn.W.weight.detach().cpu().numpy()
        wout = esn.Wout.weight.detach().cpu().numpy()
        column = state.detach().cpu().numpy().reshape(-1)
        prefix = OUT / f"tau{tau}"
        record = {
            "tau": tau,
            "reservoir_size": int(win.shape[0]),
            "leakage": float(params["leakage"][row]),
            "warmup_pts": int(warmup),
            "n_test": int(drive.shape[0]),
            "echo": write(prefix.with_suffix(".echo"), echo_table(
                int(win.shape[0]), int(win.shape[1]), 1, float(params["leakage"][row]),
                True, True, win, wrec, wout, column,
            )),
            "drive": write(Path(str(prefix) + ".drive"), series("f64", "mackey-drive", drive)),
            "auto": write(Path(str(prefix) + ".auto"), series("f64", "mackey-pred", auto)),
            "teacher": write(Path(str(prefix) + ".teacher"), series("f64", "mackey-teacher", teacher)),
            "label": write(Path(str(prefix) + ".label"), series("f64", "mackey-label", label)),
        }
        prior = MACKEY / "heldout" / f"tau{tau}_pred.f64"
        if prior.exists():
            old = np.frombuffer(prior.read_bytes(), dtype="<f8")
            record["max_abs_vs_prior_replay_file"] = float(np.max(np.abs(old - auto)))
        rows.append(record)
        print(f"tau {tau} units {record['reservoir_size']} n {record['n_test']}", flush=True)
        del esn, mg, train_data, train_labels
    return rows


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_reaching() -> list:
    import gc
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    example = PRIMATE / "example"
    example.mkdir(parents=True, exist_ok=True)
    ann_py = example / "ANN.py"
    if not ann_py.exists():
        urllib.request.urlretrieve(RAW + "ANN.py", ann_py)
    sys.path.insert(0, str(example))
    from ANN import ANNModel2D
    from neurobench.datasets import PrimateReaching
    from neurobench.models import TorchModel

    rows = []
    for name, expect in FILES.items():
        mat = PRIMATE / "data" / f"{name}.mat"
        if md5(mat) != expect:
            raise SystemExit(f"md5 {name}")
        weight = example / f"{name}_model_state_dict.pth"
        if not weight.exists():
            urllib.request.urlretrieve(RAW + f"model_data/2D_ANN_Weight/{name}_model_state_dict.pth", weight)
        dataset = PrimateReaching(
            file_path=str(PRIMATE / "data"),
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
        if float(net.batchnorm1.eps) != 1e-5 or int(net.bin_window_size) != 50:
            raise SystemExit(f"unexpected bn eps or window for {name}")
        model = TorchModel(net)
        net._buffers["data_buffer"] = torch.zeros(1, net.input_dim)
        frames, preds, labels = [], [], []
        with torch.no_grad():
            for batch in loader:
                frames.append(batch[0].detach().cpu().reshape(-1, net.input_dim).numpy())
                preds.append(model(batch[0]).detach().cpu().reshape(-1, 2).numpy())
                labels.append(batch[1].detach().cpu().reshape(-1, 2).numpy())
        frame = np.concatenate(frames)
        pred = np.concatenate(preds)
        label = np.concatenate(labels)
        prefix = OUT / name
        record = {
            "file": name + ".mat",
            "md5": expect,
            "checkpoint_sha256": file_sha(weight),
            "input_dim": int(net.input_dim),
            "window": 50,
            "eps": 1e-5,
            "n_test": int(pred.shape[0]),
            "ann": write(prefix.with_suffix(".ann"), ann_table(net.state_dict(), float(net.batchnorm1.eps))),
            "frames": write(Path(str(prefix) + ".frames"), series("f32", "reaching-frames", frame)),
            "pred": write(Path(str(prefix) + ".pred"), series("f32", "reaching-pred", pred)),
            "label": write(Path(str(prefix) + ".label"), series("f32", "reaching-label", label)),
        }
        rows.append(record)
        print(f"{name} frames {frame.shape} pred {pred.shape}", flush=True)
        del dataset, loader, net, model, frames, preds, labels, frame, pred, label
        gc.collect()
    return rows


def main() -> int:
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    manifest = {
        "format": "ferromorphic-table 1",
        "executable": False,
        "endian": "le",
        "order": "row-major",
        "neurobench_pin": NB_PIN,
        "torch": torch.__version__,
        "fit_on_held_out": False,
        "mackey_hyperparameter_sha256": HP_SHA,
        "mackey_data_sha256": DATA_SHA,
        "eig_scale": "abs(w[0]) as in EchoStateNetwork, not a search for the 40.803 comment",
    }
    if which in ("mackey", "both"):
        manifest["mackey"] = export_mackey()
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if which in ("reaching", "both"):
        if (OUT / "manifest.json").exists() and which == "reaching":
            manifest = json.loads((OUT / "manifest.json").read_text())
        manifest["reaching"] = export_reaching()
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("WROTE", OUT / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
