#!/usr/bin/env python3
"""Download the inputs tools/export_heldout_tables.py refuses to invent.

Mackey–Glass is the Hugging Face archive checked by tools/neurobench_mackey.py.
The six Primate Reaching mats are Zenodo 583331, CC-BY-4.0, checked by MD5.
Neither tree is committed. A clean cargo test fails without them because
tests/heldout_tables.rs reads target/verification/heldout_tables/manifest.json.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from neurobench_mackey import prepare

ROOT = Path(__file__).resolve().parents[1]
PRIMATE = ROOT / "target" / "verification" / "primate" / "data"
ZENODO = "https://zenodo.org/records/583331/files/{name}.mat?download=1"
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


def fetch_mat(name: str, expect: str) -> None:
    dest = PRIMATE / f"{name}.mat"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and md5(dest) == expect:
        print(f"have {name}")
        return
    part = dest.with_suffix(".mat.part")
    print(f"fetch {name}", flush=True)
    urllib.request.urlretrieve(ZENODO.format(name=name), part)
    got = md5(part)
    if got != expect:
        part.unlink(missing_ok=True)
        raise SystemExit(f"md5 {name} {got}")
    part.replace(dest)
    print(f"ok {name}")


def main() -> int:
    prepare()
    print("mackey archive checked")
    for name, expect in FILES.items():
        fetch_mat(name, expect)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
