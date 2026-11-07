#!/usr/bin/env python3
"""Official NIR HDF5 interchange, outside the std-only crate.

Reads neuromorphs/NIR at the pinned revision when the `nir` package is installed.
Exit 0 only after a write/read round trip through that package. Exit 2 when the
package is absent: that is UNAVAILABLE, not a pass.

The pin is commit 563c58b45873429e3b1b6f71763ec334ceb981d5, SPDX BSD-3-Clause.
This script does not vendor that source.
"""
import sys

PIN = "563c58b45873429e3b1b6f71763ec334ceb981d5"


def main() -> int:
    try:
        import nir  # noqa: F401
    except ImportError:
        print(f"UNAVAILABLE nir package (pin {PIN})")
        print("install the package built from that commit, then re-run this script")
        return 2
    import nir as nir_mod

    graph = nir_mod.NIRGraph(
        nodes={
            "in": nir_mod.Input(input_type=[2]),
            "li": nir_mod.CubaLI(
                tau_syn=[1e-3, 1e-3],
                tau_mem=[2e-2, 2e-2],
                r=[1.0, 1.0],
                v_leak=[0.0, 0.0],
                w_in=[1.0, 1.0],
            ),
            "out": nir_mod.Output(output_type=[2]),
        },
        edges=[("in", "li"), ("li", "out")],
    )
    path = "target/verification/oracle.nir"
    import os

    os.makedirs("target/verification", exist_ok=True)
    nir_mod.write(path, graph)
    back = nir_mod.read(path)
    if type(back.nodes["li"]).__name__ != "CubaLI":
        print("FAIL round trip dropped CubaLI")
        return 1
    print(f"PASS hdf5 round trip pin {PIN} file {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
