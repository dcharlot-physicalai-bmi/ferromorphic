#!/usr/bin/env python3
"""Show that the analytic and convergence gates fail when the science, or the gate, is broken.

The specimen is `tools/scientific_gates/fixture.rs`. This harness copies it, compiles the copy with
`rustc`, and requires a clean run to exit 0. It then applies each mutation below to a fresh copy
and requires that program to exit non-zero. A mutation that does not apply, a clean run that fails,
or a mutated run that still passes, is a failure of this self-test.

The mutations are the ones that have already shipped as silent holes in this repository's history:
a tolerance widened until a wrong method passes, a closed form with the wrong sign, and an order
window moved until a first-order method is rejected or a nonconvergent one is accepted.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIXTURE = os.path.join(HERE, "scientific_gates", "fixture.rs")

# (label, old, new). `old` must occur exactly once.
MUTATIONS = [
    (
        "tolerance widened from 1e-12 to 1e-3",
        "const CLOSED_FORM_TOL: f64 = 1e-12;",
        "const CLOSED_FORM_TOL: f64 = 1e-3;",
    ),
    (
        "closed form grows instead of decaying",
        "v_rest + (v0 - v_rest) * (-t / tau).exp()",
        "v_rest + (v0 - v_rest) * (t / tau).exp()",
    ),
    (
        "order window demands fourth order of forward Euler",
        "const ORDER_MIN: f64 = 0.8;",
        "const ORDER_MIN: f64 = 3.0;",
    ),
    (
        "order gate accepts a nonconvergent stepper",
        "order >= ORDER_MIN && order <= ORDER_MAX",
        "order >= -1.0e9 && order <= ORDER_MAX",
    ),
    (
        "forward Euler drops the step length",
        "v + dt * (-(v - v_rest) / tau)",
        "v + (-(v - v_rest) / tau)",
    ),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def compile_and_run(src, work):
    bin_path = os.path.join(work, "fixture")
    # Compile in a temp directory, so rust-toolchain.toml is not visible. Pin the same
    # release the crate names, rather than whatever default rustup has installed.
    if os.environ.get("RUSTC"):
        cmd = [os.environ["RUSTC"]]
    else:
        cmd = ["rustup", "run", "1.98.1", "rustc"]
    compiled = subprocess.run(
        [*cmd, "--edition", "2024", "-O", src, "-o", bin_path],
        cwd=work,
        capture_output=True,
        text=True,
    )
    if compiled.returncode != 0:
        return compiled.returncode, compiled.stderr
    ran = subprocess.run([bin_path], cwd=work, capture_output=True, text=True)
    return ran.returncode, ran.stdout + ran.stderr


def apply_once(text, old, new):
    n = text.count(old)
    if n != 1:
        return None, n
    return text.replace(old, new, 1), 1


def main():
    if not os.path.isfile(FIXTURE):
        print(f"mutation_selftest: specimen missing: {FIXTURE}")
        return 1
    original = open(FIXTURE, encoding="utf-8").read()
    digest = sha256(FIXTURE)
    print(f"mutation_selftest: specimen sha256 {digest}")

    work = tempfile.mkdtemp(prefix="ferromorphic-mutation-")
    try:
        clean = os.path.join(work, "clean.rs")
        with open(clean, "w", encoding="utf-8") as f:
            f.write(original)
        code, out = compile_and_run(clean, work)
        if code != 0:
            print(f"mutation_selftest: CLEAN RUN FAILED ({code})")
            print(out[-4000:])
            return 1
        print("mutation_selftest: clean specimen passed")
        for line in out.splitlines():
            if line.startswith("MEASURED "):
                print("mutation_selftest:", line)

        failed_to_catch = []
        for label, old, new in MUTATIONS:
            mutated, n = apply_once(original, old, new)
            if mutated is None:
                print(f"mutation_selftest: NOT-APPLIED {label} ({n} matches)")
                failed_to_catch.append(label)
                continue
            path = os.path.join(work, "mut.rs")
            with open(path, "w", encoding="utf-8") as f:
                f.write(mutated)
            code, out = compile_and_run(path, work)
            if code == 0:
                print(f"mutation_selftest: SURVIVED {label}")
                print(out[-2000:])
                failed_to_catch.append(label)
            else:
                print(f"mutation_selftest: caught {label} (exit {code})")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        after = sha256(FIXTURE)
        if after != digest:
            print("mutation_selftest: specimen was modified; restoring is the harness's job and it failed")
            return 1

    if failed_to_catch:
        print(f"mutation_selftest: {len(failed_to_catch)} mutation(s) were not caught")
        return 1
    print(f"mutation_selftest: {len(MUTATIONS)} of {len(MUTATIONS)} mutations caught; specimen unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
