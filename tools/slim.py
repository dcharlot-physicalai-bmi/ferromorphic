#!/usr/bin/env python3
"""Make a slim copy of this crate holding ONE module and what it needs, for mutation runs.

    python3 tools/slim.py . /tmp/slim_nef nef
    python3 tools/mutate.py --root /tmp/slim_nef nef

`tools/mutate.py` rebuilds the crate once per mutation, and building all sixty-six modules takes
about a hundred seconds. A copy holding only `nef` and the modules it names through `crate::`
builds in about six, which is the difference between auditing one module in an afternoon and
auditing nine at once.

What it copies: `src/<module>.rs`, the transitive closure of the `crate::<name>` paths found in it
(comments stripped first, so a module mentioned only in prose is not dragged in), `Cargo.toml`
with its `[[example]]` sections removed, and `Cargo.lock`. It writes a `src/lib.rs` holding the
original's attributes — `#![forbid(unsafe_code)]` and the lint tables live there — a one-line
crate doc so `missing_docs` is satisfied, and `pub mod` for each module kept.

The copy is a build artefact, not a record: the mutation lists in `tools/mutations/` are the
record, and they are written against `src/`, so the two stay in step by construction. Delete the
copy when the run is done. Re-make it after ANY edit to the module or its dependencies — a copy
made before the edit audits the code you used to have.
"""
import io
import os
import re
import shutil
import sys


def dependencies(repo, module):
    """The modules `module` names through `crate::`, ignoring anything inside a comment."""
    src = io.open(os.path.join(repo, "src", module + ".rs"), encoding="utf-8").read()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("//"))
    return set(re.findall(r"crate::([a-z_0-9]+)", code))


def main():
    if len(sys.argv) != 4:
        sys.exit("usage: slim.py <repo> <destination> <module>")
    repo, dest, module = sys.argv[1], sys.argv[2], sys.argv[3]

    keep, todo = set(), [module]
    while todo:
        m = todo.pop()
        if m in keep:
            continue
        keep.add(m)
        todo += [d for d in dependencies(repo, m) if os.path.exists(os.path.join(repo, "src", d + ".rs"))]

    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(os.path.join(dest, "src"))
    for f in ["Cargo.toml", "Cargo.lock"]:
        shutil.copy(os.path.join(repo, f), dest)
    # The examples name modules this copy may not hold, and `cargo test` would build them.
    toml = io.open(os.path.join(dest, "Cargo.toml"), encoding="utf-8").read()
    toml = re.sub(r"\n\[\[example\]\]\n.*?(?=\n\[|\Z)", "", toml, flags=re.S)
    io.open(os.path.join(dest, "Cargo.toml"), "w", encoding="utf-8").write(toml)

    lib = io.open(os.path.join(repo, "src", "lib.rs"), encoding="utf-8").read()
    attributes = "\n".join(l for l in lib[: lib.index("pub mod ")].splitlines() if not l.startswith("//!"))
    io.open(os.path.join(dest, "src", "lib.rs"), "w", encoding="utf-8").write(
        "//! A slim copy of one module and what it needs, for mutation runs.\n"
        + attributes
        + "\n"
        + "".join(f"pub mod {m};\n" for m in sorted(keep))
        # The crate's own re-exports, for the modules this copy kept. Without them a doc example
        # that writes `ferromorphic::Rng::new(1)` does not compile — which `cargo test --release
        # --lib` never notices, because it does not build doc tests, so a copy could be red in a
        # way the harness structurally could not see. Found by a `bayes` audit whose repair agent
        # ran the full `cargo test` rather than the harness's `--lib` subset.
        + "".join(
            l + "\n"
            for l in lib.splitlines()
            if l.startswith("pub use ") and l[len("pub use ") :].split("::")[0] in keep
        )
    )
    for m in keep:
        shutil.copy(os.path.join(repo, "src", m + ".rs"), os.path.join(dest, "src"))
    print(f"{module} -> {sorted(keep)}")


if __name__ == "__main__":
    main()
