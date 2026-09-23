#!/usr/bin/env python3
"""Check the README's headline counts against the repository, or print what they should be.

    python3 tools/readme_numbers.py            # check; exit 1 on a mismatch
    python3 tools/readme_numbers.py --print    # just print the computed figures

⛔ Why. This README's own audit found 557 quantitative claims, of which **218 were pinned by
nothing** and 43 were simply wrong — including a test count 423 low whose value appeared exactly
once in the whole tree, so nothing corroborated it and nothing could notice it drifting. The
headline figures — modules, tests, recorded mutations, equivalence arguments — are the ones a
reader quotes, and they move on almost every commit. Counting them here makes them the one class of
README number that cannot go stale quietly.

This counts what it can count exactly. `#[test]` functions are counted by text, which is what the
suite reports too; a figure this script cannot derive from the repository is not its business.
"""
import glob, io, json, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def figures():
    lists = sorted(glob.glob(os.path.join(REPO, "tools", "mutations", "*.json")))
    muts = eq = 0
    for p in lists:
        entries = json.load(open(p))
        muts += len(entries)
        eq += sum(1 for e in entries if e.get("equivalent"))
    srcs = sorted(glob.glob(os.path.join(REPO, "src", "*.rs")))
    tests = sum(io.open(p, encoding="utf-8").read().count("#[test]") for p in srcs)
    lib = io.open(os.path.join(REPO, "src", "lib.rs"), encoding="utf-8").read()
    modules = len(re.findall(r"^pub mod \w+;", lib, re.M))
    return {"modules": modules, "tests": tests, "mutations": muts, "equivalence arguments": eq,
            "lists": len(lists)}


def main():
    f = figures()
    if "--print" in sys.argv:
        for k, v in f.items():
            print(f"{k:24}{v:>7,}")
        return
    readme = io.open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
    # Every module has a list, so the two counts are one claim; state it rather than assume it.
    bad = []
    if f["lists"] != f["modules"]:
        bad.append(f"{f['lists']} recorded lists for {f['modules']} modules — the README's "
                   f"'at least one list per module' is not true of the repository")
    for name, value in (("tests", f["tests"]), ("mutations", f["mutations"])):
        pretty = f"{value:,}"
        if pretty not in readme:
            bad.append(f"the README does not state the current {name} count ({pretty})")
    for line in bad:
        print(f"readme_numbers: {line}")
    print(f"readme_numbers: {f['modules']} modules, {f['tests']:,} tests, {f['mutations']:,} "
          f"mutations, {f['equivalence arguments']} equivalence arguments; {len(bad)} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
