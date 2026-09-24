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
# Everything after this heading is the audit history, which quotes each release's own counts.
HISTORY = "## Every module has been audited, and the audit found things"


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
    # "Somewhere in the file" is not enough, and the first version of this check proved it: it
    # passed while the headline on line 12 and a sentence in the verification section still said
    # 2,252 tests, because the Status paragraph stated 2,597. So the rule is EVERY count in the
    # README's current-state part must equal the repository's. The audit history below it quotes
    # the counts of its own releases on purpose, and is exempt.
    head, sep, _ = readme.partition(HISTORY)
    if not sep:
        bad.append(f"the heading {HISTORY!r} that separates current state from history is gone, so "
                   f"this check can no longer tell a stale count from a historical one")
    for name, value, pattern in (
        ("tests", f["tests"], r"\b(\d[\d,]*) (?:unit )?tests\b"),
        ("mutations", f["mutations"], r"\b(\d[\d,]*) (?:recorded )?mutations\b"),
    ):
        seen = [int(m.replace(",", "")) for m in re.findall(pattern, head)]
        if value not in seen:
            bad.append(f"the README's current-state part never states the {name} count ({value:,})")
        for n in sorted(set(seen) - {value}):
            bad.append(f"the README's current-state part says {n:,} {name}; the repository has {value:,}")
    for line in bad:
        print(f"readme_numbers: {line}")
    print(f"readme_numbers: {f['modules']} modules, {f['tests']:,} tests, {f['mutations']:,} "
          f"mutations, {f['equivalence arguments']} equivalence arguments; {len(bad)} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
