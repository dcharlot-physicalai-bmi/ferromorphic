#!/usr/bin/env python3
"""Append new mutations to the recorded lists, re-verifying every anchor against today's source.

    python3 tools/extend_list.py --from target/held/deepen target/held/deepen2
    python3 tools/extend_list.py --from /somewhere/new --only nef vsa

Each directory holds `<module>.json`: a JSON array of ONLY THE NEW entries for that module, in the
same shape as `tools/mutations/<module>.json`. They are appended to the module's recorded list.

A list is the audit's record, so the bar to enter it is the anchor, not the author. Per module, the
whole file is refused unless every new entry passes:

  - its `old` occurs EXACTLY ONCE in today's `src/<module>.rs`. A list written against a source that
    has since been repaired is stale, and `mutate.py` reports a stale anchor as `NOT-APPLIED` --
    which is a verdict that says nothing, not a catch;
  - its `old` and `new` differ. A no-op mutation is always `SURVIVED`, or always `caught`, for
    reasons that have nothing to do with the tests;
  - its `(old, new)` pair repeats neither an already-recorded entry nor another new one. A duplicate
    inflates the count, which is the number everything else in this repository is divided by.

A module that fails is left exactly as it was and named in the output; the others still install. The
unit of all-or-nothing is the module, because the module is the unit a sweep runs and reports.

⛔ An arriving `equivalent` key is STRIPPED, and the argument is written to
`<dir>/../predicted_equivalent.json` instead. An `equivalent` argument is the one verdict this
repository's harness asserts rather than measures: it tells `mutate.py` to relabel a survival as
expected and let the run pass. Of 142 such arguments carried here, 48 were false when somebody
built the fixture -- and every one of those was written by reasoning from what the existing tests
happen to contain. An argument written BEFORE the mutation was ever run is a prediction with no
measurement behind it at all, and installing it would let a mutation excuse itself. Run it first.
A prediction that comes back `caught` was wrong about the arithmetic; one that comes back
`SURVIVED` has earned the right to be read, and then recorded by hand.
"""
import argparse, io, json, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(module, new, old, src):
    """Every reason this module's new entries may not enter the record, or an empty list."""
    problems, pairs, seen = [], {(e["old"], e["new"]) for e in old}, set()
    for e in new:
        if not all(k in e for k in ("label", "old", "new")):
            problems.append(f"an entry is missing one of label/old/new: {e!r:.80}")
            continue
        n = src.count(e["old"])
        if n != 1:
            problems.append(f"anchor occurs {n} times, not once: {e['label']}")
        if e["old"] == e["new"]:
            problems.append(f"no-op: {e['label']}")
        k = (e["old"], e["new"])
        if k in pairs:
            problems.append(f"already recorded: {e['label']}")
        elif k in seen:
            problems.append(f"repeated within this file: {e['label']}")
        seen.add(k)
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="dirs", nargs="+", required=True, help="directories of <module>.json new-entry files")
    ap.add_argument("--only", nargs="*", default=None, help="install only these modules")
    ap.add_argument("--dry-run", action="store_true", help="report and change nothing")
    a = ap.parse_args()

    installed, refused, predictions, total = [], [], [], 0
    for d in a.dirs:
        d = d if os.path.isabs(d) else os.path.join(REPO, d)
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json"):
                continue
            m = f[:-5]
            if a.only and m not in a.only:
                continue
            lst = os.path.join(REPO, "tools", "mutations", f"{m}.json")
            if not os.path.exists(lst):
                refused.append(f"{m:12} REFUSED: no recorded list to extend")
                continue
            new = json.load(open(os.path.join(d, f)))
            old = json.load(open(lst))
            src = io.open(os.path.join(REPO, "src", f"{m}.rs"), encoding="utf-8").read()
            problems = check(m, new, old, src)
            if problems:
                refused.append(f"{m:12} REFUSED, {len(problems)} problems, {len(new)} entries held back")
                refused += [f"{'':12}   {p}" for p in problems[:6]]
                continue
            for e in new:
                arg = e.pop("equivalent", None)
                if arg is not None:
                    predictions.append({"module": m, "label": e["label"], "argument": arg})
            if not a.dry_run:
                json.dump(old + new, open(lst, "w"), indent=1, ensure_ascii=False)
            installed.append(f"{m:12} {len(old):5} -> {len(old) + len(new):5}  (+{len(new)})")
            total += len(new)

    print("\n".join(installed))
    if predictions:
        out = os.path.join(REPO, "target", "held", "predicted_equivalent.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if not a.dry_run:
            json.dump(predictions, open(out, "w"), indent=1, ensure_ascii=False)
        print(f"\nSTRIPPED {len(predictions)} equivalence arguments written before the mutation was run.")
        print(f"They are predictions, not arguments; filed in {out}. Score them against the sweep.")
    if refused:
        print("\n" + "\n".join(refused))
    verb = "would install" if a.dry_run else "installed"
    print(f"\n{verb} {total} new mutations across {len(installed)} modules; {len(refused) and 'SOME REFUSED' or 'none refused'}")
    sys.exit(1 if refused else 0)


if __name__ == "__main__":
    main()
