#!/usr/bin/env python3
"""The hand mutation harness this crate's audits are run with.

    python3 tools/mutate.py nef vsa            # the recorded mutations of those modules
    python3 tools/mutate.py                    # every module with a list in tools/mutations/
    python3 tools/mutate.py --root /copy nef   # against a copy of the repo (git archive HEAD)

A mutation is one exact-text edit to `src/<module>.rs`. For each one the harness applies the edit,
runs that module's tests (`cargo test --release --lib -- <module>::`), restores the file, and prints
a verdict:

    caught         a test failed, or the test process was killed — the edit is visible
    SURVIVED       every test passed with the edit in place: a test that cannot fail
    equivalent     survived, and the list states why no test could tell (read the reason)
    COMPILE-ERROR  the edit does not compile; it says nothing either way — fix the mutation
    NOT-APPLIED    the `old` text is not in the file exactly once; the list has gone stale
    TIMEOUT        the tests did not finish; look at it by hand, do not count it as caught

Rules the harness exists to enforce:
  - One mutation in flight at a time, and the file is restored even if the harness is killed: the
    in-flight edit is recorded in target/mutate/ and undone at the next start. (An interrupted
    batch once left a live mutant in src/; nothing but `git diff` said so.)
  - The number of tests that ran is printed for the unmutated module first. A test that has
    vanished is indistinguishable from a test that passes in every output except that tally.
  - A repair is not kept until the mutation it exists to catch has been re-run and comes back
    `caught`. Two repairs in 0.8.0 failed that re-run.
"""
import argparse, glob, io, json, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(HERE)


def run_tests(root, module, timeout):
    try:
        p = subprocess.run(["cargo", "test", "--release", "--lib", "--", f"{module}::"], cwd=root,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", None
    ran = re.search(r"test result: \w+\. (\d+) passed; (\d+) failed", p.stdout)
    if "could not compile" in p.stderr and ran is None:
        return "COMPILE-ERROR", None
    if p.returncode == 0 and ran and ran.group(2) == "0":
        return "SURVIVED", int(ran.group(1))
    return "caught", None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("modules", nargs="*", help="module names; default: every list in tools/mutations/")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="the crate to mutate (default: this repo)")
    ap.add_argument("--lists", default=os.path.join(HERE, "mutations"), help="directory of <module>.json lists")
    ap.add_argument("--only", default=None, help="run only mutations whose label contains this text")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per test run")
    args = ap.parse_args()

    state = os.path.join(args.root, "target", "mutate")
    os.makedirs(state, exist_ok=True)
    mark = os.path.join(state, "inflight.json")
    if os.path.exists(mark):
        info = json.load(open(mark))
        shutil.copyfile(info["backup"], info["path"])
        os.remove(mark)
        print(f"RESTORED a mutant left in flight: {info['path']} ({info['label']})", flush=True)

    modules = args.modules or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(args.lists, "*.json")))
    tally = {}
    for module in modules:
        muts = json.load(open(os.path.join(args.lists, f"{module}.json")))
        if args.only:
            muts = [m for m in muts if args.only in m["label"]]
        path = os.path.join(args.root, "src", f"{module}.rs")
        verdict, count = run_tests(args.root, module, args.timeout)
        if verdict != "SURVIVED":
            print(f"{module:12} BASELINE-{verdict}: the unmutated module does not pass; nothing below would mean anything", flush=True)
            tally["BASELINE-FAILED"] = tally.get("BASELINE-FAILED", 0) + 1
            continue
        print(f"{module:12} baseline       {count} tests pass unmutated; {len(muts)} mutations", flush=True)
        for m in muts:
            src = io.open(path, encoding="utf-8").read()
            if src.count(m["old"]) != 1:
                verdict = f"NOT-APPLIED({src.count(m['old'])})"
            else:
                backup = os.path.join(state, f"backup_{module}.rs")
                io.open(backup, "w", encoding="utf-8").write(src)
                json.dump({"path": path, "backup": backup, "label": m["label"]}, open(mark, "w"))
                t0 = time.time()
                try:
                    io.open(path, "w", encoding="utf-8").write(src.replace(m["old"], m["new"], 1))
                    verdict, _ = run_tests(args.root, module, args.timeout)
                finally:
                    io.open(path, "w", encoding="utf-8").write(src)
                    os.remove(mark)
                if verdict == "SURVIVED" and m.get("equivalent"):
                    verdict = "equivalent"
                verdict = f"{verdict} {round(time.time() - t0)}s"
            key = verdict.split()[0].split("(")[0]
            tally[key] = tally.get(key, 0) + 1
            note = f"   [{m['equivalent']}]" if verdict.startswith("equivalent") else ""
            print(f"{module:12} {verdict:18} {m['label']}{note}", flush=True)
    print("TOTAL " + ", ".join(f"{k} {v}" for k, v in sorted(tally.items())), flush=True)
    bad = sum(v for k, v in tally.items() if k not in ("caught", "equivalent"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
