#!/usr/bin/env python3
"""The hand mutation harness this crate's audits are run with.

    python3 tools/mutate.py nef vsa            # the recorded mutations of those modules
    python3 tools/mutate.py                    # every module with a list in tools/mutations/
    python3 tools/mutate.py --root /copy nef   # against a copy of the repo (git archive HEAD)

A mutation is one exact-text edit to `src/<module>.rs`. For each one the harness applies the edit,
runs that module's tests (selected by exact name, so `field` does not drag in `meanfield`),
restores the file, and prints
a verdict:

    caught         a test failed, or the test process was killed — the edit is visible
    caught(const)  the edit falsifies a `const { assert!(..) }`: it cannot even be compiled
    SURVIVED       every test passed with the edit in place: a test that cannot fail
    equivalent     survived, and the list states why no test could tell (read the reason)
    COMPILE-ERROR  the edit does not compile; it says nothing either way — fix the mutation
    NOT-APPLIED    the `old` text is not in the file exactly once; the list has gone stale
    TIMEOUT        the tests did not finish; look at it by hand, do not count it as caught
    KILLED         a signal stopped the test run; it says nothing either way — run it again

A run that is KILLED exits 128 + the signal (143 for SIGTERM) and restores the file on its way out.
Read the exit code, not the tail: a truncated sweep and a finished one print the same kind of lines.

What this harness does NOT see, stated because a limit that is not written down reads as absent:

  - DOCTESTS. Every run is `cargo test --release --lib`, which does not build them. A mutation that
    only a doc example could catch is therefore reported `SURVIVED`, and a slim copy that does not
    compile its doc examples is green here — which is exactly how `tools/slim.py` came to omit the
    crate's `pub use` re-exports for six releases. The bound on that gap is the release gate, which
    runs the full `cargo test --release`; what is lost is the ATTRIBUTION, not the coverage. Running
    doctests per mutation costs about five seconds on top of six to fifteen, so it is a deliberate
    trade and not an oversight.
  - Anything a test does not ASSERT. The harness measures whether the suite fails, and a suite that
    computes a quantity and never reads it fails at nothing. That is what the numbered register of
    vacuous-test mechanisms is for.
  - The `equivalent` key. It is prose, and the harness believes it. Of 139 such arguments in this
    repository, 45 turned out to be false when somebody built the fixture. Audit them separately;
    they are the one verdict here that is an assertion rather than a measurement.

Rules the harness exists to enforce:
  - One mutation in flight at a time, and the file is restored even if the harness is killed: the
    in-flight edit is recorded in target/mutate/ and undone at the next start. (An interrupted
    batch once left a live mutant in src/; nothing but `git diff` said so.)
  - The number of tests that ran is printed for the unmutated module first. A test that has
    vanished is indistinguishable from a test that passes in every output except that tally.
  - A repair is not kept until the mutation it exists to catch has been re-run and comes back
    `caught`. Two repairs in 0.8.0 failed that re-run.
"""
import argparse, glob, io, json, os, re, shutil, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(HERE)


def test_names(root, module):
    """The module's own tests, by exact name: the filter `field::` would also run `meanfield::`."""
    p = subprocess.run(["cargo", "test", "--release", "--lib", "--", "--list"], cwd=root, capture_output=True, text=True)
    return [l.split(": ")[0] for l in p.stdout.splitlines() if l.startswith(f"{module}::") and l.endswith(": test")]


def run_tests(root, module, timeout, names=None):
    selector = ["--exact", *names] if names else [f"{module}::"]
    try:
        p = subprocess.run(["cargo", "test", "--release", "--lib", "--", *selector], cwd=root,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "TIMEOUT", None
    ran = re.search(r"test result: \w+\. (\d+) passed; (\d+) failed", p.stdout)
    # A TEST RUN THAT WAS KILLED IS NOT A CATCH. `subprocess` reports a signal-terminated child
    # with a negative return code, and without this line the fall-through below reads "non-zero
    # exit, no test result" as `caught` — so a sibling process's `pkill -f "cargo test"` would
    # turn every mutation it interrupted into a mutation this repository believes is covered.
    # That is the one failure mode a mutation harness must not have, and it was reachable: two
    # agents sharing this machine ran exactly that command.
    if p.returncode < 0 and ran is None:
        return "KILLED", None
    if "could not compile" in p.stderr and ran is None:
        # A `const { assert!(..) }` that the edit falsifies fails the BUILD, with E0080 and the
        # assertion's own message. That is the strongest catch there is — the mutant cannot be
        # produced at all — and reporting it as the inconclusive COMPILE-ERROR sent an audit
        # looking for a malformed edit three times over. Only that one error code is read this
        # way; a syntax error or a type error still says nothing either way.
        if "E0080" in p.stderr and "evaluation panicked" in p.stderr:
            return "caught(const)", None
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
    ap.add_argument("--skip", type=int, default=0, help="skip the first N mutations of each list (run only what was added since)")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per test run")
    args = ap.parse_args()

    # A SIGTERM is not hypothetical on a shared machine: six of this repository's module audits
    # were killed mid-run by a sibling process's `pkill -f mutate.py`, and Python does not unwind
    # `finally` blocks for a signal it does not handle, so each kill LEFT A LIVE MUTANT in `src/`.
    # The in-flight marker below repairs that at the next start, but only for the same `--root` and
    # only if someone runs it again. Handling the signal restores the file immediately instead, and
    # exits 143 so the caller can still tell a killed run from a finished one.
    def _restore_and_die(signum, _frame):
        mark = os.path.join(args.root, "target", "mutate", "inflight.json")
        if os.path.exists(mark):
            info = json.load(open(mark))
            shutil.copyfile(info["backup"], info["path"])
            os.remove(mark)
            print(f"RESTORED on signal {signum}: {info['path']} ({info['label']})", flush=True)
        sys.exit(128 + signum)

    for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(_sig, _restore_and_die)

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
        muts = muts[args.skip:]
        if args.only:
            muts = [m for m in muts if args.only in m["label"]]
        path = os.path.join(args.root, "src", f"{module}.rs")
        names = test_names(args.root, module)
        verdict, count = run_tests(args.root, module, args.timeout, names)
        if verdict != "SURVIVED" or count != len(names):
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
                    verdict, _ = run_tests(args.root, module, args.timeout, names)
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
