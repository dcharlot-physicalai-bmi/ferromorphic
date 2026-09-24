#!/usr/bin/env python3
"""Refuse prose that an anchored edit has doubled: a repeated sentence, stem or list marker.

    python3 tools/prose_check.py              # README.md, the other .md files, and every doc comment
    python3 tools/prose_check.py FILE...      # just those files (this is how the gate is proved able to fail)

⛔ Why this exists. The commit that corrected 43 wrong claims in this README also DOUBLED five of
its paragraphs -- the anchored replacements re-stated text whose original tail was still there, so
`README.md` shipped to crates.io reading "And the direction matters: And the direction is not
uniform. ... **Exactly one price in this crate came from fabricated silicon**, and a test asserts
the count. A field benchmarking against a pre-silicon simulation ... and a test asserts the count."
All five release gates were green, because **no gate reads prose**: `cargo test`, clippy, the wasm
build, the examples and `cargo doc` all check code, and `README.md` is published verbatim. A
repository whose whole argument is that a claim must be checkable had its front page break in a way
nothing could see. The class is not typography -- it is the repair introducing the next defect.

What it refuses, all within ONE paragraph or ONE contiguous doc-comment block, so that two methods
carrying the same `# Errors` line are not a finding:

  - the same sentence twice, or two sentences over 90% alike (the doubling that also changed a
    number, which an equality test reads as two distinct claims);
  - a sentence whose opening words recur after a colon inside it (a restated stem);
  - a doubled list marker (`- -`, `* *`) at the start of a line;
  - a Markdown table row that spills onto a second line or has the wrong number of cells, which
    ends the table where it is rendered.

It is deliberately narrow. A check that fires on prose people wrote on purpose gets switched off.

⚠ A THIRD shape of this defect is known and is NOT checked here, deliberately. The same anchored
edit can double a clause INSIDE one sentence — this README shipped "gives a median of 42.8%, a
maximum of 94.7% and gives a median of 42.8%, a maximum of 94.7% and a floor of 1.7%" — which is
unique as a sentence and does not restate its stem, so neither check above sees it. A repeated word
n-gram inside one sentence was tried at n = 6, 8, 10 and 12, with a contiguity window, with tables
and code excluded, and with a digit required in the repeat. **No variant separated the defect from
ordinary prose**: parallel construction ("exactly half the samples of class k … of class j"),
inline code and figure lists all repeat phrases on purpose, and at every setting the clean tree
scored as many findings as the broken file. It was removed rather than shipped noisy. If you are
about to re-add it, that is the bar it has to clear.
"""
import difflib, glob, io, os, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def blocks(path):
    """Paragraphs, or -- for Rust -- each run of adjacent `///`/`//!` lines as one block."""
    txt = io.open(path, encoding="utf-8").read()
    if not path.endswith(".rs"):
        return re.split(r"\n\s*\n", txt)
    out, cur = [], []
    for line in txt.splitlines():
        m = re.match(r"^\s*(///|//!)\s?(.*)$", line)
        if m:
            cur.append(m.group(2))
        elif cur:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def cells(line):
    """Cells of a Markdown table row, ignoring pipes inside code spans and escaped pipes."""
    n, code, prev = 0, False, ""
    for ch in line:
        if ch == "`":
            code = not code
        elif ch == "|" and not code and prev != "\\":
            n += 1
        prev = ch
    return n


def broken_tables(path):
    """Every line of a Markdown table must be ONE row with the header's cell count.

    Unlike a doubled sentence this is structural, so it can be checked exactly: a table is a
    header line, a separator line, and then rows until a blank line, and a row that spills onto a
    second line ends the table right there when it is rendered. This README shipped two such rows
    in its module table, and every module listed after them printed as loose text on crates.io.
    """
    lines = io.open(path, encoding="utf-8").read().split("\n")
    out = []
    for i in range(len(lines) - 1):
        head, sep = lines[i], lines[i + 1]
        if not (head.startswith("|") and sep.startswith("|") and set(sep.replace("|", "").strip()) <= set("-: ")):
            continue
        want = cells(head)
        k = i + 2
        while k < len(lines) and lines[k].strip():
            row = lines[k]
            if not row.startswith("|") or cells(row) != want:
                out.append((f"a table row broken across lines or with the wrong cell count (line {k + 1})", row[:90]))
            k += 1
    return out


def findings(path):
    out = []
    for raw in blocks(path):
        flat = re.sub(r"\s+", " ", raw).strip()
        sents = [s.strip() for s in re.split(r"(?<=[.!?]) ", flat) if len(s.strip()) > 40]
        for i, s in enumerate(sents):
            if i and s == sents[i - 1]:
                out.append(("the same sentence twice, adjacent", s))
            elif s in sents[:i]:
                out.append(("the same sentence twice in one block", s))
            else:
                # NEARLY the same sentence is the worse case, not the milder one: an anchored
                # edit that re-stated a paragraph AND changed a figure inside it leaves two
                # sentences that contradict each other, and an equality test reads them as
                # distinct prose. This repository shipped exactly that -- "There were 139 of
                # them" beside "There were 142 of them", and "Forty-eight keys are gone" beside
                # "Forty-five keys are gone", in one paragraph, past an exact-match check.
                for j, t in enumerate(sents[:i]):
                    if len(s) > 60 and len(t) > 60 and difflib.SequenceMatcher(None, s, t).ratio() > 0.9:
                        out.append((f"two near-identical sentences in one block ({j + 1} and {i + 1})", s))
                        break
            head = " ".join(s.split()[:3])
            if len(head) > 8 and ": " + head in s:
                out.append(("a restated stem after a colon", s))
            # A clause doubled INSIDE one sentence is the third shape the same botched edit takes,
            # and neither of the checks above can see it: the sentence is unique, and its stem is
            # not restated. This repository shipped "gives a median of 42.8%, a maximum of 94.7%
            # and gives a median of 42.8%, a maximum of 94.7% and a floor of 1.7%" past both.
    if path.endswith(".md"):
        out += broken_tables(path)
    for n, line in enumerate(io.open(path, encoding="utf-8"), 1):
        if re.match(r"^\s*(- -[^-]|\* \*[^*])", line):
            out.append((f"a doubled list marker on line {n}", line.strip()))
    return out


def main():
    files = sys.argv[1:] or ([os.path.join(REPO, "README.md")]
             + sorted(p for p in glob.glob(f"{REPO}/*.md") if not p.endswith("README.md"))
             + sorted(glob.glob(f"{REPO}/src/*.rs")) + sorted(glob.glob(f"{REPO}/examples/*.rs")))
    bad = 0
    for f in files:
        for why, text in findings(f):
            bad += 1
            print(f"{os.path.relpath(f, REPO)}: {why}\n    {text[:150]}")
    print(f"prose_check: {len(files)} files, {bad} doubled passages")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
