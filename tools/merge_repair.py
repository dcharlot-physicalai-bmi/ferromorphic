#!/usr/bin/env python3
"""Merge one repaired module back from a private crate copy into this repository.

    python3 tools/merge_repair.py --copy /tmp/slim_nef nef

A repair is done in a copy of the crate -- `tools/slim.py` makes one -- so that several modules can
be worked on at once and a mutation run can never touch `src/`. This is the one path from such a
copy back into the repository, and it exists to be strict about the RECORD rather than about the
source: the module's `src/<m>.rs` is copied wholesale, but the mutation list is merged entry by
entry and the merge REFUSES anything that would weaken it.

What it accepts, and what it refuses:

  - an `equivalent` argument ADDED to an entry that had none      accepted
  - an `equivalent` argument REMOVED (a retraction)               accepted, and reported by label
  - an `equivalent` argument REPLACED by a LONGER one             accepted, and counted
  - an `equivalent` argument REPLACED by a SHORTER one            REFUSED
  - any change to an entry's `label`, `old` or `new`              REFUSED
  - any entry added or removed                                    REFUSED

The asymmetry on length is deliberate. A rewrite that shortens an argument is what softening one
looks like, and 45 of this repository's 139 recorded equivalence arguments turned out to be false
when somebody built the fixture -- every one of them by reasoning from what the existing tests
happen to contain rather than from the arithmetic. A retraction is the record getting STRICTER and
is always taken; a retracted entry must then come back `caught` in the next full run.

⛔ This file used to live in a session scratch directory, and the README described its guarantees
while a reader had no way to inspect them. That is why it is here.
"""
import argparse, json, io, os, shutil, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def merge(copy_root, m):
    copy = copy_root
    src_new, src_old = f'{copy}/src/{m}.rs', f'{REPO}/src/{m}.rs'
    if not os.path.exists(src_new):
        return f'{m:12} NO SOURCE IN COPY'
    a = json.load(open(f'{REPO}/tools/mutations/{m}.json'))
    bpath, added, retracted, rewritten = f'{copy}/mutations/{m}.json', 0, [], []
    if os.path.exists(bpath):
        b = json.load(open(bpath))
        if len(a) != len(b):
            return f'{m:12} REFUSED: list length {len(a)} -> {len(b)}'
        for i, (x, y) in enumerate(zip(a, b)):
            for k in ('label', 'old', 'new'):
                if x[k] != y[k]:
                    return f'{m:12} REFUSED: entry {i} changed its {k}'
            if y.get('equivalent') and not x.get('equivalent'):
                x['equivalent'] = y['equivalent']; added += 1
            elif x.get('equivalent') and not y.get('equivalent'):
                del x['equivalent']; retracted.append(x['label'])
            elif x.get('equivalent') and y.get('equivalent', '').startswith('SUPERSEDED'):
                del x['equivalent']; retracted.append(x['label'])
            elif x.get('equivalent') and y.get('equivalent') != x.get('equivalent'):
                if len(y['equivalent']) <= len(x['equivalent']):
                    return (f'{m:12} REFUSED: entry {i} replaced an equivalence argument with a '
                            f'SHORTER one ({len(x["equivalent"])} -> {len(y["equivalent"])} chars). '
                            f'That is what softening one looks like.')
                x['equivalent'] = y['equivalent']
                rewritten.append(x['label'])
        json.dump(a, open(f'{REPO}/tools/mutations/{m}.json', 'w'), indent=1, ensure_ascii=False)
    before = io.open(src_old, encoding='utf-8').read()
    after = io.open(src_new, encoding='utf-8').read()
    shutil.copyfile(src_new, src_old)
    grew = len(after.split('\n')) - len(before.split('\n'))
    ntests = after.count('    #[test]') - before.count('    #[test]')
    note = f', RETRACTED {len(retracted)}: {retracted}' if retracted else ''
    note += f', REWRITTEN {len(rewritten)}' if rewritten else ''
    return f'{m:12} merged: {grew:+5} lines, {ntests:+3} tests, {added} equivalence arguments{note}'

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--copy', required=True, help='the crate copy the repair was done in')
    ap.add_argument('modules', nargs='+')
    a = ap.parse_args()
    bad = 0
    for m in a.modules:
        line = merge(a.copy, m)
        print(line)
        bad += 'REFUSED' in line
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
