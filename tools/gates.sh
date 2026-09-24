#!/bin/zsh
# The nine gates a release must pass. `tools/gates.sh [LOG]`; the log defaults to target/gates.log.
#
# ⛔ Read the EXIT lines, never the tail. Every gate appends `EXIT <name> <code>` and this script
# refuses unless EVERY code is 0. That is not a formality: a release was once committed with the
# rustdoc gate RED, because the gates and the commit had been chained so the commit ran whatever
# the gates said. A gate whose result nothing reads is not a gate.
#
# One gate is prose. The other eight check code, and the commit that corrected 43 false claims in
# the README doubled five of its paragraphs and shipped them to crates.io with every code gate
# green -- `readme = "README.md"` is served verbatim, and nothing was reading it.
set -u
REPO="${0:A:h:h}"
LOG="${1:-$REPO/target/gates.log}"
cd "$REPO" || exit 1
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

run() { local name="$1"; shift; { "$@"; } >> "$LOG" 2>&1; print "EXIT $name $?" >> "$LOG"; }

run tests        cargo test --release
run clippy       cargo clippy --all-targets --release -- -D warnings
run wasm         cargo build --release --target wasm32-unknown-unknown
run example1     cargo run --release --example event_driven_saves
run example2     cargo run --release --example lif_closed_form
run example3     cargo run --release --example stdp_window
run prose        python3 tools/prose_check.py
run numbers      python3 tools/readme_numbers.py
{ RUSTDOCFLAGS="-D warnings" cargo doc --no-deps; } >> "$LOG" 2>&1; print "EXIT rustdoc $?" >> "$LOG"

print "\n--- gate results ---"
grep '^EXIT ' "$LOG"
bad=$(grep '^EXIT ' "$LOG" | awk '$3 != 0' | wc -l | tr -d ' ')
want=9
got=$(grep -c '^EXIT ' "$LOG")
if [[ "$got" -ne "$want" ]]; then
  print "REFUSED: $got gate results for $want gates; the run did not finish, whatever it says"
  exit 1
fi
if [[ "$bad" -ne 0 ]]; then
  print "REFUSED: $bad of $want gates failed. Full output in $LOG"
  exit 1
fi
print "all $want gates green; full output in $LOG"
