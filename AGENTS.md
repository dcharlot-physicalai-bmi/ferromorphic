# ferromorphic — guidance for AI agents

This crate is designed to be driven by coding agents. Everything below is enforced by tests, so you
can trust it without reading the source first.

## What this library is

Pure-Rust neuromorphic computing: spiking neuron models verified against their closed forms, sparse
directed networks with per-synapse delays, spike encoders and decoders, and a joules ledger that
prices every synaptic operation, memory fetch, membrane update, spike and readout against swappable
device models. Zero dependencies, std-only, compiles unchanged to `wasm32-unknown-unknown`,
deterministic for a fixed seed.

## Invariants you can rely on

1. **Every neuron model is verified against a closed form, not against a previous run.** `Lif` free
   decay matches `v_rest + (v0 - v_rest)·exp(-t/τ)` to 1e-12; the simulated firing rate matches
   `Lif::isi` to 0.1%; `IntegrateAndFire` spikes within one tick of its exact linear prediction. If
   you change something and `cargo test` passes, these still hold.
2. **Determinism**: same seed, same spikes, every platform. Never use wall-clock or OS randomness
   inside the library. `rng::Rng` is the only source, and changing its algorithm is a BREAKING change
   because it changes every spike train in every downstream experiment.
3. **No dependencies.** Do not add any. The zero-dep property is a product feature — auditability,
   wasm size, longevity — not an accident.
4. **The ledger refuses rather than guesses.** `Prices` fields are `Option` where a figure may be
   unpublished, and `Ledger::joules` returns `None` when any term with a NON-ZERO COUNT has no price.
   Every device table in the crate leaves `e_syn_fetch` as `None`, so `joules` refuses for all of
   them. **That is the finding, not a defect.** If you add a fetch price, it needs a citation and an
   `Evidence` grade, and `no_device_table_in_this_crate_prices_a_synapse_fetch` is where you argue
   for it.
5. **A price table names its subject.** `Prices::source` is not documentation. A `Prices` without a
   subject can be applied to any machine, which is how a laptop came to report another company's
   unfabricated accelerator's energy in the sibling crate.
6. **Event-driven simulation is gated on a declared property, not on judgement.**
   `Neuron::EXACT_OVER_GAPS` says whether a model can be jumped across quiet ticks.
   `Sim::new` REFUSES `Mode::EventDriven` for a model where it is false. If you add a model,
   set it honestly: it is true only if one step of `k·dt` with zero input gives exactly the state of
   `k` steps of `dt`.
7. **The two simulation modes must agree.** `the_two_modes_produce_the_same_spike_train` compares
   them spike for spike. If you touch `Sim::step` or `Sim::catch_up`, that test is the gate. Note
   that the refractory catch-up ROUNDS UP to a tick boundary specifically so the modes agree; a
   "cleaner" jump to the exact real-valued end of the period breaks it by up to one tick.
8. **Units are SI at every interface.** `dt` in seconds, current in amperes, potential in volts,
   `bump` in volts. `Izhikevich` keeps the paper's millisecond/millivolt constants INSIDE the model
   where they can be compared against the source, and converts at the boundary. Do not "tidy" those
   constants into SI; they become unrecognisable against the paper.
9. **A synapse is stored ONCE.** `Net::n_syn` is `post.len()`, not `post.len() / 2`. The sibling
   crate's `Graph` is undirected and stores each edge twice; transferring that `/ 2` here halves
   every synapse count and therefore every energy figure, in the direction that flatters the result.
10. **A default is not a fallback.** An input the code cannot understand is an error naming what was
    sent. A NaN weight is refused at `NetBuilder::connect` rather than allowed to poison one membrane
    potential, then every spike time that neuron produces, then the whole run — which completes and
    reports zero spikes.

## How to do common tasks

- **Simulate a network**: `NetBuilder::new(n)` → `connect(pre, post, weight_volts, delay_ticks)` →
  `build()` → `Sim::new(net, vec![Lif::default(); n], dt, Mode::EventDriven)?` → `run(ticks, &ext)`.
- **Price it**: `sim.ledger.bill(&PRICES)`. Read `bill.unpriced` before `bill.total`. Use
  `joules_synops_only` only to reproduce someone else's published number, never to make one.
- **Check an event-driven claim**: `sim.ledger.idle_fraction()` on a CLOCKED run. That fraction is
  the work the hardware would avoid, and it is workload-dependent — measure it, do not assume it.
- **Encode an input**: `RateEncoder` for compatibility with the literature, `LatencyEncoder` when the
  spike budget matters, `DeltaEncoder` to imitate an event sensor. Call
  `RateEncoder::ticks_for_precision` before choosing rate coding; it is usually the answer.
- **Add a neuron model**: implement `Neuron`, set `EXACT_OVER_GAPS` honestly, and add a test that
  checks it against a closed form. A model with no analytic check does not belong here yet.

## Before you commit

`tools/gates.sh`. Nine results, every one of which must be 0: the test suite, clippy at
`-D warnings` over all targets, the wasm build, the three examples, `tools/prose_check.py`,
`tools/readme_numbers.py` (which recomputes the README's headline counts from the repository and
refuses if the file disagrees — the last time those were typed by hand the test count was 423 low),
and rustdoc with `-D warnings`. The script refuses unless it sees all eight, because a gate runner that
reads the tail cannot tell a truncated run from a clean one — and a release here was once committed
with the rustdoc gate red, the gates and the commit having been chained so the commit ran regardless.

The prose gate is there because the other seven check code. `readme = "README.md"` means crates.io
serves that file verbatim, and the commit that corrected 43 false claims in it doubled five of its
paragraphs and shipped them with everything else green.

## Traps this crate has already hit

- A test that started a `Lif` at −40 mV to watch it decay: that is ABOVE the −50 mV threshold, so the
  neuron fired and reset on its first step and the test compared a reset potential against an
  exponential. Start sub-threshold, or you are testing the reset path.
- A `1 - exp(-x)` assertion written as `p < 1.0`: at `x = 100` the nearest `f64` to `1 - 3.7e-44` IS
  `1.0`, and the assertion was pinning a floating-point accident rather than a property. Assert
  `p <= 1.0`.
- A tolerance of 1e-4 V after five membrane constants: the residual there is 1.7e-4 V. Ten constants
  gives 1.1e-6 V. Compute the residual before choosing the tolerance.
