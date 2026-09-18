# ferromorphic

**Neuromorphic computing in pure Rust — the open corpus in one library.** Spiking neuron models from
Hodgkin-Huxley to `AdEx`, synapse kernels, plasticity rules, surrogate-gradient and local learning,
ANN-to-SNN conversion, reservoir computing, every neural code, the standard topologies, spiking
convolutions and attention, a silicon cochlea and an olfactory bulb, event-based vision and
multimodal fusion, spiking control, mean-field theory, model compression and core mapping,
hardware constraint models, analog device non-idealities, NIR, event-camera decoders, benchmark
metrics, teaching tasks — and a joules ledger that charges for the memory traffic a synaptic
operation needs.

**Zero dependencies. `std` only. `wasm32` clean. Deterministic by seed. 40 modules, 1,544 tests.**

Run it in a browser without installing anything:
**[energy.physicalai-bmi.org/neuromorphic](https://energy.physicalai-bmi.org/neuromorphic)** — the
same gates this crate runs, on your own machine. It refuses to report a number until the simulator
reproduces the closed-form inter-spike interval, produces exactly nothing below threshold, and gets
the same spike train from both simulation modes.

The Institute's position is that in the era of AI a group should build one source of truth, not a
constellation of thin wrappers. So this is an ingestion, not a sampler: what is open, public and
academic in neuromorphic computing belongs in one auditable codebase that a student can read end to
end and a robot can run on any fabric.

The science is old and open — Lapicque 1907, Hodgkin & Huxley 1952, Mead 1990, Mahowald 1992,
Bi & Poo 1998, Maass 2002, Izhikevich 2003, Brette & Gerstner 2005. What a neuromorphic chip
accelerates is exactly these loops; what it charges for is moving the weights.

## What is in it

| module | what it carries |
|---|---|
| `neuron` | LIF, integrate-and-fire, adaptive LIF, Izhikevich |
| `hh` | Hodgkin-Huxley, full four-variable squid axon, with the ionic currents exposed |
| `exponential` | EIF, `AdEx` with the Naud firing-pattern taxonomy, QIF, theta |
| `synapse` | delta / exponential / alpha / bi-exponential kernels, CUBA vs COBA, AMPA / GABA / NMDA with the magnesium block, Tsodyks-Markram short-term plasticity |
| `plasticity` | pair and triplet STDP, Hebbian, Oja, BCM, reward-modulated three-factor, homeostatic scaling |
| `surrogate` | surrogate gradients (`SuperSpike`, arctan, triangular, boxcar, straight-through) and a working BPTT path |
| `convert` | ANN-to-SNN: threshold balancing, percentile normalisation, reset-by-subtraction vs reset-to-zero |
| `reservoir` | liquid state machines and echo state networks, with a pure-Rust ridge solve and power iteration |
| `encode`, `coding` | rate, latency, delta; population, rank-order, phase, burst, BSA/HSA, temporal contrast — and their decoders |
| `topology` | Erdős-Rényi, Watts-Strogatz, Barabási-Albert, distance-dependent, layered, winner-take-all, Dale's law |
| `hardware` | constraint models for Loihi, Loihi 2, `TrueNorth`, `NorthPole`, Akida, `SpiNNaker`, Xylo, Speck, ODIN, DYNAP and more — every field graded and provenanced |
| `nir` | the Neuromorphic Intermediate Representation graph, validation, and a bridge to this crate's networks |
| `aer` | AEDAT and Prophesee EVT2/EVT3 decoders, total and panic-free, with rollover-correct timestamps |
| `metrics` | `NeuroBench` complexity metrics: activation sparsity, effective MACs and ACs, footprint |
| `ledger`, `crossover` | joules with the fetch term, and the published SNN-vs-ANN thresholds as a runnable check |
| `tasks` | deterministic teaching problems: temporal XOR, coincidence detection, delayed match-to-sample, synthetic event streams |
| `spikeconv` | spiking convolutional networks — the architecture almost all deployed spiking vision runs — with tdBN, `SEW` residual blocks and pooling |
| `attention` | spiking attention and spiking transformers, with an honest count of what is actually spiking |
| `eprop` | local learning rules: training forward in time without storing the past |
| `continual` | learning without forgetting, on-chip, with the forgetting measured first |
| `meanfield` | mean-field theory: what a spiking network does in aggregate, in closed form |
| `bayes` | spikes as samples: Bayesian inference by firing |
| `vision` | event-based vision: the algorithms that consume what an event camera emits |
| `cochlea` | the silicon cochlea: gammatone bank, Meddis hair cell, gain control, onset and offset channels, every stage against its closed form |
| `olfaction` | the olfactory bulb's external plexiform layer, learning an odour from one presentation |
| `fusion` | multimodal fusion in spikes: the clock offset and drift between event streams, with an error bar calibrated against the scatter it describes |
| `control` | closing a loop with spikes: a spiking PID, Matsuoka and half-centre pattern generators, a Kalman and a population estimator, and the plants to check them against |
| `device` | analog non-idealities: what a weight becomes when it is a physical conductance |
| `compress` | pruning, quantisation, distillation and the rate-budget trade, to fit the part you can buy |
| `mapping` | placing a network on cores: partitioning, multicast trees, fabric hops, and where the energy goes |
| `nef` | the Neural Engineering Framework: tuning curves, least-squares decoders, factorised weights, and dynamics through a synapse — Nengo's three principles, each against its closed form |
| `vsa` | vector symbolic architectures / hyperdimensional computing: bipolar, binary and holographic models, codebooks, sequences, and a resonator network, with the bundle capacity as a binomial |
| `sparse` | sparse coding by local competition: the LCA and its spiking form, checked against the LASSO's own optimality conditions |
| `resonate` | resonate-and-fire neurons stepped by the exact complex exponential, and the Legendre Memory Unit as a delay line in `d` numbers |
| `sim`, `net`, `spike`, `rng` | the event-driven simulator, sparse connectivity, spike trains, seeded PCG32 |

## Use it

```sh
cargo add ferromorphic
```

```rust
use ferromorphic::{ledger::TRUENORTH_2014, net::NetBuilder, neuron::Lif, sim::{Mode, Sim}};

// A five-neuron chain: each cell drives the next after two ticks.
let mut b = NetBuilder::new(5);
for i in 0..4 { b.connect(i, i + 1, 20e-3, 2)?; }

let mut ext = vec![0.0; 5];
ext[0] = 3e-9;                                    // 3 nA into the first cell
let mut sim = Sim::new(b.build(), vec![Lif::default(); 5], 1e-4, Mode::EventDriven)?;
let train = sim.run(2_000, &ext);                 // 200 ms at 0.1 ms per tick

// What it would cost on TrueNorth. The answer is a refusal, and the reason comes with it.
let bill = sim.ledger.bill(&TRUENORTH_2014);
assert!(bill.total.is_none());
assert!(bill.unpriced.contains(&"synapse memory fetch"));
assert!(bill.synaptic.unwrap() > 0.0);            // the term that IS priced still reports
# Ok::<(), Box<dyn std::error::Error>>(())
```

## The position this crate takes

**A synaptic operation count is not an energy measurement.** The standard figure in this field is a
SOP count multiplied by a datasheet joule-per-SOP. That prices the arithmetic and sets the cost of
fetching the weight to zero — and the fetch is the term that scales with *where the model lives*
rather than with how often it fires, so it is exactly the term that separates a benchmark from a
deployment. A chip whose synapses fit on-core and the same chip running a model that spills have the
same `e_syn_op` and a different bill.

This Institute has made the analogous mistake before. In a sibling crate, omitting attention's score
matrix from a published arithmetic-intensity figure made it **wrong by 847×**, and the omission was
invisible because the number it produced looked entirely reasonable.

So `Prices` carries `e_syn_fetch`, **every device table in this crate leaves it `None`**, and
`Ledger::joules` therefore refuses. That is the finding, not an unfinished implementation:

> This review did not locate a published per-synapse memory-fetch energy for any commercially
> available neuromorphic processor.

If you have one — measured, for a stated device, at a stated boundary — supply it and the ledger
prices your workload. The flattering figure the literature reports is still computable as
`Ledger::joules_synops_only`, with its doc saying exactly what it omits, because a user needs to
reproduce published numbers in order to argue with them and hiding it would not stop anyone using it.

### What this crate does not claim

Measured joules for spiking workloads are not unheard of, and saying so would be false. The
**NeuroBench** system track mandates and publishes them — SynSense Xylo Audio 2 at 0.028 mJ per
inference against an Arduino Nano 33 BLE at 0.934 mJ, idle and active and dynamic power reported
separately, analog front end priced on its own. **Rockpool**'s `XyloSamna(record_power=True)`
returns per-rail watts off the board, and BrainChip's tooling divides on-SoC power-meter samples by
frames. Those are real, and all three do more than this crate: **ferromorphic measures nothing and
has no hardware.**

What survived checking is narrower: no published per-synapse memory-**fetch** energy for any
commercial part, and no study that instrumented a DRAM rail while running a spiking network and
reconciled the reading against the models. The models exist — SATA_Sim and EnforceSNN put memory at
**50–78% of the bill** from CACTI and DRAMPower — and nothing appears to have checked them against a
meter.

**How badly the unchecked models disagree is measurable.** SpikingJelly ships its own
cross-validation of five literature-sourced energy models. On identical workloads they disagree by a
**median factor of 556, spanning 218× to 753×** — one network priced at 553.4 µJ by one model and
1.13 µJ by another. Five published methods, one workload, three orders of magnitude. This ledger
declines to be the sixth.

### ⛔ A correction to this crate, in this crate

0.1.0 and 0.2.0 shipped `LOIHI_2018` at `Evidence::Measured`. **It is not measured.** The 23.6
pJ/SynOp figure comes from Davies et al., IEEE Micro 38(1), 2018, **Table 2 — captioned
"pre-silicon"**, sourced from pre-silicon SDF and SPICE simulations. No Loihi was ever on a meter
for it. The field cites it as measured throughout 2024–2026, and this crate joined them, **inside
the module written to stop exactly that**.

It is now `Evidence::Simulated`, with a regression test pinning it and the caption quoted in the
source line. The error is left visible because an `Evidence` enum does not help if the value handed
to it was copied from the citing literature instead of the cited table.

And the direction matters: per-synaptic-operation energies actually measured on fabricated silicon
sit **above** this simulated figure — TrueNorth at 26 pJ in 28 nm. A field benchmarking against a
pre-silicon simulation is benchmarking against a number that flatters it, and every efficiency ratio
computed from it inherits that. **Exactly one price in this crate came from fabricated silicon**, and
a test asserts the count.

### The field already published the threshold, and no library checks it

The quantity that decides whether a spiking network can beat its dense equivalent is **spikes per
synapse per inference** — each spike crossing a synapse costs an accumulate and, far more, a weight
fetch, and once a network re-reads its weights more often than a dense pass would, the dense pass
wins. At least six papers give a number for it. **Every one is below 2. Several are below 1.**

| threshold | source | what it says |
|---|---|---|
| **~1.72** | Davidson & Furber, *Front. Neurosci.* 15:651141 (2021) | "most rate-coded spiking network implementations will not be more energy or resource efficient than the original ANN" |
| **0.15 – 1.38** | Dampfhoffer et al., *IEEE TETCI* 7(3):731–741 (2023) | "many previous studies did not consider **memory accesses**, which account for an important fraction of the energy consumption" |
| **~0.06 – 0.35** | Yan, Bai, Tang & Wong (NUS), arXiv:2409.08290 | op-count evaluations "neglect critical overheads like comprehensive data movements and memory accesses" and reach "misleading conclusions"; under a fair mapping the SNN reaches **0.78×** — a 22% saving, not a 100× one |

Steve Furber designed SpiNNaker, so the first of those is the field auditing itself, not an outsider
objecting to it.

This review located that argument in the literature and **did not locate it implemented as a check
in any spiking-network library**. It gets quoted in related-work sections and then not applied —
which is a strange fate for a number that decides whether the whole approach helps. So it is a check
here, and it costs nothing, because the left-hand side is a ratio of two integer counts the simulator
already keeps:

```rust
let sps = sim.ledger.spikes_per_synapse(net.n_syn as u64, inferences).unwrap();
for (name, verdict) in sim.ledger.crossover_verdicts(net.n_syn as u64, inferences).unwrap() {
    println!("{name:24} {verdict:?}");   // Plausible | Marginal | Refuted
}
```

Reported against **all three** rather than against a chosen one, because picking the threshold your
workload passes is the same move as picking the price table that flatters your device. A band gives
three verdicts and not two: inside it, the cited work's answer depended on assumptions it states, and
collapsing that to a pass or a fail would be inventing a precision nobody published.

And a verdict is not a measurement of your workload on your hardware. Every threshold was derived for
a stated technology and dataflow; transplanting it is exactly the borrowing the ledger refuses to do
with joules. A verdict says *which side of somebody else's published line you fall on* — weaker, and
checkable.

**An event-driven claim is a measurement, not an adjective.** `Mode::Clocked` and
`Mode::EventDriven` run the same network and are *required to produce the same spike train*; the
difference between them is `Ledger::idle_fraction`, a number. On the chain above the clocked run
spends over 80% of its membrane updates on neurons that received nothing, and the event-driven run
spends none — which is the case for the hardware, stated as a count rather than asserted.

And jumping a neuron across quiet ticks is only legal for a model whose state composes across an
interval. That is `Neuron::EXACT_OVER_GAPS`, declared per model and **enforced by `Sim::new`
refusing to build**:

| model | exact over gaps | why |
|---|---|---|
| `Lif`, `AdaptiveLif` | yes | exponential Euler is the exact solution, and exponentials compose |
| `IntegrateAndFire` | yes | linear, and motionless at zero current |
| `Izhikevich` | **no** | quadratic, forward Euler; one step of `2h` ≠ two steps of `h` |

Ask for an event-driven `Izhikevich` and you get `SimError::NotExactOverGaps`, not spike times that
depend on which ticks happened to be quiet.

## Verified against closed forms, not against yesterday's output

Every neuron model here has a test that runs it against an analytic solution.

- **Free decay matches the exponential to 1e-12**, not to a discretisation tolerance, because the
  integrator is exponential Euler. The same test run at `dt = 1e-5` and `dt = 1e-2` agrees to 1e-12,
  which forward Euler would fail — so you can coarsen the time step to save energy without silently
  changing the answer.
- **The simulated firing rate matches `Lif::isi`** — `τ·ln((v∞ − v_reset)/(v∞ − v_th)) + t_ref` —
  to 0.1% across 2–20 nA.
- **The perfect integrator spikes at exactly the predicted times**, within one tick, because a
  linear ramp has no error term to hide behind.
- **A sub-threshold current returns `None`, not a large number.** "Fires rarely" and "does not fire"
  are different statements and a rate-coded readout cannot recover the difference later.

1,544 unit tests and nineteen doctests, `cargo clippy --all-targets -- -D warnings` clean,
`#![forbid(unsafe_code)]`, and `cargo build --target wasm32-unknown-unknown` compiles the library
unchanged. Three of the `examples/` are verification gates that exit non-zero when a closed form
disagrees with the simulator.

## What the encoders cost, on the page

A spiking network cannot read a float, and the choice of how a number becomes spikes fixes how much
information survives, how long the answer takes, and how many synaptic operations it costs.

| scheme | spikes per value | latency | preserves |
|---|---|---|---|
| `RateEncoder` | many, Poisson | a whole window | magnitude, in the mean |
| `LatencyEncoder` | exactly one | one spike | magnitude, in the timing |
| `DeltaEncoder` | one per threshold crossing | immediate | change, not level |

`RateEncoder::ticks_for_precision` reports the number that is almost never stated beside a
rate-coded result: a Poisson count's relative error falls as `1/√n`, so **1% precision needs 10,000
spikes**, which at 100 Hz is 100 seconds. That is the arithmetic that makes latency coding
interesting and the reason a rate-coded input layer generates the synaptic operations a neuromorphic
energy figure is then divided by.

`RateEncoder` uses `1 − exp(−rate·dt)` rather than `rate·dt`. The naive form is 0.5% high at an
ordinary operating point and exceeds 1.0 outright at `rate·dt > 1`, where the encoder saturates at
one spike per tick while still reporting that it produces `max_hz`. Both facts are asserted in the
tests, in the second case against the series expansion so the size of the error is on the page.

`DeltaEncoder` is what an event camera does: it emits nothing at all for a static input, which is
why an event sensor staring at a still scene costs nothing and why it cannot tell you what it is
looking at until something moves.

## Zero dependencies, and that is a feature

`[dependencies]` is empty and stays empty. It is what lets this crate be audited end to end by one
person, compile to `wasm32-unknown-unknown` without a toolchain argument, and still build in ten
years. Anything needing a dependency — a GPU driver, a power sensor, an FPGA toolchain, a TLS client
— is a sibling crate you opt into, and deleting every sibling leaves this one intact.

## Determinism

Same seed, same spikes, every platform. `rng` is the only source of randomness in the crate and it
takes a seed; nothing here reads a clock or the operating system's entropy. A spike train that cannot
be reproduced cannot be checked against anything, including itself.

## Status

**0.6.0, with the third wave landing.** Thirty-six modules audited and repaired, four more since
(`nef`, `vsa`, `sparse`, `resonate`) written against closed forms and not yet audited by mutation;
the crate family is not built yet. Planned
siblings, each following the same rule that a dependency lives outside the core:

| crate | what it would add | why separate |
|---|---|---|
| `ferromorphic-gpu` | spiking kernels on WebGPU | needs `wgpu` |
| `ferromorphic-meter` | joules **measured on the machine that ran it** | needs a power sensor |
| `ferromorphic-silicon` | FPGA spiking fabrics and bitstreams | needs the FPGA toolchain |
| `ferromorphic-serve` | an HTTP and MCP surface | it is a binary, not a library |

## Every module has been audited, and the audit found things

0.4.0 shipped fourteen modules that were each green in isolation. An adversarial auditor then read
all fourteen looking specifically for tests that cannot fail, constants transcribed rather than
checked, and silent failure paths. **It found real defects in every one.** 0.5.0 is the repair: 237
findings fixed, 39 disputed with evidence and left alone, tests from 583 to 792.

The three that mattered most, all shipped in 0.4.0 and all now fixed:

- **Two `EXACT_OVER_GAPS = true` that were false** — `SpikingRelu` and NIR's `CubaState`. That
  constant is a safety property: `Sim::new` refuses `Mode::EventDriven` for a model where it is
  false, so a wrong `true` silently permits spike times that depend on which ticks happened to be
  quiet. Both are now `false`, each with a test that demonstrates the divergence rather than asserting
  the constant.
- **`Eif::isi` panicked** on parameters `Eif::new` accepts. `f64::clamp` panics when its bounds
  cross, and a reset above the truncation point crosses them — which the shipped firing-pattern
  taxonomy already does.
- **AEDAT 2.0 ate valid records.** Any event whose first byte is `0x23` was consumed as a `#` header
  line.

The repair brief forbade the obvious cheat — making a finding go away by loosening a tolerance or
deleting an assertion — and a second pass diffed every test module against the original to check.
Where a repairer believed the auditor wrong, they were asked to say so with evidence rather than
"fix" correct code; 39 findings were refused that way.

### The second wave, 0.6.0

0.5.0 to 0.6.0 added fourteen modules — from `spikeconv` and `attention` to the `cochlea` and the
olfactory bulb — and the same adversarial audit was run on every one of them, this time by
**mutation**: each auditor edited the module under test one line at a time and re-ran the suite,
so a "finding" is a specific edit that left every test green. Tests went from 792 to 1,497, and
every test added in the repair was itself run against the mutation it exists to catch before it
was kept.

What the mutations found, the ones that changed a number rather than a comment:

- **`fusion`'s error bar reported 0.37 of the true scatter.** `mad / sqrt(pairs)` stood in for
  `1/(2f(0)·sqrt(n))` under a doc that said the two agree "to about 25%"; for a triangular
  difference they disagree by 41%, and `pairs` over-counted the independent events by a third.
  Measured over 300 seeds, then fixed: `2·mad / sqrt(events)`, a re-centred refinement window that
  cuts the estimate's own scatter to 0.6–0.67 of the single pass, and a calibration test that
  holds the reported-to-actual ratio inside `[1.0, 1.5]` (measured 1.17 and 1.25).
- **`spikeconv` charged a multiply-accumulate at the accumulate price.** A graded and a binary
  workload produced byte-identical ledgers — the crate had silently supplied exactly the AC:MAC
  ratio it declines to supply everywhere else. A residual block's second stage had no ledger at all.
- **`cochlea`'s Nyquist guard was argued, not measured.** At the old `0.45·fs` the driven peak sat
  1.9× further from `f_c` than the tolerance every other channel is held to. It is `0.40` now, the
  peak test runs at the guard, and every filter order from 2 to 6 is driven rather than only
  computed — four closed forms had carried an `n` no test had ever run.
- **`mapping` used half of Fennel's balance penalty** (`0.75` for the paper's `1.5`), reported the
  fan-in wall as a function of the placement, and wrapped its hop counts at `u64::MAX` spikes.
- **`control`'s push-pull controller had never been driven negative.** Delete the entire negative
  half and 45 tests stayed green while a controller asked to go down burnt 60,000 spikes and never
  moved the plant. The sigma-delta encoder overflowed `u64` in one call at a finite input, under a
  comment saying it would take 1e11 years.
- **`olfaction`'s learned-mask limb was never exercised with two odours**, so recall could deliver
  the wrong odour's mask; its ledger let half the loop's traffic be deleted; its gamma rhythm is
  imposed by a duty gate and the doc now says so.
- **`continual`'s stationary distribution degraded past depth 48** in every linear-algebra route
  tried; it is now solved by flux balance, exact to 1e-14 at every depth the `f64` transition
  matrix can represent, and refuses by name the depth it cannot.

## Not here, and said so

- **No hardware, and no measurement.** This crate simulates and counts; nothing in it has been on a
  meter, and every joule it reports is a count times a published price. `NeuroBench`'s system track,
  Rockpool's Xylo power readout and `BrainChip`'s tooling all measure more than this crate does.
  `ferromorphic-meter` is the planned sibling, and until it exists the ledger's refusal to total a
  bill without a fetch price is the most honest number here.
- **The crossover thresholds are other people's numbers.** `Evidence::Derived` on all three: they
  are analyses, not measurements, and the Yan band in particular is this review's reading of a
  sparsity figure rather than a number that paper prints. Read `Crossover::source` before quoting.
- **No device is claimed to be supported.** The price tables describe published figures for
  TrueNorth (2014) and Loihi (2018) and are graded as such. This crate does not talk to any chip.
- **The Izhikevich current scale is a convention.** The paper's `I` is not an ampere; the 1 nA ↔ 1
  unit mapping is stated in the source so a reader comparing against the paper knows what was done.
- **Two device tables is not a survey.** Per-operation energies for current commercial parts are
  quoted in units — TOPS/W, "1000× more efficient" — that do not reduce to a joule-per-operation
  figure anyone can put in a table. That shortness is a finding about the field, not a gap in the
  reading.

## Licence

Apache-2.0. Institute for Physical AI @ Bailey Military Institute.
