# ferromorphic

**Neuromorphic computing in pure Rust — the open corpus in one library.** Spiking neuron models from
Hodgkin-Huxley to `AdEx`, synapse kernels, plasticity rules, surrogate-gradient and local learning,
ANN-to-SNN conversion, reservoir computing, every neural code, the standard topologies, spiking
convolutions and attention, a silicon cochlea and an olfactory bulb, event-based vision and
multimodal fusion, spiking control, mean-field theory, model compression and core mapping,
hardware constraint models, analog device non-idealities, NIR, event-camera decoders, benchmark
metrics, teaching tasks — and a joules ledger that charges for the memory traffic a synaptic
operation needs.

**Zero dependencies. `std` only. `wasm32` clean. Deterministic by seed. 59 modules, 1,693 tests.**

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
| `aer` | AEDAT 2.0, 3.1 and 4.0, Prophesee EVT2/EVT3 and N-MNIST decoders, total and panic-free, with rollover-correct timestamps; the DVS128 Gesture label reader that cuts a recording into its gestures |
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
| `nef` | the Neural Engineering Framework: tuning curves, least-squares decoders, factorised weights, PES decoder learning with its exact contraction factor, dynamics through a synapse, and a spiking recurrent loop run as a spiking Legendre Memory Unit — Nengo's three principles, each against its closed form |
| `vsa` | vector symbolic architectures / hyperdimensional computing: bipolar, binary and holographic models, codebooks, sequences, and a resonator network, with the bundle capacity as a binomial |
| `sparse` | sparse coding by local competition: the LCA and its spiking form, checked against the LASSO's own optimality conditions |
| `resonate` | resonate-and-fire neurons stepped by the exact complex exponential, and the Legendre Memory Unit as a delay line in `d` numbers |
| `hopfield` | associative memory as an energy landscape: classical, dense (Krotov-Hopfield) and modern continuous Hopfield networks, with the crosstalk, the 0.138 load and the one-step retrieval each measured |
| `dendrite` | two compartments: the soma's conductance-weighted steady state, learning by the dendritic prediction of somatic firing, and the XOR a point neuron provably cannot compute |
| `touch` | SA1, RA and Pacinian afferents as spiking cells driven by depth, velocity and vibration; a fingertip that locates a contact by its spike centroid and reports slip |
| `optimise` | QUBO by stochastic spiking: max-cut and graph colouring as energies, a Glauber sampler checked against the exact Boltzmann distribution, an annealer checked against brute force |
| `phasor` | the phasor (complex) vector symbolic architecture: binding as phase addition, symbols as spike times on a rhythm, and the threshold phasor associative memory |
| `reinforce` | three-factor learning: synaptic eligibility traces, a broadcast prediction error, TD(λ) and a softmax actor checked against the Bellman solution of a chain |
| `oscillator` | coupled phase oscillators: Kuramoto synchronisation against Adler's lock range and the Ott–Antonsen solution, a travelling-wave pattern generator for a segmented body, and an oscillator Ising machine checked against brute force |
| `predictive` | predictive coding: inference as the relaxation of error units onto the Bayesian posterior, local learning as the gradient of the free energy, and the limit — a weakly clamped output, not a small error — in which it becomes backpropagation |
| `graph` | graph algorithms done by spikes: a shortest path as the arrival time of a wavefront, exact against Bellman–Ford with the spikes counted, and boundary-value problems by random walkers against gambler's ruin |
| `attractor` | the ring attractor: a bump of activity that holds a heading after the cue is gone, with its width and height in closed form, and the amplified bump a weakly tuned input is sharpened into |
| `equilibrium` | equilibrium propagation: the gradient of a loss from the difference between two relaxations of one energy-based network, checked against the gradient obtained without it — first order in the nudge, second order when nudged both ways |
| `localise` | sound localisation by coincidence: the Jeffress delay-line array against the path-difference geometry, the half-spacing quantisation bound, the aliasing frequency and the coincidence probability under spike jitter |
| `cerebellum` | the cerebellum as a machine: an adaptive filter that converges on the Wiener solution at `1 − βλ` an epoch and stops when its error is decorrelated from every input, and Albus's CMAC with its triangular generalisation |
| `grid` | grid cells: the hexagonal firing map, path integration that is exact in phase space, the modular code — 9009 positions from 40 cells — decoded by the Chinese remainder theorem, and its error correction: two spare modules survive any corruption of any one, exhaustively |
| `proprio` | proprioception: the power-law muscle spindle, the tendon organ and its two-rate overshoot, a rate-to-spike encoder that emits exactly the integral of its rate, and a delayed reflex loop against the gain `π/2τ` at which it rings |
| `delays` | delays as a resource: the spatiotemporal pattern a set of synaptic delays is matched to, a delay-learning rule that contracts every arrival's deviation by exactly `1 − η`, the `(D+1)ⁿ − Dⁿ` patterns a neuron can stand for, and the buffer bits that costs |
| `distance` | how different two spike trains are: the Victor–Purpura edit distance and the van Rossum distance (closed form against its own quadrature), vector strength against the jitter's characteristic function, the Fano factor of a clock, `f(1 − f)/(m + f)`, and the parameter-free ISI- and SPIKE-distances, integrated exactly and refereed by the quadrature of their definitions |
| `field` | the Amari neural field of dynamic field theory: the kernel integral `W`, the narrow unstable bump below which activity dies and the wide stable one it settles at, both as roots of `W(a) + h = 0` and both found in the simulated field — on a line, and on a sheet, where the rim integral is `πσ²[1 − e^{−R²/σ²} I₀(R²/σ²)]` |
| `resonance` | noise as a resource: the noise level at which a threshold unit best tells two sub-threshold values apart, `σ*² = (b² − a²)/(2 ln(b/a))`; the exact information in a noisy population's count, including suprathreshold resonance — 63 units carrying more than one bit only when noise is added; dither that makes a step linear |
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

1,693 unit tests and nineteen doctests, `cargo clippy --all-targets -- -D warnings` clean,
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

**0.12.0.** Fifty-nine modules, every one audited by mutation: thirty-six by an adversarial
auditor in 0.5.0 and 0.6.0, the ten of the third wave by hand in 0.8.0 (156 mutations, 36
survivors, every one now caught — see below), and everything since mutated as it was written. All
290 mutations recorded before 0.10.0 were re-run against the finished code: 286 caught, 2
equivalent by their stated arguments, none survived. The harness and every mutation list — 501 of
them — are in `tools/`. The crate family is not built yet. Planned
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

### The third wave, 0.8.0

0.7.0 shipped ten modules — `nef`, `vsa`, `sparse`, `resonate`, `hopfield`, `dendrite`, `touch`,
`optimise`, `phasor`, `reinforce` — written against closed forms and said, in this file, that none
had been mutated. 0.8.0 is that audit, done by hand: one edit at a time, the module's tests re-run,
the edit reversed. **156 mutations, 36 survivors.** Thirty-five were tests that could not fail,
each now joined by one that can and that was re-run against its mutation before it was kept; one is
a quantity that is zero by construction, and the doc now says so instead of implying a check. The
sweep also turned up one defect in the code itself.

- **The defect: `optimise`'s 64-variable wall multiplied two `usize` unchecked.** A
  `vertices × colours` that wrapped came back small and was accepted. Found by reading the guard a
  surviving mutation pointed at.
- **A default that makes two parameters equal hides which is which.** `dendrite`'s round defaults
  have `g_D = g_L`, and with them four mutations survived: the two conductances swapped in the
  somatic prediction, their ratio inverted in the dendritic target, the sign of the prediction
  error, and a learning step that ignored `dt`. The same mechanism, elsewhere: `nef`'s radius of
  1, a `K` of 1 in the Ott–Antonsen test, an excitatory reversal of 0 V, a softmax policy at
  exactly ½ where `p` and `1 − p` are the same number.
- **A monitor with nothing to report on correct input had never been seen to fire.** `hopfield`
  reports the worst single-update energy rise; on a symmetric network that is always zero, so
  blinding it changed nothing. It is now fired on an asymmetric network. For the dense memory the
  same field is zero *by construction* — a flip is taken only when it lowers the energy — and the
  doc says that; what is checked there instead is the running energy against a direct evaluation.
- **Counters nobody read**: `sparse`'s idle/driven split and its spikes-per-burst, `touch`'s spike
  and tick counters, `optimise`'s flips and its restart totals.
- **Boundaries one past the edge**: a maximum rate *at* the refractory bound, a transition *to*
  state `n`, 65 variables, a membrane that lands *exactly* on threshold, an energy tie, and
  `phasor`'s `wrap`, which without its guard returns a full turn for an angle of `−1e-20`.
- **Two repairs were themselves blind** and were caught only because every repair is re-run
  against its mutation: a decode test that refused a two-colour vertex for the wrong reason, and a
  threshold-case test written at `K = 1`.

The four new modules were mutated as they were written, and the writing found things the mutations
did not: `predictive`'s first draft claimed the local updates approach backpropagation as the
output error shrinks — measured, the mismatch fell 1.01-fold for a tenfold smaller error, because
the hidden layer absorbs a fixed *fraction* of any error; the limit that works is a weakly clamped
output, and the module now says which. `attractor`'s first bump reported every neuron active:
Euler decay stalls on a denormal (`4 × 5e-324 × 0.9` rounds back to itself), so a silent neuron is
never silent until sub-normal rates are flushed. `graph`'s path reconstruction walked a corrupted
parent record for ever and was killed for the memory; it now refuses.

0.9.0 added `equilibrium`, `localise` and PES learning in `nef` the same way, and three more first
drafts did not survive their own tests. Equilibrium propagation's symmetric estimator is second
order in the nudge — but read between `β = 0.3` and `0.03` the ratio is 111, not 100, because the
`β⁴` term is 9% of the `β²` one out there; the test now measures the law where it holds and asserts
the far reading separately. The Jeffress array flagged an ambiguous reading only on an exact tie
between peaks, which cannot happen: on `N` cycles an alias peak is exactly one coincidence short of
the true one, so it now counts peaks at half maximum. And PES "within 3× of least squares after
400 sweeps" was a number with nothing behind it — measured, 3.6× — replaced by what is exact (it
never beats the least-squares floor) and what is measured and falsifiable (it keeps closing, slowly).

0.10.0 added `cerebellum`, `grid`, `proprio` and `delays`: 85 mutations, 2 survivors closed and 1
equivalent by a stated argument. Both survivors were the mechanism named above — a fixture at the
value that hides the term: quadrature inputs have no off-diagonal correlation, so a stability
bound that ignored the off-diagonals passed; and 15 ms − 12 ms is `2.9999999999999996` ms in
binary, so a test of a *closed* 3 ms window never stood on its edge until it was rewritten in
binary fractions. `proprio` ships the FORM of the cat hamstring spindle model with its exponent
and baseline, and leaves the two gains as parameters: this review confirmed the first two from
open sources and did not locate the others in one it could read.

0.11.0 added `distance`, `field` and `resonance`: 65 mutations, 2 survivors closed. A spike a
million cycles late lost 7e-10 of a radian when its phase was not reduced before multiplying by
`2π`, under a tolerance of 1e-9 — it is tested at 10¹² cycles now, where the loss is a
milliradian. And every bump in `field` sat mid-domain, where a kernel that forgot to wrap its
distances on the ring is indistinguishable; one now straddles the seam. Two first drafts fell to
their own tests again: `field` bounded the activation next to a bump's edge by the wrong
derivative (the slope there is `w(0) − w(a)`), and `resonance`'s doc claimed Stocks's
suprathreshold effect for a two-valued signal, which cannot show it — a straddling binary signal
is already one clean bit at zero noise. The doc now says what is shown and what is not.

0.12.0 added no modules; it finished eight. The Kreuz ISI- and SPIKE-distances, integrated exactly
and refereed by the quadrature of their own definitions; suprathreshold stochastic resonance with
a many-valued signal; the ring's answer to a tuned input; single-error correction on the grid
code, exhaustively; the Amari field on a sheet, through a Bessel function; the tendon organ's
two-rate overshoot; an `AEDAT` 3.1 decoder transcribed from the vendor's specification with the
DVS128 Gesture label reader; and a spiking recurrent loop run as a spiking Legendre Memory Unit.
62 mutations, 2 survivors closed. One was luck of exactly the kind this file keeps naming: the
`AEDAT` 3.1 round trip crossed a timestamp-overflow boundary on event 192, which is `3 × 64`, the
packet size — so the encoder's split at the boundary never ran. The test now asserts the crossing
is NOT on a packet boundary.

The spiking LMU measured something worth a line of its own. At a 1 ms tick the network's state
error was 0.47 of the state's size and more neurons did not help; at 0.1 ms it was 0.046. Rate
mode, which has no ticks to round, showed the mapping was right: the error was the TICK. A cell
firing at 300 Hz has an interval of 3.3 ticks, which the tick rounds up to 4, so every rate is
biased low, systematically. A spiking loop in this crate needs a tick well below its shortest
interspike interval, and `nef` now says so.

### Re-running the audit

The harness and every mutation it has run are in the repository: `tools/mutate.py` and one list per
module in `tools/mutations/`. `python3 tools/mutate.py nef` applies each recorded edit to
`src/nef.rs` in turn, runs that module's tests, restores the file and prints `caught`, `SURVIVED`,
or — for the two mutations no test could distinguish, each with its stated reason — `equivalent`.
It prints the number of tests that ran unmutated first, because a test that has vanished looks
exactly like a test that passes. The lists for the third wave were nearly lost: they lived in a
scratch directory that was cleaned, and were recovered from the session transcript. They are in
the repository now so that cannot happen twice, and so that anyone can check the claim above
instead of reading it.

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
