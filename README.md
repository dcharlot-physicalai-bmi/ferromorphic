# ferromorphic

**Neuromorphic computing in pure Rust — the open corpus in one library.** Spiking neuron models from
Hodgkin-Huxley to `AdEx`, synapse kernels, plasticity rules, surrogate-gradient and local learning,
ANN-to-SNN conversion, reservoir computing, every neural code, the standard topologies, spiking
convolutions and attention, a silicon cochlea and an olfactory bulb, event-based vision and
multimodal fusion, spiking control, mean-field theory, model compression and core mapping,
hardware constraint models, analog device non-idealities, NIR, event-camera decoders, benchmark
metrics, teaching tasks — and a joules ledger that charges for the memory traffic a synaptic
operation needs.

**Zero dependencies. `std` only. `wasm32` clean. Deterministic by seed. 71 modules, 1,829 tests.**

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
| `neuron` | LIF, integrate-and-fire, adaptive LIF, Izhikevich; and exact spike timing — counts and TIMES that do not depend on the tick — for the LIF and the adaptive LIF, whose adapted interval is the root of a self-consistency equation |
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
| `proprio` | proprioception: the power-law muscle spindle and the fusimotor gains that retune it (dynamic `γ_d`, static `γ_s`, and the synaptic gain of the reflex arc), an intrafusal fibre with its gamma drive, spike-driven fusimotor activation, the tendon organ and its two-rate overshoot, a rate-to-spike encoder that emits exactly the integral of its rate, and a stretch reflex whose stiffness is `g γ_s k_L` and whose clonus threshold is the delay's ceiling `π/2τ` on the PRODUCT of those gains |
| `delays` | delays as a resource: the spatiotemporal pattern a set of synaptic delays is matched to, a delay-learning rule that contracts every arrival's deviation by exactly `1 − η`, the `(D+1)ⁿ − Dⁿ` patterns a neuron can stand for, and the buffer bits that costs |
| `distance` | how different two spike trains are: the Victor–Purpura edit distance and the van Rossum distance (closed form against its own quadrature), vector strength against the jitter's characteristic function, the Fano factor of a clock, `f(1 − f)/(m + f)`, and the parameter-free ISI- and SPIKE-distances, integrated exactly and refereed by the quadrature of their definitions |
| `field` | the Amari neural field of dynamic field theory: the kernel integral `W`, the narrow unstable bump below which activity dies and the wide stable one it settles at, both as roots of `W(a) + h = 0` and both found in the simulated field — on a line, and on a sheet, where the rim integral is `πσ²[1 − e^{−R²/σ²} I₀(R²/σ²)]` |
| `resonance` | noise as a resource: the noise level at which a threshold unit best tells two sub-threshold values apart, `σ*² = (b² − a²)/(2 ln(b/a))`; the exact information in a noisy population's count, including suprathreshold resonance — 63 units carrying more than one bit only when noise is added; dither that makes a step linear |
| `ttfs` | learning with spike TIMES: two neuron models whose first-spike time is a closed form (Mostafa's non-leaky integrator; the leaky cell with `τ_m = 2τ_s`), its exact gradient by the implicit function theorem, backpropagation through spike times checked against finite differences, and XOR learned with one spike per neuron |
| `ssm` | diagonal state-space models: the S4D sequence model and a bank of multi-timescale synapses shown to be the same object — zero-order hold exact for held input, convolution and recurrence required to agree to rounding, and the continuous impulse response checked against `srm`'s postsynaptic potential |
| `sdr` | sparse distributed representations: the hypergeometric arithmetic of why a few active bits out of many are unconfusable, computed in logarithms so it holds at a hundred thousand bits, with subsampling, noise and union capacity |
| `srm` | the Spike Response Model: the kernel form the crate's time-coded learning is written in, checked against `eventprop`'s own integration, plus escape noise — a hazard, a survivor function and an interval distribution — and the exact term SRM₀ drops when the synapse has a time constant |
| `ottt` | online training through time: the gradient of a spiking layer computed FORWARD in constant memory, proven equal to the true gradient when the reset path is off and measured against it when it is on |
| `polychron` | polychronous groups: the time-locked patterns axonal delays buy, enumerated by simulation — with the count shown to be combinatorics rather than evidence, and a prediction about cascade length refuted by the measurement |
| `eventprop` | `EventProp`: exact gradients for LIF networks with exponential synapses whose neurons fire any number of times and may be recurrent — an event-driven forward pass with spike times found to the last bit, an adjoint that jumps only at the spikes, checked against a closed-form spike time and its derivative and against finite differences of every weight |
| `alignment` | learning without weight transport: feedback alignment and direct feedback alignment beside the backpropagation they replace — identical to it when the feedback IS the transpose, and measured aligning from the output downward |
| `decolle` | deep continuous local learning: every spiking layer descends its own loss through a fixed random readout — three local factors, no gradient between layers or through time — with the update checked as the exact gradient of that loss |
| `forwardforward` | Hinton's forward-forward algorithm: layer-local goodness on positive and negative data, the length normalisation that hides a layer's goodness from the next, labels written into the input |
| `force` | FORCE learning: recursive least squares — shown identical to ridge regression at every step — taming a chaotic rate network into holding a sine on its own, with the measured limit where it stops working |
| `biosignal` | the ECGSYN synthetic electrocardiogram with its published constants, a level-crossing encoder, a spiking R-peak detector and rhythm monitor that flags an ectopic beat, and the EMG envelope read straight off an event rate |
| `sim`, `net`, `spike`, `rng` | the simulator, clocked and event-driven, with optional exact spike timing (several spikes a tick, counts independent of the tick); sparse connectivity, spike trains, seeded PCG32 |

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

1,829 unit tests and nineteen doctests, `cargo clippy --all-targets -- -D warnings` clean,
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

**0.18.0.** Seventy-one modules, every one audited by mutation: thirty-six by an adversarial
auditor in 0.5.0 and 0.6.0, the ten of the third wave by hand in 0.8.0 (156 mutations, 36
survivors, every one now caught — see below), and everything since mutated as it was written.

**What of that is RE-RUNNABLE, which is a different question and the one that matters.** As of
2026-09-21, **all seventy-one**. The repository holds **5,826 recorded mutations, at least one
list per module**, and every one of them can be applied to today's source by
`python3 tools/mutate.py`. For six releases that sentence had an exception in it: thirty-six
modules were audited in 0.5.0 and 0.6.0 by an adversarial auditor that never wrote its edits down,
so their verdict was a historical claim about the code as it stood then rather than a property of
the code as it stands now. Closing that gap is what the backfill below was for, and it is closed.

Every one of the 999 mutations recorded before this release was re-run in full against 0.18.0:
**985 caught, 14 equivalent by their stated arguments, none survived, none stale.** That is the
whole record, not a sample of it. The crate family is not built yet. Planned
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
biased low, systematically.

**0.13.0 is the repair.** `Lif::step_exact` solves for the threshold crossing INSIDE the tick,
resets there, serves the refractory period and goes on with what is left of the tick. The spike
count over a run of constant current is then the closed form's at every tick tried, from 0.1 ms to
half a second — 175 spikes each time, where the tick-based step gives 174, 169, 166, 160, 83
and 2. A population's measured rates match the rate curve to one spike per window at a 10 ms
tick. And the spiking LMU at 1 ms goes from 47% error to 8.7% with 300 neurons and 1.7% with
1500 — which is the rate-mode network's error, the floor the mapping sets. More neurons now help
all the way down, because what was left was never noise. The tick-based `Neuron::step` is
unchanged, because 1,600 tests and every hardware model in this crate are defined by it; exact
timing is a second method, and the spiking loop's default.

0.14.0 takes it two steps further. `step_exact_times` returns the spike TIMES the membrane
equation gives — `isi − t_ref`, then one every `isi`, the same list at a 0.2 ms tick and a 250 ms
one — and a test across three modules feeds the exact times of two cells to the ISI-distance and
gets the closed form of their two intervals, where tick-boundary times give the distance between
12 ms and 21 ms instead of 11.4 and 20.3. And the ADAPTIVE cell gets exact timing too: its
crossing has no closed form, so it is bisected inside the tick on a bracket that holds exactly one
root — including the case where a cell fires on the rebound of its falling threshold, mid-tick,
and is below threshold again by the tick's end. What it settles into is the root of
`V_∞ + (V_reset − V_∞)e^{−(T − t_ref)/τ_m} = θ₀ + β/(e^{T/τ_a} − 1)`, and the simulated late
intervals equal that root to 1e-10 at every tick tried.

0.15.0 adds `ttfs`, the other thing exact spike times make possible: if the quantity a network
computes is WHEN each neuron first fires, that time is a smooth function of the weights, and its
gradient is exact — no surrogate. For two neuron models the first-spike time has a closed form
(linear in `e^{t/τ}` for Mostafa's non-leaky integrator; a quadratic in `e^{−t/τ_m}` for the leaky
cell with `τ_m = 2τ_s`, the model trained for the BrainScaleS-2 chip), checked against a
brute-force search of the explicit membrane for its first upward crossing; the gradient is the
implicit function theorem at that crossing, checked against finite differences of the spike time
and, through two layers, of the loss; and gradient descent on spike times solves XOR for both
models with at most one spike per neuron. 24 mutations: 23 caught, 1 equivalent by a stated
argument.

0.16.0 is the wave that closed the list this README had been carrying under "still missing". Six
modules and one simulator change, 229 mutations, 24 survivors — every one now caught — and 1
equivalent by a stated argument.

`eventprop` is the one that matters most: EXACT gradients for spiking networks, with no surrogate
anywhere. A spike is not differentiable, but a spike TIME is — it is the implicit solution of
`V(t) = ϑ` — and Wunderlich and Pehle's adjoint turns that into a backward pass that visits only
the spikes. `ttfs` had this for networks where every neuron fires once; this has it for neurons
that fire any number of times, reset, and feed each other in a loop. The forward pass finds each
crossing to the last bit by bisecting the interval before the potential's single maximum, and the
backward pass carries `λ_V` and `λ_I` back through the same intervals, jumping only where the
forward pass spiked. Checked against the closed-form spike time of the `τ_m = 2τ_s` cell and its
derivative, and then against central finite differences of EVERY weight of a recurrent network
whose six neurons each fire several times — with the perturbed runs required to produce the same
spikes in the same order, which is the condition under which a finite difference of a spike time
means anything at all.

Three more ways to learn without backpropagation join the two already here. `alignment` is
feedback alignment and its direct variant: the error crosses the network through a FIXED RANDOM
matrix instead of the transpose of the forward weights, which is the wiring a chip can actually
build. Two closed forms pin it — with `B = Wᵀ` it IS backpropagation entry for entry, and in a
LINEAR network direct feedback through the collapsed product `W₂ᵀW₃ᵀ⋯` is backpropagation too —
and then the alignment itself is measured: the layer nearest the output starts unrelated to the
gradient and ends about 60° from it, while the layer below is still unaligned, so alignment
arrives from the output downward. `decolle` gives every spiking layer its own loss through a fixed
random readout: three factors available at the synapse, nothing stored over time, nothing from the
layer above — and the update is checked as the exact gradient of that layer's own loss, with a
test that scrambles every layer above and requires the update below to come back bit for bit
identical. `forwardforward` is Hinton's two-pass algorithm, whose length normalisation is the
subtle part: scaling a layer's weights by `c` multiplies its goodness by `c²` and leaves what it
passes up untouched, so the layer above cannot read its predecessor's answer.

`force` completes the set from the other end — the readout that is not trained by gradient descent
at all. Recursive least squares is shown to BE ridge regression at every step, against a Cholesky
solve that shares no code with it, and each update is shown to divide its own error by exactly
`1 + rᵀPr`. A 300-unit chaotic network learns to hold a sine and then holds it with learning off,
its chaos gone. The limit is measured and shipped as a test rather than left out: at `g = 2` the
same network, seed and ridge do NOT learn it, and `a_network_that_is_too_chaotic_is_not_tamed`
pins that.

`biosignal` is the application these were missing: the ECGSYN model with the constants read out of
the authors' own `ecgsyn.m`, a level-crossing encoder whose event count is the signal's total
variation over `δ`, a one-integrator R-peak detector with a rate threshold in closed form, and a
rhythm monitor that catches a premature beat and its compensatory pause. Two findings came out of
building it. The detector fires on the R wave's UPSTROKE, a few milliseconds before the peak, and
because every wave in this model scales with the beat's length, a premature beat is a SMALLER one
whose upstroke is caught later in its own wave — so intervals between beats of different lengths
carry a bias that the test states as a bound rather than hides in a tolerance. And `ecgsyn.m`
itself takes `rem(θ − θᵢ, 2π)`, which does not bring the difference into `(−π, π]`: the T wave's
tail is cut where the phase wraps. This module wraps it, and says so.

The simulator gained exact spike timing in both modes (`Sim::with_exact_timing`), which is the
0.13.0 repair carried into the network: a neuron may now fire more than once in a tick, every
spike is posted, and the spike count stops depending on `dt`. One thing it does NOT change is when
a spike ARRIVES — deliveries stay on the tick grid — and the module documentation says so rather
than letting the name imply more. The event-driven catch-up had to change with it: the exact
solvers keep the refractory clock in seconds, and a remainder of 4e-19 s is enough to make a
neuron ignore a delivery the clocked run accepted, so the refractory part of a gap is now walked
tick by tick and only the quiet remainder is jumped.

The survivors were the usual education. Three modules' `apply` could leave the biases exactly
where they were and every gradient test still passed. `decolle`'s network could hand back the
trace from AFTER the step instead of the one that produced the potentials, and nothing noticed
until a test asserted the identity `U = W P − ρ R + b` against the trace it returns. `force` could
feed back the UNCORRECTED readout — the one detail that makes FORCE work — and still learn the
sine. A detector that never emptied its integrator was invisible because the fixture's refractory
period happened to equal its recharge time. And two mutations of the exact catch-up survived
because every neuron in the fixtures was busy: the test that catches them drives one cell at just
over half the voltage it needs, so whether it fires is decided by how much leaked away in the
quiet.

0.17.0 adds three more: 123 mutations, 13 survivors — every one now caught — and 3 equivalent by
stated arguments.  `srm` is the formalism the rest of the crate's time-coded
learning is written in — the neuron as two kernels and a threshold — and writing it down produced
a correction to a claim that is usually made loosely. SRM₀ says a reset erases the past, and that
is exact for a delta synapse and FALSE for a synapse with a time constant: the reset empties the
membrane but not the synaptic current, so input that arrived before the spike keeps flowing in
afterwards. The term SRM₀ drops is `ε(t − t̂) · Σ_{f < t̂} w e^{−(t̂ − t_f)/τ_s}`, it is a few
percent of the potential in the fixture here, and adding it back closes the gap to rounding —
which is how the module knows it has named the right term rather than a plausible one. The
kernel itself is checked against `eventprop`'s integration of the same two equations, and the
escape-noise half against the exponential interval distribution it implies.

`ottt` is the current answer to the question `eprop` asks: can a spiking network be trained
without storing the run? Its gradient is accumulated forward, and the state it carries is two
filters of the input — ten numbers here, whatever the sequence length. The theorem is exact and
worth stating precisely: with the reset path off, the online gradient IS the gradient, checked
against central finite differences of the loss on every weight and, separately, against
backpropagation through time on the same layer. With the reset on it is not, and the gap is
measured rather than waved at. Its forward pass is required to agree with `surrogate::LifLayer`'s
to the last bit, which is what makes the comparison a comparison of gradients rather than of two
different networks — and that requirement caught a one-ulp disagreement from summing the readout
in a different order.

`polychron` closes the gap `delays` had been carrying, and is the module that most changed its
mind. It enumerates the time-locked groups a network with axonal delays holds: anchors timed so
their spikes coincide at a shared target, plus everything that coincidence goes on to fire. The
count does exceed the neuron count — 3,736 groups for 120 neurons — but the module says plainly
what that number is not. With as many anchors as the firing threshold, the anchors are timed to
coincide BY CONSTRUCTION, so the target always fires and every anchor set sharing a target is a
group; the count is combinatorics of the connectivity and barely moves when the delays change
(3,736 spread, 3,556 narrow, 3,123 uniform). Worse for the expectation this module started with:
it was written expecting a spread of delays to make cascades LONGER, and the measurement says
the opposite. Four delays gave 167 groups of length four or more and a longest of 6; a single
delay gave 657 and a longest of 25. Identical delays put every arrival on one grid, so
coincidences downstream are easy, while scattered arrivals rarely land inside a 0.1 ms window
together. The test asserts the measured direction, and the documentation says which claim was
refuted.

0.18.0 adds two modules whose value is in what they identify with something the crate already had.

`ssm` is the diagonal state-space model of S4D, and the claim it makes is not an analogy: a bank of
multi-timescale synapses IS a diagonal SSM. One mode with `A = −1/τ` is an exponential synapse; two
with the right `C` are the double-exponential postsynaptic potential, and `double_exponential` is
checked against `srm::Kernel::epsilon` entry for entry. So the sequence-modelling literature's
initialisations are choices of synaptic time constants, and a spiking SSM is not a new mechanism
bolted onto a sequence model. Three things are pinned: zero-order hold is EXACT for input held
across the step (not an approximation, and the bilinear alternative's error is measured falling as
`Δ²`); convolution and recurrence — the trainer's form and the chip's form — agree to rounding;
and the continuous impulse response is kept distinct from the discrete kernel, because `K` answers
a held sample and `h` answers a delta, and they meet only in the limit, which is also measured.

Its audit found one thing worth the name. `stable()` asks whether `|Ā| < 1`, and under the
bilinear transform a pole can land on the NEGATIVE real axis outside the unit circle — `A > 2/Δ`
sends `(1 + AΔ/2)/(1 − AΔ/2)` below −1 — so a test on the value rather than the magnitude would
call a diverging model stable. Zero-order hold never does this, because `e^{AΔ}` is positive
whatever `A` is, so only the bilinear form can show it; the test now drives that mode and watches
it grow AND alternate.

`sdr` is the hypergeometric arithmetic of sparse distributed representations — why a few active
bits out of many are unconfusable, and exactly how much of the pattern may be thrown away before
they are not. It is computed in logarithms, which is what lets it answer at `n = 100,000` where
`C(n, w)` is past `10^2000`, and the combinatorics are checked against directly enumerated bit
vectors so that the counting and the bits agree.

It also contains a correction this crate made to itself. The module was first written quoting a
published false-positive rate of `10⁻¹⁰` for `n = 2048, w = 40, θ = 20`. Its own arithmetic gave
`2.5 × 10⁻²⁶`, so the source was checked — and the abstract does not carry the numerical examples,
and this review has no access to the body. No figure in the module is now attributed to the paper.
What it states are its own measured values, and the ordering among them is not the obvious one and
is asserted: demanding ALL of a twenty-bit subsample (`2.2 × 10⁻³⁷`) is rarer than demanding ANY
twenty of the forty (`2.5 × 10⁻²⁶`), because there are fewer ways to do it.

Ten of `sdr`'s forty-five mutations survived its first audit, and five of them turned out to be
genuinely equivalent — a clamp that no configuration can reach, a loop bound past which every term
is zero, a guard that returns what an empty sum returns anyway, a cap that the expectation it caps
can never exceed, and a two-cursor walk that gets the same answer with one cursor because both
lists are strictly increasing. Each is recorded with its argument. The other five were real gaps,
and all of the same kind: they were invisible to a SINGLE draw. A subsample taken from the wrong
end satisfies every property except being the one documented; noise that drops the wrong bits
still drops the right NUMBER of them. The tests now draw thousands of times and check the
distribution rather than the instance.

The last of those five is worth stating because the first repair for it was WRONG. The mutation
draws the shuffle's swap partner from the whole pool instead of the unplaced part, and the repair
asserted that every draw still came back with the full complement of bits — on the theory that
duplicates would be de-duplicated away. They would not: a swap cannot produce a duplicate however
the partner is chosen, so the pool stays a permutation and the count is always right. What the
naive shuffle breaks is not the count but the DISTRIBUTION, and the second repair measures that
instead: at twenty bits with five active, correct sampling holds every marginal within a percent
of `w/n` and the naive version misses by more than half.

One more thing the audit found, and it was about the audit itself. Running every list against the
finished code showed nine entries whose text no longer matched the source EXACTLY ONCE — seven of
them stale since before this release, because a later edit had duplicated the line they anchor to
(a second `Regime::Bistable` when the two-dimensional field arrived, a second `spikes.saturating_add`
when the adaptive cell got its exact solver) or replaced it outright (`nef`'s spiking loop, rewritten
in 0.13.0 to count spikes rather than test a boolean). The harness reports those as `NOT-APPLIED`
and refuses, which is why they were visible at all — but they had been sitting in the record as
though they were evidence. All nine now anchor to a unique site, and the four modules involved were
re-audited in full rather than spot-checked, because an entry that has never applied has never
caught anything.

0.18.1 is not a feature release. It is what happened when the audit record was audited.

Counting the lists in `tools/mutations/` against the modules in `src/` showed 38 lists for 71
modules. The other 33 had been audited in 0.5.0 and 0.6.0 by an adversarial auditor whose edits
were never written down — so "every module audited by mutation" was true as history and
unverifiable as a present-tense claim, and this README had been saying that the harness and every
mutation it has run were in the repository. That sentence has been corrected, and the thirty
modules still without a list are named, because a reader who runs the harness should know what
fraction of the claim it checks.

Then the whole recorded record was re-run against 0.18.0: **999 mutations, 985 caught, 14
equivalent by their stated arguments, none survived and none stale.** That is the first time the
entire record has been re-run in one pass.

And three of the thirty were written: `rng`, `spike` and `encode`, 72 mutations between them. The
result is the argument for doing the other twenty-seven. `spike` had TWELVE of its twenty-two
mutations survive, and `encode` seven of twenty-nine — a coefficient of variation that could
report a variance, a standard deviation or a population estimator and pass either way, because
the only test of it used a perfectly regular train whose variance is zero however you normalise
it; interval extraction that could read every source at once; a `len` that could report the
capacity; a latency window that could span the wrong number of gaps. None of that was caught by
the tests those modules already had.

`rng` was worse, and it was not the tests. Writing its mutations meant reading `below` closely
enough to check what each edit would break, and the rejection zone was wrong:
`u32::MAX - u32::MAX % n - (n - 1)` leaves an accepted count of `(k − 1)n + 2`, which is two too
many for every `n > 2` — a residual modulo bias where the documentation said there was none — and
which collapses to an accepted count of exactly TWO once `n` passes `2³¹`, where the loop then
spins about two billion times per draw. Nothing in the crate calls `below` with an `n` that large,
which is the only reason it had never been seen. It is now `u32::MAX - 2³² % n`, the loop is
bounded so that a wrong zone says so instead of hanging, and the module has the thing it most
needed and did not have: a golden vector. Its own documentation opens by promising the same seed
gives the same sequence on every platform and every release, and not one test pinned a value —
every test asked whether the stream was well behaved, which a different generator would also be.
The expected values were computed by a separate implementation of PCG32 XSH-RR written from the
algorithm rather than from this code.

`net` and `crossover` were the next two lists written, and both came back the same way the first
three did: six of `net`'s twenty-six mutations survived and two of `crossover`'s twenty-one. `net`
could read every row of its sparse index from the start of the array, report an out-degree that
ran past the end, divide its fan-out the wrong way round, return a `NaN` for an empty network, and
accept a neuron index exactly equal to the neuron count — none of which its tests could see,
because every one of them used the FIRST row of a network whose neuron count and synapse count
happened to differ from nothing. `crossover` could not see the thing it exists for: nothing
asserted the evidence GRADE on a published threshold, so a table could quietly upgrade an analysis
to a measurement.

One more thing happened while that work was going on, and it belongs here because it is a finding
about method rather than about code. A subagent asked to READ a module and return a mutation list
as data instead ran the mutations against the live working tree, and when it was stopped it left
one behind. The mutant it left was the worst one available: `Evidence::Simulated` to
`Evidence::Measured` on the Loihi price table — the single line that keeps this crate from saying
a pre-silicon simulation was a measurement. Three tests failed instantly and the tree was
restored, which is the system working; but the reason `tools/mutate.py` writes an in-flight marker
and restores on its next start, and the reason `tools/slim.py` exists to give it a COPY to work
on, is exactly this. A mutation harness that edits the tree you are about to commit is one
interruption away from shipping the bug it was hunting.

`ledger` and `metrics` were the next two, and `ledger` is the one that mattered: THIRTEEN of its
thirty-seven mutations survived. This is the module the crate is named for in argument — the one
that refuses to price a workload it cannot price — and nothing tested the ORDER of its evidence
ladder, nothing checked that a device table carried the grade its source earns, nothing asked
whether each of the five terms individually reaches the total, and nothing held a bill's grade to
the price it was built from. A ladder in the wrong order silently promotes a claim; a bill graded
`Metered` because that is what its accumulator starts from is the exact failure the type exists to
prevent.

Writing those tests also found a disagreement inside the module. `Prices::is_complete` and
`Prices::unpriced` asked only whether an optional price was INHABITED, while `Ledger::bill`
charges one only under `Some(e) if e.is_finite()`. So a table carrying `Some(NaN)` called itself
complete, listed nothing as unpriced, and then produced a bill that refused and named the term —
two answers to the same question under different names. Both now go through one `priced` helper.

`metrics` came back with three survivors, two of them worth the name: R-squared centres its total
sum of squares on the TARGET's mean, and every existing test used predictions whose mean was the
target's — a perfect fit, the mean itself, a symmetric error — so all of them passed with the
centre taken from the wrong list. The new fixture has means of 2 and 4, where the two choices give
−0.75 and 0.3.

`synapse` (81 mutations) and `plasticity` (99) are the two largest modules in the crate and came
back with six survivors between them. Three of `synapse`'s mutations are not caught by a test at
all — they are caught by the **compiler**, because the receptor table's claims about itself are
written as `const { assert!(..) }` and a mutant that falsifies one cannot be built. The harness
used to report that as `COMPILE-ERROR`, the verdict that means "your edit was malformed, it says
nothing either way", and it sent this audit looking for a broken list entry three times before the
error code was read. It now distinguishes them: `caught(const)` for an `E0080` evaluation panic,
`COMPILE-ERROR` for everything else.

What the six survivors were:

- **`RECEPTORS` was never asked what is in it.** The table's doc names four rows in a stated order
  and quotes the factor of 100 they span; the test iterated the array and asserted properties of
  each row separately, all of which are true of `[AMPA, NMDA, GABA_A, GABA_A]`. The slow inhibitory
  row could be dropped and the fast one repeated with nothing failing.
- **The `GABA_B` cascade's removable singularity.** The first mutation written for it — keep the
  accurate `expm1` form and switch on the rate difference instead of on `z` — survived, and it is
  *equivalent*: the two branches differ by less than `1e-9` relative. The threshold was never what
  was wrong with the first version of that code. The **subtraction** was, and the mutation that
  restores `(er - e4)/(k4 - a)` is caught by the same test whose doc says so, on the assertion that
  the centring error must shrink quadratically.
- **`MultiplicativeDepression`'s factor is the weight in its own unit, not a fraction of the span.**
  Every fixture in `plasticity` bounds weights to `[0, 1]`, whose span is exactly one, and the one
  test that varies the floor asserts only ratios of two steps — a common normalisation cancels out
  of a ratio. Dividing by the span passed the whole module. The new test uses a span of four and
  compares against the additive rule at one unit above the floor, where the factor is exactly one.
- **A refused spike was allowed to register itself, three times over.** `on_pre` and `on_post` both
  document that a refused call leaves the traces exactly as it found them. The test that checked it
  asserted on the trace the call *reads* rather than the one it *writes*: after a refused `on_pre`
  it checked the post trace, and after a refused triplet `on_post` it checked `r1` — neither of
  which the registration touches. Moving `note_pre` above the guard passed. That is a new entry in
  the vacuous-test register.
- **The tag and the modulator accumulate, and nothing had ever delivered two of either.** `c += ..`
  and `d += ..` both read as `=` under a test that plays exactly one pair and delivers exactly one
  reward, which was every test in the module. A tag that is overwritten turns a burst into its last
  pair alone — the whole quantity a three-factor rule exists to carry.

`hh` (101 mutations) and `surrogate` (128) came back with sixteen survivors between them, and the
two modules failed in the same two ways.

**A parameter that is 1 in every fixture makes the operation on it invisible.** `hh`'s
`derivatives` divides `dV/dt` by the membrane capacitance; every cell in the module is the default,
whose `c_m` is exactly `1.0`. Dropping the division passed both convergence tests and the
independent-integrator cross-check. The new test runs the same cell at `c_m = 2` and asserts the
initial slope halves exactly. This is the third module in a row with this shape — `plasticity`'s
span was 1, `synapse`'s receptor table was never asked what was in it.

**An assertion made of differences cannot see a constant.**
`the_antiderivative_is_the_integral_of_the_backward_pass` checks the rise `Phi(big) - Phi(-big)`
against the analytic mass and the slope by central differences. Both are differences. `ArcTan`
could drop the `0.5` that makes `Phi(-inf)` zero, `StraightThrough` its `+ half_width`, and
`Triangular` the offset from either interior branch — the last two of which are not a shift at all
but a JUMP inside the support, which the twelve central-difference probes are positioned to miss
because they are positioned to miss the kinks. The trait doc has said `Phi(-inf) == 0` all along.
The new test asserts that absolutely, asserts `Phi(+inf) == mass` absolutely, and walks 40,000
points asserting `Phi` never rises by more than `step * peak` — the bound an integral of a bounded
function obeys, and the one a jump breaks.

The rest, briefly:

- **`Scaled` was in no property test.** Every one of them iterates `catalogue()`, which is the eight
  published families and not the wrapper. Three survivors lived there: an unscaled antiderivative, a
  width that moved with a vertical scale, and a `NaN` gain from sharpening a zero-mass inner at
  constant mass. The fixtures now run `Scaled` too.
- **The logistic's two branches are not cosmetic.** `1 / (1 + exp(-z))` overflows its denominator
  below `z = -709.78` and returns exactly `0.0` where the true value is a representable subnormal;
  measured across `[-900, 900]` the two forms differ by up to 25% relative. A `SigmoidDeriv` neuron
  710 thresholds below firing gets a real gradient from the branch form and none from the naive one.
- **The spike detector re-arms below `detect_reset`, not below `v_detect`,** and on a healthy action
  potential the two are indistinguishable — a spike that reaches +40 mV passes -20 mV on the way
  down anyway. They part on the small oscillation, which is the regime `voltage_range_mv`'s doc is
  about. Tested now against `detect_crossing` itself on a synthetic trajectory.
- **`repetitive_onset_ua_cm2` scans before it bisects**, and every fixture passed an `i_max` of 40 —
  inside the firing band, where a plain bisection happens to land on the same answer. At `i_max =
  200`, above the band, a plain bisection converges on `i_max` itself. The method's own doc warns
  about exactly this and nothing held it to the warning.
- **`Integrator::Rk4`'s order was never measured.** Uniform `(1,1,1,1)/4` weights — a consistent
  second-order method — passed the cross-check against the exponential scheme at both steps it uses.
  The error ratio under step halving separates them: 16 for fourth order, about 4 for second.
- **A `substeps` of zero froze the cell and reported `Ok`.** Both models clamp it with `.max(1)`;
  nothing checked that the clamp was on the loop bound and not only on the step width.
- **`ReducedHh`'s `Neuron` boundary is a second copy of the unit conversions** and the module's SI
  test only ever exercised the full model's.
- **A real fix, not a test.** `bisect_rest` chose its half of the bracket with `flo * fmid <= 0.0`.
  Two same-signed values below about `1e-162` multiply to `+0.0`, which satisfies `<= 0.0`, so the
  loop keeps the half without the root. It is sign comparisons now, and the fixture scales a root
  into that range.

### The backfill, finished (2026-09-21)

Twenty-two modules had no recorded list. Writing them by hand was running at two modules a session,
so this round used **one reader per module in parallel**, each with the same brief: read the module
and its tests, aim one mutation at every claim the docs make, verify every anchor occurs exactly
once, and **write nothing into the repository** — return the list as data. That last rule is not
politeness. An earlier subagent asked to draft a list had instead run the harness against the live
tree, and a kill between mutate and restore left a live mutant in `src/`.

**4,228 new mutations.** Eight anchors in `vision` came back ambiguous — the same statement appears
in the production path and again in the test module's independent reimplementation of it — and were
widened by hand. Nothing else needed fixing.

Then the audits, six at a time against slim copies. **The survivor rate is 12 to 13 per cent**,
several times the rate the crate's hand-written waves were finding. That number is the honest size
of what an audit that does not write its edits down leaves behind.

Repairs run the same way: one agent per module, each in its own crate copy, iterating until its
module's full list comes back with zero survivors. What they found, in the first eleven:

- **`convert::Reset::spikes_in` read an OVERFLOWED interval as saturation.** It returns `ticks`
  when the inter-spike interval `ceil(1/z)` is not finite — but the only way that happens for an
  activation the method has already accepted is that `1/z` overflowed, at `z` below about
  5.6e-309, and an infinite interval means the unit **never** fires. Measured: at `z = 1e-320` over
  1000 ticks the method returned `Some(1000)`, one spike on every tick, where the membrane gains
  1e-320 of a threshold per tick and the answer is `Some(0)`. The module's own asymptotic form for
  the same rule, `Reset::rate_limit`, returned `Some(0.0)` on that input — so the exact count and
  the asymptote disagreed by the full dynamic range, and the documented `floor(T / ceil(1/z))`
  sided with the asymptote.
- **`olfaction::Epl::new` named the wrong parameter.** A caller who set `recall_cycles: 0` was
  handed `field: "learn_cycles"` and sent to look at a parameter that was already correct, which
  defeats the only reason the field is in the error.
- **`compress::softmax_t`'s `# Errors` was missing a variant it really returns.** At a temperature
  of 1e-308 — finite and strictly positive, so the function accepts it — a logit of 1e10 overflows,
  the shift becomes `inf - inf`, and the sum is `NaN`. Refusing is right, and now documented, with
  the reason: once two logits have both overflowed, a uniform distribution and a one-hot are the
  same pair of infinities.
- **Two recorded equivalence arguments turned out to be FALSE and were retracted.** Both argued
  from the fixtures rather than from the arithmetic — `compress`'s said in so many words "every
  fixture in this module has a positive scaled maximum" — and both fell to a fixture the repair
  agent built specifically to break them. A retracted equivalence is the audit record getting
  stricter, and the merge script now accepts a retraction while still refusing any edit that
  changes an entry's label, `old` or `new`.

Two things about the harness came out of the same week, and both were reachable rather than
theoretical:

- **A killed test run was being counted as a catch.** `mutate.py` read "non-zero exit and no test
  result line" as `caught`. On a shared machine a sibling process's `pkill -f "cargo test"` turns
  every mutation it interrupts into a mutation this repository believes is covered — the one
  failure mode a mutation harness must not have. A signal-terminated child is now `KILLED`, which
  counts as a failure of the run, not as coverage.
- **A killed harness left a live mutant behind.** Python does not unwind `finally` for a signal it
  does not handle, so a `SIGTERM` between mutate and restore left the edit in place; the in-flight
  marker repaired it at the next start, but only for the same `--root` and only if someone ran it
  again. The harness now handles `SIGTERM`, `SIGINT` and `SIGHUP`, restores, and exits `128 + n` so
  a killed sweep can still be told from a finished one. **Read the exit code, not the tail** — six
  of this backfill's audits were killed partway through by exactly that command, and their logs
  look like completed runs.

`slim.py` gained one line for the same reason: it wrote `pub mod` for every module it kept but not
the crate's `pub use` re-exports, so a doc example that writes `ferromorphic::Rng::new(1)` did not
compile in a slim copy — invisible to the harness, which runs `cargo test --release --lib` and does
not build doc tests.

### Re-running the audit

The harness and every mutation THIS repository has recorded are in it: `tools/mutate.py` and one
list per module in `tools/mutations/`, for **all seventy-one modules**. There is no longer a
subset to name: the claim anyone can check by running the harness is the whole crate.
`python3 tools/mutate.py nef` applies each recorded edit to
`src/nef.rs` in turn, runs that module's tests, restores the file and prints `caught`, `SURVIVED`,
or — for the mutations no test could distinguish, each with its stated reason —
`equivalent`.
It prints the number of tests that ran unmutated first, because a test that has vanished looks
exactly like a test that passes.

`tools/slim.py` is what makes that affordable. The harness rebuilds the crate once per mutation,
and building all seventy-one modules takes about a hundred seconds; a copy holding only the module
under audit and the modules it names through `crate::` builds in about six. `python3
tools/slim.py . /tmp/slim_nef nef` writes that copy and `python3 tools/mutate.py --root
/tmp/slim_nef nef` audits it, which is the difference between one module in an afternoon and nine
at once. The copy is a build artefact; the lists in `tools/mutations/` are written against `src/`
and remain the record. The lists for the third wave were nearly lost: they lived in a
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
