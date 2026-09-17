# ferromorphic

Neuromorphic computing in pure Rust. Spiking neuron models checked against their closed forms,
sparse event-driven networks with per-synapse delays, spike encoders that state what they cost, and
a first-class joules ledger that **charges for the memory traffic a synaptic operation needs** —
zero dependencies, std-only, wasm-clean, deterministic by seed.

The neuroscience is open and old: Lapicque's integrate-and-fire (1907), Hodgkin and Huxley (1952),
Mead's *Neuromorphic Electronic Systems* (1990), Mahowald's address-event representation (1992),
Izhikevich (2003), spike-timing-dependent plasticity (Bi and Poo, 1998). What a neuromorphic chip
accelerates is exactly these loops; what it charges for is moving the weights. Both belong in the
open commons, runnable on every compute fabric: CPU today, GPU and wasm in the browser, event-driven
silicon where there is silicon anyone can get.

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

72 unit tests and two doctests, `cargo clippy --all-targets -- -D warnings` clean, `#![forbid(unsafe_code)]`,
and `cargo build --target wasm32-unknown-unknown` compiles the library unchanged.

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

**0.2.0.** The core is real and tested; the crate family is not built yet. Planned siblings, each
following the same rule that a dependency lives outside the core:

| crate | what it would add | why separate |
|---|---|---|
| `ferromorphic-gpu` | spiking kernels on WebGPU | needs `wgpu` |
| `ferromorphic-meter` | joules **measured on the machine that ran it** | needs a power sensor |
| `ferromorphic-silicon` | FPGA spiking fabrics and bitstreams | needs the FPGA toolchain |
| `ferromorphic-serve` | an HTTP and MCP surface | it is a binary, not a library |

## Not here, and said so

- **No training.** No surrogate gradients, no ANN-to-SNN conversion, no plasticity rule yet. STDP is
  next; nothing in this release learns.
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
