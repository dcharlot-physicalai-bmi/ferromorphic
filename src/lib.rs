//! # ferromorphic — neuromorphic computing for Physical AI
//!
//! The IPAI @ BMI neuromorphic stack, in pure Rust: spiking neuron models checked against their
//! closed forms, sparse event-driven networks with per-synapse delays, spike encoders that state
//! what they cost, and — first-class, not an appendix — an energy ledger that **charges for the
//! memory traffic a synaptic operation needs** and refuses to report a figure when the device's
//! prices are unstated.
//!
//! The neuroscience is old and open: Lapicque's integrate-and-fire (1907), Hodgkin and Huxley
//! (1952), Mahowald's address-event representation (1992), Mead's *Neuromorphic Electronic
//! Systems* (1990), Izhikevich (2003), spike-timing-dependent plasticity (Bi and Poo, 1998). What a
//! neuromorphic chip accelerates is exactly these loops; what it charges for is moving the weights.
//! Both belong in the open commons, runnable on every compute fabric — CPU today, GPU and wasm in
//! the browser, event-driven silicon where there is silicon anyone can get.
//!
//! ## The position this crate takes
//!
//! **A synaptic operation count is not an energy measurement.** The standard figure in this field
//! is a SOP count multiplied by a datasheet joule-per-SOP. That prices the arithmetic and sets the
//! cost of fetching the weight to zero — and the fetch is the term that scales with where the model
//! lives rather than with how often it fires, so it is exactly the term that separates a benchmark
//! from a deployment. This Institute has made the analogous mistake before, in a different crate,
//! where omitting attention's score matrix made a published arithmetic-intensity figure wrong by
//! 847x while looking entirely reasonable.
//!
//! So [`ledger::Prices`] carries `e_syn_fetch`, **every device table in this crate leaves it
//! `None`**, and [`ledger::Ledger::joules`] therefore refuses. That is the finding, not an
//! unfinished implementation: *this review did not locate a published per-synapse memory-fetch
//! energy for any commercially available neuromorphic processor.* The flattering number the
//! literature reports is still computable, as [`ledger::Ledger::joules_synops_only`], with its doc
//! saying what it omits.
//!
//! **And the field already published the threshold that decides it.** The quantity that says
//! whether a spiking network can beat its dense equivalent is *spikes per synapse per inference*,
//! and at least six papers give a number for it — every one of them **below 2**, several below 1.
//! Davidson and Furber (Frontiers in Neuroscience 15:651141, 2021) derive ~1.72 and conclude that
//! "most rate-coded spiking network implementations will not be more energy or resource efficient
//! than the original ANN"; Steve Furber designed `SpiNNaker`, so this is the field auditing itself.
//! This review located that argument in the literature and **did not locate it implemented as a
//! check in any spiking-network library**, which is odd for a number that decides whether the whole
//! approach helps. [`crossover`] makes it a check: the left-hand side is a ratio of two integer
//! counts the simulator already keeps, so it costs nothing to run on every workload.
//!
//! **An event-driven claim is a measurement, not an adjective.** [`sim::Mode::Clocked`] and
//! [`sim::Mode::EventDriven`] run the same network and are required to produce the same spike
//! train; the difference between them is [`ledger::Ledger::idle_fraction`], a number. And an
//! event-driven simulation is only legal for a model that can be jumped across quiet ticks, which
//! is [`neuron::Neuron::EXACT_OVER_GAPS`] — declared per model and enforced by [`sim::Sim::new`]
//! refusing to build rather than by a warning nobody reads.
//!
//! ## Quickstart
//!
//! ```
//! use ferromorphic::{
//!     ledger::TRUENORTH_2014,
//!     net::NetBuilder,
//!     neuron::Lif,
//!     sim::{Mode, Sim},
//! };
//!
//! // A five-neuron chain: each cell drives the next after 2 ticks.
//! let mut b = NetBuilder::new(5);
//! for i in 0..4 {
//!     b.connect(i, i + 1, 20e-3, 2)?;
//! }
//! let net = b.build();
//!
//! // Drive the first cell with 3 nA and run for 200 ms at 0.1 ms per tick.
//! let mut ext = vec![0.0; 5];
//! ext[0] = 3e-9;
//! let mut sim = Sim::new(net, vec![Lif::default(); 5], 1e-4, Mode::EventDriven)?;
//! let train = sim.run(2_000, &ext);
//! assert!(!train.is_empty());
//!
//! // What it would cost on TrueNorth — and the answer is a refusal, with the reason.
//! let bill = sim.ledger.bill(&TRUENORTH_2014);
//! assert!(bill.total.is_none());
//! assert!(bill.unpriced.contains(&"synapse memory fetch"));
//! // The synaptic term alone still prices, so the refusal is informative rather than blank.
//! assert!(bill.synaptic.unwrap() > 0.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Zero dependencies, and that is a feature
//!
//! `[dependencies]` is empty and stays empty. It is what lets this crate be audited end to end by
//! one person, compile to `wasm32-unknown-unknown` without a toolchain argument, and still build in
//! ten years. Anything needing a dependency — a GPU driver, a power sensor, an FPGA toolchain — is a
//! sibling crate you opt into, and deleting every sibling leaves this one intact.
//!
//! ## Determinism
//!
//! Same seed, same spikes, every platform. [`rng`] is the only source of randomness in the crate and
//! it takes a seed; nothing here reads a clock or the operating system's entropy. A spike train that
//! cannot be reproduced cannot be checked against anything, including itself.

#![forbid(unsafe_code)]

pub mod aer;
pub mod attention;
pub mod bayes;
pub mod cochlea;
pub mod coding;
pub mod compress;
pub mod continual;
pub mod control;
pub mod convert;
pub mod crossover;
pub mod dendrite;
pub mod device;
pub mod encode;
pub mod eprop;
pub mod exponential;
pub mod fusion;
pub mod hardware;
pub mod hh;
pub mod hopfield;
pub mod ledger;
pub mod mapping;
pub mod meanfield;
pub mod metrics;
pub mod nef;
pub mod net;
pub mod neuron;
pub mod nir;
pub mod olfaction;
pub mod optimise;
pub mod phasor;
pub mod plasticity;
pub mod reinforce;
pub mod reservoir;
pub mod resonate;
pub mod rng;
pub mod sim;
pub mod sparse;
pub mod spike;
pub mod spikeconv;
pub mod surrogate;
pub mod synapse;
pub mod tasks;
pub mod topology;
pub mod touch;
pub mod vision;
pub mod vsa;

pub use crossover::{Crossover, Verdict};
pub use ledger::{Bill, Evidence, Ledger, Prices};
pub use net::{Net, NetBuilder, NetError};
pub use neuron::{AdaptiveLif, IntegrateAndFire, Izhikevich, Lif, Neuron};
pub use rng::Rng;
pub use sim::{Mode, Sim, SimError};
pub use spike::{Event, Polarity, Spike, Train};

/// This crate's version, for a run that wants to record what produced it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
