//! Network topology: the wiring, and the measurements that say what kind of wiring it is.
//!
//! # The lesson
//!
//! A spiking network is a dynamical system whose equations you already know — [`crate::neuron`]
//! has them — and whose behaviour you do not, because behaviour is set by **who is connected to
//! whom**. The same 1,000 leaky integrate-and-fire cells are a silent sheet, a synchronous
//! oscillator, a memory, or a classifier depending only on the adjacency matrix. That is why a
//! paper's methods section spends a paragraph on neuron parameters and a page on connectivity, and
//! it is why this module exists: the graph is the model.
//!
//! There is a small, standard set of graphs the field reaches for, and each one buys something at a
//! stated price:
//!
//! | family | what it buys | what it costs |
//! |---|---|---|
//! | Erdős-Rényi / Gilbert random | a null model — anything your structured net does, it must beat this | no locality, no hubs, nothing a cortex has |
//! | Watts-Strogatz small-world | short paths *and* dense local clustering at once | needs a rewiring parameter nobody can measure in tissue |
//! | Barabási-Albert scale-free | hubs, and a degree distribution with no characteristic scale | a hub's fan-out breaks a neuromorphic core's fan-in limit |
//! | distance-dependent (Maass) | the actual cortical statistic: connection probability falls with distance | needs coordinates, so the network has to be embedded |
//! | layered feedforward | trainable by backpropagation-through-time, mappable to a systolic array | no recurrence, so no memory in the wiring |
//! | winner-take-all | a decision, in spikes, with no readout layer | only works inside a stated inhibition regime, given below |
//!
//! # The two numbers that define a small world
//!
//! Watts and Strogatz (*Collective dynamics of 'small-world' networks*, Nature 393:440-442, 1998)
//! made one observation with two measurements. The **clustering coefficient** `C` is the
//! probability that two neighbours of a node are themselves neighbours — how cliquish the wiring
//! is. The **characteristic path length** `L` is the mean number of hops between two nodes — how
//! far information has to travel. A ring lattice has high `C` and terrible `L`; a random graph has
//! excellent `L` and a clustering no better than its density. Their finding is that rewiring a
//! **small fraction** of a lattice's edges takes `L` most of the way to the random-graph value
//! while `C` is still essentially the lattice's. That regime — high `C`, low `L` — is the small
//! world, and this module computes both numbers ([`clustering_coefficient`],
//! [`characteristic_path_length`]) so the claim can be checked rather than cited.
//!
//! **The size of the effect depends on the graph, so here is this module's own measurement rather
//! than the paper's.** At `n = 400`, `k = 10`, rewiring 1% of the edges costs 1.4% of the
//! clustering and 58% of the path length; the paper's `n = 1000` shows the collapse at a smaller
//! `beta` than 1% because `L` of a lattice grows with `n` while `L` of a random graph does not.
//! `watts_strogatz_reproduces_figure_2` sweeps `beta` and asserts the crossover exists.
//!
//! # Directed, so an undirected model becomes a reciprocal pair
//!
//! [`crate::net::Net`] is **directed** and stores each synapse **once** (see that module's doc for
//! how loudly it means it). Watts-Strogatz and Barabási-Albert are undirected models, so each
//! undirected edge is emitted here as **two** synapses, `a -> b` and `b -> a`. Their
//! [`crate::net::Net::n_syn`] is therefore twice the graph's edge count, and the measures in this
//! module ([`clustering_coefficient`], [`undirected_degrees`]) read the network as undirected by
//! taking the union of the two directions. [`path_stats`] does **not**: hops follow synapses, which
//! is what a spike does.
//!
//! # Dale's law, and why it gets its own checker
//!
//! A neuron releases the same transmitter at all of its terminals, so its outgoing synapses share
//! a sign: a cell is excitatory or inhibitory, not both. The principle is Dale's (H. H. Dale,
//! Proc. R. Soc. Med. 28:319-332, 1935) as reformulated by Eccles; Strata and Harvey (Brain Res.
//! Bull. 50:349-350, 1999) document that Dale himself said something narrower and that the modern
//! statement is a useful misreading.
//!
//! Every **[`Wiring`]-based** generator here is Dale-compliant by construction, because `Wiring`
//! assigns a sign per **presynaptic** neuron and never per synapse. [`winner_take_all`] is the
//! exception and takes no `Wiring`: with `w_self > 0` a unit excites itself while inhibiting
//! everyone else, which is a mixed-sign neuron and a Dale violation by construction — stated here
//! rather than discovered later, and asserted in
//! `winner_take_all_refuses_a_positive_inhibition_or_a_single_unit`. The reason [`dale_check`]
//! exists anyway is that a network which quietly violates Dale's law is **biologically meaningless
//! and computationally fine** — it trains, it fires, it produces a plausible raster — which is
//! exactly how such a network survives review. The violation has to be caught by an assertion or it
//! is not caught.
//!
//! The 80/20 excitatory/inhibitory split is the modelling convention, from Brunel (J. Comput.
//! Neurosci. 8:183-208, 2000, p. 185: "`N_E = 0.8N`, `N_I = 0.2N`", so `N_E = 4 N_I`) and Maass et al. (Neural Computation 14(11):2531-2560,
//! 2002, 20% inhibitory). **Caveat beside the figure:** the anatomy it abstracts varies by area and
//! species — Gabbott and Somogyi (Exp. Brain Res. 61:323-331, 1986) counted 20.6% (20.60 ± 0.48%,
//! mean ± SEM, five cats) of the neurons of cat area 17 as GABA-immunoreactive, which is the 80/20
//! convention itself rather than a figure below it (releases through 0.22.0 said "roughly 15%"
//! and cited them for it), and Braitenberg and
//! Schüz (*Cortex: Statistics and Geometry of Neuronal Connectivity*, 2nd ed., Springer, 1998) put
//! pyramidal cells near 85% of mouse cortex. 80/20 is a round number chosen inside that range, not
//! a measurement.
//!
//! **Correction, and a caution about second-hand citations:** an earlier revision of this doc
//! attributed that 15% to Beaulieu and Colonnier, J. Comp. Neurol. 231:180-189, 1985. That paper
//! is a laminar count of round-asymmetrical and flat-symmetrical **synapses** in cat area 17, not
//! a count of GABA-immunoreactive **neurons**; the neuron proportion belongs to Gabbott and
//! Somogyi. This module read neither source directly — both attributions are from the secondary
//! literature — so check them before a published figure rests on either.
//!
//! # Units, determinism, and what a hardware fabric will refuse
//!
//! Weights are **volts of membrane displacement per arriving spike** and delays are **ticks**, the
//! conventions [`crate::net`] fixes. Distances in [`Grid3`] are **metres**; Maass's dimensionless
//! `lambda = 2` is a distance in grid steps, so it becomes `2 * spacing` here and
//! [`Grid3::lambda_of`] does that conversion where a reader can see it.
//!
//! Every generator takes a `seed` and draws only from [`crate::rng::Rng`]. Same seed, same graph,
//! every platform — checked by `every_generator_is_deterministic_by_seed`, which builds each one
//! twice and compares the `Net` field for field.
//!
//! One cross-fabric note, since the point of this crate is that a model should run on whatever
//! silicon exists: a neuromorphic core has a **hard fan-in limit** (256 synapses per neuron on
//! `TrueNorth`, a configurable ceiling on `SpiNNaker` and `Loihi`), and a scale-free graph's hubs
//! are precisely the neurons that exceed it. [`crate::net::Net::in_degrees`] is how you find out
//! before the mapper does, and [`barabasi_albert`] is the generator most likely to trip it.

use crate::net::{Net, NetBuilder, NetError};
use crate::neuron::Lif;
use crate::rng::Rng;
use core::ops::Range;
use std::collections::BTreeSet;

/// Why a topology could not be generated.
///
/// Every variant names the offending value, because a generator that refuses without saying what it
/// refused sends the caller back to the source to guess which of six arguments was wrong.
#[derive(Debug, Clone, PartialEq)]
pub enum TopologyError {
    /// The network had fewer neurons than the construction needs.
    TooSmall {
        /// The neuron count that was asked for.
        n: usize,
        /// The smallest count this construction accepts.
        needed: usize,
        /// What needed them, e.g. `"watts_strogatz needs n > k"`.
        what: &'static str,
    },
    /// More neurons than a `u32` synapse index can name, which is the limit
    /// [`crate::net::NetBuilder`] imposes on every index it stores.
    IndexSpace {
        /// The neuron count that overflowed the index space.
        n: usize,
    },
    /// A probability argument was not a finite number in `[0, 1]`.
    Probability {
        /// Which argument, e.g. `"p"`, `"beta"`, `"MaassC::ee"`.
        name: &'static str,
        /// The value as supplied.
        value: f64,
    },
    /// `G(n, m)` was asked for more distinct edges than the pair space contains.
    EdgeBudget {
        /// Edges requested.
        requested: usize,
        /// Ordered pairs `(a, b)` with `a != b` available, which is `n * (n - 1)`.
        available: u64,
    },
    /// The Watts-Strogatz neighbourhood size was odd or zero. `k` counts **both** sides of the
    /// ring, so it must be even: `k / 2` neighbours clockwise and `k / 2` anticlockwise.
    RingDegree {
        /// The value supplied.
        k: usize,
    },
    /// A weight was not finite, or carried the wrong sign for the population it belongs to.
    Weight {
        /// Which weight, e.g. `"w_exc"`.
        name: &'static str,
        /// The value as supplied, volts per spike.
        value: f64,
    },
    /// A length argument was not a finite, strictly positive number of metres.
    Length {
        /// Which length, e.g. `"lambda"`, `"Grid3::spacing"`.
        name: &'static str,
        /// The value as supplied, metres.
        value: f64,
    },
    /// A per-neuron Dale partition whose length is not the network's neuron count.
    SignsLength {
        /// Signs supplied.
        declared: usize,
        /// Neurons the generator was asked to build.
        neurons: usize,
    },
    /// A layered network needs at least an input and an output layer.
    TooFewLayers {
        /// How many layer sizes were supplied.
        layers: usize,
    },
    /// A layer of zero neurons, which would silently disconnect everything downstream of it.
    EmptyLayer {
        /// Position of the empty layer in the slice.
        layer: usize,
    },
    /// The builder refused a synapse this generator produced. Unreachable by construction — the
    /// generators only emit in-range indices and finite weights — and propagated rather than
    /// unwrapped so that a future generator's bug surfaces as an error instead of a panic.
    Net(NetError),
}

impl core::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooSmall { n, needed, what } => {
                write!(f, "{n} neurons is below the {needed} this needs: {what}")
            }
            Self::IndexSpace { n } => {
                write!(f, "{n} neurons cannot be named by a u32 synapse index")
            }
            Self::Probability { name, value } => {
                write!(f, "{name} = {value} is not a probability in [0, 1]")
            }
            Self::EdgeBudget { requested, available } => {
                write!(f, "{requested} distinct edges asked of a pair space holding {available}")
            }
            Self::RingDegree { k } => {
                write!(f, "ring degree k = {k} must be even and at least 2")
            }
            Self::Weight { name, value } => {
                write!(f, "{name} = {value} V is not finite or has the wrong sign")
            }
            Self::Length { name, value } => {
                write!(f, "{name} = {value} m is not a finite positive length")
            }
            Self::SignsLength { declared, neurons } => {
                write!(f, "{declared} declared signs for a network of {neurons} neurons")
            }
            Self::TooFewLayers { layers } => {
                write!(f, "{layers} layer sizes given; a feedforward net needs at least 2")
            }
            Self::EmptyLayer { layer } => write!(f, "layer {layer} has no neurons"),
            Self::Net(e) => write!(f, "the network builder refused a generated synapse: {e}"),
        }
    }
}

impl std::error::Error for TopologyError {}

impl From<NetError> for TopologyError {
    fn from(e: NetError) -> Self {
        Self::Net(e)
    }
}

/// The Dale type of a neuron: which sign all of its outgoing synapses carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    /// Every outgoing weight is positive — the neuron depolarises what it contacts.
    Excitatory,
    /// Every outgoing weight is negative — the neuron hyperpolarises what it contacts.
    Inhibitory,
    /// No outgoing synapse carries a non-zero weight, so this implementation cannot assign a sign.
    ///
    /// Returned only by [`dale_signs`]; no generator here produces it as a *declaration*. It is a
    /// refusal, not a third biological class: a neuron with no output has no transmitter to be
    /// consistent about.
    Silent,
}

impl Sign {
    /// The sign as a multiplier: `+1.0`, `-1.0`, or `0.0` for [`Sign::Silent`].
    #[must_use]
    pub fn as_f64(self) -> f64 {
        match self {
            Self::Excitatory => 1.0,
            Self::Inhibitory => -1.0,
            Self::Silent => 0.0,
        }
    }
}

/// The inhibitory fraction of the standard 80/20 cortical split.
///
/// From the modelling convention — Brunel, J. Comput. Neurosci. 8:183-208, 2000 (`N_E = 4 N_I`) and
/// Maass et al., Neural Computation 14(11):2531-2560, 2002 (20% inhibitory). The anatomy behind it
/// is area- and species-dependent (20.6% in cat area 17); see the module doc. Dimensionless, in
/// `[0, 1]`.
pub const CORTICAL_INHIBITORY_FRACTION: f64 = 0.2;

/// Brunel's inhibition-to-excitation weight ratio for a network balanced at the 80/20 split.
///
/// `g = |J_I| / J_E` of Brunel, J. Comput. Neurosci. 8:183-208, 2000 — **not** a constant of Maass
/// et al. 2002, which defines no `g`: with four times as many excitatory neurons as inhibitory ones, the
/// mean drive on a neuron cancels exactly at `g = 4` — Brunel's "line `g = 4` is where feedback
/// excitation exactly balances inhibition" (p. 188). Dimensionless. [`Wiring::balanced`] applies
/// it and [`Wiring::expected_drive`] is the arithmetic that makes the cancellation checkable.
pub const BALANCED_G: f64 = 4.0;

/// How a generated graph turns into weighted, signed, delayed synapses.
///
/// This is the object that makes Dale's law structural rather than hoped for: the sign attaches to
/// the **presynaptic neuron index**, so no generator can emit a mixed-sign cell even by accident.
/// Neurons `0 .. n - n_inhibitory` are excitatory and the remaining block is inhibitory — a
/// contiguous partition, so `Wiring` plus `n` is enough to reconstruct which is which without
/// carrying a per-neuron vector around.
///
/// **Contiguity is a real modelling choice, not a detail, as soon as the index means something.**
/// [`Grid3`] maps index to position with x fastest, so a contiguous inhibitory block on a grid is a
/// spatial **slab** rather than a scattered 20%: for [`Grid3::maass_column`] all 27 inhibitory
/// neurons land on the single face `z = 2`. Under a distance-dependent rule that puts a systematic
/// drive gradient across the sheet — measured on that column at 1.60, −1.58 and −0.38 mV of mean
/// incoming weight by z-slab. Use [`shuffled_partition`] with [`distance_dependent_signed`] when
/// the geometry matters; see [`distance_dependent`]'s doc for which papers it matters for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wiring {
    /// Weight of a synapse whose **presynaptic** neuron is excitatory, in volts of membrane
    /// displacement per arriving spike. Must be finite and `>= 0`.
    pub w_exc: f64,
    /// Weight of a synapse whose **presynaptic** neuron is inhibitory, volts per spike. Must be
    /// finite and `<= 0`; the sign is checked rather than applied, so a caller who passes `+4e-3`
    /// gets an error instead of an excitatory "inhibitory" population.
    pub w_inh: f64,
    /// Delay of every synapse, in **ticks** (see [`crate::net::Net::delay`]). Uniform here; a
    /// caller wanting heterogeneous delays writes into `net.delay` afterwards, which is public for
    /// that reason.
    pub delay: u32,
    /// Fraction of the neurons that are inhibitory, in `[0, 1]`. [`CORTICAL_INHIBITORY_FRACTION`]
    /// is the 0.2 convention; `0.0` gives a purely excitatory network.
    pub inhibitory_fraction: f64,
}

impl Default for Wiring {
    /// 1 mV excitatory, −4 mV inhibitory, 1 tick, 20% inhibitory: the balanced cortical convention.
    ///
    /// `|w_inh| / w_exc = 4` against four times as many excitatory neurons means the expected
    /// summed input weight of a neuron is exactly zero ([`Wiring::expected_drive`]), which is the
    /// balanced-state setup of Brunel 2000 and van Vreeswijk and Sompolinsky (Science 274:1724-1726,
    /// 1996). 1 mV per spike is a round number in the cortical range, not a measurement.
    fn default() -> Self {
        Self {
            w_exc: 1e-3,
            w_inh: -BALANCED_G * 1e-3,
            delay: 1,
            inhibitory_fraction: CORTICAL_INHIBITORY_FRACTION,
        }
    }
}

impl Wiring {
    /// A network with no inhibitory population at all: every synapse is `w` volts per spike.
    ///
    /// The right choice for a feedforward classifier and for measuring a connection rule, where a
    /// second weight would confound the statistic being measured.
    #[must_use]
    pub fn excitatory_only(w: f64, delay: u32) -> Self {
        Self { w_exc: w, w_inh: 0.0, delay, inhibitory_fraction: 0.0 }
    }

    /// Excitatory weight `w_exc`, inhibitory weight set to `-BALANCED_G * w_exc`, 20% inhibitory.
    ///
    /// Balanced in expectation: see [`Wiring::expected_drive`], which returns exactly `0.0` for the
    /// wiring this returns at any `n` where the 80/20 rounding is exact.
    #[must_use]
    pub fn balanced(w_exc: f64, delay: u32) -> Self {
        Self {
            w_exc,
            w_inh: -BALANCED_G * w_exc,
            delay,
            inhibitory_fraction: CORTICAL_INHIBITORY_FRACTION,
        }
    }

    /// Reject a wiring whose weights are non-finite or wrongly signed, or whose inhibitory fraction
    /// is not a probability.
    ///
    /// # Errors
    ///
    /// [`TopologyError::Weight`] naming `w_exc` or `w_inh`, or [`TopologyError::Probability`]
    /// naming `inhibitory_fraction`.
    pub fn validate(&self) -> Result<(), TopologyError> {
        if !self.w_exc.is_finite() || self.w_exc < 0.0 {
            return Err(TopologyError::Weight { name: "w_exc", value: self.w_exc });
        }
        if !self.w_inh.is_finite() || self.w_inh > 0.0 {
            return Err(TopologyError::Weight { name: "w_inh", value: self.w_inh });
        }
        check_probability("inhibitory_fraction", self.inhibitory_fraction)
    }

    /// How many of `n` neurons are inhibitory: `round(inhibitory_fraction * n)`, clamped to `n`.
    ///
    /// Rounding rather than truncation so that the 20% of 10 neurons is 2 and not 1. A non-finite
    /// fraction is treated as 0 here rather than panicking; [`Wiring::validate`] is where it is
    /// rejected.
    #[must_use]
    pub fn n_inhibitory(&self, n: usize) -> usize {
        let f = if self.inhibitory_fraction.is_finite() {
            self.inhibitory_fraction.clamp(0.0, 1.0)
        } else {
            0.0
        };
        ((f * n as f64).round() as usize).min(n)
    }

    /// The Dale type of neuron `index` in a network of `n`: excitatory below the inhibitory block,
    /// inhibitory inside it. Never returns [`Sign::Silent`].
    #[must_use]
    pub fn sign_of(&self, n: usize, index: usize) -> Sign {
        if index < n - self.n_inhibitory(n) { Sign::Excitatory } else { Sign::Inhibitory }
    }

    /// The Dale type of every neuron, for handing to [`dale_check`].
    #[must_use]
    pub fn signs(&self, n: usize) -> Vec<Sign> {
        (0..n).map(|i| self.sign_of(n, i)).collect()
    }

    /// The weight of a synapse leaving neuron `pre` in a network of `n`, volts per spike.
    #[must_use]
    pub fn weight_of(&self, n: usize, pre: usize) -> f64 {
        self.weight_for(self.sign_of(n, pre))
    }

    /// The weight a synapse leaving a neuron of Dale type `sign` carries, volts per spike.
    ///
    /// `0.0` for [`Sign::Silent`], which is what "no transmitter" has to mean at a synapse: a
    /// weight of zero transmits nothing and carries no sign back through [`dale_signs`]. This is
    /// the entry point [`distance_dependent_signed`] uses, where the partition is a slice rather
    /// than an index range.
    #[must_use]
    pub fn weight_for(&self, sign: Sign) -> f64 {
        match sign {
            Sign::Excitatory => self.w_exc,
            Sign::Inhibitory => self.w_inh,
            Sign::Silent => 0.0,
        }
    }

    /// The expected sum of a neuron's **incoming** weights in a `G(n, p)` graph, volts per spike,
    /// averaged over the `n` neurons.
    ///
    /// `p * (n - 1) / n * (n_exc * w_exc + n_inh * w_inh)`. **The `(n - 1) / n` is the self term**:
    /// every generator in this module excludes self-loops, so neuron `i` has `n - 1` possible
    /// presynaptic partners and not `n`, and its own expectation is `p * (S - w_i)` with
    /// `S = n_exc * w_exc + n_inh * w_inh`. Averaging that over `i` removes the dependence on which
    /// neuron is asked and leaves `p * S * (n - 1) / n`. Dropping the factor overstates the drive
    /// by `1 / n` — 11.1% at `n = 10`, 0.200% at `n = 500` — and
    /// `the_expected_drive_counts_n_minus_one_presynaptic_partners` measures it exactly at
    /// `p = 1.0`, where the graph is complete-minus-diagonal and there is no sampling noise at all.
    ///
    /// This is the balance condition in closed form: it is zero exactly when `|w_inh| / w_exc`
    /// equals the excitatory-to-inhibitory count ratio, which is what [`Wiring::balanced`]
    /// arranges — the `(n - 1) / n` is a positive factor and moves no zero. A network whose
    /// expected drive is far from zero either saturates or falls silent, and which one it does is
    /// decided here rather than in simulation.
    ///
    /// `0.0` at `n == 0` and at `n == 1`: a lone neuron with no self-loop receives nothing.
    #[must_use]
    pub fn expected_drive(&self, n: usize, p: f64) -> f64 {
        if n == 0 {
            return 0.0;
        }
        let n_inh = self.n_inhibitory(n);
        let n_exc = n - n_inh;
        let self_excluded = (n - 1) as f64 / n as f64;
        p * self_excluded * (n_exc as f64 * self.w_exc + n_inh as f64 * self.w_inh)
    }
}

/// How a network violates Dale's law.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DaleViolation {
    /// One neuron's outgoing synapses carry both signs, which no transmitter does.
    Mixed {
        /// The offending presynaptic neuron.
        neuron: u32,
        /// How many of its outgoing weights are positive.
        positive: usize,
        /// How many are negative.
        negative: usize,
    },
    /// A neuron's weights contradict the partition it was declared to belong to.
    Contradicts {
        /// The offending presynaptic neuron.
        neuron: u32,
        /// The sign the caller declared.
        declared: Sign,
        /// The sign its outgoing weights actually carry.
        found: Sign,
    },
    /// The declared partition does not cover the network.
    LengthMismatch {
        /// Length of the declared slice.
        declared: usize,
        /// Neurons in the network.
        neurons: usize,
    },
}

impl core::fmt::Display for DaleViolation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Mixed { neuron, positive, negative } => write!(
                f,
                "neuron {neuron} has {positive} excitatory and {negative} inhibitory outgoing synapses"
            ),
            Self::Contradicts { neuron, declared, found } => {
                write!(f, "neuron {neuron} was declared {declared:?} and is wired {found:?}")
            }
            Self::LengthMismatch { declared, neurons } => {
                write!(f, "{declared} declared signs for {neurons} neurons")
            }
        }
    }
}

impl std::error::Error for DaleViolation {}

/// The Dale type of every neuron, read off the network's own weights.
///
/// Zero-weight synapses are ignored: a synapse that transmits nothing has no sign to be consistent
/// with. A neuron whose outgoing weights are all zero, or which has none, comes back
/// [`Sign::Silent`].
///
/// # Errors
///
/// [`DaleViolation::Mixed`] on the first neuron whose outgoing weights carry both signs, naming how
/// many of each it found.
pub fn dale_signs(net: &Net) -> Result<Vec<Sign>, DaleViolation> {
    let mut out = Vec::with_capacity(net.n);
    for pre in 0..net.n {
        let mut positive = 0usize;
        let mut negative = 0usize;
        for (_, w, _) in net.out_of(pre) {
            if w > 0.0 {
                positive += 1;
            } else if w < 0.0 {
                negative += 1;
            }
        }
        if positive > 0 && negative > 0 {
            return Err(DaleViolation::Mixed { neuron: pre as u32, positive, negative });
        }
        out.push(if positive > 0 {
            Sign::Excitatory
        } else if negative > 0 {
            Sign::Inhibitory
        } else {
            Sign::Silent
        });
    }
    Ok(out)
}

/// Verify a network against a declared excitatory/inhibitory partition.
///
/// A neuron that comes back [`Sign::Silent`] passes whatever it was declared: having no output is
/// not a contradiction of having a transmitter.
///
/// # Errors
///
/// [`DaleViolation::LengthMismatch`] if `declared` is not `net.n` long, [`DaleViolation::Mixed`]
/// from [`dale_signs`], or [`DaleViolation::Contradicts`] on the first neuron wired against its
/// declaration.
pub fn dale_check(net: &Net, declared: &[Sign]) -> Result<(), DaleViolation> {
    if declared.len() != net.n {
        return Err(DaleViolation::LengthMismatch { declared: declared.len(), neurons: net.n });
    }
    let found = dale_signs(net)?;
    for i in 0..net.n {
        if found[i] != Sign::Silent && found[i] != declared[i] {
            return Err(DaleViolation::Contradicts {
                neuron: i as u32,
                declared: declared[i],
                found: found[i],
            });
        }
    }
    Ok(())
}

/// The contiguous 80/20-style partition: excitatory block first, inhibitory block last.
///
/// # Errors
///
/// [`TopologyError::Probability`] if `inhibitory_fraction` is not a finite number in `[0, 1]`.
pub fn dale_partition(n: usize, inhibitory_fraction: f64) -> Result<Vec<Sign>, TopologyError> {
    check_probability("inhibitory_fraction", inhibitory_fraction)?;
    let w = Wiring { inhibitory_fraction, ..Wiring::default() };
    Ok(w.signs(n))
}

/// The same 80/20-style partition as [`dale_partition`], with the inhibitory neurons chosen **at
/// random** instead of as a trailing block.
///
/// Exactly `round(inhibitory_fraction * n)` neurons are inhibitory — the count is
/// [`Wiring::n_inhibitory`]'s, not a per-neuron coin flip — and *which* ones is a uniform
/// permutation drawn from `seed` by Fisher-Yates, so the same seed gives the same partition on
/// every platform.
///
/// **Why this exists.** A contiguous partition is fine when the index means nothing. The moment it
/// means a position — [`Grid3`] maps index to coordinate, x fastest — a contiguous inhibitory block
/// becomes a spatial slab, and a distance-dependent rule reads that slab as anatomy. Maass,
/// Natschläger and Markram choose their 20% inhibitory neurons at random, so this is the partition
/// to hand [`distance_dependent_signed`] when reproducing them.
///
/// # Errors
///
/// [`TopologyError::Probability`] if `inhibitory_fraction` is not a finite number in `[0, 1]`.
pub fn shuffled_partition(
    n: usize,
    inhibitory_fraction: f64,
    seed: u64,
) -> Result<Vec<Sign>, TopologyError> {
    let mut signs = dale_partition(n, inhibitory_fraction)?;
    let mut rng = Rng::new(seed);
    // Fisher-Yates downward: element i trades places with a uniform index in 0..=i, which is the
    // permutation that is uniform over all n! orders rather than the naive swap-with-anything.
    for i in (1..signs.len()).rev() {
        let j = below_u64(&mut rng, i as u64 + 1) as usize;
        signs.swap(i, j);
    }
    Ok(signs)
}

// ---------------------------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------------------------

/// `G(n, p)`: every ordered pair of distinct neurons gets a synapse independently with probability
/// `p`.
///
/// This is **Gilbert's** model (E. N. Gilbert, Ann. Math. Statist. 30(4):1141-1144, 1959) rather
/// than Erdős and Rényi's, who fixed the edge *count* (Publ. Math. Debrecen 6:290-297, 1959). The
/// field calls both Erdős-Rényi and so does this crate's naming, with this sentence as the
/// correction; [`erdos_renyi_gnm`] is the one they actually wrote.
///
/// Directed and self-loop-free, so the pair space is `n * (n - 1)` and the synapse count is
/// `Binomial(n(n-1), p)`: mean `p n (n - 1)`, standard deviation `sqrt(n (n-1) p (1-p))`. The test
/// `gnp_edge_count_lands_within_four_sigma` checks the built graph against exactly that, in sigma
/// rather than in a percentage, because a percentage tolerance says nothing about whether the
/// sampler is correct.
///
/// Cost is `O(n^2)` draws whatever `p` is — every pair is asked. For a sparse graph on a large `n`,
/// [`erdos_renyi_gnm`] draws `m` times instead.
///
/// # Errors
///
/// [`TopologyError::Probability`] for a `p` outside `[0, 1]`, [`TopologyError::IndexSpace`] for an
/// `n` past `u32`, or the [`Wiring`] validation errors.
pub fn erdos_renyi_gnp(
    n: usize,
    p: f64,
    wiring: &Wiring,
    seed: u64,
) -> Result<Net, TopologyError> {
    check_n(n)?;
    check_probability("p", p)?;
    wiring.validate()?;
    let mut rng = Rng::new(seed);
    let mut b = NetBuilder::new(n);
    for a in 0..n {
        let w = wiring.weight_of(n, a);
        for c in 0..n {
            if a == c {
                continue;
            }
            if rng.next_f64() < p {
                b.connect(a as u32, c as u32, w, wiring.delay)?;
            }
        }
    }
    Ok(b.build())
}

/// `G(n, m)`: exactly `m` distinct directed synapses, chosen uniformly without replacement.
///
/// Erdős and Rényi, Publ. Math. Debrecen 6:290-297, 1959. Sampled by Floyd's algorithm, which draws
/// exactly `m` times — no rejection loop, no `O(n^2)` scan — so a sparse graph on a million neurons
/// costs `m` draws rather than `10^12`.
///
/// Self-loops are excluded, so the pair space is `n * (n - 1)` and the synapse count is exactly `m`
/// with no variance at all. That exactness is the reason to prefer this model when the thing under
/// test is sensitive to synapse count: a `G(n, p)` graph's count moves by a few hundred between
/// seeds, and a joules figure moves with it.
///
/// # Errors
///
/// [`TopologyError::EdgeBudget`] if `m` exceeds `n * (n - 1)`, [`TopologyError::IndexSpace`], or the
/// [`Wiring`] validation errors.
pub fn erdos_renyi_gnm(
    n: usize,
    m: usize,
    wiring: &Wiring,
    seed: u64,
) -> Result<Net, TopologyError> {
    check_n(n)?;
    wiring.validate()?;
    let space = (n as u64) * (n as u64).saturating_sub(1);
    if m as u64 > space {
        return Err(TopologyError::EdgeBudget { requested: m, available: space });
    }
    let mut rng = Rng::new(seed);
    // Floyd's algorithm for a uniform m-subset of 0..space, in m draws and no rejection.
    let mut chosen: BTreeSet<u64> = BTreeSet::new();
    let mut codes: Vec<u64> = Vec::with_capacity(m);
    for j in (space - m as u64)..space {
        let t = below_u64(&mut rng, j + 1);
        if chosen.insert(t) {
            codes.push(t);
        } else {
            chosen.insert(j);
            codes.push(j);
        }
    }
    let mut b = NetBuilder::new(n);
    for code in codes {
        // Decode into an ordered pair with a != b: `a` picks the row, `r` picks among the n - 1
        // targets that are not `a`, and the shift skips over the diagonal.
        let a = code / (n as u64 - 1);
        let r = code % (n as u64 - 1);
        let c = if r >= a { r + 1 } else { r };
        let w = wiring.weight_of(n, a as usize);
        b.connect(a as u32, c as u32, w, wiring.delay)?;
    }
    Ok(b.build())
}

/// Watts-Strogatz small-world: a ring lattice with each edge rewired with probability `beta`.
///
/// Watts and Strogatz, Nature 393:440-442, 1998. Start with `n` nodes on a ring, each joined to its
/// `k / 2` nearest neighbours on each side. Then walk the edges lap by lap — all the nearest-
/// neighbour edges in ring order, then all the next-nearest, as the paper specifies — and with
/// probability `beta` replace each edge's far endpoint with a uniformly chosen node, forbidding
/// self-loops and duplicates.
///
/// `beta = 0` leaves the lattice: clustering `3(k-2) / (4(k-1))` exactly, path length about
/// `n / (2k)`. `beta = 1` is close to a random graph. In between is the small world, and
/// `watts_strogatz_reproduces_figure_2` sweeps it and asserts the crossover exists.
///
/// The model is undirected; each edge becomes a **reciprocal pair** of synapses here, so
/// `net.n_syn == n * k` at any `beta`. A node whose degree has already reached `n - 1` cannot be
/// rewired anywhere new and keeps its edge, which is the standard implementation's behaviour and
/// only bites at `k` comparable to `n`.
///
/// # Errors
///
/// [`TopologyError::RingDegree`] if `k` is odd or zero, [`TopologyError::TooSmall`] if `n <= k`,
/// [`TopologyError::Probability`] for `beta`, [`TopologyError::IndexSpace`], or the [`Wiring`]
/// validation errors.
pub fn watts_strogatz(
    n: usize,
    k: usize,
    beta: f64,
    wiring: &Wiring,
    seed: u64,
) -> Result<Net, TopologyError> {
    check_n(n)?;
    check_probability("beta", beta)?;
    wiring.validate()?;
    if k == 0 || !k.is_multiple_of(2) {
        return Err(TopologyError::RingDegree { k });
    }
    if n <= k {
        return Err(TopologyError::TooSmall {
            n,
            needed: k + 1,
            what: "watts_strogatz needs n > k so the ring is not already complete",
        });
    }
    let half = k / 2;
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(n * half);
    for lap in 1..=half {
        for i in 0..n {
            edges.push((i as u32, ((i + lap) % n) as u32));
        }
    }
    let mut present: BTreeSet<(u32, u32)> = edges.iter().map(|&(a, b)| norm(a, b)).collect();
    let mut deg = vec![k; n];
    let mut rng = Rng::new(seed);
    for e in 0..edges.len() {
        let (u, v) = edges[e];
        if rng.next_f64() >= beta {
            continue;
        }
        // A saturated node has nowhere left to go; the edge stays as it is.
        if deg[u as usize] >= n - 1 {
            continue;
        }
        // Terminates with probability one: at least one target is free, and the draw is uniform
        // over all n, so the expected number of attempts is n / (n - 1 - deg[u]).
        loop {
            let w = rng.below(n as u32);
            if w == u || present.contains(&norm(u, w)) {
                continue;
            }
            present.remove(&norm(u, v));
            present.insert(norm(u, w));
            deg[v as usize] -= 1;
            deg[w as usize] += 1;
            edges[e] = (u, w);
            break;
        }
    }
    reciprocal(n, &edges, wiring)
}

/// Barabási-Albert scale-free: growth plus preferential attachment.
///
/// Barabási and Albert, Science 286:509-512, 1999. Each new node attaches `m` edges to existing
/// nodes with probability proportional to their degree, sampled in constant time per edge by the
/// repeated-endpoint array of Batagelj and Brandes (Phys. Rev. E 71:036113, 2005) — an array in
/// which every node appears once per incident edge, so a uniform draw from it *is* a degree-weighted
/// draw from the nodes.
///
/// The stationary degree distribution is exactly
///
/// ```text
/// p(k) = 2m(m+1) / (k(k+1)(k+2)),    k >= m
/// ```
///
/// Dorogovtsev, Mendes and Samukhin's Eq. (6a) (Phys. Rev. Lett. 85:4633-4636, 2000), written
/// there in the in-degree `q = k − m`. Krapivsky, Redner and Leyvraz (Phys. Rev. Lett.
/// 85:4629-4632, 2000) derived the `m = 1` case, `n_k = 4/(k(k+1)(k+2))`, by the same rate-equation
/// method; releases through 0.22.0 credited them with the general form. This module did not
/// re-derive it; it measured it, and
/// `barabasi_albert_degrees_match_the_exact_stationary_distribution` checks every degree from `m`
/// to `m + 4` against that expression within binomial error.
///
/// **A fitted exponent will not come back as 3, and that is not a defect.** The asymptote is
/// `k^-3`, but at finite `k` the exact distribution is *shallower* than its asymptote — its local
/// slope is `k(1/k + 1/(k+1) + 1/(k+2))`, which is 2.61 at `k = 6` and only reaches 2.97 by
/// `k = 50`. So [`power_law_exponent`] above `k_min = 6` returns about **2.78**, and 2.78 is what
/// the model itself predicts: the test computes the estimator's expectation under `p(k)` in closed
/// form and finds 2.7844 against a measured 2.787. Raising `k_min` walks it to 3.
///
/// The seed graph is a complete graph on `m + 1` nodes, so every node has degree `m` before growth
/// starts; the paper leaves the seed unspecified and the exponent does not depend on it
/// asymptotically, but the edge count does — it is exactly `m(m+1)/2 + m(n - m - 1)` undirected
/// edges, hence twice that many synapses.
///
/// Undirected, so each edge becomes a reciprocal pair. **The hubs are the reason to be careful
/// with this one on hardware:** the largest degree grows as `sqrt(n)`, and a neuromorphic core's
/// fan-in limit does not.
///
/// # Errors
///
/// [`TopologyError::TooSmall`] if `m < 1` or `n < m + 1`, [`TopologyError::IndexSpace`], or the
/// [`Wiring`] validation errors.
pub fn barabasi_albert(
    n: usize,
    m: usize,
    wiring: &Wiring,
    seed: u64,
) -> Result<Net, TopologyError> {
    check_n(n)?;
    wiring.validate()?;
    if m < 1 {
        return Err(TopologyError::TooSmall {
            n: m,
            needed: 1,
            what: "barabasi_albert needs m >= 1 edges per arriving node",
        });
    }
    if n < m + 1 {
        return Err(TopologyError::TooSmall {
            n,
            needed: m + 1,
            what: "barabasi_albert needs n >= m + 1 for the seed clique",
        });
    }
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut repeated: Vec<u32> = Vec::new();
    for a in 0..=m {
        for c in (a + 1)..=m {
            edges.push((a as u32, c as u32));
            repeated.push(a as u32);
            repeated.push(c as u32);
        }
    }
    let mut rng = Rng::new(seed);
    let mut targets: Vec<u32> = Vec::with_capacity(m);
    for v in (m + 1)..n {
        targets.clear();
        for _ in 0..m {
            let start = below_u64(&mut rng, repeated.len() as u64) as usize;
            // Linear probe from the drawn position rather than a resample loop: it always
            // terminates (the array holds at least m + 1 distinct nodes at this point) and it keeps
            // the generator's draw count independent of the graph, which is what makes two runs of
            // the same seed produce the same graph on any platform.
            //
            // "Always" is an argument, not a check, so the failure is an error and not a silent
            // short edge list: an arriving node that got fewer than m edges would leave the degree
            // distribution wrong in a way no assertion downstream looks for.
            let Some(cand) = probe_distinct(&repeated, &targets, start) else {
                return Err(TopologyError::TooSmall {
                    n: targets.len(),
                    needed: m,
                    what: "barabasi_albert's endpoint array ran out of distinct nodes to attach to",
                });
            };
            targets.push(cand);
        }
        for &t in &targets {
            edges.push((v as u32, t));
            repeated.push(v as u32);
            repeated.push(t);
        }
    }
    reciprocal(n, &edges, wiring)
}

/// A rectangular lattice of neuron positions in metres.
///
/// Neuron `i` sits at `(x, y, z) * spacing` with `i = x + nx * (y + ny * z)` — x fastest, the order
/// a `Vec` of neurons is already in. Set `nz = 1` for a 2-D sheet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid3 {
    /// Neurons along x. Must be at least 1.
    pub nx: usize,
    /// Neurons along y. Must be at least 1.
    pub ny: usize,
    /// Neurons along z. Must be at least 1; `1` makes the grid a 2-D sheet.
    pub nz: usize,
    /// Lattice spacing in **metres**. Cortical microcircuit models use tens of micrometres; the
    /// value only ever appears divided by `lambda`, so the pair `(spacing, lambda)` is what matters
    /// and either alone is meaningless.
    pub spacing: f64,
}

impl Grid3 {
    /// A grid, unchecked; [`Grid3::validate`] is where the dimensions are enforced.
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, spacing: f64) -> Self {
        Self { nx, ny, nz, spacing }
    }

    /// The 15 x 3 x 3 column of Maass, Natschläger and Markram (Neural Computation
    /// 14(11):2531-2560, 2002): 135 neurons, the liquid-state-machine microcircuit.
    ///
    /// The paper's grid is dimensionless; `spacing` is what gives it metres.
    #[must_use]
    pub fn maass_column(spacing: f64) -> Self {
        Self { nx: 15, ny: 3, nz: 3, spacing }
    }

    /// Neurons in the grid: `nx * ny * nz`, **saturating** at `usize::MAX`.
    ///
    /// Saturating and not wrapping: a wrapped product is a small number that
    /// [`crate::net::NetBuilder`] would accept, so `Grid3 { nx: usize::MAX, ny: 2, nz: 2, .. }`
    /// would build a four-neuron network and call it a lattice. [`Grid3::validate`] rejects any
    /// grid whose product does not fit, so no generator here ever sees the saturated value.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nx.saturating_mul(self.ny).saturating_mul(self.nz)
    }

    /// Whether any dimension is zero, which makes the grid hold no neurons.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Maass's dimensionless `lambda` in grid steps, converted to metres.
    ///
    /// The paper's `lambda = 2` means "two lattice steps"; this returns `2 * spacing`. The
    /// conversion is a method rather than a comment so that the paper's number survives in the
    /// caller's source where it can be compared against the source.
    #[must_use]
    pub fn lambda_of(&self, lambda_steps: f64) -> f64 {
        lambda_steps * self.spacing
    }

    /// Position of neuron `index` in metres, or `None` past the end of the grid.
    #[must_use]
    pub fn position(&self, index: usize) -> Option<[f64; 3]> {
        if index >= self.len() {
            return None;
        }
        let x = index % self.nx;
        let y = (index / self.nx) % self.ny;
        let z = index / (self.nx * self.ny);
        Some([x as f64 * self.spacing, y as f64 * self.spacing, z as f64 * self.spacing])
    }

    /// Euclidean distance between two neurons in metres, or `None` if either index is past the end.
    #[must_use]
    pub fn distance(&self, a: usize, b: usize) -> Option<f64> {
        let p = self.position(a)?;
        let q = self.position(b)?;
        let d2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
        Some(d2.sqrt())
    }

    /// Reject a grid with a zero dimension, a product that overflows a `usize`, or a non-positive
    /// spacing.
    ///
    /// # Errors
    ///
    /// [`TopologyError::TooSmall`] for a zero dimension, [`TopologyError::IndexSpace`] if
    /// `nx * ny * nz` overflows — [`Grid3::len`] saturates there, and a saturated length is a
    /// different grid from the one that was asked for — and [`TopologyError::Length`] for a spacing
    /// that is not finite and strictly positive.
    pub fn validate(&self) -> Result<(), TopologyError> {
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err(TopologyError::TooSmall {
                n: 0,
                needed: 1,
                what: "Grid3 needs every dimension to be at least 1",
            });
        }
        let fits = self.nx.checked_mul(self.ny).and_then(|a| a.checked_mul(self.nz));
        let Some(n) = fits else {
            return Err(TopologyError::IndexSpace { n: usize::MAX });
        };
        check_n(n)?;
        if !self.spacing.is_finite() || self.spacing <= 0.0 {
            return Err(TopologyError::Length { name: "Grid3::spacing", value: self.spacing });
        }
        Ok(())
    }
}

/// The four connection-probability scale factors of the Maass cortical microcircuit rule.
///
/// `C` in `P(a -> b) = C * exp(-(D(a,b) / lambda)^2)`, chosen by the Dale types of the pair, with
/// `a` presynaptic. The values in [`MAASS_2002`] are transcribed from Maass, Natschläger and
/// Markram, Neural Computation 14(11):2531-2560, 2002; check them against the paper before
/// reproducing a figure with them, because a scale factor that is wrong by a factor of two produces
/// a network that still behaves plausibly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaassC {
    /// Excitatory presynaptic to excitatory postsynaptic. Dimensionless, `[0, 1]`.
    pub ee: f64,
    /// Excitatory presynaptic to inhibitory postsynaptic. Dimensionless, `[0, 1]`.
    pub ei: f64,
    /// Inhibitory presynaptic to excitatory postsynaptic. Dimensionless, `[0, 1]`.
    pub ie: f64,
    /// Inhibitory presynaptic to inhibitory postsynaptic. Dimensionless, `[0, 1]`.
    pub ii: f64,
}

/// The scale factors of Maass et al. 2002: 0.3 (EE), 0.2 (EI), 0.4 (IE), 0.1 (II).
///
/// Transcribed from the paper's parameter list, with the hedge in [`MaassC`]'s doc: this review
/// read them off the source rather than deriving them, and they should be checked before a
/// reproduction rests on them.
pub const MAASS_2002: MaassC = MaassC { ee: 0.3, ei: 0.2, ie: 0.4, ii: 0.1 };

impl MaassC {
    /// One scale factor for every pair type — the right choice when measuring the distance rule
    /// itself, because four factors confound the measurement with the partition.
    #[must_use]
    pub fn uniform(c: f64) -> Self {
        Self { ee: c, ei: c, ie: c, ii: c }
    }

    /// The scale factor for a presynaptic `pre` contacting a postsynaptic `post`.
    ///
    /// A [`Sign::Silent`] endpoint returns `0.0`: this rule has nothing to say about a neuron whose
    /// transmitter is unknown, and inventing a probability for it would put synapses in the graph
    /// that no partition asked for.
    #[must_use]
    pub fn c_for(&self, pre: Sign, post: Sign) -> f64 {
        match (pre, post) {
            (Sign::Excitatory, Sign::Excitatory) => self.ee,
            (Sign::Excitatory, Sign::Inhibitory) => self.ei,
            (Sign::Inhibitory, Sign::Excitatory) => self.ie,
            (Sign::Inhibitory, Sign::Inhibitory) => self.ii,
            _ => 0.0,
        }
    }

    /// Reject a scale factor that is not a probability.
    ///
    /// # Errors
    ///
    /// [`TopologyError::Probability`] naming the offending field.
    pub fn validate(&self) -> Result<(), TopologyError> {
        check_probability("MaassC::ee", self.ee)?;
        check_probability("MaassC::ei", self.ei)?;
        check_probability("MaassC::ie", self.ie)?;
        check_probability("MaassC::ii", self.ii)
    }
}

/// Distance-dependent connectivity on a lattice: `P(a -> b) = C * exp(-(D(a,b) / lambda)^2)`.
///
/// Maass, Natschläger and Markram, Neural Computation 14(11):2531-2560, 2002 — the connectivity of
/// the liquid state machine, and the closest thing the field has to a standard cortical
/// microcircuit. `lambda` is in **metres**, so Maass's dimensionless 2 is `grid.lambda_of(2.0)`.
///
/// The Gaussian falloff is what makes this different from every other generator here: connection
/// probability is a function of physical distance, so the network has a geometry, and a mapper can
/// place it on a fabric where locality costs less than a long route. That is also the property the
/// test measures — `distance_dependent_probability_falls_off_as_specified` bins the built graph by
/// exact lattice distance and compares each bin's empirical rate against the rule within three
/// binomial standard deviations.
///
/// Self-connections are excluded. At `D = 0` the rule would give `P = C`, which for a neuron onto
/// itself is an autapse; they exist in tissue and are not what this generator is for.
///
/// # This generator's E/I geometry is not the paper's, unless you give it a partition
///
/// [`Wiring`] assigns the inhibitory population as a **contiguous index block** and [`Grid3`] maps
/// index to position with x fastest. Compose the two and the inhibitory population is a **slab**,
/// not a scattered fifth: on [`Grid3::maass_column`] with [`Wiring::default`] — 135 neurons, 27
/// inhibitory — every inhibitory neuron lands on the single face `z = 2`, and the mean incoming
/// weight of the built network then runs 1.60, −1.58, −0.38 mV by z-slab and 1.52, 1.04, −0.48,
/// −1.41, −1.26 mV by x-slab (measured at seed 1 with [`MAASS_2002`] and `lambda = 2` grid steps).
/// That 3.18 mV spread is an artefact of the index convention and of nothing in the model; a random
/// partition leaves 0.88 mV, averaged over five seeds.
///
/// Maass, Natschläger and Markram choose their 20% inhibitory neurons **at random**, which is the
/// only assignment under which a distance-dependent rule means what their paper says. So: a figure
/// reproduced by calling this function with a `Wiring` is not their figure. Use
/// [`distance_dependent_signed`] with [`shuffled_partition`] for that. This entry point keeps the
/// contiguous partition because it is the one every other generator here uses and because changing
/// it would silently move every network already built with it.
///
/// # Errors
///
/// [`TopologyError::Length`] for a `lambda` that is not finite and positive, plus the [`Grid3`],
/// [`MaassC`] and [`Wiring`] validation errors.
pub fn distance_dependent(
    grid: &Grid3,
    c: &MaassC,
    lambda: f64,
    wiring: &Wiring,
    seed: u64,
) -> Result<Net, TopologyError> {
    // Every argument is validated here as well as in the delegate, so that the *order* in which a
    // caller learns about two bad arguments is the one this function has always had — and so that
    // a grid too large to index is refused before `wiring.signs(n)` allocates for it.
    grid.validate()?;
    c.validate()?;
    wiring.validate()?;
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(TopologyError::Length { name: "lambda", value: lambda });
    }
    let n = grid.len();
    check_n(n)?;
    distance_dependent_signed(grid, c, lambda, wiring, &wiring.signs(n), seed)
}

/// [`distance_dependent`] with the excitatory/inhibitory partition supplied per neuron.
///
/// The rule, the lattice and the draw order are identical; only the source of each neuron's Dale
/// type changes, from `Wiring`'s contiguous index block to `signs[i]`. Hand it
/// [`shuffled_partition`] to get the random 20% that Maass, Natschläger and Markram specify, or a
/// partition of your own when the geometry is the experiment.
///
/// A [`Sign::Silent`] entry is honoured rather than ignored: [`MaassC::c_for`] gives it probability
/// zero and [`Wiring::weight_for`] gives it weight zero, so a neuron declared `Silent` is generated
/// with no outgoing synapses at all. That is the one way to leave a neuron out of the graph without
/// changing `n`.
///
/// `distance_dependent(grid, c, lambda, wiring, seed)` is exactly
/// `distance_dependent_signed(grid, c, lambda, wiring, &wiring.signs(grid.len()), seed)` — the same
/// synapses, bit for bit, which `the_contiguous_partition_is_the_shuffled_path_with_a_block_input`
/// asserts.
///
/// # Errors
///
/// [`TopologyError::SignsLength`] if `signs` is not `grid.len()` long,
/// [`TopologyError::Length`] for a `lambda` that is not finite and positive, plus the [`Grid3`],
/// [`MaassC`] and [`Wiring`] validation errors.
pub fn distance_dependent_signed(
    grid: &Grid3,
    c: &MaassC,
    lambda: f64,
    wiring: &Wiring,
    signs: &[Sign],
    seed: u64,
) -> Result<Net, TopologyError> {
    grid.validate()?;
    c.validate()?;
    wiring.validate()?;
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(TopologyError::Length { name: "lambda", value: lambda });
    }
    let n = grid.len();
    check_n(n)?;
    if signs.len() != n {
        return Err(TopologyError::SignsLength { declared: signs.len(), neurons: n });
    }
    let mut rng = Rng::new(seed);
    let mut b = NetBuilder::new(n);
    for a in 0..n {
        let sa = signs[a];
        let w = wiring.weight_for(sa);
        for d in 0..n {
            if a == d {
                continue;
            }
            let dist = grid.distance(a, d).unwrap_or(f64::INFINITY);
            let ratio = dist / lambda;
            let p = c.c_for(sa, signs[d]) * (-(ratio * ratio)).exp();
            if rng.next_f64() < p {
                b.connect(a as u32, d as u32, w, wiring.delay)?;
            }
        }
    }
    Ok(b.build())
}

/// The neuron index range of each layer, given the layer sizes.
///
/// Layers are laid out contiguously in index order, so layer `i` occupies
/// `offset .. offset + layers[i]`. This is what a caller needs to build the external current vector
/// that drives the input layer, and it is a separate function so that the ranges can be computed
/// before the network is.
#[must_use]
pub fn layer_ranges(layers: &[usize]) -> Vec<Range<usize>> {
    let mut out = Vec::with_capacity(layers.len());
    let mut at = 0usize;
    for &size in layers {
        out.push(at..at + size);
        at += size;
    }
    out
}

/// A layered feedforward network: layer `i` projects to layer `i + 1` and nowhere else.
///
/// `p == 1.0` is all-to-all, which makes the synapse count exactly
/// `sum(layers[i] * layers[i+1])`; a smaller `p` drops each potential synapse independently, which
/// is the "sparse feedforward" used to keep fan-in under a hardware limit.
///
/// There is no recurrence at all — `feedforward_has_no_recurrent_synapse` asserts that every
/// synapse crosses exactly one layer boundary forwards — so this is the one family here with no
/// memory in its wiring. Everything it remembers has to live in a neuron's state.
///
/// Pass [`Wiring::excitatory_only`] unless you want the trailing block of neurons, which in a
/// layered network is the output layer, to be inhibitory.
///
/// # Errors
///
/// [`TopologyError::TooFewLayers`] for fewer than two layers, [`TopologyError::EmptyLayer`] for a
/// layer of size zero, [`TopologyError::Probability`] for `p`, [`TopologyError::IndexSpace`], or
/// the [`Wiring`] validation errors.
pub fn feedforward(
    layers: &[usize],
    p: f64,
    wiring: &Wiring,
    seed: u64,
) -> Result<Net, TopologyError> {
    if layers.len() < 2 {
        return Err(TopologyError::TooFewLayers { layers: layers.len() });
    }
    for (i, &size) in layers.iter().enumerate() {
        if size == 0 {
            return Err(TopologyError::EmptyLayer { layer: i });
        }
    }
    check_probability("p", p)?;
    wiring.validate()?;
    let n: usize = layers.iter().sum();
    check_n(n)?;
    let ranges = layer_ranges(layers);
    let mut rng = Rng::new(seed);
    let mut b = NetBuilder::new(n);
    for l in 0..ranges.len() - 1 {
        for a in ranges[l].clone() {
            let w = wiring.weight_of(n, a);
            for c in ranges[l + 1].clone() {
                if p >= 1.0 || rng.next_f64() < p {
                    b.connect(a as u32, c as u32, w, wiring.delay)?;
                }
            }
        }
    }
    Ok(b.build())
}

/// Winner-take-all: `n` units, every one inhibiting every other, optionally exciting itself.
///
/// The motif behind selection in cortex (Hahnloser, Sarpeshkar, Mahowald, Douglas and Seung,
/// Nature 405:947-951, 2000) and the one Maass analysed as a computational primitive (Neural
/// Computation 12(11):2519-2535, 2000). Each unit receives its own external current; the unit with
/// the largest current fires, and its spikes hold everyone else below threshold.
///
/// **The regime is not automatic and is the whole point.** [`wta_inhibition_floor`] gives the
/// mean-field weight at which the runner-up is held sub-threshold, and the module's tests show that
/// floor is *necessary but not sufficient*: at exactly the floor the runner-up's sawtooth still
/// crosses threshold between the winner's spikes. Measured for the default [`crate::neuron::Lif`]
/// with currents `[6, 4, 3.5, 3]` nA, the steady-state winner count is 2 at `1.0x` the floor and 1
/// from `1.5x` upward. Both directions are asserted.
///
/// `w_self` is self-excitation, which is a synapse from a unit to itself. Note the interaction with
/// the refractory period: [`crate::neuron::Neuron::bump`] is **ignored** while a neuron is
/// refractory, so a self-synapse whose delay is shorter than `t_ref / dt` ticks does nothing at all.
/// That is a real property of delta synapses and not a bug to route around.
///
/// # Errors
///
/// [`TopologyError::TooSmall`] for `n < 2` — one unit is not a competition — or
/// [`TopologyError::Weight`] if `w_inh` is not finite and `<= 0`, or `w_self` not finite and `>= 0`.
pub fn winner_take_all(
    n: usize,
    w_inh: f64,
    w_self: f64,
    delay: u32,
) -> Result<Net, TopologyError> {
    check_n(n)?;
    if n < 2 {
        return Err(TopologyError::TooSmall {
            n,
            needed: 2,
            what: "winner_take_all needs at least two competing units",
        });
    }
    if !w_inh.is_finite() || w_inh > 0.0 {
        return Err(TopologyError::Weight { name: "w_inh", value: w_inh });
    }
    if !w_self.is_finite() || w_self < 0.0 {
        return Err(TopologyError::Weight { name: "w_self", value: w_self });
    }
    let mut b = NetBuilder::new(n);
    for a in 0..n {
        if w_self > 0.0 {
            b.connect(a as u32, a as u32, w_self, delay)?;
        }
        for c in 0..n {
            if a != c {
                b.connect(a as u32, c as u32, w_inh, delay)?;
            }
        }
    }
    Ok(b.build())
}

/// The mean-field lateral inhibition needed to hold a runner-up below threshold, volts per spike.
///
/// A train of delta bumps of amplitude `w` arriving at rate `r` into a membrane of time constant
/// `tau_m` displaces the potential by `w * r * tau_m` on average: each bump decays as
/// `exp(-t / tau_m)`, whose time integral is `tau_m`, and there are `r` of them per second. So the
/// loser's effective steady state is `v_inf(i_other) - |w| * r_win * tau_m`, and it stays
/// sub-threshold when
///
/// ```text
/// |w| > (v_inf(i_other) - v_th) / (r_win * tau_m)
/// ```
///
/// which is what this returns, with `r_win` from [`crate::neuron::Lif::rate`] at `i_win`.
///
/// **This is a floor on the average, and the average is not what fires a neuron.** The loser's
/// potential sawtooths around that mean and its peaks sit above it, so the true threshold for a
/// single winner is higher — measured in this module's tests at between `1.0x` and `1.5x` the
/// floor for the default [`crate::neuron::Lif`]. Treat the return value as a lower bound that
/// rules weights out, not one that rules them in.
///
/// `None` when the intended winner is itself sub-threshold at `i_win`: nobody fires, so no
/// inhibition produces a winner and there is no number to return. `Some(0.0)` when the runner-up is
/// already sub-threshold on its own and needs no help.
#[must_use]
pub fn wta_inhibition_floor(lif: &Lif, i_win: f64, i_other: f64) -> Option<f64> {
    let r_win = lif.rate(i_win)?;
    let excess = lif.v_inf(i_other) - lif.v_th;
    if excess <= 0.0 {
        return Some(0.0);
    }
    Some(excess / (r_win * lif.tau_m))
}

// ---------------------------------------------------------------------------------------------
// Measures
// ---------------------------------------------------------------------------------------------

/// Each neuron's neighbours, reading the network as **undirected**: `a` and `b` are neighbours if a
/// synapse runs either way between them.
///
/// Sorted and deduplicated, self-loops dropped. This is the reading the small-world measures need,
/// because clustering and degree distributions are defined on undirected graphs and a reciprocal
/// pair is one edge, not two.
#[must_use]
pub fn undirected_neighbours(net: &Net) -> Vec<Vec<u32>> {
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); net.n];
    for pre in 0..net.n {
        for (post, _, _) in net.out_of(pre) {
            if post as usize == pre {
                continue;
            }
            adj[pre].push(post);
            adj[post as usize].push(pre as u32);
        }
    }
    for a in &mut adj {
        a.sort_unstable();
        a.dedup();
    }
    adj
}

/// Undirected degree of every neuron, indexed by neuron.
///
/// A reciprocal pair counts **once**, so a Watts-Strogatz graph built here has degrees of `k` and
/// not `2k` even though its [`crate::net::Net::n_syn`] is `n * k`. The sum over all neurons is
/// twice the undirected edge count.
#[must_use]
pub fn undirected_degrees(net: &Net) -> Vec<usize> {
    undirected_neighbours(net).iter().map(Vec::len).collect()
}

/// The Watts-Strogatz clustering coefficient: the mean over neurons of the fraction of a neuron's
/// neighbour pairs that are themselves connected.
///
/// `C_v = 2 * (edges among the neighbours of v) / (d_v * (d_v - 1))`, averaged over neurons. Read
/// undirected, per [`undirected_neighbours`].
///
/// **Neurons of degree below 2 are skipped, not counted as zero.** `C_v` is undefined for them —
/// there is no pair of neighbours to be connected or not — and the other common convention, scoring
/// them 0, drags the average down in proportion to how sparse the graph is, which turns a
/// connectivity statistic into a density statistic. `None` when no neuron has two neighbours.
///
/// Exact values worth checking against: a complete graph gives 1, a ring lattice with `k`
/// neighbours gives `3(k-2) / (4(k-1))` (Barrat and Weigt, Eur. Phys. J. B 13:547-560, 2000), a
/// feedforward network gives 0, and `G(n, p)` gives `p`.
#[must_use]
pub fn clustering_coefficient(net: &Net) -> Option<f64> {
    let adj = undirected_neighbours(net);
    let mut sum = 0.0;
    let mut counted = 0usize;
    for v in 0..net.n {
        let d = adj[v].len();
        if d < 2 {
            continue;
        }
        let mut links = 0usize;
        for &u in &adj[v] {
            links += intersection_size(&adj[u as usize], &adj[v]);
        }
        // Each edge among the neighbours was seen from both of its ends.
        let links = links as f64 / 2.0;
        sum += 2.0 * links / (d as f64 * (d as f64 - 1.0));
        counted += 1;
    }
    if counted == 0 { None } else { Some(sum / counted as f64) }
}

/// Hop counts from `source` to every neuron, following synapses in their own direction.
///
/// `Some(0)` for the source itself, `None` for a neuron with no directed path from the source, and
/// the whole result is `None` if `source` is past the end of the network. Breadth-first, so the
/// counts are shortest paths and the cost is one pass over the synapse list.
#[must_use]
pub fn hops_from(net: &Net, source: usize) -> Option<Vec<Option<u32>>> {
    if source >= net.n {
        return None;
    }
    let mut dist: Vec<Option<u32>> = vec![None; net.n];
    let mut queue: Vec<u32> = Vec::with_capacity(net.n);
    dist[source] = Some(0);
    queue.push(source as u32);
    let mut head = 0usize;
    while head < queue.len() {
        let u = queue[head] as usize;
        head += 1;
        let du = dist[u].unwrap_or(0);
        for (v, _, _) in net.out_of(u) {
            if dist[v as usize].is_none() {
                dist[v as usize] = Some(du + 1);
                queue.push(v);
            }
        }
    }
    Some(dist)
}

/// A shortest-path census over every ordered pair of distinct neurons.
///
/// Reachability is reported rather than assumed, because a rewired or grown graph is not guaranteed
/// connected and a mean over "the pairs that happened to be reachable" is a different quantity from
/// the characteristic path length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathStats {
    /// Ordered pairs `(i, j)` with `i != j` considered: `n * (n - 1)`.
    pub pairs: u64,
    /// How many of those pairs have a directed path from `i` to `j`.
    pub reachable: u64,
    /// Mean hop count **over the reachable pairs only**, or `None` when none are reachable. Read it
    /// together with `reachable`: a small mean over 3% of the pairs is not a short path length.
    pub mean_hops: Option<f64>,
    /// The largest finite hop count found — the directed diameter of the reachable part — or `None`
    /// when no pair is reachable.
    pub diameter: Option<u32>,
}

impl PathStats {
    /// Whether every ordered pair is reachable, i.e. the network is strongly connected.
    ///
    /// **A network with fewer than two neurons has no ordered pair and is reported `false`, not
    /// vacuously `true`.** Graph theory calls the one-vertex graph strongly connected; this is the
    /// same refusal [`clustering_coefficient`] makes when no neuron has two neighbours and
    /// [`PathStats::mean_hops`] makes when nothing is reachable, and for the same reason — the
    /// quantity a caller is about to compute from it is a mean over pairs, and there are none.
    /// `a_network_with_no_ordered_pair_is_not_reported_strongly_connected` pins it.
    #[must_use]
    pub fn is_strongly_connected(&self) -> bool {
        self.pairs > 0 && self.reachable == self.pairs
    }
}

/// Breadth-first search from every neuron, at `O(n * (n + n_syn))`.
///
/// This is the expensive measure in the module and the honest cost of a path length: there is no
/// closed form for `L` on a rewired lattice, which is why Watts and Strogatz computed it.
#[must_use]
pub fn path_stats(net: &Net) -> PathStats {
    let n = net.n as u64;
    let pairs = n * n.saturating_sub(1);
    let mut reachable = 0u64;
    let mut sum = 0u64;
    let mut diameter = 0u32;
    for s in 0..net.n {
        let Some(dist) = hops_from(net, s) else { continue };
        for j in 0..net.n {
            if j == s {
                continue;
            }
            if let Some(d) = dist[j] {
                reachable += 1;
                sum += u64::from(d);
                diameter = diameter.max(d);
            }
        }
    }
    PathStats {
        pairs,
        reachable,
        mean_hops: if reachable > 0 { Some(sum as f64 / reachable as f64) } else { None },
        diameter: if reachable > 0 { Some(diameter) } else { None },
    }
}

/// The characteristic path length `L`: the mean hop count over every ordered pair.
///
/// `None` unless **every** pair is reachable. Watts and Strogatz define `L` on a connected graph,
/// and a disconnected graph's `L` is infinite rather than large — the same refusal [`crate::neuron`]
/// makes for a sub-threshold neuron's firing rate. Use [`path_stats`] when you want the mean over
/// the reachable part with the reachable count beside it.
#[must_use]
pub fn characteristic_path_length(net: &Net) -> Option<f64> {
    let s = path_stats(net);
    if s.is_strongly_connected() { s.mean_hops } else { None }
}

/// Weakly connected component label of every neuron, reading the network as undirected.
///
/// Labels are 0-based in order of first appearance, so the component count is
/// `labels.iter().max().map_or(0, |m| m + 1)`.
#[must_use]
pub fn weak_components(net: &Net) -> Vec<usize> {
    let adj = undirected_neighbours(net);
    let mut label: Vec<Option<usize>> = vec![None; net.n];
    let mut next = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    for s in 0..net.n {
        if label[s].is_some() {
            continue;
        }
        label[s] = Some(next);
        stack.push(s);
        while let Some(u) = stack.pop() {
            for &v in &adj[u] {
                if label[v as usize].is_none() {
                    label[v as usize] = Some(next);
                    stack.push(v as usize);
                }
            }
        }
        next += 1;
    }
    label.into_iter().map(|l| l.unwrap_or(0)).collect()
}

/// Maximum-likelihood power-law exponent of a degree sample above `k_min`.
///
/// The discrete approximation of Clauset, Shalizi and Newman (*Power-law distributions in empirical
/// data*, SIAM Review 51(4):661-703, 2009, eq. 3.7):
///
/// ```text
/// alpha = 1 + N / sum_i ln( k_i / (k_min - 0.5) )
/// ```
///
/// The `- 0.5` is the continuity correction that makes the continuous estimator usable on integer
/// degrees; it is accurate to about 1% for `k_min >= 6` and degrades below that. `None` when
/// `k_min < 1`, when fewer than two samples reach `k_min`, or when the sum is non-positive.
///
/// **Fitting an exponent is not the same as establishing a power law.** This returns the exponent
/// the estimator gives whatever the data is; Clauset et al. spend most of their paper on the
/// goodness-of-fit test that this does not implement. Use it to check that a generator produces
/// what its theory says, which is what `barabasi_albert_degrees_follow_a_power_law_near_three`
/// does, and not to claim a measured network is scale-free.
#[must_use]
pub fn power_law_exponent(degrees: &[usize], k_min: usize) -> Option<f64> {
    if k_min < 1 {
        return None;
    }
    let shift = k_min as f64 - 0.5;
    let mut sum = 0.0;
    let mut count = 0u64;
    for &k in degrees {
        if k >= k_min {
            sum += (k as f64 / shift).ln();
            count += 1;
        }
    }
    if count < 2 || sum <= 0.0 {
        return None;
    }
    Some(1.0 + count as f64 / sum)
}

/// The sum of every neuron's **incoming** synaptic weights, volts per presynaptic spike.
///
/// The measured counterpart of [`Wiring::expected_drive`]: in a balanced network these are
/// distributed about zero, and a systematic offset is the network telling you it will saturate or
/// fall silent before a single spike has been simulated.
#[must_use]
pub fn in_weight_sums(net: &Net) -> Vec<f64> {
    let mut s = vec![0.0; net.n];
    for pre in 0..net.n {
        for (post, w, _) in net.out_of(pre) {
            s[post as usize] += w;
        }
    }
    s
}

// ---------------------------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------------------------

/// The first node at or after `start` in the repeated-endpoint array that is not already a target,
/// scanning cyclically; `None` if every entry is already taken.
///
/// Factored out of [`barabasi_albert`] so that the `None` branch — unreachable from the generator,
/// which always probes an array holding at least `m + 1` distinct nodes — can be reached by a test
/// instead of only argued about.
fn probe_distinct(repeated: &[u32], targets: &[u32], start: usize) -> Option<u32> {
    if repeated.is_empty() {
        return None;
    }
    (0..repeated.len())
        .map(|step| repeated[(start + step) % repeated.len()])
        .find(|cand| !targets.contains(cand))
}

/// Normalise an undirected edge so that the duplicate test sees `(a, b)` and `(b, a)` as one edge.
fn norm(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Emit an undirected edge list as reciprocal directed synapses, signed by presynaptic neuron.
fn reciprocal(n: usize, edges: &[(u32, u32)], wiring: &Wiring) -> Result<Net, TopologyError> {
    let mut b = NetBuilder::new(n);
    for &(u, v) in edges {
        b.connect(u, v, wiring.weight_of(n, u as usize), wiring.delay)?;
        b.connect(v, u, wiring.weight_of(n, v as usize), wiring.delay)?;
    }
    Ok(b.build())
}

/// The largest `zone` such that accepting a 64-bit draw `v <= zone` and returning `v % n` is
/// exactly uniform on `[0, n)`.
///
/// The acceptance region must hold a **whole number of residue classes**, so its size `zone + 1`
/// has to be the largest multiple of `n` that fits in `2^64`. That size is `2^64 - (2^64 mod n)`,
/// and `2^64 mod n` is computed as `(u64::MAX mod n + 1) mod n` because `2^64` does not fit in a
/// `u64`. The trailing `mod n` is what makes `n` a power of two accept every draw rather than
/// rejecting one class.
///
/// **The obvious-looking `u64::MAX - (u64::MAX % n) - (n - 1)` is wrong**, and was what this module
/// shipped: its acceptance region is `(q - 1) n + 2` wide, so residues 0 and 1 are over-represented
/// by one slot in `2^64 / n` for *every* `n`. The practical bias is under `2.3e-10` at any `n` this
/// crate reaches, but the failure at large `n` is not cosmetic: for `n > 2^63` the old region
/// collapses to `{0, 1}`, so the draw returns only 0 or 1 and its rejection loop expects `2^63`
/// iterations — a hang. `n = 2^63 + 7` is reachable through the type system, since
/// [`check_n`] admits `n` up to `u32::MAX` and `erdos_renyi_gnm`'s pair space `n(n-1)` crosses
/// `2^63` at `n` near `3.04e9`.
///
/// `unbiased_zone_holds_whole_residue_classes` asserts `(zone + 1) % n == 0` exactly, in `u128`,
/// over the whole range including that one.
///
/// `n` must be at least 1; `n == 0` divides by zero, and the only caller has already ruled it out.
fn unbiased_zone(n: u64) -> u64 {
    u64::MAX - (u64::MAX % n).wrapping_add(1) % n
}

/// A uniform integer in `[0, n)` for an `n` that may exceed `u32`, without modulo bias.
///
/// The pair space of `G(n, m)` is `n(n-1)`, which passes `u32::MAX` at about 65,000 neurons — a
/// perfectly ordinary network size — so the 32-bit draw is not enough on its own. Delegates to
/// [`crate::rng::Rng::below`] below that point so that small graphs consume exactly the draws they
/// consumed before this path existed.
///
/// # Panics
///
/// Never for `n >= 1`. `n == 0` panics inside [`crate::rng::Rng::below`], which has no value to
/// return; no caller here passes it.
fn below_u64(rng: &mut Rng, n: u64) -> u64 {
    if n <= u64::from(u32::MAX) {
        return u64::from(rng.below(n as u32));
    }
    let zone = unbiased_zone(n);
    loop {
        let v = (u64::from(rng.next_u32()) << 32) | u64::from(rng.next_u32());
        if v <= zone {
            return v % n;
        }
    }
}

/// Reject anything that is not a finite number in `[0, 1]`, naming the argument.
fn check_probability(name: &'static str, p: f64) -> Result<(), TopologyError> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(TopologyError::Probability { name, value: p });
    }
    Ok(())
}

/// Reject a neuron count a `u32` synapse index cannot name.
fn check_n(n: usize) -> Result<(), TopologyError> {
    if n as u64 > u64::from(u32::MAX) {
        return Err(TopologyError::IndexSpace { n });
    }
    Ok(())
}

/// Number of elements two sorted, deduplicated slices share.
fn intersection_size(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut c) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            core::cmp::Ordering::Less => i += 1,
            core::cmp::Ordering::Greater => j += 1,
            core::cmp::Ordering::Equal => {
                c += 1;
                i += 1;
                j += 1;
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::{
        BALANCED_G, CORTICAL_INHIBITORY_FRACTION, DaleViolation, Grid3, MAASS_2002, MaassC,
        PathStats, Sign, TopologyError, Wiring, barabasi_albert, characteristic_path_length,
        clustering_coefficient, dale_check, dale_partition, dale_signs, distance_dependent,
        distance_dependent_signed, erdos_renyi_gnm, erdos_renyi_gnp, feedforward, hops_from,
        in_weight_sums, layer_ranges, norm, path_stats, power_law_exponent, shuffled_partition,
        undirected_degrees, watts_strogatz, weak_components, winner_take_all, wta_inhibition_floor,
    };
    use crate::net::{NetBuilder, NetError};
    use std::collections::BTreeSet;
    use crate::neuron::Lif;
    use crate::rng::Rng;
    use crate::sim::{Mode, Sim};

    fn exc(w: f64) -> Wiring {
        Wiring::excitatory_only(w, 1)
    }

    /// (a) The edge count of `G(n, p)` is `Binomial(n(n-1), p)`. Asserted in **sigma**, not in a
    /// percentage: a 5% tolerance on 3,980 expected edges is 3.3 sigma at this `n` and 33 sigma at
    /// a hundred times the size, so a percentage tolerance tests a different thing at every scale.
    #[test]
    fn gnp_edge_count_lands_within_four_sigma() {
        let (n, p) = (200usize, 0.1);
        let trials = (n * (n - 1)) as f64;
        let mean = trials * p;
        let sigma = (trials * p * (1.0 - p)).sqrt();
        for seed in 0..8u64 {
            let net = erdos_renyi_gnp(n, p, &exc(1e-3), seed).unwrap();
            let z = (net.n_syn as f64 - mean) / sigma;
            assert!(z.abs() < 4.0, "seed {seed}: {} synapses is {z:.2} sigma out", net.n_syn);
        }
        // And the whole distribution has to sit around the mean, not just each draw inside a band.
        let total: usize =
            (0..8u64).map(|s| erdos_renyi_gnp(n, p, &exc(1e-3), s).unwrap().n_syn).sum();
        let z = (total as f64 / 8.0 - mean) / (sigma / 8.0f64.sqrt());
        assert!(z.abs() < 3.0, "mean over 8 seeds is {z:.2} sigma out");
    }

    /// `G(n, p)` clustering is `p` in expectation — a second, independent check that the sampler is
    /// uniform over pairs rather than correlated within a row.
    #[test]
    fn gnp_clustering_is_about_p() {
        let net = erdos_renyi_gnp(150, 0.2, &exc(1e-3), 5).unwrap();
        // Read undirected, a pair is connected if either direction exists: 1 - (1 - p)^2 = 0.36.
        let want = 1.0 - (1.0 - 0.2f64).powi(2);
        let c = clustering_coefficient(&net).unwrap();
        assert!((c - want).abs() < 0.02, "clustering {c:.4} vs expected {want:.4}");
    }

    #[test]
    fn gnp_refuses_a_probability_that_is_not_one() {
        for bad in [-0.1, 1.5, f64::NAN, f64::INFINITY] {
            let e = erdos_renyi_gnp(10, bad, &exc(1e-3), 0).unwrap_err();
            assert!(matches!(e, TopologyError::Probability { name: "p", .. }), "{e}");
        }
    }

    #[test]
    fn gnm_returns_exactly_m_distinct_synapses_and_no_self_loops() {
        let n = 60usize;
        for &m in &[0usize, 1, 100, 1_000, 60 * 59] {
            let net = erdos_renyi_gnm(n, m, &exc(1e-3), 3).unwrap();
            assert_eq!(net.n_syn, m, "asked for {m} edges");
            let mut seen = std::collections::BTreeSet::new();
            for pre in 0..n {
                for (post, _, _) in net.out_of(pre) {
                    assert_ne!(post as usize, pre, "self-loop at {pre}");
                    assert!(seen.insert((pre, post)), "duplicate synapse {pre} -> {post}");
                }
            }
        }
    }

    #[test]
    fn gnm_refuses_more_edges_than_the_pair_space_holds() {
        let e = erdos_renyi_gnm(10, 91, &exc(1e-3), 0).unwrap_err();
        assert_eq!(e, TopologyError::EdgeBudget { requested: 91, available: 90 });
    }

    /// (b.1) The ring lattice's clustering coefficient has a closed form: `3(k-2) / (4(k-1))`.
    /// Checked to floating-point noise, for several `k`, against a measure that knows nothing about
    /// rings.
    #[test]
    fn the_ring_lattice_clustering_matches_its_closed_form() {
        for &k in &[4usize, 6, 8, 10, 20] {
            let net = watts_strogatz(200, k, 0.0, &exc(1e-3), 1).unwrap();
            let want = 3.0 * (k as f64 - 2.0) / (4.0 * (k as f64 - 1.0));
            let got = clustering_coefficient(&net).unwrap();
            assert!((got - want).abs() < 1e-12, "k = {k}: C = {got} vs closed form {want}");
        }
    }

    /// (b.2) On the ring lattice, the shortest path from 0 to `j` is exactly
    /// `ceil(min(j, n - j) / (k / 2))`. That is an exact integer prediction for every one of 200
    /// destinations, which pins the breadth-first search itself before it is used to measure `L`.
    #[test]
    fn the_ring_lattice_hop_counts_match_the_exact_prediction() {
        let (n, k) = (201usize, 6usize);
        let net = watts_strogatz(n, k, 0.0, &exc(1e-3), 1).unwrap();
        let dist = hops_from(&net, 0).unwrap();
        for j in 1..n {
            let ring = j.min(n - j);
            let want = ring.div_ceil(k / 2) as u32;
            assert_eq!(dist[j], Some(want), "hop 0 -> {j}");
        }
        assert_eq!(dist[0], Some(0));
    }

    /// (b.3) **Watts-Strogatz figure 2.** Sweep the rewiring probability and require the regime the
    /// paper reports to exist: a `beta` at which the path length has collapsed while the clustering
    /// is still essentially the lattice's.
    ///
    /// Measured here at `n = 400`, `k = 10` (the paper uses `n = 1000`, `k = 10`), and the numbers
    /// this run produces are printed by the assertion messages on failure. `L(0) = 20.451`,
    /// `C(0) = 0.66667`; at `beta = 0.01` the run gives `L/L(0) = 0.420` against `C/C(0) = 0.986`
    /// — the path length has lost 58% and the clustering 1.4%, which is the whole phenomenon. By
    /// `beta = 1` they are 0.139 and 0.035.
    #[test]
    fn watts_strogatz_reproduces_figure_2() {
        let (n, k) = (400usize, 10usize);
        let betas = [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0];
        let mut l = Vec::new();
        let mut c = Vec::new();
        for &beta in &betas {
            let net = watts_strogatz(n, k, beta, &exc(1e-3), 20_250_917).unwrap();
            assert_eq!(net.n_syn, n * k, "beta {beta}: rewiring must not change the edge count");
            let stats = path_stats(&net);
            assert!(stats.is_strongly_connected(), "beta {beta}: graph fell apart");
            l.push(stats.mean_hops.unwrap());
            c.push(clustering_coefficient(&net).unwrap());
        }
        // The lattice end: both measures at their closed forms. `L(0)` has one too — the ring
        // lattice's shortest path from any node to the node `j` places away is exactly
        // `ceil(min(j, n - j) / (k / 2))`, which test (b.2) checks hop by hop — so the mean over
        // destinations is an exact prediction and not a "> 15".
        assert!((c[0] - 3.0 * 8.0 / (4.0 * 9.0)).abs() < 1e-12, "C(0) = {}", c[0]);
        let want_l0: f64 = (1..n).map(|j| j.min(n - j).div_ceil(k / 2) as f64).sum::<f64>()
            / (n as f64 - 1.0);
        assert!((want_l0 - 20.451_127_820).abs() < 1e-9, "the closed form itself moved: {want_l0}");
        assert!((l[0] - want_l0).abs() < 1e-12, "L(0) = {} against the closed form {want_l0}", l[0]);
        // The random end, against the random-graph values rather than against a fraction of the
        // lattice's. Clustering is the edge density `k / (n - 1)` = 0.02506 (measured 0.02354, and
        // 0.0212 to 0.0251 over five seeds), and the characteristic path length of a random graph
        // of mean degree `k` is `(ln n - gamma) / ln k + 1/2` = 2.8514 with Euler's gamma
        // (measured 2.8375, within 0.5% for every seed tried).
        let want_c1 = k as f64 / (n as f64 - 1.0);
        assert!(
            c[8] > 0.7 * want_c1 && c[8] < 1.3 * want_c1,
            "C(1) = {} is not the random-graph density {want_c1}",
            c[8]
        );
        let gamma = 0.577_215_664_901_532_9f64;
        let want_l1 = ((n as f64).ln() - gamma) / (k as f64).ln() + 0.5;
        assert!(
            (l[8] - want_l1).abs() < 0.05 * want_l1,
            "L(1) = {} is not the random-graph estimate {want_l1}",
            l[8]
        );
        // And the collapse itself, which is what the figure is about.
        assert!(c[8] < 0.1 * c[0], "C(1) = {} vs C(0) = {}", c[8], c[0]);
        assert!(l[8] < 0.25 * l[0], "L(1) = {} vs L(0) = {}", l[8], l[0]);
        // The finding: a regime where L has collapsed and C has not.
        let crossover = (0..betas.len()).any(|i| l[i] < 0.6 * l[0] && c[i] > 0.8 * c[0]);
        assert!(
            crossover,
            "no small-world regime found\nbeta {betas:?}\nL/L0 {:?}\nC/C0 {:?}",
            l.iter().map(|x| x / l[0]).collect::<Vec<_>>(),
            c.iter().map(|x| x / c[0]).collect::<Vec<_>>()
        );
        // And path length has to fall faster than clustering, which is the whole phenomenon.
        let i = betas.iter().position(|&b| b == 0.01).unwrap();
        assert!(
            (l[0] - l[i]) / l[0] > (c[0] - c[i]) / c[0],
            "at beta = 0.01, L fell {:.3} and C fell {:.3}",
            (l[0] - l[i]) / l[0],
            (c[0] - c[i]) / c[0]
        );
    }

    /// (finding 3) The rewiring's own promise: **no self-loop and no duplicate**, at every `beta`.
    ///
    /// `assert_eq!(net.n_syn, n * k)` cannot see either, because `reciprocal` emits exactly two
    /// synapses per edge slot however corrupt the slot is. Two mutations survive every other test
    /// in this module and both are caught here: making [`norm`] the identity, which lets a
    /// rewiring re-create an edge that already exists in the other direction, and writing
    /// `edges[e] = (w, v)`, which moves the endpoint Watts and Strogatz hold fixed and
    /// desynchronises `edges` from the `present` set that polices duplicates.
    ///
    /// A duplicate is not merely untidy: [`crate::net::Net`] stores each synapse once, so two of
    /// them double-bill every joule the ledger charges for that connection.
    #[test]
    fn watts_strogatz_rewiring_keeps_every_synapse_distinct() {
        let (n, k) = (200usize, 6usize);
        for &beta in &[0.0f64, 0.1, 0.5, 1.0] {
            for seed in [1u64, 7] {
                let net = watts_strogatz(n, k, beta, &exc(1e-3), seed).unwrap();
                assert_eq!(net.n_syn, n * k, "beta {beta} seed {seed}");
                let mut seen = BTreeSet::new();
                for pre in 0..n {
                    for (post, _, _) in net.out_of(pre) {
                        assert_ne!(post as usize, pre, "beta {beta} seed {seed}: self-loop at {pre}");
                        assert!(
                            seen.insert((pre as u32, post)),
                            "beta {beta} seed {seed}: duplicate synapse {pre} -> {post}"
                        );
                    }
                }
                // Undirected too: every synapse must be half of a reciprocal pair, so the
                // undirected degree sum is exactly n * k and the edge set is half the size.
                let undirected: BTreeSet<(u32, u32)> =
                    seen.iter().map(|&(a, b)| norm(a, b)).collect();
                assert_eq!(undirected.len(), n * k / 2, "beta {beta} seed {seed}");
                assert_eq!(
                    undirected_degrees(&net).iter().sum::<usize>(),
                    n * k,
                    "beta {beta} seed {seed}"
                );
            }
        }
    }

    #[test]
    fn watts_strogatz_refuses_an_odd_ring_degree_and_a_ring_that_is_already_complete() {
        assert_eq!(
            watts_strogatz(10, 3, 0.1, &exc(1e-3), 0).unwrap_err(),
            TopologyError::RingDegree { k: 3 }
        );
        assert_eq!(
            watts_strogatz(10, 0, 0.1, &exc(1e-3), 0).unwrap_err(),
            TopologyError::RingDegree { k: 0 }
        );
        assert!(matches!(
            watts_strogatz(10, 10, 0.1, &exc(1e-3), 0).unwrap_err(),
            TopologyError::TooSmall { n: 10, needed: 11, .. }
        ));
    }

    /// The exact edge count of the growth model, which the seed clique makes predictable:
    /// `m(m+1)/2 + m(n - m - 1)` undirected edges, twice that many synapses.
    #[test]
    fn barabasi_albert_has_exactly_the_predicted_edge_count() {
        for &(n, m) in &[(50usize, 1usize), (200, 2), (500, 3), (300, 5)] {
            let net = barabasi_albert(n, m, &exc(1e-3), 4).unwrap();
            let want = m * (m + 1) / 2 + m * (n - m - 1);
            assert_eq!(net.n_syn, 2 * want, "n = {n}, m = {m}");
            // Undirected degrees sum to twice the edge count.
            let sum: usize = undirected_degrees(&net).iter().sum();
            assert_eq!(sum, 2 * want);
        }
    }

    /// The exponent [`power_law_exponent`] returns in the large-sample limit on integers produced
    /// by a given continuous draw, summed in closed form rather than sampled.
    ///
    /// `lower_edge(k)` is the value of the underlying Pareto variate `T` (supported on `[1, inf)`,
    /// with `P(T >= t) = t^-(alpha - 1)`) at which the integer result becomes `k`, so
    /// `P(K = k) = P(T >= lower_edge(k)) - P(T >= lower_edge(k + 1))`. Flooring `k_min * T` and
    /// rounding `(k_min - 0.5) * T` are two different discretisations of the same continuous law
    /// and this evaluates the estimator's expectation under either.
    fn exact_discretised_mle(k_min: usize, alpha: f64, lower_edge: impl Fn(f64) -> f64) -> f64 {
        let shift = k_min as f64 - 0.5;
        let tail = |t: f64| t.powf(-(alpha - 1.0));
        let (mut mass, mut weighted) = (0.0f64, 0.0f64);
        for k in k_min..=10_000_000usize {
            let kf = k as f64;
            let p = tail(lower_edge(kf)) - tail(lower_edge(kf + 1.0));
            mass += p;
            weighted += p * (kf / shift).ln();
        }
        1.0 + 1.0 / (weighted / mass)
    }

    /// The fitter is checked before it is trusted: draw from a continuous power law by inverse CDF,
    /// discretise, and require the estimator to return **the number the estimator must return on
    /// that sample**, which is not always the exponent the sample was drawn at.
    ///
    /// `x = k_min * (1 - u)^(-1 / (alpha - 1))` has `P(X > x) ~ x^(1 - alpha)`, which is exponent
    /// `alpha`. **Flooring it does not preserve the exponent.** At `alpha = 2.5`, `k_min = 10` the
    /// estimator's expectation on `floor(X)` is 2.45237, not 2.5: the discretisation and the
    /// continuity correction disagree, and the 0.0476 offset is *ten times* the sampling error
    /// `(alpha - 1) / sqrt(N) = 0.00474` at 100,000 samples. This test used to assert
    /// `|got - 2.5| < 0.05` and passed with 6% of its tolerance to spare, on a bias it did not
    /// name. Both the biased value and the sampling error are now predicted in closed form and
    /// checked against the draw.
    ///
    /// The sampler Clauset, Shalizi and Newman give for the *discrete* law —
    /// `round((k_min - 0.5) * (1 - u)^(-1 / (alpha - 1)))` — is the one the estimator is derived
    /// for, and it recovers 2.4982 against an expectation of 2.49734, inside half a sigma of both.
    #[test]
    fn the_power_law_fitter_recovers_a_known_exponent() {
        let alpha = 2.5f64;
        let k_min = 10usize;
        let draw = |corrected: bool| {
            let mut rng = Rng::new(31);
            let mut sample = Vec::with_capacity(100_000);
            for _ in 0..100_000 {
                let t = (1.0 - rng.next_f64()).powf(-1.0 / (alpha - 1.0));
                let x = if corrected {
                    ((k_min as f64 - 0.5) * t + 0.5).floor()
                } else {
                    (k_min as f64 * t).floor()
                };
                sample.push(x as usize);
            }
            sample
        };
        // The estimator's own standard error, which is what every tolerance below is written in.
        let sigma = (alpha - 1.0) / 100_000.0f64.sqrt();
        assert!((sigma - 0.004_743).abs() < 1e-6, "sigma {sigma}");

        // (i) Floored continuous Pareto: biased, by a predictable amount.
        let biased = exact_discretised_mle(k_min, alpha, |k| k / k_min as f64);
        assert!((biased - 2.452_373).abs() < 1e-5, "the closed form moved: {biased}");
        assert!(
            (alpha - biased) / sigma > 8.0,
            "the discretisation bias is {:.1} sigma, so it is not noise",
            (alpha - biased) / sigma
        );
        let sample = draw(false);
        let got = power_law_exponent(&sample, k_min).unwrap();
        assert!(
            (got - biased).abs() < 4.0 * sigma,
            "fitted {got:.5} against the floored-Pareto expectation {biased:.5}, \
             {:.1} sigma out",
            (got - biased).abs() / sigma
        );

        // (ii) The discrete sampler the continuity correction is derived for: unbiased, so the
        // estimator recovers the exponent it was drawn at.
        let corrected_prediction =
            exact_discretised_mle(k_min, alpha, |k| (k - 0.5) / (k_min as f64 - 0.5));
        assert!((corrected_prediction - 2.497_344).abs() < 1e-5, "{corrected_prediction}");
        let got_corrected = power_law_exponent(&draw(true), k_min).unwrap();
        assert!(
            (got_corrected - corrected_prediction).abs() < 4.0 * sigma,
            "fitted {got_corrected:.5} against {corrected_prediction:.5}"
        );
        assert!(
            (got_corrected - alpha).abs() < 4.0 * sigma,
            "fitted {got_corrected:.5} for a sample drawn at {alpha}"
        );
        // The two discretisations really do differ by more than the noise they are measured in.
        assert!((got - got_corrected).abs() > 8.0 * sigma);
        // And the estimator's arithmetic at a degenerate input, where it has a closed form: every
        // sample equal to k_min gives alpha = 1 + 1 / ln(k_min / (k_min - 0.5)).
        let flat = vec![4usize; 1000];
        let want = 1.0 + 1.0 / (4.0f64 / 3.5).ln();
        let got = power_law_exponent(&flat, 4).unwrap();
        assert!((got - want).abs() < 1e-12, "degenerate fit {got} vs closed form {want}");
        assert!(power_law_exponent(&[5, 6, 7], 0).is_none(), "k_min = 0 has no shift");
        assert!(power_law_exponent(&[1, 1], 5).is_none(), "nothing reaches k_min");
    }

    /// The maximum-likelihood exponent this estimator returns under the exact Barabási-Albert
    /// degree distribution, computed by summing the closed form rather than by sampling.
    ///
    /// `alpha = 1 + 1 / E[ln(k / (k_min - 0.5))]` with the expectation taken over
    /// `p(k) ~ 1 / (k(k+1)(k+2))` restricted to `k >= k_min`. The normalisation `2m(m+1)` cancels,
    /// which is why the prediction does not depend on `m` — and neither does Barabási and Albert's
    /// exponent.
    fn exact_ba_mle(k_min: usize, k_max: usize) -> f64 {
        let shift = k_min as f64 - 0.5;
        let (mut mass, mut weighted) = (0.0f64, 0.0f64);
        for k in k_min..=k_max {
            let kf = k as f64;
            let p = 1.0 / (kf * (kf + 1.0) * (kf + 2.0));
            mass += p;
            weighted += p * (kf / shift).ln();
        }
        1.0 + 1.0 / (weighted / mass)
    }

    /// (c) The degree distribution of a grown graph against the exact stationary form
    /// `p(k) = 2m(m+1) / (k(k+1)(k+2))` — every degree from `m` to `m + 4`, in binomial sigma, for
    /// two values of `m` and three seeds.
    ///
    /// This is a stronger statement than "the exponent is about 3", and it is checkable: for
    /// `m = 2` the closed form says exactly half the nodes have degree 2, a fifth have degree 3 and
    /// a tenth have degree 4.
    #[test]
    fn barabasi_albert_degrees_match_the_exact_stationary_distribution() {
        for &m in &[2usize, 3] {
            for &seed in &[77u64, 5, 1234] {
                let net = barabasi_albert(20_000, m, &exc(1e-3), seed).unwrap();
                let deg = undirected_degrees(&net);
                let n = deg.len() as f64;
                for k in m..=(m + 4) {
                    let got = deg.iter().filter(|&&x| x == k).count() as f64 / n;
                    let kf = k as f64;
                    let want =
                        2.0 * m as f64 * (m as f64 + 1.0) / (kf * (kf + 1.0) * (kf + 2.0));
                    let sigma = (want * (1.0 - want) / n).sqrt();
                    assert!(
                        (got - want).abs() < 3.5 * sigma,
                        "m = {m}, seed {seed}, k = {k}: p = {got:.5} vs closed form {want:.5} ({:.1} sigma)",
                        (got - want).abs() / sigma
                    );
                }
            }
        }
    }

    /// (c, continued) The fitted exponent against the value the estimator must return on this
    /// model, computed in closed form by `exact_ba_mle`.
    ///
    /// **The fit is not 3 and must not be.** The asymptote is `k^-3`, but the exact distribution is
    /// shallower than its asymptote at finite `k`, so an honest fit above `k_min = 6` returns
    /// 2.7844 — and that is the number this test predicts analytically and then measures, agreeing
    /// to better than 0.03. Asserting "near 3" instead would have been a looser test that passed
    /// for the wrong reason.
    #[test]
    fn the_fitted_exponent_matches_the_estimator_prediction_and_walks_to_three() {
        // The prediction converges: 2.784 at k_min = 6, 2.987 at 100, 2.99989 at 10,000. That the
        // limit is 3 is Barabási and Albert's result; that the finite-k value is below it is why
        // this test does not assert 3.
        let ladder: Vec<f64> = [6usize, 10, 50, 100, 1_000, 10_000]
            .iter()
            .map(|&k| exact_ba_mle(k, 10_000_000))
            .collect();
        for w in ladder.windows(2) {
            assert!(w[1] > w[0], "the prediction is not monotone in k_min: {ladder:?}");
        }
        assert!((ladder[0] - 2.78438).abs() < 1e-4, "k_min = 6 predicts {}", ladder[0]);
        assert!((ladder[5] - 3.0).abs() < 1e-3, "k_min = 10,000 predicts {}", ladder[5]);

        for &m in &[2usize, 3] {
            for &seed in &[77u64, 5, 1234, 0, 4, 6, 9, 11] {
                let net = barabasi_albert(20_000, m, &exc(1e-3), seed).unwrap();
                let deg = undirected_degrees(&net);
                let got = power_law_exponent(&deg, 6).unwrap();
                // In sigma, like the sibling test, and for the same reason. The maximum-likelihood
                // exponent's own standard error is `(alpha - 1) / sqrt(N)` with `N` the number of
                // samples at or above `k_min`: 0.0333 at m = 2 (N is about 2,870 of 20,000 nodes)
                // and 0.0234 at m = 3. The `< 0.03` this test used to assert is tighter than one
                // sigma at m = 2 — seed 6 returns 2.74804, which is a perfectly ordinary 1.1 sigma
                // and misses that band, as would about one seed in six.
                //
                // The nominal sigma is conservative here: measured over twelve seeds at m = 2 the
                // seed-to-seed spread is 0.0165, about half of it, because a single grown graph's
                // degree sum is fixed by construction rather than sampled. Every seed below lands
                // inside 1.2 nominal sigma, so the 3.0 band is not fitted to them.
                let n_at = deg.iter().filter(|&&d| d >= 6).count();
                let sigma = (got - 1.0) / (n_at as f64).sqrt();
                assert!(
                    (got - ladder[0]).abs() < 3.0 * sigma,
                    "m = {m}, seed {seed}: fitted {got:.4} vs the model's own MLE {:.4}, \
                     which is {:.2} sigma at sigma = {sigma:.4} over {n_at} samples",
                    ladder[0],
                    (got - ladder[0]).abs() / sigma
                );
                // Hubs: the largest degree of a BA graph grows as sqrt(n), so at n = 20,000 it is
                // hundreds against a mean of 2m. This is the property that breaks a fan-in limit.
                let max = *deg.iter().max().unwrap();
                assert!(max > 20 * m, "largest degree {max} for m = {m} is not a hub");
            }
        }
    }

    /// (f) The distance rule, measured from the graph it built. Bins are **exact**: the squared
    /// lattice distance is an integer, so pairs are grouped with no float binning at all, and each
    /// bin's empirical rate is compared to `C exp(-(d / lambda)^2)` within three binomial sigma.
    #[test]
    fn distance_dependent_probability_falls_off_as_specified() {
        let grid = Grid3::new(20, 20, 1, 50e-6);
        let c = 0.5;
        let lambda = grid.lambda_of(3.0);
        let net = distance_dependent(&grid, &MaassC::uniform(c), lambda, &exc(1e-3), 8).unwrap();
        let n = grid.len();
        // key = dx^2 + dy^2 in lattice units; value = (pairs, connections).
        let mut bins: std::collections::BTreeMap<usize, (u64, u64)> = std::collections::BTreeMap::new();
        let mut connected: std::collections::BTreeSet<(usize, usize)> = std::collections::BTreeSet::new();
        for pre in 0..n {
            for (post, _, _) in net.out_of(pre) {
                connected.insert((pre, post as usize));
            }
        }
        for a in 0..n {
            let (ax, ay) = (a % 20, a / 20);
            for b in 0..n {
                if a == b {
                    continue;
                }
                let (bx, by) = (b % 20, b / 20);
                let dx = ax as i64 - bx as i64;
                let dy = ay as i64 - by as i64;
                let key = (dx * dx + dy * dy) as usize;
                let e = bins.entry(key).or_insert((0, 0));
                e.0 += 1;
                if connected.contains(&(a, b)) {
                    e.1 += 1;
                }
            }
        }
        let mut checked = 0usize;
        for (&key, &(pairs, hits)) in &bins {
            if pairs < 400 {
                continue;
            }
            let d = (key as f64).sqrt() * grid.spacing;
            let ratio = d / lambda;
            let want = c * (-(ratio * ratio)).exp();
            let got = hits as f64 / pairs as f64;
            let sigma = (want * (1.0 - want) / pairs as f64).sqrt();
            assert!(
                (got - want).abs() < 3.0 * sigma.max(1e-4),
                "d^2 = {key}: measured p = {got:.5} vs rule {want:.5} ({:.1} sigma over {pairs} pairs)",
                (got - want).abs() / sigma
            );
            checked += 1;
        }
        assert!(checked > 12, "only {checked} distance bins had enough pairs to test");
        // The rule must actually fall off, not merely agree on average.
        let near = bins[&1];
        let far = bins[&64];
        assert!(
            (near.1 as f64 / near.0 as f64) > 10.0 * (far.1 as f64 / far.0 as f64),
            "connection probability did not fall with distance"
        );
    }

    #[test]
    fn distance_dependent_refuses_a_degenerate_grid_or_lambda() {
        let good = Grid3::new(4, 4, 1, 50e-6);
        assert!(matches!(
            distance_dependent(&good, &MAASS_2002, 0.0, &Wiring::default(), 0).unwrap_err(),
            TopologyError::Length { name: "lambda", .. }
        ));
        assert!(matches!(
            distance_dependent(&good, &MAASS_2002, f64::NAN, &Wiring::default(), 0).unwrap_err(),
            TopologyError::Length { name: "lambda", .. }
        ));
        let flat = Grid3::new(4, 0, 1, 50e-6);
        assert!(matches!(
            distance_dependent(&flat, &MAASS_2002, 1e-4, &Wiring::default(), 0).unwrap_err(),
            TopologyError::TooSmall { .. }
        ));
        let bad_c = MaassC { ee: 1.5, ..MAASS_2002 };
        assert!(matches!(
            distance_dependent(&good, &bad_c, 1e-4, &Wiring::default(), 0).unwrap_err(),
            TopologyError::Probability { name: "MaassC::ee", .. }
        ));
    }

    /// The Maass column is 135 neurons and its geometry is the paper's, which is worth pinning
    /// because the index order is a convention this module chose.
    #[test]
    fn the_maass_column_has_the_papers_shape() {
        let g = Grid3::maass_column(50e-6);
        assert_eq!(g.len(), 135);
        assert_eq!(g.position(0), Some([0.0, 0.0, 0.0]));
        assert_eq!(g.position(1), Some([50e-6, 0.0, 0.0]));
        assert_eq!(g.position(15), Some([0.0, 50e-6, 0.0]));
        assert_eq!(g.position(45), Some([0.0, 0.0, 50e-6]));
        assert!(g.position(135).is_none());
        assert!((g.distance(0, 1).unwrap() - 50e-6).abs() < 1e-18);
        assert!((g.lambda_of(2.0) - 100e-6).abs() < 1e-18);
        let w = Wiring::default();
        let net = distance_dependent(&g, &MAASS_2002, g.lambda_of(2.0), &w, 1).unwrap();
        // Not `n_syn > 0`, which the four scale factors cannot move: the synapse count of a column
        // built by this rule is a Poisson-binomial sum with a closed-form mean and variance, and
        // this one has to land on it. (Measured 619 against a predicted 623.7 +/- 21.4.)
        let signs = w.signs(135);
        let (mut mean, mut var) = (0.0f64, 0.0f64);
        for a in 0..135 {
            for d in 0..135 {
                if a == d {
                    continue;
                }
                let ratio = g.distance(a, d).unwrap() / g.lambda_of(2.0);
                let p = MAASS_2002.c_for(signs[a], signs[d]) * (-(ratio * ratio)).exp();
                mean += p;
                var += p * (1.0 - p);
            }
        }
        assert!(
            (net.n_syn as f64 - mean).abs() < 4.0 * var.sqrt(),
            "{} synapses against a predicted {mean:.1} +/- {:.1}",
            net.n_syn,
            var.sqrt()
        );
        // Dale by construction, with the 80/20 split the wiring declares. The *geometry* of that
        // split is not the paper's; see `a_contiguous_partition_makes_the_inhibitory_population_a_slab`.
        dale_check(&net, &signs).unwrap();
    }

    #[test]
    fn feedforward_has_no_recurrent_synapse_and_the_predicted_count() {
        let layers = [8usize, 16, 4];
        let net = feedforward(&layers, 1.0, &exc(2e-3), 0).unwrap();
        assert_eq!(net.n, 28);
        assert_eq!(net.n_syn, 8 * 16 + 16 * 4);
        let ranges = layer_ranges(&layers);
        let layer_of = |i: usize| ranges.iter().position(|r| r.contains(&i)).unwrap();
        for pre in 0..net.n {
            for (post, _, _) in net.out_of(pre) {
                assert_eq!(
                    layer_of(post as usize),
                    layer_of(pre) + 1,
                    "synapse {pre} -> {post} does not cross exactly one layer boundary"
                );
            }
        }
        // No cycles at all, so clustering is zero and the input layer is unreachable from anywhere.
        assert_eq!(clustering_coefficient(&net), Some(0.0));
        assert!(characteristic_path_length(&net).is_none(), "a feedforward net is not connected");
        let stats = path_stats(&net);
        assert_eq!(stats.pairs, 28 * 27);
        assert_eq!(stats.reachable, (8 * 16 + 16 * 4 + 8 * 4) as u64);
        assert_eq!(stats.diameter, Some(2));
    }

    #[test]
    fn feedforward_refuses_a_missing_or_empty_layer() {
        assert_eq!(
            feedforward(&[4], 1.0, &exc(1e-3), 0).unwrap_err(),
            TopologyError::TooFewLayers { layers: 1 }
        );
        assert_eq!(
            feedforward(&[4, 0, 2], 1.0, &exc(1e-3), 0).unwrap_err(),
            TopologyError::EmptyLayer { layer: 1 }
        );
    }

    /// A sparse feedforward layer's synapse count is binomial too, so the same sigma discipline
    /// applies as for `G(n, p)`.
    #[test]
    fn sparse_feedforward_drops_synapses_at_the_stated_rate() {
        let layers = [100usize, 100];
        let p = 0.3;
        let net = feedforward(&layers, p, &exc(1e-3), 12).unwrap();
        let trials = 100.0 * 100.0;
        let mean = trials * p;
        let sigma = (trials * p * (1.0 - p)).sqrt();
        let z = (net.n_syn as f64 - mean) / sigma;
        assert!(z.abs() < 4.0, "{} synapses is {z:.2} sigma from {mean}", net.n_syn);
    }

    /// (d) The checker has to catch a network that violates Dale's law, because such a network
    /// behaves perfectly well and nothing else will notice.
    #[test]
    fn dale_check_catches_a_mixed_sign_neuron_and_passes_a_conforming_net() {
        // Conforming: neuron 0 excitatory, neuron 1 inhibitory.
        let mut b = NetBuilder::new(3);
        b.connect(0, 1, 1e-3, 0).unwrap();
        b.connect(0, 2, 2e-3, 0).unwrap();
        b.connect(1, 0, -3e-3, 0).unwrap();
        b.connect(1, 2, -1e-3, 0).unwrap();
        let ok = b.build();
        assert_eq!(
            dale_signs(&ok).unwrap(),
            vec![Sign::Excitatory, Sign::Inhibitory, Sign::Silent]
        );
        dale_check(&ok, &[Sign::Excitatory, Sign::Inhibitory, Sign::Excitatory]).unwrap();

        // Violating: neuron 0 now excites 1 and inhibits 2. Nothing else in the crate objects.
        let mut b = NetBuilder::new(3);
        b.connect(0, 1, 1e-3, 0).unwrap();
        b.connect(0, 2, -2e-3, 0).unwrap();
        let bad = b.build();
        assert_eq!(
            dale_signs(&bad).unwrap_err(),
            DaleViolation::Mixed { neuron: 0, positive: 1, negative: 1 }
        );
        assert!(dale_check(&bad, &[Sign::Excitatory; 3]).is_err());
    }

    #[test]
    fn dale_check_catches_a_declaration_the_weights_contradict() {
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, -1e-3, 0).unwrap();
        let net = b.build();
        assert_eq!(
            dale_check(&net, &[Sign::Excitatory, Sign::Excitatory]).unwrap_err(),
            DaleViolation::Contradicts {
                neuron: 0,
                declared: Sign::Excitatory,
                found: Sign::Inhibitory
            }
        );
        assert_eq!(
            dale_check(&net, &[Sign::Excitatory]).unwrap_err(),
            DaleViolation::LengthMismatch { declared: 1, neurons: 2 }
        );
    }

    /// A zero-weight synapse has no sign, and a neuron with only those is silent rather than
    /// excitatory-by-default. The distinction matters because `Silent` passes any declaration.
    #[test]
    fn a_neuron_with_only_zero_weights_is_silent_not_excitatory() {
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, 0.0, 0).unwrap();
        let net = b.build();
        assert_eq!(dale_signs(&net).unwrap(), vec![Sign::Silent, Sign::Silent]);
        dale_check(&net, &[Sign::Inhibitory, Sign::Excitatory]).unwrap();
        assert!((Sign::Silent.as_f64()).abs() < 1e-18);
        assert!((Sign::Excitatory.as_f64() - 1.0).abs() < 1e-18);
        assert!((Sign::Inhibitory.as_f64() + 1.0).abs() < 1e-18);
    }

    #[test]
    fn the_partition_is_eighty_twenty_and_contiguous() {
        let p = dale_partition(100, CORTICAL_INHIBITORY_FRACTION).unwrap();
        assert_eq!(p.iter().filter(|s| **s == Sign::Inhibitory).count(), 20);
        assert_eq!(p[79], Sign::Excitatory);
        assert_eq!(p[80], Sign::Inhibitory);
        // Rounding, not truncation: 20% of 10 is 2.
        assert_eq!(Wiring::default().n_inhibitory(10), 2);
        assert_eq!(Wiring::default().n_inhibitory(3), 1);
        assert_eq!(Wiring::default().n_inhibitory(0), 0);
        assert!(matches!(
            dale_partition(10, 1.5).unwrap_err(),
            TopologyError::Probability { name: "inhibitory_fraction", .. }
        ));
    }

    /// Every generator, checked against the partition it was given rather than against itself.
    ///
    /// **Seven generators, not six.** The module doc used to say "every generator here is
    /// Dale-compliant by construction" and this list quietly held six of them:
    /// [`winner_take_all`] takes no [`Wiring`], and with `w_self > 0` it builds a unit that excites
    /// itself while inhibiting everyone else, which is a mixed-sign neuron and the one thing
    /// Dale's law forbids. Both of its forms are here, with the partition each one actually has.
    #[test]
    fn every_generator_obeys_dale_by_construction() {
        let w = Wiring::default();
        let grid = Grid3::new(5, 4, 3, 50e-6);
        let scattered = shuffled_partition(60, CORTICAL_INHIBITORY_FRACTION, 1).unwrap();
        let nets: Vec<(&str, crate::net::Net, Vec<Sign>)> = vec![
            ("gnp", erdos_renyi_gnp(60, 0.1, &w, 1).unwrap(), w.signs(60)),
            ("gnm", erdos_renyi_gnm(60, 300, &w, 1).unwrap(), w.signs(60)),
            ("ws", watts_strogatz(60, 6, 0.1, &w, 1).unwrap(), w.signs(60)),
            ("ba", barabasi_albert(60, 2, &w, 1).unwrap(), w.signs(60)),
            (
                "maass",
                distance_dependent(&grid, &MAASS_2002, grid.lambda_of(2.0), &w, 1).unwrap(),
                w.signs(60),
            ),
            (
                "maass_signed",
                distance_dependent_signed(
                    &grid,
                    &MAASS_2002,
                    grid.lambda_of(2.0),
                    &w,
                    &scattered,
                    1,
                )
                .unwrap(),
                scattered,
            ),
            ("ff", feedforward(&[20, 20, 20], 1.0, &w, 1).unwrap(), w.signs(60)),
            ("wta", winner_take_all(60, -1e-3, 0.0, 1).unwrap(), vec![Sign::Inhibitory; 60]),
        ];
        // Seven generators, and the distance rule twice — once per partition it accepts.
        assert_eq!(nets.len(), 8, "the generator list changed without this test changing");
        for (name, net, declared) in nets {
            assert!(net.n_syn > 0, "{name} produced an empty network");
            dale_check(&net, &declared)
                .unwrap_or_else(|e| panic!("{name} violates Dale's law: {e}"));
        }
        // The exception, stated rather than discovered: self-excitation makes a mixed-sign unit,
        // and no declaration can rescue it.
        let self_exciting = winner_take_all(5, -1e-3, 1e-3, 1).unwrap();
        assert_eq!(
            dale_signs(&self_exciting).unwrap_err(),
            DaleViolation::Mixed { neuron: 0, positive: 1, negative: 4 }
        );
        for declared in [Sign::Excitatory, Sign::Inhibitory, Sign::Silent] {
            assert!(dale_check(&self_exciting, &[declared; 5]).is_err(), "{declared:?}");
        }
    }

    /// (e) Same seed, same graph — the property every reproducible experiment downstream rests on.
    /// Compared field by field through `Net`'s `PartialEq`, not by a summary statistic.
    #[test]
    fn every_generator_is_deterministic_by_seed() {
        let w = Wiring::default();
        let g = Grid3::new(4, 4, 2, 50e-6);
        assert_eq!(erdos_renyi_gnp(40, 0.2, &w, 9).unwrap(), erdos_renyi_gnp(40, 0.2, &w, 9).unwrap());
        assert_eq!(erdos_renyi_gnm(40, 200, &w, 9).unwrap(), erdos_renyi_gnm(40, 200, &w, 9).unwrap());
        assert_eq!(
            watts_strogatz(40, 6, 0.3, &w, 9).unwrap(),
            watts_strogatz(40, 6, 0.3, &w, 9).unwrap()
        );
        assert_eq!(barabasi_albert(80, 3, &w, 9).unwrap(), barabasi_albert(80, 3, &w, 9).unwrap());
        assert_eq!(
            distance_dependent(&g, &MAASS_2002, g.lambda_of(2.0), &w, 9).unwrap(),
            distance_dependent(&g, &MAASS_2002, g.lambda_of(2.0), &w, 9).unwrap()
        );
        assert_eq!(
            feedforward(&[10, 10, 5], 0.5, &w, 9).unwrap(),
            feedforward(&[10, 10, 5], 0.5, &w, 9).unwrap()
        );
        assert_eq!(winner_take_all(8, -1e-3, 0.0, 1).unwrap(), winner_take_all(8, -1e-3, 0.0, 1).unwrap());
        // And a different seed has to give a different graph, or the seed is not being used.
        assert_ne!(erdos_renyi_gnp(40, 0.2, &w, 9).unwrap(), erdos_renyi_gnp(40, 0.2, &w, 10).unwrap());
        assert_ne!(barabasi_albert(80, 3, &w, 9).unwrap(), barabasi_albert(80, 3, &w, 10).unwrap());
    }

    /// The balance condition in closed form, then measured off the graph. `expected_drive` is
    /// exactly zero for the balanced wiring, and the built network's mean incoming weight sits
    /// within a few standard errors of it.
    ///
    /// **The grand mean is not enough and this test used to stop there.** Summing
    /// `in_weight_sums` over all neurons gives the total weight in the graph, which is the same
    /// number whichever endpoint of a synapse it is attributed to: attributing every weight to its
    /// *presynaptic* neuron instead — a one-word mutation of [`in_weight_sums`] — moves that mean
    /// by 4.2e-17 V and nothing else here noticed. Balance is a **per-neuron** property, so the
    /// per-neuron spread is checked too, and the two readings are 7.5x apart: incoming sums have
    /// closed-form sd 1.340e-2 V, outgoing sums 1.007e-1 V, because an outgoing row is a single
    /// sign times a binomial count and is therefore bimodal.
    #[test]
    fn a_balanced_wiring_has_zero_expected_drive_and_the_graph_agrees() {
        let w = Wiring::balanced(1e-3, 1);
        let (n, p) = (500usize, 0.1);
        assert!(w.expected_drive(n, p).abs() < 1e-18, "drive {}", w.expected_drive(n, p));
        assert!((w.w_inh / w.w_exc + BALANCED_G).abs() < 1e-18);
        let n_inh = w.n_inhibitory(n);
        // Sum of w^2 over the whole population, the quantity both spreads are built from.
        let s2 = (n - n_inh) as f64 * w.w_exc * w.w_exc + n_inh as f64 * w.w_inh * w.w_inh;
        // Incoming: n - 1 independent Bernoulli(p) draws, one per possible presynaptic partner.
        let var_in = p * (1.0 - p) * s2 * (n as f64 - 1.0) / n as f64;
        // Outgoing: one Binomial(n - 1, p) count times this neuron's own weight. The spread is
        // dominated by which sign the row carries, not by the count.
        let e_w2 = s2 / n as f64;
        let var_out =
            p * (1.0 - p) * (n as f64 - 1.0) * e_w2 + (n as f64 - 1.0).powi(2) * p * p * e_w2;
        // The sd of an sd estimated from n samples is sd / sqrt(2n).
        let sd_err = var_in.sqrt() / (2.0 * n as f64).sqrt();
        let spread = |v: &[f64]| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt()
        };
        for seed in [2u64, 3, 4, 5] {
            let net = erdos_renyi_gnp(n, p, &w, seed).unwrap();
            let sums = in_weight_sums(&net);
            let mean = sums.iter().sum::<f64>() / n as f64;
            let sem = (var_in / n as f64).sqrt();
            assert!(
                mean.abs() < 4.0 * sem,
                "seed {seed}: mean incoming weight {mean:.3e} vs 4 sem {:.3e}",
                4.0 * sem
            );
            let mut outs = vec![0.0f64; n];
            for pre in 0..n {
                for (_, weight, _) in net.out_of(pre) {
                    outs[pre] += weight;
                }
            }
            let (sd_in, sd_out) = (spread(&sums), spread(&outs));
            assert!(
                (sd_in - var_in.sqrt()).abs() < 4.0 * sd_err,
                "seed {seed}: incoming sd {sd_in:.4e} vs closed form {:.4e}, 4 sd_err {:.4e}",
                var_in.sqrt(),
                4.0 * sd_err
            );
            // Measured deviation from the closed form is 0.9% at seed 2; the band is 5%.
            assert!(
                (sd_out - var_out.sqrt()).abs() < 0.05 * var_out.sqrt(),
                "seed {seed}: outgoing sd {sd_out:.4e} vs closed form {:.4e}",
                var_out.sqrt()
            );
            // And the two really are different measurements, which is the point.
            assert!(sd_out > 5.0 * sd_in, "seed {seed}: sd_out {sd_out:.4e} sd_in {sd_in:.4e}");
        }
        // A fixture with no statistics in it at all. A 5-chain's incoming sums are [0, w, w, w, w]
        // and its outgoing sums are [w, w, w, w, 0]: the head receives nothing, the tail sends
        // nothing, and no reading of the graph can confuse the two lists.
        let mut b = NetBuilder::new(5);
        for i in 0..4u32 {
            b.connect(i, i + 1, 2e-3, 0).unwrap();
        }
        let chain = b.build();
        assert_eq!(in_weight_sums(&chain), vec![0.0, 2e-3, 2e-3, 2e-3, 2e-3]);
        // An unbalanced wiring must fail the same measurement, or the measurement proves nothing.
        let hot = Wiring { w_inh: -1e-3, ..w };
        assert!(hot.expected_drive(n, p) > 0.0);
        let hot_net = erdos_renyi_gnp(n, p, &hot, 2).unwrap();
        let hot_sums = in_weight_sums(&hot_net);
        let hot_mean = hot_sums.iter().sum::<f64>() / n as f64;
        // Against its own closed form, not merely against a threshold: the hot wiring's expected
        // drive is 3.98e-2 V and the measured mean has to land on it within a few sem.
        let hot_s2 = (n - n_inh) as f64 * hot.w_exc * hot.w_exc + n_inh as f64 * hot.w_inh * hot.w_inh;
        let hot_sem = (p * (1.0 - p) * hot_s2 * (n as f64 - 1.0) / n as f64 / n as f64).sqrt();
        let want = hot.expected_drive(n, p);
        assert!(
            (hot_mean - want).abs() < 4.0 * hot_sem,
            "unbalanced drive: measured {hot_mean:.6e} vs expected {want:.6e}, 4 sem {:.3e}",
            4.0 * hot_sem
        );
        assert!(hot_mean > 4.0 * (var_in / n as f64).sqrt());
    }

    /// (finding 7) `expected_drive` counts `n - 1` presynaptic partners, because every generator
    /// here excludes self-loops.
    ///
    /// Measured at `p = 1.0`, where `G(n, 1)` is the complete graph minus its diagonal and the
    /// measurement has **no sampling error at all**: neuron `i`'s incoming sum is exactly `S - w_i`
    /// and the mean over neurons is exactly `S (n - 1) / n`. The old formula returned `S`, which is
    /// 11.1% high at `n = 10` and 0.200% high at `n = 500` — small, systematic, and invisible in
    /// the balanced test above because `S = 0` makes both formulas exactly zero.
    #[test]
    fn the_expected_drive_counts_n_minus_one_presynaptic_partners() {
        let hot = Wiring { w_inh: -1e-3, ..Wiring::balanced(1e-3, 1) };
        for &n in &[10usize, 100, 500] {
            let net = erdos_renyi_gnp(n, 1.0, &hot, 0).unwrap();
            assert_eq!(net.n_syn, n * (n - 1), "p = 1 is the complete graph minus the diagonal");
            let sums = in_weight_sums(&net);
            let measured = sums.iter().sum::<f64>() / n as f64;
            let want = hot.expected_drive(n, 1.0);
            // Twelve significant digits: the measurement is a sum of n - 1 float additions in the
            // builder's order and the closed form multiplies once, so they agree to rounding and
            // not bit for bit.
            assert!(
                (measured - want).abs() < 1e-12 * measured.abs(),
                "n = {n}: measured {measured:.14e} vs expected_drive {want:.14e}"
            );
            // And the per-neuron form the mean is an average of: S - w_i, exactly.
            let s: f64 = (0..n).map(|i| hot.weight_of(n, i)).sum();
            for i in 0..n {
                let want_i = s - hot.weight_of(n, i);
                assert!((sums[i] - want_i).abs() < 1e-15 * s.abs().max(1e-3), "n = {n}, neuron {i}");
            }
            // The self term is the whole difference, and it is not zero.
            let without_self_term = s; // what the dropped-factor formula returns at p = 1
            assert!(
                (without_self_term - want).abs() > 0.9 * s.abs() / n as f64,
                "n = {n}: the n - 1 factor made no difference"
            );
        }
        // Degenerate counts: no neuron, and one neuron with nobody to hear from.
        assert!(Wiring::default().expected_drive(0, 0.5).abs() < 1e-18);
        assert!(Wiring::default().expected_drive(1, 0.5).abs() < 1e-18);
    }

    #[test]
    fn a_wiring_with_the_wrong_signs_is_refused_at_the_boundary() {
        let bad = Wiring { w_exc: -1e-3, ..Wiring::default() };
        assert!(matches!(
            bad.validate().unwrap_err(),
            TopologyError::Weight { name: "w_exc", .. }
        ));
        let bad = Wiring { w_inh: 1e-3, ..Wiring::default() };
        assert!(matches!(
            bad.validate().unwrap_err(),
            TopologyError::Weight { name: "w_inh", .. }
        ));
        let bad = Wiring { w_exc: f64::NAN, ..Wiring::default() };
        assert!(erdos_renyi_gnp(10, 0.5, &bad, 0).is_err(), "a NaN weight reached the builder");
        let bad = Wiring { inhibitory_fraction: f64::INFINITY, ..Wiring::default() };
        assert!(matches!(
            bad.validate().unwrap_err(),
            TopologyError::Probability { name: "inhibitory_fraction", .. }
        ));
    }

    /// Path length has closed forms on two graphs, and both are checked: a complete graph is 1 hop
    /// everywhere, and a directed chain's mean over its reachable pairs is exactly 2 at `n = 5`.
    #[test]
    fn path_length_matches_its_closed_forms() {
        let complete = erdos_renyi_gnp(20, 1.0, &exc(1e-3), 0).unwrap();
        assert_eq!(complete.n_syn, 20 * 19);
        assert_eq!(characteristic_path_length(&complete), Some(1.0));
        assert_eq!(clustering_coefficient(&complete), Some(1.0));

        let mut b = NetBuilder::new(5);
        for i in 0..4u32 {
            b.connect(i, i + 1, 1e-3, 0).unwrap();
        }
        let chain = b.build();
        let s = path_stats(&chain);
        assert_eq!(s.pairs, 20);
        assert_eq!(s.reachable, 10);
        assert_eq!(s.diameter, Some(4));
        // sum over i < j of (j - i) = 20 over 10 reachable pairs.
        assert_eq!(s.mean_hops, Some(2.0));
        assert!(!s.is_strongly_connected());
        assert_eq!(characteristic_path_length(&chain), None, "a chain is not strongly connected");
        assert!(hops_from(&chain, 5).is_none(), "a source past the end has no distances");
        assert_eq!(hops_from(&chain, 4).unwrap()[0], None);
    }

    #[test]
    fn an_empty_or_disconnected_network_refuses_rather_than_reporting_zero() {
        let empty = NetBuilder::new(0).build();
        assert_eq!(
            path_stats(&empty),
            PathStats { pairs: 0, reachable: 0, mean_hops: None, diameter: None }
        );
        assert_eq!(characteristic_path_length(&empty), None);
        assert_eq!(clustering_coefficient(&empty), None);
        let isolated = NetBuilder::new(4).build();
        assert_eq!(clustering_coefficient(&isolated), None, "no neuron has two neighbours");
        assert_eq!(weak_components(&isolated), vec![0, 1, 2, 3]);
        let mut b = NetBuilder::new(4);
        b.connect(0, 1, 1e-3, 0).unwrap();
        b.connect(2, 3, 1e-3, 0).unwrap();
        assert_eq!(weak_components(&b.build()), vec![0, 0, 1, 1]);
    }

    /// (finding 2) **Weak** means the edge direction is ignored, and only an in-star says so.
    ///
    /// Every fixture above is a fixed point of the mistake: four isolated neurons and `0 -> 1`,
    /// `2 -> 3` come out the same whether the search follows synapses or forgets their direction,
    /// because forward reachability from the lowest-indexed member already covers each component.
    /// An in-star does not: `1 -> 0`, `2 -> 0` is one weak component and three strong ones, so
    /// replacing [`undirected_neighbours`] with the plain out-edge adjacency — deleting the word
    /// the function is named for — turns `[0, 0, 0]` into `[0, 1, 2]`.
    #[test]
    fn weak_components_ignore_the_direction_of_a_synapse() {
        let mut b = NetBuilder::new(3);
        b.connect(1, 0, 1e-3, 0).unwrap();
        b.connect(2, 0, 1e-3, 0).unwrap();
        let in_star = b.build();
        assert_eq!(weak_components(&in_star), vec![0, 0, 0], "an in-star is one weak component");
        // The same graph read directedly is three separate reachability classes, which is what
        // `hops_from` reports and what `weak_components` must not.
        assert_eq!(hops_from(&in_star, 1).unwrap()[2], None);
        assert_eq!(hops_from(&in_star, 2).unwrap()[1], None);
        // A directed cycle and a reversed chain, for the same reason at a second shape.
        let mut b = NetBuilder::new(5);
        for i in 0..4u32 {
            b.connect(i + 1, i, 1e-3, 0).unwrap();
        }
        b.connect(0, 4, 1e-3, 0).unwrap();
        assert_eq!(weak_components(&b.build()), vec![0; 5]);
        // Two in-stars stay two components: the undirected reading must not merge everything.
        let mut b = NetBuilder::new(6);
        b.connect(1, 0, 1e-3, 0).unwrap();
        b.connect(2, 0, 1e-3, 0).unwrap();
        b.connect(4, 3, 1e-3, 0).unwrap();
        b.connect(5, 3, 1e-3, 0).unwrap();
        assert_eq!(weak_components(&b.build()), vec![0, 0, 0, 1, 1, 1]);
    }

    /// The winner-take-all regime, simulated. Above the mean-field floor there is exactly one
    /// winner in the steady state, and its rate is the rate it would have alone.
    #[test]
    fn the_winner_take_all_regime_produces_exactly_one_winner() {
        let lif = Lif::default();
        let currents = [6e-9, 4e-9, 3.5e-9, 3e-9];
        let floor = wta_inhibition_floor(&lif, currents[0], currents[1]).unwrap();
        // The closed form, re-derived from the LIF's own equations with the default's printed
        // constants as literals — NOT by calling `v_inf` and `rate` again, which is what this test
        // used to do and which cannot fail, because it is the body of `wta_inhibition_floor`
        // written out a second time:
        //   v_inf(4 nA)  = -65 mV + 10 MΩ * 4 nA              = -25 mV, so the excess is 25 mV
        //   v_inf(6 nA)  = -65 mV + 10 MΩ * 6 nA              = -5 mV
        //   isi(6 nA)    = 2 ms + 20 ms ln((-5 + 65)/(-5 + 50)) = 7.75364 ms -> 128.9717 Hz
        //   floor        = 25 mV / (128.9717 Hz * 20 ms)      = 9.692052 mV
        let by_hand = 25e-3 / ((1.0 / (2e-3 + 20e-3 * (60.0f64 / 45.0).ln())) * 20e-3);
        assert!((by_hand - 9.692_051_811_3e-3).abs() < 1e-12, "the derivation moved: {by_hand}");
        assert!((floor - by_hand).abs() < 1e-15, "floor {floor:.12e} vs {by_hand:.12e}");
        assert!((floor - 9.69e-3).abs() < 0.2e-3, "floor {floor:.4} V is not the expected 9.7 mV");
        // A second parameter set, so that the check is of the formula and not of one cancellation:
        //   tau_m 15 ms, rest -70 mV, threshold -52 mV, reset -60 mV, 8 MΩ, t_ref 3 ms,
        //   i_win 8 nA, i_other 6 nA -> excess 30 mV, isi 5.40514 ms, rate 185.0048 Hz,
        //   floor 30 mV / (185.0048 * 15 ms) = 10.81028 mV.
        let other = Lif {
            tau_m: 15e-3,
            v_rest: -70e-3,
            v_th: -52e-3,
            v_reset: -60e-3,
            r_m: 8e6,
            t_ref: 3e-3,
            v: -70e-3,
            refractory: 0.0,
        };
        let other_by_hand = 30e-3 / ((1.0 / (3e-3 + 15e-3 * (54.0f64 / 46.0).ln())) * 15e-3);
        assert!((other_by_hand - 1.081_027_950_2e-2).abs() < 1e-12, "{other_by_hand}");
        assert!(
            (wta_inhibition_floor(&other, 8e-9, 6e-9).unwrap() - other_by_hand).abs() < 1e-15,
            "the floor disagrees with the hand derivation at a second parameter set"
        );
        // The floor is the weight at which the loser's MEAN potential sits exactly at threshold,
        // which is the statement its doc makes. Checked forwards, in volts, from the definition:
        // v_inf(i_other) - |w| * r_win * tau_m == v_th.
        let r_win = lif.rate(currents[0]).unwrap();
        let held_at = lif.v_inf(currents[1]) - floor * r_win * lif.tau_m;
        assert!((held_at - lif.v_th).abs() < 1e-15, "the floor holds the loser at {held_at} V");

        let dt = 1e-4;
        let counts = |w_inh: f64| -> Vec<u32> {
            let net = winner_take_all(4, w_inh, 0.0, 1).unwrap();
            let mut sim = Sim::new(net, vec![lif; 4], dt, Mode::Clocked).unwrap();
            let mut c = vec![0u32; 4];
            for t in 0..5_000u64 {
                for src in sim.step(&currents) {
                    // Count only the steady state; the first half is the transient in which more
                    // than one unit fires before the winner's inhibition has built up.
                    if t >= 2_500 {
                        c[src as usize] += 1;
                    }
                }
            }
            c
        };

        // 1.5x is where the competition resolves for this LIF; 2x and 3x are margin. All three are
        // asserted so that a change which moves the boundary upward is caught rather than absorbed.
        for mult in [1.5, 2.0, 3.0] {
            let strong = counts(-mult * floor);
            let firing: Vec<usize> = (0..4).filter(|&i| strong[i] > 0).collect();
            assert_eq!(firing, vec![0], "at {mult}x: winners {firing:?} from counts {strong:?}");
            // The winner is unopposed, so it fires at the solo rate its own closed form predicts.
            let expect = r_win * 0.25;
            let got = f64::from(strong[0]);
            assert!(
                (got - expect).abs() / expect < 0.05,
                "at {mult}x the winner fired {got} times in 250 ms against a solo rate of {expect:.1}"
            );
        }
    }

    /// The other half of the regime statement, and the reason the floor's doc calls itself a lower
    /// bound: **at exactly the mean-field floor the competition is not resolved.** The loser's
    /// potential sawtooths around the mean and its peaks still reach threshold.
    #[test]
    fn at_the_mean_field_floor_alone_more_than_one_unit_survives() {
        let lif = Lif::default();
        let currents = [6e-9, 4e-9, 3.5e-9, 3e-9];
        let floor = wta_inhibition_floor(&lif, currents[0], currents[1]).unwrap();
        let net = winner_take_all(4, -floor, 0.0, 1).unwrap();
        let mut sim = Sim::new(net, vec![lif; 4], 1e-4, Mode::Clocked).unwrap();
        let mut c = vec![0u32; 4];
        for t in 0..5_000u64 {
            for src in sim.step(&currents) {
                if t >= 2_500 {
                    c[src as usize] += 1;
                }
            }
        }
        let firing = (0..4).filter(|&i| c[i] > 0).count();
        assert!(firing > 1, "the mean-field floor alone resolved the competition: {c:?}");
    }

    #[test]
    fn the_winner_take_all_floor_refuses_when_nobody_can_win() {
        let lif = Lif::default();
        // v_inf = -55 mV at 1 nA, below the -50 mV threshold: the leader never fires, so no
        // inhibition weight produces a winner and there is no number to return.
        assert!(wta_inhibition_floor(&lif, 1e-9, 0.5e-9).is_none());
        // A runner-up that is already sub-threshold needs no inhibition at all.
        assert_eq!(wta_inhibition_floor(&lif, 6e-9, 1e-9), Some(0.0));
    }

    #[test]
    fn winner_take_all_refuses_a_positive_inhibition_or_a_single_unit() {
        assert!(matches!(
            winner_take_all(1, -1e-3, 0.0, 1).unwrap_err(),
            TopologyError::TooSmall { n: 1, needed: 2, .. }
        ));
        assert!(matches!(
            winner_take_all(4, 1e-3, 0.0, 1).unwrap_err(),
            TopologyError::Weight { name: "w_inh", .. }
        ));
        assert!(matches!(
            winner_take_all(4, -1e-3, -1e-3, 1).unwrap_err(),
            TopologyError::Weight { name: "w_self", .. }
        ));
        assert!(matches!(
            winner_take_all(4, f64::NAN, 0.0, 1).unwrap_err(),
            TopologyError::Weight { name: "w_inh", .. }
        ));
        // The motif itself: all-to-all inhibition, no self-loop unless asked for.
        let net = winner_take_all(5, -1e-3, 0.0, 1).unwrap();
        assert_eq!(net.n_syn, 5 * 4);
        let with_self = winner_take_all(5, -1e-3, 1e-3, 1).unwrap();
        assert_eq!(with_self.n_syn, 5 * 5);
        // A self-excitatory unit is mixed-sign, which is exactly what Dale's law forbids — stated
        // here so that nobody discovers it by failing `dale_check` on a network they trusted.
        assert!(dale_signs(&with_self).is_err());
    }

    /// The 64-bit draw path, which no graph in this test suite is large enough to reach.
    ///
    /// `below_u64` delegates to the 32-bit generator below `u32::MAX` and only then falls through
    /// to its own rejection loop — so the branch that matters for a network of more than about
    /// 65,000 neurons is the one nothing else here exercises. Checked for range and for a mean
    /// near `n / 2`: a modulo-biased or truncating draw shows up in both.
    #[test]
    fn the_sixty_four_bit_draw_covers_its_range_without_bias() {
        let mut rng = Rng::new(404);
        let n = (1u64 << 40) + 12_345;
        let mut sum = 0.0f64;
        let draws = 100_000u32;
        let mut high = 0u32;
        for _ in 0..draws {
            let v = super::below_u64(&mut rng, n);
            assert!(v < n, "draw {v} escaped [0, {n})");
            sum += v as f64;
            if v >= n / 2 {
                high += 1;
            }
        }
        let mean = sum / f64::from(draws);
        let want = n as f64 / 2.0;
        // Standard error of the mean of a uniform on [0, n) is n / sqrt(12 * draws).
        let sem = n as f64 / (12.0 * f64::from(draws)).sqrt();
        assert!((mean - want).abs() < 4.0 * sem, "mean {mean:.3e} vs {want:.3e}");
        assert!(
            (f64::from(high) / f64::from(draws) - 0.5).abs() < 0.01,
            "{high} of {draws} draws in the upper half"
        );
        // The delegating path has to agree with `Rng::below` exactly, or a graph's layout would
        // depend on which side of the u32 boundary its neuron count fell.
        let mut a = Rng::new(9);
        let mut b = Rng::new(9);
        for _ in 0..1000 {
            assert_eq!(super::below_u64(&mut a, 1000), u64::from(b.below(1000)));
        }
    }

    /// The saturation guard in the rewiring loop: a ring with `k = n - 1` is already complete, so
    /// every node has nowhere new to go and `beta = 1` must leave the graph exactly as it was.
    ///
    /// This is the one branch of [`watts_strogatz`] that a normal `k << n` sweep never reaches, and
    /// without it the rewiring loop would spin forever looking for a free target.
    #[test]
    fn a_saturated_ring_keeps_its_edges_however_hard_it_is_rewired() {
        let lattice = watts_strogatz(11, 10, 0.0, &exc(1e-3), 1).unwrap();
        let rewired = watts_strogatz(11, 10, 1.0, &exc(1e-3), 1).unwrap();
        assert_eq!(lattice.n_syn, 11 * 10, "a k = n - 1 ring is the complete graph");
        assert_eq!(lattice, rewired, "a saturated ring was rewired anyway");
        assert_eq!(clustering_coefficient(&rewired), Some(1.0));
        assert_eq!(characteristic_path_length(&rewired), Some(1.0));
    }

    /// (finding 17) The two probability boundaries of `G(n, p)`, which `feedforward` short-circuits
    /// and this generator does not.
    ///
    /// `p = 1.0` gives the complete graph only because [`crate::rng::Rng::next_f64`] never returns
    /// exactly 1.0 — a guarantee that module makes and tests, and that
    /// `path_length_matches_its_closed_forms` silently depends on when it asserts 380 synapses.
    /// Asserted here over several seeds so the dependency is visible at least once.
    #[test]
    fn gnp_at_the_probability_boundaries_is_complete_or_empty() {
        for seed in 0..6u64 {
            let full = erdos_renyi_gnp(40, 1.0, &exc(1e-3), seed).unwrap();
            assert_eq!(full.n_syn, 40 * 39, "p = 1 dropped a synapse at seed {seed}");
            let empty = erdos_renyi_gnp(40, 0.0, &exc(1e-3), seed).unwrap();
            assert_eq!(empty.n_syn, 0, "p = 0 invented a synapse at seed {seed}");
        }
    }

    /// (finding 14) The `Silent` arm of [`MaassC::c_for`], which no [`Wiring`] can reach and which
    /// [`distance_dependent_signed`] can.
    ///
    /// Zero, both ways round, for every scale factor: a neuron whose transmitter is unknown gets no
    /// synapses rather than a probability this rule invented for it. Changing the arm to `1.0`
    /// leaves every other test in this module green.
    #[test]
    fn the_scale_factor_of_a_silent_endpoint_is_zero() {
        for c in [MAASS_2002, MaassC::uniform(1.0), MaassC::uniform(0.0)] {
            for other in [Sign::Excitatory, Sign::Inhibitory, Sign::Silent] {
                assert!(c.c_for(Sign::Silent, other).abs() < 1e-18, "{c:?} pre = Silent");
                assert!(c.c_for(other, Sign::Silent).abs() < 1e-18, "{c:?} post = Silent");
            }
        }
        // The four live arms, in the paper's order, which also pins the pre/post convention: a
        // swap of the two cross terms shows up here and nowhere else.
        assert!((MAASS_2002.c_for(Sign::Excitatory, Sign::Excitatory) - 0.3).abs() < 1e-18);
        assert!((MAASS_2002.c_for(Sign::Excitatory, Sign::Inhibitory) - 0.2).abs() < 1e-18);
        assert!((MAASS_2002.c_for(Sign::Inhibitory, Sign::Excitatory) - 0.4).abs() < 1e-18);
        assert!((MAASS_2002.c_for(Sign::Inhibitory, Sign::Inhibitory) - 0.1).abs() < 1e-18);
        // And a Silent neuron really does drop out of a built graph rather than joining it.
        let g = Grid3::new(4, 4, 1, 50e-6);
        let mut signs = vec![Sign::Excitatory; 16];
        signs[7] = Sign::Silent;
        let net = distance_dependent_signed(
            &g,
            &MaassC::uniform(1.0),
            g.lambda_of(2.0),
            &Wiring::default(),
            &signs,
            3,
        )
        .unwrap();
        assert_eq!(net.out_of(7).count(), 0, "a Silent neuron sent a synapse");
        assert!(net.n_syn > 0, "the rest of the network was built");
        assert_eq!(dale_signs(&net).unwrap()[7], Sign::Silent);
    }

    /// (finding 15) The preferential-attachment probe reports failure instead of returning a short
    /// edge list.
    ///
    /// [`barabasi_albert`] can argue that the probe always finds a node — its array holds at least
    /// `m + 1` distinct ones — but an argument is not a check, and the old loop could fall through
    /// in silence and leave an arriving node with fewer than `m` edges. The branch is unreachable
    /// from the generator, so it is tested on the helper directly.
    #[test]
    fn the_preferential_attachment_probe_refuses_rather_than_returning_a_short_edge_list() {
        // Every entry already taken: the probe has nothing to return and says so.
        assert_eq!(super::probe_distinct(&[3, 3, 4], &[3, 4], 0), None);
        assert_eq!(super::probe_distinct(&[], &[], 0), None);
        // The ordinary case: the first entry at or after `start`, cyclically, that is free.
        assert_eq!(super::probe_distinct(&[5, 6, 7], &[], 1), Some(6));
        assert_eq!(super::probe_distinct(&[5, 6, 7], &[6, 7], 1), Some(5));
        assert_eq!(super::probe_distinct(&[5, 6, 7], &[5], 0), Some(6));
        // Degree weighting survives the refactor: a node that appears twice is twice as likely to
        // be the first one found, which is the whole of Batagelj and Brandes' trick.
        let repeated = [1u32, 1, 2];
        let hits = (0..3).filter(|&s| super::probe_distinct(&repeated, &[], s) == Some(1)).count();
        assert_eq!(hits, 2, "the repeated endpoint lost its extra weight");
        // And the generator itself still gives every arriving node exactly m edges.
        for &(n, m) in &[(60usize, 3usize), (60, 5)] {
            let net = barabasi_albert(n, m, &exc(1e-3), 11).unwrap();
            let deg = undirected_degrees(&net);
            for (v, d) in deg.iter().enumerate().skip(m + 1) {
                assert!(*d >= m, "node {v} arrived with {d} edges, fewer than m = {m}");
            }
        }
    }

    /// (finding 16) A grid whose dimensions overflow a `usize` is refused, not wrapped.
    ///
    /// `nx * ny * nz` wrapping in release is the dangerous case: the wrapped product is a small,
    /// plausible neuron count that [`crate::net::NetBuilder`] would accept, so the caller gets a
    /// four-neuron network where they asked for an impossible one. [`Grid3::len`] saturates and
    /// [`Grid3::validate`] rejects.
    #[test]
    fn a_grid_whose_dimensions_overflow_is_refused_rather_than_wrapped() {
        let huge = Grid3::new(usize::MAX, 2, 2, 50e-6);
        assert_eq!(huge.len(), usize::MAX, "the product wrapped instead of saturating");
        assert!(matches!(huge.validate().unwrap_err(), TopologyError::IndexSpace { .. }));
        assert!(matches!(
            distance_dependent(&huge, &MAASS_2002, 1e-4, &Wiring::default(), 0).unwrap_err(),
            TopologyError::IndexSpace { .. }
        ));
        // Past u32 but inside usize: still refused, by the index-space rule every generator obeys.
        let wide = Grid3::new(5_000_000_000, 1, 1, 50e-6);
        assert_eq!(wide.len(), 5_000_000_000);
        assert!(matches!(
            wide.validate().unwrap_err(),
            TopologyError::IndexSpace { n: 5_000_000_000 }
        ));
        // A grid that fits is untouched.
        assert!(Grid3::new(15, 3, 3, 50e-6).validate().is_ok());
        assert!(!Grid3::new(15, 3, 3, 50e-6).is_empty());
        assert!(Grid3::new(15, 0, 3, 50e-6).is_empty());
    }

    /// (finding 9) The rejection zone holds a **whole number of residue classes**, which is the
    /// entire content of the phrase "without modulo bias".
    ///
    /// The shipped form, `u64::MAX - (u64::MAX % n) - (n - 1)`, does not: its acceptance region is
    /// `(q - 1) n + 2` wide, so residues 0 and 1 are over-represented for every `n`, and for
    /// `n > 2^63` it collapses to `{0, 1}` — a draw that returns only 0 or 1 and loops about `2^63`
    /// times to do it. This assertion is exact and in `u128`, so it fails in microseconds rather
    /// than hanging.
    #[test]
    fn unbiased_zone_holds_whole_residue_classes() {
        let ns = [
            2u64,
            3,
            7,
            400,
            1000,
            1_000_000,
            (1u64 << 32) + 1,
            (1u64 << 40) + 12_345,
            1u64 << 62,
            (1u64 << 62) + 1,
            (1u64 << 63) + 7,
            u64::MAX,
            u64::MAX - 1,
        ];
        for n in ns {
            let zone = super::unbiased_zone(n);
            let size = u128::from(zone) + 1;
            assert_eq!(size % u128::from(n), 0, "n = {n}: acceptance region {size} is not n * q");
            // It is also the LARGEST such region: one more class would not fit in 2^64.
            assert!(size + u128::from(n) > 1u128 << 64, "n = {n}: the zone left a class on the table");
            // Which means at least half of the 64-bit range is accepted, so the rejection loop
            // terminates in about two draws however large n is.
            assert!(size * 2 > 1u128 << 64, "n = {n}: rejection is more likely than acceptance");
        }
        // A power of two rejects nothing at all.
        assert_eq!(super::unbiased_zone(1u64 << 62), u64::MAX);
        // And the draw itself stays in range and spreads over it for an n past 2^63, where the old
        // formula returned only 0 and 1.
        let mut rng = Rng::new(17);
        let n = (1u64 << 63) + 7;
        let mut above = 0u32;
        for _ in 0..400 {
            let v = super::below_u64(&mut rng, n);
            assert!(v < n, "draw {v} escaped [0, {n})");
            if v > (1u64 << 62) {
                above += 1;
            }
        }
        assert!(above > 100, "only {above} of 400 draws from a 2^63 range exceeded 2^62");
    }

    /// (finding 4a) **The four Maass scale factors are load-bearing here and nowhere else.**
    ///
    /// Until this test existed, `MAASS_2002` was inert: swapping `ei` and `ie`, or setting all four
    /// factors to 0.01, left every test in this module green, because the only one that built a
    /// network with them asserted `n_syn > 0`. What pins them is the **rate of each pair type**:
    /// with a random partition the four types all have thousands of candidate pairs, so the
    /// expected count of each is a Poisson-binomial sum with a closed-form mean and variance, and
    /// each factor is separately visible in the result.
    ///
    /// The last two assertions are the test checking itself: the same measurement is compared
    /// against the predictions a **wrong** parameter set would make, and has to be many sigma from
    /// them. A test that cannot reject the alternative is not testing the constant.
    #[test]
    fn the_maass_scale_factors_set_the_rate_of_each_pair_type() {
        let grid = Grid3::new(12, 12, 3, 50e-6);
        let n = grid.len();
        let lambda = grid.lambda_of(2.0);
        let wiring = Wiring::default();
        let slot = |pre: Sign, post: Sign| {
            2 * usize::from(pre == Sign::Inhibitory) + usize::from(post == Sign::Inhibitory)
        };
        // The paper's four factors, transcribed a second time and independently of the constant
        // under test, so that this measurement fails when `MAASS_2002` moves rather than moving
        // with it: a prediction computed from the same constant the generator used would agree
        // with any value at all.
        let paper = MaassC { ee: 0.3, ei: 0.2, ie: 0.4, ii: 0.1 };
        assert_eq!(paper, MAASS_2002, "MAASS_2002 is no longer the paper's parameter list");
        // Mean and variance of the count of each pair type under a given parameter set.
        let predict = |c: &MaassC, signs: &[Sign]| {
            let (mut mean, mut var) = ([0.0f64; 4], [0.0f64; 4]);
            for a in 0..n {
                for d in 0..n {
                    if a == d {
                        continue;
                    }
                    let ratio = grid.distance(a, d).unwrap() / lambda;
                    let p = c.c_for(signs[a], signs[d]) * (-(ratio * ratio)).exp();
                    mean[slot(signs[a], signs[d])] += p;
                    var[slot(signs[a], signs[d])] += p * (1.0 - p);
                }
            }
            (mean, var)
        };
        for seed in [5u64, 6] {
            let signs = shuffled_partition(n, CORTICAL_INHIBITORY_FRACTION, seed).unwrap();
            let net =
                distance_dependent_signed(&grid, &MAASS_2002, lambda, &wiring, &signs, seed).unwrap();
            let mut obs = [0u64; 4];
            for a in 0..n {
                for (post, _, _) in net.out_of(a) {
                    obs[slot(signs[a], signs[post as usize])] += 1;
                }
            }
            let (mean, var) = predict(&paper, &signs);
            let names = ["EE", "EI", "IE", "II"];
            for t in 0..4 {
                let sigma = var[t].sqrt();
                assert!(
                    (obs[t] as f64 - mean[t]).abs() < 4.0 * sigma,
                    "seed {seed} {}: {} synapses against a predicted {:.1} +/- {sigma:.1}",
                    names[t],
                    obs[t],
                    mean[t]
                );
            }
            // Would this measurement notice if `ei` and `ie` were transposed? They are 0.2 and 0.4,
            // so the two cross terms swap and the observed counts land 10+ sigma from the wrong
            // prediction. (Measured at seed 5: EI 313 against a swapped prediction of 642.8.)
            let swapped = MaassC { ei: paper.ie, ie: paper.ei, ..paper };
            let (bad_mean, bad_var) = predict(&swapped, &signs);
            for t in [1usize, 2] {
                let z = (obs[t] as f64 - bad_mean[t]).abs() / bad_var[t].sqrt();
                assert!(z > 8.0, "seed {seed} {}: a swapped ei/ie is only {z:.1} sigma away", names[t]);
            }
            // And if every factor were 0.01 — the other mutation that used to survive.
            let flat = MaassC::uniform(0.01);
            let (flat_mean, flat_var) = predict(&flat, &signs);
            for t in 0..4 {
                let z = (obs[t] as f64 - flat_mean[t]).abs() / flat_var[t].sqrt();
                assert!(z > 8.0, "seed {seed} {}: a 0.01 rule is only {z:.1} sigma away", names[t]);
            }
        }
    }

    /// (finding 4b) **A contiguous E/I partition on a lattice is a slab, and the slab is visible in
    /// the built network's drive.**
    ///
    /// [`Wiring`] numbers the inhibitory neurons last and [`Grid3`] numbers positions x-fastest, so
    /// on the 15x3x3 Maass column all 27 inhibitory neurons land on the single face `z = 2`. Under
    /// a distance-dependent rule that is a systematic gradient across the column — 1.600, −1.578,
    /// −0.378 mV of mean incoming weight by z-slab, a 3.18 mV spread that exists only because of
    /// the index convention. Maass, Natschläger and Markram pick their inhibitory neurons at
    /// random, which is what [`shuffled_partition`] does, and the spread then averages 0.88 mV over
    /// five seeds.
    ///
    /// Both halves are asserted, because the first is the one a reader has to know before
    /// reproducing a figure with [`distance_dependent`].
    #[test]
    fn a_contiguous_partition_makes_the_inhibitory_population_a_slab() {
        let g = Grid3::maass_column(50e-6);
        let lambda = g.lambda_of(2.0);
        let wiring = Wiring::default();
        let block = wiring.signs(135);
        assert_eq!(block.iter().filter(|s| **s == Sign::Inhibitory).count(), 27);
        // Every inhibitory neuron is on the face z = 2, and z = 2 is 45 of the 135 neurons.
        for (i, sign) in block.iter().enumerate() {
            if *sign == Sign::Inhibitory {
                let z = g.position(i).unwrap()[2];
                assert!((z - 2.0 * 50e-6).abs() < 1e-18, "inhibitory neuron {i} is not on z = 2");
            }
        }
        let zmeans = |net: &crate::net::Net| -> [f64; 3] {
            let ins = in_weight_sums(net);
            let mut out = [0.0f64; 3];
            for (i, v) in ins.iter().enumerate() {
                out[i / 45] += v / 45.0;
            }
            out
        };
        let spread = |m: [f64; 3]| {
            m.iter().cloned().fold(f64::MIN, f64::max) - m.iter().cloned().fold(f64::MAX, f64::min)
        };
        let contiguous = distance_dependent(&g, &MAASS_2002, lambda, &wiring, 1).unwrap();
        let cz = zmeans(&contiguous);
        // The exact gradient, as a golden value: this is a deterministic generator on a fixed seed,
        // and the numbers are the ones the module doc quotes.
        for (got, want) in cz.iter().zip([1.600e-3, -1.577_777_777_777_777e-3, -0.377_777_777_777_777e-3]) {
            assert!((got - want).abs() < 1e-15, "z-slab drive {cz:?} moved");
        }
        assert!((spread(cz) - 3.177_777_777_777_78e-3).abs() < 1e-15, "spread {}", spread(cz));

        // The random partition: the same 27 inhibitory neurons, scattered.
        let mut total = 0.0f64;
        let seeds = [1u64, 2, 3, 4, 5];
        for seed in seeds {
            let signs = shuffled_partition(135, CORTICAL_INHIBITORY_FRACTION, seed).unwrap();
            assert_eq!(
                signs.iter().filter(|s| **s == Sign::Inhibitory).count(),
                27,
                "seed {seed}: the shuffle changed the inhibitory count"
            );
            // Hypergeometric: 9 per z-slab in expectation, sd 2.2, so every slab is populated.
            for z in 0..3 {
                let c = (0..135).filter(|i| i / 45 == z && signs[*i] == Sign::Inhibitory).count();
                assert!(
                    c.abs_diff(9) <= 8,
                    "seed {seed}: z-slab {z} holds {c} of the 27 inhibitory neurons"
                );
            }
            let net =
                distance_dependent_signed(&g, &MAASS_2002, lambda, &wiring, &signs, 1).unwrap();
            // The network is wired to the partition it was given, not to the index block.
            dale_check(&net, &signs).unwrap();
            total += spread(zmeans(&net));
        }
        let mean_spread = total / seeds.len() as f64;
        assert!(
            mean_spread < 0.5 * spread(cz),
            "the scattered partition's mean z-slab spread is {mean_spread:.3e} V against the \
             contiguous {:.3e} V",
            spread(cz)
        );
    }

    /// [`shuffled_partition`] is a permutation of [`dale_partition`], drawn from the seed.
    #[test]
    fn a_shuffled_partition_is_a_seeded_permutation_of_the_contiguous_one() {
        for n in [1usize, 2, 10, 135, 500] {
            let block = dale_partition(n, CORTICAL_INHIBITORY_FRACTION).unwrap();
            let shuffled = shuffled_partition(n, CORTICAL_INHIBITORY_FRACTION, 9).unwrap();
            assert_eq!(shuffled.len(), n);
            for sign in [Sign::Excitatory, Sign::Inhibitory] {
                assert_eq!(
                    block.iter().filter(|s| **s == sign).count(),
                    shuffled.iter().filter(|s| **s == sign).count(),
                    "n = {n}: the shuffle changed how many neurons are {sign:?}"
                );
            }
        }
        // Same seed, same partition; a different seed moves it. (n = 135, 27 inhibitory: the
        // chance of two seeds agreeing by accident is 1 / C(135, 27), about 1e-29.)
        assert_eq!(shuffled_partition(135, 0.2, 4).unwrap(), shuffled_partition(135, 0.2, 4).unwrap());
        assert_ne!(shuffled_partition(135, 0.2, 4).unwrap(), shuffled_partition(135, 0.2, 5).unwrap());
        assert_ne!(shuffled_partition(135, 0.2, 4).unwrap(), dale_partition(135, 0.2).unwrap());
        // Degenerate fractions still land exactly on the block partition's counts.
        assert!(shuffled_partition(20, 0.0, 1).unwrap().iter().all(|s| *s == Sign::Excitatory));
        assert!(shuffled_partition(20, 1.0, 1).unwrap().iter().all(|s| *s == Sign::Inhibitory));
        assert!(matches!(
            shuffled_partition(10, 1.5, 0).unwrap_err(),
            TopologyError::Probability { name: "inhibitory_fraction", .. }
        ));
    }

    /// [`distance_dependent`] is [`distance_dependent_signed`] fed the contiguous block, **bit for
    /// bit** — the refactor that introduced the signed entry point moved no synapse.
    ///
    /// Pinned by a fingerprint as well as by equality, so that a future change to the draw order
    /// has to be deliberate: the Maass column at seed 1 is 619 synapses hashing to
    /// `0x10faf783b3eee553` under FNV-1a over `(pre, post, weight bits, delay)`.
    #[test]
    fn the_contiguous_partition_is_the_shuffled_path_with_a_block_input() {
        let g = Grid3::maass_column(50e-6);
        let w = Wiring::default();
        let lambda = g.lambda_of(2.0);
        let direct = distance_dependent(&g, &MAASS_2002, lambda, &w, 1).unwrap();
        let signed =
            distance_dependent_signed(&g, &MAASS_2002, lambda, &w, &w.signs(135), 1).unwrap();
        assert_eq!(direct, signed, "the two entry points disagree");
        assert_eq!(direct.n_syn, 619);
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for pre in 0..direct.n {
            for (post, weight, delay) in direct.out_of(pre) {
                for b in [pre as u64, u64::from(post), weight.to_bits(), u64::from(delay)] {
                    h ^= b;
                    h = h.wrapping_mul(0x100_0000_01b3);
                }
            }
        }
        assert_eq!(h, 0x10fa_f783_b3ee_e553, "the Maass column at seed 1 is not the graph it was");
        // The length of the partition is checked, not assumed.
        assert_eq!(
            distance_dependent_signed(&g, &MAASS_2002, lambda, &w, &w.signs(134), 1).unwrap_err(),
            TopologyError::SignsLength { declared: 134, neurons: 135 }
        );
        assert_eq!(
            distance_dependent_signed(&g, &MAASS_2002, lambda, &w, &[], 1).unwrap_err(),
            TopologyError::SignsLength { declared: 0, neurons: 135 }
        );
        let e = TopologyError::SignsLength { declared: 134, neurons: 135 };
        assert!(e.to_string().contains("134") && e.to_string().contains("135"), "{e}");
    }

    #[test]
    fn the_error_messages_name_the_offending_value() {
        let e = erdos_renyi_gnp(10, 2.0, &exc(1e-3), 0).unwrap_err();
        assert!(e.to_string().contains("p = 2"), "{e}");
        let e = erdos_renyi_gnm(10, 91, &exc(1e-3), 0).unwrap_err();
        assert!(e.to_string().contains("91") && e.to_string().contains("90"), "{e}");
        let e = watts_strogatz(10, 3, 0.0, &exc(1e-3), 0).unwrap_err();
        assert!(e.to_string().contains("k = 3"), "{e}");
        let e = TopologyError::IndexSpace { n: 5_000_000_000 };
        assert!(e.to_string().contains("5000000000"), "{e}");
        let v = DaleViolation::Mixed { neuron: 7, positive: 3, negative: 2 };
        assert!(v.to_string().contains("neuron 7"), "{v}");
    }

    /// (survivor 1) [`Wiring::excitatory_only`] declares **no** inhibitory neuron, at any size, and
    /// every synapse a network built with it carries is exactly `w`.
    ///
    /// The suite could not see this because every `exc(w)` fixture in this module asserts a count,
    /// a distance, a clustering coefficient or a hop, and not one of them reads a weight. Setting
    /// that constructor's `inhibitory_fraction` to [`CORTICAL_INHIBITORY_FRACTION`] therefore left
    /// the whole module green: the trailing fifth of the neurons became "inhibitory" carrying
    /// `w_inh = 0.0`, which [`crate::net::NetBuilder`] accepts, so the graph kept its shape and
    /// only its weights went to zero — and [`dale_signs`] would have read that fifth as
    /// [`Sign::Silent`].
    #[test]
    fn the_excitatory_only_wiring_declares_no_inhibitory_neuron_at_any_size() {
        let w = exc(2.5e-3);
        assert_eq!(w.inhibitory_fraction, 0.0, "excitatory_only carried an inhibitory fraction");
        assert_eq!(w.w_inh, 0.0);
        for n in [1usize, 2, 5, 10, 135, 1000] {
            assert_eq!(w.n_inhibitory(n), 0, "n = {n}: excitatory_only grew an inhibitory block");
            for i in [0, n - 1] {
                assert_eq!(w.sign_of(n, i), Sign::Excitatory, "n = {n}, neuron {i}");
                assert_eq!(w.weight_of(n, i), 2.5e-3, "n = {n}, neuron {i}");
            }
            // The whole point of the constructor, stated as the arithmetic its doc states: with no
            // inhibitory population every neuron counts as excitatory, so the drive is
            // `p * (n-1)/n * (n * w_exc)` with no cancelling term. Written in that order, and
            // asserted as equality rather than within a tolerance, because at `p = 1.0` — where
            // multiplication by one is exact — it is the same two multiplications the method
            // itself performs on the same numbers.
            let drive = w.expected_drive(n, 1.0);
            let closed_form = ((n - 1) as f64 / n as f64) * (n as f64 * 2.5e-3);
            assert_eq!(drive, closed_form, "n = {n}: expected drive {drive}");
        }
        // And in a built network, which is where the mistake would have been felt.
        let net = erdos_renyi_gnp(50, 0.4, &w, 3).unwrap();
        assert!(net.n_syn > 0, "the fixture built no synapse");
        assert!(
            net.w.iter().all(|x| *x == 2.5e-3),
            "a synapse left an excitatory_only wiring carrying something other than w"
        );
        assert_eq!(dale_signs(&net).unwrap(), vec![Sign::Excitatory; 50]);
    }

    /// (survivor 2) A non-finite weight is refused by [`Wiring::validate`] **itself**, naming the
    /// field, rather than by [`crate::net::NetBuilder`] downstream.
    ///
    /// The hole was the shape of the old assertion — `erdos_renyi_gnp(10, 0.5, &bad, 0).is_err()`.
    /// `NetBuilder::connect` refuses a non-finite weight on its own with
    /// `NetError::NonFiniteWeight`, which `From<NetError>` folds into [`TopologyError::Net`], so
    /// that call is `Err` whether or not `Wiring::validate` ever calls `is_finite`. Deleting the
    /// finiteness test from `validate` left every assertion in the module green. `NaN != NaN`, so
    /// the variant is matched and its payload read rather than compared for equality.
    #[test]
    fn a_non_finite_weight_is_refused_by_the_wiring_and_not_by_the_builder() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let w = Wiring { w_exc: bad, ..Wiring::default() };
            match w.validate().unwrap_err() {
                TopologyError::Weight { name, value } => {
                    assert_eq!(name, "w_exc", "w_exc = {bad} was blamed on {name}");
                    assert!(!value.is_finite(), "the error reported a finite {value} for {bad}");
                }
                e => panic!("w_exc = {bad} gave {e:?} instead of a Weight error"),
            }
            let w = Wiring { w_inh: bad, ..Wiring::default() };
            match w.validate().unwrap_err() {
                TopologyError::Weight { name, value } => {
                    assert_eq!(name, "w_inh", "w_inh = {bad} was blamed on {name}");
                    assert!(!value.is_finite(), "the error reported a finite {value} for {bad}");
                }
                e => panic!("w_inh = {bad} gave {e:?} instead of a Weight error"),
            }
            // The generator reports the wiring's refusal, not the builder's: a `Net` error here
            // means the bad weight travelled all the way to a synapse before anyone looked at it.
            let w = Wiring { w_exc: bad, ..Wiring::default() };
            let e = erdos_renyi_gnp(10, 0.5, &w, 0).unwrap_err();
            assert!(
                matches!(e, TopologyError::Weight { name: "w_exc", .. }),
                "w_exc = {bad} reached the builder and came back as {e:?}"
            );
        }
    }

    /// (survivor 3) [`Wiring::weight_for`] gives a [`Sign::Silent`] neuron weight **zero**, whatever
    /// the wiring's two live weights are.
    ///
    /// No built network can show this, which is why it survived: the only caller that can reach the
    /// `Silent` arm is [`distance_dependent_signed`], and there [`MaassC::c_for`] has already given
    /// a `Silent` endpoint probability zero, so the weight is computed and then never attached to a
    /// synapse. The arm is reachable only through the public method, so that is where it is pinned.
    /// Both fixtures have a **non-zero** `w_exc`, or returning `w_exc` for `Silent` would be the
    /// same number as returning zero and the assertion could not fail.
    #[test]
    fn a_silent_neuron_is_given_no_weight_by_any_wiring() {
        for w in [Wiring::default(), exc(7e-3), Wiring::balanced(2e-3, 4)] {
            assert_ne!(w.w_exc, 0.0, "the fixture's w_exc is zero, so this assertion is blind");
            assert_eq!(w.weight_for(Sign::Silent), 0.0, "{w:?} gave a Silent neuron a weight");
            assert_eq!(w.weight_for(Sign::Excitatory), w.w_exc, "{w:?}");
            assert_eq!(w.weight_for(Sign::Inhibitory), w.w_inh, "{w:?}");
        }
        // The three-way table is the same one [`Sign::as_f64`] reports, and the product of the two
        // is what a caller uses to sign a drive: zero times anything stays zero.
        assert_eq!(Sign::Silent.as_f64() * Wiring::default().w_exc, 0.0);
    }

    /// (survivor 4) [`shuffled_partition`] draws its swap partner from `0..=i`, **including `i`**,
    /// so every position is equally likely to end up holding the inhibitory neuron.
    ///
    /// Narrowing that range to `0..i` is Sattolo's algorithm, which is uniform over the `(n-1)!`
    /// **cyclic** permutations rather than all `n!` orders — and a cyclic permutation has no fixed
    /// point, so the element that started at the last index can never stay there. Nothing in the
    /// suite measured uniformity: `a_shuffled_partition_is_a_seeded_permutation_of_the_contiguous_one`
    /// checks the inhibitory *count*, that a seed repeats, and that two seeds differ, all of which
    /// Sattolo's satisfies exactly.
    ///
    /// The instrument is `n = 4` at a quarter inhibitory, where the partition holds exactly one
    /// inhibitory neuron and its position is directly observable. Over 1,000 seeds each position
    /// should hold it `1000 / 4 = 250` times, `sd = sqrt(1000 * 0.25 * 0.75) = 13.69`; the band is
    /// four of those. This implementation measures 251, 252, 235 and 262. Sattolo's measures 351,
    /// 333, 316 and **0** — the last position starved by construction, at 18 sigma.
    #[test]
    fn the_partition_shuffle_can_leave_the_inhibitory_neuron_where_it_started() {
        let trials = 1000usize;
        let mut at = [0usize; 4];
        for seed in 0..trials as u64 {
            let p = shuffled_partition(4, 0.25, seed).unwrap();
            let inh: Vec<usize> =
                (0..4).filter(|&i| p[i] == Sign::Inhibitory).collect();
            assert_eq!(inh.len(), 1, "seed {seed}: round(0.25 * 4) is one inhibitory neuron");
            at[inh[0]] += 1;
        }
        let sd = (trials as f64 * 0.25 * 0.75).sqrt();
        for (i, &count) in at.iter().enumerate() {
            let z = (count as f64 - trials as f64 / 4.0) / sd;
            assert!(z.abs() < 4.0, "position {i} held the inhibitory neuron {count} times, {z:.1} sigma from 250 (all four: {at:?})");
        }
        // The same statement at its smallest: with two neurons and one of them inhibitory, the
        // single swap is `i = 1, j in 0..=1`, so both orders have to appear. Sattolo's `j in 0..1`
        // is the constant 0 and produces `[I, E]` for every seed there is.
        let mut orders: BTreeSet<String> = BTreeSet::new();
        for seed in 0..64u64 {
            let p = shuffled_partition(2, 0.5, seed).unwrap();
            orders.insert(p.iter().map(|s| if *s == Sign::Inhibitory { 'I' } else { 'E' }).collect());
        }
        assert_eq!(orders.len(), 2, "the n = 2 shuffle reached only {orders:?}");
    }

    /// (survivor 5) [`shuffled_partition`]'s stream is pinned by a **golden vector**, because its
    /// direction is not visible in any distribution.
    ///
    /// Fisher-Yates run upward — `for i in 1..len` with the partner still drawn from `0..=i` — is
    /// *also* uniform over all `n!` orders; enumerating `n = 3` gives all six permutations once
    /// each either way round. So no uniformity test, no count test and no "two seeds differ" test
    /// can separate the two walks. What separates them is the stream, and the stream is the
    /// promise: "the same seed gives the same partition on every platform".
    /// `every_generator_is_deterministic_by_seed` builds each generator twice and compares, which
    /// can see nondeterminism and cannot see a change of algorithm.
    ///
    /// The vector below is this implementation's output, recorded here so a future change to the
    /// walk has to be deliberate — the same instrument
    /// `the_contiguous_partition_is_the_shuffled_path_with_a_block_input` uses on the Maass column.
    /// Measured: `n = 12` at a quarter inhibitory, seed 7, is `EEIEEIEEEEIE`. The upward walk gives
    /// `EEEIEIEEEEIE` and Sattolo's gives `EIIEEEIEEEEE` on the same seed.
    #[test]
    fn the_partition_shuffle_reproduces_a_recorded_stream_and_not_merely_itself() {
        let letters = |p: &[Sign]| -> String {
            p.iter().map(|s| if *s == Sign::Inhibitory { 'I' } else { 'E' }).collect()
        };
        assert_eq!(letters(&shuffled_partition(12, 0.25, 7).unwrap()), "EEIEEIEEEEIE");
        assert_eq!(letters(&shuffled_partition(8, 0.25, 7).unwrap()), "EEEEEEII");
        // A second seed, so the vector pins the stream and not one lucky draw.
        assert_eq!(letters(&shuffled_partition(12, 0.25, 8).unwrap()), "IEEIEEEIEEEE");
        // The first 24 neurons of the 135-neuron Maass column at seed 4, the partition the
        // distance-dependent tests hand to `distance_dependent_signed`.
        let column = shuffled_partition(135, CORTICAL_INHIBITORY_FRACTION, 4).unwrap();
        assert_eq!(&letters(&column)[..24], "EIEEIIEEEEEEEEEEIIEEEEEE");
        assert_eq!(column.iter().filter(|s| **s == Sign::Inhibitory).count(), 27);
    }

    /// (survivor 6) `G(n, m)` draws Floyd's algorithm over the **top** of the code space, so the
    /// `m` chosen codes are a uniform subset of all `n(n-1)` ordered pairs.
    ///
    /// Running the loop over `0..m` instead of `(space - m)..space` still returns `m` distinct
    /// codes and still emits no self-loop, so `gnm_returns_exactly_m_distinct_synapses_and_no_self_loops`
    /// cannot see it — but every code then lands below `m`, and since a code decodes as
    /// `a = code / (n - 1)`, only the first `ceil(m / (n - 1))` neurons can ever be presynaptic.
    /// At `n = 50, m = 100` that is three rows out of fifty.
    ///
    /// Each synapse's code is recovered here — `pre * (n - 1) + r` with `r` the post index past the
    /// skipped diagonal — and two statistics of the `m` codes are checked against sampling without
    /// replacement from `0..space`. The mean of `m` draws has expectation `(space - 1) / 2` and
    /// variance `(space^2 - 1) / 12 / m * (space - m) / (space - 1)`; the count in the top half is
    /// hypergeometric with mean `m / 2` and variance `m / 4 * (space - m) / (space - 1)`. Both
    /// bands are four sigma. This implementation measures the mean 0.59 sigma high at seed 5;
    /// drawing from the floor puts it 17 sigma low and empties the top half entirely.
    #[test]
    fn gnm_samples_the_whole_pair_space_and_not_only_its_floor() {
        let (n, m) = (50usize, 100usize);
        let space = (n as u64) * (n as u64 - 1);
        for seed in [5u64, 11, 23] {
            let net = erdos_renyi_gnm(n, m, &exc(1e-3), seed).unwrap();
            assert_eq!(net.n_syn, m);
            let mut codes: Vec<u64> = Vec::with_capacity(m);
            for pre in 0..net.n {
                for (post, _, _) in net.out_of(pre) {
                    let r = if (post as usize) < pre { u64::from(post) } else { u64::from(post) - 1 };
                    codes.push(pre as u64 * (n as u64 - 1) + r);
                }
            }
            assert_eq!(codes.len(), m);
            let mean = codes.iter().sum::<u64>() as f64 / m as f64;
            let expected = (space - 1) as f64 / 2.0;
            let finite_population = (space - m as u64) as f64 / (space - 1) as f64;
            let sd_mean = ((((space * space) as f64 - 1.0) / 12.0) / m as f64 * finite_population).sqrt();
            let z = (mean - expected) / sd_mean;
            assert!(z.abs() < 4.0, "seed {seed}: code mean {mean:.1} is {z:.1} sigma from {expected:.1}");
            let top = codes.iter().filter(|c| **c >= space / 2).count();
            let sd_top = (m as f64 / 4.0 * finite_population).sqrt();
            let z_top = (top as f64 - m as f64 / 2.0) / sd_top;
            assert!(z_top.abs() < 4.0, "seed {seed}: {top} of {m} codes in the top half, {z_top:.1} sigma from 50");
            // And the presynaptic side reaches past the first few rows, which is the visible
            // consequence: a floor-drawn sample cannot name a neuron above `m / (n - 1)`.
            let rows: BTreeSet<u64> = codes.iter().map(|c| c / (n as u64 - 1)).collect();
            assert!(rows.len() > n / 2, "seed {seed}: only {} of {n} neurons were presynaptic", rows.len());
        }
    }

    /// (survivor 7) Every node of [`barabasi_albert`]'s seed clique enters the repeated-endpoint
    /// array **once per incident edge**, so every one of them can be attached to.
    ///
    /// The array is the urn that makes attachment degree-weighted, and pushing one end of each seed
    /// edge twice keeps its *length* at two per edge — which is why
    /// `barabasi_albert_has_exactly_the_predicted_edge_count` stays green — while leaving the
    /// highest-numbered seed node out of it altogether. That node is then never drawn, never gains
    /// an edge, and keeps the clique degree `m` forever, and the degree tests average over the
    /// whole network where one starved node is invisible.
    ///
    /// The sharpest instrument is the smallest graph the generator builds: `n = 3, m = 1` is the
    /// single seed edge `0 - 1` and one arriving node that draws exactly one target. With the urn
    /// `[0, 1]` the target is 0 or 1 with equal probability; with `[0, 0]` it is 0 every time.
    /// Over 400 seeds this implementation measures 196 and 204, `sd = sqrt(400 * 0.25) = 10`.
    #[test]
    fn every_seed_clique_node_enters_the_preferential_attachment_urn() {
        let trials = 400usize;
        let mut to_one = 0usize;
        for seed in 0..trials as u64 {
            let net = barabasi_albert(3, 1, &exc(1e-3), seed).unwrap();
            let neighbours = undirected_degrees(&net);
            assert_eq!(neighbours[2], 1, "seed {seed}: the arriving node did not get m = 1 edge");
            // Node 2's only neighbour, read off the reciprocal pair it was given.
            let target = net.out_of(2).map(|(post, _, _)| post).next().unwrap();
            if target == 1 {
                to_one += 1;
            }
        }
        let sd = (trials as f64 * 0.25).sqrt();
        let z = (to_one as f64 - trials as f64 / 2.0) / sd;
        assert!(
            z.abs() < 4.0,
            "the arriving node chose the second seed node {to_one} times of {trials}, {z:.1} sigma from 200"
        );
        // At a larger seed clique the same claim reads as a degree: the last seed node is drawn
        // like any other, so it leaves the clique degree behind. Starved, it would sit at exactly m.
        for &(n, m) in &[(400usize, 3usize), (200, 2)] {
            for seed in [1u64, 2, 3] {
                let deg = undirected_degrees(&barabasi_albert(n, m, &exc(1e-3), seed).unwrap());
                assert!(
                    deg[m] > m,
                    "n = {n}, m = {m}, seed {seed}: the last seed node kept its clique degree {}",
                    deg[m]
                );
            }
        }
    }

    /// (survivor 8) [`barabasi_albert`] refuses a network too small for its seed clique **by name**.
    ///
    /// Deleting the guard does not make the call succeed — the seed loop runs over `0..=m`, and
    /// `NetBuilder::connect` then refuses an index past `n` with `NetError::OutOfRange`, which
    /// arrives as [`TopologyError::Net`]. So `is_err()` holds either way and says nothing about
    /// whether the generator checked its own precondition. The variant and its payload are what
    /// separate a stated refusal from an accident downstream.
    #[test]
    fn barabasi_albert_refuses_a_network_too_small_for_its_seed_clique() {
        assert_eq!(
            barabasi_albert(4, 4, &exc(1e-3), 0).unwrap_err(),
            TopologyError::TooSmall {
                n: 4,
                needed: 5,
                what: "barabasi_albert needs n >= m + 1 for the seed clique",
            }
        );
        assert_eq!(
            barabasi_albert(2, 7, &exc(1e-3), 0).unwrap_err(),
            TopologyError::TooSmall {
                n: 2,
                needed: 8,
                what: "barabasi_albert needs n >= m + 1 for the seed clique",
            }
        );
        // The boundary is where it says it is: `n == m + 1` is the bare seed clique and builds.
        let clique = barabasi_albert(5, 4, &exc(1e-3), 0).unwrap();
        assert_eq!(clique.n_syn, 2 * (4 * 5 / 2), "n = m + 1 is the complete graph on m + 1 nodes");
        assert_eq!(undirected_degrees(&clique), vec![4; 5]);
    }

    /// (survivor 9) [`Grid3::validate`] refuses a spacing of **zero**, not merely a negative one.
    ///
    /// No fixture passed zero, and the doc's phrase is "finite and strictly positive". A zero
    /// spacing is the case that matters: it puts every neuron at the origin, so every pairwise
    /// distance is `0`, the Gaussian `exp(-(0 / lambda)^2)` is `1`, and the distance-dependent
    /// generator emits a complete graph at probability `C` while still calling itself
    /// distance-dependent. Loosening `<= 0.0` to `< 0.0` is the one-character version of that.
    #[test]
    fn a_zero_lattice_spacing_is_refused_rather_than_collapsing_every_distance() {
        for bad in [0.0, -0.0, -50e-6, f64::NAN, f64::INFINITY] {
            let g = Grid3::new(4, 4, 2, bad);
            match g.validate().unwrap_err() {
                TopologyError::Length { name, value } => {
                    assert_eq!(name, "Grid3::spacing", "spacing = {bad} was blamed on {name}");
                    assert!(!value.is_finite() || value <= 0.0, "the error reported {value}");
                }
                e => panic!("spacing = {bad} gave {e:?} instead of a Length error"),
            }
            assert!(matches!(
                distance_dependent(&g, &MAASS_2002, 1e-4, &Wiring::default(), 0).unwrap_err(),
                TopologyError::Length { name: "Grid3::spacing", .. }
            ));
        }
        // What a zero spacing would have produced, had it been let through: every distance zero.
        let flat = Grid3::new(4, 4, 2, 0.0);
        assert_eq!(flat.distance(0, 31), Some(0.0), "a zero spacing collapses the lattice");
        assert!(Grid3::new(4, 4, 2, f64::MIN_POSITIVE).validate().is_ok(), "a tiny spacing is legal");
    }

    /// (survivor 10) [`MaassC::uniform`] sets **all four** pair types to the same factor, including
    /// inhibitory-to-inhibitory.
    ///
    /// Every call site of `uniform` in this module hands it an all-excitatory partition or reads
    /// only the `Silent` arms, so the `ii` field it writes is never consulted and zeroing it
    /// changed nothing. The exact consequence is the point of the constructor: because `c_for`
    /// returns the same number for every non-`Silent` pair, and
    /// [`distance_dependent_signed`] draws once per ordered pair in a fixed order whatever the
    /// probability is, the **set of synapses is identical for every partition** — only the weights
    /// move. That equality is exact, so it is asserted as equality.
    #[test]
    fn a_uniform_scale_factor_is_the_same_factor_for_all_four_pair_types() {
        for c in [0.37, 1.0, 0.0] {
            let u = MaassC::uniform(c);
            assert_eq!(u.c_for(Sign::Excitatory, Sign::Excitatory), c);
            assert_eq!(u.c_for(Sign::Excitatory, Sign::Inhibitory), c);
            assert_eq!(u.c_for(Sign::Inhibitory, Sign::Excitatory), c);
            assert_eq!(u.c_for(Sign::Inhibitory, Sign::Inhibitory), c, "the II term is not uniform");
            assert_eq!(u, MaassC { ee: c, ei: c, ie: c, ii: c });
        }
        let g = Grid3::new(6, 6, 1, 50e-6);
        let n = g.len();
        let w = Wiring::default();
        let uniform = MaassC::uniform(0.4);
        let lambda = g.lambda_of(2.0);
        let all_exc = vec![Sign::Excitatory; n];
        let mixed = shuffled_partition(n, 0.5, 2).unwrap();
        let a = distance_dependent_signed(&g, &uniform, lambda, &w, &all_exc, 9).unwrap();
        let b = distance_dependent_signed(&g, &uniform, lambda, &w, &mixed, 9).unwrap();
        assert!(a.n_syn > 0, "the fixture built no synapse");
        assert_eq!(a.n_syn, b.n_syn, "a uniform C made the synapse count depend on the partition");
        assert_eq!(a.offset, b.offset, "the two graphs differ in shape");
        assert_eq!(a.post, b.post, "the two graphs differ in which pairs they connect");
        assert_eq!(a.delay, b.delay);
        // Not vacuous: the mixed partition really does contain inhibitory-to-inhibitory synapses,
        // which are the ones a zeroed `ii` would have removed.
        let ii = (0..n)
            .filter(|&pre| mixed[pre] == Sign::Inhibitory)
            .flat_map(|pre| b.out_of(pre))
            .filter(|(post, _, _)| mixed[*post as usize] == Sign::Inhibitory)
            .count();
        assert!(ii > 0, "the fixture has no inhibitory-to-inhibitory pair to be sensitive to");
    }

    /// (survivor 11) [`MaassC::validate`] checks each field against **itself** and names the one it
    /// refused.
    ///
    /// Only `ee` was ever exercised, so `check_probability("MaassC::ii", self.ie)` — the right name
    /// against the wrong field — passed the suite while letting an out-of-range `ii` through into
    /// the generator, where it becomes a probability above one and silently a complete graph.
    /// Each of the four is bad in turn, alone, with the other three legal.
    #[test]
    fn each_maass_scale_factor_is_validated_against_its_own_field() {
        let good = MAASS_2002;
        let cases: [(&str, MaassC); 4] = [
            ("MaassC::ee", MaassC { ee: 1.5, ..good }),
            ("MaassC::ei", MaassC { ei: -0.1, ..good }),
            ("MaassC::ie", MaassC { ie: f64::NAN, ..good }),
            ("MaassC::ii", MaassC { ii: 1.5, ..good }),
        ];
        for (field, c) in cases {
            match c.validate().unwrap_err() {
                TopologyError::Probability { name, value } => {
                    assert_eq!(name, field, "{c:?} was blamed on {name} instead of {field}");
                    assert!(
                        value.is_nan() || !(0.0..=1.0).contains(&value),
                        "{field}: the error reported the legal value {value}"
                    );
                }
                e => panic!("{c:?} gave {e:?} instead of a Probability error"),
            }
            let g = Grid3::new(4, 4, 1, 50e-6);
            assert!(matches!(
                distance_dependent(&g, &c, 1e-4, &Wiring::default(), 0).unwrap_err(),
                TopologyError::Probability { .. }
            ));
        }
        assert!(good.validate().is_ok());
        assert!(MaassC::uniform(1.0).validate().is_ok());
        assert!(MaassC::uniform(0.0).validate().is_ok());
    }

    /// (survivor 12) [`distance_dependent_signed`] validates `lambda` **itself**, because it is a
    /// public entry point and not only a delegate.
    ///
    /// The check is written twice on purpose — [`distance_dependent`] validates before it allocates
    /// `wiring.signs(n)` — and only the outer copy was ever called, so loosening the inner
    /// `lambda <= 0.0` to `< 0.0` was invisible. A zero `lambda` is not harmless: `dist / 0.0` is
    /// `+inf` for every distinct pair, `exp(-inf)` is zero, and the generator returns a network
    /// with **no synapses at all** and no error, which is the failure a caller is least likely to
    /// notice.
    #[test]
    fn the_signed_entry_point_validates_its_own_lambda() {
        let g = Grid3::maass_column(50e-6);
        let w = Wiring::default();
        let signs = shuffled_partition(g.len(), CORTICAL_INHIBITORY_FRACTION, 1).unwrap();
        for bad in [0.0, -0.0, -1e-4, f64::NAN, f64::INFINITY] {
            match distance_dependent_signed(&g, &MAASS_2002, bad, &w, &signs, 1).unwrap_err() {
                TopologyError::Length { name, value } => {
                    assert_eq!(name, "lambda", "lambda = {bad} was blamed on {name}");
                    assert!(!value.is_finite() || value <= 0.0, "the error reported {value}");
                }
                e => panic!("lambda = {bad} gave {e:?} instead of a Length error"),
            }
            // And through the outer entry point, which has always refused it.
            assert!(matches!(
                distance_dependent(&g, &MAASS_2002, bad, &w, 1).unwrap_err(),
                TopologyError::Length { name: "lambda", .. }
            ));
        }
        // The smallest positive lambda is legal and builds: the boundary is at zero, not above it.
        assert!(distance_dependent_signed(&g, &MAASS_2002, 1e-9, &w, &signs, 1).is_ok());
    }

    /// (survivor 13) Every [`feedforward`] synapse carries **its own** presynaptic neuron's weight,
    /// not neuron zero's.
    ///
    /// The hole is named in the function's own doc and is exactly why the suite could not see it:
    /// `Wiring` puts the inhibitory population in the **trailing** index block, and in a layered
    /// network that block is the output layer, which projects nowhere. So with `Wiring::default`'s
    /// 20% there is no inhibitory neuron that emits anything, and every other feedforward fixture
    /// uses `excitatory_only`, where all the weights are the same number anyway. Reading
    /// `wiring.weight_of(n, 0)` instead of `wiring.weight_of(n, a)` was invisible in both.
    ///
    /// A half-inhibitory wiring on `[4, 4, 4]` moves the boundary into the hidden layer: `n = 12`,
    /// `round(0.5 * 12) = 6` inhibitory, so neurons `6` and `7` are inhibitory *and* sit in layer 1,
    /// where they project into layer 2.
    #[test]
    fn a_feedforward_hidden_layer_neuron_carries_its_own_sign() {
        let w = Wiring { w_exc: 1e-3, w_inh: -4e-3, delay: 1, inhibitory_fraction: 0.5 };
        let net = feedforward(&[4, 4, 4], 1.0, &w, 1).unwrap();
        assert_eq!(net.n, 12);
        assert_eq!(net.n_syn, 4 * 4 + 4 * 4);
        assert_eq!(layer_ranges(&[4, 4, 4]), vec![0..4, 4..8, 8..12]);
        for pre in 0..12usize {
            let expected = if pre < 6 { 1e-3 } else { -4e-3 };
            for (post, weight, _) in net.out_of(pre) {
                assert_eq!(weight, expected, "neuron {pre} -> {post} carried {weight}");
            }
        }
        // Both signs are actually present among the emitting neurons, or the assertion above would
        // hold for a wiring that had lost the distinction entirely.
        assert_eq!(net.out_of(4).count(), 4, "an excitatory hidden neuron projects forward");
        assert_eq!(net.out_of(6).count(), 4, "an inhibitory hidden neuron projects forward");
        let signs = dale_signs(&net).unwrap();
        assert_eq!(signs[4], Sign::Excitatory);
        assert_eq!(signs[6], Sign::Inhibitory, "neuron 6 is in layer 1 and inhibitory");
        assert_eq!(signs[8], Sign::Silent, "the output layer emits nothing, which is the old hole");
        assert!(dale_check(&net, &w.signs(12)).is_ok());
    }

    /// (survivor 14) [`undirected_neighbours`] drops a **self-loop**, so a unit that excites itself
    /// is not its own neighbour.
    ///
    /// The only network in this module that has a self-loop is [`winner_take_all`] with
    /// `w_self > 0`, and the suite reads that one with [`dale_signs`] and a simulation — never with
    /// a degree or a clustering measure. Deleting the guard costs each such neuron one spurious
    /// neighbour, which is visible three ways at once on a four-unit competition: the undirected
    /// degree goes from 3 to 4, the neuron appears in its own adjacency list, and the clustering
    /// coefficient goes from exactly 1 to `2 * 8 / (4 * 3) = 4 / 3` — a fraction of neighbour pairs
    /// above one, which is not a number this statistic can take.
    #[test]
    fn the_undirected_reading_drops_a_self_loop() {
        let wta = winner_take_all(4, -1e-3, 1e-3, 1).unwrap();
        assert_eq!(wta.n_syn, 4 + 4 * 3, "four self-synapses and twelve lateral ones");
        assert_eq!(undirected_degrees(&wta), vec![3; 4], "a unit counted itself as a neighbour");
        let adj = super::undirected_neighbours(&wta);
        for (v, near) in adj.iter().enumerate() {
            assert!(!near.contains(&(v as u32)), "neuron {v} is in its own neighbour list");
            assert_eq!(near.len(), 3, "neuron {v} has neighbours {near:?}");
        }
        assert_eq!(
            clustering_coefficient(&wta),
            Some(1.0),
            "the four units are a complete graph once the self-loops are dropped"
        );
        // Without a self-loop the same competition reads identically, which is what "dropped" means.
        let plain = winner_take_all(4, -1e-3, 0.0, 1).unwrap();
        assert_eq!(plain.n_syn, 4 * 3);
        assert_eq!(undirected_degrees(&plain), undirected_degrees(&wta));
        assert_eq!(clustering_coefficient(&plain), clustering_coefficient(&wta));
    }

    /// (survivor 15) [`clustering_coefficient`] divides by the number of neurons it **counted**,
    /// not by the number in the network.
    ///
    /// The function's doc spends a paragraph on this — a neuron of degree below two is skipped, not
    /// scored zero, because scoring it zero "turns a connectivity statistic into a density
    /// statistic". Nothing asserted it, because no fixture in the suite **mixes** the two
    /// populations: every graph whose clustering is asserted has every neuron countable (the ring
    /// lattice, `G(n, p)`, the complete graph) or none of them (four isolated neurons, which
    /// returns `None`), and `sum / counted` and `sum / net.n` agree exactly whenever `counted == n`.
    ///
    /// A triangle plus two isolated neurons separates them with no arithmetic slack at all: three
    /// neurons of degree 2 each score `2 * 1 / (2 * 1) = 1`, so the answer is `3 / 3 = 1` exactly,
    /// and dividing by the neuron count gives `3 / 5 = 0.6`. Both are exact in binary, so this is
    /// asserted as equality rather than within a tolerance.
    #[test]
    fn clustering_averages_over_the_neurons_it_is_defined_for() {
        let mut b = NetBuilder::new(5);
        for &(u, v) in &[(0u32, 1u32), (1, 2), (2, 0)] {
            b.connect(u, v, 1e-3, 1).unwrap();
            b.connect(v, u, 1e-3, 1).unwrap();
        }
        let triangle_plus_two = b.build();
        assert_eq!(undirected_degrees(&triangle_plus_two), vec![2, 2, 2, 0, 0]);
        assert_eq!(
            clustering_coefficient(&triangle_plus_two),
            Some(1.0),
            "the two isolated neurons were averaged in as zeros"
        );
        // A second shape where the counted neurons do not all score the same, so the numerator is
        // not a whole number of neurons either: a triangle sharing one vertex with a path.
        // Neighbourhoods: 0 {1, 2, 3}, 1 {0, 2}, 2 {0, 1}, 3 {0, 4}, 4 {3}, 5 {}.
        // C_0 = 2 * 1 / (3 * 2) = 1/3, C_1 = C_2 = 1, C_3 = 0; neuron 4 has one neighbour and
        // neuron 5 none, so counted = 4 and the mean is (1/3 + 1 + 1 + 0) / 4 = 7 / 12.
        let mut b = NetBuilder::new(6);
        for &(u, v) in &[(0u32, 1u32), (1, 2), (2, 0), (0, 3), (3, 4)] {
            b.connect(u, v, 1e-3, 1).unwrap();
            b.connect(v, u, 1e-3, 1).unwrap();
        }
        let kite = b.build();
        assert_eq!(undirected_degrees(&kite), vec![3, 2, 2, 2, 1, 0]);
        let c = clustering_coefficient(&kite).unwrap();
        assert!((c - 7.0 / 12.0).abs() < 1e-15, "clustering {c} is not 7/12; averaged over 6 it is {}", 7.0 / 3.0 / 6.0);
    }

    /// (survivor 16) A network with **no ordered pair** is not reported strongly connected.
    ///
    /// `pairs > 0 && reachable == pairs` is the guard, and dropping it makes the empty network —
    /// and the one-neuron network — vacuously "strongly connected", since `0 == 0`. The suite could
    /// not see that, because the only path it took to the predicate was
    /// [`characteristic_path_length`], which returns `mean_hops`, and `mean_hops` is already `None`
    /// when nothing is reachable. So the answer was `None` either way and the guard was decorative.
    ///
    /// Graph theory would call the one-vertex graph strongly connected; this module refuses instead,
    /// for the same reason [`clustering_coefficient`] returns `None` when no neuron has two
    /// neighbours — the quantity the caller is about to take a mean of does not exist. That choice
    /// is now in [`PathStats::is_strongly_connected`]'s doc as well as here.
    #[test]
    fn a_network_with_no_ordered_pair_is_not_reported_strongly_connected() {
        for n in [0usize, 1] {
            let net = NetBuilder::new(n).build();
            let stats = path_stats(&net);
            assert_eq!(stats.pairs, 0, "n = {n} has no ordered pair of distinct neurons");
            assert!(!stats.is_strongly_connected(), "n = {n} was called strongly connected");
            assert_eq!(characteristic_path_length(&net), None, "n = {n}");
        }
        assert!(
            !PathStats { pairs: 0, reachable: 0, mean_hops: None, diameter: None }
                .is_strongly_connected()
        );
        // The predicate still says yes where it should: two neurons joined both ways.
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, 1e-3, 1).unwrap();
        b.connect(1, 0, 1e-3, 1).unwrap();
        let pair = b.build();
        assert!(path_stats(&pair).is_strongly_connected());
        assert_eq!(characteristic_path_length(&pair), Some(1.0));
        // And no when one direction is missing, which is the case the `reachable == pairs` half
        // catches and the `pairs > 0` half does not.
        let mut b = NetBuilder::new(2);
        b.connect(0, 1, 1e-3, 1).unwrap();
        let one_way = b.build();
        assert_eq!(path_stats(&one_way).reachable, 1);
        assert!(!path_stats(&one_way).is_strongly_connected());
    }

    /// (survivor 17) `below_u64` **applies** its rejection zone, rather than computing one and
    /// returning `v % n` regardless.
    ///
    /// `unbiased_zone_holds_whole_residue_classes` tests the zone function; nothing tested the loop
    /// that uses it. At the sizes this crate normally reaches the residual bias of skipping the
    /// rejection is about `2.3e-10`, which no sample mean can see — so the instrument has to be an
    /// `n` where rejection is common. `n = 3 * 2^62` is one: the largest multiple of `n` below
    /// `2^64` is `n` itself, so a quarter of all 64-bit draws are rejected, and the codes those
    /// draws would have folded onto are exactly `[0, 2^62)`.
    ///
    /// That makes the difference a factor of `3 / 2` on a measurable event. A correct draw is
    /// uniform on `[0, n)`, so `P(v < 2^62) = 2^62 / (3 * 2^62) = 1 / 3` exactly. Accepting every
    /// draw gives `[0, 2^62)` twice the density of the rest and `P = 1 / 2` exactly. Over 20,000
    /// draws that is 6,667 against 10,000, `sd = sqrt(20000 * (1/3) * (2/3)) = 66.7`: this
    /// implementation measures 6,613, and skipping the rejection would measure 50 sigma away.
    ///
    /// Such an `n` is reachable rather than hypothetical: [`check_n`] admits `n` up to `u32::MAX`
    /// and `erdos_renyi_gnm`'s pair space `n(n - 1)` passes `3 * 2^62` at about 3.72 billion
    /// neurons, which is inside that limit.
    #[test]
    fn the_sixty_four_bit_draw_applies_its_rejection_zone() {
        let n = 3u64 << 62;
        assert_eq!(super::unbiased_zone(n), n - 1, "the zone should hold exactly one residue class");
        let trials = 20_000usize;
        let mut rng = Rng::new(31);
        let mut low = 0usize;
        for _ in 0..trials {
            let v = super::below_u64(&mut rng, n);
            assert!(v < n, "draw {v} escaped [0, {n})");
            if v < (1u64 << 62) {
                low += 1;
            }
        }
        let expected = trials as f64 / 3.0;
        let sd = (trials as f64 * (1.0 / 3.0) * (2.0 / 3.0)).sqrt();
        let z = (low as f64 - expected) / sd;
        assert!(
            z.abs() < 4.0,
            "{low} of {trials} draws fell in the bottom third, {z:.1} sigma from {expected:.0} — accepting every draw would give {}",
            trials / 2
        );
        // The mechanism, directly: a rejected draw costs two more `next_u32` calls, so at this `n`
        // some seeds consume four or six where an accept-everything loop always consumes two.
        let mut extra = 0usize;
        for seed in 0..12u64 {
            let mut after = Rng::new(seed);
            let _ = super::below_u64(&mut after, n);
            let mut walk = Rng::new(seed);
            let mut steps = 0usize;
            while walk != after {
                walk.next_u32();
                steps += 1;
                assert!(steps < 200, "seed {seed}: the draw did not land on a reachable state");
            }
            assert_eq!(steps % 2, 0, "seed {seed} consumed {steps} halves of a 64-bit draw");
            if steps > 2 {
                extra += 1;
            }
        }
        assert!(extra > 0, "no seed of the first twelve rejected a draw at n = 3 * 2^62");
    }

    /// Pins that `reciprocal` hands the FORWARD synapse of each pair to the builder before the
    /// return one, by the index the builder names when it refuses.
    ///
    /// `NetBuilder::connect` validates `for idx in [pre, post]` — `pre` first — so whichever call
    /// is made first decides which endpoint the error carries, and the same holds for the
    /// non-finite-weight check, whose payload is the whole ordered pair. The suite could not see
    /// the order because both callers of `reciprocal` (`watts_strogatz` and `barabasi_albert`)
    /// only ever hand it in-range endpoints and finite weights, so its two error paths are
    /// unreachable from the generators: the hole is a private helper's refusal that no public
    /// entry point can reach. This module already closes that shape of hole by calling the helper
    /// directly — see
    /// `the_preferential_attachment_probe_refuses_rather_than_returning_a_short_edge_list`.
    #[test]
    fn the_forward_synapse_of_a_reciprocal_pair_is_offered_to_the_builder_first() {
        let w = Wiring::excitatory_only(1e-3, 1);
        // Both endpoints are past the two neurons, so `connect` refuses on whichever it was given
        // as `pre`: 5 for the forward call `connect(5, 7, ..)`, 7 for the return one.
        assert_eq!(
            super::reciprocal(2, &[(5u32, 7u32)], &w).unwrap_err(),
            TopologyError::Net(NetError::OutOfRange { index: 5, n: 2 }),
            "the forward endpoint is the one the builder sees first"
        );
        // The same order shows through the weight check, which `connect` reaches only once both
        // endpoints are in range. The forward synapse of the pair (0, 1) is 0 -> 1.
        let nan = Wiring { w_exc: f64::NAN, w_inh: f64::NAN, delay: 1, inhibitory_fraction: 0.0 };
        assert_eq!(
            super::reciprocal(2, &[(0u32, 1u32)], &nan).unwrap_err(),
            TopologyError::Net(NetError::NonFiniteWeight { pre: 0, post: 1 }),
            "the forward synapse is 0 -> 1, not 1 -> 0"
        );
    }
}
