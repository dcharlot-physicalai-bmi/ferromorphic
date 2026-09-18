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
//! Every generator here is Dale-compliant by construction, because [`Wiring`] assigns a sign per
//! **presynaptic** neuron and never per synapse. The reason [`dale_check`] exists anyway is that a
//! network which quietly violates Dale's law is **biologically meaningless and computationally
//! fine** — it trains, it fires, it produces a plausible raster — which is exactly how such a
//! network survives review. The violation has to be caught by an assertion or it is not caught.
//!
//! The 80/20 excitatory/inhibitory split is the modelling convention, from Brunel (J. Comput.
//! Neurosci. 8:183-208, 2000, `N_E = 4 N_I`) and Maass et al. (Neural Computation 14(11):2531-2560,
//! 2002, 20% inhibitory). **Caveat beside the figure:** the anatomy it abstracts is 15-20%
//! inhibitory and varies by area and species — Beaulieu and Colonnier (J. Comp. Neurol.
//! 231:180-189, 1985) count roughly 15% GABA-immunoreactive neurons in cat area 17, and
//! Braitenberg and Schüz (*Cortex: Statistics and Geometry of Neuronal Connectivity*, 2nd ed.,
//! Springer, 1998) put pyramidal cells near 85% of mouse cortex. 80/20 is a round number chosen
//! inside that range, not a measurement.
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
/// is 15-20% and area-dependent; see the module doc. Dimensionless, in `[0, 1]`.
pub const CORTICAL_INHIBITORY_FRACTION: f64 = 0.2;

/// Maass's inhibition-to-excitation weight ratio for a network balanced at the 80/20 split.
///
/// Brunel's `g = |J_I| / J_E`: with four times as many excitatory neurons as inhibitory ones, the
/// mean drive on a neuron cancels exactly at `g = 4`. Dimensionless. [`Wiring::balanced`] applies
/// it and [`Wiring::expected_drive`] is the arithmetic that makes the cancellation checkable.
pub const BALANCED_G: f64 = 4.0;

/// How a generated graph turns into weighted, signed, delayed synapses.
///
/// This is the object that makes Dale's law structural rather than hoped for: the sign attaches to
/// the **presynaptic neuron index**, so no generator can emit a mixed-sign cell even by accident.
/// Neurons `0 .. n - n_inhibitory` are excitatory and the remaining block is inhibitory — a
/// contiguous partition, so `Wiring` plus `n` is enough to reconstruct which is which without
/// carrying a per-neuron vector around.
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
        match self.sign_of(n, pre) {
            Sign::Excitatory => self.w_exc,
            Sign::Inhibitory => self.w_inh,
            Sign::Silent => 0.0,
        }
    }

    /// The expected sum of a neuron's **incoming** weights in a `G(n, p)` graph, volts per spike.
    ///
    /// `p * (n_exc * w_exc + n_inh * w_inh)`. This is the balance condition in closed form: it is
    /// zero exactly when `|w_inh| / w_exc` equals the excitatory-to-inhibitory count ratio, which is
    /// what [`Wiring::balanced`] arranges. A network whose expected drive is far from zero either
    /// saturates or falls silent, and which one it does is decided here rather than in simulation.
    #[must_use]
    pub fn expected_drive(&self, n: usize, p: f64) -> f64 {
        let n_inh = self.n_inhibitory(n);
        let n_exc = n - n_inh;
        p * (n_exc as f64 * self.w_exc + n_inh as f64 * self.w_inh)
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
/// derived by Krapivsky, Redner and Leyvraz (Phys. Rev. Lett. 85:4629-4632, 2000) and by
/// Dorogovtsev, Mendes and Samukhin (Phys. Rev. Lett. 85:4633-4636, 2000). This module did not
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
            for step in 0..repeated.len() {
                let cand = repeated[(start + step) % repeated.len()];
                if !targets.contains(&cand) {
                    targets.push(cand);
                    break;
                }
            }
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

    /// Neurons in the grid: `nx * ny * nz`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nx * self.ny * self.nz
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

    /// Reject a grid with a zero dimension or a non-positive spacing.
    ///
    /// # Errors
    ///
    /// [`TopologyError::TooSmall`] for a zero dimension, [`TopologyError::Length`] for a spacing
    /// that is not finite and strictly positive.
    pub fn validate(&self) -> Result<(), TopologyError> {
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err(TopologyError::TooSmall {
                n: 0,
                needed: 1,
                what: "Grid3 needs every dimension to be at least 1",
            });
        }
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
    grid.validate()?;
    c.validate()?;
    wiring.validate()?;
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(TopologyError::Length { name: "lambda", value: lambda });
    }
    let n = grid.len();
    check_n(n)?;
    let mut rng = Rng::new(seed);
    let mut b = NetBuilder::new(n);
    for a in 0..n {
        let sa = wiring.sign_of(n, a);
        let w = wiring.weight_of(n, a);
        for d in 0..n {
            if a == d {
                continue;
            }
            let dist = grid.distance(a, d).unwrap_or(f64::INFINITY);
            let ratio = dist / lambda;
            let p = c.c_for(sa, wiring.sign_of(n, d)) * (-(ratio * ratio)).exp();
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

/// A uniform integer in `[0, n)` for an `n` that may exceed `u32`, without modulo bias.
///
/// The pair space of `G(n, m)` is `n(n-1)`, which passes `u32::MAX` at about 65,000 neurons — a
/// perfectly ordinary network size — so the 32-bit draw is not enough on its own. Delegates to
/// [`crate::rng::Rng::below`] below that point so that small graphs consume exactly the draws they
/// consumed before this path existed.
fn below_u64(rng: &mut Rng, n: u64) -> u64 {
    if n <= u64::from(u32::MAX) {
        return u64::from(rng.below(n as u32));
    }
    let zone = u64::MAX - (u64::MAX % n) - (n - 1);
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
        erdos_renyi_gnm, erdos_renyi_gnp, feedforward, hops_from, in_weight_sums, layer_ranges,
        path_stats, power_law_exponent, undirected_degrees, watts_strogatz, weak_components,
        winner_take_all, wta_inhibition_floor,
    };
    use crate::net::NetBuilder;
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
        // The lattice end: both measures at their closed forms.
        assert!((c[0] - 3.0 * 8.0 / (4.0 * 9.0)).abs() < 1e-12, "C(0) = {}", c[0]);
        assert!(l[0] > 15.0, "L(0) = {} should be about n / (2k) = 20", l[0]);
        // The random end: clustering has collapsed with it.
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

    /// The fitter is checked before it is trusted: draw from a continuous power law by inverse CDF,
    /// floor to integers, and require the estimator to recover the exponent it was given.
    ///
    /// `x = k_min * (1 - u)^(-1 / (alpha - 1))` has `P(X > x) ~ x^(1 - alpha)`, which is exponent
    /// `alpha`. Recovery at `alpha = 2.5`, `k_min = 10`, 100,000 samples lands inside 0.02 here.
    #[test]
    fn the_power_law_fitter_recovers_a_known_exponent() {
        let mut rng = Rng::new(31);
        let alpha = 2.5f64;
        let k_min = 10usize;
        let mut sample = Vec::with_capacity(100_000);
        for _ in 0..100_000 {
            let u = rng.next_f64();
            let x = k_min as f64 * (1.0 - u).powf(-1.0 / (alpha - 1.0));
            sample.push(x.floor() as usize);
        }
        let got = power_law_exponent(&sample, k_min).unwrap();
        assert!((got - alpha).abs() < 0.05, "fitted {got:.4} for a sample drawn at {alpha}");
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
            for &seed in &[77u64, 5, 1234] {
                let net = barabasi_albert(20_000, m, &exc(1e-3), seed).unwrap();
                let deg = undirected_degrees(&net);
                let got = power_law_exponent(&deg, 6).unwrap();
                assert!(
                    (got - ladder[0]).abs() < 0.03,
                    "m = {m}, seed {seed}: fitted {got:.4} vs the model's own MLE {:.4}",
                    ladder[0]
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
        let net = distance_dependent(&g, &MAASS_2002, g.lambda_of(2.0), &Wiring::default(), 1)
            .unwrap();
        assert!(net.n_syn > 0);
        // Dale by construction, with the 80/20 split the wiring declares.
        dale_check(&net, &Wiring::default().signs(135)).unwrap();
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

    /// Every generator is Dale-compliant by construction, checked against the partition the wiring
    /// declares rather than against itself.
    #[test]
    fn every_generator_obeys_dale_by_construction() {
        let w = Wiring::default();
        let nets = vec![
            ("gnp", erdos_renyi_gnp(60, 0.1, &w, 1).unwrap()),
            ("gnm", erdos_renyi_gnm(60, 300, &w, 1).unwrap()),
            ("ws", watts_strogatz(60, 6, 0.1, &w, 1).unwrap()),
            ("ba", barabasi_albert(60, 2, &w, 1).unwrap()),
            (
                "maass",
                distance_dependent(
                    &Grid3::new(5, 4, 3, 50e-6),
                    &MAASS_2002,
                    Grid3::new(5, 4, 3, 50e-6).lambda_of(2.0),
                    &w,
                    1,
                )
                .unwrap(),
            ),
            ("ff", feedforward(&[20, 20, 20], 1.0, &w, 1).unwrap()),
        ];
        for (name, net) in nets {
            assert!(net.n_syn > 0, "{name} produced an empty network");
            dale_check(&net, &w.signs(net.n))
                .unwrap_or_else(|e| panic!("{name} violates Dale's law: {e}"));
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
    #[test]
    fn a_balanced_wiring_has_zero_expected_drive_and_the_graph_agrees() {
        let w = Wiring::balanced(1e-3, 1);
        let (n, p) = (500usize, 0.1);
        assert!(w.expected_drive(n, p).abs() < 1e-18, "drive {}", w.expected_drive(n, p));
        assert!((w.w_inh / w.w_exc + BALANCED_G).abs() < 1e-18);
        let net = erdos_renyi_gnp(n, p, &w, 2).unwrap();
        let sums = in_weight_sums(&net);
        let mean = sums.iter().sum::<f64>() / n as f64;
        // Per-neuron variance: p(1-p) * sum of w^2 over presynaptic candidates.
        let n_inh = w.n_inhibitory(n);
        let var = p * (1.0 - p)
            * ((n - n_inh) as f64 * w.w_exc * w.w_exc + n_inh as f64 * w.w_inh * w.w_inh);
        let sem = (var / n as f64).sqrt();
        assert!(mean.abs() < 4.0 * sem, "mean incoming weight {mean:.3e} vs 4 sem {:.3e}", 4.0 * sem);
        // An unbalanced wiring must fail the same measurement, or the measurement proves nothing.
        let hot = Wiring { w_inh: -1e-3, ..w };
        assert!(hot.expected_drive(n, p) > 0.0);
        let hot_net = erdos_renyi_gnp(n, p, &hot, 2).unwrap();
        let hot_mean = in_weight_sums(&hot_net).iter().sum::<f64>() / n as f64;
        assert!(hot_mean > 4.0 * sem, "unbalanced network's drive {hot_mean:.3e} is not positive");
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

    /// The winner-take-all regime, simulated. Above the mean-field floor there is exactly one
    /// winner in the steady state, and its rate is the rate it would have alone.
    #[test]
    fn the_winner_take_all_regime_produces_exactly_one_winner() {
        let lif = Lif::default();
        let currents = [6e-9, 4e-9, 3.5e-9, 3e-9];
        let floor = wta_inhibition_floor(&lif, currents[0], currents[1]).unwrap();
        // 9.7 mV per spike at the default LIF; sanity-check the closed form itself.
        let r_win = lif.rate(currents[0]).unwrap();
        let want = (lif.v_inf(currents[1]) - lif.v_th) / (r_win * lif.tau_m);
        assert!((floor - want).abs() < 1e-15);
        assert!((floor - 9.69e-3).abs() < 0.2e-3, "floor {floor:.4} V is not the expected 9.7 mV");

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
}
